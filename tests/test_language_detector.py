"""U1 detector: the three-part gate, the census decision table, and the flag's edges.

The grading check for queue row 3 unit (1). Stub seeded by the lead: fill each body in
place, never rename a case and never delete one. Add a case with the next free name where
the contract needs more.

Contract, in the order the cases sit below:

- `lid_accept(argmax, score, ja, en)` -- three parts and no fourth. The GLOBAL 107-way
  argmax must be `ja` or `en`; that argmax's score must be >= `LID_MIN_SCORE`;
  `abs(ja - en)` must be >= `LID_MIN_MARGIN`. The pair is NEVER renormalized. Both floors
  are inclusive. No duration gate lives here -- unit (2) owns the schedule.
- `tests/lid_census.json` -- the recorded spike scores, read but never edited. Keys:
  `view_fields` = `["spoken", "utterance", "bucket", "argmax", "argmax_score", "ja",
  "en"]` over 8,628 `views` (buckets `1s` 1925, `2s` 1516, `3s` 1453, `5s` 1188, `8s` 620,
  `VADfin` 1926) for 1,926 utterances, plus `synthetic_fields` = `["id", "seconds",
  "argmax", "argmax_score", "ja", "en"]` over 25 `synthetic` probes. The law this table
  must reproduce: at the 2 s bucket, 1,338 correct / 178 abstain / 0 false of 1,516; 25 of
  25 synthetic probes rejected; and the single 1 s English view that the rule accepts as
  `ja` at score 0.98221, which is the input the schedule must never hand it.
- `--two-way` -- `action="store_true"`, default OFF. With `--source-lang`, and with
  `--engine k2v2` or `--engine parakeet`, it is a parse error (`SystemExit` 2).
  `--source-lang` alone keeps its one-way meaning. Flag off constructs no detector and
  opens no ONNX session; flag on constructs exactly one. `check_models` preflights the LID
  weights under the flag and not otherwise.

Weights-free by construction: this suite reads the committed census and stubs the session,
so it runs in a fresh clone. Any case that would need `models/lid/d2-ecapa/` belongs to the
unit that measures, not here.
"""

from __future__ import annotations

import asyncio
import json
import sys
from argparse import Namespace
from collections import Counter
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import live_stt

CENSUS = json.loads((Path(__file__).parent / "lid_census.json").read_text(encoding="utf-8"))
VIEWS = [dict(zip(CENSUS["view_fields"], values, strict=True)) for values in CENSUS["views"]]
SYNTHETIC = [
    dict(zip(CENSUS["synthetic_fields"], values, strict=True)) for values in CENSUS["synthetic"]
]


def census_decision(row) -> str | None:
    return live_stt.lid_accept(row["argmax"], row["argmax_score"], row["ja"], row["en"])


def parse_cli(monkeypatch, *argv):
    seen = {}
    monkeypatch.setattr(sys, "argv", ["live-stt", *argv])

    def supervise(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(live_stt, "_supervise_session", supervise)
    live_stt.main()
    return seen["args"]


def run_cli_to_session(monkeypatch, *argv):
    seen: dict[str, object] = {}
    detectors = 0
    monkeypatch.setattr(sys, "argv", ["live-stt", *argv])
    monkeypatch.setattr(live_stt, "check_models", lambda *_args: None)
    monkeypatch.setattr(live_stt, "check_device", lambda *_args: None)

    class FakeDetector:
        def __init__(self, *_args, **_kwargs):
            nonlocal detectors
            detectors += 1

    async def run_session(args):
        seen["args"] = args
        seen["source_lang"] = live_stt.ASR_LANGUAGE

    def supervise(args):
        live_stt._run_cli(args)
        return 0

    monkeypatch.setattr(live_stt, "LanguageDetector", FakeDetector)
    monkeypatch.setattr(live_stt, "run_session", run_session)
    monkeypatch.setattr(live_stt, "_supervise_session", supervise)
    live_stt.main()
    seen["detectors"] = detectors
    return seen


def detector_constructions_before_device_probe(monkeypatch, two_way: bool) -> int:
    constructions = 0

    class ReachedDeviceProbe(RuntimeError):
        pass

    class FakeDetector:
        def __init__(self, *_args, **_kwargs):
            nonlocal constructions
            constructions += 1

    def query_devices(*_args, **_kwargs):
        raise ReachedDeviceProbe

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(query_devices=query_devices))
    monkeypatch.setattr(live_stt, "load_recognizer", lambda *_args: object())
    monkeypatch.setattr(live_stt, "make_vad", lambda: (object(), 0))
    monkeypatch.setattr(live_stt, "LanguageDetector", FakeDetector)
    args = SimpleNamespace(engine="whisper", asr_device="CPU", device=None, two_way=two_way)

    with pytest.raises(ReachedDeviceProbe):
        asyncio.run(live_stt.run_session(args))
    return constructions


def configure_present_one_way_models(monkeypatch, tmp_path: Path) -> None:
    model_root = tmp_path / "models"
    vad_model = model_root / "silero_vad.onnx"
    engine_dir = model_root / "whisper"
    engine_dir.mkdir(parents=True)
    vad_model.touch()
    (engine_dir / "openvino_encoder_model.xml").touch()

    monkeypatch.setattr(live_stt, "MODELS_DIR", model_root)
    monkeypatch.setattr(live_stt, "VAD_MODEL", vad_model)
    monkeypatch.setattr(live_stt, "ENGINE_DIRS", {"whisper": engine_dir})
    monkeypatch.setattr(live_stt, "LID_MODEL_DIR", model_root / "lid/d2-ecapa")


def fake_language_detector_runtime(monkeypatch, tmp_path: Path, logits: np.ndarray):
    model_dir = tmp_path / "d2-ecapa"
    model_dir.mkdir()
    (model_dir / "voxlingua107.onnx").touch()
    language_map = {}
    for index in range(107):
        iso = "en" if index == 20 else "ja" if index == 45 else f"x{index}"
        language_map[str(index)] = {"iso": iso, "name": iso}
    (model_dir / "lang_map.json").write_text(json.dumps(language_map), encoding="utf-8")

    calls = {"sessions": [], "runs": []}

    class SessionOptions:
        pass

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="waveform")]

        def run(self, output_names, feeds):
            calls["runs"].append((output_names, feeds))
            return [logits[np.newaxis, :]]

    def inference_session(*args, **kwargs):
        calls["sessions"].append((args, kwargs))
        return FakeSession()

    fake_ort = SimpleNamespace(
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        InferenceSession=inference_session,
        SessionOptions=SessionOptions,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(live_stt, "ort", fake_ort, raising=False)
    monkeypatch.setattr(live_stt, "onnxruntime", fake_ort, raising=False)
    monkeypatch.setattr(live_stt, "InferenceSession", inference_session, raising=False)
    monkeypatch.setattr(live_stt, "SessionOptions", SessionOptions, raising=False)
    return model_dir, calls


def test_the_gate_accepts_a_clear_japanese_view() -> None:
    assert live_stt.lid_accept("ja", 0.80, 0.80, 0.10) == "ja"


def test_the_gate_accepts_a_clear_english_view() -> None:
    assert live_stt.lid_accept("en", 0.75, 0.05, 0.75) == "en"


def test_a_third_language_argmax_is_rejected_however_the_pair_splits() -> None:
    assert live_stt.lid_accept("fr", 0.55, 0.40, 0.01) is None


def test_a_score_below_the_floor_abstains() -> None:
    assert live_stt.lid_accept("ja", live_stt.LID_MIN_SCORE - 1e-9, 0.80, 0.10) is None


def test_a_margin_below_the_floor_abstains() -> None:
    en = 0.60 - (live_stt.LID_MIN_MARGIN - 1e-9)
    assert live_stt.lid_accept("ja", 0.60, 0.60, en) is None


def test_the_margin_is_never_renormalized_over_the_pair() -> None:
    # Raw gap = 0.25; pair-renormalized gap = 0.56 and would cross the floor.
    assert live_stt.lid_accept("ja", 0.35, 0.35, 0.10) is None


def test_both_floors_are_inclusive() -> None:
    assert live_stt.lid_accept("ja", live_stt.LID_MIN_SCORE, live_stt.LID_MIN_MARGIN, 0.0) == "ja"


def test_the_rule_reads_the_recorded_scores_and_nothing_else() -> None:
    assert tuple(signature(live_stt.lid_accept).parameters) == (
        "argmax",
        "score",
        "ja",
        "en",
    )
    assert live_stt.lid_accept(argmax="en", score=0.90, ja=0.01, en=0.90) == "en"


def test_the_two_second_bucket_reproduces_the_law_counts() -> None:
    outcomes: Counter[str] = Counter()
    for row in VIEWS:
        if row["bucket"] != "2s":
            continue
        decision = census_decision(row)
        if decision is None:
            outcomes["abstain"] += 1
        elif decision == row["spoken"]:
            outcomes["correct"] += 1
        else:
            outcomes["false"] += 1

    # Counter-to-Counter, because a Counter that never incremented "false" compares
    # unequal to a plain dict carrying an explicit zero for it.
    assert outcomes == Counter(correct=1338, abstain=178, false=0)
    assert outcomes.total() == 1516


def test_every_synthetic_probe_is_rejected() -> None:
    assert len(SYNTHETIC) == 25
    accepted = [(row["id"], census_decision(row)) for row in SYNTHETIC]
    assert all(decision is None for _, decision in accepted)


def test_the_one_second_english_view_the_rule_routes_to_ja() -> None:
    misrouted = [
        row
        for row in VIEWS
        if row["spoken"] == "en" and row["bucket"] == "1s" and census_decision(row) == "ja"
    ]

    assert len(misrouted) == 1
    assert misrouted[0]["argmax"] == "ja"
    assert misrouted[0]["argmax_score"] == pytest.approx(0.98221)


def test_the_census_fixture_is_whole() -> None:
    assert CENSUS["view_fields"] == [
        "spoken",
        "utterance",
        "bucket",
        "argmax",
        "argmax_score",
        "ja",
        "en",
    ]
    assert CENSUS["synthetic_fields"] == [
        "id",
        "seconds",
        "argmax",
        "argmax_score",
        "ja",
        "en",
    ]
    assert CENSUS["utterances"] == 1926
    assert len(VIEWS) == 8628
    assert Counter(row["bucket"] for row in VIEWS) == {
        "1s": 1925,
        "2s": 1516,
        "3s": 1453,
        "5s": 1188,
        "8s": 620,
        "VADfin": 1926,
    }
    assert len(SYNTHETIC) == 25


def test_two_way_with_source_lang_is_a_parse_error(monkeypatch) -> None:
    for language in ("ja", "en"):
        with pytest.raises(SystemExit) as exit_info:
            parse_cli(monkeypatch, "--two-way", "--source-lang", language)
        assert exit_info.value.code == 2


def test_two_way_with_a_sherpa_engine_is_a_parse_error(monkeypatch) -> None:
    for engine in ("k2v2", "parakeet"):
        with pytest.raises(SystemExit) as exit_info:
            parse_cli(monkeypatch, "--two-way", "--engine", engine)
        assert exit_info.value.code == 2


def test_two_way_with_whisper_parses(monkeypatch) -> None:
    args = parse_cli(monkeypatch, "--two-way")

    assert args.engine == "whisper"
    assert args.two_way is True
    assert args.source_lang is None


def test_source_lang_alone_keeps_its_one_way_meaning(monkeypatch) -> None:
    bare = run_cli_to_session(monkeypatch)
    english = run_cli_to_session(monkeypatch, "--source-lang", "en")
    bare_args = bare["args"]
    english_args = english["args"]

    assert isinstance(bare_args, Namespace)
    assert isinstance(english_args, Namespace)
    assert bare_args.source_lang is None
    assert bare["source_lang"] == "ja"
    assert english_args.source_lang == "en"
    assert english["source_lang"] == "en"
    assert bare["detectors"] == english["detectors"] == 0


def test_two_way_off_constructs_no_detector(monkeypatch) -> None:
    assert detector_constructions_before_device_probe(monkeypatch, two_way=False) == 0


def test_two_way_on_constructs_exactly_one_detector(monkeypatch) -> None:
    assert detector_constructions_before_device_probe(monkeypatch, two_way=True) == 1


def test_check_models_preflights_the_lid_weights_under_two_way(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    configure_present_one_way_models(monkeypatch, tmp_path)
    reached_session = False

    async def run_session(_args):
        nonlocal reached_session
        reached_session = True

    def supervise(args):
        live_stt._run_cli(args)
        return 0

    monkeypatch.setattr(sys, "argv", ["live-stt", "--two-way"])
    monkeypatch.setattr(live_stt, "check_device", lambda *_args: None)
    monkeypatch.setattr(live_stt, "run_session", run_session)
    monkeypatch.setattr(live_stt, "_supervise_session", supervise)

    with pytest.raises(SystemExit) as exit_info:
        live_stt.main()

    assert exit_info.value.code == 1
    assert "d2-ecapa" in capsys.readouterr().err
    assert reached_session is False


def test_check_models_ignores_the_lid_weights_without_two_way(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    configure_present_one_way_models(monkeypatch, tmp_path)
    reached_session = False

    async def run_session(_args):
        nonlocal reached_session
        reached_session = True

    def supervise(args):
        live_stt._run_cli(args)
        return 0

    monkeypatch.setattr(sys, "argv", ["live-stt"])
    monkeypatch.setattr(live_stt, "check_device", lambda *_args: None)
    monkeypatch.setattr(live_stt, "run_session", run_session)
    monkeypatch.setattr(live_stt, "_supervise_session", supervise)
    live_stt.main()

    assert reached_session is True
    assert "d2-ecapa" not in capsys.readouterr().err


def test_language_detector_opens_one_cpu_session_with_the_measured_threads(
    monkeypatch, tmp_path: Path
) -> None:
    logits = np.zeros(107, dtype=np.float32)
    model_dir, calls = fake_language_detector_runtime(monkeypatch, tmp_path, logits)

    live_stt.LanguageDetector(model_dir)

    assert len(calls["sessions"]) == 1
    args, kwargs = calls["sessions"][0]
    assert Path(args[0]) == model_dir / "voxlingua107.onnx"
    providers = kwargs.get("providers", args[2] if len(args) > 2 else ())
    assert tuple(providers) == ("CPUExecutionProvider",)
    options = kwargs.get("sess_options", args[1] if len(args) > 1 else None)
    assert options is not None
    assert options.intra_op_num_threads == live_stt.LID_INTRA_OP_THREADS
    assert options.inter_op_num_threads == live_stt.LID_INTER_OP_THREADS
    assert options.execution_mode == "sequential"


def test_language_detector_scores_the_global_softmax_without_touching_pcm(
    monkeypatch, tmp_path: Path
) -> None:
    logits = np.full(107, -10.0, dtype=np.float32)
    logits[0] = 3.0
    logits[20] = 1.0
    logits[45] = 2.0
    model_dir, calls = fake_language_detector_runtime(monkeypatch, tmp_path, logits)
    detector = live_stt.LanguageDetector(model_dir)
    pcm = np.array([-0.50, 0.25, 0.75], dtype=np.float32)

    argmax, score, ja, en = detector.score(pcm)

    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    assert argmax == "x0"
    assert score == pytest.approx(float(probabilities[0]))
    assert ja == pytest.approx(float(probabilities[45]))
    assert en == pytest.approx(float(probabilities[20]))
    assert len(calls["runs"]) == 1
    output_names, feeds = calls["runs"][0]
    # The graph's own names and its ['batch', 'samples'] input rank: the only defence the
    # gate has against a rename, weights being gitignored and absent in a fresh clone.
    assert output_names == ["logits"]
    assert list(feeds) == ["audio"]
    fed = np.asarray(feeds["audio"])
    assert fed.shape == (1, pcm.size)
    assert fed.dtype == np.float32
    np.testing.assert_array_equal(fed.reshape(-1), pcm)


def test_language_detector_decide_passes_one_score_to_the_pure_gate(monkeypatch) -> None:
    detector = object.__new__(live_stt.LanguageDetector)
    pcm = np.array([0.125, -0.25], dtype=np.float32)
    seen = {}

    def score(_self, actual_pcm):
        seen["pcm"] = actual_pcm
        return "en", 0.8, 0.1, 0.8

    def accept(*values):
        seen["values"] = values
        return "en"

    monkeypatch.setattr(live_stt.LanguageDetector, "score", score)
    monkeypatch.setattr(live_stt, "lid_accept", accept)

    assert detector.decide(pcm) == "en"
    assert seen["pcm"] is pcm
    assert seen["values"] == ("en", 0.8, 0.1, 0.8)
