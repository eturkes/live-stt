"""JA-tuned checkpoint: one-way JA=760M, EN=turbo, two-way=both resident.

Contract: .agent/deferred.md's JA-tuned Whisper checkpoint + spec Decisions.
Fake bindings/model markers isolate routing from weights, devices and codex.
"""

import asyncio
import sys
import types
from typing import Any, cast

import numpy as np
import pytest

import live_stt
from tests.test_shipped_path import _Chunk, _Result
from tests.test_shipped_path import model_tree as model_tree
from tests.test_shipped_path import openvino as openvino

WHISPER_MARKER = "openvino_encoder_model.xml"
DEVICES = ("CPU", "GPU", "NPU")
MODES = (("ja", False), ("en", False), ("ja", True))


@pytest.fixture
def routing_models(monkeypatch, model_tree, tmp_path):
    english_dir = tmp_path / "whisper-en"
    english_dir.mkdir()
    lid_dir = tmp_path / "lid/d2-ecapa"
    lid_dir.mkdir(parents=True)
    for name in ("voxlingua107.onnx", "lang_map.json"):
        (lid_dir / name).touch()
    monkeypatch.setattr(live_stt, "WHISPER_EN_DIR", english_dir, raising=False)
    monkeypatch.setattr(live_stt, "LID_MODEL_DIR", lid_dir)
    return model_tree["whisper"], english_dir, lid_dir


def _engine(tmp_path, device="CPU", english=False):
    # The extended API must remain typecheckable on the unfixed revision too.
    constructor = cast(Any, live_stt.WhisperEngine)
    kwargs = {"english_dir": tmp_path / "english"} if english else {}
    return constructor(tmp_path / "japanese", device, **kwargs)


def test_checkpoint_directories_keep_japanese_and_english_separate():
    assert live_stt.ENGINE_DIRS["whisper"] == (
        live_stt.MODELS_DIR / "openvino/whisper-ja-760m-int8-ov"
    )
    assert cast(Any, live_stt).WHISPER_EN_DIR == (
        live_stt.MODELS_DIR / "openvino/whisper-large-v3-turbo-int8-ov"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("english", [False, True], ids=["primary-only", "with-english"])
def test_constructor_builds_only_the_requested_resident_pipelines(
    openvino, tmp_path, device, english
):
    _engine(tmp_path, device, english)

    expected_dirs = [tmp_path / "japanese"] + ([tmp_path / "english"] if english else [])
    assert [pipeline.model_dir for pipeline in openvino] == [str(d) for d in expected_dirs]
    assert [pipeline.device for pipeline in openvino] == [device] * len(expected_dirs)
    assert [pipeline.kwargs for pipeline in openvino] == [
        {"CACHE_DIR": str(live_stt.OPENVINO_CACHE_DIR)}
    ] * len(expected_dirs)
    assert live_stt.OPENVINO_CACHE_DIR.is_dir()


def test_constructor_defaults_to_the_shipped_device(openvino, tmp_path):
    live_stt.WhisperEngine(tmp_path / "japanese")

    assert len(openvino) == 1
    assert openvino[0].device == live_stt.ASR_DEVICE == "NPU"


def test_explicit_none_english_directory_still_constructs_one_pipeline(openvino, tmp_path):
    cast(Any, live_stt.WhisperEngine)(tmp_path / "japanese", "CPU", english_dir=None)

    assert len(openvino) == 1
    assert openvino[0].model_dir == str(tmp_path / "japanese")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("language", ["ja", "en"])
@pytest.mark.parametrize("timestamps", [False, True])
def test_routed_generate_keeps_audio_and_generation_options(
    openvino, tmp_path, device, language, timestamps
):
    engine = _engine(tmp_path, device, english=True)
    primary, english = openvino
    primary.result, english.result = _Result(["日本語"]), _Result(["English"])
    engine.set_hotwords("東京、タワー")
    samples = np.linspace(-0.75, 0.75, 53, dtype=np.float32)
    selected, unused = (english, primary) if language == "en" else (primary, english)

    result = engine.generate(samples, timestamps=timestamps, language=language)

    assert result is selected.result
    assert unused.calls == []
    (call,) = selected.calls
    np.testing.assert_array_equal(call["samples"], samples)
    assert call["language"] == f"<|{language}|>"
    assert call["task"] == "transcribe"
    assert call["return_timestamps"] is timestamps
    assert call["repetition_penalty"] == live_stt.ASR_REPETITION_PENALTY
    if device in live_stt.ASR_HOTWORDS_DEVICES:
        assert call["hotwords"] == "東京、タワー"
    else:
        assert "hotwords" not in call


@pytest.mark.parametrize("language", ["ja", "en", "fr"])
def test_missing_english_pipeline_routes_every_language_to_primary(openvino, tmp_path, language):
    engine = _engine(tmp_path)
    (primary,) = openvino

    result = engine.generate(np.ones(5, dtype=np.float32), timestamps=False, language=language)

    assert result is primary.result
    assert primary.calls[0]["language"] == f"<|{language}|>"
    assert len(openvino) == 1


@pytest.mark.parametrize("english", [False, True], ids=["primary-only", "with-english"])
def test_generated_language_sequences_route_without_loading_more_pipelines(
    monkeypatch, openvino, tmp_path, english
):
    """Routing property: exact en selects the second resident pipeline, if present."""
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja")
    engine = _engine(tmp_path, english=english)
    rng = np.random.default_rng(760)
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz-<>|日本語"))
    languages = [None, "", "ja", "en", "EN", "en-US", "<|en|>"]
    for _ in range(128):
        languages.extend(["".join(rng.choice(alphabet, int(rng.integers(1, 17)))), "en"])
    expected_counts = [0] * (2 if english else 1)
    samples = np.linspace(-1, 1, 19, dtype=np.float32)
    for language in languages:
        index = int(english and language == "en")
        result = engine.generate(samples, timestamps=False, language=language)
        expected_counts[index] += 1
        assert result is openvino[index].result
        assert [len(pipeline.calls) for pipeline in openvino] == expected_counts
        assert len(openvino) == len(expected_counts)


@pytest.mark.parametrize("language", ["ja", "en"])
def test_timestamped_decode_returns_the_routed_pipeline_spans(openvino, tmp_path, language):
    engine = _engine(tmp_path, english=True)
    primary, english = openvino
    primary.result = _Result(["あい"], [_Chunk(0.0, 0.4, "あ"), _Chunk(0.4, 0.9, "い")])
    english.result = _Result(["hello"], [_Chunk(0.0, 0.7, "hello")])
    selected = english if language == "en" else primary

    text, spans = engine.decode_segments(np.ones(5, dtype=np.float32), language=language)

    assert text == "".join(selected.result.texts)
    assert [(span.start_s, span.end_s, span.text) for span in spans] == [
        (chunk.start_ts, chunk.end_ts, chunk.text) for chunk in selected.result.chunks
    ]
    assert selected.calls[0]["return_timestamps"] is True
    assert [len(pipeline.calls) for pipeline in openvino] == (
        [0, 1] if language == "en" else [1, 0]
    )


@pytest.mark.parametrize("language", ["ja", "en"])
def test_default_language_decode_selects_the_matching_pipeline(
    monkeypatch, openvino, tmp_path, language
):
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    engine = _engine(tmp_path, english=True)
    primary, english = openvino
    primary.result, english.result = _Result(["日本語"]), _Result(["English"])
    selected = english if language == "en" else primary

    assert engine.decode(np.ones(5, dtype=np.float32)) == selected.result.texts[0]
    assert selected.calls[0]["language"] == f"<|{language}|>"
    assert selected.calls[0]["return_timestamps"] is False
    assert [len(pipeline.calls) for pipeline in openvino] == (
        [0, 1] if language == "en" else [1, 0]
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "language,two_way", [*MODES, ("en", True)], ids=["ja", "en", "two-way", "en-priority"]
)
def test_loader_constructs_the_mode_specific_pipelines(
    monkeypatch, openvino, routing_models, device, language, two_way
):
    primary, english, _lid = routing_models
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)

    engine = cast(Any, live_stt.load_recognizer)("whisper", device, two_way=two_way)

    expected = [english] if language == "en" else [primary] + ([english] if two_way else [])
    assert isinstance(engine, live_stt.WhisperEngine)
    assert [pipeline.model_dir for pipeline in openvino] == [str(d) for d in expected]
    assert [pipeline.device for pipeline in openvino] == [device] * len(expected)


@pytest.mark.parametrize("language", ["ja", "en"])
def test_loader_default_two_way_constructs_only_the_pinned_model(
    monkeypatch, openvino, routing_models, language
):
    primary, english, _lid = routing_models
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)

    live_stt.load_recognizer("whisper")

    assert len(openvino) == 1
    assert openvino[0].model_dir == str(english if language == "en" else primary)
    assert openvino[0].device == live_stt.ASR_DEVICE


@pytest.mark.parametrize("engine_name", ["k2v2", "parakeet"])
def test_sherpa_loader_keeps_its_own_model_without_openvino(monkeypatch, openvino, engine_name):
    seen = []
    recognizer = object()

    def build(**kwargs):
        seen.append(kwargs)
        return recognizer

    monkeypatch.setattr(
        live_stt,
        "sherpa_onnx",
        types.SimpleNamespace(
            OfflineRecognizer=types.SimpleNamespace(from_transducer=build, from_nemo_ctc=build)
        ),
    )
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "en")

    assert live_stt.load_recognizer(engine_name, "NPU") is recognizer
    assert openvino == []
    (kwargs,) = seen
    model_arg = kwargs["encoder"] if engine_name == "k2v2" else kwargs["model"]
    assert model_arg.startswith(str(live_stt.ENGINE_DIRS[engine_name]))


@pytest.mark.parametrize("present", range(4), ids=["neither", "ja-only", "en-only", "both"])
@pytest.mark.parametrize("language,two_way", MODES, ids=["ja", "en", "two-way"])
def test_preflight_requires_only_the_models_that_the_mode_can_decode(
    monkeypatch, routing_models, language, two_way, present
):
    primary, english, _lid = routing_models
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    for index, directory in enumerate((primary, english)):
        if present & (1 << index):
            (directory / WHISPER_MARKER).touch()
    required = [primary, english] if two_way else [english if language == "en" else primary]
    missing = [directory for directory in required if not (directory / WHISPER_MARKER).is_file()]
    # Omit the default argument in one-way mode: these cases isolate path selection
    # from the newly added two_way API, rather than all reddening on a TypeError.
    kwargs = {"two_way": True} if two_way else {}

    error = cast(Any, live_stt.check_models)("whisper", **kwargs)

    if missing:
        assert error is not None
        for directory in missing:
            assert f"{directory.name}/" in error
    else:
        assert error is None


@pytest.mark.parametrize("language", ["ja", "en"])
def test_wrong_marker_does_not_satisfy_the_selected_whisper_model(
    monkeypatch, routing_models, language
):
    primary, english, _lid = routing_models
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    selected = english if language == "en" else primary
    (selected / "tokens.txt").touch()

    error = live_stt.check_models("whisper")

    assert error is not None
    assert f"{selected.name}/" in error


@pytest.mark.parametrize("missing_name", ["voxlingua107.onnx", "lang_map.json"])
def test_two_way_preflight_retains_both_lid_requirements(monkeypatch, routing_models, missing_name):
    primary, english, lid = routing_models
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja")
    for directory in (primary, english):
        (directory / WHISPER_MARKER).touch()
    (lid / missing_name).unlink()

    error = cast(Any, live_stt.check_models)("whisper", two_way=True)

    assert error is not None
    assert "d2-ecapa" in error


@pytest.mark.parametrize("language", ["ja", "en"])
def test_one_way_preflight_keeps_lid_optional(monkeypatch, routing_models, language):
    primary, english, lid = routing_models
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    selected = english if language == "en" else primary
    (selected / WHISPER_MARKER).touch()
    for path in lid.iterdir():
        path.unlink()

    assert live_stt.check_models("whisper") is None


@pytest.mark.parametrize("two_way", [False, True], ids=["one-way", "two-way"])
def test_run_session_passes_its_two_way_flag_to_the_loader(monkeypatch, tmp_path, two_way):
    seen = []
    captured = []

    def load_recognizer(engine_name, device, two_way=None):
        seen.append((engine_name, device, two_way))
        return object()

    class InputStream:
        def __init__(self, callback, **_kwargs):
            self.callback = callback

        def start(self):
            block = np.full((160, 1), 0.25, dtype=np.float32)
            self.callback(block, len(block), None, None)

        def stop(self):
            pass

        def close(self):
            pass

    async def worker(_rec, _vad, _window, audio_q, state, *_args, **_kwargs):
        while (chunk := await audio_q.get()) is not None:
            captured.append(chunk)
            state.request_stop()

    async def meter(*_args, **_kwargs):
        pass

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(
            InputStream=InputStream,
            query_devices=lambda *_a, **_k: {"default_samplerate": 16000, "name": "fake"},
        ),
    )
    monkeypatch.setattr(live_stt, "load_recognizer", load_recognizer)
    monkeypatch.setattr(live_stt, "make_vad", lambda: (None, 512))
    monkeypatch.setattr(live_stt, "LanguageDetector", lambda *_a, **_k: object())
    monkeypatch.setattr(live_stt, "worker", worker)
    monkeypatch.setattr(live_stt, "meter", meter)
    monkeypatch.setattr(live_stt, "_install_signal_handlers", lambda _state: None)
    monkeypatch.setattr(live_stt, "TRANSCRIPT_DIR", tmp_path)
    args = types.SimpleNamespace(
        engine="whisper",
        asr_device="GPU",
        two_way=two_way,
        device=None,
        output=None,
        no_save=True,
        no_translate=True,
        save_audio=False,
        context="",
    )

    asyncio.run(asyncio.wait_for(live_stt.run_session(args), 3.0))

    assert seen == [("whisper", "GPU", two_way)]
    assert len(captured) == 1
