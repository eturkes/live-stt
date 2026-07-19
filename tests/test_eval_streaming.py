"""Model-free locks for M10.5d's full-corpus streaming evaluator machinery."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests import eval_models as shared
from tests import eval_streaming as evaluator

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "streaming_baseline.json"


class FakeStream:
    def __init__(self):
        self.accepted_blocks = 0
        self.current_text = ""
        self.eof_count = 0
        self.eof_pending = False
        self.option: dict[str, str] = {}
        self.pending = False
        self.reset_count = 0

    def set_option(self, name: str, value: str) -> None:
        self.option[name] = value

    def get_option(self, name: str) -> str:
        return self.option[name]

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        assert sample_rate == evaluator.shared.SAMPLE_RATE
        assert samples.size == evaluator.BLOCK_SAMPLES
        self.accepted_blocks += 1
        self.pending = True

    def input_finished(self) -> None:
        self.eof_count += 1
        self.eof_pending = True
        self.pending = True


class FakeRecognizer:
    def __init__(self, *, endpoint_text: str = "一", final_text: str = "二"):
        self.stream = FakeStream()
        self.endpoint_collected = False
        self.endpoint_text = endpoint_text
        self.final_text = final_text

    def create_stream(self) -> FakeStream:
        return self.stream

    def is_ready(self, stream: FakeStream) -> bool:
        return stream.pending

    def decode_stream(self, stream: FakeStream) -> None:
        if stream.eof_pending:
            stream.eof_pending = False
        elif stream.accepted_blocks == 1:
            stream.current_text = ""
        elif stream.accepted_blocks == 2:
            stream.current_text = self.endpoint_text
        else:
            stream.current_text = self.final_text
        stream.pending = False

    def get_result(self, stream: FakeStream) -> str:
        return stream.current_text

    def is_endpoint(self, stream: FakeStream) -> bool:
        return stream.accepted_blocks == 2 and not self.endpoint_collected

    def reset(self, stream: FakeStream) -> None:
        self.endpoint_collected = True
        stream.current_text = ""
        stream.reset_count += 1


def _committed_baseline() -> dict:
    if not BASELINE.exists():
        pytest.skip("M10.5c streaming baseline not generated yet")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _fake_decode(case: shared.EvalCase, hypothesis: str) -> evaluator.StreamingDecodeObservation:
    recognizer = FakeRecognizer(
        endpoint_text=hypothesis[:1],
        final_text=hypothesis[1:],
    )
    return evaluator._decode_samples(
        recognizer,
        np.zeros(case.duration_samples, dtype=np.float32),
        adapter_id=f"fake/{case.case_id}",
        rss_sampler=lambda: 100.0,
    )


def _synthetic_corpus() -> tuple[list[shared.EvalCase], dict[str, str]]:
    duration_samples = 3 * evaluator.BLOCK_SAMPLES
    cases = [
        shared.EvalCase(
            "cv-1",
            "common_voice_8",
            Path("unused-cv-1.wav"),
            "一二",
            duration_samples,
            duration_bucket="0-5",
        ),
        shared.EvalCase(
            "cv-2",
            "common_voice_8",
            Path("unused-cv-2.wav"),
            "一三",
            duration_samples,
            duration_bucket="0-5",
        ),
        shared.EvalCase(
            "fleurs-1",
            "fleurs",
            Path("unused-fleurs-1.wav"),
            "四五",
            duration_samples,
            gender="female",
            duration_bucket="0-5",
        ),
    ]
    return cases, {"cv-1": "一二", "cv-2": "一二", "fleurs-1": "四"}


def _passing_candidate(comparator: dict) -> dict:
    common_voice = comparator["aggregates"]["by_source"]["common_voice_8"]["micro"]
    fleurs = comparator["aggregates"]["by_source"]["fleurs"]["micro"]
    long_form = copy.deepcopy(comparator["compatibility"]["long_form"][shared.LONG_FORM_ID])
    return {
        "aggregates": {
            "by_source": {
                "common_voice_8": {"micro": {"D": 0, "I": 0, "N": common_voice["N"], "S": 7000}},
                "fleurs": {
                    "micro": {
                        "D": 0,
                        "I": 0,
                        "N": fleurs["N"],
                        "S": fleurs["S"] + fleurs["D"] + fleurs["I"],
                    }
                },
            },
            "completion": {
                "complete": True,
                "empty_hypotheses": 0,
                "rows": 3,
            },
        },
        "compatibility": {
            "long_form": {shared.LONG_FORM_ID: long_form},
            "stressors": {
                case_id: {
                    "D": 0,
                    "I": 0,
                    "N": 100,
                    "S": 10,
                    "excess_D": 4,
                }
                for case_id in comparator["compatibility"]["stressors"]
            },
        },
        "diagnostics": {
            "content_event_case_ids": [],
            "content_events": [],
        },
    }


def test_direct_stream_loop_accounts_partial_endpoint_reset_and_eof_once():
    recognizer = FakeRecognizer()
    rss_values = iter((100.0, 101.0, 102.0, 103.0, 104.0))
    observation = evaluator._decode_samples(
        recognizer,
        np.zeros(3 * evaluator.BLOCK_SAMPLES, dtype=np.float32),
        adapter_id="fake",
        rss_sampler=lambda: next(rss_values),
        rss_sample_blocks=1,
    )

    transcript = observation.content
    assert transcript.hypothesis == "一二"
    assert transcript.accepted_samples == 3 * evaluator.BLOCK_SAMPLES
    assert transcript.eof_count == recognizer.stream.eof_count == 1
    assert transcript.complete is True
    assert transcript.partial_update_count == 2
    assert transcript.finalization_count == 2
    assert transcript.segment_reset_count == recognizer.stream.reset_count == 1
    assert transcript.first_text_logical_audio_s == 0.04
    assert transcript.finalization_logical_audio_s == (0.04, 0.06)
    assert transcript.segments == (
        {
            "finalization_logical_audio_s": 0.04,
            "first_text_logical_audio_s": 0.04,
            "index": 0,
            "partial_update_count": 1,
            "reason": "endpoint",
            "text": "一",
        },
        {
            "finalization_logical_audio_s": 0.06,
            "first_text_logical_audio_s": 0.06,
            "index": 1,
            "partial_update_count": 1,
            "reason": "eof",
            "text": "二",
        },
    )
    assert observation.rss_peak_mib == 104.0
    assert [sample["phase"] for sample in observation.rss_samples_mib] == [
        "start",
        "feed",
        "feed",
        "feed",
        "eof",
    ]


def test_streaming_score_keeps_logical_events_out_of_measurements():
    transcript = evaluator.StreamingTranscript(
        hypothesis="一二",
        segments=(
            {
                "finalization_logical_audio_s": 0.06,
                "first_text_logical_audio_s": 0.04,
                "index": 0,
                "partial_update_count": 2,
                "reason": "eof",
                "text": "一二",
            },
        ),
        accepted_samples=3 * evaluator.BLOCK_SAMPLES,
        eof_count=1,
        complete=True,
        partial_update_count=2,
        finalization_count=1,
        segment_reset_count=0,
        first_text_logical_audio_s=0.04,
        finalization_logical_audio_s=(0.06,),
    )
    case = shared.EvalCase(
        "fake",
        "unit",
        Path("unused.wav"),
        "一三",
        3 * evaluator.BLOCK_SAMPLES,
        duration_bucket="0-5",
    )

    row = evaluator._streaming_score(case, transcript)

    assert (row["N"], row["S"], row["D"], row["I"]) == (2, 1, 0, 0)
    assert row["finalization_count"] == row["n_segments"] == 1
    assert row["segment_reset_count"] == 0
    assert row["finalization_logical_audio_s"] == [0.06]
    assert not {"decode_seconds", "decode_rtf", "rss_mib", "wall_seconds", "wall_rtf"} & row.keys()
    evaluator._validate_streaming_row(case, row)


def test_shared_model_set_validator_accepts_streaming_ids_and_rejects_partial_sets():
    model_ids = ("stream-a", "stream-b")
    summaries = {model_id: {} for model_id in model_ids}
    details = {model_id: Path(f"{model_id}.jsonl") for model_id in model_ids}

    assert (
        shared.validate_evidence_model_set(summaries, details, model_ids, label="streaming test")
        == model_ids
    )
    with pytest.raises(RuntimeError, match="required model set"):
        shared.validate_evidence_model_set(
            summaries,
            {"stream-a": details["stream-a"]},
            model_ids,
            label="streaming test",
        )


def test_full_tournament_orders_fast_variants_before_long_poles():
    assert evaluator.TOURNAMENT_VARIANTS == ("1120ms", "560ms", "160ms", "80ms")
    assert evaluator.STREAMING_MODEL_IDS == tuple(
        evaluator.streaming_models.CANDIDATE_SPECS[name].model_id
        for name in evaluator.TOURNAMENT_VARIANTS
    )


def test_fixed_parakeet_comparator_snapshot_includes_both_corpus_sources():
    comparator = evaluator._comparator_snapshot()

    assert comparator["engine"] == "parakeet"
    assert comparator["common_voice_micro_cer"] == 0.08426233
    assert comparator["fleurs_micro_cer"] == 0.10444744
    assert comparator["long_form_cer"] == 0.23571945
    assert comparator["aggregates"]["by_source"]["common_voice_8"]["micro"]["cer"] == 0.08426233
    assert comparator["aggregates"]["by_source"]["fleurs"]["micro"]["cer"] == 0.10444744


def test_runtime_incompatibility_produces_failed_content_and_resource_verdicts():
    summary = evaluator._runtime_failure_summary(
        "fake-model",
        {"sha256": "model"},
        {"corpus_index_sha256": "corpus"},
        RuntimeError("forced-ja unsupported"),
    )
    variant = evaluator._runtime_failure_result(summary)
    resource = shared.resource_verdict({})

    assert summary["status"] == variant["status"] == "runtime_incompatible"
    assert variant["failure"] == {
        "message": "forced-ja unsupported",
        "phase": "adapter_initialization",
        "type": "RuntimeError",
    }
    assert variant["content_verdict"]["qualified"] is False
    assert resource == {"paths": {}, "qualified": False}
    assert evaluator._displacement_verdict(
        variant["content_verdict"], resource, non_dominated=False
    ) == {
        "content_qualified": False,
        "displacement_qualified": False,
        "non_dominated": False,
        "resource_qualified": False,
    }


def test_synthetic_streaming_corpus_aggregates_by_source_and_strata(tmp_path: Path):
    cases, hypotheses = _synthetic_corpus()
    detail = tmp_path / "synthetic.jsonl"
    result = evaluator._write_corpus_detail_resumable(
        detail,
        cases,
        lambda case: _fake_decode(case, hypotheses[case.case_id]),
        {"engine": "fake", "schema_version": evaluator.SCHEMA_VERSION},
    )
    rows = evaluator._detail_rows(detail, cases, result["sha256"])

    aggregates = shared.aggregate_rows(rows)
    common_voice = aggregates["by_source"]["common_voice_8"]
    fleurs = aggregates["by_source"]["fleurs"]
    assert common_voice["rows"] == 2
    assert common_voice["micro"] == {"D": 0, "I": 0, "N": 4, "S": 1, "cer": 0.25}
    assert common_voice["macro"]["cer"] == 0.25
    assert fleurs["rows"] == 1
    assert fleurs["micro"] == {"D": 1, "I": 0, "N": 2, "S": 0, "cer": 0.5}
    assert aggregates["fleurs_gender"]["female"]["micro"]["cer"] == 0.5
    assert aggregates["fleurs_duration"]["0-5"]["rows"] == 1
    assert aggregates["completion"] == {
        "accepted_all_audio_rows": 3,
        "complete": True,
        "complete_rows": 3,
        "empty_case_ids": [],
        "empty_hypotheses": 0,
        "eof_once_rows": 3,
        "rows": 3,
    }
    assert result["measurement"]["rows"] == 3
    assert result["measurement"]["rows_reused_on_resume"] == 0
    assert "runs" not in result["measurement"]


def _synthetic_manifest_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    corpus_cases, hypotheses = _synthetic_corpus()
    detail = tmp_path / "manifest-corpus.jsonl"
    detail_result = evaluator._write_corpus_detail_resumable(
        detail,
        corpus_cases,
        lambda case: _fake_decode(case, hypotheses[case.case_id]),
        {"engine": "manifest-fake", "schema_version": evaluator.SCHEMA_VERSION},
    )

    def compatibility_case(case_id: str, source: str, reference: str) -> shared.EvalCase:
        wav = tmp_path / f"{case_id}.wav"
        wav.write_bytes(case_id.encode())
        return shared.EvalCase(
            case_id,
            source,
            wav,
            reference,
            3 * evaluator.BLOCK_SAMPLES,
            duration_bucket="0-5",
        )

    groups = {
        "short": [compatibility_case("short-a", "compatibility", "一二")],
        "stressors": [
            compatibility_case("stress_long", "stressor", "一二"),
            compatibility_case("stress_med", "stressor", "一二"),
        ],
        "long_form": [compatibility_case(shared.LONG_FORM_ID, "long_form", "一二")],
        "retention": [compatibility_case("retention_probe", "retention_probe", "一二")],
    }
    compatibility: dict[str, dict[str, dict]] = {group: {} for group in evaluator.GROUP_ORDER}
    expected = {
        "stressor_manifest": {
            "stressors": {
                "stress_long": {"order": ["short-a"]},
                "stress_med": {"order": ["short-a"]},
            }
        }
    }
    for group in evaluator.GROUP_ORDER:
        for case in groups[group]:
            row = evaluator._streaming_score(case, _fake_decode(case, "一二").content)
            if group == "stressors":
                evaluator._add_stressor_baseline(row, compatibility["short"], expected)
            compatibility[group][case.case_id] = row

    def synthetic_artifact(path: Path) -> dict:
        return {
            "bytes": path.stat().st_size,
            "path": path.name,
            "sha256": shared.file_sha256(path),
        }

    monkeypatch.setattr(evaluator.shared, "_artifact", synthetic_artifact)
    corpus_manifest = {
        "cache": {"index_sha256": "synthetic-index", "rows": len(corpus_cases)},
        "sources": {
            source: {
                "revision": "synthetic",
                "source_identity": source,
                "statistics": {"rows": sum(case.source == source for case in corpus_cases)},
            }
            for source in ("common_voice_8", "fleurs")
        },
    }
    inputs = evaluator.evaluation_inputs(groups, corpus_manifest)
    complete_id = evaluator.streaming_models.CANDIDATE_SPECS["1120ms"].model_id
    failed_id = evaluator.streaming_models.CANDIDATE_SPECS["80ms"].model_id
    models = {
        complete_id: {"fingerprint": "complete"},
        failed_id: {"fingerprint": "failed"},
    }
    monkeypatch.setattr(
        evaluator,
        "model_fingerprint",
        lambda spec: models[spec.model_id],
    )
    complete_summary = {
        "adapter": {
            "actual_execution_device": "CPU",
            "model": models[complete_id],
        },
        "compatibility": compatibility,
        "details": {
            "rows": len(corpus_cases),
            "sha256": detail_result["sha256"],
        },
        "diagnostics": evaluator._streaming_diagnostics(),
        "engine": complete_id,
        "inputs": inputs,
        "measurements": {
            "cases": {group: {} for group in evaluator.GROUP_ORDER},
            "corpus": {"rows": len(corpus_cases)},
            "post_warm": {"median_decode_rtf": 0.1},
        },
        "schema_version": evaluator.SCHEMA_VERSION,
        "status": "complete",
    }
    failed_summary = evaluator._runtime_failure_summary(
        failed_id,
        models[failed_id],
        inputs,
        RuntimeError("forced-ja unsupported"),
    )
    return {
        "compatibility": compatibility,
        "complete_id": complete_id,
        "corpus_cases": corpus_cases,
        "corpus_manifest": corpus_manifest,
        "detail": detail,
        "detail_result": detail_result,
        "expected": expected,
        "failed_id": failed_id,
        "groups": groups,
        "inputs": inputs,
        "models": models,
        "summaries": {
            complete_id: complete_summary,
            failed_id: failed_summary,
        },
    }


def test_build_manifest_separates_corpus_detail_compatibility_and_runtime_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _synthetic_manifest_setup(tmp_path, monkeypatch)
    complete_id = setup["complete_id"]
    failed_id = setup["failed_id"]
    manifest = evaluator.build_manifest(
        setup["corpus_manifest"],
        setup["corpus_cases"],
        setup["summaries"],
        {complete_id: setup["detail"], failed_id: tmp_path / "no-detail.jsonl"},
        (complete_id, failed_id),
        setup["groups"],
        setup["expected"],
    )

    complete = manifest["deterministic"]["variants"][complete_id]
    failed = manifest["deterministic"]["variants"][failed_id]
    assert manifest["deterministic"]["scope"]["full_short_corpus_rows"] == 3
    assert complete["aggregates"] == shared.aggregate_rows(
        evaluator._detail_rows(
            setup["detail"],
            setup["corpus_cases"],
            setup["detail_result"]["sha256"],
        )
    )
    assert complete["compatibility"] == setup["compatibility"]
    assert complete["details"]["rows"] == 3
    assert failed["status"] == "runtime_incompatible"
    assert failed["content_verdict"]["qualified"] is False
    assert manifest["measurements"]["non_dominated"]["variants"] == [complete_id]
    assert manifest["measurements"]["variants"][failed_id]["resource_verdict"] == {
        "paths": {},
        "qualified": False,
    }


def test_reusable_child_accepts_exact_summary_and_rejects_input_or_logic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _synthetic_manifest_setup(tmp_path, monkeypatch)
    model_id = setup["complete_id"]
    summary = setup["summaries"][model_id]
    summary_path = tmp_path / "exact-summary.json"
    shared.write_atomic(summary_path, [shared._json_bytes(summary)])

    reused = evaluator._reusable_child(
        model_id,
        summary_path,
        setup["detail"],
        setup["corpus_manifest"],
        setup["corpus_cases"],
        setup["groups"],
        setup["expected"],
    )

    assert reused == summary
    assert summary_path.is_file()
    assert setup["detail"].is_file()

    detail_bytes = setup["detail"].read_bytes()
    for drift in ("corpus_input", "decision_logic"):
        stale_summary = copy.deepcopy(summary)
        if drift == "corpus_input":
            stale_summary["inputs"]["corpus_index_sha256"] = "stale-index"
        else:
            stale_summary["inputs"]["pipeline"]["contract_sha256"] = "0" * 64
        stale_summary_path = tmp_path / f"{drift}-summary.json"
        stale_detail_path = tmp_path / f"{drift}.jsonl"
        shared.write_atomic(stale_summary_path, [shared._json_bytes(stale_summary)])
        stale_detail_path.write_bytes(detail_bytes)

        assert (
            evaluator._reusable_child(
                model_id,
                stale_summary_path,
                stale_detail_path,
                setup["corpus_manifest"],
                setup["corpus_cases"],
                setup["groups"],
                setup["expected"],
            )
            is None
        )
        assert not stale_summary_path.exists()
        assert not stale_detail_path.exists()


def test_reaggregate_parent_rebuilds_full_manifest_and_rejects_measurement_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    setup = _synthetic_manifest_setup(tmp_path, monkeypatch)
    complete_id = setup["complete_id"]
    failed_id = setup["failed_id"]
    model_ids = (complete_id, failed_id)
    manifest = evaluator.build_manifest(
        setup["corpus_manifest"],
        setup["corpus_cases"],
        setup["summaries"],
        {complete_id: setup["detail"], failed_id: tmp_path / "no-detail.jsonl"},
        model_ids,
        setup["groups"],
        setup["expected"],
    )
    baseline = tmp_path / "streaming-baseline.json"
    shared.write_atomic(baseline, [shared._json_bytes(manifest)])

    detail_relative = Path(manifest["deterministic"]["variants"][complete_id]["details"]["path"])
    cached_detail = tmp_path / detail_relative
    cached_detail.parent.mkdir(parents=True)
    cached_detail.write_bytes(setup["detail"].read_bytes())

    monkeypatch.setattr(evaluator, "BASELINE", baseline)
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(
        evaluator.shared,
        "load_corpus_cases",
        lambda *, verify_pcm: (setup["corpus_manifest"], setup["corpus_cases"]),
    )
    monkeypatch.setattr(
        evaluator,
        "small_clip_cases",
        lambda *, verify_audio: (setup["groups"], setup["expected"]),
    )
    monkeypatch.setattr(
        evaluator,
        "evaluation_inputs",
        lambda groups, corpus_manifest: setup["inputs"],
    )
    installed = []

    def capture_install(installed_manifest, summaries, details, installed_model_ids):
        installed.append(
            {
                "manifest": installed_manifest,
                "model_ids": tuple(installed_model_ids),
            }
        )

    monkeypatch.setattr(evaluator, "install_evidence", capture_install)
    evaluator.reaggregate_parent()

    assert installed == [{"manifest": manifest, "model_ids": model_ids}]
    assert baseline.read_bytes() == shared._json_bytes(manifest)

    build_manifest = evaluator.build_manifest

    def drifted_build_manifest(*args, **kwargs):
        rebuilt = build_manifest(*args, **kwargs)
        rebuilt["measurements"]["non_dominated"]["objectives"].append("logic-drift")
        return rebuilt

    monkeypatch.setattr(evaluator, "build_manifest", drifted_build_manifest)
    with pytest.raises(RuntimeError, match="byte rebuild drifted"):
        evaluator.reaggregate_parent()
    assert len(installed) == 1


def test_resumed_streaming_corpus_detail_is_byte_identical_to_fresh(tmp_path: Path):
    cases, hypotheses = _synthetic_corpus()
    identity = {"engine": "fake", "schema_version": evaluator.SCHEMA_VERSION}
    fresh = tmp_path / "fresh.jsonl"
    resumed = tmp_path / "resumed.jsonl"
    fresh_result = evaluator._write_corpus_detail_resumable(
        fresh,
        cases,
        lambda case: _fake_decode(case, hypotheses[case.case_id]),
        identity,
    )

    decoded = 0

    def interrupt_after_two(
        case: shared.EvalCase,
    ) -> evaluator.StreamingDecodeObservation:
        nonlocal decoded
        if decoded == 2:
            raise RuntimeError("injected corpus interruption")
        decoded += 1
        return _fake_decode(case, hypotheses[case.case_id])

    with pytest.raises(RuntimeError, match="injected corpus interruption"):
        evaluator._write_corpus_detail_resumable(
            resumed,
            cases,
            interrupt_after_two,
            identity,
        )
    part_path, measurement_path, _ = evaluator._resume_paths(resumed)
    assert not resumed.exists()
    assert part_path.exists()
    with part_path.open("ab") as partial:
        partial.write(b'{"truncated"')
    measurement_lines = measurement_path.read_bytes().splitlines(keepends=True)
    assert len(measurement_lines) == 2
    measurement_path.write_bytes(measurement_lines[0])

    resumed_result = evaluator._write_corpus_detail_resumable(
        resumed,
        cases,
        lambda case: _fake_decode(case, hypotheses[case.case_id]),
        identity,
    )

    assert resumed.read_bytes() == fresh.read_bytes()
    assert resumed_result["sha256"] == fresh_result["sha256"]
    assert resumed_result["measurement"]["rows_reused_on_resume"] == 1
    assert not part_path.exists()

    redecoded = 0

    def decode_after_identity_change(
        case: shared.EvalCase,
    ) -> evaluator.StreamingDecodeObservation:
        nonlocal redecoded
        redecoded += 1
        return _fake_decode(case, hypotheses[case.case_id])

    invalidated = evaluator._write_corpus_detail_resumable(
        resumed,
        cases,
        decode_after_identity_change,
        {"engine": "different-fake", "schema_version": evaluator.SCHEMA_VERSION},
    )
    assert redecoded == len(cases)
    assert invalidated["measurement"]["rows_reused_on_resume"] == 0
    assert resumed.read_bytes() == fresh.read_bytes()


def test_streaming_content_verdict_passes_every_fixed_parakeet_gate():
    comparator = evaluator._comparator_snapshot()
    verdict = shared.content_verdict(_passing_candidate(comparator), comparator)

    assert verdict["qualified"] is True
    assert all(check["pass"] for check in verdict["checks"].values())


@pytest.mark.parametrize(
    "failed_check",
    [
        "common_voice_relative_cer",
        "complete_run",
        "decoder_diagnostics",
        "fleurs_micro_cer_regression",
        "long_form_cer_regression",
        "nonempty_run",
        "stressors_cer",
        "stressors_complete",
        "stressors_excess_deletion",
    ],
)
def test_streaming_content_verdict_fails_each_gate_independently(failed_check: str):
    comparator = evaluator._comparator_snapshot()
    candidate = _passing_candidate(comparator)
    if failed_check == "common_voice_relative_cer":
        candidate["aggregates"]["by_source"]["common_voice_8"]["micro"]["S"] = 7100
    elif failed_check == "complete_run":
        candidate["aggregates"]["completion"]["complete"] = False
    elif failed_check == "decoder_diagnostics":
        candidate["diagnostics"] = {
            "content_event_case_ids": ["cv-1"],
            "content_events": [{"case_id": "cv-1"}],
        }
    elif failed_check == "fleurs_micro_cer_regression":
        micro = candidate["aggregates"]["by_source"]["fleurs"]["micro"]
        micro.update({"D": 0, "I": 0, "S": 3800})
    elif failed_check == "long_form_cer_regression":
        row = candidate["compatibility"]["long_form"][shared.LONG_FORM_ID]
        row.update({"D": 0, "I": 0, "S": 340})
    elif failed_check == "nonempty_run":
        completion = candidate["aggregates"]["completion"]
        completion["empty_hypotheses"] = completion["rows"]
    elif failed_check == "stressors_cer":
        row = next(iter(candidate["compatibility"]["stressors"].values()))
        row.update({"D": 0, "I": 0, "S": 16})
    elif failed_check == "stressors_complete":
        candidate["compatibility"]["stressors"].pop(
            next(iter(candidate["compatibility"]["stressors"]))
        )
    else:
        row = next(iter(candidate["compatibility"]["stressors"].values()))
        row["excess_D"] = 5

    verdict = shared.content_verdict(candidate, comparator)

    assert verdict["qualified"] is False
    assert verdict["checks"][failed_check]["pass"] is False
    assert [name for name, check in verdict["checks"].items() if not check["pass"]] == [
        failed_check
    ]


def test_streaming_non_dominated_set_and_combined_displacement_verdict():
    def variant(errors: int) -> dict:
        return {
            "aggregates": {
                "by_source": {"common_voice_8": {"micro": {"D": 0, "I": 0, "N": 100, "S": errors}}}
            }
        }

    variants = {
        "accuracy": variant(8),
        "dominated": variant(10),
        "speed": variant(9),
    }
    measurements = {
        "accuracy": {"post_warm": {"median_decode_rtf": 0.18}},
        "dominated": {"post_warm": {"median_decode_rtf": 0.20}},
        "speed": {"post_warm": {"median_decode_rtf": 0.10}},
    }

    assert evaluator._non_dominated_variants(variants, measurements) == ["accuracy", "speed"]

    exact_variants = {
        "exact": {
            "aggregates": {
                "by_source": {
                    "common_voice_8": {
                        "micro": {"D": 0, "I": 0, "N": 1_000_000_000, "S": 80_000_000}
                    }
                }
            }
        },
        "rounded_worse": {
            "aggregates": {
                "by_source": {
                    "common_voice_8": {
                        "micro": {"D": 0, "I": 0, "N": 1_000_000_000, "S": 80_000_001}
                    }
                }
            }
        },
    }
    equal_rtf = {model_id: {"post_warm": {"median_decode_rtf": 0.1}} for model_id in exact_variants}
    assert round(80_000_000 / 1_000_000_000, 8) == round(80_000_001 / 1_000_000_000, 8)
    assert evaluator._non_dominated_variants(exact_variants, equal_rtf) == ["exact"]

    resource = shared.resource_verdict(
        {
            "sherpa_cpu": {
                "actual_execution_device": "CPU",
                "post_warm_rtf": 0.18,
                "production_eligible": True,
            }
        }
    )
    assert evaluator._displacement_verdict({"qualified": True}, resource, non_dominated=True) == {
        "content_qualified": True,
        "displacement_qualified": True,
        "non_dominated": True,
        "resource_qualified": True,
    }

    slow_resource = shared.resource_verdict(
        {
            "sherpa_cpu": {
                "actual_execution_device": "CPU",
                "post_warm_rtf": 0.200001,
                "production_eligible": True,
            }
        }
    )
    assert slow_resource["qualified"] is False
    assert evaluator._displacement_verdict(
        {"qualified": True}, slow_resource, non_dominated=False
    ) == {
        "content_qualified": True,
        "displacement_qualified": False,
        "non_dominated": False,
        "resource_qualified": False,
    }


def test_committed_streaming_baseline_locks_rows_events_sources_and_measurements():
    baseline = _committed_baseline()
    deterministic = baseline["deterministic"]
    scope = deterministic.get("scope", {})
    if scope.get("full_short_corpus_rows") != 5133:
        pytest.skip("committed streaming baseline remains the M10.5c schema-1 evidence")
    groups, expected = evaluator.small_clip_cases(verify_audio=False)

    assert baseline["schema_version"] == evaluator.SCHEMA_VERSION == 2
    assert (
        baseline["deterministic_sha256"]
        == hashlib.sha256(shared._json_bytes(deterministic)).hexdigest()
    )
    assert deterministic["metric_contract"] == evaluator.STREAMING_METRIC_CONTRACT
    assert deterministic["displacement_gates"] == shared.DISPLACEMENT_GATES
    assert scope["m10_unit"] == "M10.5d"
    assert scope["case_groups"] == {group: len(groups[group]) for group in evaluator.GROUP_ORDER}
    assert deterministic["corpus"]["rows"] == 5133
    assert deterministic["inputs"]["corpus_index_sha256"] == deterministic["corpus"]["index_sha256"]
    assert deterministic["comparator"] == evaluator._comparator_snapshot()
    assert set(deterministic["variants"]) == set(evaluator.STREAMING_MODEL_IDS)
    assert baseline["measurements"]["excluded_from_deterministic_equality"] is True

    for artifact in deterministic["inputs"]["pipeline"]["manifests"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert shared.file_sha256(path) == artifact["sha256"]

    measurements_by_variant = baseline["measurements"]["variants"]
    for model_id, variant in deterministic["variants"].items():
        measurements = measurements_by_variant[model_id]
        if variant["status"] == "runtime_incompatible":
            assert variant["content_verdict"]["qualified"] is False
            assert variant["failure"]["phase"] == "adapter_initialization"
            assert measurements["resource_verdict"]["qualified"] is False
            assert measurements["displacement_verdict"] == {
                "content_qualified": False,
                "displacement_qualified": False,
                "non_dominated": False,
                "resource_qualified": False,
            }
            continue
        assert variant["status"] == "complete"
        adapter = variant["adapter"]
        assert adapter["architecture"] == "cache_aware_streaming_transducer"
        assert adapter["block_ms"] == 20
        assert adapter["forced_language"]["value"] == "ja"
        assert adapter["provider"] == "cpu"
        assert adapter["actual_execution_device"] == "CPU"
        assert adapter["endpoint_detection"]["enabled"] is True
        assert adapter["model"]["provenance"]["license"]["spdx"] == "OpenMDW-1.1"
        assert variant["details"]["rows"] == 5133
        assert variant["diagnostics"] == evaluator._streaming_diagnostics()
        evaluator._validate_compatibility_rows(groups, variant["compatibility"], expected)
        assert variant["content_verdict"] == shared.content_verdict(
            variant, deterministic["comparator"]
        )

        detail_path = ROOT / variant["details"]["path"]
        if detail_path.exists():
            assert shared.file_sha256(detail_path) == variant["details"]["sha256"]
            corpus_manifest, corpus_cases = shared.load_corpus_cases(verify_pcm=False)
            assert (
                corpus_manifest["cache"]["index_sha256"] == deterministic["corpus"]["index_sha256"]
            )
            rows = evaluator._detail_rows(detail_path, corpus_cases, variant["details"]["sha256"])
            assert variant["aggregates"] == shared.aggregate_rows(rows)

        assert set(measurements["cases"]) == set(evaluator.GROUP_ORDER)
        assert measurements["corpus"]["rows"] == 5133
        assert measurements["corpus"]["total_decode_s"] >= 0
        assert measurements["corpus"]["overall_rtf"] >= 0
        assert "runs" not in measurements["corpus"]
        assert len(measurements["post_warm"]["runs"]) == evaluator.TIMING_RUNS
        assert measurements["rss_mib"]["sample_interval_audio_s"] == 1.0
        assert "does not prove" in measurements["rss_mib"]["finite_observation_note"]
        assert (
            measurements["resource_verdict"]["paths"]["sherpa_cpu"]["actual_execution_device"]
            == "CPU"
        )
        for group in evaluator.GROUP_ORDER:
            assert set(measurements["cases"][group]) == set(variant["compatibility"][group])
            for measurement in measurements["cases"][group].values():
                assert measurement["decode_seconds"] >= 0
                assert measurement["wall_seconds"] >= measurement["decode_seconds"]
                assert measurement["rss_mib"]["samples"][0]["phase"] == "start"
                assert measurement["rss_mib"]["samples"][-1]["phase"] == "eof"
                assert measurement["rss_mib"]["observed_peak"] == max(
                    sample["rss_mib"] for sample in measurement["rss_mib"]["samples"]
                )

    completed = {
        model_id: variant
        for model_id, variant in deterministic["variants"].items()
        if variant["status"] == "complete"
    }
    non_dominated = evaluator._non_dominated_variants(
        completed,
        {model_id: measurements_by_variant[model_id] for model_id in completed},
    )
    assert baseline["measurements"]["non_dominated"]["variants"] == non_dominated
    for model_id, variant in deterministic["variants"].items():
        measurements = measurements_by_variant[model_id]
        assert measurements["displacement_verdict"] == evaluator._displacement_verdict(
            variant["content_verdict"],
            measurements["resource_verdict"],
            non_dominated=model_id in non_dominated,
        )


def test_installed_560ms_adapter_reproduces_one_committed_row_when_available():
    baseline = _committed_baseline()
    spec = evaluator.streaming_models.CANDIDATE_SPECS["560ms"]
    try:
        evaluator.streaming_models.validate_installed(spec)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    groups, _ = evaluator.small_clip_cases(verify_audio=True)
    case = next(case for case in groups["short"] if case.case_id == "cv_short")
    try:
        adapter = evaluator.StreamingOnlineAdapter(spec)
        observation = adapter.decode(case)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    row = evaluator._streaming_score(case, observation.content)

    committed = baseline["deterministic"]["variants"][spec.model_id]
    if committed.get("status", "complete") != "complete":
        pytest.skip("committed 560ms variant is runtime-incompatible")
    compatibility = committed.get("compatibility", committed.get("cases"))
    assert isinstance(compatibility, dict)
    assert adapter.identity() == committed["adapter"]
    assert row == compatibility["short"][case.case_id]


def test_installed_1120ms_adapter_decodes_bounded_real_corpus_slice_when_available(
    tmp_path: Path,
):
    spec = evaluator.streaming_models.CANDIDATE_SPECS["1120ms"]
    try:
        evaluator.streaming_models.validate_installed(spec)
        corpus_manifest, corpus_cases = shared.load_corpus_cases(verify_pcm=False)
        adapter = evaluator.StreamingOnlineAdapter(spec)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert corpus_manifest["cache"]["rows"] == len(corpus_cases) == 5133
    cases = corpus_cases[:3]
    assert 0 < len(cases) <= 10
    detail = tmp_path / "bounded-real-1120ms.jsonl"
    result = evaluator._write_corpus_detail_resumable(
        detail,
        cases,
        adapter.decode,
        {
            "corpus_index_sha256": corpus_manifest["cache"]["index_sha256"],
            "engine": spec.model_id,
            "model": adapter.identity()["model"],
            "schema_version": evaluator.SCHEMA_VERSION,
        },
    )
    rows = evaluator._detail_rows(detail, cases, result["sha256"])
    aggregates = shared.aggregate_rows(rows)

    assert aggregates["completion"]["complete"] is True
    assert aggregates["completion"]["rows"] == len(cases)
    assert all(row["accepted_samples"] == row["duration_samples"] for row in rows)
    assert all(row["eof_count"] == 1 for row in rows)
    assert all(row["finalization_count"] >= 1 for row in rows)
    assert result["measurement"]["rows"] == len(cases)
    assert result["measurement"]["overall_rtf"] >= 0
    print(
        "bounded-real-1120ms "
        + json.dumps(
            {
                "case_cer": {row["case_id"]: row["cer"] for row in rows},
                "eof_once_rows": aggregates["completion"]["eof_once_rows"],
                "rows": len(rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
