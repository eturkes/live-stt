"""Model-free locks for M10.5c's direct-streaming evaluator evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests import eval_models as shared
from tests import eval_streaming as evaluator

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "streaming_baseline.json"
BASELINE_DETERMINISTIC_SHA256 = "9d067943502c0a1d51de8cf2c08d9802bf3bc14b6e1d5eb9753f31b36fded135"
DETAIL_SHA256 = {
    "nemotron_streaming_560ms": ("1a9a52a59784aa0bcd53e2da20c03ffe5207421bf5c280ef61a37da3060c6f80")
}


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
    def __init__(self):
        self.stream = FakeStream()
        self.endpoint_collected = False

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
            stream.current_text = "一"
        else:
            stream.current_text = "二"
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


def test_committed_streaming_baseline_locks_rows_events_sources_and_measurements():
    baseline = _committed_baseline()
    deterministic = baseline["deterministic"]
    groups, expected = evaluator.small_clip_cases(verify_audio=False)

    assert baseline["schema_version"] == evaluator.SCHEMA_VERSION
    assert baseline["deterministic_sha256"] == BASELINE_DETERMINISTIC_SHA256
    assert (
        hashlib.sha256(shared._json_bytes(deterministic)).hexdigest()
        == BASELINE_DETERMINISTIC_SHA256
    )
    assert deterministic["metric_contract"] == evaluator.STREAMING_METRIC_CONTRACT
    assert deterministic["displacement_gates"] == shared.DISPLACEMENT_GATES
    assert deterministic["scope"]["full_short_corpus_rows"] == 0
    assert deterministic["scope"]["case_groups"] == {
        group: len(groups[group]) for group in evaluator.GROUP_ORDER
    }
    assert set(deterministic["variants"]) == set(DETAIL_SHA256)
    assert baseline["measurements"]["excluded_from_deterministic_equality"] is True

    comparator = deterministic["comparator"]
    model_baseline = json.loads(shared.BASELINE.read_text(encoding="utf-8"))
    assert comparator["engine"] == model_baseline["deterministic"]["comparator"]["engine"]
    assert comparator["engine"] == "parakeet"
    assert comparator["source"]["sha256"] == shared.file_sha256(shared.BASELINE)

    for artifact in deterministic["inputs"]["pipeline"]["manifests"]:
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert shared.file_sha256(path) == artifact["sha256"]

    for model_id, variant in deterministic["variants"].items():
        adapter = variant["adapter"]
        assert adapter["architecture"] == "cache_aware_streaming_transducer"
        assert adapter["block_ms"] == 20
        assert adapter["forced_language"]["value"] == "ja"
        assert adapter["provider"] == "cpu"
        assert adapter["actual_execution_device"] == "CPU"
        assert adapter["endpoint_detection"]["enabled"] is True
        assert adapter["model"]["provenance"]["license"]["spdx"] == "OpenMDW-1.1"
        assert variant["details"]["sha256"] == DETAIL_SHA256[model_id]
        assert variant["details"]["rows"] == sum(len(cases) for cases in groups.values())

        ordered_rows: list[dict[str, Any]] = []
        for group in evaluator.GROUP_ORDER:
            rows = variant["cases"][group]
            assert set(rows) == {case.case_id for case in groups[group]}
            for case in groups[group]:
                row = rows[case.case_id]
                evaluator._validate_streaming_row(case, row)
                ordered_rows.append(row)
                assert row["accepted_samples"] == row["duration_samples"]
                assert row["eof_count"] == 1
                assert row["finalization_count"] >= 1
                assert row["segment_reset_count"] == row["finalization_count"] - 1
                assert row["finalization_logical_audio_s"][-1] == round(
                    row["duration_samples"] / shared.SAMPLE_RATE, 6
                )
                if row["first_text_logical_audio_s"] is not None:
                    assert (
                        0
                        <= row["first_text_logical_audio_s"]
                        <= row["finalization_logical_audio_s"][-1]
                    )

        for row in variant["cases"]["stressors"].values():
            expected_row = dict(row)
            evaluator._add_stressor_baseline(expected_row, variant["cases"]["short"], expected)
            assert expected_row == row
            assert shared.normalize(row["hyp"])
        assert shared.normalize(variant["cases"]["retention"]["retention_probe"]["hyp"])
        assert variant["aggregates"] == shared.aggregate_rows(ordered_rows)
        verdict = variant["small_clip_verdict"]
        assert verdict["displacement_qualified"] is None
        assert verdict["deferred_to_full_corpus"] == [
            "common_voice_relative_cer",
            "fleurs_micro_cer_regression",
            "complete_5133_row_run",
        ]

        detail_path = ROOT / variant["details"]["path"]
        if detail_path.exists():
            assert shared.file_sha256(detail_path) == DETAIL_SHA256[model_id]

        measurements = baseline["measurements"]["variants"][model_id]
        assert set(measurements["cases"]) == set(evaluator.GROUP_ORDER)
        assert len(measurements["post_warm"]["runs"]) == evaluator.TIMING_RUNS
        assert measurements["rss_mib"]["sample_interval_audio_s"] == 1.0
        assert "does not prove" in measurements["rss_mib"]["finite_observation_note"]
        assert (
            measurements["resource_verdict"]["paths"]["sherpa_cpu"]["actual_execution_device"]
            == "CPU"
        )
        for group in evaluator.GROUP_ORDER:
            assert set(measurements["cases"][group]) == set(variant["cases"][group])
            for measurement in measurements["cases"][group].values():
                assert measurement["decode_seconds"] >= 0
                assert measurement["wall_seconds"] >= measurement["decode_seconds"]
                assert measurement["rss_mib"]["samples"][0]["phase"] == "start"
                assert measurement["rss_mib"]["samples"][-1]["phase"] == "eof"
                assert measurement["rss_mib"]["observed_peak"] == max(
                    sample["rss_mib"] for sample in measurement["rss_mib"]["samples"]
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
    adapter = evaluator.StreamingOnlineAdapter(spec)
    observation = adapter.decode(case)
    row = evaluator._streaming_score(case, observation.content)

    committed = baseline["deterministic"]["variants"][spec.model_id]
    assert adapter.identity() == committed["adapter"]
    assert row == committed["cases"]["short"][case.case_id]
