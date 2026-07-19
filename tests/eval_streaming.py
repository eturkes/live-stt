#!/usr/bin/env python3
"""Direct-online M10.5 streaming ASR evaluator.

The evaluator feeds canonical mono 16 kHz PCM directly to sherpa-onnx's
``OnlineRecognizer`` in production-like 20 ms blocks. It deliberately bypasses
live-stt's offline VAD, ring buffer, and decode chunker. Endpoint resets and the
single EOF finalization are explicit deterministic events; elapsed time and
sampled RSS remain in a separate measurements block.

M10.5d adds the complete 5,133-row corpus to those compatibility clips and makes
the corpus JSONL resumable at row boundaries. The default remains the pinned 560 ms
variant for bounded checks; pass ``all`` only for the separately scheduled tournament.

Run from the repository root:

    UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python tests/eval_streaming.py
    UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python tests/eval_streaming.py all
    UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python tests/eval_streaming.py --aggregate-only
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import time
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

import eval_models as shared  # noqa: E402
import fetch_eval_streaming as streaming_models  # noqa: E402

SCHEMA_VERSION = 2
BASELINE = TESTS / "streaming_baseline.json"
RETENTION_PROBE = TESTS / "retention_probe.json"
CACHE = ROOT / "spike" / "backends" / "cache"
DETAILS_DIR = CACHE / "streaming_eval-v2"
STAGING_DIR = CACHE / "streaming_eval-v2.staging"
DEFAULT_VARIANTS = ("560ms",)
STREAMING_SPECS = {spec.model_id: spec for spec in streaming_models.CANDIDATE_SPECS.values()}
TOURNAMENT_VARIANTS = ("1120ms", "560ms", "160ms", "80ms")
STREAMING_MODEL_IDS = tuple(
    streaming_models.CANDIDATE_SPECS[name].model_id for name in TOURNAMENT_VARIANTS
)
GROUP_ORDER = ("short", "stressors", "long_form", "retention")
BLOCK_SAMPLES = shared.SAMPLE_RATE // 50
BLOCK_MS = 20
RSS_SAMPLE_BLOCKS = 50
TIMING_RUNS = shared.TIMING_RUNS
FORCED_LANGUAGE = "ja"
PROVIDER = "cpu"

STREAMING_METRIC_CONTRACT = {
    **shared.METRIC_CONTRACT,
    "corpus_detail": (
        "one deterministic streaming score row per pinned corpus index row; no timing or RSS"
    ),
    "corpus_measurement": (
        "aggregate total decode/wall seconds and overall RTF only; per-row timing/RSS "
        "stays staging-only"
    ),
    "diagnostics": (
        "direct OnlineRecognizer has no generation/context truncation path; no native "
        "fd-2 capture, "
        "so the decoder-diagnostic content gate is explicitly vacuously clean"
    ),
    "finalization": (
        "each recognizer endpoint is collected once then reset; EOF is sent once and "
        "its trailing result is collected once without reset"
    ),
    "first_text_logical_audio_s": (
        "audio seconds accepted when the first nonempty recognizer result appears"
    ),
    "finalization_logical_audio_s": (
        "ordered audio-second positions at which endpoint/EOF results are finalized"
    ),
    "hypothesis": "raw finalized segment texts concatenated in event order",
    "partial_update_count": "nonempty recognizer-result text changes",
    "resume": (
        "append validated deterministic row bytes, reconcile with an fsynced measurement journal, "
        "then atomically rename the complete JSONL"
    ),
    "rss": (
        "per-case samples for the 16 compatibility clips and post-warm runs; corpus "
        "measurements retain timing aggregates only"
    ),
    "segment_reset_count": "endpoint resets only; equals finalization_count - 1",
}


@dataclass(frozen=True)
class StreamingTranscript(shared.Transcript):
    partial_update_count: int
    finalization_count: int
    segment_reset_count: int
    first_text_logical_audio_s: float | None
    finalization_logical_audio_s: tuple[float, ...]


@dataclass(frozen=True)
class StreamingDecodeObservation(shared.DecodeObservation):
    content: StreamingTranscript
    rss_samples_mib: tuple[dict, ...]
    rss_peak_mib: float


def _audio_s(samples: int) -> float:
    return round(samples / shared.SAMPLE_RATE, 6)


def _endpoint_rule(rule: Any) -> dict:
    return {
        "min_trailing_silence": rule.min_trailing_silence,
        "min_utterance_length": rule.min_utterance_length,
        "must_contain_nonsilence": rule.must_contain_nonsilence,
    }


def model_fingerprint(spec: streaming_models.CandidateSpec) -> dict:
    model_dir = streaming_models.validate_installed(spec)
    artifacts = [shared._artifact(model_dir / artifact.path) for artifact in spec.artifacts]
    runtime_paths = {
        "decoder.int8.onnx",
        "encoder.int8.onnx",
        "joiner.int8.onnx",
        "tokens.txt",
    }
    return {
        "artifacts": artifacts,
        "bytes": sum(row["bytes"] for row in artifacts),
        "directory": shared._relative(model_dir),
        "provenance": streaming_models.provenance(spec),
        "runtime_model_bytes": sum(
            row["bytes"]
            for artifact, row in zip(spec.artifacts, artifacts, strict=True)
            if artifact.path in runtime_paths
        ),
    }


class StreamingOnlineAdapter:
    """Forced-Japanese direct OnlineRecognizer adapter with explicit events."""

    def __init__(
        self,
        spec: streaming_models.CandidateSpec,
        *,
        model: Mapping[str, Any] | None = None,
    ):
        self.adapter_id = spec.model_id
        self.spec = spec
        self.model = dict(model) if model is not None else model_fingerprint(spec)
        target = ROOT / self.model["directory"]
        try:
            self.recognizer: Any = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(target / "tokens.txt"),
                encoder=str(target / "encoder.int8.onnx"),
                decoder=str(target / "decoder.int8.onnx"),
                joiner=str(target / "joiner.int8.onnx"),
                provider=PROVIDER,
                decoding_method="greedy_search",
                modeling_unit="cjkchar",
                num_threads=shared.NUM_THREADS,
                enable_endpoint_detection=True,
            )
            language_probe = self.recognizer.create_stream()
            language_probe.set_option("language", FORCED_LANGUAGE)
            if language_probe.get_option("language") != FORCED_LANGUAGE:
                raise RuntimeError("stream did not retain forced-ja option")
        except Exception as exc:
            version = getattr(sherpa_onnx, "__version__", "unknown")
            raise RuntimeError(
                f"{spec.model_id}: sherpa_onnx {version} streaming initialization failed: {exc}"
            ) from exc

    def identity(self) -> dict:
        cfg = self.recognizer.config
        model = cfg.model_config
        endpoint = cfg.endpoint_config
        feature = cfg.feat_config
        provider = model.provider_config.provider
        return {
            "actual_execution_device": provider.upper(),
            "adapter": "sherpa_online_direct_pcm20",
            "adapter_id": self.adapter_id,
            "architecture": "cache_aware_streaming_transducer",
            "block_ms": BLOCK_MS,
            "chunk_ms": self.spec.chunk_ms,
            "decoding_method": cfg.decoding_method,
            "endpoint_detection": {
                "enabled": cfg.enable_endpoint,
                "rule1": _endpoint_rule(endpoint.rule1),
                "rule2": _endpoint_rule(endpoint.rule2),
                "rule3": _endpoint_rule(endpoint.rule3),
            },
            "execution_device_evidence": {
                "field": "OnlineRecognizer.config.model_config.provider_config.provider",
                "value": provider,
            },
            "feature_config": {
                "dither": feature.dither,
                "feature_dim": feature.feature_dim,
                "normalize_samples": feature.normalize_samples,
                "sample_rate_hz": feature.sampling_rate,
                "snip_edges": feature.snip_edges,
            },
            "forced_language": {
                "method": "OnlineStream.set_option",
                "value": FORCED_LANGUAGE,
            },
            "model": self.model,
            "modeling_unit": model.modeling_unit,
            "num_threads": model.num_threads,
            "provider": provider,
        }

    def decode(self, case: shared.EvalCase) -> StreamingDecodeObservation:
        samples = _read_pcm(case)
        return _decode_samples(
            self.recognizer,
            samples,
            adapter_id=self.adapter_id,
        )


def _read_pcm(case: shared.EvalCase) -> np.ndarray:
    try:
        with wave.open(str(case.wav), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != shared.SAMPLE_RATE
                or source.getcomptype() != "NONE"
            ):
                raise RuntimeError(
                    f"{case.case_id}: streaming evaluator requires canonical mono PCM16"
                )
            frames = source.getnframes()
            raw = source.readframes(frames)
    except (EOFError, OSError, wave.Error) as exc:
        raise RuntimeError(f"{case.case_id}: cannot read evaluator WAV: {exc}") from exc
    if frames != case.duration_samples or len(raw) != frames * 2:
        raise RuntimeError(
            f"{case.case_id}: WAV geometry drift: {frames} != {case.duration_samples}"
        )
    return np.ascontiguousarray(
        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0,
        dtype=np.float32,
    )


def _decode_samples(
    recognizer: Any,
    samples: np.ndarray,
    *,
    adapter_id: str,
    rss_sampler: Callable[[], float] = shared._current_rss_mib,
    rss_sample_blocks: int = RSS_SAMPLE_BLOCKS,
) -> StreamingDecodeObservation:
    """Drive one online stream; all logical timestamps derive from accepted PCM."""
    if samples.ndim != 1 or samples.dtype != np.float32 or not samples.flags.c_contiguous:
        raise RuntimeError(f"{adapter_id}: samples must be contiguous mono float32")
    if samples.size <= 0:
        raise RuntimeError(f"{adapter_id}: cannot decode empty audio")
    if rss_sample_blocks <= 0:
        raise ValueError("rss_sample_blocks must be positive")

    wall_started = time.perf_counter()
    stream = recognizer.create_stream()
    stream.set_option("language", FORCED_LANGUAGE)
    if stream.get_option("language") != FORCED_LANGUAGE:
        raise RuntimeError(f"{adapter_id}: stream did not retain forced-ja option")

    accepted_samples = 0
    eof_count = 0
    decode_seconds = 0.0
    partial_update_count = 0
    segment_reset_count = 0
    first_text_logical_audio_s: float | None = None
    segment_first_text_s: float | None = None
    segment_partial_updates = 0
    last_result = ""
    segments: list[dict] = []
    finalization_times: list[float] = []
    rss_samples: list[dict] = [
        {"logical_audio_s": 0.0, "phase": "start", "rss_mib": round(rss_sampler(), 3)}
    ]

    def observe_text(logical_audio_s: float) -> str:
        nonlocal first_text_logical_audio_s
        nonlocal last_result
        nonlocal partial_update_count
        nonlocal segment_first_text_s
        nonlocal segment_partial_updates
        text = recognizer.get_result(stream)
        if text != last_result:
            if text:
                partial_update_count += 1
                segment_partial_updates += 1
                if first_text_logical_audio_s is None:
                    first_text_logical_audio_s = logical_audio_s
                if segment_first_text_s is None:
                    segment_first_text_s = logical_audio_s
            last_result = text
        return text

    def finalize(reason: str, logical_audio_s: float) -> None:
        nonlocal last_result
        nonlocal segment_first_text_s
        nonlocal segment_partial_updates
        text = observe_text(logical_audio_s)
        segments.append(
            {
                "finalization_logical_audio_s": logical_audio_s,
                "first_text_logical_audio_s": segment_first_text_s,
                "index": len(segments),
                "partial_update_count": segment_partial_updates,
                "reason": reason,
                "text": text,
            }
        )
        finalization_times.append(logical_audio_s)
        last_result = ""
        segment_first_text_s = None
        segment_partial_updates = 0

    total_blocks = (samples.size + BLOCK_SAMPLES - 1) // BLOCK_SAMPLES
    for block_index, start in enumerate(range(0, samples.size, BLOCK_SAMPLES), 1):
        end = min(start + BLOCK_SAMPLES, samples.size)
        stream.accept_waveform(shared.SAMPLE_RATE, samples[start:end])
        accepted_samples += end - start
        logical_audio_s = _audio_s(accepted_samples)
        while recognizer.is_ready(stream):
            decode_started = time.perf_counter()
            recognizer.decode_stream(stream)
            decode_seconds += time.perf_counter() - decode_started
            observe_text(logical_audio_s)
        if recognizer.is_endpoint(stream):
            finalize("endpoint", logical_audio_s)
            recognizer.reset(stream)
            segment_reset_count += 1
        if block_index % rss_sample_blocks == 0 or block_index == total_blocks:
            rss_samples.append(
                {
                    "logical_audio_s": logical_audio_s,
                    "phase": "feed",
                    "rss_mib": round(rss_sampler(), 3),
                }
            )

    stream.input_finished()
    eof_count += 1
    eof_audio_s = _audio_s(accepted_samples)
    while recognizer.is_ready(stream):
        decode_started = time.perf_counter()
        recognizer.decode_stream(stream)
        decode_seconds += time.perf_counter() - decode_started
        observe_text(eof_audio_s)
    finalize("eof", eof_audio_s)
    rss_samples.append(
        {
            "logical_audio_s": eof_audio_s,
            "phase": "eof",
            "rss_mib": round(rss_sampler(), 3),
        }
    )

    if accepted_samples != samples.size:
        raise RuntimeError(f"{adapter_id}: accepted-sample accounting drift")
    if eof_count != 1:
        raise RuntimeError(f"{adapter_id}: EOF accounting drift")
    if segment_reset_count != len(segments) - 1:
        raise RuntimeError(f"{adapter_id}: endpoint/finalization accounting drift")
    transcript = StreamingTranscript(
        hypothesis="".join(segment["text"] for segment in segments),
        segments=tuple(segments),
        accepted_samples=accepted_samples,
        eof_count=eof_count,
        complete=True,
        partial_update_count=partial_update_count,
        finalization_count=len(segments),
        segment_reset_count=segment_reset_count,
        first_text_logical_audio_s=first_text_logical_audio_s,
        finalization_logical_audio_s=tuple(finalization_times),
    )
    return StreamingDecodeObservation(
        content=transcript,
        decode_seconds=decode_seconds,
        wall_seconds=time.perf_counter() - wall_started,
        rss_samples_mib=tuple(rss_samples),
        rss_peak_mib=max(sample["rss_mib"] for sample in rss_samples),
    )


def _streaming_score(case: shared.EvalCase, transcript: StreamingTranscript) -> dict:
    row = shared._score(case, transcript)
    row.update(
        {
            "finalization_count": transcript.finalization_count,
            "finalization_logical_audio_s": list(transcript.finalization_logical_audio_s),
            "first_text_logical_audio_s": transcript.first_text_logical_audio_s,
            "n_segments": len(transcript.segments),
            "partial_update_count": transcript.partial_update_count,
            "segment_reset_count": transcript.segment_reset_count,
        }
    )
    return row


def small_clip_cases(*, verify_audio: bool) -> tuple[dict[str, list[shared.EvalCase]], dict]:
    probes, expected = shared.compatibility_cases()
    retention = shared._load_json(RETENTION_PROBE)
    probe = retention["probe"]
    duration_samples = round(probe["audio_s"] * shared.SAMPLE_RATE)
    groups = {
        "short": probes["short"],
        "stressors": probes["stressors"],
        "long_form": probes["long_form"],
        "retention": [
            shared.EvalCase(
                case_id="retention_probe",
                source="retention_probe",
                wav=CACHE / "retention_probe.wav",
                reference=probe["ja_ref"],
                duration_samples=duration_samples,
                duration_bucket=shared._duration_bucket(duration_samples),
            )
        ],
    }
    if tuple(groups) != GROUP_ORDER:
        raise RuntimeError("streaming small-clip group order drifted")
    case_ids = [case.case_id for cases in groups.values() for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("streaming small-clip case IDs are not unique")
    if verify_audio:
        for case in (case for cases in groups.values() for case in cases):
            if shared._wav_samples(case.wav) != case.duration_samples:
                raise RuntimeError(f"{case.case_id}: duration-sample drift")
        retention_wav = groups["retention"][0].wav
        if shared.file_sha256(retention_wav) != probe["audio_sha256"]:
            raise RuntimeError("retention-probe WAV fingerprint mismatch")
    expected["retention_manifest"] = retention
    return groups, expected


def pipeline_fingerprint(
    groups: Mapping[str, Sequence[shared.EvalCase]],
    corpus_manifest: Mapping[str, Any],
) -> dict:
    contract = "\n\n".join(
        inspect.getsource(obj)
        for obj in (
            StreamingTranscript,
            StreamingDecodeObservation,
            StreamingOnlineAdapter,
            _read_pcm,
            _decode_samples,
            _streaming_score,
            small_clip_cases,
            shared.load_corpus_cases,
            _validate_streaming_row,
            _validate_compatibility_rows,
            _validated_jsonl_prefix,
            _write_corpus_detail_resumable,
            _comparator_snapshot,
            _streaming_diagnostics,
            _runtime_failure_summary,
            _runtime_failure_result,
            _non_dominated_variants,
            _displacement_verdict,
            _case_measurement,
            _benchmark,
            run_worker,
            build_manifest,
        )
    ).encode()
    return {
        "case_inputs": [
            {
                "case_id": case.case_id,
                "group": group,
                **shared._artifact(case.wav),
            }
            for group in GROUP_ORDER
            for case in groups[group]
        ],
        "contract_sha256": hashlib.sha256(contract).hexdigest(),
        "corpus_index_sha256": corpus_manifest["cache"]["index_sha256"],
        "implementation": [
            shared._artifact(ROOT / "cer.py"),
            shared._artifact(TESTS / "eval_models.py"),
            shared._artifact(TESTS / "fetch_eval_streaming.py"),
        ],
        "manifests": [
            shared._artifact(path)
            for path in (
                shared.SHORT_CORPUS,
                shared.REPLAY_GOLDENS,
                shared.STRESSORS,
                shared.LONG_FORM,
                RETENTION_PROBE,
                shared.BASELINE,
            )
        ],
        "values": {
            "block_ms": BLOCK_MS,
            "block_samples": BLOCK_SAMPLES,
            "endpoint_detection": True,
            "forced_language": FORCED_LANGUAGE,
            "num_threads": shared.NUM_THREADS,
            "provider": PROVIDER,
            "rss_sample_blocks": RSS_SAMPLE_BLOCKS,
            "sample_rate_hz": shared.SAMPLE_RATE,
            "timing_runs": TIMING_RUNS,
            "tournament_model_order": list(STREAMING_MODEL_IDS),
        },
    }


def evaluation_inputs(
    groups: Mapping[str, Sequence[shared.EvalCase]],
    corpus_manifest: Mapping[str, Any],
) -> dict:
    return {
        "corpus_index_sha256": corpus_manifest["cache"]["index_sha256"],
        "pipeline": pipeline_fingerprint(groups, corpus_manifest),
        "runtime": shared.runtime_fingerprint(),
    }


def _add_stressor_baseline(
    row: dict,
    short_rows: Mapping[str, dict],
    expected: Mapping[str, Any],
) -> None:
    order = expected["stressor_manifest"]["stressors"][row["case_id"]]["order"]
    baseline_d = sum(short_rows[case_id]["D"] for case_id in order)
    row["baseline_D"] = baseline_d
    row["excess_D"] = row["D"] - baseline_d
    row["excess_del_rate"] = round(row["excess_D"] / row["N"], 4)


def _validate_streaming_row(case: shared.EvalCase, row: Mapping[str, Any]) -> None:
    ref = shared.normalize(case.reference)
    hyp = shared.normalize(row.get("hyp", ""))
    substitutions, deletions, insertions = shared.align(ref, hyp)
    if (
        row.get("case_id") != case.case_id
        or row.get("source") != case.source
        or row.get("ref") != case.reference
        or row.get("duration_samples") != case.duration_samples
        or row.get("duration_bucket") != case.duration_bucket
        or row.get("gender") != case.gender
        or row.get("accepted_samples") != case.duration_samples
        or row.get("eof_count") != 1
        or row.get("complete") is not True
        or tuple(row.get(key) for key in ("N", "S", "D", "I"))
        != (len(ref), substitutions, deletions, insertions)
        or row.get("cer") != (substitutions + deletions + insertions) / len(ref)
    ):
        raise RuntimeError(f"{case.case_id}: streaming score/accounting drift")

    segments = row.get("segments")
    finalizations = row.get("finalization_logical_audio_s")
    finalization_count = row.get("finalization_count")
    reset_count = row.get("segment_reset_count")
    if (
        not isinstance(segments, list)
        or not isinstance(finalizations, list)
        or not isinstance(finalization_count, int)
        or finalization_count < 1
        or row.get("n_segments") != finalization_count
        or len(segments) != finalization_count
        or len(finalizations) != finalization_count
        or reset_count != finalization_count - 1
    ):
        raise RuntimeError(f"{case.case_id}: finalization/reset shape drift")

    duration_s = _audio_s(case.duration_samples)
    previous = -1.0
    segment_updates = 0
    segment_first_times: list[float] = []
    for index, segment in enumerate(segments):
        expected_reason = "eof" if index == len(segments) - 1 else "endpoint"
        finalization = segment.get("finalization_logical_audio_s")
        first_text = segment.get("first_text_logical_audio_s")
        updates = segment.get("partial_update_count")
        if (
            segment.get("index") != index
            or segment.get("reason") != expected_reason
            or not isinstance(segment.get("text"), str)
            or not isinstance(updates, int)
            or updates < 0
            or not isinstance(finalization, (int, float))
            or finalization != finalizations[index]
            or finalization < previous
            or finalization < 0
            or finalization > duration_s
            or (first_text is not None and not 0 <= first_text <= finalization)
        ):
            raise RuntimeError(f"{case.case_id}: segment event drift")
        if segment["text"] and updates < 1:
            raise RuntimeError(f"{case.case_id}: finalized text lacks a partial update")
        if first_text is not None:
            segment_first_times.append(first_text)
        segment_updates += updates
        previous = finalization

    expected_first = min(segment_first_times) if segment_first_times else None
    if (
        finalizations[-1] != duration_s
        or row.get("hyp") != "".join(segment["text"] for segment in segments)
        or row.get("partial_update_count") != segment_updates
        or row.get("first_text_logical_audio_s") != expected_first
    ):
        raise RuntimeError(f"{case.case_id}: streaming transition accounting drift")


def _validate_compatibility_rows(
    groups: Mapping[str, Sequence[shared.EvalCase]],
    grouped_rows: Mapping[str, Mapping[str, dict]],
    expected: Mapping[str, Any],
) -> None:
    if set(grouped_rows) != set(GROUP_ORDER):
        raise RuntimeError("streaming compatibility groups drifted")
    for group in GROUP_ORDER:
        expected_ids = [case.case_id for case in groups[group]]
        if set(grouped_rows[group]) != set(expected_ids):
            raise RuntimeError(f"streaming compatibility IDs drifted: {group}")
        for case in groups[group]:
            _validate_streaming_row(case, grouped_rows[group][case.case_id])
    for row in grouped_rows["stressors"].values():
        expected_row = dict(row)
        _add_stressor_baseline(expected_row, grouped_rows["short"], expected)
        if expected_row != row:
            raise RuntimeError("streaming stressor baseline drifted")


def _resume_paths(detail_path: Path) -> tuple[Path, Path, Path]:
    return (
        detail_path.with_name(f"{detail_path.name}.part"),
        detail_path.with_name(f"{detail_path.name}.measurements.part"),
        detail_path.with_name(f"{detail_path.name}.resume.json"),
    )


def _clear_corpus_resume_artifacts(detail_path: Path, *, include_detail: bool = False) -> None:
    part_path, measurement_path, state_path = _resume_paths(detail_path)
    for path in (part_path, measurement_path, state_path):
        path.unlink(missing_ok=True)
    if include_detail:
        detail_path.unlink(missing_ok=True)


def _validated_jsonl_prefix(
    path: Path,
    validate: Callable[[int, Mapping[str, Any]], None],
) -> tuple[list[dict], list[bytes]]:
    rows: list[dict] = []
    chunks: list[bytes] = []
    if not path.is_file():
        return rows, chunks
    with path.open("rb") as source:
        while line := source.readline():
            if not line.endswith(b"\n"):
                break
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    break
                validate(len(rows), row)
            except (KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
                break
            rows.append(row)
            chunks.append(line)
    return rows, chunks


def _truncate_jsonl(path: Path, chunks: Sequence[bytes]) -> None:
    if not path.exists():
        return
    with path.open("r+b") as output:
        output.truncate(sum(len(chunk) for chunk in chunks))
        output.flush()
        os.fsync(output.fileno())


def _write_corpus_detail_resumable(
    detail_path: Path,
    cases: Sequence[shared.EvalCase],
    decode: Callable[[shared.EvalCase], StreamingDecodeObservation],
    resume_identity: Mapping[str, Any],
    *,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict:
    """Append a validated corpus prefix, then atomically expose only the full JSONL."""
    if not cases:
        raise RuntimeError("streaming corpus cannot be empty")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    part_path, measurement_path, state_path = _resume_paths(detail_path)
    expected_state = dict(resume_identity)

    def reset() -> None:
        _clear_corpus_resume_artifacts(detail_path, include_detail=True)

    if state_path.is_file():
        try:
            state = shared._load_json(state_path)
        except (OSError, UnicodeError, json.JSONDecodeError):
            reset()
        else:
            if state != expected_state:
                reset()
    elif any(path.exists() for path in (detail_path, part_path, measurement_path)):
        reset()
    if not state_path.is_file():
        shared.write_atomic(state_path, [shared._json_bytes(expected_state)])

    detail_source = detail_path if detail_path.is_file() else part_path

    def validate_detail(index: int, row: Mapping[str, Any]) -> None:
        if index >= len(cases):
            raise RuntimeError("streaming detail has extra rows")
        _validate_streaming_row(cases[index], row)

    def validate_measurement(index: int, row: Mapping[str, Any]) -> None:
        if index >= len(cases) or row.get("case_id") != cases[index].case_id:
            raise RuntimeError("streaming measurement journal identity drifted")
        values = [row.get(key) for key in ("decode_seconds", "wall_seconds")]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise RuntimeError("streaming measurement journal value drifted")
        if row["wall_seconds"] < row["decode_seconds"]:
            raise RuntimeError("streaming measurement journal timing drifted")

    rows, detail_chunks = _validated_jsonl_prefix(detail_source, validate_detail)
    measurements, measurement_chunks = _validated_jsonl_prefix(
        measurement_path, validate_measurement
    )
    detail_fully_valid = (
        not detail_source.exists()
        or sum(len(chunk) for chunk in detail_chunks) == detail_source.stat().st_size
    )
    measurement_fully_valid = (
        not measurement_path.exists()
        or sum(len(chunk) for chunk in measurement_chunks) == measurement_path.stat().st_size
    )
    if detail_path.is_file():
        if (
            len(rows) != len(cases)
            or len(measurements) != len(cases)
            or not detail_fully_valid
            or not measurement_fully_valid
        ):
            reset()
            shared.write_atomic(state_path, [shared._json_bytes(expected_state)])
            detail_source = part_path
            rows, detail_chunks, measurements, measurement_chunks = [], [], [], []
    else:
        prefix_rows = min(len(rows), len(measurements))
        rows = rows[:prefix_rows]
        detail_chunks = detail_chunks[:prefix_rows]
        measurements = measurements[:prefix_rows]
        measurement_chunks = measurement_chunks[:prefix_rows]
        _truncate_jsonl(part_path, detail_chunks)
        _truncate_jsonl(measurement_path, measurement_chunks)

    resumed_rows = len(rows)
    if len(rows) < len(cases):
        with (
            part_path.open("ab") as detail_output,
            measurement_path.open("ab") as measurement_output,
        ):
            for index, case in enumerate(cases[len(rows) :], len(rows) + 1):
                observation = decode(case)
                row = _streaming_score(case, observation.content)
                _validate_streaming_row(case, row)
                measurement = {
                    "case_id": case.case_id,
                    "decode_seconds": observation.decode_seconds,
                    "wall_seconds": observation.wall_seconds,
                }
                detail_data = shared._json_bytes(row, compact=True)
                measurement_data = shared._json_bytes(measurement, compact=True)

                # Detail reaches stable storage first. If interruption lands between the
                # two fsyncs, prefix reconciliation drops and re-decodes this one row.
                detail_output.write(detail_data)
                detail_output.flush()
                os.fsync(detail_output.fileno())
                measurement_output.write(measurement_data)
                measurement_output.flush()
                os.fsync(measurement_output.fileno())

                rows.append(row)
                detail_chunks.append(detail_data)
                measurements.append(measurement)
                measurement_chunks.append(measurement_data)
                if progress is not None:
                    progress(index, len(cases), row)

    if len(rows) != len(cases) or len(measurements) != len(cases):
        raise RuntimeError("streaming corpus detail did not reach every row")
    if not detail_path.is_file():
        part_path.replace(detail_path)
    detail_sha256 = shared.file_sha256(detail_path)
    _detail_rows(detail_path, cases, detail_sha256)

    audio_s = sum(case.duration_samples for case in cases) / shared.SAMPLE_RATE
    decode_s = sum(row["decode_seconds"] for row in measurements)
    wall_s = sum(row["wall_seconds"] for row in measurements)
    return {
        "measurement": {
            "audio_s": round(audio_s, 6),
            "overall_rtf": round(decode_s / audio_s, 6),
            "overall_wall_rtf": round(wall_s / audio_s, 6),
            "rows": len(cases),
            "rows_reused_on_resume": resumed_rows,
            "total_decode_s": round(decode_s, 6),
            "total_wall_s": round(wall_s, 6),
        },
        "rows": len(cases),
        "sha256": detail_sha256,
    }


def _detail_rows(
    path: Path,
    cases: Sequence[shared.EvalCase],
    expected_sha256: str,
) -> list[dict]:
    if not path.is_file() or shared.file_sha256(path) != expected_sha256:
        raise RuntimeError(f"streaming detail fingerprint mismatch: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read streaming detail: {path}") from exc
    if [row.get("case_id") for row in rows] != [case.case_id for case in cases]:
        raise RuntimeError("streaming detail is incomplete, duplicated, or reordered")
    for case, row in zip(cases, rows, strict=True):
        _validate_streaming_row(case, row)
    return rows


def _comparator_snapshot() -> dict:
    baseline = shared._load_json(shared.BASELINE)
    deterministic = baseline["deterministic"]
    comparator = deterministic["comparator"]
    if comparator["engine"] != "parakeet":
        raise RuntimeError("streaming evaluator requires M10.3's fixed parakeet comparator")
    control = deterministic["controls"]["parakeet"]
    common_voice = control["aggregates"]["by_source"]["common_voice_8"]["micro"]
    fleurs = control["aggregates"]["by_source"]["fleurs"]["micro"]
    long_form = control["compatibility"]["long_form"][shared.LONG_FORM_ID]
    observed = {
        "common_voice_micro_cer": common_voice["cer"],
        "fleurs_micro_cer": fleurs["cer"],
        "long_form_cer": round(long_form["cer"], 8),
    }
    expected = {
        "common_voice_micro_cer": 0.08426233,
        "fleurs_micro_cer": 0.10444744,
        "long_form_cer": 0.23571945,
    }
    if observed != expected or comparator["micro_cer"] != common_voice["cer"]:
        raise RuntimeError("fixed parakeet comparator values drifted")

    score_keys = ("D", "I", "N", "S", "cer", "hyp", "ref")
    stressors = control["compatibility"]["stressors"]
    return {
        **observed,
        "aggregates": {
            "by_source": {
                "common_voice_8": {"micro": dict(common_voice)},
                "fleurs": {"micro": dict(fleurs)},
            }
        },
        "compatibility": {
            "long_form": {shared.LONG_FORM_ID: {key: long_form[key] for key in score_keys}},
            "stressors": {
                case_id: {key: row[key] for key in score_keys if key in row}
                for case_id, row in sorted(stressors.items())
            },
        },
        "engine": comparator["engine"],
        "source": {
            "path": shared._relative(shared.BASELINE),
            "sha256": shared.file_sha256(shared.BASELINE),
        },
    }


def _streaming_diagnostics() -> dict:
    return {
        "capture": "none",
        "content_event_case_ids": [],
        "content_events": [],
        "note": (
            "No native fd-2 capture: the direct OnlineRecognizer path has no "
            "generation/context truncation mechanism, so this content gate is vacuously clean."
        ),
    }


def _runtime_failure_summary(
    model_id: str,
    model: Mapping[str, Any],
    inputs: Mapping[str, Any],
    exc: RuntimeError,
) -> dict:
    return {
        "engine": model_id,
        "failure": {
            "message": str(exc),
            "phase": "adapter_initialization",
            "type": type(exc).__name__,
        },
        "inputs": dict(inputs),
        "model": dict(model),
        "schema_version": SCHEMA_VERSION,
        "status": "runtime_incompatible",
    }


def _runtime_failure_result(summary: Mapping[str, Any]) -> dict:
    failure = dict(summary["failure"])
    return {
        "content_verdict": {
            "checks": {
                "runtime_compatible": {
                    "observed": failure,
                    "pass": False,
                }
            },
            "qualified": False,
        },
        "failure": failure,
        "model": dict(summary["model"]),
        "status": "runtime_incompatible",
    }


def _non_dominated_variants(
    variants: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if set(variants) != set(measurements):
        raise RuntimeError("non-dominated inputs do not cover the same variants")
    objectives: dict[str, tuple[Fraction, Fraction]] = {}
    for model_id, variant in variants.items():
        micro = variant["aggregates"]["by_source"]["common_voice_8"]["micro"]
        if micro["N"] <= 0:
            raise RuntimeError(f"{model_id}: Common Voice objective has no reference units")
        rtf = measurements[model_id]["post_warm"]["median_decode_rtf"]
        if isinstance(rtf, bool) or not isinstance(rtf, (int, float)) or rtf < 0:
            raise RuntimeError(f"{model_id}: invalid post-warm RTF objective")
        objectives[model_id] = (
            Fraction(micro["S"] + micro["D"] + micro["I"], micro["N"]),
            Fraction(str(rtf)),
        )

    non_dominated = []
    for model_id in sorted(objectives):
        cer, rtf = objectives[model_id]
        dominated = any(
            other_id != model_id
            and other_cer <= cer
            and other_rtf <= rtf
            and (other_cer < cer or other_rtf < rtf)
            for other_id, (other_cer, other_rtf) in objectives.items()
        )
        if not dominated:
            non_dominated.append(model_id)
    return non_dominated


def _displacement_verdict(
    content: Mapping[str, Any],
    resource: Mapping[str, Any],
    *,
    non_dominated: bool,
) -> dict:
    content_qualified = content.get("qualified") is True
    resource_qualified = resource.get("qualified") is True
    return {
        "content_qualified": content_qualified,
        "displacement_qualified": content_qualified and resource_qualified,
        "non_dominated": non_dominated,
        "resource_qualified": resource_qualified,
    }


def _case_measurement(
    case: shared.EvalCase,
    observation: StreamingDecodeObservation,
) -> dict:
    audio_seconds = case.duration_samples / shared.SAMPLE_RATE
    return {
        "audio_s": round(audio_seconds, 6),
        "decode_rtf": round(observation.decode_seconds / audio_seconds, 6),
        "decode_seconds": round(observation.decode_seconds, 6),
        "rss_mib": {
            "observed_peak": observation.rss_peak_mib,
            "samples": list(observation.rss_samples_mib),
        },
        "wall_rtf": round(observation.wall_seconds / audio_seconds, 6),
        "wall_seconds": round(observation.wall_seconds, 6),
    }


def _benchmark(
    adapter: StreamingOnlineAdapter,
    case: shared.EvalCase,
    expected_row: Mapping[str, Any],
    short_rows: Mapping[str, dict],
    expected: Mapping[str, Any],
) -> dict:
    runs = []
    for index in range(1, TIMING_RUNS + 1):
        observation = adapter.decode(case)
        row = _streaming_score(case, observation.content)
        _add_stressor_baseline(row, short_rows, expected)
        if row != expected_row:
            raise RuntimeError(f"{adapter.adapter_id}: post-warm content drift on run {index}")
        runs.append({"run": index, **_case_measurement(case, observation)})
    return {
        "median_decode_rtf": round(median(run["decode_rtf"] for run in runs), 6),
        "median_wall_rtf": round(median(run["wall_rtf"] for run in runs), 6),
        "runs": runs,
    }


def run_worker(model_id: str, detail_path: Path, summary_path: Path) -> None:
    spec = STREAMING_SPECS[model_id]
    corpus_manifest, corpus_cases = shared.load_corpus_cases(verify_pcm=False)
    groups, expected = small_clip_cases(verify_audio=True)
    inputs = evaluation_inputs(groups, corpus_manifest)

    model = model_fingerprint(spec)
    rss_before_load = shared._current_rss_mib()
    peak_before_load = shared._peak_rss_mib()
    load_started = time.perf_counter()
    try:
        adapter = StreamingOnlineAdapter(spec, model=model)
    except RuntimeError as exc:
        _clear_corpus_resume_artifacts(detail_path, include_detail=True)
        summary = _runtime_failure_summary(model_id, model, inputs, exc)
        shared.write_atomic(summary_path, [shared._json_bytes(summary)])
        print(f"{model_id}: recorded runtime incompatibility: {exc}", flush=True)
        return
    cold_load_s = time.perf_counter() - load_started
    identity = adapter.identity()
    rss_after_load = shared._current_rss_mib()
    peak_after_load = shared._peak_rss_mib()
    print(
        f"{model_id}: loaded in {cold_load_s:.3f}s; RSS={rss_after_load:.1f} MiB",
        flush=True,
    )

    grouped_rows: dict[str, dict[str, dict]] = {group: {} for group in GROUP_ORDER}
    case_measurements: dict[str, dict[str, dict]] = {group: {} for group in GROUP_ORDER}
    for group in GROUP_ORDER:
        for case in groups[group]:
            observation = adapter.decode(case)
            row = _streaming_score(case, observation.content)
            if group == "stressors":
                _add_stressor_baseline(row, grouped_rows["short"], expected)
            _validate_streaming_row(case, row)
            grouped_rows[group][case.case_id] = row
            case_measurements[group][case.case_id] = _case_measurement(case, observation)
            print(
                f"{model_id}: compatibility/{group}/{case.case_id} "
                f"CER={row['cer']:.2%} final={row['finalization_count']} "
                f"reset={row['segment_reset_count']}",
                flush=True,
            )
    _validate_compatibility_rows(groups, grouped_rows, expected)

    benchmark_case = next(case for case in groups["stressors"] if case.case_id == "stress_long")
    post_warm = _benchmark(
        adapter,
        benchmark_case,
        grouped_rows["stressors"]["stress_long"],
        grouped_rows["short"],
        expected,
    )
    print(
        f"{model_id}: post-warm median decode RTF={post_warm['median_decode_rtf']:.3f}",
        flush=True,
    )

    def progress(index: int, total: int, row: Mapping[str, Any]) -> None:
        if index == 1 or index % shared.PROGRESS_ROWS == 0 or index == total:
            print(
                f"{model_id}: corpus {index}/{total}; {row['case_id']} CER={row['cer']:.2%}",
                flush=True,
            )

    corpus_result = _write_corpus_detail_resumable(
        detail_path,
        corpus_cases,
        adapter.decode,
        {
            "adapter": identity,
            "engine": model_id,
            "inputs": inputs,
            "schema_version": SCHEMA_VERSION,
        },
        progress=progress,
    )

    observed_peaks = [
        measurement["rss_mib"]["observed_peak"]
        for group in GROUP_ORDER
        for measurement in case_measurements[group].values()
    ]
    observed_peaks.extend(run["rss_mib"]["observed_peak"] for run in post_warm["runs"])
    measurements = {
        "cases": case_measurements,
        "cold_recognizer_load_s": round(cold_load_s, 6),
        "corpus": corpus_result["measurement"],
        "device_memory_mib": None,
        "device_memory_note": "CPU path has no separate accelerator-memory allocation.",
        "isolated_process": True,
        "post_warm": post_warm,
        "rss_mib": {
            "current_after_load": round(rss_after_load, 3),
            "current_before_load": round(rss_before_load, 3),
            "finite_observation_note": (
                "Time-sampled RSS covers this finite compatibility, benchmark, and "
                "corpus run only; "
                "it does not prove streaming memory is bounded."
            ),
            "observed_streaming_peak": max(observed_peaks),
            "ru_maxrss_after_load": round(peak_after_load, 3),
            "ru_maxrss_before_load": round(peak_before_load, 3),
            "ru_maxrss_process": round(shared._peak_rss_mib(), 3),
            "sample_interval_audio_s": RSS_SAMPLE_BLOCKS * BLOCK_MS / 1000,
        },
    }
    summary = {
        "adapter": identity,
        "compatibility": grouped_rows,
        "details": {
            "rows": corpus_result["rows"],
            "sha256": corpus_result["sha256"],
        },
        "diagnostics": _streaming_diagnostics(),
        "engine": model_id,
        "inputs": inputs,
        "measurements": measurements,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
    }
    shared.write_atomic(summary_path, [shared._json_bytes(summary)])
    _clear_corpus_resume_artifacts(detail_path)
    print(
        f"{model_id}: complete; detail={corpus_result['sha256']}; "
        f"corpus decode RTF={corpus_result['measurement']['overall_rtf']:.3f}",
        flush=True,
    )


def _reusable_child(
    model_id: str,
    summary_path: Path,
    detail_path: Path,
    corpus_manifest: Mapping[str, Any],
    corpus_cases: Sequence[shared.EvalCase],
    groups: Mapping[str, Sequence[shared.EvalCase]],
    expected: Mapping[str, Any],
) -> dict | None:
    if not summary_path.is_file():
        return None
    try:
        summary = shared._load_json(summary_path)
        if (
            summary.get("schema_version") != SCHEMA_VERSION
            or summary.get("engine") != model_id
            or summary.get("inputs") != evaluation_inputs(groups, corpus_manifest)
        ):
            raise RuntimeError("staged streaming child fingerprint drift")
        status = summary.get("status")
        if status == "runtime_incompatible":
            if summary.get("model") != model_fingerprint(STREAMING_SPECS[model_id]):
                raise RuntimeError("staged streaming failure model drift")
            _runtime_failure_result(summary)
            _clear_corpus_resume_artifacts(detail_path, include_detail=True)
            return summary
        if status != "complete" or not detail_path.is_file():
            raise RuntimeError("staged streaming child is not complete")
        if summary["adapter"]["model"] != model_fingerprint(STREAMING_SPECS[model_id]) or summary[
            "details"
        ]["rows"] != len(corpus_cases):
            raise RuntimeError("staged streaming child fingerprint drift")
        _detail_rows(detail_path, corpus_cases, summary["details"]["sha256"])
        _validate_compatibility_rows(groups, summary["compatibility"], expected)
        if summary["diagnostics"] != _streaming_diagnostics():
            raise RuntimeError("staged streaming diagnostic contract drift")
        measurement = summary["measurements"]
        if set(measurement["cases"]) != set(GROUP_ORDER) or measurement["corpus"]["rows"] != len(
            corpus_cases
        ):
            raise RuntimeError("staged streaming measurements drift")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"{model_id}: discard unusable staged streaming child ({exc})", flush=True)
        summary_path.unlink(missing_ok=True)
        _clear_corpus_resume_artifacts(detail_path, include_detail=True)
        return None
    _clear_corpus_resume_artifacts(detail_path)
    return summary


def build_manifest(
    corpus_manifest: Mapping[str, Any],
    corpus_cases: Sequence[shared.EvalCase],
    summaries: Mapping[str, dict],
    detail_paths: Mapping[str, Path],
    model_ids: Sequence[str],
    groups: Mapping[str, Sequence[shared.EvalCase]],
    expected: Mapping[str, Any],
) -> dict:
    models = shared.validate_evidence_model_set(
        summaries,
        detail_paths,
        model_ids,
        label="streaming evaluator",
    )
    inputs = evaluation_inputs(groups, corpus_manifest)
    comparator = _comparator_snapshot()
    variants: dict[str, dict] = {}
    variant_measurements: dict[str, dict] = {}
    for model_id in models:
        summary = summaries[model_id]
        if summary.get("engine") != model_id or summary.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"invalid streaming child summary: {model_id}")
        if summary.get("inputs") != inputs:
            raise RuntimeError(f"{model_id}: child input fingerprint drift")
        status = summary.get("status")
        if status == "runtime_incompatible":
            if summary.get("model") != model_fingerprint(STREAMING_SPECS[model_id]):
                raise RuntimeError(f"{model_id}: failed-child model fingerprint drift")
            result = _runtime_failure_result(summary)
            variants[model_id] = result
            resource = shared.resource_verdict({})
            variant_measurements[model_id] = {
                "resource_verdict": resource,
            }
            continue
        if status != "complete":
            raise RuntimeError(f"{model_id}: child status drift")
        if summary["adapter"]["model"] != model_fingerprint(STREAMING_SPECS[model_id]):
            raise RuntimeError(f"{model_id}: child model fingerprint drift")
        details = summary["details"]
        if details["rows"] != len(corpus_cases):
            raise RuntimeError(f"{model_id}: incomplete streaming child row count")
        rows = _detail_rows(detail_paths[model_id], corpus_cases, details["sha256"])
        aggregates = shared.aggregate_rows(rows)
        if not aggregates["completion"]["complete"]:
            raise RuntimeError(f"{model_id}: incomplete transcript contract")
        compatibility = summary["compatibility"]
        _validate_compatibility_rows(groups, compatibility, expected)
        diagnostics = summary["diagnostics"]
        if diagnostics != _streaming_diagnostics():
            raise RuntimeError(f"{model_id}: streaming diagnostic contract drift")
        detail_destination = DETAILS_DIR / f"{model_id}-{details['sha256'][:16]}.jsonl"
        result = {
            "adapter": summary["adapter"],
            "aggregates": aggregates,
            "compatibility": compatibility,
            "details": {
                "path": shared._relative(detail_destination),
                "rows": details["rows"],
                "sha256": details["sha256"],
            },
            "diagnostics": diagnostics,
            "status": "complete",
        }
        result["content_verdict"] = shared.content_verdict(result, comparator)
        variants[model_id] = result

        measurement = summary["measurements"]
        if set(measurement["cases"]) != set(GROUP_ORDER) or measurement["corpus"]["rows"] != len(
            corpus_cases
        ):
            raise RuntimeError(f"{model_id}: measurement groups drifted")
        path = {
            "actual_execution_device": summary["adapter"]["actual_execution_device"],
            "post_warm_rtf": measurement["post_warm"]["median_decode_rtf"],
            "production_eligible": True,
        }
        variant_measurements[model_id] = {
            **measurement,
            "resource_verdict": shared.resource_verdict({"sherpa_cpu": path}),
        }

    completed_ids = [model_id for model_id in models if variants[model_id]["status"] == "complete"]
    non_dominated = _non_dominated_variants(
        {model_id: variants[model_id] for model_id in completed_ids},
        {model_id: variant_measurements[model_id] for model_id in completed_ids},
    )
    non_dominated_set = set(non_dominated)
    for model_id in models:
        measurement = variant_measurements[model_id]
        measurement["displacement_verdict"] = _displacement_verdict(
            variants[model_id]["content_verdict"],
            measurement["resource_verdict"],
            non_dominated=model_id in non_dominated_set,
        )

    deterministic = {
        "comparator": comparator,
        "corpus": shared._corpus_fingerprint(dict(corpus_manifest)),
        "displacement_gates": shared.DISPLACEMENT_GATES,
        "inputs": inputs,
        "metric_contract": STREAMING_METRIC_CONTRACT,
        "scope": {
            "case_groups": {group: len(groups[group]) for group in GROUP_ORDER},
            "full_short_corpus_rows": len(corpus_cases),
            "m10_unit": "M10.5d",
            "note": (
                "Full-corpus streaming tournament machinery; measurements identify the "
                "content/RTF non-dominated set."
            ),
        },
        "variants": variants,
    }
    return {
        "deterministic": deterministic,
        "deterministic_sha256": hashlib.sha256(shared._json_bytes(deterministic)).hexdigest(),
        "measurements": {
            "excluded_from_deterministic_equality": True,
            "non_dominated": {
                "objectives": ["common_voice_8.micro.cer", "post_warm.median_decode_rtf"],
                "variants": non_dominated,
            },
            "note": (
                "Elapsed time and sampled RSS vary; scored content, logical-audio "
                "events, and fingerprints live under deterministic."
            ),
            "variants": variant_measurements,
        },
        "schema_version": SCHEMA_VERSION,
    }


def install_evidence(
    manifest: dict,
    summaries: Mapping[str, dict],
    staged_details: Mapping[str, Path],
    model_ids: Sequence[str],
) -> None:
    complete_ids = tuple(
        model_id for model_id in model_ids if summaries[model_id].get("status") == "complete"
    )
    complete_summaries = {model_id: summaries[model_id] for model_id in complete_ids}
    complete_details = {model_id: staged_details[model_id] for model_id in complete_ids}
    destinations = shared.install_content_addressed_details(
        complete_summaries,
        complete_details,
        complete_ids,
        details_dir=DETAILS_DIR,
    )
    for model_id, destination in destinations.items():
        recorded = manifest["deterministic"]["variants"][model_id]["details"]["path"]
        if recorded != shared._relative(destination):
            raise RuntimeError(f"{model_id}: streaming detail destination drift")
    shared.write_atomic(BASELINE, [shared._json_bytes(manifest)])


def reaggregate_parent() -> None:
    previous = shared._load_json(BASELINE)
    if previous.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("streaming aggregate-only refused: evaluator schema changed")
    corpus_manifest, corpus_cases = shared.load_corpus_cases(verify_pcm=True)
    groups, expected = small_clip_cases(verify_audio=True)
    deterministic = previous["deterministic"]
    if deterministic.get("inputs") != evaluation_inputs(groups, corpus_manifest):
        raise RuntimeError("streaming aggregate-only refused: input fingerprint changed")
    model_ids = tuple(deterministic["variants"])
    summaries = {}
    details = {}
    for model_id in model_ids:
        result = deterministic["variants"][model_id]
        current_model = model_fingerprint(STREAMING_SPECS[model_id])
        measurement = dict(previous["measurements"]["variants"][model_id])
        measurement.pop("displacement_verdict", None)
        measurement.pop("resource_verdict", None)
        if result.get("status") == "runtime_incompatible":
            if result["model"] != current_model:
                raise RuntimeError(f"streaming aggregate-only refused: {model_id} model changed")
            summaries[model_id] = {
                "engine": model_id,
                "failure": result["failure"],
                "inputs": deterministic["inputs"],
                "model": result["model"],
                "schema_version": SCHEMA_VERSION,
                "status": "runtime_incompatible",
            }
            details[model_id] = STAGING_DIR / f"{model_id}.jsonl"
            continue
        if result.get("status") != "complete" or result["adapter"]["model"] != current_model:
            raise RuntimeError(f"streaming aggregate-only refused: {model_id} model changed")
        summaries[model_id] = {
            "adapter": result["adapter"],
            "compatibility": result["compatibility"],
            "details": {
                "rows": result["details"]["rows"],
                "sha256": result["details"]["sha256"],
            },
            "diagnostics": result["diagnostics"],
            "engine": model_id,
            "inputs": deterministic["inputs"],
            "measurements": measurement,
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
        }
        details[model_id] = ROOT / result["details"]["path"]
    manifest = build_manifest(
        corpus_manifest,
        corpus_cases,
        summaries,
        details,
        model_ids,
        groups,
        expected,
    )
    if manifest != previous:
        raise RuntimeError("streaming aggregate-only byte rebuild drifted")
    install_evidence(manifest, summaries, details, model_ids)
    print(
        f"PASS: streaming aggregate rebuild byte-stable; "
        f"deterministic={manifest['deterministic_sha256']}",
        flush=True,
    )


def _selected_variants(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(values) or DEFAULT_VARIANTS
    if names == ("all",):
        return STREAMING_MODEL_IDS
    if "all" in names:
        raise RuntimeError("'all' cannot be combined with named streaming variants")
    model_ids = tuple(streaming_models.CANDIDATE_SPECS[name].model_id for name in names)
    if len(model_ids) != len(set(model_ids)):
        raise RuntimeError("duplicate streaming variant")
    return model_ids


def run_parent(model_ids: Sequence[str]) -> None:
    print("validating pinned corpus PCM + index before isolated runs", flush=True)
    corpus_manifest, corpus_cases = shared.load_corpus_cases(verify_pcm=True)
    groups, expected = small_clip_cases(verify_audio=True)
    for model_id in model_ids:
        model_fingerprint(STREAMING_SPECS[model_id])

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    installed = False
    try:

        def reusable_child(model_id: str, summary: Path, detail: Path) -> dict | None:
            return _reusable_child(
                model_id,
                summary,
                detail,
                corpus_manifest,
                corpus_cases,
                groups,
                expected,
            )

        summaries, details = shared.run_isolated_workers(
            model_ids,
            staging=STAGING_DIR,
            worker_script=Path(__file__).resolve(),
            reusable_child=reusable_child,
        )
        manifest = build_manifest(
            corpus_manifest,
            corpus_cases,
            summaries,
            details,
            model_ids,
            groups,
            expected,
        )
        install_evidence(manifest, summaries, details, model_ids)
        installed = True
    finally:
        if installed:
            shutil.rmtree(STAGING_DIR, ignore_errors=True)

    for model_id in model_ids:
        variant = manifest["deterministic"]["variants"][model_id]
        measurement = manifest["measurements"]["variants"][model_id]
        verdict = measurement["displacement_verdict"]
        if variant["status"] == "runtime_incompatible":
            print(
                f"{model_id}: runtime_incompatible={variant['failure']['message']}",
                flush=True,
            )
            continue
        print(
            f"{model_id}: content_qualified={variant['content_verdict']['qualified']} "
            f"cpu_resource_qualified={measurement['resource_verdict']['qualified']} "
            f"non_dominated={verdict['non_dominated']} "
            f"decode_RTF={measurement['post_warm']['median_decode_rtf']:.3f} "
            f"RSS_peak={measurement['rss_mib']['observed_streaming_peak']:.1f} MiB",
            flush=True,
        )
    print(
        f"PASS: wrote {shared._relative(BASELINE)}; "
        f"non_dominated={manifest['measurements']['non_dominated']['variants']}; "
        f"deterministic={manifest['deterministic_sha256']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "variants",
        nargs="*",
        choices=[*streaming_models.CANDIDATE_SPECS, "all"],
        help="Streaming variants (default: 560ms; use 'all' for the M10.5e tournament).",
    )
    parser.add_argument("--worker", choices=STREAMING_MODEL_IDS, help=argparse.SUPPRESS)
    parser.add_argument("--details", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--summary", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild the exact cached full-corpus baseline without model decode.",
    )
    args = parser.parse_args()
    if args.worker:
        if args.aggregate_only or args.variants:
            parser.error("--worker cannot be combined with public modes")
        if args.details is None or args.summary is None:
            parser.error("--worker requires --details and --summary")
        run_worker(args.worker, args.details, args.summary)
    elif args.details is not None or args.summary is not None:
        parser.error("--details/--summary are internal worker arguments")
    elif args.aggregate_only:
        if args.variants:
            parser.error("--aggregate-only does not accept variants")
        reaggregate_parent()
    else:
        run_parent(_selected_variants(args.variants))


if __name__ == "__main__":
    main()
