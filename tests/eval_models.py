#!/usr/bin/env python3
"""Model-neutral M10 ASR evaluator + offline-candidate tournament.

The default command validates the pinned corpus, evaluates both shipped controls
and both M10 offline candidates in separate child processes, then atomically publishes:

- ignored, content-addressed per-clip JSONL under ``spike/backends/cache/``;
- ``tests/model_baseline.json`` with deterministic aggregates/fingerprints;
- a structurally separate measurements block (cold load, RSS, repeated warm RTF).

``ASRAdapter.decode()`` is the tournament contract. The shipped-controls adapter
feeds sherpa OfflineRecognizer through replay.py -> the production worker. Later
direct-online and direct-OpenVINO adapters can return the same content/completion
record without pretending to share the buffered-offline execution path.

Run from the repository root after M10.2's corpus build:

    UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python tests/eval_models.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Any, Protocol

import sherpa_onnx

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

import fetch_eval_models as candidate_models  # noqa: E402
from fetch_real_clips import (  # noqa: E402
    file_sha256,
    validate_cached_index,
    write_atomic,
)

import replay  # noqa: E402
from cer import align, normalize  # noqa: E402
from live_stt import (  # noqa: E402
    DECODE_CHUNK_OVERLAP_S,
    DECODE_CHUNK_S,
    DECODE_SPLIT_RMS_WINDOW_S,
    DECODE_SPLIT_SEARCH_S,
    DECODE_SPLIT_TRIGGER_S,
    ENGINE_DIRS,
    NUM_THREADS,
    RING_SECONDS,
    SAMPLE_RATE,
    VAD_MAX_SPEECH_S,
    VAD_MIN_SILENCE_S,
    VAD_MIN_SPEECH_S,
    VAD_MODEL,
    VAD_PRE_PAD_S,
    check_models,
    load_recognizer,
)

CANDIDATE_SPECS = candidate_models.CANDIDATE_SPECS
candidate_provenance = candidate_models.provenance
validate_candidate_model = candidate_models.validate_installed

SCHEMA_VERSION = 2
BASELINE = TESTS / "model_baseline.json"
SHORT_CORPUS = TESTS / "short_corpus.json"
REPLAY_GOLDENS = TESTS / "replay_goldens.json"
CER_BASELINE = TESTS / "cer_baseline.json"
STRESSORS = TESTS / "stressor_clips.json"
LONG_FORM = TESTS / "long_form.json"
CACHE = ROOT / "spike" / "backends" / "cache"
DETAILS_DIR = CACHE / "model_eval-v2"

CONTROL_ENGINES = ("k2v2", "parakeet")
OFFLINE_CANDIDATES = ("qwen3_asr", "cohere_transcribe")
OFFLINE_MODELS = (*CONTROL_ENGINES, *OFFLINE_CANDIDATES)
STRESSOR_IDS = ("stress_long", "stress_med")
LONG_FORM_ID = "gongitsune_01"
TAIL_EXEMPLARS = 10
TIMING_RUNS = 3
PROGRESS_ROWS = 25

# Frozen before expanded-control scores. A candidate's content result and each
# execution path's resource result are evaluated independently (M10 plan gate).
DISPLACEMENT_GATES = {
    "comparator": {
        "metric": "common_voice_8.micro.cer",
        "selection": "minimum",
        "tie_break": "engine_id_ascending",
    },
    "content": {
        "common_voice_relative_cer_improvement_min": 0.10,
        "fleurs_micro_cer_regression_max_abs": 0.01,
        "long_form_cer_regression_max_abs": 0.01,
        "require_clean_decoder_diagnostics": True,
        "require_complete_run": True,
        "require_nonempty_run": True,
        "stressor_cer_max": 0.15,
        "stressor_excess_deletion_rate_max": 0.04,
    },
    "resource": {
        "cpu_miss_disqualifies_content": False,
        "production_eligible_paths_min": 1,
        "post_warm_rtf_max": 0.20,
    },
}

METRIC_CONTRACT = {
    "alignment": "cer.normalize + diagonal-preferred Levenshtein S/D/I",
    "duration_buckets_s": ["[0,5)", "[5,10)", "[10,20)", "[20,+inf)"],
    "empty_hypothesis": "complete row; scored as D=N and retained by case ID",
    "macro": "arithmetic mean of per-row S/D/I/CER rates; repeated recordings remain rows",
    "micro": "sum(S/D/I) over sum(N)",
    "worst_tail": "top 10 rows/source by exact CER descending, then case ID ascending",
}

CONTROL_MODEL_FILES = {
    "k2v2": (
        "encoder-epoch-99-avg-1.int8.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.onnx",
        "tokens.txt",
    ),
    "parakeet": ("model.int8.onnx", "tokens.txt"),
}

NATIVE_DIAGNOSTIC_CODES = {
    "Result is truncated. max_new_tokens": "generation_max_new_tokens",
    "Truncating audio placeholders": "audio_context_truncated",
}


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    source: str
    wav: Path
    reference: str
    duration_samples: int
    gender: str | None = None
    duration_bucket: str | None = None


@dataclass(frozen=True)
class Transcript:
    hypothesis: str
    segments: tuple[dict, ...]
    accepted_samples: int
    eof_count: int
    complete: bool


@dataclass(frozen=True)
class DecodeObservation:
    """Content/completion is deterministic; elapsed fields are measurements."""

    content: Transcript
    decode_seconds: float
    wall_seconds: float


class ASRAdapter(Protocol):
    adapter_id: str

    def identity(self) -> dict: ...

    def decode(self, case: EvalCase) -> DecodeObservation: ...


def _native_diagnostic_code(message: str) -> str:
    return next(
        (code for marker, code in NATIVE_DIAGNOSTIC_CODES.items() if marker in message),
        "native_stderr",
    )


def _decode_with_native_diagnostics(
    decode: Callable[[], DecodeObservation],
) -> tuple[DecodeObservation, tuple[str, ...]]:
    """Capture C/C++ fd-2 output around one synchronous candidate decode."""
    with tempfile.TemporaryFile() as captured:
        saved_stderr = os.dup(2)
        try:
            os.dup2(captured.fileno(), 2)
            observation = decode()
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        captured.seek(0)
        raw = captured.read()
    if raw:
        os.write(2, raw)
    lines = tuple(line for line in raw.decode("utf-8", errors="replace").splitlines() if line)
    return observation, lines


def candidate_diagnostic_contract() -> dict:
    source = "\n\n".join(
        inspect.getsource(obj) for obj in (_native_diagnostic_code, _decode_with_native_diagnostics)
    ).encode()
    return {
        "codes": NATIVE_DIAGNOSTIC_CODES,
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "scope": (
            "native fd-2 lines captured per candidate decode; content and timing kept distinct"
        ),
    }


class SherpaOfflineAdapter:
    """Sherpa offline recognizer behind the exact production VAD/ring/chunker."""

    def __init__(self, engine: str):
        self.adapter_id = engine
        self.recognizer: Any = load_evaluator_recognizer(engine)
        model_config = self.recognizer.config.model_config
        if model_config.provider != "cpu":
            raise RuntimeError(
                f"{engine}: expected sherpa CPU provider, got {model_config.provider!r}"
            )
        if model_config.num_threads != NUM_THREADS:
            raise RuntimeError(
                f"{engine}: recognizer threads {model_config.num_threads} != {NUM_THREADS}"
            )

    def identity(self) -> dict:
        cfg = self.recognizer.config
        model = cfg.model_config
        feat = cfg.feat_config
        return {
            "adapter": "sherpa_offline_production_worker",
            "adapter_id": self.adapter_id,
            "architecture": "buffered_offline_vad",
            "actual_execution_device": "CPU",
            "execution_device_evidence": {
                "field": "OfflineRecognizer.config.model_config.provider",
                "value": model.provider,
            },
            "feature_config": {
                "dither": feat.dither,
                "feature_dim": feat.feature_dim,
                "normalize_samples": feat.normalize_samples,
                "sample_rate_hz": feat.sampling_rate,
                "snip_edges": feat.snip_edges,
            },
            "model": model_fingerprint(self.adapter_id),
            "recognizer_config": recognizer_config(self.adapter_id, cfg),
        }

    def decode(self, case: EvalCase) -> DecodeObservation:
        started = time.perf_counter()
        report = replay.replay_recognizer(case.wav, self.recognizer, self.adapter_id)
        wall_seconds = time.perf_counter() - started
        observed_samples = round(report["audio_s"] * SAMPLE_RATE)
        if observed_samples != case.duration_samples:
            raise RuntimeError(
                f"{case.case_id}: audio samples {observed_samples} != {case.duration_samples}"
            )
        segments = tuple(
            {
                "n": row["n"],
                "seg_len": row["seg_len"],
                "start": row["start"],
                "text": row["text"],
            }
            for row in report["segments"]
        )
        return DecodeObservation(
            content=Transcript(
                hypothesis="".join(row["text"] for row in segments),
                segments=segments,
                accepted_samples=observed_samples,
                eof_count=1,
                complete=True,
            ),
            decode_seconds=report["total_decode_s"],
            wall_seconds=wall_seconds,
        )


def load_evaluator_recognizer(engine: str) -> sherpa_onnx.OfflineRecognizer:
    """Construct controls through production; candidates remain evaluator-only."""
    if engine in CONTROL_ENGINES:
        rec = load_recognizer(engine)
        # Production `load_recognizer` also builds the OpenVINO engine, which this
        # evaluator cannot score: a WhisperEngine here is a mis-specified control.
        if not isinstance(rec, sherpa_onnx.OfflineRecognizer):
            raise TypeError(f"control engine {engine} is not a sherpa recognizer")
        return rec
    spec = CANDIDATE_SPECS[engine]
    model = validate_candidate_model(spec)
    if engine == "qwen3_asr":
        return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
            conv_frontend=str(model / "conv_frontend.onnx"),
            encoder=str(model / "encoder.int8.onnx"),
            decoder=str(model / "decoder.int8.onnx"),
            tokenizer=str(model / "tokenizer"),
            num_threads=NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            feature_dim=128,
            decoding_method="greedy_search",
            debug=False,
            provider="cpu",
            max_total_len=512,
            max_new_tokens=128,
            temperature=1e-6,
            top_p=0.8,
            seed=42,
            hotwords="",
        )
    return sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
        encoder=str(model / "encoder.int8.onnx"),
        decoder=str(model / "decoder.int8.onnx"),
        tokens=str(model / "tokens.txt"),
        num_threads=NUM_THREADS,
        language="ja",
        use_punct=True,
        use_itn=True,
        decoding_method="greedy_search",
        debug=False,
        provider="cpu",
    )


def recognizer_config(engine: str, cfg) -> dict:
    model = cfg.model_config
    output = {
        "decoding_method": cfg.decoding_method,
        "num_threads": model.num_threads,
        "provider": model.provider,
    }
    if engine in CONTROL_ENGINES:
        output["model_type"] = model.model_type
    elif engine == "qwen3_asr":
        qwen = model.qwen3_asr
        output["qwen3_asr"] = {
            "hotwords": qwen.hotwords,
            "max_new_tokens": qwen.max_new_tokens,
            "max_total_len": qwen.max_total_len,
            "seed": qwen.seed,
            "temperature": qwen.temperature,
            "top_p": qwen.top_p,
        }
    else:
        cohere = model.cohere_transcribe
        output["cohere_transcribe"] = {
            "language": cohere.language,
            "use_itn": cohere.use_itn,
            "use_punct": cohere.use_punct,
        }
    return output


def _json_bytes(value: object, *, compact: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=None if compact else 2,
            separators=(",", ":") if compact else None,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _duration_bucket(samples: int) -> str:
    if samples < 5 * SAMPLE_RATE:
        return "0-5"
    if samples < 10 * SAMPLE_RATE:
        return "5-10"
    if samples < 20 * SAMPLE_RATE:
        return "10-20"
    return "20+"


def _wav_samples(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != SAMPLE_RATE
                or source.getcomptype() != "NONE"
            ):
                raise RuntimeError(f"non-canonical evaluator WAV: {_relative(path)}")
            return source.getnframes()
    except (EOFError, OSError, wave.Error) as exc:
        raise RuntimeError(f"cannot read evaluator WAV {_relative(path)}: {exc}") from exc


def _safe_corpus_wav(corpus_dir: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".wav":
        raise RuntimeError(f"unsafe corpus WAV path: {relative!r}")
    wav = corpus_dir.joinpath(*path.parts)
    if not wav.is_file():
        raise RuntimeError(f"missing corpus WAV: {relative}")
    return wav


def load_corpus_cases(*, verify_pcm: bool) -> tuple[dict, list[EvalCase]]:
    manifest = _load_json(SHORT_CORPUS)
    cache_meta = manifest["cache"]
    corpus_dir = ROOT / cache_meta["directory"]
    index = corpus_dir / cache_meta["index"]
    if not index.is_file() or file_sha256(index) != cache_meta["index_sha256"]:
        raise RuntimeError("short-corpus index fingerprint mismatch")
    if verify_pcm:
        entries = validate_cached_index(corpus_dir, cache_meta["index_sha256"])
    else:
        entries = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    if len(entries) != cache_meta["rows"]:
        raise RuntimeError(f"short-corpus rows {len(entries)} != {cache_meta['rows']}")

    cases = []
    seen: set[str] = set()
    for entry in entries:
        case_id = entry["corpus_id"]
        if case_id in seen:
            raise RuntimeError(f"duplicate corpus ID: {case_id}")
        seen.add(case_id)
        duration_samples = entry["duration_samples"]
        if entry["normalized_reference"] != normalize(entry["reference"]):
            raise RuntimeError(f"normalized reference drift: {case_id}")
        cases.append(
            EvalCase(
                case_id=case_id,
                source=entry["source"],
                wav=_safe_corpus_wav(corpus_dir, entry["wav"]),
                reference=entry["reference"],
                duration_samples=duration_samples,
                gender=entry.get("gender"),
                duration_bucket=_duration_bucket(duration_samples),
            )
        )
    return manifest, cases


def compatibility_cases() -> tuple[dict[str, list[EvalCase]], dict]:
    goldens = _load_json(REPLAY_GOLDENS)
    cer_baseline = _load_json(CER_BASELINE)
    stressors = _load_json(STRESSORS)
    long_form = _load_json(LONG_FORM)

    short = []
    first_engine = CONTROL_ENGINES[0]
    # Subset, not equality: the golden matrix also carries accelerator-bound engines
    # this evaluator never scores, and only the controls reach a case below.
    if not set(CONTROL_ENGINES) <= set(goldens):
        raise RuntimeError("replay-golden controls drifted")
    for case_id, meta in goldens[first_engine].items():
        if any(goldens[engine][case_id]["ja_ref"] != meta["ja_ref"] for engine in CONTROL_ENGINES):
            raise RuntimeError(f"cross-engine short reference drift: {case_id}")
        wav = CACHE / f"{case_id}.wav"
        short.append(
            EvalCase(
                case_id=case_id,
                source="compat_short",
                wav=wav,
                reference=meta["ja_ref"],
                duration_samples=_wav_samples(wav),
            )
        )

    stress = []
    for case_id in STRESSOR_IDS:
        wav = CACHE / f"{case_id}.wav"
        stress.append(
            EvalCase(
                case_id=case_id,
                source="continuous_stressor",
                wav=wav,
                reference=stressors["stressors"][case_id]["ja_ref"],
                duration_samples=_wav_samples(wav),
            )
        )

    long_wav = ROOT / long_form["build"]["wav"]
    long = [
        EvalCase(
            case_id=LONG_FORM_ID,
            source="long_form",
            wav=long_wav,
            reference=long_form["reference"]["text"],
            duration_samples=_wav_samples(long_wav),
        )
    ]
    expected = {
        "goldens": goldens,
        "cer_baseline": cer_baseline,
        "stressor_manifest": stressors,
        "long_form": long_form,
    }
    return {"short": short, "stressors": stress, "long_form": long}, expected


def _score(case: EvalCase, transcript: Transcript) -> dict:
    ref = normalize(case.reference)
    if not ref:
        raise RuntimeError(f"reference normalizes to empty: {case.case_id}")
    hyp = normalize(transcript.hypothesis)
    substitutions, deletions, insertions = align(ref, hyp)
    n = len(ref)
    return {
        "D": deletions,
        "I": insertions,
        "N": n,
        "S": substitutions,
        "accepted_samples": transcript.accepted_samples,
        "case_id": case.case_id,
        "cer": (substitutions + deletions + insertions) / n,
        "complete": transcript.complete,
        "duration_bucket": case.duration_bucket,
        "duration_samples": case.duration_samples,
        "eof_count": transcript.eof_count,
        "gender": case.gender,
        "hyp": transcript.hypothesis,
        "ref": case.reference,
        "segments": list(transcript.segments),
        "source": case.source,
    }


def _expected_score(ref: str, hyp: str) -> tuple[int, int, int, int]:
    normalized_ref = normalize(ref)
    substitutions, deletions, insertions = align(normalized_ref, normalize(hyp))
    return len(normalized_ref), substitutions, deletions, insertions


def _compatibility_output(row: dict) -> dict:
    keys = (
        "D",
        "I",
        "N",
        "S",
        "accepted_samples",
        "cer",
        "complete",
        "duration_samples",
        "eof_count",
        "hyp",
        "ref",
    )
    output = {key: row[key] for key in keys}
    output["n_segments"] = len(row["segments"])
    return output


def _control_compatibility_row(
    engine: str, group: str, case: EvalCase, row: dict, expected: dict
) -> dict:
    if group == "short":
        golden = expected["goldens"][engine][case.case_id]
        expected_hyp = "".join(segment["text"] for segment in golden["segments"])
        expected_segments = [segment["text"] for segment in golden["segments"]]
        observed_segments = [segment["text"] for segment in row["segments"]]
        if observed_segments != expected_segments:
            raise RuntimeError(f"{engine}/{case.case_id}: short segment hypotheses drifted")
        legacy = expected["cer_baseline"]["corpus"][engine].get(case.case_id)
    elif group == "stressors":
        legacy = expected["cer_baseline"]["stressors"][engine][case.case_id]
        expected_hyp = legacy["hyp"]
    else:
        legacy = expected["long_form"]["scores"][engine]
        expected_hyp = legacy["hyp"]

    n, substitutions, deletions, insertions = _expected_score(case.reference, expected_hyp)
    expected_counts = (n, substitutions, deletions, insertions)
    observed_counts = (row["N"], row["S"], row["D"], row["I"])
    if row["hyp"] != expected_hyp or observed_counts != expected_counts:
        raise RuntimeError(
            f"{engine}/{case.case_id}: compatibility drift; "
            f"counts={observed_counts} expected={expected_counts}"
        )
    if legacy is not None:
        legacy_counts = (legacy["N"], legacy["S"], legacy["D"], legacy["I"])
        if legacy["hyp"] != row["hyp"] or legacy_counts != observed_counts:
            raise RuntimeError(f"{engine}/{case.case_id}: legacy score drift")

    output = _compatibility_output(row)
    if group == "stressors":
        if legacy is None:
            raise RuntimeError(f"{engine}/{case.case_id}: missing stressor baseline")
        order = expected["stressor_manifest"]["stressors"][case.case_id]["order"]
        baseline_d = sum(
            expected["stressor_manifest"]["components"][component]["baseline"][engine]["D"]
            for component in order
        )
        output["baseline_D"] = baseline_d
        output["excess_D"] = row["D"] - baseline_d
        output["excess_del_rate"] = round(output["excess_D"] / row["N"], 4)
        if (
            output["excess_D"] != legacy["excess_D"]
            or output["excess_del_rate"] != legacy["excess_del_rate"]
        ):
            raise RuntimeError(f"{engine}/{case.case_id}: excess-deletion drift")
    return output


def _candidate_compatibility_row(
    group: str,
    case: EvalCase,
    row: dict,
    compatibility: Mapping[str, dict],
    expected: dict,
) -> dict:
    output = _compatibility_output(row)
    if group == "stressors":
        order = expected["stressor_manifest"]["stressors"][case.case_id]["order"]
        short = compatibility["short"]
        if not set(order) <= short.keys():
            raise RuntimeError(f"{case.case_id}: candidate stressor baselines are incomplete")
        baseline_d = sum(short[component]["D"] for component in order)
        output["baseline_D"] = baseline_d
        output["excess_D"] = row["D"] - baseline_d
        output["excess_del_rate"] = round(output["excess_D"] / row["N"], 4)
    return output


def validate_compatibility_snapshot(engine: str, snapshot: dict, expected: dict) -> None:
    """Recheck a child/committed legacy snapshot without running either model."""
    if set(snapshot) != {"short", "stressors", "long_form"}:
        raise RuntimeError(f"{engine}: compatibility groups drifted")

    goldens = expected["goldens"][engine]
    if set(snapshot["short"]) != set(goldens):
        raise RuntimeError(f"{engine}: short compatibility IDs drifted")
    for case_id, golden in goldens.items():
        row = snapshot["short"][case_id]
        hyp = "".join(segment["text"] for segment in golden["segments"])
        counts = _expected_score(golden["ja_ref"], hyp)
        if (
            row["hyp"] != hyp
            or row["ref"] != golden["ja_ref"]
            or row["n_segments"] != golden["n_segments"]
            or row["accepted_samples"] != row["duration_samples"]
            or row["complete"] is not True
            or row["eof_count"] != 1
            or tuple(row[key] for key in ("N", "S", "D", "I")) != counts
        ):
            raise RuntimeError(f"{engine}/{case_id}: short compatibility snapshot drifted")

    legacy_stress = expected["cer_baseline"]["stressors"][engine]
    if set(snapshot["stressors"]) != set(legacy_stress):
        raise RuntimeError(f"{engine}: stressor compatibility IDs drifted")
    for case_id, legacy in legacy_stress.items():
        row = snapshot["stressors"][case_id]
        keys = ("D", "I", "N", "S", "cer", "excess_D", "excess_del_rate", "hyp", "ref")
        if any(row[key] != legacy[key] for key in keys) or row["n_segments"] <= 0:
            raise RuntimeError(f"{engine}/{case_id}: stressor compatibility snapshot drifted")
        if row["baseline_D"] + row["excess_D"] != row["D"]:
            raise RuntimeError(f"{engine}/{case_id}: stressor baseline arithmetic drifted")

    legacy_long = expected["long_form"]["scores"][engine]
    if set(snapshot["long_form"]) != {LONG_FORM_ID}:
        raise RuntimeError(f"{engine}: long-form compatibility IDs drifted")
    long_row = snapshot["long_form"][LONG_FORM_ID]
    keys = ("D", "I", "N", "S", "cer", "hyp", "ref")
    if any(long_row[key] != legacy_long[key] for key in keys) or long_row["n_segments"] <= 0:
        raise RuntimeError(f"{engine}: long-form compatibility snapshot drifted")

    probes, _ = compatibility_cases()
    expected_samples = {
        case.case_id: case.duration_samples for cases in probes.values() for case in cases
    }
    for rows in snapshot.values():
        for case_id, row in rows.items():
            if (
                row["accepted_samples"] != expected_samples[case_id]
                or row["duration_samples"] != expected_samples[case_id]
                or row["complete"] is not True
                or row["eof_count"] != 1
            ):
                raise RuntimeError(f"{engine}/{case_id}: compatibility completion drifted")


def validate_candidate_compatibility_snapshot(engine: str, snapshot: dict, expected: dict) -> None:
    if set(snapshot) != {"short", "stressors", "long_form"}:
        raise RuntimeError(f"{engine}: compatibility groups drifted")
    probes, _ = compatibility_cases()
    for group, cases in probes.items():
        if set(snapshot[group]) != {case.case_id for case in cases}:
            raise RuntimeError(f"{engine}: {group} compatibility IDs drifted")
        for case in cases:
            row = snapshot[group][case.case_id]
            counts = _expected_score(case.reference, row["hyp"])
            if (
                row["ref"] != case.reference
                or row["accepted_samples"] != case.duration_samples
                or row["duration_samples"] != case.duration_samples
                or row["complete"] is not True
                or row["eof_count"] != 1
                or not isinstance(row["n_segments"], int)
                or row["n_segments"] < 0
                or tuple(row[key] for key in ("N", "S", "D", "I")) != counts
            ):
                raise RuntimeError(f"{engine}/{case.case_id}: compatibility snapshot drifted")

    short = snapshot["short"]
    stressor_manifest = expected["stressor_manifest"]["stressors"]
    for case_id, row in snapshot["stressors"].items():
        baseline_d = sum(short[name]["D"] for name in stressor_manifest[case_id]["order"])
        excess_d = row["D"] - baseline_d
        if (
            row["baseline_D"] != baseline_d
            or row["excess_D"] != excess_d
            or row["excess_del_rate"] != round(excess_d / row["N"], 4)
        ):
            raise RuntimeError(f"{engine}/{case_id}: candidate stressor baseline drifted")


def _candidate_decode(
    adapter: ASRAdapter,
    case: EvalCase,
    phase: str,
    events: list[dict],
) -> DecodeObservation:
    observation, messages = _decode_with_native_diagnostics(lambda: adapter.decode(case))
    events.extend(
        {
            "case_id": case.case_id,
            "code": _native_diagnostic_code(message),
            "message": message,
            "phase": phase,
        }
        for message in messages
    )
    return observation


def _is_content_diagnostic(event: Mapping[str, str]) -> bool:
    return event["phase"] == "corpus" or event["phase"].startswith("compatibility/")


def summarize_candidate_diagnostics(events: Sequence[dict]) -> dict:
    content = [event for event in events if _is_content_diagnostic(event)]
    return {
        "content_event_case_ids": sorted({event["case_id"] for event in content}),
        "content_events": content,
        "contract": candidate_diagnostic_contract(),
        "events": list(events),
    }


def candidate_diagnostic_phase_cases(
    corpus_cases: Sequence[EvalCase], probes: Mapping[str, Sequence[EvalCase]]
) -> dict[str, set[str]]:
    benchmark_id = next(
        case.case_id for case in probes["stressors"] if case.case_id == "stress_long"
    )
    phases = {
        "corpus": {case.case_id for case in corpus_cases},
        **{
            f"compatibility/{group}": {case.case_id for case in cases}
            for group, cases in probes.items()
        },
    }
    phases.update(
        {f"timing/post_warm/{index}": {benchmark_id} for index in range(1, TIMING_RUNS + 1)}
    )
    return phases


def validate_candidate_diagnostics(
    engine: str,
    diagnostics: dict,
    phase_cases: Mapping[str, set[str]] | None = None,
) -> None:
    if diagnostics.get("contract") != candidate_diagnostic_contract():
        raise RuntimeError(f"{engine}: candidate diagnostic contract drifted")
    events = diagnostics.get("events")
    if not isinstance(events, list):
        raise RuntimeError(f"{engine}: candidate diagnostics are malformed")
    for event in events:
        if not isinstance(event, dict) or set(event) != {"case_id", "code", "message", "phase"}:
            raise RuntimeError(f"{engine}: candidate diagnostic event is malformed")
        if not all(isinstance(event[key], str) and event[key] for key in event):
            raise RuntimeError(f"{engine}: candidate diagnostic event is empty")
        if event["code"] != _native_diagnostic_code(event["message"]):
            raise RuntimeError(f"{engine}: candidate diagnostic code drifted")
        if phase_cases is not None and event["case_id"] not in phase_cases.get(
            event["phase"], set()
        ):
            raise RuntimeError(f"{engine}: candidate diagnostic phase/case drifted")
    expected = summarize_candidate_diagnostics(events)
    if diagnostics != expected:
        raise RuntimeError(f"{engine}: candidate diagnostic summary drifted")


def _artifact(path: Path) -> dict:
    return {
        "bytes": path.stat().st_size,
        "path": _relative(path),
        "sha256": file_sha256(path),
    }


def _replay_control_manifest() -> dict:
    """Decode-scoped stand-in for the whole replay-goldens file.

    The goldens are an engine x clip matrix, but only CONTROL_ENGINES rows reach a
    compatibility case, so whole-file bytes would refuse a rebuild for every
    accelerator-bound row the matrix gains while proving nothing extra (L-030).
    """
    goldens = _load_json(REPLAY_GOLDENS)
    payload = _json_bytes({engine: goldens[engine] for engine in CONTROL_ENGINES})
    return {
        "bytes": len(payload),
        "path": f"{_relative(REPLAY_GOLDENS)}#controls",
        "sha256": _sha256_bytes(payload),
    }


def model_fingerprint(engine: str) -> dict:
    if engine in CONTROL_ENGINES:
        model_dir = ENGINE_DIRS[engine]
        artifacts = [_artifact(model_dir / name) for name in CONTROL_MODEL_FILES[engine]]
        provenance = None
    else:
        spec = CANDIDATE_SPECS[engine]
        model_dir = validate_candidate_model(spec)
        artifacts = [_artifact(model_dir / artifact.path) for artifact in spec.artifacts]
        provenance = candidate_provenance(spec)
    output = {
        "artifacts": artifacts,
        "bytes": sum(row["bytes"] for row in artifacts),
        "directory": _relative(model_dir),
    }
    if provenance is not None:
        output["provenance"] = provenance
    return output


def _current_rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    raise RuntimeError("cannot read current RSS from /proc/self/status")


def _peak_rss_mib() -> float:
    # Linux ru_maxrss is KiB. The evaluator records its Linux runtime fingerprint.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or "unknown"


def runtime_fingerprint() -> dict:
    packages = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "sherpa-onnx", "sherpa-onnx-core")
    }
    return {
        "host": {
            "cpu": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "packages": packages,
        # Recorded, never compared. Neither the OS kernel nor a whole-lock hash
        # participates in sherpa CPU decode, and gating on them refused rebuilds after
        # a host kernel update and after M11.1 added OpenVINO to uv.lock -- the same
        # whole-artifact defect `asr_contract_sha256` retires above. `packages` already
        # pins every decode-relevant distribution, more precisely than the lock does.
        "provenance": {
            "kernel": platform.release(),
            "uv_lock_sha256": file_sha256(ROOT / "uv.lock"),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


# Keys in `runtime_fingerprint()` that record provenance instead of gating a rebuild.
# The two bare names are the pre-M11.3 spelling, kept so an older baseline reduces to
# the same comparable set instead of needing its own migration clause.
_RUNTIME_PROVENANCE_KEYS = frozenset({"kernel", "provenance", "uv_lock_sha256"})


def _comparable_runtime(value: dict) -> dict:
    return {key: item for key, item in value.items() if key not in _RUNTIME_PROVENANCE_KEYS}


# Names `replay.py` and this evaluator import from live_stt: the whole entry surface
# of the sherpa offline row path. The contract is the AST closure over these, so a new
# call enters the hash automatically.
ASR_CONTRACT_SEEDS = (
    "DECODE_CHUNK_OVERLAP_S",
    "DECODE_CHUNK_S",
    "DECODE_SPLIT_RMS_WINDOW_S",
    "DECODE_SPLIT_SEARCH_S",
    "DECODE_SPLIT_TRIGGER_S",
    "ENGINE_DIRS",
    "NUM_THREADS",
    "RING_SECONDS",
    "SAMPLE_RATE",
    "State",
    "VAD_MAX_SPEECH_S",
    "VAD_MIN_SILENCE_S",
    "VAD_MIN_SPEECH_S",
    "VAD_MODEL",
    "VAD_PRE_PAD_S",
    "check_models",
    "load_recognizer",
    "make_vad",
    "resample",
    "worker",
)

# Reached from the seeds but excluded, each with the reason it cannot change a decoded
# row. Inclusion is derived; exclusion is declared. `tests/test_eval_models.py` asserts
# closure == contract | cuts, so a new call forces a decision here instead of silently
# entering or escaping the hash. Whole-file hashing of live_stt.py was the M10-carried
# defect: every unrelated production edit refused an aggregate-only rebuild (M11.3).
ASR_CONTRACT_CUTS = {
    "ASR_DEVICE": "whisper-only; reached through load_recognizer's WHISPER_ENGINES branch",
    "ASR_HOTWORDS_DEVICES": "whisper-only; WhisperEngine capability gate",
    "ENGINE_DIRS": "path map; the model bytes it points at are hashed by model_fingerprint",
    "MODELS_DIR": "path root; see ENGINE_DIRS",
    "OPENVINO_CACHE_DIR": "whisper-only; WhisperEngine compile cache",
    "State": "counter bag (queue depth, drops, stop event) — measurement, never decoder input",
    "VAC_CHUNK_S": "VAC-only cadence constant",
    "VAC_TRIM_S": "VAC-only trim constant",
    "VAD_MODEL": "path constant; the model bytes are hashed by pipeline['vad_model']",
    "WHISPER_ENGINES": "whisper-only; selects the WhisperEngine branch",
    "WhisperEngine": "whisper constructor; load_recognizer returns it only for WHISPER_ENGINES",
    "_LINE_CLEAR": "terminal presentation",
    "_STDOUT_TTY": "terminal presentation",
    "_vac_segments": "VAC branch; worker enters it only for a recogniser exposing"
    " decode_segments, which sherpa never does",
    "check_models": "readiness preflight; raises or returns None, never reaches the decoder",
    "emit_line": "presentation; evaluator rows arrive through replay.py's on_segment hook",
    "logger": "logging handle",
}


@dataclass(frozen=True)
class ModuleSurface:
    """One module's top-level definitions plus the `from X import Y` edges out of it."""

    symbols: dict[str, str]
    imports: dict[str, tuple[str, str]]


def module_surface(path: Path) -> ModuleSurface:
    """Top-level definitions by name → exact source text, and the module's import map."""
    source = path.read_text(encoding="utf-8")
    symbols: dict[str, str] = {}
    imports: dict[str, tuple[str, str]] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom):
            # Fail closed on the two forms a name-keyed import map cannot represent: a
            # star import would drop the whole imported surface out of the closure
            # silently, which is exactly the escape a derived contract exists to stop.
            if node.level or node.module is None:
                raise RuntimeError(f"{path.name}: relative import is not a contract edge")
            if any(alias.name == "*" for alias in node.names):
                raise RuntimeError(f"{path.name}: star import hides the contract surface")
            imports.update(
                {alias.asname or alias.name: (node.module, alias.name) for alias in node.names}
            )
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        segment = ast.get_source_segment(source, node)
        if segment is not None:
            symbols.update(dict.fromkeys(names, segment))
    return ModuleSurface(symbols, imports)


def contract_closure(
    surfaces: Mapping[str, ModuleSurface],
    roots: Iterable[tuple[str, str]],
    *,
    cuts: Mapping[str, str] | frozenset[str] = frozenset(),
    require: bool = False,
) -> set[tuple[str, str]]:
    """`(module, name)` reachable from `roots`, stopping at `cuts`.

    Cross-module edges follow `from X import Y` only while X is in `surfaces`, which is
    what keeps the sherpa contract single-module while the VAC contract spans
    live_stt + streaming. `require` turns a missing root/dependency into an error rather
    than a silent drop; the sherpa contract predates that check and keeps the old
    skip so its hash cannot move (M11.3b P1.5).
    """
    seen: set[tuple[str, str]] = set()
    stack = [key for key in roots if require or key[1] in surfaces[key[0]].symbols]
    while stack:
        key = stack.pop()
        module, name = key
        if key in seen or name in cuts:
            continue
        surface = surfaces[module]
        if name not in surface.symbols:
            raise RuntimeError(f"contract closure: {module}.{name} is not a top-level definition")
        seen.add(key)
        for node in ast.walk(ast.parse(surface.symbols[name])):
            if not isinstance(node, ast.Name):
                continue
            target = (
                (module, node.id) if node.id in surface.symbols else surface.imports.get(node.id)
            )
            if target is None or target[0] not in surfaces:
                continue
            if target[1] in surfaces[target[0]].symbols and target not in seen:
                stack.append(target)
    return seen


def _live_stt_symbols() -> dict[str, str]:
    """Top-level live_stt definitions by name, mapped to their exact source text."""
    return module_surface(ROOT / "live_stt.py").symbols


def asr_contract_closure(cut: bool = True) -> set[str]:
    """Names reachable from the seeds, stopping at cuts when `cut` is set."""
    surfaces = {"live_stt": module_surface(ROOT / "live_stt.py")}
    closure = contract_closure(
        surfaces,
        [("live_stt", name) for name in ASR_CONTRACT_SEEDS],
        cuts=ASR_CONTRACT_CUTS if cut else frozenset(),
    )
    return {name for _, name in closure}


def _structural_dump(source: str) -> str:
    """Executable structure of `source`, with docstrings dropped.

    `ast.dump` omits positions, so comments, blank lines, docstrings and `ruff format`
    rewrapping cannot move the hash while every executed expression still can. A
    whole-file byte hash conflated the two: M11.1's rewrap of one slice expression in
    `_split_decode_segment` was a fingerprint change under bytes and is a no-op here.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                del body[0]
    return ast.dump(tree)


def asr_contract_source() -> bytes:
    """Executable structure of everything that can change a decoded sherpa row.

    The decode-reaching closure of live_stt, plus the scorer and the replay driver
    whole -- each of those two files is dependency-scoped by construction.
    """
    symbols = _live_stt_symbols()
    parts = [
        f"live_stt.{name}\n{_structural_dump(symbols[name])}"
        for name in sorted(asr_contract_closure())
    ]
    parts += [
        f"{name}\n{_structural_dump((ROOT / name).read_text(encoding='utf-8'))}"
        for name in ("cer.py", "replay.py")
    ]
    return "\n\n".join(parts).encode()


def pipeline_fingerprint() -> dict:
    values = {
        "decode_chunk_overlap_s": DECODE_CHUNK_OVERLAP_S,
        "decode_chunk_s": DECODE_CHUNK_S,
        "decode_split_rms_window_s": DECODE_SPLIT_RMS_WINDOW_S,
        "decode_split_search_s": DECODE_SPLIT_SEARCH_S,
        "decode_split_trigger_s": DECODE_SPLIT_TRIGGER_S,
        "num_threads": NUM_THREADS,
        "ring_seconds": RING_SECONDS,
        "sample_rate_hz": SAMPLE_RATE,
        "vad_max_speech_s": VAD_MAX_SPEECH_S,
        "vad_min_silence_s": VAD_MIN_SILENCE_S,
        "vad_min_speech_s": VAD_MIN_SPEECH_S,
        "vad_pre_pad_s": VAD_PRE_PAD_S,
    }
    contract_source = "\n\n".join(
        inspect.getsource(obj)
        for obj in (
            EvalCase,
            Transcript,
            DecodeObservation,
            SherpaOfflineAdapter,
            load_evaluator_recognizer,
            recognizer_config,
            _json_bytes,
            _duration_bucket,
            load_corpus_cases,
            _score,
        )
    ).encode()
    probes, _ = compatibility_cases()
    compatibility_wavs = [
        {"case_id": case.case_id, "group": group, **_artifact(case.wav)}
        for group, cases in probes.items()
        for case in cases
    ]
    return {
        "compatibility_inputs": {
            "manifests": [
                _artifact(CER_BASELINE),
                _artifact(LONG_FORM),
                _replay_control_manifest(),
                _artifact(STRESSORS),
            ],
            "wavs": compatibility_wavs,
        },
        "asr_contract_sha256": _sha256_bytes(asr_contract_source()),
        "evaluator_contract_sha256": _sha256_bytes(contract_source),
        # Only acquisition stays a whole-file byte hash; it has its own reaggregate
        # clause. The scorer, the driver and live_stt's decode closure moved into
        # `asr_contract_sha256` above, because whole-file bytes refused a rebuild after
        # every unrelated edit -- the M10-carried defect M11.3 repairs.
        "implementation": [_artifact(ROOT / "tests/fetch_eval_models.py")],
        "values": values,
        "vad_model": _artifact(VAD_MODEL),
    }


def evaluation_inputs(corpus_manifest: dict) -> dict:
    """Fingerprint every non-model input that can change deterministic rows."""
    return {
        "corpus_index_sha256": corpus_manifest["cache"]["index_sha256"],
        "pipeline": pipeline_fingerprint(),
        "runtime": runtime_fingerprint(),
    }


def _metric(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "audio_s": 0.0,
            "empty_hypotheses": 0,
            "macro": {"d_rate": None, "i_rate": None, "s_rate": None, "cer": None},
            "micro": {"D": 0, "I": 0, "N": 0, "S": 0, "cer": None},
            "rows": 0,
        }
    n = sum(row["N"] for row in rows)
    substitutions = sum(row["S"] for row in rows)
    deletions = sum(row["D"] for row in rows)
    insertions = sum(row["I"] for row in rows)
    count = len(rows)

    def macro(key: str) -> float:
        return round(float(sum(Fraction(row[key], row["N"]) for row in rows) / count), 8)

    return {
        "audio_s": round(sum(row["duration_samples"] for row in rows) / SAMPLE_RATE, 6),
        "empty_hypotheses": sum(not normalize(row["hyp"]) for row in rows),
        "macro": {
            "d_rate": macro("D"),
            "i_rate": macro("I"),
            "s_rate": macro("S"),
            "cer": round(
                float(
                    sum(Fraction(row["S"] + row["D"] + row["I"], row["N"]) for row in rows) / count
                ),
                8,
            ),
        },
        "micro": {
            "D": deletions,
            "I": insertions,
            "N": n,
            "S": substitutions,
            "cer": round((substitutions + deletions + insertions) / n, 8),
        },
        "rows": count,
    }


def aggregate_rows(rows: Sequence[dict]) -> dict:
    by_source = {
        source: _metric([row for row in rows if row["source"] == source])
        for source in sorted({row["source"] for row in rows})
    }
    fleurs = [row for row in rows if row["source"] == "fleurs"]
    gender = {
        value: _metric([row for row in fleurs if row["gender"] == value])
        for value in sorted({row["gender"] for row in fleurs})
        if value is not None
    }
    duration = {
        bucket: _metric([row for row in fleurs if row["duration_bucket"] == bucket])
        for bucket in ("0-5", "5-10", "10-20", "20+")
    }
    completion = {
        "accepted_all_audio_rows": sum(
            type(row["accepted_samples"]) is int
            and row["accepted_samples"] == row["duration_samples"]
            for row in rows
        ),
        "complete_rows": sum(row["complete"] is True for row in rows),
        "empty_case_ids": [row["case_id"] for row in rows if not normalize(row["hyp"])],
        "empty_hypotheses": sum(not normalize(row["hyp"]) for row in rows),
        "eof_once_rows": sum(
            type(row["eof_count"]) is int and row["eof_count"] == 1 for row in rows
        ),
        "rows": len(rows),
    }
    completion["complete"] = all(
        completion[key] == len(rows)
        for key in ("accepted_all_audio_rows", "complete_rows", "eof_once_rows")
    )
    worst_tail = {}
    for source in by_source:
        selected = [row for row in rows if row["source"] == source]
        selected.sort(
            key=lambda row: (
                -Fraction(row["S"] + row["D"] + row["I"], row["N"]),
                row["case_id"],
            )
        )
        worst_tail[source] = [
            {
                key: row[key]
                for key in ("D", "I", "N", "S", "case_id", "cer", "gender", "hyp", "ref")
            }
            for row in selected[:TAIL_EXEMPLARS]
        ]
    return {
        "by_source": by_source,
        "completion": completion,
        "fleurs_duration": duration,
        "fleurs_gender": gender,
        "worst_tail": worst_tail,
    }


def select_comparator(controls: Mapping[str, dict]) -> dict:
    ranked = []
    for engine, control in controls.items():
        micro = control["aggregates"]["by_source"]["common_voice_8"]["micro"]
        ranked.append((Fraction(micro["S"] + micro["D"] + micro["I"], micro["N"]), engine))
    score, engine = min(ranked)
    return {
        "engine": engine,
        "micro_cer": round(float(score), 8),
        "rule": DISPLACEMENT_GATES["comparator"],
    }


def _score_fraction(row: Mapping[str, int]) -> Fraction:
    return Fraction(row["S"] + row["D"] + row["I"], row["N"])


def _source_micro_fraction(control: dict, source: str) -> Fraction:
    return _score_fraction(control["aggregates"]["by_source"][source]["micro"])


def content_verdict(candidate: dict, comparator: dict) -> dict:
    """Mechanically apply content gates; resource evidence is intentionally absent."""
    gates = DISPLACEMENT_GATES["content"]
    candidate_cv = _source_micro_fraction(candidate, "common_voice_8")
    comparator_cv = _source_micro_fraction(comparator, "common_voice_8")
    relative_gain = (comparator_cv - candidate_cv) / comparator_cv if comparator_cv else Fraction(0)
    candidate_fleurs = _source_micro_fraction(candidate, "fleurs")
    comparator_fleurs = _source_micro_fraction(comparator, "fleurs")
    candidate_long = _score_fraction(candidate["compatibility"]["long_form"][LONG_FORM_ID])
    comparator_long = _score_fraction(comparator["compatibility"]["long_form"][LONG_FORM_ID])
    candidate_stress = candidate["compatibility"]["stressors"]
    required_stressors = set(comparator["compatibility"]["stressors"])
    relative_min = Fraction(str(gates["common_voice_relative_cer_improvement_min"]))
    fleurs_regression_max = Fraction(str(gates["fleurs_micro_cer_regression_max_abs"]))
    long_regression_max = Fraction(str(gates["long_form_cer_regression_max_abs"]))
    stressor_cer_max = Fraction(str(gates["stressor_cer_max"]))
    stressor_deletion_max = Fraction(str(gates["stressor_excess_deletion_rate_max"]))

    checks = {
        "common_voice_relative_cer": {
            "observed_improvement": round(float(relative_gain), 8),
            "pass": relative_gain >= relative_min,
        },
        "complete_run": {
            "observed": candidate["aggregates"]["completion"]["complete"],
            "pass": candidate["aggregates"]["completion"]["complete"] is True,
        },
        "decoder_diagnostics": {
            "case_ids": candidate["diagnostics"]["content_event_case_ids"],
            "observed_events": len(candidate["diagnostics"]["content_events"]),
            "pass": not candidate["diagnostics"]["content_events"],
        },
        "nonempty_run": {
            "observed": (
                candidate["aggregates"]["completion"]["rows"]
                - candidate["aggregates"]["completion"]["empty_hypotheses"]
            ),
            "pass": (
                candidate["aggregates"]["completion"]["rows"]
                > candidate["aggregates"]["completion"]["empty_hypotheses"]
            ),
        },
        "fleurs_micro_cer_regression": {
            "observed_abs": round(float(candidate_fleurs - comparator_fleurs), 8),
            "pass": candidate_fleurs - comparator_fleurs <= fleurs_regression_max,
        },
        "long_form_cer_regression": {
            "observed_abs": round(float(candidate_long - comparator_long), 8),
            "pass": candidate_long - comparator_long <= long_regression_max,
        },
        "stressors_complete": {
            "observed": sorted(candidate_stress),
            "pass": set(candidate_stress) == required_stressors,
        },
        "stressors_cer": {
            "observed_max": round(
                float(
                    max(
                        (_score_fraction(row) for row in candidate_stress.values()),
                        default=Fraction(1),
                    )
                ),
                8,
            ),
            "pass": bool(candidate_stress)
            and all(_score_fraction(row) <= stressor_cer_max for row in candidate_stress.values()),
        },
        "stressors_excess_deletion": {
            "observed_max": round(
                float(
                    max(
                        (Fraction(row["excess_D"], row["N"]) for row in candidate_stress.values()),
                        default=Fraction(1),
                    )
                ),
                8,
            ),
            "pass": bool(candidate_stress)
            and all(
                Fraction(row["excess_D"], row["N"]) <= stressor_deletion_max
                for row in candidate_stress.values()
            ),
        },
    }
    return {"checks": checks, "qualified": all(check["pass"] for check in checks.values())}


def resource_verdict(paths: Mapping[str, dict]) -> dict:
    """Apply only execution-path gates; accuracy/content is intentionally absent."""
    maximum = DISPLACEMENT_GATES["resource"]["post_warm_rtf_max"]
    checks = {
        name: {
            "actual_execution_device": row["actual_execution_device"],
            "post_warm_rtf": row["post_warm_rtf"],
            "production_eligible": row["production_eligible"],
            "qualified": row["production_eligible"] and row["post_warm_rtf"] <= maximum,
        }
        for name, row in sorted(paths.items())
    }
    qualified = sum(row["qualified"] for row in checks.values())
    return {
        "paths": checks,
        "qualified": qualified >= DISPLACEMENT_GATES["resource"]["production_eligible_paths_min"],
    }


def _detail_rows(path: Path, cases: Sequence[EvalCase], expected_sha256: str) -> list[dict]:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise RuntimeError(f"detail fingerprint mismatch: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse detail rows: {path}") from exc
    expected_ids = [case.case_id for case in cases]
    if [row.get("case_id") for row in rows] != expected_ids:
        raise RuntimeError("detail rows are incomplete, duplicated, or reordered")
    for row, case in zip(rows, cases, strict=True):
        identity = {
            "case_id": case.case_id,
            "duration_bucket": case.duration_bucket,
            "duration_samples": case.duration_samples,
            "gender": case.gender,
            "ref": case.reference,
            "source": case.source,
        }
        if any(row.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"detail identity drift: {case.case_id}")
        segments = row.get("segments")
        if not isinstance(segments, list) or any(
            not isinstance(segment, dict) or not isinstance(segment.get("text"), str)
            for segment in segments
        ):
            raise RuntimeError(f"invalid detail segments: {case.case_id}")
        if row.get("hyp") != "".join(segment["text"] for segment in segments):
            raise RuntimeError(f"detail hypothesis/segments drift: {case.case_id}")
        ref = normalize(case.reference)
        substitutions, deletions, insertions = align(ref, normalize(row["hyp"]))
        counts = (len(ref), substitutions, deletions, insertions)
        if tuple(row.get(key) for key in ("N", "S", "D", "I")) != counts:
            raise RuntimeError(f"detail score drift: {case.case_id}")
        if row.get("cer") != (substitutions + deletions + insertions) / len(ref):
            raise RuntimeError(f"detail CER drift: {case.case_id}")
    return rows


def _corpus_fingerprint(manifest: dict) -> dict:
    return {
        "index_sha256": manifest["cache"]["index_sha256"],
        "manifest": _relative(SHORT_CORPUS),
        "manifest_sha256": file_sha256(SHORT_CORPUS),
        "rows": manifest["cache"]["rows"],
        "sources": {
            source: {
                "revision": row["revision"],
                "rows": row["statistics"]["rows"],
                "source_identity": row["source_identity"],
            }
            for source, row in sorted(manifest["sources"].items())
        },
    }


def validate_evidence_model_set(
    summaries: Mapping[str, dict],
    detail_paths: Mapping[str, Path],
    model_ids: Sequence[str],
    *,
    label: str,
) -> tuple[str, ...]:
    """Validate an exact ordered model set for any isolated evaluator."""
    expected = tuple(model_ids)
    if len(set(expected)) != len(expected):
        raise RuntimeError(f"{label}: duplicate model IDs")
    if set(summaries) != set(expected) or set(detail_paths) != set(expected):
        raise RuntimeError(f"{label}: required model set is {list(expected)}")
    return expected


def build_manifest(
    corpus_manifest: dict,
    corpus_cases: Sequence[EvalCase],
    summaries: Mapping[str, dict],
    detail_paths: Mapping[str, Path],
) -> dict:
    validate_evidence_model_set(
        summaries,
        detail_paths,
        OFFLINE_MODELS,
        label="offline evaluator",
    )
    expected_inputs = evaluation_inputs(corpus_manifest)
    probes, compatibility_expected = compatibility_cases()
    diagnostic_phase_cases = candidate_diagnostic_phase_cases(corpus_cases, probes)
    controls = {}
    candidates = {}
    control_measurements = {}
    candidate_measurements = {}
    for engine in OFFLINE_MODELS:
        summary = summaries[engine]
        if summary.get("engine") != engine or summary.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"invalid child summary: {engine}")
        if summary.get("inputs") != expected_inputs:
            raise RuntimeError(f"{engine}: child input fingerprint drift")
        if summary["adapter"]["model"] != model_fingerprint(engine):
            raise RuntimeError(f"{engine}: child model fingerprint drift")
        if engine in CONTROL_ENGINES:
            validate_compatibility_snapshot(
                engine, summary["compatibility"], compatibility_expected
            )
        else:
            validate_candidate_compatibility_snapshot(
                engine, summary["compatibility"], compatibility_expected
            )
            validate_candidate_diagnostics(engine, summary["diagnostics"], diagnostic_phase_cases)
        detail = summary["details"]
        if detail["rows"] != len(corpus_cases):
            raise RuntimeError(f"{engine}: incomplete child row count")
        rows = _detail_rows(detail_paths[engine], corpus_cases, detail["sha256"])
        aggregates = aggregate_rows(rows)
        if not aggregates["completion"]["complete"]:
            raise RuntimeError(f"{engine}: incomplete transcript contract")
        result = {
            "adapter": summary["adapter"],
            "aggregates": aggregates,
            "compatibility": summary["compatibility"],
            "details": {
                "rows": detail["rows"],
                "sha256": detail["sha256"],
            },
        }
        if engine in OFFLINE_CANDIDATES:
            result["diagnostics"] = summary["diagnostics"]
        destination = controls if engine in CONTROL_ENGINES else candidates
        destination[engine] = result
        measurement = summary["measurements"]
        path = {
            "actual_execution_device": summary["adapter"]["actual_execution_device"],
            "post_warm_rtf": measurement["post_warm"]["median_decode_rtf"],
            "production_eligible": True,
        }
        measurement_result = {
            **measurement,
            "resource_verdict": resource_verdict({"sherpa_cpu": path}),
        }
        measurement_destination = (
            control_measurements if engine in CONTROL_ENGINES else candidate_measurements
        )
        measurement_destination[engine] = measurement_result

    comparator = select_comparator(controls)
    comparator_result = controls[comparator["engine"]]
    for candidate in candidates.values():
        candidate["content_verdict"] = content_verdict(candidate, comparator_result)

    deterministic = {
        "candidates": candidates,
        "comparator": comparator,
        "controls": controls,
        "corpus": _corpus_fingerprint(corpus_manifest),
        "displacement_gates": DISPLACEMENT_GATES,
        "metric_contract": METRIC_CONTRACT,
        "pipeline": pipeline_fingerprint(),
        "runtime": runtime_fingerprint(),
    }
    return {
        "deterministic": deterministic,
        "measurements": {
            "candidates": candidate_measurements,
            "controls": control_measurements,
            "excluded_from_deterministic_equality": True,
            "note": (
                "Elapsed time and RSS vary; scored content and fingerprints live "
                "under deterministic."
            ),
        },
        "schema_version": SCHEMA_VERSION,
    }


def install_content_addressed_details(
    summaries: Mapping[str, dict],
    staged_details: Mapping[str, Path],
    model_ids: Sequence[str],
    *,
    details_dir: Path,
) -> dict[str, Path]:
    """Validate then install immutable detail files for any evaluator model set."""
    models = validate_evidence_model_set(
        summaries,
        staged_details,
        model_ids,
        label="content-addressed evidence",
    )
    destinations: dict[str, Path] = {}
    for model_id in models:
        expected = summaries[model_id]["details"]["sha256"]
        source = staged_details[model_id]
        if file_sha256(source) != expected:
            raise RuntimeError(f"{model_id}: staged detail hash changed before install")
        destinations[model_id] = details_dir / f"{model_id}-{expected[:16]}.jsonl"

    # All validation precedes writes. Interrupted installs can only leave an
    # unreferenced immutable detail file; the tracked manifest is written last.
    for model_id, destination in destinations.items():
        expected = summaries[model_id]["details"]["sha256"]
        if destination.exists():
            if file_sha256(destination) != expected:
                raise RuntimeError(f"content-addressed detail collision: {destination}")
        else:
            write_atomic(destination, [staged_details[model_id].read_bytes()])
    return destinations


def install_evidence(
    manifest: dict,
    summaries: Mapping[str, dict],
    staged_details: Mapping[str, Path],
    *,
    baseline: Path = BASELINE,
    details_dir: Path = DETAILS_DIR,
) -> None:
    """Install validated immutable details first, then atomically replace the manifest."""
    destinations = install_content_addressed_details(
        summaries,
        staged_details,
        OFFLINE_MODELS,
        details_dir=details_dir,
    )
    for engine, destination in destinations.items():
        group = "controls" if engine in CONTROL_ENGINES else "candidates"
        manifest["deterministic"][group][engine]["details"]["path"] = _relative(destination)
    write_atomic(baseline, [_json_bytes(manifest)])


def _benchmark(
    adapter: ASRAdapter,
    case: EvalCase,
    expected_hyp: str,
    diagnostic_events: list[dict] | None = None,
) -> dict:
    runs = []
    for index in range(1, TIMING_RUNS + 1):
        observation = (
            adapter.decode(case)
            if diagnostic_events is None
            else _candidate_decode(adapter, case, f"timing/post_warm/{index}", diagnostic_events)
        )
        if observation.content.hypothesis != expected_hyp or not observation.content.complete:
            raise RuntimeError(f"{adapter.adapter_id}: timing replay content drift at run {index}")
        audio_s = case.duration_samples / SAMPLE_RATE
        runs.append(
            {
                "decode_rtf": round(observation.decode_seconds / audio_s, 6),
                "decode_s": round(observation.decode_seconds, 6),
                "wall_rtf": round(observation.wall_seconds / audio_s, 6),
                "wall_s": round(observation.wall_seconds, 6),
            }
        )
    return {
        "audio_s": round(case.duration_samples / SAMPLE_RATE, 6),
        "case_id": case.case_id,
        "median_decode_rtf": round(median(row["decode_rtf"] for row in runs), 6),
        "median_wall_rtf": round(median(row["wall_rtf"] for row in runs), 6),
        "runs": runs,
        "warmup": "compatibility/stress_long decode before measured repetitions",
    }


def validate_model_available(engine: str) -> None:
    if engine in CONTROL_ENGINES:
        err = check_models(engine)
        if err:
            raise RuntimeError(f"{engine}: {err.splitlines()[0]}")
        return
    try:
        validate_candidate_model(CANDIDATE_SPECS[engine])
    except RuntimeError as exc:
        raise RuntimeError(
            f"{engine}: {exc}; run tests/fetch_eval_models.py to acquire exact artifacts"
        ) from exc


def run_worker(engine: str, details_path: Path, summary_path: Path) -> None:
    validate_model_available(engine)
    corpus_manifest, corpus_cases = load_corpus_cases(verify_pcm=False)
    probes, expected = compatibility_cases()

    rss_before = _current_rss_mib()
    started = time.perf_counter()
    adapter = SherpaOfflineAdapter(engine)
    cold_load_s = time.perf_counter() - started
    rss_loaded = _current_rss_mib()
    peak_loaded = _peak_rss_mib()
    identity = adapter.identity()
    diagnostic_events: list[dict] | None = [] if engine in OFFLINE_CANDIDATES else None
    print(
        f"{engine}: loaded {identity['model']['bytes'] / 1_000_000:.1f} MB model "
        f"in {cold_load_s:.3f}s; RSS={rss_loaded:.1f} MiB",
        flush=True,
    )

    compatibility: dict[str, dict] = {group: {} for group in probes}
    for group, cases in probes.items():
        for case in cases:
            observation = (
                adapter.decode(case)
                if diagnostic_events is None
                else _candidate_decode(adapter, case, f"compatibility/{group}", diagnostic_events)
            )
            row = _score(case, observation.content)
            if engine in CONTROL_ENGINES:
                result = _control_compatibility_row(engine, group, case, row, expected)
            else:
                result = _candidate_compatibility_row(group, case, row, compatibility, expected)
            compatibility[group][case.case_id] = result
        status = "exact" if engine in CONTROL_ENGINES else "scored"
        print(f"{engine}: compatibility/{group} {status} ({len(cases)} cases)", flush=True)

    benchmark_case = next(case for case in probes["stressors"] if case.case_id == "stress_long")
    post_warm = _benchmark(
        adapter,
        benchmark_case,
        compatibility["stressors"]["stress_long"]["hyp"],
        diagnostic_events,
    )
    print(
        f"{engine}: post-warm median decode RTF={post_warm['median_decode_rtf']:.3f}",
        flush=True,
    )

    digest = hashlib.sha256()
    rows_written = 0

    def detail_chunks() -> Iterable[bytes]:
        nonlocal rows_written
        for index, case in enumerate(corpus_cases, 1):
            observation = (
                adapter.decode(case)
                if diagnostic_events is None
                else _candidate_decode(adapter, case, "corpus", diagnostic_events)
            )
            row = _score(case, observation.content)
            data = _json_bytes(row, compact=True)
            digest.update(data)
            rows_written += 1
            if index == 1 or index % PROGRESS_ROWS == 0 or index == len(corpus_cases):
                print(f"{engine}: corpus {index}/{len(corpus_cases)}", flush=True)
            yield data

    write_atomic(details_path, detail_chunks())
    detail_sha256 = digest.hexdigest()
    if rows_written != len(corpus_cases) or file_sha256(details_path) != detail_sha256:
        raise RuntimeError(f"{engine}: detail write incomplete")

    measurements = {
        "cold_recognizer_load_s": round(cold_load_s, 6),
        "device_memory_mib": None,
        "device_memory_note": "CPU path has no separate accelerator-memory allocation.",
        "isolated_process": True,
        "post_warm": post_warm,
        "rss_mib": {
            "current_before_recognizer_load": round(rss_before, 3),
            "current_model_load_delta": round(rss_loaded - rss_before, 3),
            "current_model_loaded": round(rss_loaded, 3),
            "ru_maxrss_after_load": round(peak_loaded, 3),
            "ru_maxrss_process": round(_peak_rss_mib(), 3),
        },
    }
    summary = {
        "adapter": identity,
        "compatibility": compatibility,
        "details": {"rows": rows_written, "sha256": detail_sha256},
        "engine": engine,
        "inputs": evaluation_inputs(corpus_manifest),
        "measurements": measurements,
        "schema_version": SCHEMA_VERSION,
    }
    if diagnostic_events is not None:
        summary["diagnostics"] = summarize_candidate_diagnostics(diagnostic_events)
    write_atomic(summary_path, [_json_bytes(summary)])
    print(
        f"{engine}: complete; detail={detail_sha256}; peak RSS="
        f"{measurements['rss_mib']['ru_maxrss_process']:.1f} MiB",
        flush=True,
    )


def _reusable_child(
    engine: str,
    summary_path: Path,
    detail_path: Path,
    corpus_manifest: dict,
    corpus_cases: Sequence[EvalCase],
) -> dict | None:
    """Return a complete matching staged child, else remove it for a clean rerun."""
    if not summary_path.is_file() or not detail_path.is_file():
        summary_path.unlink(missing_ok=True)
        detail_path.unlink(missing_ok=True)
        return None
    try:
        summary = _load_json(summary_path)
        if (
            summary.get("schema_version") != SCHEMA_VERSION
            or summary.get("engine") != engine
            or summary.get("inputs") != evaluation_inputs(corpus_manifest)
            or summary["adapter"]["model"] != model_fingerprint(engine)
            or summary["details"]["rows"] != len(corpus_cases)
        ):
            raise RuntimeError("staged child fingerprint drift")
        if engine in OFFLINE_CANDIDATES:
            probes, _ = compatibility_cases()
            validate_candidate_diagnostics(
                engine,
                summary["diagnostics"],
                candidate_diagnostic_phase_cases(corpus_cases, probes),
            )
        _detail_rows(detail_path, corpus_cases, summary["details"]["sha256"])
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"{engine}: discard unusable staged child ({exc})", flush=True)
        summary_path.unlink(missing_ok=True)
        detail_path.unlink(missing_ok=True)
        return None
    return summary


def _child_measurement(measurement: dict) -> dict:
    """Strip parent verdicts and migrate pre-review RSS labels without new timing."""
    output = {key: value for key, value in measurement.items() if key != "resource_verdict"}
    rss = dict(output["rss_mib"])
    aliases = {
        "before_recognizer_load": "current_before_recognizer_load",
        "model_load_delta": "current_model_load_delta",
        "model_loaded": "current_model_loaded",
        "peak_after_load": "ru_maxrss_after_load",
        "peak_process": "ru_maxrss_process",
    }
    for old, new in aliases.items():
        if old in rss:
            rss[new] = rss.pop(old)
    output["rss_mib"] = rss
    output.setdefault("device_memory_mib", None)
    output.setdefault(
        "device_memory_note", "CPU path has no separate accelerator-memory allocation."
    )
    return output


def _pipeline_reaggregate_compatible(previous: dict, current: dict) -> bool:
    if previous == current:
        return True
    # One-time M10.3 hardening: old evidence lacked explicit compatibility-WAV
    # and evaluator-contract fields. Current row + legacy-snapshot validation
    # substantiates adding them without rerunning unchanged decode code.
    legacy_current = {
        key: value
        for key, value in current.items()
        if key not in {"compatibility_inputs", "evaluator_contract_sha256"}
    }
    if previous == legacy_current:
        return True

    # Acquisition/provenance code cannot alter cached decode rows once the exact
    # installed artifact identity is independently rechecked below. Permit only
    # that implementation entry to move; every worker/config/input hash stays exact.
    acquisition_path = "tests/fetch_eval_models.py"

    def without_acquisition(value: dict) -> dict:
        return {
            **value,
            "implementation": [
                row
                for row in value.get("implementation", [])
                if row.get("path") != acquisition_path
            ],
        }

    previous_acquisition = [
        row for row in previous.get("implementation", []) if row.get("path") == acquisition_path
    ]
    current_acquisition = [
        row for row in current.get("implementation", []) if row.get("path") == acquisition_path
    ]
    if len(previous_acquisition) == len(current_acquisition) == 1 and without_acquisition(
        previous
    ) == without_acquisition(current):
        return True

    if _asr_contract_migration_compatible(previous, current):
        return True
    return _goldens_manifest_migration_compatible(previous, current)


# One-time M11.3c migration, pinned to exactly one prior manifest row so it can never
# waive a second transition. The replay goldens became an engine x clip MATRIX carrying
# accelerator-bound rows this evaluator never scores, and whole-file bytes cannot tell a
# whisper row from a control-reference edit -- the same false-refusal class the ASR
# contract retired (L-030). Behavioural licence: `compatibility_cases()` reads
# `goldens[first_engine]` and cross-checks CONTROL_ENGINES alone, so no non-control row
# can reach a case, and the 24 control rows regenerate byte-identically at HEAD.
_M11_3C_RETIRED_GOLDENS_MANIFEST = {
    "bytes": 10673,
    "path": "tests/replay_goldens.json",
    "sha256": "35760b8303101f787e2071e2a8711e09ebd979c974d925832bf8717a64011fda",
}


def _goldens_manifest_migration_compatible(previous: dict, current: dict) -> bool:
    def manifests(value: dict) -> list[dict]:
        return value.get("compatibility_inputs", {}).get("manifests", [])

    scoped_path = f"{_relative(REPLAY_GOLDENS)}#controls"
    if _M11_3C_RETIRED_GOLDENS_MANIFEST not in manifests(previous):
        return False
    if any(row["path"] == scoped_path for row in manifests(previous)):
        return False
    if not any(row["path"] == scoped_path for row in manifests(current)):
        return False

    # Everything the migration does not retire must still match exactly, including
    # every other manifest, the compatibility WAVs, and both contract hashes.
    def survivors(value: dict) -> dict:
        kept = [
            row
            for row in manifests(value)
            if row != _M11_3C_RETIRED_GOLDENS_MANIFEST and row["path"] != scoped_path
        ]
        return {
            **value,
            "compatibility_inputs": {**value.get("compatibility_inputs", {}), "manifests": kept},
        }

    return survivors(previous) == survivors(current)


# One-time M11.3 migration, pinned to exactly one prior fingerprint so it can never
# waive a second transition. Evidence written before this commit byte-hashed whole files,
# which every unrelated production edit invalidated. Each retired value is adjudicated:
#
#   live_stt.py  the defect itself. Its 42-symbol decode closure is 26 unchanged / 8
#                changed / 2 new since e2a8d9c, and all 8 changes are additive
#                whisper/VAC/context branches a sherpa recogniser never enters, or a
#                `ruff format` rewrap. Behavioural proof the sherpa path is unchanged:
#                tests/test_replay.py = 29 passed at HEAD on real k2v2 + parakeet models
#                against committed byte-exact goldens.
#   replay.py    `git diff e2a8d9c -- replay.py` is module docstring, one function
#                docstring, and one argparse help string. No executable change.
#   evaluator    `load_evaluator_recognizer` gained M11.1's pyright type guard, which
#                raises for a WhisperEngine and returns the same object for both
#                controls. Inert on every scored row.
#
# Replay from a clean base:
#   git checkout e2a8d9c -- tests/model_baseline.json
#   uv run --no-sync python tests/eval_models.py --aggregate-only
_M11_3_RETIRED_IMPLEMENTATION = {
    "cer.py": "2dd7c47ad28beb614b5305dbe784495d0c997e840b0820d27d000a8dca6a2bd7",
    "live_stt.py": "7a25cece8cf102e2c4d685ccb7bc848562d43bc1044890f15a404320331e08be",
    "replay.py": "469fb8c3b52eb455d73dfb40da99a2a65f3278c6af515f253bd408ce7a67e445",
}
_M11_3_RETIRED_EVALUATOR_CONTRACT = (
    "ae2aab95888ae651d57a96bf052d2f28154dea12be233341b383ab076fbbc8e6"
)


def _asr_contract_migration_compatible(previous: dict, current: dict) -> bool:
    if "asr_contract_sha256" in previous or "asr_contract_sha256" not in current:
        return False
    retired = {
        row["path"]: row["sha256"]
        for row in previous.get("implementation", [])
        if row["path"] in _M11_3_RETIRED_IMPLEMENTATION
    }
    if retired != _M11_3_RETIRED_IMPLEMENTATION:
        return False
    if previous.get("evaluator_contract_sha256") != _M11_3_RETIRED_EVALUATOR_CONTRACT:
        return False

    # Everything the migration does not retire must still match exactly: the decode
    # constants, the compatibility WAVs and manifests, the VAD model, and acquisition.
    def survivors(value: dict) -> dict:
        return {
            **{
                key: item
                for key, item in value.items()
                if key not in {"asr_contract_sha256", "evaluator_contract_sha256"}
            },
            "implementation": [
                row
                for row in value.get("implementation", [])
                if row["path"] not in _M11_3_RETIRED_IMPLEMENTATION
            ],
        }

    return survivors(previous) == survivors(current)


def _model_decode_identity(model: dict) -> dict:
    return {key: model[key] for key in ("artifacts", "bytes", "directory")}


def reaggregate_parent() -> None:
    """Rebuild aggregate logic from exact details when decode inputs are unchanged."""
    previous = _load_json(BASELINE)
    if previous.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("aggregate-only refused: evaluator schema changed")
    deterministic = previous["deterministic"]
    corpus_manifest, corpus_cases = load_corpus_cases(verify_pcm=True)
    current = {
        "corpus": _corpus_fingerprint(corpus_manifest),
        "pipeline": pipeline_fingerprint(),
        "runtime": runtime_fingerprint(),
    }
    for key, value in current.items():
        if key == "pipeline":
            matches = _pipeline_reaggregate_compatible(deterministic.get(key, {}), value)
        elif key == "runtime":
            matches = _comparable_runtime(deterministic.get(key, {})) == _comparable_runtime(value)
        else:
            matches = deterministic.get(key) == value
        if not matches:
            raise RuntimeError(f"aggregate-only refused: {key} fingerprint changed")

    summaries = {}
    details = {}
    for engine in OFFLINE_MODELS:
        group = "controls" if engine in CONTROL_ENGINES else "candidates"
        result = deterministic[group][engine]
        current_model = model_fingerprint(engine)
        if _model_decode_identity(result["adapter"]["model"]) != _model_decode_identity(
            current_model
        ):
            raise RuntimeError(f"aggregate-only refused: {engine} model changed")
        measurement = previous["measurements"][group][engine]
        summaries[engine] = {
            "adapter": {**result["adapter"], "model": current_model},
            "compatibility": result["compatibility"],
            "details": {
                "rows": result["details"]["rows"],
                "sha256": result["details"]["sha256"],
            },
            "engine": engine,
            "inputs": evaluation_inputs(corpus_manifest),
            "measurements": _child_measurement(measurement),
            "schema_version": SCHEMA_VERSION,
        }
        if engine in OFFLINE_CANDIDATES:
            summaries[engine]["diagnostics"] = result["diagnostics"]
        details[engine] = ROOT / result["details"]["path"]

    rebuilt = build_manifest(corpus_manifest, corpus_cases, summaries, details)
    install_evidence(rebuilt, summaries, details)
    print(f"wrote {_relative(BASELINE)} from exact cached details", flush=True)


def run_isolated_workers(
    model_ids: Sequence[str],
    *,
    staging: Path,
    worker_script: Path,
    reusable_child: Callable[[str, Path, Path], dict | None],
) -> tuple[dict[str, dict], dict[str, Path]]:
    """Run or resume fingerprint-matched evaluator children in model order."""
    summaries: dict[str, dict] = {}
    details: dict[str, Path] = {}
    for model_id in model_ids:
        detail = staging / f"{model_id}.jsonl"
        summary = staging / f"{model_id}.summary.json"
        reusable = reusable_child(model_id, summary, detail)
        if reusable is None:
            command = [
                sys.executable,
                "-u",
                str(worker_script),
                "--worker",
                model_id,
                "--details",
                str(detail),
                "--summary",
                str(summary),
            ]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode:
                raise RuntimeError(f"isolated {model_id} evaluator exited {completed.returncode}")
            reusable = _load_json(summary)
        else:
            print(f"{model_id}: reuse complete fingerprint-matched staged child", flush=True)
        summaries[model_id] = reusable
        details[model_id] = detail
    return summaries, details


def run_parent() -> None:
    print("validating pinned corpus PCM + index before isolated runs", flush=True)
    corpus_manifest, corpus_cases = load_corpus_cases(verify_pcm=True)
    compatibility_cases()  # fail before staging when any legacy probe input is absent
    for engine in OFFLINE_MODELS:
        validate_model_available(engine)

    staging = CACHE / "model_eval-v2.staging"
    staging.mkdir(parents=True, exist_ok=True)
    installed = False
    try:

        def reusable_child(engine: str, summary: Path, detail: Path) -> dict | None:
            return _reusable_child(engine, summary, detail, corpus_manifest, corpus_cases)

        summaries, details = run_isolated_workers(
            OFFLINE_MODELS,
            staging=staging,
            worker_script=Path(__file__).resolve(),
            reusable_child=reusable_child,
        )
        manifest = build_manifest(corpus_manifest, corpus_cases, summaries, details)
        install_evidence(manifest, summaries, details)
        installed = True
    finally:
        if installed:
            shutil.rmtree(staging, ignore_errors=True)

    comparator = manifest["deterministic"]["comparator"]
    print(
        f"PASS: complete controls + offline candidates; comparator={comparator['engine']} "
        f"CER={comparator['micro_cer']:.2%}",
        flush=True,
    )
    for engine, candidate in manifest["deterministic"]["candidates"].items():
        resource = manifest["measurements"]["candidates"][engine]["resource_verdict"]
        print(
            f"{engine}: content_qualified={candidate['content_verdict']['qualified']} "
            f"cpu_resource_qualified={resource['qualified']}",
            flush=True,
        )
    print(f"wrote {_relative(BASELINE)} + immutable ignored JSONL details", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--worker", choices=OFFLINE_MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--details", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--summary", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild aggregates from exact cached details; refuse changed decode inputs.",
    )
    args = parser.parse_args()
    if args.worker:
        if args.aggregate_only:
            parser.error("--worker and --aggregate-only are mutually exclusive")
        if args.details is None or args.summary is None:
            parser.error("--worker requires --details and --summary")
        run_worker(args.worker, args.details, args.summary)
    elif args.details is not None or args.summary is not None:
        parser.error("--details/--summary are internal worker arguments")
    elif args.aggregate_only:
        reaggregate_parent()
    else:
        run_parent()


if __name__ == "__main__":
    main()
