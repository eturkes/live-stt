#!/usr/bin/env python3
"""Evaluator kernel for the shipped whisper + VAC path (D-016).

Publishes one deterministic row per ``arm_id/corpus_id`` -- content scored against the
pinned reference -- beside a structurally separate measurements block, immutable
content-addressed detail JSONL, and one atomically replaced ``tests/vac_baseline.json``.

Row work resumes automatically from three journals; there is no resume flag and no
user-facing path flag, because evidence identity is what decides reuse. Decode reaches
stable storage before its measurement, so an interruption between the paired fsyncs
costs exactly one re-decoded row.

The fingerprint is ``vac_contract_sha256``: the AST closure of the VAC decode path over
live_stt + streaming. It is deliberately NOT ``asr_contract_sha256``. That hash is the
sherpa offline-row contract, and folding VAC into it would make every VAC edit refuse a
sherpa rebuild -- the false-refusal class M11.3 fixed (L-030).

Run from the repository root:

    UV_PROJECT_ENVIRONMENT=.venv env -u PYTHONPATH uv run --no-sync python tests/eval_vac.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

import eval_models as shared  # noqa: E402

import live_stt  # noqa: E402
import replay  # noqa: E402

VAC_SCHEMA_VERSION = 1
VAC_BASELINE = TESTS / "vac_baseline.json"
VAC_DETAILS_DIR = shared.CACHE / "vac_eval-v1"
VAC_STAGING = shared.CACHE / "vac_eval-v1.staging"

# Roots of the VAC decode contract. Inclusion past these is derived by closure and
# spans streaming.py through live_stt's `from streaming import ...` edge; exclusion is
# the whole sherpa contract, which stays rooted in its own seeds (L-030).
VAC_CONTRACT_ROOTS = (
    "ASR_DEVICE",
    "OPENVINO_CACHE_DIR",
    "VAC_CHUNK_S",
    "VAC_TRIM_S",
    "WHISPER_ENGINES",
    "WhisperEngine",
    "_vac_segments",
)
VAC_CONTRACT_MODULES = ("live_stt", "streaming")


@dataclass(frozen=True)
class VacArm:
    """One evaluated configuration of the shipped path."""

    arm_id: str
    engine: str
    device: str


@dataclass(frozen=True)
class VacCorpus:
    """A pinned corpus, addressed through its committed manifest."""

    corpus_id: str
    manifest: Path
    wav: Path
    reference_key: str
    sha256_key: str


# Committed constants, never flags: the corpora and arms ARE the evidence identity.
VAC_ARMS = (VacArm("vac_npu", "whisper", live_stt.ASR_DEVICE),)
VAC_CORPORA = (
    VacCorpus(
        "long_form",
        TESTS / "long_form.json",
        shared.CACHE / "gongitsune_01.wav",
        "reference.text",
        "build.wav_sha256",
    ),
    VacCorpus(
        "retention",
        TESTS / "retention_probe.json",
        shared.CACHE / "retention_probe.wav",
        "probe.ja_ref",
        "probe.audio_sha256",
    ),
)


@dataclass(frozen=True)
class VacCase:
    arm: VacArm
    corpus: VacCorpus
    wav: Path
    reference: str
    duration_samples: int

    @property
    def row_id(self) -> str:
        return f"{self.arm.arm_id}/{self.corpus.corpus_id}"


@dataclass(frozen=True)
class VacTranscript:
    """Everything a rerun must reproduce exactly."""

    hypothesis: str
    segments: tuple[dict, ...]
    accepted_samples: int


@dataclass(frozen=True)
class VacObservation:
    """Content is deterministic; elapsed fields are measurements."""

    content: VacTranscript
    decode_seconds: float
    wall_seconds: float


VacDecoder = Callable[[VacCase], VacObservation]

DETERMINISTIC_ROW_KEYS = frozenset(
    {
        "D",
        "I",
        "N",
        "S",
        "accepted_samples",
        "arm_id",
        "cer",
        "corpus_id",
        "hyp",
        "ref",
        "row_id",
        "schema_version",
        "segments",
    }
)
SEGMENT_KEYS = frozenset({"n", "seg_len", "start", "text"})
MEASUREMENT_KEYS = frozenset({"decode_seconds", "row_id", "wall_seconds"})


# --- contract fingerprint ----------------------------------------------------------


def vac_contract_source(root: Path = ROOT) -> bytes:
    """Executable structure of everything that can change a decoded VAC row.

    `root` is what lets a probe hash an edited copy of the two modules without touching
    the working tree, which is how P1.3/P1.4 are proved.
    """
    surfaces = {name: shared.module_surface(root / f"{name}.py") for name in VAC_CONTRACT_MODULES}
    closure = shared.contract_closure(
        surfaces,
        [("live_stt", name) for name in VAC_CONTRACT_ROOTS],
        require=True,
    )
    parts = [
        f"{module}.{name}\n{shared._structural_dump(surfaces[module].symbols[name])}"
        for module, name in sorted(closure)
    ]
    return "\n\n".join(parts).encode()


def vac_contract_sha256(root: Path = ROOT) -> str:
    return shared._sha256_bytes(vac_contract_source(root))


def _path_ref(path: Path) -> str:
    """Repo-relative where possible; an out-of-tree probe tree records its real path."""
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _ref_path(ref: str) -> Path:
    return path if (path := Path(ref)).is_absolute() else ROOT / ref


def vac_pipeline_fingerprint() -> dict:
    """The VAC constants a reader needs, beside the hash that actually gates a rebuild."""
    return {
        "asr_device": live_stt.ASR_DEVICE,
        "sample_rate_hz": live_stt.SAMPLE_RATE,
        "vac_chunk_s": live_stt.VAC_CHUNK_S,
        "vac_contract_sha256": vac_contract_sha256(),
        "vac_trim_s": live_stt.VAC_TRIM_S,
        "vad_pre_pad_s": live_stt.VAD_PRE_PAD_S,
    }


# --- corpus ------------------------------------------------------------------------


def _key_path(manifest: Mapping[str, Any], key_path: str, corpus_id: str) -> Any:
    value: Any = manifest
    for token in key_path.split("."):
        if not isinstance(value, Mapping) or token not in value:
            raise RuntimeError(f"{corpus_id}: manifest key path does not resolve: {key_path}")
        value = value[token]
    return value


def _wav_samples(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != live_stt.SAMPLE_RATE or handle.getnchannels() != 1:
            raise RuntimeError(f"{path}: corpus WAV must be mono {live_stt.SAMPLE_RATE} Hz")
        return handle.getnframes()


def load_vac_cases(*, verify_wav: bool = True) -> list[VacCase]:
    """Arm-major ordered cases; the order IS the row order of every journal."""
    cases: list[VacCase] = []
    for arm in VAC_ARMS:
        for corpus in VAC_CORPORA:
            manifest = shared._load_json(corpus.manifest)
            reference = _key_path(manifest, corpus.reference_key, corpus.corpus_id)
            if not isinstance(reference, str) or not reference.strip():
                raise RuntimeError(f"{corpus.corpus_id}: reference text is missing")
            if verify_wav:
                expected = _key_path(manifest, corpus.sha256_key, corpus.corpus_id)
                if not corpus.wav.is_file():
                    raise RuntimeError(f"{corpus.corpus_id}: pinned WAV absent: {corpus.wav}")
                if shared.file_sha256(corpus.wav) != expected:
                    raise RuntimeError(f"{corpus.corpus_id}: pinned WAV hash changed")
            cases.append(
                VacCase(
                    arm=arm,
                    corpus=corpus,
                    wav=corpus.wav,
                    reference=reference,
                    duration_samples=_wav_samples(corpus.wav) if verify_wav else 0,
                )
            )
    validate_case_order(cases)
    return cases


def validate_case_order(cases: Sequence[VacCase]) -> None:
    """Reject a duplicate or out-of-order row id before any journal is written."""
    if not cases:
        raise RuntimeError("VAC corpus cannot be empty")
    row_ids = [case.row_id for case in cases]
    if len(set(row_ids)) != len(row_ids):
        raise RuntimeError("VAC row identity is not unique")
    arm_order = [case.arm.arm_id for case in cases]
    if arm_order != sorted(arm_order, key=list(dict.fromkeys(arm_order)).index):
        raise RuntimeError("VAC cases must be ordered arm-major")


# --- rows --------------------------------------------------------------------------


def score_row(case: VacCase, content: VacTranscript) -> dict:
    ref = shared.normalize(case.reference)
    if not ref:
        raise RuntimeError(f"reference normalizes to empty: {case.row_id}")
    substitutions, deletions, insertions = shared.align(ref, shared.normalize(content.hypothesis))
    return {
        "D": deletions,
        "I": insertions,
        "N": len(ref),
        "S": substitutions,
        "accepted_samples": content.accepted_samples,
        "arm_id": case.arm.arm_id,
        "cer": (substitutions + deletions + insertions) / len(ref),
        "corpus_id": case.corpus.corpus_id,
        "hyp": content.hypothesis,
        "ref": case.reference,
        "row_id": case.row_id,
        "schema_version": VAC_SCHEMA_VERSION,
        "segments": [dict(segment) for segment in content.segments],
    }


def validate_vac_row(case: VacCase, row: Mapping[str, Any]) -> None:
    """Structural + arithmetic acceptance for one deterministic row.

    Re-derives the score from the row's own `ref`/`hyp`, so a resumed prefix is trusted
    for what it contains rather than for having been written by this process.
    """
    if not isinstance(row, Mapping) or set(row) != DETERMINISTIC_ROW_KEYS:
        raise RuntimeError(f"VAC row keys drifted: {case.row_id}")
    if row["schema_version"] != VAC_SCHEMA_VERSION or isinstance(row["schema_version"], bool):
        raise RuntimeError(f"VAC row schema version drifted: {case.row_id}")
    if (row["row_id"], row["arm_id"], row["corpus_id"]) != (
        case.row_id,
        case.arm.arm_id,
        case.corpus.corpus_id,
    ):
        raise RuntimeError(f"VAC row identity drifted: {case.row_id}")
    if row["ref"] != case.reference:
        raise RuntimeError(f"VAC row reference drifted: {case.row_id}")
    if not isinstance(row["hyp"], str) or not isinstance(row["segments"], list):
        raise RuntimeError(f"VAC row content types drifted: {case.row_id}")
    if not isinstance(row["accepted_samples"], int) or isinstance(row["accepted_samples"], bool):
        raise RuntimeError(f"VAC row sample accounting drifted: {case.row_id}")
    expected = score_row(
        case,
        VacTranscript(row["hyp"], tuple(row["segments"]), row["accepted_samples"]),
    )
    if expected != dict(row):
        raise RuntimeError(f"VAC row score does not match its own content: {case.row_id}")
    for index, segment in enumerate(row["segments"], 1):
        if not isinstance(segment, Mapping) or set(segment) != SEGMENT_KEYS:
            raise RuntimeError(f"VAC segment keys drifted: {case.row_id}")
        if segment["n"] != index:
            raise RuntimeError(f"VAC segment numbering drifted: {case.row_id}")


def validate_measurement(case: VacCase, row: Mapping[str, Any]) -> None:
    if not isinstance(row, Mapping) or set(row) != MEASUREMENT_KEYS:
        raise RuntimeError(f"VAC measurement keys drifted: {case.row_id}")
    if row["row_id"] != case.row_id:
        raise RuntimeError(f"VAC measurement identity drifted: {case.row_id}")
    values = [row["decode_seconds"], row["wall_seconds"]]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise RuntimeError(f"VAC measurement value drifted: {case.row_id}")
    if row["wall_seconds"] < row["decode_seconds"]:
        raise RuntimeError(f"VAC measurement timing drifted: {case.row_id}")


# --- resumable serialization --------------------------------------------------------
# Mechanism ported from tests/eval_streaming.py:656-846; VAC carries its own schema,
# composite row id and validators. polish.md P-006 holds the unification.


def resume_paths(detail_path: Path) -> tuple[Path, Path, Path]:
    return (
        detail_path.with_name(f"{detail_path.name}.part"),
        detail_path.with_name(f"{detail_path.name}.measurements.part"),
        detail_path.with_name(f"{detail_path.name}.resume.json"),
    )


def clear_resume_artifacts(detail_path: Path, *, include_detail: bool = False) -> None:
    for path in resume_paths(detail_path):
        path.unlink(missing_ok=True)
    if include_detail:
        detail_path.unlink(missing_ok=True)


def resume_identity(cases: Sequence[VacCase]) -> dict:
    """What a journal must agree with to be reusable at all."""
    return {
        "row_ids": [case.row_id for case in cases],
        "schema_version": VAC_SCHEMA_VERSION,
        "vac_contract_sha256": vac_contract_sha256(),
    }


def _validated_jsonl_prefix(
    path: Path,
    validate: Callable[[int, Mapping[str, Any]], None],
) -> tuple[list[dict], list[bytes]]:
    """Rows up to the first that is truncated, unparsable, or invalid."""
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


def write_detail_resumable(
    detail_path: Path,
    cases: Sequence[VacCase],
    decode: VacDecoder,
    *,
    progress: Callable[[int, int, Mapping[str, Any]], None] | None = None,
) -> dict:
    """Append a validated row prefix, then atomically expose only the complete JSONL."""
    validate_case_order(cases)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    part_path, measurement_path, state_path = resume_paths(detail_path)
    expected_state = resume_identity(cases)

    def reset() -> None:
        clear_resume_artifacts(detail_path, include_detail=True)
        shared.write_atomic(state_path, [shared._json_bytes(expected_state)])

    if state_path.is_file():
        try:
            reusable = shared._load_json(state_path) == expected_state
        except (OSError, UnicodeError, json.JSONDecodeError):
            reusable = False
        if not reusable:
            reset()
    else:
        # No state file, but journals present: they belong to an unknown identity.
        if any(path.exists() for path in (detail_path, part_path, measurement_path)):
            clear_resume_artifacts(detail_path, include_detail=True)
        shared.write_atomic(state_path, [shared._json_bytes(expected_state)])

    def validate_detail(index: int, row: Mapping[str, Any]) -> None:
        if index >= len(cases):
            raise RuntimeError("VAC detail has extra rows")
        validate_vac_row(cases[index], row)

    def validate_measurement_row(index: int, row: Mapping[str, Any]) -> None:
        if index >= len(cases):
            raise RuntimeError("VAC measurement journal has extra rows")
        validate_measurement(cases[index], row)

    detail_source = detail_path if detail_path.is_file() else part_path
    rows, detail_chunks = _validated_jsonl_prefix(detail_source, validate_detail)
    measurements, measurement_chunks = _validated_jsonl_prefix(
        measurement_path, validate_measurement_row
    )

    def fully_valid(path: Path, chunks: Sequence[bytes]) -> bool:
        return not path.exists() or sum(map(len, chunks)) == path.stat().st_size

    if detail_path.is_file():
        # An immutable detail is all-or-nothing: anything short of every row paired and
        # byte-complete means the file cannot be what it claims, so start over.
        if (
            len(rows) != len(cases)
            or len(measurements) != len(cases)
            or not fully_valid(detail_source, detail_chunks)
            or not fully_valid(measurement_path, measurement_chunks)
        ):
            reset()
            detail_source = part_path
            rows, detail_chunks, measurements, measurement_chunks = [], [], [], []
    else:
        prefix = min(len(rows), len(measurements))
        rows, detail_chunks = rows[:prefix], detail_chunks[:prefix]
        measurements, measurement_chunks = measurements[:prefix], measurement_chunks[:prefix]
        _truncate_jsonl(part_path, detail_chunks)
        _truncate_jsonl(measurement_path, measurement_chunks)

    reused = len(rows)
    if reused < len(cases):
        with (
            part_path.open("ab") as detail_output,
            measurement_path.open("ab") as measurement_output,
        ):
            for index, case in enumerate(cases[reused:], reused + 1):
                observation = decode(case)
                row = score_row(case, observation.content)
                validate_vac_row(case, row)
                measurement = {
                    "decode_seconds": observation.decode_seconds,
                    "row_id": case.row_id,
                    "wall_seconds": observation.wall_seconds,
                }
                validate_measurement(case, measurement)
                detail_data = shared._json_bytes(row, compact=True)
                measurement_data = shared._json_bytes(measurement, compact=True)

                # Detail reaches stable storage first. Interruption between the two
                # fsyncs leaves an unpaired row, and prefix reconciliation re-decodes
                # exactly that one on the next run.
                detail_output.write(detail_data)
                detail_output.flush()
                os.fsync(detail_output.fileno())
                measurement_output.write(measurement_data)
                measurement_output.flush()
                os.fsync(measurement_output.fileno())

                rows.append(row)
                measurements.append(measurement)
                if progress is not None:
                    progress(index, len(cases), row)

    if len(rows) != len(cases) or len(measurements) != len(cases):
        raise RuntimeError("VAC detail did not reach every row")
    if not detail_path.is_file():
        part_path.replace(detail_path)
    detail_sha256 = shared.file_sha256(detail_path)
    audio_s = sum(case.duration_samples for case in cases) / live_stt.SAMPLE_RATE
    decode_s = sum(row["decode_seconds"] for row in measurements)
    wall_s = sum(row["wall_seconds"] for row in measurements)
    return {
        "deterministic": aggregate_rows(rows),
        "details": {"rows": len(cases), "sha256": detail_sha256},
        "measurement": {
            "audio_s": round(audio_s, 6),
            "overall_rtf": round(decode_s / audio_s, 6) if audio_s else 0.0,
            "rows_reused_on_resume": reused,
            "total_decode_s": round(decode_s, 6),
            "total_wall_s": round(wall_s, 6),
        },
    }


# --- aggregate + publish ------------------------------------------------------------


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Micro CER over the arm's rows, plus each row's own score."""
    if not rows:
        raise RuntimeError("VAC aggregate needs at least one row")
    totals = {key: sum(int(row[key]) for row in rows) for key in ("S", "D", "I", "N")}
    return {
        "corpora": {
            row["corpus_id"]: {key: row[key] for key in ("S", "D", "I", "N", "cer")} for row in rows
        },
        "micro_cer": (totals["S"] + totals["D"] + totals["I"]) / totals["N"],
        "rows": len(rows),
        "totals": totals,
    }


def build_vac_manifest(
    cases: Sequence[VacCase],
    summaries: Mapping[str, dict],
    details: Mapping[str, Path],
) -> dict:
    """Deterministic evidence, with every measured second in a sibling block."""
    arms = {case.arm.arm_id: case.arm for case in cases}
    arm_ids = tuple(arms)
    shared.validate_evidence_model_set(summaries, details, arm_ids, label="VAC evidence")
    corpora = {
        case.corpus.corpus_id: {
            "manifest": _path_ref(case.corpus.manifest),
            "reference_sha256": shared._sha256_bytes(case.reference.encode()),
            "wav": _path_ref(case.wav),
        }
        for case in cases
    }
    return {
        "deterministic": {
            "arms": {
                arm_id: {
                    "aggregate": summaries[arm_id]["deterministic"],
                    "details": dict(summaries[arm_id]["details"]),
                    "device": arm.device,
                    "engine": arm.engine,
                }
                for arm_id, arm in arms.items()
            },
            "corpora": corpora,
            "pipeline": vac_pipeline_fingerprint(),
            "row_ids": [case.row_id for case in cases],
            "schema_version": VAC_SCHEMA_VERSION,
        },
        "measurements": {
            "arms": {arm_id: summaries[arm_id]["measurement"] for arm_id in arm_ids},
            "excluded_from_deterministic_equality": True,
        },
    }


def install_vac_evidence(
    manifest: dict,
    summaries: Mapping[str, dict],
    staged_details: Mapping[str, Path],
    *,
    baseline: Path = VAC_BASELINE,
    details_dir: Path = VAC_DETAILS_DIR,
) -> None:
    """Immutable details first, then the manifest -- the aggregate is always last."""
    destinations = shared.install_content_addressed_details(
        summaries,
        staged_details,
        tuple(manifest["deterministic"]["arms"]),
        details_dir=details_dir,
    )
    for arm_id, destination in destinations.items():
        manifest["deterministic"]["arms"][arm_id]["details"]["path"] = _path_ref(destination)
    shared.write_atomic(baseline, [shared._json_bytes(manifest)])


# --- orchestration ------------------------------------------------------------------


def arm_decoder(arm: VacArm) -> VacDecoder:
    """Production replay drives the shipped VAC path; no scratch policy is copied."""
    recognizer = live_stt.load_recognizer(arm.engine, arm.device)

    def decode(case: VacCase) -> VacObservation:
        started = time.perf_counter()
        report = replay.replay_recognizer(case.wav, recognizer, arm.engine)
        wall_seconds = time.perf_counter() - started
        segments = tuple(
            {key: segment[key] for key in sorted(SEGMENT_KEYS)} for segment in report["segments"]
        )
        return VacObservation(
            content=VacTranscript(
                hypothesis="".join(segment["text"] for segment in segments),
                segments=segments,
                accepted_samples=round(report["audio_s"] * live_stt.SAMPLE_RATE),
            ),
            decode_seconds=report["total_decode_s"],
            wall_seconds=wall_seconds,
        )

    return decode


def arm_cases(cases: Sequence[VacCase], arm_id: str) -> list[VacCase]:
    return [case for case in cases if case.arm.arm_id == arm_id]


def _no_child_reuse(*_: object) -> None:
    """VAC resumes per ROW inside the child journals, so a whole-child summary reuse
    check would only shadow the finer-grained mechanism. Always run the child."""
    return None


def run_worker(arm_id: str, details_path: Path, summary_path: Path) -> None:
    cases = arm_cases(load_vac_cases(), arm_id)
    if not cases:
        raise RuntimeError(f"unknown VAC arm: {arm_id}")

    def progress(index: int, total: int, row: Mapping[str, Any]) -> None:
        print(f"{row['row_id']}: {index}/{total} cer={row['cer']:.4f}", flush=True)

    summary = write_detail_resumable(
        details_path, cases, arm_decoder(cases[0].arm), progress=progress
    )
    shared.write_atomic(summary_path, [shared._json_bytes(summary)])


def run_parent() -> None:
    cases = load_vac_cases()
    arm_ids = tuple(dict.fromkeys(case.arm.arm_id for case in cases))
    VAC_STAGING.mkdir(parents=True, exist_ok=True)
    installed = False
    try:
        summaries, details = shared.run_isolated_workers(
            arm_ids,
            staging=VAC_STAGING,
            worker_script=Path(__file__).resolve(),
            reusable_child=_no_child_reuse,
        )
        manifest = build_vac_manifest(cases, summaries, details)
        install_vac_evidence(manifest, summaries, details)
        installed = True
    finally:
        if installed:
            shutil.rmtree(VAC_STAGING, ignore_errors=True)
    for arm_id in arm_ids:
        aggregate = manifest["deterministic"]["arms"][arm_id]["aggregate"]
        print(f"{arm_id}: micro_cer={aggregate['micro_cer']:.4%} rows={aggregate['rows']}")
    print(f"wrote {shared._relative(VAC_BASELINE)} + immutable ignored JSONL details", flush=True)


def reaggregate_parent(
    *, baseline: Path = VAC_BASELINE, details_dir: Path = VAC_DETAILS_DIR
) -> None:
    """Rebuild the manifest from installed details; refuse any changed decode input."""
    previous = shared._load_json(baseline)
    cases = load_vac_cases(verify_wav=False)
    recorded = previous["deterministic"]["pipeline"]["vac_contract_sha256"]
    current = vac_contract_sha256()
    if recorded != current:
        raise RuntimeError(f"VAC contract changed: recorded {recorded}, current {current}")
    if previous["deterministic"]["row_ids"] != [case.row_id for case in cases]:
        raise RuntimeError("VAC row set changed; rerun the full evaluator")
    summaries: dict[str, dict] = {}
    staged: dict[str, Path] = {}
    for arm_id, arm in previous["deterministic"]["arms"].items():
        detail = _ref_path(arm["details"]["path"])
        if not detail.is_file():
            raise RuntimeError(f"{arm_id}: installed detail is missing: {detail}")
        if shared.file_sha256(detail) != arm["details"]["sha256"]:
            raise RuntimeError(f"{arm_id}: installed detail hash changed: {detail}")
        rows = [json.loads(line) for line in detail.read_text(encoding="utf-8").splitlines()]
        for case, row in zip(arm_cases(cases, arm_id), rows, strict=True):
            validate_vac_row(case, row)
        summaries[arm_id] = {
            "deterministic": aggregate_rows(rows),
            "details": {"rows": len(rows), "sha256": arm["details"]["sha256"]},
            # Timing is measured, never re-derived: carry the recorded block through so
            # repeated rebuilds are byte-identical.
            "measurement": previous["measurements"]["arms"][arm_id],
        }
        staged[arm_id] = detail
    manifest = build_vac_manifest(cases, summaries, staged)
    for arm_id, arm in previous["deterministic"]["arms"].items():
        manifest["deterministic"]["arms"][arm_id]["details"]["path"] = arm["details"]["path"]
    shared.write_atomic(baseline, [shared._json_bytes(manifest)])
    print(f"rebuilt {_path_ref(baseline)} from cached details", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--worker", help=argparse.SUPPRESS)
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
