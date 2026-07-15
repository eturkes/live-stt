#!/usr/bin/env python3
"""Build the complete pinned Japanese short-form evaluation corpus.

Sources:
- Common Voice 8 Japanese test: revision-pinned Parquet mirror of the complete
  4,483-row crowd-recorded split (CC0-1.0). The mirror exposes no speaker fields.
- FLEURS Japanese test: revision-pinned TSV + WAV archive, 650 read-speech
  recordings (CC-BY-4.0). Sentence IDs repeat for independently recorded reads.

Verified source payloads stay in ``spike/backends/cache/``. Canonical mono 16 kHz
PCM16 WAVs + the detailed JSONL index live in a content-addressed directory there;
git receives only ``tests/short_corpus.json`` (provenance, statistics, fingerprints).
The seven historical Common Voice replay clips + ``tests/real_clips.json`` remain
compatibility outputs of the same verified rows.

Run from the repository root:

    uv run --with soundfile==0.14.0 --with pyarrow==25.0.0 \
      python tests/fetch_real_clips.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import wave
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, NotRequired, TypedDict, cast

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cer import normalize  # noqa: E402  (after sys.path injection)
from live_stt import SAMPLE_RATE, resample  # noqa: E402

CACHE = ROOT / "spike" / "backends" / "cache"
MANIFEST = ROOT / "tests" / "short_corpus.json"
REPLAY_MANIFEST = ROOT / "tests" / "real_clips.json"
SCHEMA_VERSION = 1
DOWNLOAD_TIMEOUT_S = 90
BUILD_DEPENDENCIES = {
    "libsndfile": "1.2.2",
    "numpy": "2.4.6",
    "pyarrow": "25.0.0",
    "soundfile": "0.14.0",
}
EXPECTED_INDEX_SHA256 = "98e0d8a40fbc2d6e819ddd8db22fd23c2d7f050ac2da5773ac207a1bd0a14d36"

CV_DATASET = "japanese-asr/ja_asr.common_voice_8_0"
CV_REVISION = "bf8819e8d9a5feb51b0c718686bd20ea67a3c729"
CV_SPLIT = "test"
CV_ROWS = 4483

FLEURS_DATASET = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_CONFIG = "ja_jp"
FLEURS_SPLIT = "test"
FLEURS_ROWS = 650


@dataclass(frozen=True)
class SourceSpec:
    filename: str
    path: str
    url: str
    sha256: str
    size: int


CV_PARQUET = SourceSpec(
    filename="common_voice_8_test.parquet",
    path="data/test-00000-of-00001.parquet",
    url=(
        f"https://huggingface.co/datasets/{CV_DATASET}/resolve/"
        f"{CV_REVISION}/data/test-00000-of-00001.parquet"
    ),
    sha256="44a9141bc16cfa34877955fb39003ad34d3b730417a05c9eb50d8e90ba3ec40a",
    size=151_322_876,
)
FLEURS_TSV = SourceSpec(
    filename="fleurs_ja_test.tsv",
    path="data/ja_jp/test.tsv",
    url=(
        f"https://huggingface.co/datasets/{FLEURS_DATASET}/resolve/"
        f"{FLEURS_REVISION}/data/ja_jp/test.tsv"
    ),
    sha256="5dd9643511437414681ad3f23508596c621cdf78978724a09f1f06fefe9d300b",
    size=361_174,
)
FLEURS_AUDIO = SourceSpec(
    filename="fleurs_ja_test.tar.gz",
    path="data/ja_jp/audio/test.tar.gz",
    url=(
        f"https://huggingface.co/datasets/{FLEURS_DATASET}/resolve/"
        f"{FLEURS_REVISION}/data/ja_jp/audio/test.tar.gz"
    ),
    sha256="5de465fa7aaafc4e2c13aba44771550b8cd2dd29bb9b265daeb6d92ca8e0c136",
    size=448_762_391,
)

# Historical replay fixtures derived from exact Common Voice row offsets.
LEAD_S, TAIL_S = 0.3, 0.6
SINGLES = [
    ("cv_short", 65, "real CV: short utterance -> 1 segment"),
    ("cv_med", 10, "real CV: medium sentence -> 1 segment"),
    ("cv_long", 2, "real CV: longer sentence with a name -> 1 segment"),
    ("cv_kana", 15, "real CV: katakana loanword (フィリピン) -> proper-noun acoustics"),
    ("cv_xlong", 4, "real CV: long katakana-heavy sentence -> sustained real decode"),
]
CONCATS = [
    ("cv_multi", [11, 81, 93], 0.7, "real CV x3, 0.7 s gaps (> VAD_MIN_SILENCE_S) -> 3 segments"),
    ("cv_paused", [46, 55], 2.0, "real CV x2, 2.0 s gap -> 2 segments"),
]
REPLAY_ROWS = {row for _, row, _ in SINGLES}
for _, rows, _, _ in CONCATS:
    REPLAY_ROWS.update(rows)


@dataclass(frozen=True)
class FleursRow:
    source_row: int
    sentence_id: int
    filename: str
    corpus_id: str
    reference: str
    transcription: str
    normalized_reference: str
    num_samples: int
    gender: str


class SummaryEntry(TypedDict):
    corpus_id: str
    source: str
    normalized_reference: str
    duration_samples: int
    pcm_sha256: str
    gender: str | None


class CorpusEntry(SummaryEntry):
    source_row: int
    wav: str
    reference: str
    duration_seconds: float
    sentence_id: NotRequired[int]
    source_filename: NotRequired[str]
    source_transcription: NotRequired[str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_atomic(path: Path, chunks: Iterable[bytes]) -> None:
    """Commit bytes with one rename; interrupted writes preserve the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.part")
    part.unlink(missing_ok=True)
    try:
        with part.open("xb") as output:
            for chunk in chunks:
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def _source_valid(path: Path, spec: SourceSpec) -> bool:
    return path.is_file() and path.stat().st_size == spec.size and file_sha256(path) == spec.sha256


def fetch_source(spec: SourceSpec, cache: Path = CACHE) -> Path:
    """Return an exact payload; a failed refresh cannot replace cached bytes."""
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / spec.filename
    if _source_valid(path, spec):
        print(f"source: cached + verified {path.name}")
        return path

    part = path.with_name(f"{path.name}.part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(spec.url, headers={"User-Agent": "live-stt-corpus/2"})
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response,
            part.open("xb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual_hash = digest.hexdigest()
        if actual_hash != spec.sha256:
            raise RuntimeError(
                f"source SHA-256 mismatch for {spec.path}: expected {spec.sha256}, "
                f"got {actual_hash} (size {size})"
            )
        if size != spec.size:
            raise RuntimeError(
                f"source size mismatch for {spec.path}: expected {spec.size}, got {size}"
            )
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    print(f"source: downloaded + verified {path.name}")
    return path


def _decode_audio(
    raw: bytes,
    *,
    context: str,
    expected_rate: int | None = None,
    expected_samples: int | None = None,
) -> np.ndarray:
    try:
        import soundfile as sf  # pyright: ignore[reportMissingImports]  (transient dep)

        samples, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"cannot decode audio for {context}: {exc}") from exc
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise RuntimeError(f"invalid decoded audio for {context}")
    if expected_rate is not None and rate != expected_rate:
        raise RuntimeError(f"unexpected sample rate for {context}: {rate} != {expected_rate}")
    if expected_samples is not None and len(samples) != expected_samples:
        raise RuntimeError(
            f"sample-count mismatch for {context}: {len(samples)} != {expected_samples}"
        )
    if rate != SAMPLE_RATE:
        samples = resample(samples, rate, SAMPLE_RATE)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise RuntimeError(f"invalid canonical audio for {context}")
    return np.ascontiguousarray(samples, dtype=np.float32)


def _check_build_dependencies() -> None:
    try:
        import pyarrow  # pyright: ignore[reportMissingImports]  (transient dep)
        import soundfile  # pyright: ignore[reportMissingImports]  (transient dep)
    except ImportError as exc:
        raise RuntimeError(
            "corpus build needs soundfile==0.14.0 and pyarrow==25.0.0; use the documented command"
        ) from exc
    actual = {
        "libsndfile": soundfile.__libsndfile_version__,
        "numpy": np.__version__,
        "pyarrow": pyarrow.__version__,
        "soundfile": soundfile.__version__,
    }
    if actual != BUILD_DEPENDENCIES:
        raise RuntimeError(
            f"unqualified corpus build dependencies: expected {BUILD_DEPENDENCIES}, got {actual}"
        )


def _pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _write_pcm_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)


def _read_pcm_wav(path: Path, context: str) -> bytes:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != SAMPLE_RATE
                or source.getcomptype() != "NONE"
            ):
                raise RuntimeError(f"non-canonical cached PCM for {context}")
            frames = source.readframes(source.getnframes())
            if len(frames) != source.getnframes() * 2:
                raise RuntimeError(f"truncated cached PCM for {context}")
            return frames
    except (EOFError, OSError, wave.Error) as exc:
        raise RuntimeError(f"cannot read cached PCM for {context}: {exc}") from exc


def _validate_cv_metadata(names: list[str], rows: int) -> None:
    if names != ["audio", "transcription"]:
        raise RuntimeError(f"unexpected Common Voice schema: {names}")
    if rows != CV_ROWS:
        raise RuntimeError(f"unexpected Common Voice row count: {rows} != {CV_ROWS}")


def _parse_cv_row(row: object, source_row: int) -> tuple[str, str, bytes]:
    audio = row.get("audio") if isinstance(row, dict) else None
    reference = row.get("transcription") if isinstance(row, dict) else None
    raw = audio.get("bytes") if isinstance(audio, dict) else None
    if not isinstance(raw, bytes) or not isinstance(reference, str):
        raise RuntimeError(f"unexpected Common Voice row schema at {source_row}")
    reference = reference.strip()
    normalized = normalize(reference)
    if not normalized:
        raise RuntimeError(f"empty normalized Common Voice reference at {source_row}")
    return reference, normalized, raw


def _cv_rows(path: Path):
    try:
        import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]  (transient dep)

        parquet = pq.ParquetFile(path)
    except Exception as exc:
        raise RuntimeError(f"cannot open Common Voice Parquet: {exc}") from exc
    _validate_cv_metadata(parquet.schema_arrow.names, parquet.metadata.num_rows)
    source_row = 0
    for batch in parquet.iter_batches(columns=["audio", "transcription"], batch_size=64):
        for row in batch.to_pylist():
            reference, normalized, raw = _parse_cv_row(row, source_row)
            yield source_row, reference, normalized, raw
            source_row += 1
    if source_row != CV_ROWS:
        raise RuntimeError(f"incomplete Common Voice scan: {source_row} != {CV_ROWS}")


def parse_fleurs_tsv(path: Path, *, expected_rows: int = FLEURS_ROWS) -> list[FleursRow]:
    rows: list[FleursRow] = []
    seen_filenames: set[str] = set()
    seen_corpus_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read FLEURS TSV: {exc}") from exc
    for source_row, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) != 7:
            raise RuntimeError(f"unexpected FLEURS TSV schema at row {source_row}")
        sentence_raw, filename, reference, transcription, _characters, samples_raw, gender = fields
        try:
            sentence_id = int(sentence_raw)
            num_samples = int(samples_raw)
        except ValueError as exc:
            raise RuntimeError(f"invalid FLEURS integer at row {source_row}") from exc
        name = PurePosixPath(filename)
        if (
            sentence_id < 0
            or num_samples <= 0
            or name.name != filename
            or name.suffix.lower() != ".wav"
            or not name.stem.isdecimal()
        ):
            raise RuntimeError(f"invalid FLEURS identity at row {source_row}")
        if gender not in {"MALE", "FEMALE", "OTHER"}:
            raise RuntimeError(f"invalid FLEURS gender at row {source_row}: {gender!r}")
        reference = reference.strip()
        transcription = transcription.strip()
        normalized = normalize(reference)
        if not normalized or normalized != normalize(transcription):
            raise RuntimeError(f"invalid FLEURS reference at row {source_row}")
        corpus_id = f"fleurs-ja-test-{name.stem}"
        if filename in seen_filenames or corpus_id in seen_corpus_ids:
            raise RuntimeError(f"duplicate FLEURS audio identity at row {source_row}")
        seen_filenames.add(filename)
        seen_corpus_ids.add(corpus_id)
        rows.append(
            FleursRow(
                source_row=source_row,
                sentence_id=sentence_id,
                filename=filename,
                corpus_id=corpus_id,
                reference=reference,
                transcription=transcription,
                normalized_reference=normalized,
                num_samples=num_samples,
                gender=gender.lower(),
            )
        )
    if len(rows) != expected_rows:
        raise RuntimeError(f"unexpected FLEURS row count: {len(rows)} != {expected_rows}")
    return rows


def validate_fleurs_archive(
    archive: tarfile.TarFile, expected_filenames: set[str]
) -> dict[str, tarfile.TarInfo]:
    """Return exact expected WAV members; reject links, traversal, extras, gaps."""
    found: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe FLEURS archive path: {member.name!r}")
        if member.isdir():
            continue
        if not member.isfile():
            raise RuntimeError(f"non-regular FLEURS archive member: {member.name!r}")
        filename = path.name
        if filename not in expected_filenames:
            raise RuntimeError(f"unexpected FLEURS archive member: {member.name!r}")
        if filename in found:
            raise RuntimeError(f"duplicate FLEURS archive member: {filename!r}")
        found[filename] = member
    missing = expected_filenames - found.keys()
    if missing:
        preview = sorted(missing)[:5]
        raise RuntimeError(f"missing FLEURS archive members: {preview}")
    return found


def _entry(
    *,
    corpus_id: str,
    source: str,
    source_row: int,
    wav: str,
    reference: str,
    normalized_reference: str,
    pcm: bytes,
    gender: str | None,
) -> CorpusEntry:
    samples = len(pcm) // 2
    return {
        "corpus_id": corpus_id,
        "source": source,
        "source_row": source_row,
        "wav": wav,
        "reference": reference,
        "normalized_reference": normalized_reference,
        "duration_samples": samples,
        "duration_seconds": round(samples / SAMPLE_RATE, 6),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "gender": gender,
    }


def _materialize_common_voice(
    source: Path, staging: Path
) -> tuple[list[CorpusEntry], dict[int, bytes]]:
    entries: list[CorpusEntry] = []
    replay_pcm: dict[int, bytes] = {}
    for source_row, reference, normalized, raw in _cv_rows(source):
        corpus_id = f"cv8-ja-test-{source_row:06d}"
        relative = f"common_voice_8/{corpus_id}.wav"
        samples = _decode_audio(raw, context=corpus_id)
        pcm = _pcm16(samples)
        _write_pcm_wav(staging / relative, pcm)
        entries.append(
            _entry(
                corpus_id=corpus_id,
                source="common_voice_8",
                source_row=source_row,
                wav=relative,
                reference=reference,
                normalized_reference=normalized,
                pcm=pcm,
                gender=None,
            )
        )
        if source_row in REPLAY_ROWS:
            replay_pcm[source_row] = pcm
    if replay_pcm.keys() != REPLAY_ROWS:
        raise RuntimeError(f"missing replay rows: {sorted(REPLAY_ROWS - replay_pcm.keys())}")
    return entries, replay_pcm


def _materialize_fleurs(source: Path, rows: list[FleursRow], staging: Path) -> list[CorpusEntry]:
    entries: list[CorpusEntry] = []
    try:
        archive = tarfile.open(source, "r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"cannot open FLEURS archive: {exc}") from exc
    with archive:
        members = validate_fleurs_archive(archive, {row.filename for row in rows})
        for row in rows:
            extracted = archive.extractfile(members[row.filename])
            if extracted is None:
                raise RuntimeError(f"cannot read FLEURS member: {row.filename}")
            raw = extracted.read()
            if len(raw) != members[row.filename].size:
                raise RuntimeError(f"truncated FLEURS member: {row.filename}")
            samples = _decode_audio(
                raw,
                context=row.corpus_id,
                expected_rate=SAMPLE_RATE,
                expected_samples=row.num_samples,
            )
            pcm = _pcm16(samples)
            relative = f"fleurs/{row.corpus_id}.wav"
            _write_pcm_wav(staging / relative, pcm)
            entry = _entry(
                corpus_id=row.corpus_id,
                source="fleurs",
                source_row=row.source_row,
                wav=relative,
                reference=row.reference,
                normalized_reference=row.normalized_reference,
                pcm=pcm,
                gender=row.gender,
            )
            entry.update(
                sentence_id=row.sentence_id,
                source_filename=row.filename,
                source_transcription=row.transcription,
            )
            entries.append(entry)
    return entries


def _fingerprint(
    entries: Sequence[SummaryEntry], field: Literal["pcm_sha256", "normalized_reference"]
) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item["corpus_id"]):
        digest.update(entry["corpus_id"].encode("utf-8"))
        digest.update(b"\0")
        value = entry["pcm_sha256"] if field == "pcm_sha256" else entry["normalized_reference"]
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def summarize(entries: Sequence[SummaryEntry]) -> dict:
    if not entries:
        raise RuntimeError("cannot summarize an empty corpus")
    ids = [entry["corpus_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate corpus ID")
    references = Counter(entry["normalized_reference"] for entry in entries)
    pcm = Counter(entry["pcm_sha256"] for entry in entries)
    recordings = Counter(references.values())
    durations = [entry["duration_samples"] for entry in entries]
    bucket_counts = {
        "0-5": sum(samples < 5 * SAMPLE_RATE for samples in durations),
        "5-10": sum(5 * SAMPLE_RATE <= samples < 10 * SAMPLE_RATE for samples in durations),
        "10-20": sum(10 * SAMPLE_RATE <= samples < 20 * SAMPLE_RATE for samples in durations),
        "20+": sum(samples >= 20 * SAMPLE_RATE for samples in durations),
    }
    genders: dict[str, dict[str, float | int]] = {}
    present_genders = {
        gender for entry in entries if isinstance((gender := entry.get("gender")), str)
    }
    for gender in sorted(present_genders):
        selected = [entry for entry in entries if entry.get("gender") == gender]
        genders[str(gender)] = {
            "audio_seconds": round(
                sum(entry["duration_samples"] for entry in selected) / SAMPLE_RATE, 6
            ),
            "rows": len(selected),
        }
    total_samples = sum(durations)
    return {
        "rows": len(entries),
        "audio_hours": round(total_samples / SAMPLE_RATE / 3600, 6),
        "duration_seconds": {
            "buckets": bucket_counts,
            "max": round(max(durations) / SAMPLE_RATE, 6),
            "mean": round(total_samples / len(entries) / SAMPLE_RATE, 6),
            "min": round(min(durations) / SAMPLE_RATE, 6),
            "total": round(total_samples / SAMPLE_RATE, 6),
        },
        "gender": genders,
        "references": {
            "duplicate_groups": sum(count > 1 for count in references.values()),
            "duplicate_rows": len(entries) - len(references),
            "max_recordings_per_reference": max(references.values()),
            "recordings_per_reference": {
                str(count): groups for count, groups in sorted(recordings.items())
            },
            "unique": len(references),
        },
        "pcm": {
            "duplicate_groups": sum(count > 1 for count in pcm.values()),
            "duplicate_rows": len(entries) - len(pcm),
            "unique": len(pcm),
        },
        "fingerprints": {
            "pcm_sha256": _fingerprint(entries, "pcm_sha256"),
            "references_sha256": _fingerprint(entries, "normalized_reference"),
        },
    }


def _payload(spec: SourceSpec) -> dict:
    return {
        "path": spec.path,
        "sha256": spec.sha256,
        "size_bytes": spec.size,
        "url": spec.url,
    }


def _manifest(entries: list[CorpusEntry], index_sha256: str, corpus_dir: Path) -> dict:
    cv = [entry for entry in entries if entry["source"] == "common_voice_8"]
    fleurs = [entry for entry in entries if entry["source"] == "fleurs"]
    if len(cv) != CV_ROWS or len(fleurs) != FLEURS_ROWS:
        raise RuntimeError(f"incomplete corpus: Common Voice={len(cv)}, FLEURS={len(fleurs)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "audio": {"channels": 1, "encoding": "PCM16", "sample_rate_hz": SAMPLE_RATE},
        "builder": {
            "dependencies": BUILD_DEPENDENCIES,
            "script": "tests/fetch_real_clips.py",
        },
        "cache": {
            "directory": corpus_dir.relative_to(ROOT).as_posix(),
            "index": "index.jsonl",
            "index_sha256": index_sha256,
            "rows": len(entries),
        },
        "sources": {
            "common_voice_8": {
                "attribution": "Mozilla Common Voice contributors",
                "dataset_card": (
                    f"https://huggingface.co/datasets/{CV_DATASET}/blob/{CV_REVISION}/README.md"
                ),
                "dataset": CV_DATASET,
                "license": "CC0-1.0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "limitations": [
                    "crowd-recorded scripted speech",
                    "mirror omits speaker metadata",
                    "training overlap with evaluated models is unknown",
                ],
                "payloads": [_payload(CV_PARQUET)],
                "revision": CV_REVISION,
                "source_identity": f"{CV_SPLIT}[0:{CV_ROWS}]",
                "speaker_metadata": "absent",
                "statistics": summarize(cv),
            },
            "fleurs": {
                "attribution": (
                    "FLEURS authors and recording contributors; cite Conneau et al. (2022)"
                ),
                "citation": "https://arxiv.org/abs/2205.12446",
                "dataset_card": (
                    f"https://huggingface.co/datasets/{FLEURS_DATASET}/blob/"
                    f"{FLEURS_REVISION}/README.md"
                ),
                "dataset": FLEURS_DATASET,
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "limitations": [
                    "read speech; not an unseen production-speech proxy",
                    "repeated sentences have one to three independent recordings",
                    "Nemotron publishes FLEURS results, so candidate overlap is explicit",
                    "gender labels are corpus metadata, not stable speaker identities",
                ],
                "payloads": [_payload(FLEURS_TSV), _payload(FLEURS_AUDIO)],
                "revision": FLEURS_REVISION,
                "source_identity": f"{FLEURS_CONFIG}/{FLEURS_SPLIT}[0:{FLEURS_ROWS}]",
                "speaker_metadata": "gender only; no speaker ID",
                "statistics": summarize(fleurs),
            },
        },
    }


def _json_bytes(value: object, *, compact: bool = False) -> bytes:
    separators = (",", ":") if compact else None
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=None if compact else 2,
            separators=separators,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _index_bytes(entries: list[CorpusEntry]) -> bytes:
    return b"".join(_json_bytes(entry, compact=True) for entry in entries)


def _safe_cached_wav(corpus_dir: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".wav":
        raise RuntimeError(f"unsafe cached WAV path: {relative!r}")
    candidate = corpus_dir.joinpath(*path.parts)
    if not candidate.is_file():
        raise RuntimeError(f"missing cached PCM: {relative}")
    return candidate


def validate_cached_index(corpus_dir: Path, expected_sha256: str) -> list[CorpusEntry]:
    index = corpus_dir / "index.jsonl"
    if not index.is_file() or file_sha256(index) != expected_sha256:
        raise RuntimeError("cached corpus index fingerprint mismatch")
    entries: list[CorpusEntry] = []
    seen: set[str] = set()
    try:
        lines = index.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read cached corpus index: {exc}") from exc
    for source_row, line in enumerate(lines):
        try:
            raw_entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid cached corpus index row {source_row}") from exc
        required = {
            "corpus_id",
            "source",
            "source_row",
            "wav",
            "reference",
            "normalized_reference",
            "duration_samples",
            "duration_seconds",
            "pcm_sha256",
            "gender",
        }
        if not isinstance(raw_entry, dict) or not required <= raw_entry.keys():
            raise RuntimeError(f"invalid cached corpus index schema at row {source_row}")
        if not (
            isinstance(raw_entry["corpus_id"], str)
            and raw_entry["source"] in {"common_voice_8", "fleurs"}
            and isinstance(raw_entry["source_row"], int)
            and raw_entry["source_row"] >= 0
            and isinstance(raw_entry["wav"], str)
            and isinstance(raw_entry["reference"], str)
            and isinstance(raw_entry["normalized_reference"], str)
            and isinstance(raw_entry["duration_samples"], int)
            and raw_entry["duration_samples"] > 0
            and isinstance(raw_entry["duration_seconds"], (int, float))
            and isinstance(raw_entry["pcm_sha256"], str)
            and len(raw_entry["pcm_sha256"]) == 64
            and (raw_entry["gender"] is None or raw_entry["gender"] in {"male", "female", "other"})
        ):
            raise RuntimeError(f"invalid cached corpus index types at row {source_row}")
        entry = cast(CorpusEntry, raw_entry)
        corpus_id = entry["corpus_id"]
        if corpus_id in seen:
            raise RuntimeError(f"duplicate/invalid cached corpus ID at row {source_row}")
        if (
            not isinstance(entry["reference"], str)
            or not isinstance(entry["normalized_reference"], str)
            or normalize(entry["reference"]) != entry["normalized_reference"]
        ):
            raise RuntimeError(f"invalid cached reference at row {source_row}")
        if entry["duration_seconds"] != round(entry["duration_samples"] / SAMPLE_RATE, 6):
            raise RuntimeError(f"invalid cached duration at row {source_row}")
        path = _safe_cached_wav(corpus_dir, entry["wav"])
        pcm = _read_pcm_wav(path, corpus_id)
        if (
            len(pcm) // 2 != entry["duration_samples"]
            or hashlib.sha256(pcm).hexdigest() != entry["pcm_sha256"]
        ):
            raise RuntimeError(f"cached PCM fingerprint mismatch for {corpus_id}")
        seen.add(corpus_id)
        entries.append(entry)
    return entries


def _replay_manifest(entries: list[CorpusEntry]) -> dict[str, dict]:
    cv = {entry["source_row"]: entry for entry in entries if entry["source"] == "common_voice_8"}
    manifest: dict[str, dict] = {}
    for corpus_id, source_row, purpose in SINGLES:
        manifest[corpus_id] = {
            "ja_ref": cv[source_row]["reference"],
            "purpose": purpose,
            "source": f"{CV_DATASET}@{CV_REVISION}#{CV_SPLIT}[{source_row}]",
        }
    for corpus_id, source_rows, gap_s, purpose in CONCATS:
        manifest[corpus_id] = {
            "ja_ref": " ".join(cv[source_row]["reference"] for source_row in source_rows),
            "purpose": purpose,
            "source": f"{CV_DATASET}@{CV_REVISION}#{CV_SPLIT}{source_rows} gap={gap_s}s",
        }
    return manifest


def _materialize_replay_clips(replay_pcm: dict[int, bytes], staging: Path) -> None:
    lead = b"\0\0" * int(LEAD_S * SAMPLE_RATE)
    tail = b"\0\0" * int(TAIL_S * SAMPLE_RATE)
    for corpus_id, source_row, _purpose in SINGLES:
        _write_pcm_wav(staging / f"{corpus_id}.wav", lead + replay_pcm[source_row] + tail)
    for corpus_id, source_rows, gap_s, _purpose in CONCATS:
        gap = b"\0\0" * int(gap_s * SAMPLE_RATE)
        parts = [lead]
        for index, source_row in enumerate(source_rows):
            if index:
                parts.append(gap)
            parts.append(replay_pcm[source_row])
        parts.append(tail)
        _write_pcm_wav(staging / f"{corpus_id}.wav", b"".join(parts))


def _replay_pcm_from_cache(entries: list[CorpusEntry], corpus_dir: Path) -> dict[int, bytes]:
    selected: dict[int, bytes] = {}
    for entry in entries:
        if entry["source"] == "common_voice_8" and entry["source_row"] in REPLAY_ROWS:
            selected[entry["source_row"]] = _read_pcm_wav(
                _safe_cached_wav(corpus_dir, entry["wav"]), entry["corpus_id"]
            )
    if selected.keys() != REPLAY_ROWS:
        raise RuntimeError("cached corpus lacks historical replay rows")
    return selected


def _load_cached() -> tuple[list[CorpusEntry], Path] | None:
    if not MANIFEST.is_file():
        corpus_dir = CACHE / f"short_corpus-v1-{EXPECTED_INDEX_SHA256[:16]}"
        if not corpus_dir.exists():
            return None
        entries = validate_cached_index(corpus_dir, EXPECTED_INDEX_SHA256)
        _manifest(entries, EXPECTED_INDEX_SHA256, corpus_dir)
        return entries, corpus_dir
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read committed corpus manifest: {exc}") from exc
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return None
    cache = manifest.get("cache")
    if not isinstance(cache, dict):
        raise RuntimeError("invalid committed corpus cache metadata")
    relative = cache.get("directory")
    index_sha256 = cache.get("index_sha256")
    if not isinstance(relative, str) or not isinstance(index_sha256, str):
        raise RuntimeError("invalid committed corpus cache identity")
    if index_sha256 != EXPECTED_INDEX_SHA256:
        raise RuntimeError("committed corpus index is not the qualified fingerprint")
    relative_path = PurePosixPath(relative)
    expected_parent = PurePosixPath(CACHE.relative_to(ROOT).as_posix())
    if relative_path.parent != expected_parent or not relative_path.name.startswith(
        "short_corpus-v1-"
    ):
        raise RuntimeError(f"unsafe committed corpus cache path: {relative!r}")
    corpus_dir = ROOT.joinpath(*relative_path.parts)
    if not corpus_dir.exists():
        return None
    entries = validate_cached_index(corpus_dir, index_sha256)
    expected = _manifest(entries, index_sha256, corpus_dir)
    if manifest != expected:
        raise RuntimeError("committed manifest disagrees with cached corpus")
    return entries, corpus_dir


def _install_replay(entries: list[CorpusEntry], replay_pcm: dict[int, bytes]) -> None:
    staging = CACHE / "replay_clips.part"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        _materialize_replay_clips(replay_pcm, staging)
        for corpus_id, *_ in [*SINGLES, *CONCATS]:
            (staging / f"{corpus_id}.wav").replace(CACHE / f"{corpus_id}.wav")
        # Preserve the historical fixture ordering; this manifest predates the
        # canonical sort used by M10's new aggregate evidence.
        replay_manifest = (
            json.dumps(_replay_manifest(entries), ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        write_atomic(REPLAY_MANIFEST, [replay_manifest])
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _build_fresh(
    cv_source: Path, fleurs_tsv: Path, fleurs_audio: Path
) -> tuple[list[CorpusEntry], Path, dict[int, bytes], dict]:
    staging = CACHE / "short_corpus.part"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        cv_entries, replay_pcm = _materialize_common_voice(cv_source, staging)
        fleurs_rows = parse_fleurs_tsv(fleurs_tsv)
        fleurs_entries = _materialize_fleurs(fleurs_audio, fleurs_rows, staging)
        entries = cv_entries + fleurs_entries
        index_bytes = _index_bytes(entries)
        write_atomic(staging / "index.jsonl", [index_bytes])
        index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        if index_sha256 != EXPECTED_INDEX_SHA256:
            raise RuntimeError(
                f"corpus index drift: expected {EXPECTED_INDEX_SHA256}, got {index_sha256}"
            )
        validate_cached_index(staging, index_sha256)
        corpus_dir = CACHE / f"short_corpus-v1-{index_sha256[:16]}"
        manifest = _manifest(entries, index_sha256, corpus_dir)
        if corpus_dir.exists():
            existing = validate_cached_index(corpus_dir, index_sha256)
            if existing != entries:
                raise RuntimeError(f"content-addressed cache collision/corruption: {corpus_dir}")
            shutil.rmtree(staging)
        else:
            staging.replace(corpus_dir)
        return entries, corpus_dir, replay_pcm, manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    _check_build_dependencies()
    CACHE.mkdir(parents=True, exist_ok=True)
    cv_source = fetch_source(CV_PARQUET)
    fleurs_tsv = fetch_source(FLEURS_TSV)
    fleurs_audio = fetch_source(FLEURS_AUDIO)

    cached = _load_cached()
    if cached is None:
        entries, corpus_dir, replay_pcm, manifest = _build_fresh(
            cv_source, fleurs_tsv, fleurs_audio
        )
        mode = "built"
    else:
        entries, corpus_dir = cached
        replay_pcm = _replay_pcm_from_cache(entries, corpus_dir)
        manifest = _manifest(entries, file_sha256(corpus_dir / "index.jsonl"), corpus_dir)
        mode = "cached + verified"

    _install_replay(entries, replay_pcm)
    manifest_bytes = _json_bytes(manifest)
    write_atomic(MANIFEST, [manifest_bytes])
    print(
        f"corpus: {mode}; rows={len(entries)}, index={manifest['cache']['index_sha256']}, "
        f"manifest={MANIFEST.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
