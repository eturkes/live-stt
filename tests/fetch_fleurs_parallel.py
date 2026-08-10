#!/usr/bin/env python3
"""Build pinned English references parallel to the Japanese FLEURS corpus.

FLEURS sentence IDs are FLoRes sentence indices shared across languages. This
script verifies the revision-pinned English TSV, joins its sentence IDs to the
pinned Japanese short-corpus index, and installs detailed reference text in a
gitignored content-addressed cache. Git receives only provenance, statistics,
and fingerprints in ``tests/fleurs_parallel.json``.

Run from the repository root:
    uv run python tests/fetch_fleurs_parallel.py
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, TypedDict, cast

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "spike" / "backends" / "cache"
MANIFEST = ROOT / "tests" / "fleurs_parallel.json"
SCHEMA_VERSION = 1
DOWNLOAD_TIMEOUT_S = 90
BUILD_DEPENDENCIES = {"python": platform.python_version()}

FLEURS_DATASET = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_CONFIG = "en_us"
FLEURS_SPLIT = "test"
EXPECTED_EN_ROWS = 647
EXPECTED_EN_SENTENCE_IDS = 350

JA_INDEX_SHA256 = "98e0d8a40fbc2d6e819ddd8db22fd23c2d7f050ac2da5773ac207a1bd0a14d36"
JA_INDEX = CACHE / f"short_corpus-v1-{JA_INDEX_SHA256[:16]}" / "index.jsonl"
EXPECTED_JA_INDEX_ROWS = 5_133
EXPECTED_JA_FLEURS_ROWS = 650
EXPECTED_JA_SENTENCE_IDS = 321
EXPECTED_REFERENCES_INDEX_SHA256 = (
    "19f0e7066cc246b53f17c1172081c7b8b759bd31efdbee70e45d660ce44335c3"
)


@dataclass(frozen=True)
class SourceSpec:
    filename: str
    path: str
    url: str
    sha256: str
    size: int


EN_TSV = SourceSpec(
    filename="fleurs_en_us_test.tsv",
    path="data/en_us/test.tsv",
    url=(
        f"https://huggingface.co/datasets/{FLEURS_DATASET}/resolve/"
        f"{FLEURS_REVISION}/data/en_us/test.tsv"
    ),
    sha256="74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca",
    size=367_864,
)


@dataclass(frozen=True)
class EnglishRow:
    source_row: int
    sentence_id: int
    filename: str
    reference: str
    transcription: str
    normalized_reference: str
    num_samples: int
    gender: str


@dataclass(frozen=True)
class JapaneseReferences:
    references: dict[int, str]
    recordings: dict[int, int]
    rows_scanned: int


@dataclass(frozen=True)
class CollapsedEnglish:
    reference: str
    normalized_reference: str
    recordings: int


class ReferenceEntry(TypedDict):
    sentence_id: int
    en_reference: str
    en_normalized: str
    ja_reference: str
    recordings: int


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
    request = urllib.request.Request(spec.url, headers={"User-Agent": "live-stt-fleurs/1"})
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


def normalize_english(text: str) -> str:
    """Lowercase, remove Unicode punctuation, then collapse Unicode whitespace."""
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def parse_english_tsv(
    path: Path,
    *,
    expected_rows: int = EXPECTED_EN_ROWS,
    expected_sentence_ids: int = EXPECTED_EN_SENTENCE_IDS,
) -> list[EnglishRow]:
    rows: list[EnglishRow] = []
    seen_filenames: set[str] = set()
    sentence_ids: set[int] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read FLEURS English TSV: {exc}") from exc

    for source_row, line in enumerate(lines):
        fields = line.split("\t")
        if len(fields) != 7:
            raise RuntimeError(f"unexpected FLEURS English TSV schema at row {source_row}")
        sentence_raw, filename, reference, transcription, characters, samples_raw, gender = fields
        try:
            sentence_id = int(sentence_raw)
            num_samples = int(samples_raw)
        except ValueError as exc:
            raise RuntimeError(f"invalid FLEURS English integer at row {source_row}") from exc
        name = PurePosixPath(filename)
        if (
            sentence_id < 0
            or num_samples <= 0
            or name.name != filename
            or name.suffix.lower() != ".wav"
            or not name.stem.isdecimal()
            or not characters.strip()
        ):
            raise RuntimeError(f"invalid FLEURS English identity at row {source_row}")
        if gender not in {"MALE", "FEMALE", "OTHER"}:
            raise RuntimeError(f"invalid FLEURS English gender at row {source_row}: {gender!r}")
        reference = reference.strip()
        transcription = transcription.strip()
        normalized = normalize_english(reference)
        if not reference or not transcription or not normalized:
            raise RuntimeError(f"empty FLEURS English reference at row {source_row}")
        if filename in seen_filenames:
            raise RuntimeError(f"duplicate FLEURS English audio identity at row {source_row}")
        seen_filenames.add(filename)
        sentence_ids.add(sentence_id)
        rows.append(
            EnglishRow(
                source_row=source_row,
                sentence_id=sentence_id,
                filename=filename,
                reference=reference,
                transcription=transcription,
                normalized_reference=normalized,
                num_samples=num_samples,
                gender=gender.lower(),
            )
        )

    if len(rows) != expected_rows:
        raise RuntimeError(f"unexpected FLEURS English row count: {len(rows)} != {expected_rows}")
    if len(sentence_ids) != expected_sentence_ids:
        raise RuntimeError(
            "unexpected FLEURS English sentence-ID count: "
            f"{len(sentence_ids)} != {expected_sentence_ids}"
        )
    return rows


def collapse_english(rows: Sequence[EnglishRow]) -> dict[int, CollapsedEnglish]:
    grouped: dict[int, list[EnglishRow]] = defaultdict(list)
    for row in rows:
        grouped[row.sentence_id].append(row)

    collapsed: dict[int, CollapsedEnglish] = {}
    for sentence_id, recordings in grouped.items():
        references = {row.reference for row in recordings}
        normalized = {row.normalized_reference for row in recordings}
        if len(references) != 1:
            raise RuntimeError(
                f"FLEURS English sentence {sentence_id} has {len(references)} reference texts"
            )
        if len(normalized) != 1:
            raise RuntimeError(
                f"FLEURS English sentence {sentence_id} has inconsistent normalized text"
            )
        collapsed[sentence_id] = CollapsedEnglish(
            reference=next(iter(references)),
            normalized_reference=next(iter(normalized)),
            recordings=len(recordings),
        )
    return collapsed


def load_japanese_references(path: Path = JA_INDEX) -> JapaneseReferences:
    if not path.is_file():
        raise RuntimeError(f"pinned Japanese corpus index is absent: {path}")
    actual_sha256 = file_sha256(path)
    if actual_sha256 != JA_INDEX_SHA256:
        raise RuntimeError(
            f"Japanese corpus index SHA-256 mismatch: expected {JA_INDEX_SHA256}, "
            f"got {actual_sha256}"
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read Japanese corpus index: {exc}") from exc
    if len(lines) != EXPECTED_JA_INDEX_ROWS:
        raise RuntimeError(
            f"unexpected Japanese corpus index row count: {len(lines)} != {EXPECTED_JA_INDEX_ROWS}"
        )

    references: dict[int, set[str]] = defaultdict(set)
    recordings: Counter[int] = Counter()
    seen_ids: set[str] = set()
    fleurs_rows = 0
    for source_row, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid Japanese corpus index row {source_row}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError(f"invalid Japanese corpus index schema at row {source_row}")
        corpus_id = raw.get("corpus_id")
        source = raw.get("source")
        if not isinstance(corpus_id, str) or corpus_id in seen_ids:
            raise RuntimeError(f"duplicate/invalid Japanese corpus ID at row {source_row}")
        if source not in {"common_voice_8", "fleurs"}:
            raise RuntimeError(f"invalid Japanese corpus source at row {source_row}")
        seen_ids.add(corpus_id)
        if source != "fleurs":
            continue
        sentence_id = raw.get("sentence_id")
        reference = raw.get("reference")
        if type(sentence_id) is not int or sentence_id < 0:
            raise RuntimeError(f"invalid Japanese FLEURS sentence ID at row {source_row}")
        if not isinstance(reference, str) or not reference.strip():
            raise RuntimeError(f"invalid Japanese FLEURS reference at row {source_row}")
        fleurs_rows += 1
        references[sentence_id].add(reference)
        recordings[sentence_id] += 1

    if fleurs_rows != EXPECTED_JA_FLEURS_ROWS:
        raise RuntimeError(
            f"unexpected Japanese FLEURS row count: {fleurs_rows} != {EXPECTED_JA_FLEURS_ROWS}"
        )
    if len(references) != EXPECTED_JA_SENTENCE_IDS:
        raise RuntimeError(
            "unexpected Japanese FLEURS sentence-ID count: "
            f"{len(references)} != {EXPECTED_JA_SENTENCE_IDS}"
        )
    disagreements = [sentence_id for sentence_id, texts in references.items() if len(texts) != 1]
    if disagreements:
        raise RuntimeError(
            "Japanese FLEURS recordings disagree for sentence IDs: "
            + ", ".join(str(sentence_id) for sentence_id in sorted(disagreements))
        )
    return JapaneseReferences(
        references={sentence_id: next(iter(texts)) for sentence_id, texts in references.items()},
        recordings=dict(recordings),
        rows_scanned=fleurs_rows,
    )


def _distribution(values: Iterable[int]) -> dict[str, int]:
    return {str(value): count for value, count in sorted(Counter(values).items())}


def build_references(
    english_rows: Sequence[EnglishRow], japanese: JapaneseReferences
) -> tuple[list[ReferenceEntry], dict]:
    english = collapse_english(english_rows)
    ja_sentence_ids = set(japanese.references)
    missing = sorted(ja_sentence_ids - english.keys())
    if missing:
        raise RuntimeError(
            f"English FLEURS coverage incomplete: {len(ja_sentence_ids) - len(missing)}/"
            f"{len(ja_sentence_ids)}; missing sentence IDs: "
            + ", ".join(str(sentence_id) for sentence_id in missing)
        )

    entries: list[ReferenceEntry] = []
    for sentence_id in sorted(ja_sentence_ids):
        parallel = english[sentence_id]
        entries.append(
            {
                "sentence_id": sentence_id,
                "en_reference": parallel.reference,
                "en_normalized": parallel.normalized_reference,
                "ja_reference": japanese.references[sentence_id],
                "recordings": parallel.recordings,
            }
        )
    if len(entries) != EXPECTED_JA_SENTENCE_IDS:
        raise RuntimeError(f"unexpected joined reference count: {len(entries)}")

    payload_recordings = [entry.recordings for entry in english.values()]
    joined_recordings = [entry["recordings"] for entry in entries]
    statistics = {
        "ja_rows_scanned": japanese.rows_scanned,
        "ja_unique_sentence_ids": len(ja_sentence_ids),
        "covered_ja_sentence_ids": len(entries),
        "missing_ja_sentence_ids": len(missing),
        "en_rows_in_payload": len(english_rows),
        "en_unique_sentence_ids_in_payload": len(english),
        "en_rows_joined": sum(joined_recordings),
        "recordings_per_sentence": {
            "ja_corpus": _distribution(japanese.recordings.values()),
            "en_payload": _distribution(payload_recordings),
            "en_joined": _distribution(joined_recordings),
        },
    }
    return entries, statistics


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


def _index_bytes(entries: Sequence[ReferenceEntry]) -> bytes:
    return b"".join(_json_bytes(entry, compact=True) for entry in entries)


def _field_fingerprint(
    entries: Sequence[ReferenceEntry], field: Literal["en_reference", "en_normalized"]
) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry[field].encode("utf-8"))
    return digest.hexdigest()


def validate_cached_references(
    corpus_dir: Path,
    expected_sha256: str,
    expected_entries: Sequence[ReferenceEntry] | None = None,
) -> list[ReferenceEntry]:
    index = corpus_dir / "references.jsonl"
    if not index.is_file() or file_sha256(index) != expected_sha256:
        raise RuntimeError("cached FLEURS parallel index fingerprint mismatch")
    if {path.name for path in corpus_dir.iterdir()} != {"references.jsonl"}:
        raise RuntimeError("cached FLEURS parallel directory has unexpected entries")
    try:
        lines = index.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read cached FLEURS parallel index: {exc}") from exc

    entries: list[ReferenceEntry] = []
    previous_sentence_id = -1
    required = {"sentence_id", "en_reference", "en_normalized", "ja_reference", "recordings"}
    for source_row, line in enumerate(lines):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid cached FLEURS parallel row {source_row}") from exc
        if not isinstance(raw, dict) or set(raw) != required:
            raise RuntimeError(f"invalid cached FLEURS parallel schema at row {source_row}")
        sentence_id = raw["sentence_id"]
        recordings = raw["recordings"]
        en_reference = raw["en_reference"]
        en_normalized = raw["en_normalized"]
        ja_reference = raw["ja_reference"]
        if (
            type(sentence_id) is not int
            or sentence_id <= previous_sentence_id
            or type(recordings) is not int
            or recordings <= 0
            or not isinstance(en_reference, str)
            or not en_reference
            or not isinstance(en_normalized, str)
            or not en_normalized
            or normalize_english(en_reference) != en_normalized
            or not isinstance(ja_reference, str)
            or not ja_reference
        ):
            raise RuntimeError(f"invalid cached FLEURS parallel values at row {source_row}")
        entries.append(cast(ReferenceEntry, raw))
        previous_sentence_id = sentence_id

    if len(entries) != EXPECTED_JA_SENTENCE_IDS:
        raise RuntimeError(
            f"unexpected cached FLEURS parallel row count: {len(entries)} != "
            f"{EXPECTED_JA_SENTENCE_IDS}"
        )
    if expected_entries is not None and list(expected_entries) != entries:
        raise RuntimeError("content-addressed FLEURS parallel cache collision/corruption")
    return entries


def install_references(entries: Sequence[ReferenceEntry]) -> tuple[Path, str, str]:
    index_bytes = _index_bytes(entries)
    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    if index_sha256 != EXPECTED_REFERENCES_INDEX_SHA256:
        raise RuntimeError(
            "FLEURS parallel index drift: expected "
            f"{EXPECTED_REFERENCES_INDEX_SHA256}, got {index_sha256}"
        )
    corpus_dir = CACHE / f"fleurs_parallel-v1-{index_sha256[:16]}"
    staging = CACHE / "fleurs_parallel.part"
    if corpus_dir.exists():
        validate_cached_references(corpus_dir, index_sha256, entries)
        shutil.rmtree(staging, ignore_errors=True)
        return corpus_dir, index_sha256, "cached + verified"

    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        write_atomic(staging / "references.jsonl", [index_bytes])
        validate_cached_references(staging, index_sha256, entries)
        if corpus_dir.exists():
            validate_cached_references(corpus_dir, index_sha256, entries)
            shutil.rmtree(staging)
            return corpus_dir, index_sha256, "cached + verified"
        staging.replace(corpus_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return corpus_dir, index_sha256, "built"


def _payload(spec: SourceSpec) -> dict:
    return {
        "path": spec.path,
        "sha256": spec.sha256,
        "size_bytes": spec.size,
        "url": spec.url,
    }


def build_manifest(
    english_rows: Sequence[EnglishRow],
    entries: Sequence[ReferenceEntry],
    statistics: dict,
    corpus_dir: Path,
    index_sha256: str,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "builder": {
            "dependencies": BUILD_DEPENDENCIES,
            "script": "tests/fetch_fleurs_parallel.py",
        },
        "cache": {
            "directory": corpus_dir.relative_to(ROOT).as_posix(),
            "index": "references.jsonl",
            "index_sha256": index_sha256,
            "rows": len(entries),
        },
        "inputs": {
            "japanese_corpus_index": {
                "fleurs_rows": EXPECTED_JA_FLEURS_ROWS,
                "fleurs_unique_sentence_ids": EXPECTED_JA_SENTENCE_IDS,
                "path": JA_INDEX.relative_to(ROOT).as_posix(),
                "rows": EXPECTED_JA_INDEX_ROWS,
                "sha256": JA_INDEX_SHA256,
                "source_manifest": "tests/short_corpus.json",
            }
        },
        "join": {
            **statistics,
            "fingerprints": {
                "contract": (
                    "SHA-256 over UTF-8 field values concatenated without separators in "
                    "ascending sentence_id order"
                ),
                "en_normalized_sha256": _field_fingerprint(entries, "en_normalized"),
                "en_reference_sha256": _field_fingerprint(entries, "en_reference"),
            },
        },
        "normalization": {
            "field": "en_normalized",
            "lowercase": "Python str.lower",
            "order": ["lowercase", "punctuation removal", "whitespace collapse"],
            "punctuation": ("remove every Unicode code point whose general category starts with P"),
            "whitespace": (
                "split on Unicode whitespace, join tokens with one U+0020, and strip edges"
            ),
        },
        "sources": {
            "fleurs_en_us": {
                "attribution": (
                    "FLEURS authors and recording contributors; cite Conneau et al. (2022)"
                ),
                "citation": "https://arxiv.org/abs/2205.12446",
                "dataset": FLEURS_DATASET,
                "dataset_card": (
                    f"https://huggingface.co/datasets/{FLEURS_DATASET}/blob/"
                    f"{FLEURS_REVISION}/README.md"
                ),
                "license": "CC-BY-4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "limitations": [
                    "read-speech translations are not an unseen production-speech proxy",
                    "valid English paraphrases can score poorly under exact string metrics",
                    "training or evaluation overlap with downstream models is unknown",
                    (
                        "reference text stays outside git in the ignored content-addressed "
                        "cache; this manifest commits only provenance, statistics, and fingerprints"
                    ),
                ],
                "payloads": [_payload(EN_TSV)],
                "revision": FLEURS_REVISION,
                "source_identity": f"{FLEURS_CONFIG}/{FLEURS_SPLIT}[0:{len(english_rows)}]",
                "statistics": {
                    "rows": len(english_rows),
                    "unique_sentence_ids": statistics["en_unique_sentence_ids_in_payload"],
                    "recordings_per_sentence": statistics["recordings_per_sentence"]["en_payload"],
                },
            }
        },
    }


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    source = fetch_source(EN_TSV)
    english_rows = parse_english_tsv(source)
    japanese = load_japanese_references()
    entries, statistics = build_references(english_rows, japanese)
    corpus_dir, index_sha256, mode = install_references(entries)
    manifest = build_manifest(
        english_rows,
        entries,
        statistics,
        corpus_dir,
        index_sha256,
    )
    write_atomic(MANIFEST, [_json_bytes(manifest)])
    print(
        f"parallel references: {mode}; rows={len(entries)}, index={index_sha256}, "
        f"manifest={MANIFEST.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
