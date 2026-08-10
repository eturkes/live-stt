"""Model-free locks for pinned FLEURS English parallel references."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Literal, cast

import pytest

from tests import fetch_fleurs_parallel as parallel

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "tests" / "fleurs_parallel.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _source(payload: bytes) -> parallel.SourceSpec:
    return parallel.SourceSpec(
        filename="source.tsv",
        path="data/source.tsv",
        url="https://example.invalid/source.tsv",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def _english_line(
    *,
    sentence_id: str = "1661",
    filename: str = "100.wav",
    reference: str = "A clear sentence.",
    transcription: str = "a clear sentence",
    characters: str = "a | clear | sentence",
    samples: str = "16000",
    gender: str = "MALE",
) -> str:
    return "\t".join(
        [
            sentence_id,
            filename,
            reference,
            transcription,
            characters,
            samples,
            gender,
        ]
    )


def _reference_entries() -> list[parallel.ReferenceEntry]:
    return [
        {
            "sentence_id": 1661,
            "en_reference": "A clear sentence.",
            "en_normalized": "a clear sentence",
            "ja_reference": "明瞭な文です。",
            "recordings": 2,
        }
    ]


def _distribution(raw: dict[str, int]) -> dict[int, int]:
    return {int(recordings): sentence_ids for recordings, sentence_ids in raw.items()}


def test_fetch_source_verifies_fresh_and_cached_paths(tmp_path, monkeypatch):
    payload = b"revision-pinned payload"
    calls = 0

    def open_source(_request, timeout):
        nonlocal calls
        assert timeout == parallel.DOWNLOAD_TIMEOUT_S
        calls += 1
        return _Response(payload)

    monkeypatch.setattr(parallel.urllib.request, "urlopen", open_source)
    spec = _source(payload)
    path = parallel.fetch_source(spec, tmp_path)
    assert path.read_bytes() == payload
    assert calls == 1

    assert parallel.fetch_source(spec, tmp_path) == path
    assert calls == 1


def test_fetch_source_failure_preserves_cached_file(tmp_path, monkeypatch):
    spec = _source(b"expected")
    destination = tmp_path / spec.filename
    destination.write_bytes(b"corrupt cache")
    monkeypatch.setattr(
        parallel.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"wrong download"),
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        parallel.fetch_source(spec, tmp_path)

    assert destination.read_bytes() == b"corrupt cache"
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_normalize_english_contract():
    assert parallel.normalize_english("  Hello,\tWORLD—again!  ") == "hello worldagain"
    assert parallel.normalize_english("A 20% Share") == "a 20 share"


def test_parse_and_collapse_english_tsv_fail_closed(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text(
        _english_line() + "\n" + _english_line(filename="101.wav") + "\n",
        encoding="utf-8",
    )
    rows = parallel.parse_english_tsv(path, expected_rows=2, expected_sentence_ids=1)
    collapsed = parallel.collapse_english(rows)
    assert collapsed[1661].reference == "A clear sentence."
    assert collapsed[1661].normalized_reference == "a clear sentence"
    assert collapsed[1661].recordings == 2

    path.write_text(
        _english_line(reference="Same meaning.", filename="100.wav")
        + "\n"
        + _english_line(reference="Same meaning!", filename="101.wav")
        + "\n",
        encoding="utf-8",
    )
    rows = parallel.parse_english_tsv(path, expected_rows=2, expected_sentence_ids=1)
    with pytest.raises(RuntimeError, match="2 reference texts"):
        parallel.collapse_english(rows)

    path.write_text("\t".join(_english_line().split("\t")[:6]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="TSV schema"):
        parallel.parse_english_tsv(path, expected_rows=1, expected_sentence_ids=1)


def test_japanese_index_hash_and_parallel_reference_agreement(tmp_path, monkeypatch):
    rows = [
        {
            "corpus_id": "fleurs-ja-test-1",
            "source": "fleurs",
            "sentence_id": 1661,
            "reference": "同じ文です。",
        },
        {
            "corpus_id": "fleurs-ja-test-2",
            "source": "fleurs",
            "sentence_id": 1661,
            "reference": "同じ文です。",
        },
    ]
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    path = tmp_path / "index.jsonl"
    path.write_bytes(payload)
    monkeypatch.setattr(parallel, "JA_INDEX_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(parallel, "EXPECTED_JA_INDEX_ROWS", 2)
    monkeypatch.setattr(parallel, "EXPECTED_JA_FLEURS_ROWS", 2)
    monkeypatch.setattr(parallel, "EXPECTED_JA_SENTENCE_IDS", 1)

    references = parallel.load_japanese_references(path)
    assert references.references == {1661: "同じ文です。"}
    assert references.recordings == {1661: 2}

    path.write_bytes(payload + b"\n")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        parallel.load_japanese_references(path)


def test_cached_install_is_validated_without_overwrite(tmp_path, monkeypatch):
    entries = _reference_entries()
    index_sha256 = hashlib.sha256(parallel._index_bytes(entries)).hexdigest()
    monkeypatch.setattr(parallel, "CACHE", tmp_path)
    monkeypatch.setattr(parallel, "EXPECTED_JA_SENTENCE_IDS", 1)
    monkeypatch.setattr(parallel, "EXPECTED_REFERENCES_INDEX_SHA256", index_sha256)

    corpus_dir, actual_sha256, mode = parallel.install_references(entries)
    assert actual_sha256 == index_sha256
    assert mode == "built"
    original = (corpus_dir / "references.jsonl").read_bytes()
    stale_staging = tmp_path / "fleurs_parallel.part"
    stale_staging.mkdir()
    (stale_staging / "references.jsonl").write_bytes(b"partial")

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("valid cache must not be rewritten")

    monkeypatch.setattr(parallel, "write_atomic", unexpected_write)
    cached_dir, cached_sha256, cached_mode = parallel.install_references(entries)
    assert (cached_dir, cached_sha256, cached_mode) == (
        corpus_dir,
        index_sha256,
        "cached + verified",
    )
    assert (corpus_dir / "references.jsonl").read_bytes() == original
    assert not stale_staging.exists()


def test_manifest_locks_provenance_normalization_and_join_geometry():
    assert set(MANIFEST) == {
        "builder",
        "cache",
        "inputs",
        "join",
        "normalization",
        "schema_version",
        "sources",
    }
    assert MANIFEST["schema_version"] == 1
    assert MANIFEST["builder"] == {
        "dependencies": parallel.BUILD_DEPENDENCIES,
        "script": "tests/fetch_fleurs_parallel.py",
    }

    cache = MANIFEST["cache"]
    assert cache == {
        "directory": f"spike/backends/cache/fleurs_parallel-v1-{cache['index_sha256'][:16]}",
        "index": "references.jsonl",
        "index_sha256": parallel.EXPECTED_REFERENCES_INDEX_SHA256,
        "rows": 321,
    }
    assert MANIFEST["inputs"]["japanese_corpus_index"] == {
        "fleurs_rows": 650,
        "fleurs_unique_sentence_ids": 321,
        "path": "spike/backends/cache/short_corpus-v1-98e0d8a40fbc2d6e/index.jsonl",
        "rows": 5_133,
        "sha256": parallel.JA_INDEX_SHA256,
        "source_manifest": "tests/short_corpus.json",
    }

    source = MANIFEST["sources"]["fleurs_en_us"]
    assert source["dataset"] == "google/fleurs"
    assert source["revision"] == parallel.FLEURS_REVISION
    assert source["source_identity"] == "en_us/test[0:647]"
    assert source["dataset_card"].endswith(f"/{parallel.FLEURS_REVISION}/README.md")
    assert source["license"] == "CC-BY-4.0"
    assert source["license_url"] == "https://creativecommons.org/licenses/by/4.0/"
    assert source["attribution"] == (
        "FLEURS authors and recording contributors; cite Conneau et al. (2022)"
    )
    assert source["citation"] == "https://arxiv.org/abs/2205.12446"
    assert source["payloads"] == [
        {
            "path": "data/en_us/test.tsv",
            "sha256": parallel.EN_TSV.sha256,
            "size_bytes": parallel.EN_TSV.size,
            "url": parallel.EN_TSV.url,
        }
    ]
    assert any("outside git" in limitation for limitation in source["limitations"])
    assert source["statistics"] == {
        "recordings_per_sentence": {"1": 53, "2": 297},
        "rows": 647,
        "unique_sentence_ids": 350,
    }

    normalization = MANIFEST["normalization"]
    assert normalization["field"] == "en_normalized"
    assert normalization["order"] == [
        "lowercase",
        "punctuation removal",
        "whitespace collapse",
    ]
    assert normalization["lowercase"] == "Python str.lower"
    assert "category starts with P" in normalization["punctuation"]
    assert "one U+0020" in normalization["whitespace"]

    join = MANIFEST["join"]
    assert join["ja_rows_scanned"] == 650
    assert join["ja_unique_sentence_ids"] == 321
    assert join["covered_ja_sentence_ids"] == cache["rows"] == 321
    assert join["missing_ja_sentence_ids"] == 0
    assert join["en_rows_in_payload"] == 647
    assert join["en_unique_sentence_ids_in_payload"] == 350
    assert join["en_rows_joined"] == 592

    geometry = {
        "ja_corpus": (321, 650),
        "en_payload": (350, 647),
        "en_joined": (321, 592),
    }
    for name, (expected_sentence_ids, expected_rows) in geometry.items():
        distribution = _distribution(join["recordings_per_sentence"][name])
        assert sum(distribution.values()) == expected_sentence_ids
        assert (
            sum(recordings * count for recordings, count in distribution.items()) == expected_rows
        )

    fingerprints = join["fingerprints"]
    assert set(fingerprints) == {
        "contract",
        "en_normalized_sha256",
        "en_reference_sha256",
    }
    assert fingerprints["contract"].endswith("ascending sentence_id order")
    assert all(len(fingerprints[field]) == 64 for field in fingerprints if field != "contract")


def _fingerprint(
    rows: list[dict[str, object]], field: Literal["en_reference", "en_normalized"]
) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(cast(str, row[field]).encode("utf-8"))
    return digest.hexdigest()


def test_cached_references_match_manifest_and_japanese_index_when_present():
    cache_relative = PurePosixPath(MANIFEST["cache"]["directory"])
    cache_dir = ROOT.joinpath(*cache_relative.parts)
    reference_index = cache_dir / MANIFEST["cache"]["index"]
    ja_relative = PurePosixPath(MANIFEST["inputs"]["japanese_corpus_index"]["path"])
    ja_index = ROOT.joinpath(*ja_relative.parts)
    if not reference_index.is_file() or not ja_index.is_file():
        pytest.skip("ignored FLEURS parallel or Japanese corpus cache absent")

    index_bytes = reference_index.read_bytes()
    assert hashlib.sha256(index_bytes).hexdigest() == MANIFEST["cache"]["index_sha256"]
    rows = cast(
        list[dict[str, object]],
        [json.loads(line) for line in index_bytes.decode("utf-8").splitlines()],
    )
    assert len(rows) == MANIFEST["cache"]["rows"] == 321
    assert all(
        set(row) == {"sentence_id", "en_reference", "en_normalized", "ja_reference", "recordings"}
        for row in rows
    )
    sentence_ids = [cast(int, row["sentence_id"]) for row in rows]
    assert all(type(sentence_id) is int for sentence_id in sentence_ids)
    assert sentence_ids == sorted(sentence_ids)
    assert len(sentence_ids) == len(set(sentence_ids)) == 321
    assert all(
        parallel.normalize_english(cast(str, row["en_reference"])) == row["en_normalized"]
        for row in rows
    )
    assert all(type(row["recordings"]) is int and cast(int, row["recordings"]) > 0 for row in rows)

    canonical = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
    assert canonical == index_bytes
    fingerprints = MANIFEST["join"]["fingerprints"]
    assert _fingerprint(rows, "en_reference") == fingerprints["en_reference_sha256"]
    assert _fingerprint(rows, "en_normalized") == fingerprints["en_normalized_sha256"]
    assert _distribution(MANIFEST["join"]["recordings_per_sentence"]["en_joined"]) == dict(
        sorted(Counter(cast(int, row["recordings"]) for row in rows).items())
    )

    ja_bytes = ja_index.read_bytes()
    assert (
        hashlib.sha256(ja_bytes).hexdigest()
        == MANIFEST["inputs"]["japanese_corpus_index"]["sha256"]
    )
    ja_references: dict[int, set[str]] = defaultdict(set)
    ja_rows = 0
    for line in ja_bytes.decode("utf-8").splitlines():
        row = json.loads(line)
        if row["source"] != "fleurs":
            continue
        ja_rows += 1
        ja_references[row["sentence_id"]].add(row["reference"])
    assert ja_rows == 650
    assert set(sentence_ids) == set(ja_references)
    assert all(len(references) == 1 for references in ja_references.values())
    assert all(
        row["ja_reference"] == next(iter(ja_references[cast(int, row["sentence_id"])]))
        for row in rows
    )
