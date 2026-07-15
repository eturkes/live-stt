"""Model-free locks for the pinned M10 short-form corpus builder."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import wave
from pathlib import Path

import pytest

from tests import fetch_real_clips as corpus


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _source(payload: bytes) -> corpus.SourceSpec:
    return corpus.SourceSpec(
        filename="source.bin",
        path="data/source.bin",
        url="https://example.invalid/source.bin",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def test_fetch_source_verifies_fresh_and_cached_paths(tmp_path, monkeypatch):
    payload = b"revision-pinned payload"
    calls = 0

    def open_source(_request, timeout):
        nonlocal calls
        assert timeout == corpus.DOWNLOAD_TIMEOUT_S
        calls += 1
        return _Response(payload)

    monkeypatch.setattr(corpus.urllib.request, "urlopen", open_source)
    spec = _source(payload)
    path = corpus.fetch_source(spec, tmp_path)
    assert path.read_bytes() == payload
    assert calls == 1

    assert corpus.fetch_source(spec, tmp_path) == path
    assert calls == 1


def test_fetch_source_hash_failure_preserves_cached_file(tmp_path, monkeypatch):
    spec = _source(b"expected")
    destination = tmp_path / spec.filename
    destination.write_bytes(b"corrupt cache")
    monkeypatch.setattr(
        corpus.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"wrong download"),
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        corpus.fetch_source(spec, tmp_path)

    assert destination.read_bytes() == b"corrupt cache"
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_atomic_write_failure_preserves_committed_file(tmp_path):
    destination = tmp_path / "manifest.json"
    destination.write_bytes(b"old\n")

    def interrupted():
        yield b"new"
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        corpus.write_atomic(destination, interrupted())

    assert destination.read_bytes() == b"old\n"
    assert not destination.with_name(f"{destination.name}.part").exists()


def _fleurs_line(
    *,
    sentence_id: str = "1822",
    filename: str = "1030930262884292674.wav",
    raw: str = "都市は、再建されました。",
    normalized: str = "都市は 再建されました",
    samples: str = "224640",
    gender: str = "MALE",
) -> str:
    return "\t".join([sentence_id, filename, raw, normalized, "都 市 は | 再 建", samples, gender])


def test_parse_fleurs_tsv_maps_stable_ids_and_rejects_schema_errors(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text(
        _fleurs_line() + "\n" + _fleurs_line(filename="9.wav", gender="FEMALE") + "\n",
        encoding="utf-8",
    )
    rows = corpus.parse_fleurs_tsv(path, expected_rows=2)
    assert [row.corpus_id for row in rows] == [
        "fleurs-ja-test-1030930262884292674",
        "fleurs-ja-test-9",
    ]
    assert rows[0].reference == "都市は、再建されました。"
    assert rows[0].normalized_reference == "都市は再建されました"
    assert rows[0].num_samples == 224640
    assert rows[1].gender == "female"

    bad_lines = [
        _fleurs_line(filename="../escape.wav"),
        _fleurs_line(samples="0"),
        _fleurs_line(gender="UNKNOWN"),
        _fleurs_line(raw="。", normalized=""),
        "too\tfew\tfields",
        _fleurs_line() + "\n" + _fleurs_line(),
    ]
    for bad in bad_lines:
        path.write_text(bad + "\n", encoding="utf-8")
        with pytest.raises(RuntimeError):
            corpus.parse_fleurs_tsv(path)


def test_common_voice_metadata_and_rows_fail_closed():
    corpus._validate_cv_metadata(["audio", "transcription"], corpus.CV_ROWS)
    assert corpus._parse_cv_row(
        {"audio": {"bytes": b"audio"}, "transcription": " 音声です。 "}, 7
    ) == ("音声です。", "音声です", b"audio")

    with pytest.raises(RuntimeError, match="schema"):
        corpus._validate_cv_metadata(["transcription"], corpus.CV_ROWS)
    with pytest.raises(RuntimeError, match="row count"):
        corpus._validate_cv_metadata(["audio", "transcription"], corpus.CV_ROWS - 1)
    invalid = [
        None,
        {"audio": None, "transcription": "音声"},
        {"audio": {"bytes": "not bytes"}, "transcription": "音声"},
        {"audio": {"bytes": b"audio"}, "transcription": "。"},
    ]
    for row in invalid:
        with pytest.raises(RuntimeError):
            corpus._parse_cv_row(row, 0)


def _write_tar(path: Path, members: list[tuple[str, bytes, str]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.type = tarfile.SYMTYPE
                info.linkname = "target"
                archive.addfile(info)


def test_fleurs_archive_requires_exact_safe_regular_file_set(tmp_path):
    archive = tmp_path / "audio.tar.gz"
    _write_tar(
        archive,
        [
            ("audio/test", b"", "dir"),
            ("audio/test/a.wav", b"A", "file"),
            ("audio/test/b.wav", b"B", "file"),
        ],
    )
    with tarfile.open(archive, "r:gz") as opened:
        members = corpus.validate_fleurs_archive(opened, {"a.wav", "b.wav"})
    assert sorted(members) == ["a.wav", "b.wav"]

    invalid = [
        [("audio/test/a.wav", b"A", "file")],
        [
            ("audio/test/a.wav", b"A", "file"),
            ("audio/test/b.wav", b"B", "file"),
            ("audio/test/c.wav", b"C", "file"),
        ],
        [
            ("audio/test/a.wav", b"A", "file"),
            ("elsewhere/a.wav", b"A", "file"),
            ("audio/test/b.wav", b"B", "file"),
        ],
        [
            ("../a.wav", b"A", "file"),
            ("audio/test/b.wav", b"B", "file"),
        ],
        [
            ("audio/test/a.wav", b"", "link"),
            ("audio/test/b.wav", b"B", "file"),
        ],
    ]
    for i, members in enumerate(invalid):
        candidate = tmp_path / f"bad-{i}.tar.gz"
        _write_tar(candidate, members)
        with tarfile.open(candidate, "r:gz") as opened, pytest.raises(RuntimeError):
            corpus.validate_fleurs_archive(opened, {"a.wav", "b.wav"})


def test_summarize_counts_duplicates_gender_duration_and_fingerprints():
    entries: list[corpus.SummaryEntry] = [
        {
            "corpus_id": "fleurs-ja-test-a",
            "source": "fleurs",
            "duration_samples": 4 * corpus.SAMPLE_RATE,
            "gender": "male",
            "normalized_reference": "同文",
            "pcm_sha256": "1" * 64,
        },
        {
            "corpus_id": "fleurs-ja-test-b",
            "source": "fleurs",
            "duration_samples": 7 * corpus.SAMPLE_RATE,
            "gender": "female",
            "normalized_reference": "同文",
            "pcm_sha256": "1" * 64,
        },
        {
            "corpus_id": "fleurs-ja-test-c",
            "source": "fleurs",
            "duration_samples": 25 * corpus.SAMPLE_RATE,
            "gender": "male",
            "normalized_reference": "別文",
            "pcm_sha256": "2" * 64,
        },
    ]
    summary = corpus.summarize(entries)

    assert summary["rows"] == 3
    assert summary["references"] == {
        "duplicate_groups": 1,
        "duplicate_rows": 1,
        "max_recordings_per_reference": 2,
        "recordings_per_reference": {"1": 1, "2": 1},
        "unique": 2,
    }
    assert summary["pcm"] == {"duplicate_groups": 1, "duplicate_rows": 1, "unique": 2}
    assert summary["gender"] == {
        "female": {"audio_seconds": 7.0, "rows": 1},
        "male": {"audio_seconds": 29.0, "rows": 2},
    }
    assert summary["duration_seconds"]["buckets"] == {
        "0-5": 1,
        "5-10": 1,
        "10-20": 0,
        "20+": 1,
    }
    assert len(summary["fingerprints"]["references_sha256"]) == 64
    assert len(summary["fingerprints"]["pcm_sha256"]) == 64


def test_validate_cached_index_rejects_pcm_corruption(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    wav = corpus_dir / "clip.wav"
    pcm = b"\0\0" * 16
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(corpus.SAMPLE_RATE)
        output.writeframes(pcm)
    entry = {
        "corpus_id": "cv8-ja-test-000000",
        "source": "common_voice_8",
        "source_row": 0,
        "wav": "clip.wav",
        "reference": "音声",
        "normalized_reference": "音声",
        "duration_samples": 16,
        "duration_seconds": 0.001,
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "gender": None,
    }
    index = corpus_dir / "index.jsonl"
    index.write_text(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    assert corpus.validate_cached_index(corpus_dir, corpus.file_sha256(index)) == [entry]
    wav.write_bytes(b"not a wav")
    with pytest.raises(RuntimeError, match="cached PCM"):
        corpus.validate_cached_index(corpus_dir, corpus.file_sha256(index))
