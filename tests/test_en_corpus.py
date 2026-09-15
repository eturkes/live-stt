"""Locks for the pinned FLEURS English evidence corpus."""

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


def test_en_sources_are_revision_pinned_and_verified_fresh_and_cached(tmp_path, monkeypatch):
    expected = [
        (
            corpus.EN_FLEURS_TSV,
            "data/en_us/test.tsv",
            367_864,
            "74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca",
        ),
        (
            corpus.EN_FLEURS_AUDIO,
            "data/en_us/audio/test.tar.gz",
            289_851_356,
            "d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356",
        ),
    ]
    for pinned, source_path, size, sha256 in expected:
        assert pinned.path == source_path
        assert pinned.size == size
        assert pinned.sha256 == sha256
        assert corpus.FLEURS_REVISION in pinned.url
        assert f"/{source_path}" in pinned.url

        payload = f"fresh probe for {source_path}".encode()
        probe = corpus.SourceSpec(
            filename=pinned.filename,
            path=pinned.path,
            url=pinned.url,
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        )
        calls = 0

        def open_source(request, timeout, *, expected_url=pinned.url, body=payload):
            nonlocal calls
            assert request.full_url == expected_url
            assert timeout == corpus.DOWNLOAD_TIMEOUT_S
            calls += 1
            return _Response(body)

        monkeypatch.setattr(corpus.urllib.request, "urlopen", open_source)
        path = corpus.fetch_source(probe, tmp_path)
        assert path.read_bytes() == payload
        assert calls == 1
        assert corpus.fetch_source(probe, tmp_path) == path
        assert calls == 1


def _fleurs_line(
    *,
    sentence_id: str = "1980",
    filename: str = "12741024238657315067.wav",
    reference: str = "A full 20 percent comes from the Amazon.",
    transcription: str = "a full 20% comes from the amazon",
    samples: str = "160000",
    gender: str = "MALE",
) -> str:
    return "\t".join(
        [sentence_id, filename, reference, transcription, "t r a f f i c", samples, gender]
    )


def test_en_tsv_schema_count_and_duplicate_identity_are_locked(tmp_path):
    assert corpus.EN_FLEURS_ROWS == 647
    path = tmp_path / "test.tsv"
    path.write_text(
        _fleurs_line() + "\n" + _fleurs_line(filename="9.wav", gender="FEMALE") + "\n",
        encoding="utf-8",
    )
    rows = corpus.parse_fleurs_tsv(path, config=corpus.EN_FLEURS_CONFIG, expected_rows=2)
    assert [row.corpus_id for row in rows] == [
        "fleurs-en-test-12741024238657315067",
        "fleurs-en-test-9",
    ]
    assert rows[0].normalized_reference == "afull20percentcomesfromtheamazon"
    assert rows[1].gender == "female"

    path.write_text(_fleurs_line() + "\n" + _fleurs_line() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate FLEURS audio identity"):
        corpus.parse_fleurs_tsv(path, config=corpus.EN_FLEURS_CONFIG, expected_rows=2)
    with pytest.raises(RuntimeError, match="unsupported FLEURS config"):
        corpus.parse_fleurs_tsv(path, config="en_unknown", expected_rows=2)


def _wav_bytes(frames: int, *, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(corpus.SAMPLE_RATE)
        output.writeframes(b"\x01\x00" * frames * channels)
    return buffer.getvalue()


def _write_tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_en_canonical_pcm_count_and_duration_are_locked(tmp_path, monkeypatch):
    assert corpus.EN_FLEURS_ROWS == 647
    assert corpus.EN_FLEURS_SAMPLES == 102_206_400
    tsv = tmp_path / "test.tsv"
    tsv.write_text(
        _fleurs_line(samples="160")
        + "\n"
        + _fleurs_line(filename="9.wav", samples="320", gender="FEMALE")
        + "\n",
        encoding="utf-8",
    )
    rows = corpus.parse_fleurs_tsv(tsv, config=corpus.EN_FLEURS_CONFIG, expected_rows=2)
    archive = tmp_path / "audio.tar.gz"
    _write_tar(
        archive,
        [
            (f"audio/test/{rows[0].filename}", _wav_bytes(160, channels=2)),
            (f"audio/test/{rows[1].filename}", _wav_bytes(320)),
        ],
    )

    def decode(raw, *, context, expected_rate, expected_samples):
        assert raw.startswith(b"RIFF")
        assert context.startswith("fleurs-en-test-")
        assert expected_rate == corpus.SAMPLE_RATE
        assert expected_samples in {160, 320}
        return corpus.np.full(expected_samples, 0.25, dtype=corpus.np.float32)

    monkeypatch.setattr(corpus, "_decode_audio", decode)
    entries, corpus_dir, index_sha256 = corpus._build_fleurs_cache(
        archive,
        rows,
        cache=tmp_path / "cache",
        directory_prefix=corpus.EN_CACHE_PREFIX,
        expected_rows=2,
        expected_samples=480,
    )
    assert corpus_dir.name == f"{corpus.EN_CACHE_PREFIX}-{index_sha256[:16]}"
    assert corpus.file_sha256(corpus_dir / "index.jsonl") == index_sha256
    assert sum(entry["duration_samples"] for entry in entries) == 480
    for entry in entries:
        with wave.open(str(corpus_dir / entry["wav"]), "rb") as source:
            assert source.getnchannels() == 1
            assert source.getsampwidth() == 2
            assert source.getframerate() == corpus.SAMPLE_RATE
            assert source.getnframes() == entry["duration_samples"]

    for expected_rows, expected_samples in [(3, 480), (2, 481)]:
        with pytest.raises(RuntimeError, match="incomplete FLEURS corpus"):
            corpus._build_fleurs_cache(
                archive,
                rows,
                cache=tmp_path / "bad-cache",
                directory_prefix=corpus.EN_CACHE_PREFIX,
                expected_rows=expected_rows,
                expected_samples=expected_samples,
            )


def test_en_compact_manifest_and_index_fingerprint_are_locked():
    manifest_bytes = corpus.EN_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected_index = "1b13a64fdc119f6c4c760a6750c80c6211d11b61e47f39327542a5842165814a"
    assert corpus.EN_EXPECTED_INDEX_SHA256 == expected_index
    assert len(manifest_bytes) < 5_000
    assert b'"wav"' not in manifest_bytes
    assert manifest["cache"] == {
        "directory": "spike/backends/cache/en_clips-v1-1b13a64fdc119f6c",
        "index": "index.jsonl",
        "index_sha256": expected_index,
        "rows": 647,
    }
    source = manifest["sources"]["fleurs"]
    assert source["source_identity"] == "en_us/test[0:647]"
    assert source["statistics"]["duration_seconds"]["total"] == 6387.9
    assert source["statistics"]["references"] == {
        "duplicate_groups": 297,
        "duplicate_rows": 297,
        "max_recordings_per_reference": 2,
        "recordings_per_reference": {"1": 53, "2": 297},
        "unique": 350,
    }
    assert source["statistics"]["pcm"] == {
        "duplicate_groups": 0,
        "duplicate_rows": 0,
        "unique": 647,
    }


def test_en_pcm_corruption_and_archive_escapes_fail_closed(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    pcm = b"\0\0" * 16
    wav = corpus_dir / "clip.wav"
    corpus._write_pcm_wav(wav, pcm)
    entry = {
        "corpus_id": "fleurs-en-test-1",
        "source": "fleurs",
        "source_row": 0,
        "wav": "clip.wav",
        "reference": "English evidence.",
        "normalized_reference": "englishevidence",
        "duration_samples": 16,
        "duration_seconds": 0.001,
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "gender": "female",
    }
    index = corpus_dir / "index.jsonl"
    index.write_text(json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")
    assert corpus.validate_cached_index(corpus_dir, corpus.file_sha256(index)) == [entry]
    wav.write_bytes(b"corrupt PCM")
    with pytest.raises(RuntimeError, match="cached PCM"):
        corpus.validate_cached_index(corpus_dir, corpus.file_sha256(index))

    archive = tmp_path / "escape.tar.gz"
    _write_tar(archive, [("../escape.wav", b"payload")])
    with (
        tarfile.open(archive, "r:gz") as opened,
        pytest.raises(RuntimeError, match="unsafe FLEURS archive path"),
    ):
        corpus.validate_fleurs_archive(opened, {"escape.wav"})


def test_japanese_corpus_identity_is_unchanged():
    assert corpus.EXPECTED_INDEX_SHA256 == (
        "98e0d8a40fbc2d6e819ddd8db22fd23c2d7f050ac2da5773ac207a1bd0a14d36"
    )
    assert corpus.file_sha256(corpus.REPLAY_MANIFEST) == (
        "8ed02f84609ab05ba179d2eeb805f7c802c4262401a987f6a1dfbd4559e043dc"
    )
    assert corpus.file_sha256(corpus.MANIFEST) == (
        "9f7ca9eb6e180dd97ae682de0ec7f55925c6bf25acac58d937e8cb45a844cfc2"
    )
