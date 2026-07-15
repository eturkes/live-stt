#!/usr/bin/env python3
"""Fetch real Japanese speech clips from Common Voice 8.0 into the replay corpus.

T5.3 provenance/regeneration tool. The replay corpus was Gemini-TTS synthetic;
this adds *real* acoustics (real speakers, real mics, real prosody/onset through
silero VAD) without a microphone — the agent cannot capture audio (L-004), so
the clips are fetched from the web instead.

Source: japanese-asr/ja_asr.common_voice_8_0 — an ungated Parquet mirror of
Mozilla Common Voice 8.0 Japanese (CC0 audio; crowd-sourced mic recordings).
The exact 144 MiB test Parquet is revision- and SHA-pinned, cached outside git,
then only the selected row group is decoded. Embedded MP3 cells are decoded
with soundfile (libsndfile decodes MP3 natively here — no ffmpeg). Each clip is
downmixed to mono and resampled to 16 kHz via the project's own
live_stt.resample for pipeline fidelity, then written as PCM16 WAV.

The WAVs land in the gitignored spike/backends/cache/. The (id, ja_ref, purpose)
manifest is written to tests/real_clips.json, which gen_replay_goldens.py merges
into its clip list. Revision, payload hash, and row indices make the source
selection reproducible:

    uv run --with soundfile --with pyarrow python tests/fetch_real_clips.py
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import wave
from pathlib import Path
from typing import TypedDict

import numpy as np
import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]  (transient tool dep)
import soundfile as sf  # pyright: ignore[reportMissingImports]  (via uv run --with soundfile)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live_stt import SAMPLE_RATE, resample  # noqa: E402  (after sys.path injection)

CACHE = ROOT / "spike" / "backends" / "cache"
MANIFEST = ROOT / "tests" / "real_clips.json"

DATASET = "japanese-asr/ja_asr.common_voice_8_0"
REVISION = "bf8819e8d9a5feb51b0c718686bd20ea67a3c729"
SPLIT = "test"
PARQUET_PATH = "data/test-00000-of-00001.parquet"
PARQUET_SHA256 = "44a9141bc16cfa34877955fb39003ad34d3b730417a05c9eb50d8e90ba3ec40a"
PARQUET_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{PARQUET_PATH}"
PARQUET_FILENAME = "common_voice_8_test.parquet"

# Silence framing: lead so the first onset is clean; tail > VAD_MIN_SILENCE_S so
# the final segment closes via VAD (not the EOF flush) for stable boundaries.
LEAD_S, TAIL_S = 0.3, 0.6

# (clip_id, row_idx, purpose) — one real utterance each.
SINGLES = [
    ("cv_short", 65, "real CV: short utterance -> 1 segment"),
    ("cv_med", 10, "real CV: medium sentence -> 1 segment"),
    ("cv_long", 2, "real CV: longer sentence with a name -> 1 segment"),
    ("cv_kana", 15, "real CV: katakana loanword (フィリピン) -> proper-noun acoustics"),
    ("cv_xlong", 4, "real CV: long katakana-heavy sentence -> sustained real decode"),
]
# (clip_id, [row_idx...], gap_s, purpose) — independent utterances joined by real
# silence (not a continuous render, so no TTS decode-collapse artifact; D-010).
CONCATS = [
    ("cv_multi", [11, 81, 93], 0.7, "real CV x3, 0.7 s gaps (> VAD_MIN_SILENCE_S) -> 3 segments"),
    ("cv_paused", [46, 55], 2.0, "real CV x2, 2.0 s gap -> 2 segments"),
]


class SourceRow(TypedDict):
    transcription: str
    audio: bytes


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        while chunk := src.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_parquet() -> Path:
    """Return the verified revision-pinned source, downloading it once."""
    path = CACHE / PARQUET_FILENAME
    if path.is_file() and file_sha256(path) == PARQUET_SHA256:
        print(f"source: cached + verified {path.name}")
        return path

    part = path.with_suffix(f"{path.suffix}.part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(PARQUET_URL, headers={"User-Agent": "live-stt-corpus/1"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=90) as response, part.open("wb") as dst:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
        actual = digest.hexdigest()
        if actual != PARQUET_SHA256:
            raise RuntimeError(f"source SHA-256 mismatch: expected {PARQUET_SHA256}, got {actual}")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    print(f"source: downloaded + verified {path.name}")
    return path


def read_rows(path: Path, indices: set[int]) -> dict[int, SourceRow]:
    """Read selected global row offsets without materializing the full table."""
    parquet = pq.ParquetFile(path)
    out: dict[int, SourceRow] = {}
    remaining = set(indices)
    start = 0
    for group_index in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(group_index).num_rows
        end = start + count
        selected = sorted(index for index in remaining if start <= index < end)
        if selected:
            rows = parquet.read_row_group(
                group_index, columns=["audio", "transcription"]
            ).to_pylist()
            for index in selected:
                row = rows[index - start]
                transcription = row["transcription"]
                raw = row["audio"]["bytes"]
                if not isinstance(transcription, str) or not isinstance(raw, bytes):
                    raise RuntimeError(f"unexpected source row schema at index {index}")
                out[index] = {"transcription": transcription.strip(), "audio": raw}
                remaining.remove(index)
        if not remaining:
            break
        start = end
    if remaining:
        raise RuntimeError(f"rows not found: {sorted(remaining)}")
    return out


def load_clip(raw: bytes) -> np.ndarray:
    """Decode one embedded MP3 to float32 mono @ SAMPLE_RATE (pipeline rep)."""
    data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        data = resample(data, sr, SAMPLE_RATE)
    return np.ascontiguousarray(data, dtype=np.float32)


def write_wav(cid: str, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(CACHE / f"{cid}.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def sil(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    needed = {idx for _, idx, _ in SINGLES}
    for _, idxs, _, _ in CONCATS:
        needed.update(idxs)
    rows = read_rows(fetch_parquet(), needed)
    audio = {idx: load_clip(rows[idx]["audio"]) for idx in needed}

    manifest: dict[str, dict] = {}
    lead, tail = sil(LEAD_S), sil(TAIL_S)

    for cid, idx, purpose in SINGLES:
        write_wav(cid, np.concatenate([lead, audio[idx], tail]))
        manifest[cid] = {
            "ja_ref": rows[idx]["transcription"],
            "purpose": purpose,
            "source": f"{DATASET}@{REVISION}#{SPLIT}[{idx}]",
        }
        print(f"{cid}: idx={idx}  {rows[idx]['transcription']}")

    for cid, idxs, gap_s, purpose in CONCATS:
        gap = sil(gap_s)
        parts = [lead]
        for j, idx in enumerate(idxs):
            if j:
                parts.append(gap)
            parts.append(audio[idx])
        parts.append(tail)
        write_wav(cid, np.concatenate(parts))
        manifest[cid] = {
            "ja_ref": " ".join(rows[idx]["transcription"] for idx in idxs),
            "purpose": purpose,
            "source": f"{DATASET}@{REVISION}#{SPLIT}{idxs} gap={gap_s}s",
        }
        print(f"{cid}: idxs={idxs} gap={gap_s}s")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} clips to {CACHE.relative_to(ROOT)}/ and {MANIFEST.name}")


if __name__ == "__main__":
    main()
