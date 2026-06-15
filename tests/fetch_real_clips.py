#!/usr/bin/env python3
"""Fetch real Japanese speech clips from Common Voice 8.0 into the replay corpus.

T5.3 provenance/regeneration tool. The replay corpus was Gemini-TTS synthetic;
this adds *real* acoustics (real speakers, real mics, real prosody/onset through
silero VAD) without a microphone — the agent cannot capture audio (L-004), but
CLAUDE.md grants network access, so the clips are fetched from the web instead.

Source: japanese-asr/ja_asr.common_voice_8_0 — an ungated, viewer-enabled
Parquet mirror of Mozilla Common Voice 8.0 Japanese (CC0 audio; crowd-sourced
mic recordings). Rows come from the HF datasets-server rows API (a few samples,
no multi-GB download); audio cells are MP3, decoded with soundfile (libsndfile
decodes MP3 natively here — no ffmpeg). Each clip is downmixed to mono and
resampled to 16 kHz via the project's own live_stt.resample for pipeline
fidelity, then written as PCM16 WAV.

The WAVs land in the deny-listed spike/backends/cache/; the path is constructed
here and never passed on a command line (L-016). The (id, ja_ref, purpose)
manifest is written to tests/real_clips.json, which gen_replay_goldens.py merges
into its clip list. Row indices + the dataset revision are pinned for
reproducibility; re-run to refresh:

    uv run --with soundfile python tests/fetch_real_clips.py
"""

from __future__ import annotations

import io
import json
import sys
import urllib.parse
import urllib.request
import wave
from pathlib import Path

import numpy as np
import soundfile as sf  # pyright: ignore[reportMissingImports]  (via uv run --with soundfile)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live_stt import SAMPLE_RATE, resample  # noqa: E402  (after sys.path injection)

CACHE = ROOT / "spike" / "backends" / "cache"
MANIFEST = ROOT / "tests" / "real_clips.json"

DATASET = "japanese-asr/ja_asr.common_voice_8_0"
# Parquet revision the rows API served when these indices were pinned (provenance;
# row order is stable within a revision). Re-pin indices if the mirror updates.
REVISION = "bf8819e8d9a5feb51b0c718686bd20ea67a3c729"
CONFIG, SPLIT = "default", "test"
ROWS_API = "https://datasets-server.huggingface.co/rows"

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


def fetch_rows(indices: set[int]) -> dict[int, dict]:
    """Return {row_idx: {"transcription", "url"}} for the requested rows.

    The rows API caps length at 100, so fetch the 100-row pages that cover the
    requested indices and keep only those rows.
    """
    out: dict[int, dict] = {}
    pages = sorted({i // 100 for i in indices})
    for page in pages:
        q = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": page * 100,
                "length": 100,
            }
        )
        with urllib.request.urlopen(f"{ROWS_API}?{q}", timeout=90) as r:
            doc = json.load(r)
        for row in doc["rows"]:
            idx = row["row_idx"]
            if idx in indices:
                out[idx] = {
                    "transcription": row["row"]["transcription"].strip(),
                    "url": row["row"]["audio"][0]["src"],
                }
    missing = indices - out.keys()
    if missing:
        raise RuntimeError(f"rows not found: {sorted(missing)}")
    return out


def load_clip(url: str) -> np.ndarray:
    """Download one MP3 and return float32 mono @ SAMPLE_RATE (pipeline rep)."""
    with urllib.request.urlopen(url, timeout=90) as r:
        raw = r.read()
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
    rows = fetch_rows(needed)
    audio = {idx: load_clip(rows[idx]["url"]) for idx in needed}

    manifest: dict[str, dict] = {}
    lead, tail = sil(LEAD_S), sil(TAIL_S)

    for cid, idx, purpose in SINGLES:
        write_wav(cid, np.concatenate([lead, audio[idx], tail]))
        manifest[cid] = {
            "ja_ref": rows[idx]["transcription"],
            "purpose": purpose,
            "source": f"{DATASET}#{SPLIT}[{idx}]",
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
            "source": f"{DATASET}#{SPLIT}{idxs} gap={gap_s}s",
        }
        print(f"{cid}: idxs={idxs} gap={gap_s}s")

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} clips to {CACHE.relative_to(ROOT)}/ and {MANIFEST.name}")


if __name__ == "__main__":
    main()
