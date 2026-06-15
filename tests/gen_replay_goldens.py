#!/usr/bin/env python3
"""Regenerate tests/replay_goldens.json from the real STT pipeline.

Replays the cached bench clips through replay.py and snapshots the
*deterministic* outputs only: per-clip segment count + each segment's
(start, n, text). Decode latency / RTF are excluded — they are CPU-variable.

Run when the pipeline's segmentation/decode behavior intentionally changes
(e.g. VAD tuning, engine swap), then review the JSON diff before committing:

    uv run python tests/gen_replay_goldens.py            # default engine k2v2

The bench WAVs live under the deny-listed spike/backends/cache/; this script's
*runtime* reads are unaffected (D-008 amendment), and the deny-listed path is
constructed here rather than passed on the command line. Requires models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import replay  # noqa: E402  (after sys.path injection)

CACHE = ROOT / "spike" / "backends" / "cache"
GOLDENS = ROOT / "tests" / "replay_goldens.json"
ENGINE = "k2v2"

# Ported from the retired spike/backends/scenarios.py — (id, ja_ref, purpose).
# `purpose` documents the expected segmentation; the asserted values come from
# the pipeline, not these refs.
CLIPS = [
    ("greet", "こんにちは。", "single sentence -> 1 segment"),
    ("short", "今日はライブAPIのテストをしています。", "single sentence -> 1 segment"),
    (
        "medium",
        "こんにちは、今日はライブAPIのテストをしています。よろしくお願いします。",
        "3 sentences, 0.7 s silences (> VAD_MIN_SILENCE_S) -> 3 segments",
    ),
    (
        "long",
        "このプロジェクトはリアルタイムの日本語音声認識ツールです。"
        "マイクから音声を取り込み、ジェミニAPIに送って、"
        "日本語の文字起こしと英語の翻訳を同時に表示します。",
        "2 sentences, 0.7 s silence -> 2 segments",
    ),
    ("paused", "最初の文です。\n二つ目の文です。", "2.0 s gap -> 2 segments"),
]


def main():
    if not CACHE.exists():
        print(f"cache dir absent: {CACHE}", file=sys.stderr)
        sys.exit(1)
    out: dict[str, dict] = {}
    for cid, ja_ref, purpose in CLIPS:
        wav = CACHE / f"{cid}.wav"
        if not wav.exists():
            print(f"skip {cid}: {wav.name} absent", file=sys.stderr)
            continue
        rep = replay.replay_wav(wav, ENGINE)
        out[cid] = {
            "ja_ref": ja_ref,
            "purpose": purpose,
            "engine": ENGINE,
            "n_segments": rep["n_segments"],
            "segments": [
                {"start": s["start"], "n": s["n"], "text": s["text"]} for s in rep["segments"]
            ],
        }
        texts = " | ".join(s["text"] for s in rep["segments"] if s["text"])
        print(f"{cid}: {rep['n_segments']} seg, rtf {rep['overall_rtf']:.3f} :: {texts}")
    GOLDENS.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {GOLDENS.relative_to(ROOT)} ({len(out)} clips)")


if __name__ == "__main__":
    main()
