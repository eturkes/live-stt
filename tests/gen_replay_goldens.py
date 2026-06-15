#!/usr/bin/env python3
"""Regenerate tests/replay_goldens.json from the real STT pipeline.

Replays the cached bench clips through replay.py for every engine and snapshots
the *deterministic* outputs only: per-engine, per-clip segment count + each
segment's (start, n, text). Decode latency / RTF are excluded — they are
CPU-variable.

Run when the pipeline's segmentation/decode behavior intentionally changes
(e.g. VAD tuning, engine swap), then review the JSON diff before committing:

    uv run python tests/gen_replay_goldens.py            # all engines (k2v2, parakeet)

An engine whose weights are absent is skipped with a warning (the others still
regenerate); rerun with that engine's models present to refresh its goldens.

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
from live_stt import check_models  # noqa: E402

CACHE = ROOT / "spike" / "backends" / "cache"
GOLDENS = ROOT / "tests" / "replay_goldens.json"
ENGINES = ["k2v2", "parakeet"]

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


MANIFEST = ROOT / "tests" / "real_clips.json"


def all_clips():
    """Inline synthetic CLIPS plus the real fetched corpus (tests/real_clips.json).

    Real clips are added by tests/fetch_real_clips.py (T5.3); the manifest is the
    single source for their (id, ja_ref, purpose). Absent manifest -> synthetic only.
    """
    clips = list(CLIPS)
    if MANIFEST.exists():
        for cid, m in json.loads(MANIFEST.read_text(encoding="utf-8")).items():
            clips.append((cid, m["ja_ref"], m["purpose"]))
    return clips


def main():
    if not CACHE.exists():
        print(f"cache dir absent: {CACHE}", file=sys.stderr)
        sys.exit(1)
    out: dict[str, dict] = {}
    for engine in ENGINES:
        err = check_models(engine)
        if err:
            print(f"skip engine {engine}: {err.splitlines()[0]}", file=sys.stderr)
            continue
        out[engine] = {}
        for cid, ja_ref, purpose in all_clips():
            wav = CACHE / f"{cid}.wav"
            if not wav.exists():
                print(f"skip {engine}/{cid}: {wav.name} absent", file=sys.stderr)
                continue
            rep = replay.replay_wav(wav, engine)
            out[engine][cid] = {
                "ja_ref": ja_ref,
                "purpose": purpose,
                "n_segments": rep["n_segments"],
                "segments": [
                    {"start": s["start"], "n": s["n"], "text": s["text"]} for s in rep["segments"]
                ],
            }
            texts = " | ".join(s["text"] for s in rep["segments"] if s["text"])
            print(
                f"{engine}/{cid}: {rep['n_segments']} seg, "
                f"rtf {rep['overall_rtf']:.3f} :: {texts}"
            )
    GOLDENS.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {GOLDENS.relative_to(ROOT)} ({len(out)} engines)")


if __name__ == "__main__":
    main()
