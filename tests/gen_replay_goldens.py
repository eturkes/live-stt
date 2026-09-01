#!/usr/bin/env python3
"""Regenerate tests/replay_goldens.json from the real STT pipeline.

Replays the cached bench clips through replay.py for every engine and snapshots
the *deterministic* outputs only: per-engine, per-clip segment count + each
segment's (start, n, text). Decode latency / RTF are excluded — they are
CPU-variable.

Run when the pipeline's segmentation/decode behavior intentionally changes
(e.g. VAD tuning, engine swap), then review the JSON diff before committing:

    uv run python tests/gen_replay_goldens.py

MATRIX, not a product: the sherpa engines take every clip because their decode is
CPU-deterministic and costs milliseconds, while whisper takes ONE clip because each
of its rows is an accelerator-bound decode. `long` is that clip on measured VAC
depth -- it drives 13 StreamingProcessor.process calls and commits 51 characters
through LocalAgreement-2, where the shorter `greet` commits 0 and exercises the
speech-end flush alone.

A cell the local machine cannot run (absent weights, absent WAV, absent
accelerator) carries its committed row forward instead of vanishing, because a
whisper regeneration on a box with no NPU would otherwise silently delete the
committed whisper row. Rows outside the matrix are dropped. Both are reported.

For whisper, source the accel farm and clear PYTHONPATH first (see
.agent/memory.md); without the farm the NPU aborts on a missing compiler loader.

The bench WAVs live under the deny-listed spike/backends/cache/; this script's
runtime reads are unaffected (L-016), and the path is constructed here rather
than passed on the command line. Requires models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import replay  # noqa: E402  (after sys.path injection)
from live_stt import ASR_DEVICE, WHISPER_ENGINES, check_device, check_models  # noqa: E402

CACHE = ROOT / "spike" / "backends" / "cache"
GOLDENS = ROOT / "tests" / "replay_goldens.json"
# The one whisper clip, chosen on measured VAC depth (see the module docstring).
WHISPER_CLIPS = ["long"]

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


def matrix() -> dict[str, list[str]]:
    """Engine -> the clip ids it snapshots. Explicit, never a Cartesian product.

    Validated here rather than at use: an unknown or repeated id would otherwise
    surface as a bare KeyError deep in the run, or as a silently overwritten row.
    """
    every = [cid for cid, _ref, _purpose in all_clips()]
    cells = {"k2v2": every, "parakeet": every, "whisper": WHISPER_CLIPS}
    for engine, cids in cells.items():
        if unknown := [c for c in cids if c not in every]:
            raise ValueError(f"matrix[{engine}]: unknown clip id(s) {unknown}; known: {every}")
        if len(set(cids)) != len(cids):
            raise ValueError(f"matrix[{engine}]: duplicate clip id(s) in {cids}")
    return cells


def main():
    if not CACHE.exists():
        print(f"cache dir absent: {CACHE}", file=sys.stderr)
        sys.exit(1)
    meta = {cid: (ja_ref, purpose) for cid, ja_ref, purpose in all_clips()}
    cells = matrix()
    prior = json.loads(GOLDENS.read_text(encoding="utf-8")) if GOLDENS.exists() else {}
    for engine, rows in prior.items():
        for cid in rows:
            if cid not in cells.get(engine, ()):
                print(f"drop {engine}/{cid}: outside the matrix", file=sys.stderr)
    out: dict[str, dict] = {}
    for engine, cids in cells.items():
        blocked = check_models(engine) or check_device(engine)
        rows = {}
        for cid in cids:
            wav = CACHE / f"{cid}.wav"
            why = blocked or (None if wav.exists() else f"{wav.name} absent")
            if why:
                # Carry the committed row forward: a whisper regeneration on a box
                # with no NPU must not silently delete the committed whisper row.
                kept = prior.get(engine, {}).get(cid)
                verb = "omit" if kept is None else "keep"
                print(f"{verb} {engine}/{cid}: {why.splitlines()[0]}", file=sys.stderr)
                if kept is not None:
                    rows[cid] = kept
                continue
            rep = replay.replay_wav(wav, engine)
            ja_ref, purpose = meta[cid]
            row: dict[str, object] = {"ja_ref": ja_ref, "purpose": purpose}
            if engine in WHISPER_ENGINES:
                row["device"] = ASR_DEVICE
            row["n_segments"] = rep["n_segments"]
            row["segments"] = [
                {"start": s["start"], "n": s["n"], "text": s["text"]} for s in rep["segments"]
            ]
            rows[cid] = row
            texts = " | ".join(s["text"] for s in rep["segments"] if s["text"])
            print(
                f"{engine}/{cid}: {rep['n_segments']} seg, rtf {rep['overall_rtf']:.3f} :: {texts}"
            )
        if rows:
            out[engine] = rows
    GOLDENS.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n_rows = sum(len(r) for r in out.values())
    print(f"wrote {GOLDENS.relative_to(ROOT)} ({len(out)} engines, {n_rows} rows)")


if __name__ == "__main__":
    main()
