#!/usr/bin/env python3
"""Deterministic WAV replay through the exact live-stt STT pipeline.

Feeds a WAV file through `live_stt.worker` — the same VAD + RingBuffer pre-pad
re-slice + sherpa-onnx decode loop the live mic uses — and reports per-segment
segmentation, decode latency, and transcript. The translation leg is omitted
(translator=None): this exercises and regression-tests the local STT half only.

This is a dev/regression tool, not part of the shipped app, so it is not a
console-script entry point. Run it directly:

    uv run python replay.py path/to.wav [--engine k2v2|parakeet] [--json]

Determinism: for a given WAV + engine, segment boundaries (start, n) and
transcript text are reproducible (silero VAD + sherpa offline decode are
deterministic on CPU). Decode latency / RTF are CPU-variable — reported for
inspection, never used as a pass/fail signal (see tests/test_replay.py).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np

from live_stt import (
    ENGINE_DIRS,
    SAMPLE_RATE,
    State,
    check_models,
    load_recognizer,
    make_vad,
    resample,
    worker,
)

# Model load is seconds; the regression test replays several clips per process,
# so cache one recognizer per engine. The VAD is stateful and must stay fresh
# per replay, so it is NOT cached.
_RECS: dict[str, object] = {}


def _recognizer(engine: str):
    rec = _RECS.get(engine)
    if rec is None:
        rec = load_recognizer(engine)
        _RECS[engine] = rec
    return rec


def load_wav_f32_16k(path: Path) -> np.ndarray:
    """Load any PCM WAV as float32 mono at SAMPLE_RATE, via the live resampler.

    Mirrors the mic path's representation (float32 mono @ 16 kHz) so replayed
    audio enters `worker` exactly as live audio does. Channels are averaged to
    mono; rate conversion reuses live_stt.resample for pipeline fidelity.
    """
    with wave.open(str(path), "rb") as w:
        nchan = w.getnchannels()
        sr = w.getframerate()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sw * 8}-bit (need 16 or 32)")
    if nchan > 1:
        data = data.reshape(-1, nchan).mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        data = resample(data, sr, SAMPLE_RATE)
    return np.ascontiguousarray(data, dtype=np.float32)


async def _run(samples: np.ndarray, engine: str) -> list[dict]:
    """Drive the real worker() over `samples` and collect per-segment rows."""
    rec = _recognizer(engine)
    vad, window = make_vad()
    audio_q: asyncio.Queue = asyncio.Queue()
    # Keep replay blocks within live AudioQueue's 2 s headroom. VAD framing is
    # unchanged, while bounded blocks let the feeder copy completed segments
    # from its 60 s ring before later long-form audio can evict them.
    for start in range(0, len(samples), SAMPLE_RATE):
        audio_q.put_nowait(samples[start : start + SAMPLE_RATE])
    audio_q.put_nowait(None)
    rows: list[dict] = []
    state = State()

    def on_segment(start, n, seg_len, decode_s, text):
        rows.append(
            {"start": start, "n": n, "seg_len": seg_len, "decode_s": decode_s, "text": text}
        )

    await worker(rec, vad, window, audio_q, state, None, None, on_segment)
    # worker() intentionally converts a stage exception into session shutdown
    # for the live app. Replay is an evaluator, so turn that signal back into a
    # hard failure rather than committing a partial/empty transcript as golden.
    if state.stopping:
        raise RuntimeError("STT worker failed during replay")
    return rows


def build_report(engine: str, wav: str, samples: np.ndarray, rows: list[dict]) -> dict:
    audio_s = len(samples) / SAMPLE_RATE
    total_decode_s = sum(r["decode_s"] for r in rows)
    segments = []
    for i, r in enumerate(rows, 1):
        dur_s = r["n"] / SAMPLE_RATE
        segments.append(
            {
                "idx": i,
                "start": r["start"],
                "n": r["n"],
                "seg_len": r["seg_len"],
                "start_s": r["start"] / SAMPLE_RATE,
                "end_s": (r["start"] + r["n"]) / SAMPLE_RATE,
                "dur_s": dur_s,
                "decode_s": r["decode_s"],
                "rtf": (r["decode_s"] / dur_s) if dur_s > 0 else 0.0,
                "text": r["text"],
            }
        )
    return {
        "wav": wav,
        "engine": engine,
        "audio_s": audio_s,
        "n_segments": len(rows),
        "n_nonempty": sum(1 for r in rows if r["text"]),
        "total_decode_s": total_decode_s,
        "overall_rtf": (total_decode_s / audio_s) if audio_s > 0 else 0.0,
        "segments": segments,
    }


def replay_wav(path, engine: str = "k2v2") -> dict:
    """Load a WAV and replay it through the pipeline. Returns the report dict.

    The real worker() prints its live `JA n:` lines via emit_line; capture that
    stdout so replay's only output is its own report (keeps --json valid).
    """
    samples = load_wav_f32_16k(Path(path))
    with contextlib.redirect_stdout(io.StringIO()):
        rows = asyncio.run(_run(samples, engine))
    return build_report(engine, str(path), samples, rows)


def render(report: dict) -> str:
    lines = [
        f"replay: {report['wav']}  engine={report['engine']}  "
        f"audio={report['audio_s']:.2f}s  "
        f"segments={report['n_segments']} ({report['n_nonempty']} non-empty)  "
        f"decode={report['total_decode_s']:.3f}s  rtf={report['overall_rtf']:.3f}",
        "",
        f"  {'#':>2}  {'start':>8}  {'end':>8}  {'dur':>7}  {'decode':>7}  {'rtf':>6}  text",
    ]
    for s in report["segments"]:
        lines.append(
            f"  {s['idx']:>2}  {s['start_s']:>7.3f}s  {s['end_s']:>7.3f}s  "
            f"{s['dur_s']:>6.3f}s  {s['decode_s']:>6.3f}s  {s['rtf']:>6.3f}  "
            f"{s['text'] or '(empty)'}"
        )
    if not report["segments"]:
        lines.append("  (no speech segments detected)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("wav", help="Path to a WAV file (any rate/channels; PCM 16/32-bit).")
    ap.add_argument(
        "--engine",
        choices=sorted(ENGINE_DIRS),
        default="k2v2",
        help="Local STT engine (default: k2v2; matches live-stt default).",
    )
    ap.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = ap.parse_args()

    err = check_models(args.engine)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
    path = Path(args.wav)
    if not path.exists():
        print(f"Error: no such WAV: {path}", file=sys.stderr)
        sys.exit(1)

    report = replay_wav(path, args.engine)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
