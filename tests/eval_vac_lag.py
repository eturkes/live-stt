#!/usr/bin/env python3
"""Per-character caption lag on the shipped VAC path, from the committed trace.

D-016 headlines a 2.483 s median lag, but the evaluator that produced it was
deleted and predates the `on_update` seam, so the number had no producer. This
re-derives it from `vac_decode_trace.json` alone: no model, no accelerator, no
audio, well under a second.

Lag is answered per CHARACTER, not per commit. A commit carries several
characters spanning an audio interval, and the reader waited longest for the
first of them, so charging every character the same emit-minus-commit-end figure
understates the head of every commit.

    end     = commit_audio_s, the audio time the committed prefix reaches
    start   = the audio time the PREVIOUS commit reached
    at_i    = start + (end - start) * (i + 0.5) / len(text)
    lag_i   = emit_s - at_i

`emit_s` runs on the virtual audio clock M11.4 paced the backpressure arm on:
`now = max(now, buffer_end_s) + decode_s`. A decode cannot start before its
audio exists, and the caption appears when the decode returns.

Two rules the trace's shape forces, both load-bearing:

- A FINAL update ends at the utterance end (`buffer_end_s`), not at its recorded
  `commit_audio_s`. `live_stt.update()` appends `processor.finish()` to the
  commit after `process()` returned that timestamp, so the flushed tail runs past
  it to the end of the buffer.
- Never derive lag from final updates ALONE. That is the measurement VAC exists
  to beat: it collapses every early in-speech commit into one utterance-close
  event and reports the VAD policy's latency instead of the streaming policy's.

An update that commits nothing does not move `start`: the committed endpoint is
where committed TEXT reaches, not where the last decode ran.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TRACE = ROOT / "tests" / "vac_decode_trace.json"


def clip_lags(clip: dict[str, Any]) -> tuple[list[float], dict[str, int]]:
    """Per-character lags for one clip, with the counts that qualify them."""
    lags: list[float] = []
    counts = {"updates": 0, "commits": 0, "unplaced": 0, "non_monotone": 0, "finals": 0}
    now = 0.0
    start: float | None = None
    for row in clip["series"]:
        counts["updates"] += 1
        now = max(now, row["buffer_end_s"]) + row["decode_s"]
        if start is None:  # first update of an utterance: text begins at its audio start
            start = row["buffer_end_s"] - row["buffer_s"]
        text = row["commit"]
        end = row["buffer_end_s"] if row["final"] else row["commit_audio_s"]
        if text:
            counts["commits"] += 1
            if end is None:
                # No spans to interpolate in, so the commit cannot be placed on
                # the audio clock at all. Counted, never guessed at.
                counts["unplaced"] += 1
            else:
                if end < start:
                    counts["non_monotone"] += 1
                span = max(end, start) - start
                lags += [now - (start + span * (i + 0.5) / len(text)) for i in range(len(text))]
                start = max(end, start)
        if row["final"]:
            counts["finals"] += 1
            start = None
    return lags, counts


def report(trace: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "source": str(TRACE.relative_to(ROOT)),
        "model": trace["model"],
        "requested_device": trace["requested_device"],
        "clips": {},
    }
    for name, clip in trace["clips"].items():
        lags, counts = clip_lags(clip)
        out["clips"][name] = {
            "audio_s": clip["audio_s"],
            "utterances": clip["utterances"],
            "characters": len(lags),
            "median_lag_s": round(statistics.median(lags), 3) if lags else None,
            "mean_lag_s": round(statistics.fmean(lags), 3) if lags else None,
            "max_lag_s": round(max(lags), 3) if lags else None,
            "min_lag_s": round(min(lags), 3) if lags else None,
            **counts,
        }
    return out


def render(out: dict[str, Any]) -> str:
    lines = [f"vac lag: {out['source']}  {out['model']} on {out['requested_device']}", ""]
    for name, row in out["clips"].items():
        lines.append(
            f"  {name}: audio={row['audio_s']:.3f}s utterances={row['utterances']} "
            f"chars={row['characters']} median={row['median_lag_s']:.3f}s "
            f"mean={row['mean_lag_s']:.3f}s max={row['max_lag_s']:.3f}s "
            f"min={row['min_lag_s']:.3f}s"
        )
        lines.append(
            f"      updates={row['updates']} commits={row['commits']} finals={row['finals']} "
            f"unplaced={row['unplaced']} non_monotone={row['non_monotone']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args()

    if not args.trace.is_file():
        print(f"error: no trace at {args.trace}", file=sys.stderr)
        sys.exit(1)
    out = report(json.loads(args.trace.read_text(encoding="utf-8")))
    print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else render(out))


if __name__ == "__main__":
    main()
