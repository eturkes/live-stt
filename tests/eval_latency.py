#!/usr/bin/env python3
"""The end-to-end latency budget of the shipped NPU path, per stage.

Re-derives every stage from committed traces alone -- no model, no accelerator,
no audio, no network, well under a second -- so a latency claim survives the
scrollback that produced it. Supersedes `eval_vac_lag.py`, whose per-character
commit lag is stage 2 here; that math is unchanged and still carries the whole
weight of the headline number.

    voice ──1──> a decode covers it ──2──> committed, on the status line
                                            │
              speech ends ──3──> `JA n:` ──4──> `EN n:`

    1+2  commit lag   `vac_decode_trace.json`   what the reader waits for mid-speech
    3    publication  `vac_decode_trace.json`   VAD_MIN_SILENCE_S + the final decode
    4    translate    `en_pairing_trace.json`   one codex turn, per caption
    decode           `vac_decode_trace.json`   the unit cost stages 1-3 are built from

Stage 2 is answered per CHARACTER, not per commit. A commit carries several
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

Two ARMS run the same clock and the same placement rule over the same trace, so
their difference is exactly the display policy and nothing else:

- `committed` -- what the status line shows today: the LocalAgreement-2 commit.
- `provisional` -- the whole latest hypothesis, its unconfirmed tail included.
  The tail is rewritten by the next decode, so this is a status-line-only policy;
  the published `JA n:` line and the transcript are the committed text either way.

Both arms measure FIRST APPEARANCE: when a character reaches the screen. Neither
measures when it stops moving, and the two are not the same character-for-character
under the provisional arm, whose text is by definition unsettled. `redraws` is that
qualifier, so the gap is never read as a settled-text speedup: it counts updates
whose hypothesis diverges from its predecessor before the predecessor ended, i.e.
where already-visible characters were rewritten. Committed text cannot be among
them -- `emitted` is append-only -- so a redraw always lands in the dimmed tail.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from live_stt import VAC_CHUNK_S, VAD_MIN_SILENCE_S  # noqa: E402
from streaming import common_prefix  # noqa: E402

TRACE = ROOT / "tests" / "vac_decode_trace.json"
PAIRING = ROOT / "tests" / "en_pairing_trace.json"


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "p50": None, "mean": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50": round(statistics.median(ordered), 3),
        "mean": round(statistics.fmean(ordered), 3),
        "p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 3),
        "max": round(ordered[-1], 3),
    }


def hypothesis(row: dict[str, Any]) -> str:
    """The decode's full text. Spans ARE the hypothesis wherever they exist."""
    spans = row["segments"]
    return "".join(span[2] for span in spans) if spans else row.get("text", "")


def _hypothesis_end(row: dict[str, Any]) -> float:
    """Absolute audio time the hypothesis reaches, or the buffer end without spans."""
    offset_s = row["buffer_end_s"] - row["buffer_s"]
    spans = row["segments"]
    return offset_s + spans[-1][1] if spans else row["buffer_end_s"]


def clip_lags(
    clip: dict[str, Any], provisional: bool = False
) -> tuple[list[float], dict[str, int]]:
    """Per-character lags for one clip, with the counts that qualify them."""
    lags: list[float] = []
    counts = {"updates": 0, "commits": 0, "unplaced": 0, "non_monotone": 0, "finals": 0}
    counts["redraws"] = 0
    now = 0.0
    start: float | None = None
    seen = ""  # previous update's hypothesis, for the redraw qualifier
    for row in clip["series"]:
        counts["updates"] += 1
        now = max(now, row["buffer_end_s"]) + row["decode_s"]
        if start is None:  # first update of an utterance: text begins at its audio start
            start = float(row["buffer_end_s"]) - float(row["buffer_s"])
        since: float = start
        end: float | None
        if provisional:
            # Everything the decode produced is on screen at once, so what is NEW
            # is the tail whose audio runs past the point the display already
            # reached. A final update's flushed tail still ends at the buffer end.
            reach: float = row["buffer_end_s"] if row["final"] else _hypothesis_end(row)
            text = hypothesis(row)
            if common_prefix(text, seen) < len(seen):
                counts["redraws"] += 1
            seen = "" if row["final"] else text
            covered = reach - (row["buffer_end_s"] - row["buffer_s"])
            fresh = (
                round(len(text) * (reach - since) / covered) if reach > since and covered > 0 else 0
            )
            text = text[len(text) - fresh :] if fresh > 0 else ""
            end = reach
        else:
            text = row["commit"]
            end = row["buffer_end_s"] if row["final"] else row["commit_audio_s"]
        if text:
            counts["commits"] += 1
            if end is None:
                # No spans to interpolate in, so the commit cannot be placed on
                # the audio clock at all. Counted, never guessed at.
                counts["unplaced"] += 1
            else:
                if end < since:
                    counts["non_monotone"] += 1
                span = max(end, since) - since
                lags += [now - (since + span * (i + 0.5) / len(text)) for i in range(len(text))]
                start = max(end, since)
        if row["final"]:
            counts["finals"] += 1
            start = None
    return lags, counts


def redraw_bound_lags(clip: dict[str, Any]) -> list[float]:
    """Pessimistic companion to the provisional arm: a redrawn slot waits again.

    `clip_lags(provisional=True)` charges a character once, at the update that
    first put it on screen. This charges every character at or past the common
    prefix on EVERY update, so a slot rewritten five times pays five waits. That
    double-counts by construction and is a loose UPPER bound, not a measured
    reader experience -- it exists so the first-appearance figure is never read
    as the whole story. Placement is over each hypothesis's own span, because a
    redraw repaints the whole tail rather than extending it.
    """
    lags: list[float] = []
    now = 0.0
    seen = ""
    for row in clip["series"]:
        now = max(now, row["buffer_end_s"]) + row["decode_s"]
        text = hypothesis(row)
        offset_s = float(row["buffer_end_s"]) - float(row["buffer_s"])
        reach = float(row["buffer_end_s"] if row["final"] else _hypothesis_end(row))
        span = max(reach - offset_s, 0.0)
        for i in range(common_prefix(text, seen), len(text)):
            lags.append(now - (offset_s + span * (i + 0.5) / len(text)))
        seen = "" if row["final"] else text
    return lags


def decode_fit(clip: dict[str, Any]) -> dict[str, float]:
    """Split per-update decode cost into its fixed and per-character halves.

    The NPU runs Whisper's encoder over a static 30 s window whatever the buffer
    holds, so decode cost is a large constant plus autoregressive token
    generation. Fitting it is what says which lever is worth pulling: shortening
    the buffer only touches the marginal half.
    """
    lengths = [float(len(hypothesis(row))) for row in clip["series"]]
    costs = [row["decode_s"] for row in clip["series"]]
    n = len(costs)
    mean_x, mean_y = statistics.fmean(lengths), statistics.fmean(costs)
    var = sum((x - mean_x) ** 2 for x in lengths)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(lengths, costs, strict=True)) / var
    return {
        "fixed_s": round(mean_y - slope * mean_x, 4),
        "per_char_s": round(slope, 5),
        "chars_p50": round(statistics.median(lengths), 1),
        "n": n,
    }


def report(trace: dict[str, Any], pairing: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model": trace["model"],
        "requested_device": trace["requested_device"],
        "vac_chunk_s": trace["vac_chunk_s"],
        "vac_trim_s": trace["vac_trim_s"],
        "clips": {},
    }
    for name, clip in trace["clips"].items():
        series = clip["series"]
        committed, counts = clip_lags(clip)
        provisional, pcounts = clip_lags(clip, provisional=True)
        counts["redraws"] = pcounts["redraws"]
        finals = [row["decode_s"] for row in series if row["final"]]
        out["clips"][name] = {
            "audio_s": clip["audio_s"],
            "utterances": clip["utterances"],
            "decode_s": _quantiles([row["decode_s"] for row in series]),
            "decode_fit": decode_fit(clip),
            "buffer_s": _quantiles([row["buffer_s"] for row in series]),
            # Wall time an update spends beyond the audio it consumed. Positive
            # means this update handed its successor a backlog (`asr-pipeline.md`).
            "carry_s": _quantiles([row["decode_s"] - VAC_CHUNK_S for row in series]),
            "commit_lag_s": _quantiles(committed),
            "provisional_lag_s": _quantiles(provisional),
            "redraw_bound_s": _quantiles(redraw_bound_lags(clip)),
            # Speech end -> `JA n:`. silero needs VAD_MIN_SILENCE_S of silence to
            # close the utterance, then one full decode publishes it. Its own
            # detection granularity (one 32 ms window) is not in the trace.
            "publication_s": _quantiles([VAD_MIN_SILENCE_S + cost for cost in finals]),
            "vad_close_s": VAD_MIN_SILENCE_S,
            **counts,
        }
    if pairing is not None:
        turns = [turn["s"] for turn in pairing["turns"] if turn.get("s")]
        rotated = [turn["s"] for turn in pairing["turns"] if turn.get("s") and turn.get("rotated")]
        plain = [
            turn["s"] for turn in pairing["turns"] if turn.get("s") and not turn.get("rotated")
        ]
        out["translate_turn_s"] = {
            "model": pairing["run"]["model"],
            "effort": pairing["run"]["effort"],
            "all": _quantiles(turns),
            # A glossary change opens a fresh thread before the turn, and that
            # thread pays an uncached prompt. It is the same turn, so it belongs
            # in the same stage -- split out because the two costs move apart.
            "rotating": _quantiles(rotated),
            "steady": _quantiles(plain),
            "rotations": pairing["run"]["rotations"],
        }
    return out


def render(out: dict[str, Any]) -> str:
    lines = [f"latency budget: {out['model']} on {out['requested_device']}", ""]
    for name, row in out["clips"].items():
        fit = row["decode_fit"]
        lines.append(f"  {name}: audio={row['audio_s']:.3f}s utterances={row['utterances']}")
        for stage in (
            "decode_s",
            "commit_lag_s",
            "provisional_lag_s",
            "redraw_bound_s",
            "publication_s",
            "carry_s",
        ):
            q = row[stage]
            lines.append(
                f"      {stage:<18} n={q['n']:<5} p50={q['p50']:>7.3f}  p90={q['p90']:>7.3f}  "
                f"max={q['max']:>7.3f}"
            )
        lines.append(
            f"      decode = {fit['fixed_s']:.3f}s fixed + {fit['per_char_s'] * 1000:.2f}ms/char "
            f"(p50 {fit['chars_p50']:.0f} chars);  buffer p50={row['buffer_s']['p50']:.3f}s "
            f"max={row['buffer_s']['max']:.3f}s"
        )
        lines.append(
            f"      updates={row['updates']} commits={row['commits']} finals={row['finals']} "
            f"unplaced={row['unplaced']} non_monotone={row['non_monotone']} "
            f"redraws={row['redraws']}"
        )
    turn = out.get("translate_turn_s")
    if turn:
        lines.append("")
        lines.append(f"  translate turn: {turn['model']} effort={turn['effort']}")
        for key in ("all", "steady", "rotating"):
            q = turn[key]
            lines.append(
                f"      {key:<18} n={q['n']:<5} p50={q['p50']:>7.3f}  p90={q['p90']:>7.3f}  "
                f"max={q['max']:>7.3f}"
            )
        lines.append(f"      thread rotations: {turn['rotations']} of {turn['all']['n']} turns")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--pairing", type=Path, default=PAIRING)
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args()

    if not args.trace.is_file():
        print(f"error: no trace at {args.trace}", file=sys.stderr)
        sys.exit(1)
    pairing = (
        json.loads(args.pairing.read_text(encoding="utf-8")) if args.pairing.is_file() else None
    )
    out = report(json.loads(args.trace.read_text(encoding="utf-8")), pairing)
    print(json.dumps(out, ensure_ascii=False, indent=2) if args.json else render(out))


if __name__ == "__main__":
    main()
