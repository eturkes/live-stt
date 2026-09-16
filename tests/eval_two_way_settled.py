#!/usr/bin/env python3
"""Does two-way's withheld first commit move time-to-SETTLED? Replayed, it does not.

Two-way publishes nothing until the LID accepts a label, which cannot happen before
`LID_MIN_SECONDS` of voiced buffer exists. The queue row assumed that withholding the
first commit shifts settled text later. It does not, on either committed clip:
LocalAgreement-2 already commits almost nothing on update 1, so at the 2.0 s gate the
shift is ZERO and 13 characters across both clips are re-dated at all.

Same virtual clock, same per-character placement and the same `_quantiles` as
`eval_latency.py`'s commit-lag arm, so the no-withholding arm reproduces that stage's
published numbers and the difference between arms is the commit rule alone. A withheld
character settles at the ACCEPTING update's wall clock; its audio placement never moves.
Time-to-FIRST-GLIMPSE is untouched in every arm, the dim tail rendering upstream of the
commit rule.

The never-accepted arm is an UPPER BOUND, not a forecast: it forces both long-form clips
into the held path, which is not what abstention does. Abstention concentrates in
utterances shorter than 2 s, where the final update arrives carrying the whole caption
at once and no earlier commit is left to withhold.

No model, no accelerator, no audio, no network, well under a second.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_latency import _quantiles  # noqa: E402

TRACE = Path(__file__).resolve().parent / "vac_decode_trace.json"
GATE_S = 2.0
# (label, withholds, the buffer length that accepts) -- the last arm never reaches its gate.
ARMS = (
    ("one-way, no withholding", False, GATE_S),
    ("two-way, accepted at the 2.0 s gate", True, GATE_S),
    ("two-way, abstained once, accepted at 3.0 s", True, 3.0),
    ("two-way, never accepted (held label)", True, 1e9),
)


def utterances(series: list[dict]) -> list[list[dict]]:
    """Split a clip's update series into utterances, each ending on its final update."""
    out: list[list[dict]] = []
    current: list[dict] = []
    for row in series:
        current.append(row)
        if row["final"]:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def clip_lags(clip: dict, withhold: bool, gate_s: float = GATE_S) -> tuple[list[float], int]:
    """Per-character committed lags for one clip, `withhold` applying the two-way rule."""
    lags: list[float] = []
    now = 0.0
    held_chars = 0
    for rows in utterances(clip["series"]):
        # The accepting update is known before any of this utterance is charged, so run
        # the same clock forward to find it: a withheld character cannot settle earlier.
        accept_at = None
        if withhold:
            wall = now
            for row in rows:
                wall = max(wall, row["buffer_end_s"]) + row["decode_s"]
                if row["buffer_s"] >= gate_s or row["final"]:
                    accept_at = wall
                    break
        start = None
        for row in rows:
            now = max(now, row["buffer_end_s"]) + row["decode_s"]
            if start is None:
                start = float(row["buffer_end_s"]) - float(row["buffer_s"])
            since = start
            text = row["commit"]
            end = row["buffer_end_s"] if row["final"] else row["commit_audio_s"]
            if not text or end is None:
                continue
            settled = now
            if withhold and accept_at is not None and row["buffer_s"] < gate_s and not row["final"]:
                settled = max(now, accept_at)
                held_chars += len(text)
            span = max(end, since) - since
            lags += [settled - (since + span * (i + 0.5) / len(text)) for i in range(len(text))]
            start = max(end, since)
    return lags, held_chars


def report(trace: dict) -> dict:
    """Every arm over every clip: the committed-lag quantiles and the re-dated count."""
    arms = []
    for name, withhold, gate_s in ARMS:
        clips = {}
        for clip_name, clip in trace["clips"].items():
            lags, held = clip_lags(clip, withhold, gate_s)
            # `re_dated` is an upper bound on how many characters MOVED: one whose own
            # clock already runs past the accepting update keeps its own.
            clips[clip_name] = dict(_quantiles(lags), re_dated=held)
        arms.append({"arm": name, "withhold": withhold, "gate_s": gate_s, "clips": clips})
    return {"arms": arms}


def render(out: dict) -> str:
    lines = []
    for arm in out["arms"]:
        lines.append(f"== {arm['arm']}")
        for clip_name, q in arm["clips"].items():
            lines.append(
                f"   {clip_name:24s} p50={q['p50']} p90={q['p90']} max={q['max']} "
                f"n={q['n']} re-dated={q['re_dated']}"
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
