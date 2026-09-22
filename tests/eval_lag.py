#!/usr/bin/env python3
"""Replay the committed lag trace: distribution, backlog split, degrade adjacency, length."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

TRACE = Path(__file__).resolve().parent / "lag_sessions.json"
NOTE_WINDOW_S = 60
BACKLOG_LIMIT = (
    "A null target is abandoned; the transcript cannot show when its translation attempt stopped."
)


def _percentile(ordered: list[int], fraction: float) -> int | None:
    return ordered[int(len(ordered) * fraction)] if ordered else None


def distribution(values: list[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    return {
        "pairs": len(ordered),
        "p50": statistics.median(ordered) if ordered else None,
        "p75": _percentile(ordered, 0.75),
        "p90": _percentile(ordered, 0.9),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": max(ordered) if ordered else None,
        "ge_6s": sum(value >= 6 for value in ordered),
        "ge_10s": sum(value >= 10 for value in ordered),
        "ge_20s": sum(value >= 20 for value in ordered),
        "ge_30s": sum(value >= 30 for value in ordered),
    }


def _lags(session: dict[str, Any]) -> list[int]:
    return [pair["tgt_s"] - pair["src_s"] for pair in session["pairs"] if pair["tgt_s"] is not None]


def _lag_summary(values: list[int]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": statistics.median(ordered) if ordered else None,
        "p90": _percentile(ordered, 0.9),
        "max": max(ordered) if ordered else None,
    }


def _backlog_values(session: dict[str, Any]) -> dict[str, list[int]]:
    split: dict[str, list[int]] = {"standalone": [], "backlogged": []}
    previous: list[dict[str, Any]] = []
    for pair in sorted(session["pairs"], key=lambda row: row["n"]):
        target = pair["tgt_s"]
        if target is not None:
            outstanding = any(
                earlier["tgt_s"] is not None and earlier["tgt_s"] > pair["src_s"]
                for earlier in previous
            )
            split["backlogged" if outstanding else "standalone"].append(target - pair["src_s"])
        previous.append(pair)
    return split


def _backlog_report(values: dict[str, list[int]]) -> dict[str, dict[str, int | float | None]]:
    return {name: _lag_summary(lags) for name, lags in values.items()}


def _degrade_adjacency(session: dict[str, Any]) -> list[dict[str, Any]]:
    notes = []
    for note in session["notes"]:
        captions = []
        for pair in session["pairs"]:
            offset = pair["src_s"] - note["at_s"]
            if abs(offset) <= NOTE_WINDOW_S:
                captions.append(
                    {
                        "n": pair["n"],
                        "src_s": pair["src_s"],
                        "offset_s": offset,
                        "lag_s": (None if pair["tgt_s"] is None else pair["tgt_s"] - pair["src_s"]),
                    }
                )
        notes.append(
            {
                "at_s": note["at_s"],
                "text": note["text"],
                "window_s": NOTE_WINDOW_S,
                "captions": captions,
            }
        )
    return notes


def _average_ranks(values: list[int]) -> list[float]:
    ranks = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_delta) * sum(value * value for value in right_delta)
    )
    if denominator == 0:
        return None
    correlation = (
        sum(x_delta * y_delta for x_delta, y_delta in zip(left_delta, right_delta, strict=True))
        / denominator
    )
    return round(correlation, 6)


def _length_values(session: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (pair["src_chars"], pair["tgt_s"] - pair["src_s"])
        for pair in session["pairs"]
        if pair["tgt_s"] is not None
    ]


def _length_report(values: list[tuple[int, int]]) -> dict[str, Any]:
    lengths = [length for length, _ in values]
    lags = [lag for _, lag in values]
    length_ranks = _average_ranks(lengths)
    lag_ranks = _average_ranks(lags)
    groups: list[list[tuple[int, int]]] = [[] for _ in range(4)]
    for row, rank in zip(values, length_ranks, strict=True):
        quartile = min(3, int((rank - 1) * 4 / len(values))) if values else 0
        groups[quartile].append(row)
    return {
        "pairs": len(values),
        "spearman_rho": _pearson(length_ranks, lag_ranks),
        "quartile_method": "average ranks; equal source lengths stay together",
        "quartiles": [
            {
                "quartile": index,
                "count": len(group),
                "src_chars_min": min((length for length, _ in group), default=None),
                "src_chars_max": max((length for length, _ in group), default=None),
                "median_lag_s": (statistics.median(lag for _, lag in group) if group else None),
            }
            for index, group in enumerate(groups, start=1)
        ],
    }


def report(trace: dict[str, Any]) -> dict[str, Any]:
    sessions = []
    pooled_lags: list[int] = []
    pooled_backlog: dict[str, list[int]] = {"standalone": [], "backlogged": []}
    pooled_notes = []
    pooled_lengths: list[tuple[int, int]] = []
    for session in trace["sessions"]:
        lags = _lags(session)
        backlog = _backlog_values(session)
        adjacency = _degrade_adjacency(session)
        lengths = _length_values(session)
        sessions.append(
            {
                "id": session["id"],
                "distribution": distribution(lags),
                "backlog": _backlog_report(backlog),
                "degrade_adjacency": adjacency,
                "length": _length_report(lengths),
            }
        )
        pooled_lags.extend(lags)
        pooled_lengths.extend(lengths)
        for name, values in backlog.items():
            pooled_backlog[name].extend(values)
        pooled_notes.extend({"session": session["id"], **note} for note in adjacency)
    return {
        "backlog_limit": BACKLOG_LIMIT,
        "sessions": sessions,
        "pooled": {
            "distribution": distribution(pooled_lags),
            "backlog": _backlog_report(pooled_backlog),
            "degrade_adjacency": pooled_notes,
            "length": _length_report(pooled_lengths),
        },
    }


def _render_distribution(values: dict[str, int | float | None]) -> str:
    return (
        f"pairs={values['pairs']} p50={values['p50']}s p75={values['p75']}s "
        f"p90={values['p90']}s p95={values['p95']}s p99={values['p99']}s "
        f"max={values['max']}s >=6s={values['ge_6s']} >=10s={values['ge_10s']} "
        f">=20s={values['ge_20s']} >=30s={values['ge_30s']}"
    )


def _render_backlog(values: dict[str, dict[str, int | float | None]]) -> str:
    halves = []
    for name in ("standalone", "backlogged"):
        row = values[name]
        halves.append(
            f"{name} count={row['count']} p50={row['p50']}s p90={row['p90']}s max={row['max']}s"
        )
    return "; ".join(halves)


def _render_adjacency(notes: list[dict[str, Any]]) -> list[str]:
    if not notes:
        return ["degrade adjacency: none"]
    lines = []
    for note in notes:
        session = f" session={note['session']}" if "session" in note else ""
        lines.append(
            f"degrade note:{session} at={note['at_s']}s window=+/-{note['window_s']}s "
            f"captions={len(note['captions'])}: {note['text']}"
        )
        for caption in note["captions"]:
            lag = "none" if caption["lag_s"] is None else f"{caption['lag_s']}s"
            lines.append(
                f"  n={caption['n']} src={caption['src_s']}s "
                f"offset={caption['offset_s']:+d}s lag={lag}"
            )
    return lines


def _render_length(values: dict[str, Any]) -> list[str]:
    rho = "none" if values["spearman_rho"] is None else f"{values['spearman_rho']:.6f}"
    lines = [f"length: pairs={values['pairs']} spearman_rho={rho}"]
    for row in values["quartiles"]:
        bounds = (
            "none"
            if row["src_chars_min"] is None
            else f"{row['src_chars_min']}..{row['src_chars_max']}"
        )
        lines.append(
            f"  q{row['quartile']} count={row['count']} src_chars={bounds} "
            f"median_lag={row['median_lag_s']}s"
        )
    return lines


def render(result: dict[str, Any]) -> str:
    lines = [f"backlog limit: {result['backlog_limit']}"]
    for session in result["sessions"]:
        lines += [
            f"== {session['id']}",
            f"distribution: {_render_distribution(session['distribution'])}",
            f"backlog: {_render_backlog(session['backlog'])}",
            *_render_adjacency(session["degrade_adjacency"]),
            *_render_length(session["length"]),
        ]
    lines += [
        "== pooled",
        f"distribution: {_render_distribution(result['pooled']['distribution'])}",
        f"backlog: {_render_backlog(result['pooled']['backlog'])}",
        *_render_adjacency(result["pooled"]["degrade_adjacency"]),
        *_render_length(result["pooled"]["length"]),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args()

    try:
        trace = json.loads(TRACE.read_text(encoding="utf-8"))
        result = report(trace)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"error: missing or unparseable trace: {error}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=2, ensure_ascii=True) if args.json else render(result))


if __name__ == "__main__":
    main()
