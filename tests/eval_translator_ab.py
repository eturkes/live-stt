#!/usr/bin/env python3
"""Which codex model + effort translates captions fastest: a paired latency screen.

On-demand, never a gate step: it needs a logged-in `codex app-server` and spends
real turns on the shared Codex bucket (~2 s each). Inputs are committed traces, so
it reruns from a clean clone with no gitignored artifact and no accelerator.

Each arm = MODEL:EFFORT on its own real `CodexTranslator` — own app-server, the
shipped argv, instructions and serviceTier — so the pair is the only difference.
The pair lives in module constants read at call time (`_new_thread`, `_turn`), so
`pinned` sets them around each of the arm's calls, and a wiretap on `_request`
records what actually went out and what thread/start echoed back. Both legs take
the production warm-up (`_handshake`) before any measured turn, and no leg reaches
TRANSLATE_ROTATE_TURNS. Per block of items, arm order follows a Williams design
(every arm at every position, every directed predecessor pair equally often),
rows shuffled and labels permuted from a fixed seed: drift, position and a
neighbour's prefix-cache warming hit every arm alike (L-026). The span is
production `_translate`: turn/start → turn/completed through the real collect.

Estimand = median per-item (challenger − FIRST arm) seconds, with a paired
bootstrap: iid at 95 % and Bonferroni-adjusted over the challengers, plus
moving-block (4, 8) sensitivity for serial dependence, and the paired p90
difference. A failed or no-attempt turn drops its item from the pairs and makes
its arm ineligible. One process per arm = one session per arm: a winner here is
a screen result, confirmed by a fresh-client finalist-vs-control rerun.

    uv run python tests/eval_translator_ab.py [--arms M:E ...] [--ja N] [--en N]
                                              [--seed S] [--start I] [--json F]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path[:0] = [str(ROOT), str(TESTS)]

from eval_latency import _quantiles  # noqa: E402

import live_stt  # noqa: E402  (after sys.path injection)

ARMS = ("gpt-6-luna:low", "gpt-6-luna:none", "gpt-6.1-terra:low", "gpt-6.1-terra:none")
CAPTIONS = TESTS / "caption_trace.json"
PAIRING = TESTS / "en_pairing_trace.json"
START = 40  # mid-story: narration, no section preamble (eval_translate_repeat's slice)
SEED = 20261105
BOOTSTRAP = 10_000
BLOCKS = (4, 8)
MIN_GAIN_S = 0.2  # a JA win smaller than this is no reason to swap a user-ruled model


def parse_arm(text: str) -> tuple[str, str]:
    model, sep, effort = text.rpartition(":")
    if not (sep and model and effort):
        raise argparse.ArgumentTypeError(f"arm {text!r} is not MODEL:EFFORT")
    return model, effort


def items(ja: int, en: int, start: int = START) -> list[tuple[str, str]]:
    """(source, text): recogniser captions JA→EN, then their English EN→JA, from `start` on."""
    captions = json.loads(CAPTIONS.read_text(encoding="utf-8"))["captions"]
    turns = json.loads(PAIRING.read_text(encoding="utf-8"))["turns"]
    picked = [("ja", c["text"]) for c in captions[start : start + ja]]
    picked += [("en", t["en"]) for t in turns[start : start + en]]
    if len(picked) != ja + en:
        raise SystemExit(f"--start {start} leaves {len(picked)} of {ja + en} items")
    return picked


def williams(k: int) -> list[list[int]]:
    """Rows where each symbol takes each position and follows each other symbol equally often."""
    first, lo, hi = [0], 1, k - 1
    while len(first) < k:
        first.append(lo)
        lo += 1
        if len(first) < k:
            first.append(hi)
            hi -= 1
    rows = [[(x + i) % k for x in first] for i in range(k)]
    return rows if k % 2 == 0 else rows + [row[::-1] for row in rows]


def schedule(arms: list[Any], n: int, seed: int = SEED) -> list[list[Any]]:
    """Item i's arm order: Williams rows per block, rows shuffled, labels permuted per block."""
    rng = random.Random(seed)  # noqa: S311 — a fixed design seed, rerun from committed state
    rows, out = williams(len(arms)), []
    while len(out) < n:
        block, labels = rows[:], list(arms)
        rng.shuffle(block)
        rng.shuffle(labels)
        out += [[labels[j] for j in row] for row in block]
    return out[:n]


def _p90(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]  # eval_latency's convention


def paired(
    control: Sequence[float | None],
    challenger: Sequence[float | None],
    *,
    block: int = 1,
    alpha: float = 0.05,
    reps: int = BOOTSTRAP,
) -> dict[str, Any]:
    """challenger − control per item: median + p90 difference, moving-block bootstrap CIs.

    A failed turn (None) on either side drops the item. Resampled index sets depend on
    (n, block, reps) alone, so both arms of a pair always see the same items.
    """
    pairs = [
        (a, b) for a, b in zip(control, challenger, strict=True) if a is not None and b is not None
    ]
    n = len(pairs)
    if not n:
        return {"n": 0, "median": None, "lo": None, "hi": None, "p90": None}
    diffs = [b - a for a, b in pairs]
    rng = random.Random(SEED)  # noqa: S311 — a seeded bootstrap, rerun from committed state
    starts = range(max(1, n - block + 1))
    meds, p90s = [], []
    for _ in range(reps):
        idx: list[int] = []
        while len(idx) < n:
            s = rng.choice(starts)
            idx += range(s, min(s + block, n))
        idx = idx[:n]
        meds.append(statistics.median(diffs[i] for i in idx))
        p90s.append(_p90([pairs[i][1] for i in idx]) - _p90([pairs[i][0] for i in idx]))
    meds.sort()
    p90s.sort()
    cut = int(alpha / 2 * reps)
    return {
        "n": n,
        "median": round(statistics.median(diffs), 3),
        "lo": round(meds[cut], 3),
        "hi": round(meds[reps - 1 - cut], 3),
        "p90": round(_p90([b for _, b in pairs]) - _p90([a for a, _ in pairs]), 3),
        "p90_lo": round(p90s[cut], 3),
        "p90_hi": round(p90s[reps - 1 - cut], 3),
    }


def verdict(ja: dict[str, Any], en: dict[str, Any] | None, clean: bool) -> str:
    """The rule frozen before the screen ran, applied to one challenger."""
    if not clean:
        return "ineligible: a failed, no-attempt, multi-line or off-wire turn"
    adjusted = ja["adjusted"]
    if adjusted["hi"] is None or not (adjusted["hi"] < 0 and -adjusted["median"] >= MIN_GAIN_S):
        return "no convincing speed win"
    if en is not None and en["iid"]["lo"] is not None and en["iid"]["lo"] > 0:
        return "vetoed: convincingly slower EN→JA"
    if any(ja[f"block{b}"]["hi"] >= 0 for b in BLOCKS):
        return "fragile: a block bootstrap spans 0"
    return "faster"


class _Usage(asyncio.Queue):
    """A translator's note queue that also keeps the latest turn's token usage."""

    def __init__(self) -> None:
        super().__init__()
        self.last: dict[str, Any] = {}

    def put_nowait(self, item: dict) -> None:
        if item.get("method") == "thread/tokenUsage/updated":
            self.last = ((item.get("params") or {}).get("tokenUsage") or {}).get("last") or {}
        super().put_nowait(item)


class _Log(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())


@dataclass
class Arm:
    name: str
    model: str
    effort: str
    translator: Any = None
    log: list[str] = field(default_factory=list)
    wire: list[tuple] = field(default_factory=list)

    def on_wire(self) -> bool:
        """Every request this arm sent carried its own pair, and the server applied the tier."""
        threads = [w for w in self.wire if w[0] == "thread"]
        turns = [w for w in self.wire if w[0] == "turn"]
        return (
            bool(threads and turns)
            and all(w[1] == self.model and w[2] == w[3] for w in threads)
            and all(w[1] == self.effort for w in turns)
        )


def _wiretap(arm: Arm) -> None:
    real = arm.translator._request

    async def request(method: str, params: Any = None) -> Any:
        if method == "turn/start":
            arm.wire.append(("turn", params.get("effort")))
        result = await real(method, params)
        if method == "thread/start":
            tier = (result or {}).get("serviceTier")
            arm.wire.append(("thread", params["model"], params.get("serviceTier"), tier))
        return result

    arm.translator._request = request


@contextlib.contextmanager
def pinned(arm: Arm, handler: _Log):
    """The arm's pair in the module constants for one call; its warnings kept per arm."""
    saved = live_stt.TRANSLATE_MODEL, live_stt.TRANSLATE_EFFORT
    live_stt.TRANSLATE_MODEL, live_stt.TRANSLATE_EFFORT = arm.model, arm.effort
    mark = len(handler.records)
    try:
        yield
    finally:
        live_stt.TRANSLATE_MODEL, live_stt.TRANSLATE_EFFORT = saved
        arm.log += handler.records[mark:]


async def run(arms: list[Arm], work: list[tuple[str, str]], seed: int = SEED) -> list[dict]:
    handler = _Log()
    live_stt.logger.addHandler(handler)
    rows: list[dict[str, Any]] = []
    try:
        startup = list(arms)
        random.Random(seed).shuffle(startup)  # noqa: S311 — fixed design seed
        for arm in startup:
            arm.translator = live_stt.CodexTranslator(sources=("ja", "en"))
            arm.translator._notes = _Usage()
            _wiretap(arm)
            with pinned(arm, handler):
                if not await arm.translator.start("ja"):
                    raise SystemExit(f"{arm.name}: codex app-server did not start; codex login?")
                await arm.translator._handshake("en")  # the lazy reverse warm-up, unmeasured
        t0, previous = time.perf_counter(), None
        orders = schedule(arms, len(work), seed)
        for i, ((source, text), order) in enumerate(zip(work, orders, strict=True)):
            for position, arm in enumerate(order):
                translator = arm.translator
                translator._notes.last = {}
                age = translator._legs[source].turns
                attempted = translator.enabled
                with pinned(arm, handler):
                    started = time.perf_counter()
                    out = await translator._translate(text, source)
                    seconds = time.perf_counter() - started
                status = "ok" if out else ("failed" if attempted else "disabled")
                rows.append(
                    {
                        "item": i,
                        "source": source,
                        "arm": arm.name,
                        "position": position,
                        "after": previous,
                        "at_s": round(started - t0, 3),
                        "age": age,
                        "status": status,
                        "seconds": round(seconds, 3) if status == "ok" else None,
                        "tokens": translator._notes.last,
                        "text": text,
                        "out": out,
                    }
                )
                previous = arm.name
                print(
                    f"{i:3d} {source} {arm.name:20s} {seconds:6.2f}s {status:8s} {out[:50]!r}",
                    flush=True,
                )
    finally:
        for arm in arms:
            if arm.translator is not None:
                await arm.translator.close()
        live_stt.logger.removeHandler(handler)
    return rows


def summarize(arms: list[Arm], rows: list[dict[str, Any]], reps: int = BOOTSTRAP) -> dict:
    out: dict[str, Any] = {}
    adjusted_alpha = 0.05 / max(1, len(arms) - 1)
    for source in ("ja", "en"):
        mine = [r for r in rows if r["source"] == source]
        if not mine:
            continue
        per = {
            a.name: sorted((r for r in mine if r["arm"] == a.name), key=lambda r: r["item"])
            for a in arms
        }
        control = [r["seconds"] for r in per[arms[0].name]]
        out[source] = {}
        for a in arms:
            turns = per[a.name]
            seconds = [r["seconds"] for r in turns]
            tokens = [r["tokens"] for r in turns]
            out[source][a.name] = {
                **_quantiles([s for s in seconds if s is not None]),
                "failed": sum(r["status"] != "ok" for r in turns),
                "multiline": sum("\n" in r["out"] for r in turns),
                "reasoning_max": max(
                    (t.get("reasoningOutputTokens") or 0 for t in tokens), default=0
                ),
                "output_tokens_p50": statistics.median(
                    [t.get("outputTokens") or 0 for t in tokens]
                ),
                "iid": paired(control, seconds, reps=reps),
                "adjusted": paired(control, seconds, alpha=adjusted_alpha, reps=reps),
                **{f"block{b}": paired(control, seconds, block=b, reps=reps) for b in BLOCKS},
            }
    out["wire"] = {a.name: a.on_wire() for a in arms}
    out["warnings"] = {a.name: a.log for a in arms}
    out["verdict"] = {}
    for a in arms[1:]:
        clean = out["wire"][a.name] and all(
            out[s][a.name]["failed"] == 0 and out[s][a.name]["multiline"] == 0
            for s in ("ja", "en")
            if s in out
        )
        en = (out.get("en") or {}).get(a.name)
        out["verdict"][a.name] = verdict(out["ja"][a.name], en, clean)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), help="MODEL:EFFORT, control first")
    parser.add_argument("--ja", type=int, default=60, help="JA→EN items (default 60, primary)")
    parser.add_argument("--en", type=int, default=40, help="EN→JA items (default 40, guardrail)")
    parser.add_argument("--seed", type=int, default=SEED, help="order + startup design seed")
    parser.add_argument("--start", type=int, default=START, help="first item of both slices")
    parser.add_argument("--json", type=Path, help="write summary + every turn to this path")
    args = parser.parse_args()
    arms = [Arm(name, *parse_arm(name)) for name in args.arms]
    rows = asyncio.run(run(arms, items(args.ja, args.en, args.start), args.seed))
    summary = summarize(arms, rows)
    for source in ("ja", "en"):
        if source not in summary:
            continue
        print(f"\n{source}→{'en' if source == 'ja' else 'ja'}  (paired = arm − {arms[0].name})")
        for name, s in summary[source].items():
            i, adj = s["iid"], s["adjusted"]
            blocks = " ".join(
                f"b{b} [{s[f'block{b}']['lo']}, {s[f'block{b}']['hi']}]" for b in BLOCKS
            )
            print(
                f"  {name:20s} n={s['n']:3d} p50={s['p50']} p90={s['p90']} max={s['max']} "
                f"failed={s['failed']} reasoning_max={s['reasoning_max']}\n"
                f"  {'':20s} paired {i['median']} [{i['lo']}, {i['hi']}] "
                f"adj [{adj['lo']}, {adj['hi']}] {blocks} "
                f"p90 {i['p90']} [{i['p90_lo']}, {i['p90_hi']}]"
            )
    print(f"\nwire {summary['wire']}\nwarnings {summary['warnings']}\nverdict {summary['verdict']}")
    if args.json:
        codex = subprocess.run(["codex", "--version"], capture_output=True, text=True, check=False)  # noqa: S607
        meta = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "codex": codex.stdout.strip(),
            "seed": args.seed,
            "start": args.start,
            "wire": {a.name: a.wire for a in arms},
        }
        payload = {"meta": meta, "arms": args.arms, "summary": summary, "rows": rows}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
