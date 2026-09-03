#!/usr/bin/env python3
"""What repetition stops the translator terminating: M13.1's calibration probe.

The producer of `TRANSLATE_REPEAT_MAX_CHARS`. On-demand, never a gate step: it
needs a live `codex app-server` and spends up to ~15 minutes of real turns. Its
inputs are literals plus the committed caption trace, so it reruns from a clean
clone with no gitignored artifact and no accelerator.

`_turn` runs under a 30 s bound while the shipped cap is TRANSLATE_TIMEOUT_S=15,
so "stalls" is a measurement rather than a timeout artifact.

Contamination control, and it is what makes the matrix readable (L-026). A
stalled turn keeps generating server-side, and `_abort_turn`'s interrupt plus
note drain answers an ERRORED turn, not a stalled one: on one shared thread the
first stall made a later real-speech CONTROL hang, and "あ"+"は"*59 read as fatal
where a fresh thread measures it at 3.4 s. So each turn opens its own thread, a
real-speech canary follows every degenerate turn, and a canary that comes back
slow or empty restarts the server. A row is credible only if the canary behind
it is healthy, so `canary_s` rides every row.

Measured 2026-09-03 (seconds per turn, 30 s = stalled):

    chars | real speech | は    | 中央の | クラブの | アーメンの
       20 | 2.9         | 2.9   | 3.4   | 2.7     | 3.2
       60 | 3.9         | 3.4   | 4.8   | 2.9     | 3.5
      120 | 5.2         | STALL | 5.4   | 6.4     | 3.3
      240 | 6.0         | STALL | 10.8  | 4.6     | 5.0
      480 | 7.0         | STALL | STALL | STALL   | STALL

So it is repetition, not length: 480 characters of real speech cost 7.0 s, while
every unit up to 5 characters stalls at 480.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path[:0] = [str(ROOT)]

import live_stt  # noqa: E402  (after sys.path injection)

BOUND_S = 30.0
CANARY_MAX_S = 12.0  # an ordinary real-speech turn is ~3 s (D-011 p50 1.7-3.0 s)
SPANS = [20, 60, 120, 240, 480]
# Every unit was observed live: 'は' is session 2's laughter run (n=43), '中央の'
# its n=22 loop, 'クラブの' and 'アーメンの' are session 1 runaways at n=144 and n=130.
UNITS = ["は", "中央の", "クラブの", "アーメンの"]


def degenerate(unit: str, span: int) -> str:
    """n=43's shape generalized: one leading あ, then the unit repeated to span."""
    return ("あ" + unit * (span // len(unit) + 1))[:span]


def controls() -> dict[int, str]:
    """Real speech at each span, concatenated from the committed NPU caption trace."""
    trace = json.loads((TESTS / "caption_trace.json").read_text(encoding="utf-8"))
    captions = [c["text"] for c in trace["captions"]]
    out = {}
    for span in SPANS:
        text = ""
        for caption in captions[40:]:  # mid-story: narration, no section preamble
            if len(text) >= span:
                break
            text += caption
        out[span] = text[:span]
    return out


class Probe:
    """One live translator, restartable, one thread per measured turn."""

    def __init__(self) -> None:
        self.translator: live_stt.CodexTranslator | None = None

    async def restart(self) -> None:
        if self.translator is not None:
            await self.translator.close()
        self.translator = live_stt.CodexTranslator()
        if not await self.translator.start():
            raise SystemExit("codex app-server did not start; run `codex login` first")

    async def turn(self, ja: str) -> tuple[float, int, str]:
        """One turn on a fresh thread: (seconds, EN characters, status)."""
        t = self.translator
        assert t is not None
        while not t._notes.empty():  # nothing from a previous turn may bleed in
            t._notes.get_nowait()
        t._thread_id = await asyncio.wait_for(t._new_thread(), live_stt.CODEX_CONTROL_TIMEOUT_S * 2)
        started = time.perf_counter()
        try:
            en = await asyncio.wait_for(t._turn(ja), BOUND_S)
            return time.perf_counter() - started, len(en), "ok" if en else "EMPTY"
        except TimeoutError:
            elapsed = time.perf_counter() - started
            await t._abort_turn()
            return elapsed, 0, "STALL"
        except Exception as e:  # noqa: BLE001 - one bad turn never ends the matrix
            elapsed = time.perf_counter() - started
            await t._abort_turn()
            return elapsed, 0, f"error:{type(e).__name__}:{str(e)[:60]}"


async def run(out_path: Path | None) -> None:
    control = controls()
    probe = Probe()
    await probe.restart()
    print(f"bound {BOUND_S}s, canary ceiling {CANARY_MAX_S}s\n", flush=True)

    rows = []
    for span in SPANS:
        arms = [("control", "real-speech", control[span])]
        arms += [("degenerate", unit, degenerate(unit, span)) for unit in UNITS]
        for arm, unit, ja in arms:
            seconds, en_chars, status = await probe.turn(ja)
            canary_s, _, canary_status = await probe.turn(control[60])
            healthy = canary_status == "ok" and canary_s <= CANARY_MAX_S
            rows.append(
                {
                    "arm": arm,
                    "unit": unit,
                    "span": span,
                    "chars": len(ja),
                    "seconds": round(seconds, 2),
                    "en_chars": en_chars,
                    "status": status,
                    "canary_s": round(canary_s, 2),
                    "canary_ok": healthy,
                }
            )
            print(
                f"{arm:11s} {unit:6s} span={span:4d} {seconds:7.2f}s en={en_chars:5d} "
                f"{status:8s} canary={canary_s:6.2f}s {'ok' if healthy else 'DIRTY -> restart'}",
                flush=True,
            )
            if not healthy:
                await probe.restart()

    assert probe.translator is not None
    await probe.translator.close()
    stalled = [r for r in rows if r["status"] == "STALL"]
    print(
        f"\n{len(rows)} turns, {len(stalled)} stalled, "
        f"{sum(not r['canary_ok'] for r in rows)} dirty rows"
    )
    if out_path:
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the matrix to this path")
    asyncio.run(run(parser.parse_args().json))


if __name__ == "__main__":
    main()
