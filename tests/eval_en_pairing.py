#!/usr/bin/env python3
"""Does a real translator ever hand the learner an English spelling? (M12.5)

M12.3 found two dead pairings, but it simulated the English side at the
learner's BEST case: every pairing opening answered with an agreeing proper
noun, by construction. `observe_en` acquires a rendering only where the English
caption carries a proper noun that is not sentence-initial (D-015), and both
keys that screen flagged are ORDINARY NOUNS in English -- 鼻腔 = "nasal cavity",
イワシ = "sardine" -- so the pairing they lose may never have existed at all.

This replays the committed 215-caption stream through the REAL production pair,
`SessionContext` + `CodexTranslator` over a live `codex app-server`, in
production's own order (observe_ja -> _translate -> observe_en), and reports what
each trust episode acquires from real English.

**The positive control is what makes a refutation readable.** ゴン (M12.4) is the
one key here whose English rendering IS a proper noun. ゴン pairing while
鼻腔/イワシ do not is the structural result; NOTHING pairing indicts this harness
instead of the mode.

Two modes, and the cheap one is the default because the artifact is committed:

    uv run python tests/eval_en_pairing.py            # re-derive the verdict, <1 s, no codex
    uv run python tests/eval_en_pairing.py --live     # spend ~215 real turns, rewrite the trace

`--live` writes `tests/en_pairing_trace.json`; the default re-derives every
verdict from it plus `caption_trace.json`, so the ruling reruns from a clean
clone with no codex, no model and no accelerator. Episode bookkeeping is
`eval_term_census.learner`, shared with M12.3, so the two tables are comparable
by construction rather than by restatement.

The JA side is arm-independent: `observe_ja` never reads `renderings`, so
`trusted_at`, `expired_at`, `mechanism` and the sighting counts are identical in
both arms and only the pairing columns can move. `--json` reports that check.

Model output is sampled, so the trace is ONE session. The verdict it supports is
structural -- an English common noun supplies no proper noun to pair -- never a
rate. One deviation from production is recorded per run: a translator disabled by
TRANSLATE_MAX_FAILURES is restarted against the same `SessionContext` rather than
running JA-only for the rest of the replay, because this measures the learner and
not the degradation path (`tests/test_translator.py` owns that). A run reporting
`restarts: 0` is production-identical.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path[:0] = [str(TESTS), str(ROOT)]

import live_stt  # noqa: E402
from live_stt import CodexTranslator, SessionContext, _en_names  # noqa: E402
from tests.eval_term_census import learner, pinned_sections  # noqa: E402

TRACE = TESTS / "caption_trace.json"
MANIFEST = TESTS / "long_form.json"
TURNS = TESTS / "en_pairing_trace.json"

# M12.3's dead-pairing candidates, and M12.4's positive control. Named here
# because the verdict is about these three and a reader should not have to
# recover them from a table of eight.
CANDIDATES = ("鼻腔", "イワシ")
CONTROL = "ゴン"


def caption_stream() -> list[dict]:
    """The committed captions, hash-gated against the pinned corpus."""
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pinned_sections(manifest, trace)  # refuses a trace decoded from another WAV
    if trace["run"]["hotwords_reachable"]:
        raise SystemExit(
            "this trace reached the recogniser with hotwords, so a replay with an empty "
            "`prompted` set would credit prompted sightings as evidence (D-015)"
        )
    return trace["captions"]


def recorded(turns: list[dict]) -> dict:
    """Index a live run's turns by caption, refusing one bound to other captions."""
    by_idx = {turn["idx"]: turn for turn in turns}
    if len(by_idx) != len(turns):
        raise SystemExit("the turn trace repeats a caption index")
    return by_idx


def replay(captions: list[dict], turns: list[dict]) -> dict:
    """Re-derive the trust episodes from a live run's recorded English."""
    by_idx = recorded(turns)
    for caption in captions:
        turn = by_idx.get(caption["idx"])
        if turn is None:
            raise SystemExit(f"caption {caption['idx']} was never translated in this trace")
        if turn["ja"] != caption["text"]:
            raise SystemExit(
                f"turn {caption['idx']} was translated from text this trace no longer "
                "carries; rerun --live against the current caption trace"
            )
    return learner(captions, lambda caption, unpaired: by_idx[caption["idx"]]["en"])


def verdict(captions: list[dict], turns: list[dict]) -> dict:
    """M12.3's simulated pairings against the live ones, episode by episode."""
    best = {e["term"]: e for e in learner(captions)["episodes"]}
    live = {e["term"]: e for e in replay(captions, turns)["episodes"]}
    by_idx = recorded(turns)
    # Every field the English side can move, `openings_before_quiet` included --
    # it counts openings, so leaving it out of this set would report the arms as
    # differing in TRUST when only the pairing moved.
    moving = {
        "openings",
        "openings_before_quiet",
        "paired_at",
        "rendering",
        "sightings_after_paired",
        "dead_pairing",
    }
    rows = []
    for term, episode in live.items():
        rows.append(
            {
                "term": term,
                "trusted_at": episode["trusted_at"],
                "expired_at": episode["expired_at"],
                "mechanism": episode["mechanism"],
                "openings": {
                    "best": len(best[term]["openings"]),
                    "live": len(episode["openings"]),
                },
                "paired_at": {"best": best[term]["paired_at"], "live": episode["paired_at"]},
                "rendering": episode["rendering"],
                "dead_pairing": {
                    "best": best[term]["dead_pairing"],
                    "live": episode["dead_pairing"],
                },
                # What the translator actually offered where the JA gate was open.
                # A rendering needs exactly one proper noun here, twice over.
                "names_at_openings": [
                    {
                        "caption": idx,
                        "names": list(dict.fromkeys(_en_names(by_idx[idx]["en"]))),
                        "en": by_idx[idx]["en"],
                    }
                    for idx in episode["openings"]
                ],
            }
        )
    return {
        "n_episodes": len(rows),
        "episodes": rows,
        # The JA side never reads `renderings`, so trust itself cannot move
        # between arms; measuring that is what confines the comparison to
        # pairing.
        "trust_identical": all(
            {k: v for k, v in live[t].items() if k not in moving}
            == {k: v for k, v in best[t].items() if k not in moving}
            for t in live
        ),
        "same_episodes": sorted(live) == sorted(best),
        "dead_pairings": {
            "best": [t for t, e in best.items() if e["dead_pairing"]],
            "live": [t for t, e in live.items() if e["dead_pairing"]],
        },
        "paired": {
            "best": [t for t, e in best.items() if e["paired_at"] is not None],
            "live": [t for t, e in live.items() if e["paired_at"] is not None],
        },
        "control_paired": live[CONTROL]["paired_at"] is not None if CONTROL in live else None,
        "candidates_paired": [t for t in CANDIDATES if t in live and live[t]["paired_at"]],
    }


async def live_turns(captions: list[dict]) -> dict:
    """Translate every caption through the real translator, production's way."""
    context = SessionContext()
    translator = CodexTranslator(context)
    if not await translator.start():
        raise SystemExit("codex app-server did not start; run `codex login` first")
    turns: list[dict] = []
    restarts = rotations = declined = 0
    started = time.perf_counter()
    for caption in captions:
        ja = caption["text"]
        if not ja:
            continue  # production publishes -- and so observes -- nothing else
        context.observe_ja(ja)  # `prompted` is empty on the NPU: nothing is ever biased
        span = live_stt.repeat_span(ja)
        if span >= live_stt.CAPTION_REPEAT_MAX_CHARS:
            declined += 1  # M13.1's screen, which lives in submit ahead of the queue
            turns.append({"idx": caption["idx"], "ja": ja, "en": "", "s": 0.0, "declined": True})
            continue
        brief = translator._brief
        at = time.perf_counter()
        en = await translator._translate(ja)
        seconds = time.perf_counter() - at
        rotated = translator._brief != brief
        rotations += rotated
        if en:
            context.observe_en(ja, en)
        turns.append(
            {
                "idx": caption["idx"],
                "ja": ja,
                "en": en,
                "s": round(seconds, 2),
                "rotated": rotated,
            }
        )
        print(
            f"{caption['idx']:>4} {seconds:6.2f}s {'ROTATE ' if rotated else ''}"
            f"{en[:88] or '(no translation)'}",
            flush=True,
        )
        if not translator.enabled:
            # Production would run JA-only from here; this measures the learner,
            # so the leg is rebuilt against the same context and the deviation
            # is reported rather than smoothed.
            print(f"  translator disabled after {caption['idx']}; restarting", flush=True)
            await translator.close()
            translator = CodexTranslator(context)
            if not await translator.start():
                raise SystemExit("codex app-server would not restart mid-replay")
            restarts += 1
    await translator.close()
    latencies = sorted(t["s"] for t in turns if t.get("en"))
    # The live loop's own context and a fresh replay of the recorded pairs must
    # agree: they run the same two observers over the same inputs, so a
    # divergence is a harness defect, not a finding.
    derived = learner(captions, lambda caption, unpaired: recorded(turns)[caption["idx"]]["en"])
    if {e["term"]: e["rendering"] for e in derived["episodes"] if e["rendering"]} != {
        t: r for t, r in context.renderings.items()
    } and not restarts:
        raise SystemExit("the replay disagrees with the live context; the harness is wrong")
    return {
        "source": {"trace": TRACE.name, "n_captions": len(captions)},
        "run": {
            "model": live_stt.TRANSLATE_MODEL,
            "effort": live_stt.TRANSLATE_EFFORT,
            "service_tier": live_stt.TRANSLATE_SERVICE_TIER,
            "instructions_sha256": hashlib.sha256(
                live_stt.TRANSLATOR_INSTRUCTIONS.encode()
            ).hexdigest()[:16],
            "n_turns": len(turns),
            "n_translated": len(latencies),
            "n_empty": sum(1 for t in turns if not t["en"] and not t.get("declined")),
            "n_declined": declined,
            "rotations": rotations,
            "restarts": restarts,
            "wall_s": round(time.perf_counter() - started, 1),
            "latency_s": {
                "p50": round(statistics.median(latencies), 2) if latencies else None,
                "p90": round(latencies[int(len(latencies) * 0.9)], 2) if latencies else None,
                "max": round(latencies[-1], 2) if latencies else None,
            },
        },
        "turns": turns,
    }


def report(result: dict, run: dict | None) -> None:
    if run:
        print(
            f"{run['n_turns']} captions: {run['n_translated']} translated, {run['n_empty']} "
            f"failed, {run['n_declined']} declined as repetition (M13.1), "
            f"{run['rotations']} thread rotations, {run['restarts']} restarts, "
            f"{run['wall_s']:.0f}s wall"
        )
        latency = run["latency_s"]
        print(
            f"  {run['model']}/{run['effort']} tier={run['service_tier']}  EN latency "
            f"p50 {latency['p50']}s  p90 {latency['p90']}s  max {latency['max']}s"
        )
    print(
        f"  {result['n_episodes']} trust episodes; the JA side is arm-independent: "
        f"same episodes={result['same_episodes']}, trust identical={result['trust_identical']}"
    )
    print(
        f"    {'term':<10} {'trusted@':>8} {'expired@':>8} {'open':>9} {'paired@':>13} "
        f"{'dead':>9}  rendering"
    )
    for row in result["episodes"]:
        openings = f"{row['openings']['best']}/{row['openings']['live']}"
        paired = f"{row['paired_at']['best']}/{row['paired_at']['live']}"
        dead = f"{int(row['dead_pairing']['best'])}/{int(row['dead_pairing']['live'])}"
        mark = "  <- CONTROL" if row["term"] == CONTROL else ""
        mark = "  <- CANDIDATE" if row["term"] in CANDIDATES else mark
        print(
            f"    {row['term']:<10} {row['trusted_at']:>8} {str(row['expired_at']):>8} "
            f"{openings:>9} {paired:>13} {dead:>9}  {row['rendering'] or '-'}{mark}"
        )
    print("    (best-case simulation / live translator)")
    for row in result["episodes"]:
        if row["term"] not in (*CANDIDATES, CONTROL):
            continue
        print(f"  {row['term']}: {len(row['names_at_openings'])} pairing openings")
        for opening in row["names_at_openings"]:
            print(
                f"    caption {opening['caption']:>4} names={opening['names'] or 'none'}  "
                f"{opening['en'][:100]}"
            )
    learned = [
        f"{r['term']} = {r['rendering']}" for r in result["episodes"] if r["rendering"] is not None
    ]
    print(
        f"  learned live: {', '.join(learned) or 'NOTHING'}\n"
        f"  dead pairings: best case {result['dead_pairings']['best']}, "
        f"live {result['dead_pairings']['live'] or 'NONE'}"
    )
    if result["control_paired"] and not result["candidates_paired"]:
        print(
            f"  VERDICT: structural refutation. The control {CONTROL} pairs, so the harness "
            f"acquires renderings; neither {'/'.join(CANDIDATES)} does, so the dead pairings "
            "M12.3 screened do not exist against a real translator."
        )
    elif result["candidates_paired"]:
        print(
            f"  VERDICT: {'/'.join(result['candidates_paired'])} really pairs, so a rendering "
            "is acquired and then lost; P-012's arm matrix is warranted."
        )
    else:
        print(
            f"  VERDICT: nothing pairs, {CONTROL} included. That indicts this harness, not the "
            "mode -- the candidates' nulls carry no weight until the control pairs."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--live", action="store_true", help="Spend real turns and rewrite the trace.")
    ap.add_argument("--json", action="store_true", help="Emit the verdict as JSON.")
    args = ap.parse_args()

    captions = caption_stream()
    if args.live:
        trace = asyncio.run(live_turns(captions))
        TURNS.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {TURNS}")
    else:
        if not TURNS.exists():
            raise SystemExit(f"no live run recorded at {TURNS}; produce one with --live")
        trace = json.loads(TURNS.read_text(encoding="utf-8"))
    result = verdict(captions, trace["turns"])
    if args.json:
        print(json.dumps({**result, "run": trace["run"]}, ensure_ascii=False, indent=2))
        return
    report(result, trace["run"])


if __name__ == "__main__":
    main()
