"""Run all three prototypes through the canonical 45-min scenario and report.

Usage: uv run python spike/t3_2/bench.py
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SPIKE_DIR = Path(__file__).parent
if str(SPIKE_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKE_DIR))

import harness  # noqa: E402
import scenarios  # noqa: E402


def make_args(**overrides):
    base = {
        "model": "gemini-3.1-flash-live-preview",
        "no_translate": False,
        "output": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _patch_for_bench(proto_mod):
    """Drop all wall-clock sleeps to bench-friendly values."""
    if hasattr(proto_mod, "RECONNECT_BACKOFF_S"):
        proto_mod.RECONNECT_BACKOFF_S = 0.05
    if hasattr(proto_mod, "EXTRACTOR_INTERVAL_S"):
        proto_mod.EXTRACTOR_INTERVAL_S = 0.3

    async def _bench_meter(state, _print_lock, _audio_q):
        while not state.stopping and not state.should_reconnect:
            await asyncio.sleep(0.02)

    proto_mod.meter = _bench_meter

    # Wrap entity_extractor so the interval override actually takes effect
    # (the original binds interval_s as a default argument at def time).
    if hasattr(proto_mod, "entity_extractor"):
        orig = proto_mod.entity_extractor

        async def _fast_ee(state, client, interval_s=0.3):
            return await orig(state, client, interval_s=interval_s)

        proto_mod.entity_extractor = _fast_ee


def _capture_state(proto_mod):
    """Monkey-patch State.__init__ so the bench can reach into state mid-run."""
    holder: list = []
    orig_init = proto_mod.State.__init__

    def _init(self, *a, **kw):
        orig_init(self, *a, **kw)
        holder.append(self)

    proto_mod.State.__init__ = _init
    return holder, orig_init


async def _run_one(
    proto_name: str,
    proto_mod,
    wall_budget_s: float,
    scripts=None,
    prior_history: list[dict] | None = None,
):
    _patch_for_bench(proto_mod)
    state_holder, orig_state_init = _capture_state(proto_mod)

    scripts = scripts or scenarios.SCENARIO_45MIN_SCRIPTS
    factory = harness.mock_client_factory(
        scripts,
        entity_response_fn=scenarios.entity_response_fn,
    )
    args = make_args()

    # Hydrate state.recent_blocks immediately after State() is created, to
    # simulate a process restart where prior history was loaded from disk.
    if prior_history:
        orig_init = proto_mod.State.__init__

        def _init_with_history(self, *a, **kw):
            orig_init(self, *a, **kw)
            if hasattr(self, "recent_blocks"):
                for b in prior_history:
                    self.recent_blocks.append(b)

        proto_mod.State.__init__ = _init_with_history

    async def _body():
        with contextlib.redirect_stdout(io.StringIO()):
            return await proto_mod.run_session(
                args,
                api_key="mock",
                client_factory=factory,
                stream_factory=harness.fake_stream_factory,
            )

    run_task = asyncio.create_task(_body())

    async def _watchdog():
        # Signal stopping the instant all scripts have completed. Any earlier
        # pad just lets the prototype start another (refused) reconnect cycle
        # before we can stop it, inflating reconnect_count in the report.
        start = time.monotonic()
        while time.monotonic() - start < wall_budget_s:
            mc = factory.holder.get("client")
            if mc is not None and mc.all_scripts_done.is_set():
                break
            await asyncio.sleep(0.02)
        if state_holder:
            for s in state_holder:
                s.stopping = True
                s.should_reconnect = True

    wd = asyncio.create_task(_watchdog())

    t_start = time.monotonic()
    try:
        state = await asyncio.wait_for(run_task, timeout=wall_budget_s)
        err = None
    except Exception as e:
        state = state_holder[-1] if state_holder else None
        err = e
    t_elapsed = time.monotonic() - t_start

    wd.cancel()
    try:
        await wd
    except (asyncio.CancelledError, Exception):
        pass

    # Restore State.__init__
    proto_mod.State.__init__ = orig_state_init

    client = factory.holder.get("client")
    return SimpleNamespace(
        name=proto_name,
        state=state,
        client=client,
        elapsed_s=t_elapsed,
        err=err,
    )


def summarize(results: list[SimpleNamespace], expected: int | None = None) -> str:
    if expected is None:
        expected = scenarios.count_expected_blocks()
    lines: list[str] = []
    header = (
        "| proto | blocks_ok | reconnects | sessions | seed_events | seed_bytes | "
        "client_content | extractor_runs | entities | wall_s | err |"
    )
    sep = "|" + "|".join(["---"] * 11) + "|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        if r.state is None:
            lines.append(
                f"| {r.name} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | "
                f"{r.elapsed_s:.2f} | {type(r.err).__name__ if r.err else '-'} |"
            )
            continue
        s = r.state
        blocks = s.transcripts_emitted
        reconnects = s.reconnect_count
        sessions = len(r.client.log.sessions) if r.client else 0
        seed_events = getattr(s, "seed_events", 0)
        seed_bytes = getattr(s, "seed_bytes_sent", 0)
        ent_runs = getattr(s, "extractor_runs", 0)
        ent_count = getattr(s, "entity_count", 0)
        cc_calls = (
            sum(len(rec.client_content_sent) for rec in r.client.log.sessions)
            if r.client
            else 0
        )
        err = type(r.err).__name__ if r.err else "-"
        lines.append(
            f"| {r.name} | {blocks}/{expected} | {reconnects} | {sessions} | "
            f"{seed_events} | {seed_bytes} | {cc_calls} | {ent_runs} | "
            f"{ent_count} | {r.elapsed_s:.2f} | {err} |"
        )
    lines.append("")
    c_result = next((r for r in results if r.name == "C"), None)
    if c_result and c_result.state and hasattr(c_result.state, "entities"):
        captured = c_result.state.entities.entities
        gt = {e["surface"] for e in scenarios.ENTITY_GROUND_TRUTH}
        got = set(captured.keys())
        tp = len(got & gt)
        fp = len(got - gt)
        fn = len(gt - got)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        lines.append("## Entity extraction (C only)")
        lines.append(
            f"- TP={tp}, FP={fp}, FN={fn}; "
            f"precision={prec:.2f}, recall={rec:.2f}, F1={f1:.2f}"
        )
        lines.append(f"- Captured: {sorted(got)}")
        lines.append(f"- Missed:   {sorted(gt - got)}")
        lines.append(f"- Spurious: {sorted(got - gt)}")
        lines.append("")
    lines.append("## Per-session mock log")
    for r in results:
        if not r.client:
            continue
        lines.append(f"### {r.name}")
        for i, rec in enumerate(r.client.log.sessions):
            cc = rec.client_content_sent
            cc_summary = (
                "no seed"
                if not cc
                else f"{len(cc)} seed(s), turns={sum(c.turn_count for c in cc)}, "
                f"text_bytes={sum(c.text_bytes for c in cc)}"
            )
            lines.append(
                f"- session {i}: handle_in={rec.handle_passed!r}, "
                f"msgs_emitted={rec.messages_emitted}, "
                f"audio_chunks_sent={len(rec.audio_sent)}, {cc_summary}"
            )
    return "\n".join(lines)


async def main():
    wall_budget_s = 45.0

    sections: list[str] = []

    # Benchmark 1: 45-min normal session (goAway + unexpected close).
    print("\n### Benchmark 1: 45-min normal session ###", flush=True)
    results_main = []
    for name, module in [("A", "prototype_a"), ("B", "prototype_b"), ("C", "prototype_c")]:
        mod = importlib.reload(importlib.import_module(module))
        print(f"\n=== {name} ===", flush=True)
        r = await _run_one(name, mod, wall_budget_s)
        if r.err:
            print(f"  err: {type(r.err).__name__}: {r.err}", flush=True)
        if r.state:
            print(
                f"  blocks={r.state.transcripts_emitted} "
                f"reconnects={r.state.reconnect_count} "
                f"elapsed={r.elapsed_s:.2f}s",
                flush=True,
            )
        results_main.append(r)

    sections.append(
        "## Benchmark 1: 45-min normal session\n\n"
        "Scenario: 3 sessions, goAway between 1-2, unexpected close between 2-3, "
        "resumption handle carried throughout (no cold reconnects).\n\n"
        + summarize(results_main)
    )

    # Benchmark 2: process-restart with prior history in memory.
    print("\n### Benchmark 2: cold start with prior history ###", flush=True)
    results_cold = []
    for name, module in [("A", "prototype_a"), ("B", "prototype_b"), ("C", "prototype_c")]:
        mod = importlib.reload(importlib.import_module(module))
        print(f"\n=== {name} ===", flush=True)
        r = await _run_one(
            name,
            mod,
            wall_budget_s,
            scripts=scenarios.SCENARIO_COLD_START_SCRIPTS,
            prior_history=scenarios.PRIOR_HISTORY_BLOCKS,
        )
        if r.err:
            print(f"  err: {type(r.err).__name__}: {r.err}", flush=True)
        if r.state:
            print(
                f"  blocks={r.state.transcripts_emitted} "
                f"elapsed={r.elapsed_s:.2f}s",
                flush=True,
            )
        results_cold.append(r)

    sections.append(
        "## Benchmark 2: cold start with prior history\n\n"
        f"Scenario: fresh State (handle=None), but recent_blocks pre-populated "
        f"with {len(scenarios.PRIOR_HISTORY_BLOCKS)} prior JA/EN pairs (simulating "
        f"restore-from-disk). One scripted session of 3 blocks. This is the only "
        f"case where prototype B's cold-seed path fires.\n\n"
        + summarize(results_cold, expected=scenarios.count_expected_blocks_cold_start())
    )

    report = "# T3.2 Benchmark Results\n\n" + "\n\n".join(sections)
    out = SPIKE_DIR / "results.md"
    out.write_text(report, encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    asyncio.run(main())
