"""Regression locks for CodexTranslator's degradation contract (D-009, D-011).

The happy-path live turn needs a real `codex app-server` + auth and stays a
user smoke (L-004). What is locked here is the failure surface a refactor can
silently break: 3-strike session disable, backlog eviction, the `_read_loop`
dispatch/EOF branches plus its oversized-line/broken-transport guard (the sole
non-local input boundary, T6-hardened), and M13.1's degeneracy screen. All in
memory — fake stdio over an asyncio.StreamReader, `asyncio.run` per test, no
subprocess, no mic, no new dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

import live_stt

TESTS = Path(__file__).resolve().parent


class _FakeStdin:
    """Captures the JSON-RPC the translator writes; `close()` for graceful shutdown."""

    def __init__(self):
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes):
        self.writes.append(data)

    def close(self):
        self.closed = True


class _FakeProc:
    """Minimal asyncio.subprocess.Process stand-in: a real StreamReader stdout
    feeds `_read_loop`; `wait`/`kill` let `close()` run its shutdown path."""

    def __init__(self, stdout: asyncio.StreamReader):
        self.stdout = stdout
        self.stdin = _FakeStdin()
        self.returncode: int | None = None

    async def wait(self):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


def _rpc_result(rid: int, result: dict) -> bytes:
    """A JSON-RPC response line, framed as _read_loop expects (one line, \\n)."""
    return (json.dumps({"id": rid, "result": result}) + "\n").encode()


def _rpc_note(method: str, params: dict) -> bytes:
    """A JSON-RPC notification line (no id) -> lands in _notes."""
    return (json.dumps({"method": method, "params": params}) + "\n").encode()


async def _await_pending(t: live_stt.CodexTranslator, rid: int, spins: int = 1000):
    """Yield until request `rid` is registered in _pending (issued and awaiting a
    response). Lets a feeder answer requests the production way -- through
    _read_loop -- instead of reaching past it with set_result."""
    for _ in range(spins):
        await asyncio.sleep(0)
        if rid in t._pending:
            return
    raise AssertionError(f"request id {rid} was never issued")


def test_consecutive_failures_disable_then_reset():
    # D-009 hard requirement: transient turn failures degrade per-block, but
    # TRANSLATE_MAX_FAILURES in a row must flip the session to JA-only; a single
    # success must reset the streak. A regression that never disables hangs every
    # block; one that never resets disables a healthy leg after a transient blip.
    async def scenario():
        t = live_stt.CodexTranslator()
        t.enabled = True
        t._proc = None  # _abort_turn early-returns -> no interrupt write needed

        async def boom(_ja):
            raise RuntimeError("turn failed")

        t._turn = boom  # type: ignore[assignment]
        n = live_stt.TRANSLATE_MAX_FAILURES
        for i in range(1, n + 1):
            assert await t._translate(f"x{i}") == ""
            assert t._failures == i
            assert t.enabled is (i < n)  # still enabled until the nth failure

        t.enabled = True  # operator-independent re-enable for the reset check

        async def ok(_ja):
            return "hello"

        t._turn = ok  # type: ignore[assignment]
        assert await t._translate("y") == "hello"
        assert t._failures == 0

    asyncio.run(scenario())


def test_submit_evicts_oldest_and_counts():
    # Backlog overflow drops the STALEST caption (newest beats oldest) and bumps
    # the eviction counter the meter surfaces as tdrop= (T8.5). An inverted
    # eviction would silently drop the freshest caption.
    async def scenario():
        t = live_stt.CodexTranslator()
        t.enabled = True
        cap = live_stt.TRANSLATE_QUEUE_MAX
        for i in range(cap):
            t.submit(i, f"ja{i}")
        assert t.queue.qsize() == cap
        assert t.dropped_translations == 0

        t.submit(cap, f"ja{cap}")  # one past full
        assert t.queue.qsize() == cap  # still capped
        assert t.dropped_translations == 1

        seqs = []
        while not t.queue.empty():
            seq, _ = t.queue.get_nowait()
            seqs.append(seq)
        assert 0 not in seqs  # oldest evicted
        assert cap in seqs  # newest survived
        assert len(seqs) == cap

    asyncio.run(scenario())


def test_submit_sentinel_lands_on_full_queue():
    # Shutdown must enqueue the None sentinel even when the backlog is full,
    # evicting to make room (mirrors the audio-side T8.1 idiom) — a blocking put
    # would hang shutdown.
    async def scenario():
        t = live_stt.CodexTranslator()
        t.enabled = True
        cap = live_stt.TRANSLATE_QUEUE_MAX
        for i in range(cap):
            t.submit(i, f"ja{i}")
        assert t.queue.qsize() == cap

        t.submit_sentinel()
        assert t.queue.qsize() == cap  # still capped, room made by eviction
        items = [t.queue.get_nowait() for _ in range(cap)]
        assert items.count(None) == 1  # exactly one sentinel landed

    asyncio.run(scenario())


def test_read_loop_dispatch_and_eof():
    # The one non-local input boundary (T6). Lock each dispatch branch and the
    # EOF cleanup: malformed line skipped; server request auto-denied; id+result
    # resolves a pending future; a notification lands in _notes; EOF flips
    # enabled off, fails remaining pending futures, and (T8.3) enqueues the wake
    # sentinel after the real notes.
    async def scenario():
        reader = asyncio.StreamReader()
        proc = _FakeProc(reader)
        t = live_stt.CodexTranslator()
        t._proc = proc  # type: ignore[assignment]
        t.enabled = True

        loop = asyncio.get_running_loop()
        resolved = loop.create_future()
        t._pending[7] = resolved

        reader.feed_data(b"not json\n")  # skipped
        reader.feed_data(
            json.dumps({"id": 99, "method": "applyPatchApproval", "params": {}}).encode() + b"\n"
        )  # server request -> auto-deny
        reader.feed_data(json.dumps({"id": 7, "result": {"ok": 1}}).encode() + b"\n")
        reader.feed_data(
            json.dumps({"method": "item/agentMessage/delta", "params": {"delta": "x"}}).encode()
            + b"\n"
        )  # notification -> _notes

        orphan = loop.create_future()
        t._pending[8] = orphan  # no response arrives; EOF must fail it
        reader.feed_eof()
        await t._read_loop()

        assert resolved.result() == {"ok": 1}
        assert any(b'"denied"' in w for w in proc.stdin.writes)
        first = t._notes.get_nowait()
        assert first["method"] == "item/agentMessage/delta"
        sentinel = t._notes.get_nowait()  # T8.3 wake sentinel, after the real note
        assert sentinel["method"] == "error"
        assert t.enabled is False
        assert orphan.done() and isinstance(orphan.exception(), RuntimeError)

    asyncio.run(scenario())


@pytest.mark.parametrize("trigger", ["oversized", "broken"])
def test_read_loop_input_boundary_degrades(trigger):
    # T6 hardening (the sole non-local input boundary): an oversized line
    # (ValueError from the 64 KiB readline limit) or a broken transport (OSError)
    # must route into the SAME post-loop cleanup as EOF -> JA-only, not crash the
    # reader task. Locks the `except (ValueError, OSError)` guard -- dropping
    # either type lets the exception escape and strands the degrade. EOF entry is
    # covered by test_read_loop_dispatch_and_eof.
    async def scenario():
        stdout = asyncio.StreamReader()  # default 64 KiB limit
        if trigger == "oversized":
            stdout.feed_data(b"x" * (2**16 + 16))  # no newline -> readline() raises ValueError
        else:

            async def _broken(*_a, **_k):
                raise OSError("transport closed")

            stdout.readline = _broken  # type: ignore[assignment]
        t = live_stt.CodexTranslator()
        t._proc = _FakeProc(stdout)  # type: ignore[assignment]
        t.enabled = True
        orphan = asyncio.get_running_loop().create_future()
        t._pending[1] = orphan  # the shared cleanup must fail it

        await t._read_loop()  # returns (degrades), does not raise

        assert t.enabled is False  # session degraded to JA-only
        assert orphan.done() and isinstance(orphan.exception(), RuntimeError)
        assert t._notes.get_nowait()["method"] == "error"  # T8.3 wake sentinel enqueued

    asyncio.run(scenario())


def test_turn_wakes_on_eof_under_timeout():
    # T8.3: codex dies after turn/start resolves but before turn/completed, so
    # the turn is parked on _notes.get() with no pending request to fail. The
    # EOF sentinel must wake it; without it the collect loop blocks until the
    # outer TRANSLATE_TIMEOUT_S (15 s). A 2 s bound proves "prompt".
    async def scenario():
        reader = asyncio.StreamReader()
        t = live_stt.CodexTranslator()
        t._proc = _FakeProc(reader)  # type: ignore[assignment]
        t._thread_id = "thread-1"
        t.enabled = True

        reader_task = asyncio.create_task(t._read_loop())
        turn_task = asyncio.create_task(t._turn("テスト"))

        for _ in range(50):  # let _turn issue turn/start
            await asyncio.sleep(0)
            if t._pending:
                break
        assert t._pending, "turn/start was never issued"
        next(iter(t._pending.values())).set_result({})  # advance into collect loop

        reader.feed_eof()  # codex dies mid-turn
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(turn_task, timeout=2.0)
        assert t.enabled is False
        await reader_task

    asyncio.run(scenario())


def test_translate_degrades_to_ja_only_on_eof_under_timeout():
    # End-to-end of T8.3 (the wake test above stops at _turn raising). Drive the
    # public _translate: resolve turn/start through _read_loop the production
    # way, then kill the server mid-turn. _translate must catch, degrade the
    # block to "" (JA-only), bump _failures, and flip the session off -- all well
    # under TRANSLATE_TIMEOUT_S (15 s). A 2 s bound proves "prompt".
    async def scenario():
        reader = asyncio.StreamReader()
        proc = _FakeProc(reader)
        t = live_stt.CodexTranslator()
        t._proc = proc  # type: ignore[assignment]
        t._thread_id = "thread-1"
        t.enabled = True

        reader_task = asyncio.create_task(t._read_loop())
        translate_task = asyncio.create_task(t._translate("テスト"))

        await _await_pending(t, 1)  # turn/start is the first request id
        reader.feed_data(_rpc_result(1, {}))  # resolve via the reader, not set_result
        proc.returncode = 0  # server is dead -> _abort_turn early-returns
        reader.feed_eof()  # codex dies mid-turn

        en = await asyncio.wait_for(translate_task, timeout=2.0)
        assert en == ""  # JA-only block, not a hang
        assert t.enabled is False  # session degraded
        assert t._failures == 1
        await reader_task

    asyncio.run(scenario())


def test_graceful_close_enqueues_no_sentinel():
    # The flip side of T8.3: close() cancels _reader_task mid-readline, raising
    # CancelledError that escapes the (ValueError, OSError) catch -> EOF cleanup
    # is never reached, so a clean shutdown enqueues no spurious wake sentinel.
    async def scenario():
        reader = asyncio.StreamReader()
        t = live_stt.CodexTranslator()
        t._proc = _FakeProc(reader)  # type: ignore[assignment]
        t.enabled = True
        t._reader_task = asyncio.create_task(t._read_loop())
        for _ in range(10):  # let the read loop park on readline()
            await asyncio.sleep(0)

        await t.close()
        try:
            await t._reader_task
        except asyncio.CancelledError:
            pass
        assert t._reader_task.cancelled()
        assert t._notes.empty()

    asyncio.run(scenario())


def test_eof_logs_once_and_disables(caplog):
    # T8.5: an EOF in an idle gap was the one silent, permanent degradation
    # (startup and 3-strike both log). The cleanup must log exactly one error
    # and flip enabled off.
    async def scenario():
        reader = asyncio.StreamReader()
        t = live_stt.CodexTranslator()
        t._proc = _FakeProc(reader)  # type: ignore[assignment]
        t.enabled = True
        reader.feed_eof()
        with caplog.at_level(logging.ERROR, logger="live_stt"):
            await t._read_loop()
        assert t.enabled is False
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert "JA-only" in errors[0].getMessage()

    asyncio.run(scenario())


def test_start_refuses_dead_server_after_warmup(monkeypatch):
    # T8.6: the warm-up turn completes, then the server dies before start()
    # enables (its turn/completed is consumed, the next readline hits EOF). The
    # EOF cleanup runs with enabled still False, so nothing logs and the only
    # trace is a finished reader task. start() must NOT enable over that corpse
    # -- doing so strands every later turn on a turn/start no one resolves until
    # TRANSLATE_TIMEOUT_S. Without the liveness guard this returns True.
    async def scenario():
        reader = asyncio.StreamReader()
        proc = _FakeProc(reader)

        async def fake_exec(*_a, **_k):
            return proc

        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", fake_exec)
        t = live_stt.CodexTranslator()

        async def feeder():
            await _await_pending(t, 1)  # initialize
            reader.feed_data(_rpc_result(1, {}))
            await _await_pending(t, 2)  # thread/start
            reader.feed_data(_rpc_result(2, {"thread": {"id": "th-1"}}))
            await _await_pending(t, 3)  # warm-up turn/start
            # Complete the warm-up turn, then die -- all buffered before the read
            # loop drains, so EOF is processed before start() reaches the guard.
            reader.feed_data(_rpc_result(3, {}))
            reader.feed_data(_rpc_note("item/agentMessage/delta", {"delta": "hi"}))
            reader.feed_data(_rpc_note("turn/completed", {}))
            reader.feed_eof()

        feeder_task = asyncio.create_task(feeder())
        ok = await asyncio.wait_for(t.start(), timeout=3.0)
        await feeder_task

        assert ok is False  # refused to enable a dead server
        assert t.enabled is False

    asyncio.run(scenario())


def test_new_thread_requests_service_tier_and_warns_when_not_applied(caplog):
    # Every GPT call rides the thread thread/start opens -- warm-up, each caption,
    # and each ~100-turn rotation -- so the tier is set once, here. Two ways this
    # silently reverts to the account default: the param stops being sent, or the
    # server drops an unrecognized tier (it answers null, never an error). Lock
    # the outgoing request AND the echo check that makes the drop visible.
    async def scenario():
        reader = asyncio.StreamReader()
        t = live_stt.CodexTranslator()
        t._proc = _FakeProc(reader)  # type: ignore[assignment]
        reader_task = asyncio.create_task(t._read_loop())

        async def open_thread(rid: int, echo: dict) -> str:
            async def feeder():
                await _await_pending(t, rid)
                reader.feed_data(_rpc_result(rid, {"thread": {"id": "th-1"}, **echo}))

            feeder_task = asyncio.create_task(feeder())
            tid = await asyncio.wait_for(t._new_thread(), timeout=3.0)
            await feeder_task
            return tid

        with caplog.at_level(logging.WARNING, logger="live_stt"):
            # Applied: the server echoes the tier back -> no warning.
            assert await open_thread(1, {"serviceTier": live_stt.TRANSLATE_SERVICE_TIER}) == "th-1"
            sent = json.loads(t._proc.stdin.writes[0])  # type: ignore[union-attr]
            assert sent["method"] == "thread/start"
            assert sent["params"]["serviceTier"] == live_stt.TRANSLATE_SERVICE_TIER
            assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

            # Dropped: the server reports null -> one warning names the miss.
            assert await open_thread(2, {"serviceTier": None}) == "th-1"

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "service tier" in warnings[0].getMessage()

        reader.feed_eof()
        await reader_task

    asyncio.run(scenario())


def test_glossary_rides_developer_instructions_not_the_turn_text():
    # The turn text is declared translatable input, so a glossary sent there comes
    # back translated instead of obeyed. developerInstructions outranks it.
    ctx = live_stt.SessionContext("神経内科の申し送り")
    for _ in range(live_stt.CONTEXT_TERM_SUPPORT):
        ctx.observe_ja("プレドニンを投与しました")
    t = live_stt.CodexTranslator(ctx)
    instructions = t._instructions()
    assert instructions.startswith(live_stt.TRANSLATOR_INSTRUCTIONS)
    assert "プレドニン" in instructions and "神経内科" in instructions
    assert t._brief == ctx.translator_brief()


def test_instructions_are_unchanged_without_context():
    t = live_stt.CodexTranslator()
    assert t._instructions() == live_stt.TRANSLATOR_INSTRUCTIONS
    assert t._brief == ""


def test_new_terms_rotate_the_thread_so_the_glossary_reaches_the_model():
    # Thread scope: a term learned at turn 5 must not wait out TRANSLATE_ROTATE_TURNS.
    async def scenario():
        ctx = live_stt.SessionContext()
        t = live_stt.CodexTranslator(ctx)
        t.enabled = True
        t._thread_id = "th-1"
        rotations = []

        async def fake_new_thread():
            rotations.append(t._instructions())
            return "th-2"

        async def fake_turn(ja):
            return "ok"

        t._new_thread = fake_new_thread  # type: ignore[method-assign]
        t._turn = fake_turn  # type: ignore[method-assign]

        assert await t._translate("こんにちは") == "ok"
        assert rotations == []  # empty glossary, nothing to refresh

        for _ in range(live_stt.CONTEXT_TERM_SUPPORT):
            ctx.observe_ja("プレドニンを投与しました")
        assert await t._translate("プレドニンです") == "ok"
        assert len(rotations) == 1 and "プレドニン" in rotations[0]

        assert await t._translate("もう一度") == "ok"
        assert len(rotations) == 1  # unchanged glossary does not rotate again

    asyncio.run(scenario())


# --- M13.1: decline a degenerate caption before it reaches the translator ----


def _runaway(unit: str, span: int) -> str:
    """One leading あ, then `unit` repeated to `span` — the measured shape."""
    return ("あ" + unit * (span // len(unit) + 1))[:span]


def _real_japanese() -> list[str]:
    """Every real caption committed in tree: 215 NPU captions + each golden text."""
    trace = json.loads((TESTS / "caption_trace.json").read_text(encoding="utf-8"))
    texts = [c["text"] for c in trace["captions"]]
    goldens = json.loads((TESTS / "replay_goldens.json").read_text(encoding="utf-8"))
    for clips in goldens.values():
        for row in clips.values():
            texts += [seg["text"] for seg in row["segments"]] + [row["ja_ref"]]
    return texts


def test_a_degenerate_caption_never_reaches_a_turn():
    # The screen sits in submit(), BEFORE the queue, so a declined caption cannot
    # reach _turn and cannot touch _failures — that placement is the whole unit.
    # Locked with it: the JA-side learner never keys a rendering on a runaway
    # (observe_en runs only on a translated block), and the eviction counter the
    # meter reads as backpressure stays untouched by a content decision.
    async def scenario():
        ctx = live_stt.SessionContext()
        t = live_stt.CodexTranslator(ctx)
        t.enabled = True
        t._thread_id = "th-1"
        turns, paired = [], []

        async def fake_turn(ja):
            turns.append(ja)
            return "Gon set out for Hyoju's house."

        t._turn = fake_turn  # type: ignore[assignment]
        ctx.observe_en = lambda ja, en: paired.append(ja)  # type: ignore[method-assign]

        real = "ごんは兵十のうちへ出かけました。"
        t.submit(1, real)
        t.submit(2, _runaway("は", 480))

        assert t.queue.qsize() == 1  # only the ordinary caption was enqueued
        assert t.degenerate_captions == 1
        assert t.dropped_translations == 0  # a content decision, never backpressure

        t.submit_sentinel()
        await t.run(None)

        assert turns == [real]  # the runaway never entered a turn
        assert paired == [real]  # nor observe_en (D-015)

    asyncio.run(scenario())


def test_a_runaway_streak_leaves_the_translation_leg_alive():
    # What killed session 1: n=195/196/197 were three consecutive runaways, so
    # three consecutive TimeoutErrors hit TRANSLATE_MAX_FAILURES and the last 47
    # turns of a 41-minute session were JA-only. Declining ahead of the queue
    # means a streak of any length costs no strike at all.
    t = live_stt.CodexTranslator()
    t.enabled = True
    streak = live_stt.TRANSLATE_MAX_FAILURES + 1

    for seq in range(streak):
        t.submit(seq, _runaway("次は、", 444))

    assert t.queue.empty()
    assert t.enabled is True
    assert t._failures == 0
    assert t.degenerate_captions == streak


def test_a_declined_caption_names_its_reason_once(caplog):
    # The JA line still prints and is still saved (the caption is evidence of what
    # was heard), so the missing EN needs a reason on stderr — session 2's whole
    # stderr was one `translation failed ()`, an empty TimeoutError str().
    t = live_stt.CodexTranslator()
    t.enabled = True

    with caplog.at_level(logging.WARNING, logger="live_stt"):
        t.submit(43, _runaway("は", 890))

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "43" in message  # which caption lost its EN line
    assert "889 of 890" in message  # and how much of it was one repeated unit


@pytest.mark.parametrize(
    "unit,span",
    [
        ("は", 120),  # smallest measured stall: 30 s bound, fresh thread per turn
        ("は", 240),
        ("は", 480),
        ("中央の", 480),
        ("クラブの", 480),
        ("アーメンの", 480),
        ("は", 890),  # session 2 n=43, the observed maximum caption
        ("次は、", 444),  # session 1 n=196, first of the three that killed the leg
        ("中央の", 333),  # session 2 n=22
    ],
)
def test_every_caption_measured_to_stall_the_translator_is_declined(unit, span):
    assert live_stt.repeat_span(_runaway(unit, span)) >= live_stt.CAPTION_REPEAT_MAX_CHARS


def test_the_screen_flags_no_real_caption():
    # The false-positive side, hardware-free and rerunnable from committed state:
    # across every real Japanese caption in tree the longest adjacent repetition
    # is 8 characters (ポンポンポンポン, an onomatopoeia the story itself uses),
    # five times under the threshold.
    spans = {text: live_stt.repeat_span(text) for text in _real_japanese()}

    assert max(spans.values()) == 8
    assert not [t for t, s in spans.items() if s >= live_stt.CAPTION_REPEAT_MAX_CHARS]


def test_the_threshold_is_a_boundary_and_the_unit_bound_is_real():
    # Two constants decide every verdict above, so pin each at its own edge.
    # A unit longer than CAPTION_REPEAT_UNIT_CHARS is a repeated PHRASE — a
    # speaker saying the same thing twice — while a decode loop repeats something
    # short, so the bound is what keeps a person out of the screen.
    limit = live_stt.CAPTION_REPEAT_MAX_CHARS
    unit = live_stt.CAPTION_REPEAT_UNIT_CHARS

    assert live_stt.repeat_span("ごんは兵十のうちへ出かけました。") == 0  # a span, not a length
    assert live_stt.repeat_span("あ" * (limit - 1)) == limit - 1
    assert live_stt.repeat_span("あ" * limit) == limit
    assert live_stt.repeat_span(("あいうえおかきくけこ"[:unit]) * 5) >= limit
    assert live_stt.repeat_span(("あいうえおかきくけこ"[: unit + 1]) * 5) < limit
