"""Regression locks for CodexTranslator's degradation contract (D-009, D-011).

The happy-path live turn needs a real `codex app-server` + auth and stays a
user smoke (L-004). What is locked here is the failure surface a refactor can
silently break: 3-strike session disable, backlog eviction, the `_read_loop`
dispatch/EOF branches plus its oversized-line/broken-transport guard (the sole
non-local input boundary, T6-hardened), the transcript marker + named cause that
make a degrade diagnosable afterwards, M13.1's degeneracy screen, and the
bounded recovery of a disabled leg — respawn when the app-server exited
(M14.2), probe when it outlived the turns (M14.3). All in memory — fake stdio over an
asyncio.StreamReader, `asyncio.run` per test, no subprocess, no mic, no new
dependency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
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


# --- P-015: the saved transcript must name a degrade, not just lack EN lines --


def _marker_lines(path: Path) -> list[str]:
    """Transcript lines that are neither JA nor EN, i.e. the degrade markers."""
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if " -- " in ln]


def test_the_three_strike_degrade_marks_the_transcript_once(tmp_path):
    # Session 1 went JA-only at n=194 and ran 47 more turns with no EN line and no
    # recorded cause; which of the two paths fired was recovered only by reading
    # the JA text. The marker must name the path, land once, and never repeat
    # afterwards — a marker per later caption would bury the transcript it saves.
    async def scenario():
        transcript = live_stt.TranscriptFile(tmp_path / "session.txt")
        t = live_stt.CodexTranslator(output_file=transcript)
        t.enabled = True
        t._proc = None  # _abort_turn early-returns -> no interrupt write needed

        async def timeout(_ja):
            raise TimeoutError  # what asyncio.wait_for raises around a stalled turn

        t._turn = timeout  # type: ignore[assignment]
        for _ in range(live_stt.TRANSLATE_MAX_FAILURES):
            assert await t._translate("テスト") == ""
        assert t.enabled is False
        for _ in range(3):  # every later caption after the flip
            assert await t._translate("テスト") == ""
        transcript.close()

    asyncio.run(scenario())
    markers = _marker_lines(tmp_path / "session.txt")
    assert len(markers) == 1
    assert markers[0].startswith("[")  # same timestamped shape as a JA/EN line
    assert "] -- translation disabled: " in markers[0]  # but outside their grammar
    assert f"{live_stt.TRANSLATE_MAX_FAILURES} consecutive failures" in markers[0]
    assert "TimeoutError" in markers[0]  # which path AND what failed


def test_the_eof_degrade_marks_the_transcript_once(tmp_path):
    # The other permanent path: codex dies in an idle gap. Feeding EOF twice
    # proves the marker tracks the enabled->disabled TRANSITION rather than the
    # event, so a second cleanup pass cannot re-mark a session already degraded.
    async def scenario():
        transcript = live_stt.TranscriptFile(tmp_path / "session.txt")
        t = live_stt.CodexTranslator(output_file=transcript)
        t.enabled = True
        for _ in range(2):
            reader = asyncio.StreamReader()
            reader.feed_eof()
            t._proc = _FakeProc(reader)  # type: ignore[assignment]
            await t._read_loop()
        assert t.enabled is False
        transcript.close()

    asyncio.run(scenario())
    markers = _marker_lines(tmp_path / "session.txt")
    assert len(markers) == 1
    assert "translation disabled: codex app-server exited" in markers[0]


def test_a_timed_out_turn_logs_the_cause_it_used_to_swallow(caplog, monkeypatch):
    # Session 2's entire stderr record of a lost EN line was `translation failed
    # ()`: TimeoutError carries no message, so %s formatted the cause away and not
    # even the word "timeout" survived. A real wait_for timeout here, so the test
    # locks the rendering of the exception production actually raises.
    async def scenario():
        t = live_stt.CodexTranslator()
        t.enabled = True
        t._proc = None

        async def hang(_ja):
            await asyncio.sleep(3600)

        t._turn = hang  # type: ignore[assignment]
        monkeypatch.setattr(live_stt, "TRANSLATE_TIMEOUT_S", 0.01)
        with caplog.at_level(logging.WARNING, logger="live_stt"):
            assert await t._translate("テスト") == ""
        assert t.enabled is True  # one block lost, the session survives

    asyncio.run(scenario())
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "TimeoutError" in warnings[0].getMessage()


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
    """Every real Japanese committed in tree: 215 NPU captions, each golden text,
    and the Aozora reference — 10.7 K characters no accelerator can change."""
    trace = json.loads((TESTS / "caption_trace.json").read_text(encoding="utf-8"))
    texts = [c["text"] for c in trace["captions"]]
    goldens = json.loads((TESTS / "replay_goldens.json").read_text(encoding="utf-8"))
    for clips in goldens.values():
        for row in clips.values():
            texts += [seg["text"] for seg in row["segments"]] + [row["ja_ref"]]
    long_form = json.loads((TESTS / "long_form.json").read_text(encoding="utf-8"))
    return texts + [s["reference"]["text"] for s in long_form["sections"].values()]


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
        await t.run()

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
        # The three the 8-character bound let through, each measured live and
        # each an escape only because its unit is longer than the old bound.
        ("いい音があるので、", 664),  # session 6 n=263, x68 — reached the translator
        ("キーパーソースになります", 53),  # session 5 n=114, x4 — the shortest true loop
        ("、彼女の人にとってもらえず", 443),  # session 1 n=227, x33 — hit max_length
    ],
)
def test_every_caption_measured_to_stall_the_translator_is_declined(unit, span):
    assert live_stt.repeat_span(_runaway(unit, span)) >= live_stt.CAPTION_REPEAT_MAX_CHARS


def test_the_screen_flags_no_real_caption():
    # The false-positive side, hardware-free and rerunnable from committed state.
    # Widening the unit bound to 13 raises the longest surviving repetition in
    # tree from 8 (ポンポンポンポン, an onomatopoeia the story itself uses) to 18 —
    # a speaker saying a 9-character phrase twice, which is what the bound now
    # reaches. It survives with better than a 2x margin, and nothing else moves.
    spans = {text: live_stt.repeat_span(text) for text in _real_japanese()}
    longest = max(spans, key=lambda t: spans[t])

    assert spans[longest] == 18
    assert "、うなぎが食べたい、うなぎが食べたい" in longest  # said twice, not looped
    assert not [t for t, s in spans.items() if s >= live_stt.CAPTION_REPEAT_MAX_CHARS]


def test_the_threshold_is_a_boundary_and_the_unit_bound_is_real():
    # Two constants decide every verdict above, so pin each at its own edge.
    # A unit longer than CAPTION_REPEAT_UNIT_CHARS is a repeated PHRASE — a
    # speaker saying the same thing twice — while a decode loop repeats something
    # short, so the bound is what keeps a person out of the screen.
    limit = live_stt.CAPTION_REPEAT_MAX_CHARS
    unit = live_stt.CAPTION_REPEAT_UNIT_CHARS
    distinct = "あいうえおかきくけこさしすせそ"  # >unit characters, none repeating
    assert len(distinct) > unit

    assert live_stt.repeat_span("ごんは兵十のうちへ出かけました。") == 0  # a span, not a length
    assert live_stt.repeat_span("あ" * (limit - 1)) == limit - 1
    assert live_stt.repeat_span("あ" * limit) == limit
    assert live_stt.repeat_span(distinct[:unit] * 4) >= limit
    assert live_stt.repeat_span(distinct[: unit + 1] * 4) == 0  # one character wider, invisible


def test_a_phrase_said_three_times_survives_at_every_size_the_screen_scans():
    # What the bound is FOR, stated as the invariant rather than as a number: a
    # drop needs ceil(limit/size) repeats, so the widest unit must still take
    # four. 13 is the last bound that does (3x13=39), which is why widening
    # stopped there — at 14 a tripled phrase drops, and a live caption shows the
    # shape (完全にどころから…x2 followed by a unique third sentence).
    unit = live_stt.CAPTION_REPEAT_UNIT_CHARS

    assert unit * 3 < live_stt.CAPTION_REPEAT_MAX_CHARS
    for size in range(1, unit + 1):
        phrase = ("あいうえおかきくけこさしすせそたちつてと" * 2)[:size]
        assert live_stt.repeat_span(phrase * 3) < live_stt.CAPTION_REPEAT_MAX_CHARS


def test_both_screens_read_one_bound(caplog):
    # The publication screen and the translator backstop are two sites deciding
    # one question, so they share repeat_span rather than each carrying a copy —
    # a caption caught at publication must also be declined if it ever reaches
    # the queue by another route.
    escape = _runaway("いい音があるので、", 612)  # session 6 n=263
    t = live_stt.CodexTranslator()
    t.enabled = True

    with caplog.at_level(logging.WARNING, logger="live_stt"):
        t.submit(263, escape)

    assert live_stt.caption_defect(escape)
    assert t.degenerate_captions == 1
    assert t.queue.empty()


# --- M14.2: respawn the EN leg after the app-server exits --------------------


class _Codex:
    """A `codex app-server` factory for create_subprocess_exec: one fresh
    _FakeProc per spawn, so a test can count spawns and script each server."""

    def __init__(self, missing: bool = False):
        self.procs: list[_FakeProc] = []
        self.missing = missing

    async def exec(self, *_a, **_k):
        if self.missing:
            raise FileNotFoundError("codex")
        self.procs.append(_FakeProc(asyncio.StreamReader()))
        return self.procs[-1]


async def _await_spawn(codex: _Codex, n: int, spins: int = 4000) -> _FakeProc:
    """Yield until spawn `n` (1-based) exists and return its process."""
    for _ in range(spins):
        await asyncio.sleep(0)
        if len(codex.procs) >= n:
            return codex.procs[n - 1]
    raise AssertionError(f"spawn {n} never happened ({len(codex.procs)} so far)")


async def _answer(t: live_stt.CodexTranslator, reader, result: dict, after: int) -> int:
    """Answer the next request issued after id `after`, through _read_loop.

    Ids keep incrementing across a respawn, so tests chain on the returned id
    rather than naming ids that shift the moment a handshake gains a step.
    """
    for _ in range(4000):
        await asyncio.sleep(0)
        new = [r for r in t._pending if r > after]
        if new:
            rid = min(new)
            reader.feed_data(_rpc_result(rid, result))
            return rid
    raise AssertionError(f"no request was issued after id {after}")


async def _serve_start(t: live_stt.CodexTranslator, proc: _FakeProc, warmup: bool = True) -> int:
    """Answer the three requests start() issues; `warmup=False` fails the turn."""
    reader = proc.stdout
    rid = await _answer(t, reader, {}, 0)  # initialize
    tier = live_stt.TRANSLATE_SERVICE_TIER
    rid = await _answer(t, reader, {"thread": {"id": f"th-{rid}"}, "serviceTier": tier}, rid)
    rid = await _answer(t, reader, {}, rid)  # warm-up turn/start
    reader.feed_data(_rpc_note("turn/completed" if warmup else "error", {}))
    return rid


async def _live_leg(t: live_stt.CodexTranslator, codex: _Codex) -> _FakeProc:
    """Bring the leg up against spawn 1 and return its process."""
    starter = asyncio.create_task(t.start())
    await _serve_start(t, await _await_spawn(codex, 1))
    assert await asyncio.wait_for(starter, timeout=3.0) is True
    return codex.procs[0]


async def _kill(t: live_stt.CodexTranslator, proc: _FakeProc):
    """The session-6 death: the app-server exits, leaving nothing to probe."""
    proc.stdout.feed_eof()
    assert t._reader_task is not None
    await t._reader_task
    assert t.enabled is False


def test_an_exited_app_server_is_respawned_and_the_leg_re_enables(tmp_path, monkeypatch):
    # M14.2(a)+(e). Session 6 lost translation at 14:38:49 on `codex app-server
    # exited` and ran JA-only to the end; the process is gone, so a re-probe has
    # nothing to talk to and only a new subprocess recovers. The transcript must
    # record the recovery too -- one carrying the disable marker alone reads as
    # JA-only from that point while EN lines resume below it.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        transcript = live_stt.TranscriptFile(tmp_path / "session.txt")
        t = live_stt.CodexTranslator(output_file=transcript)
        await _kill(t, await _live_leg(t, codex))

        run_task = asyncio.create_task(t.run())
        t.submit(7, "こんにちは。")  # the caption that finds the leg dead
        second = await _await_spawn(codex, 2)
        rid = await _serve_start(t, second)
        rid = await _answer(t, second.stdout, {}, rid)  # the caption's own turn
        item = {"item": {"type": "agentMessage", "text": "Hello."}}
        second.stdout.feed_data(_rpc_note("item/completed", item))
        second.stdout.feed_data(_rpc_note("turn/completed", {}))

        t.submit_sentinel()
        await asyncio.wait_for(run_task, timeout=3.0)
        assert t.enabled is True  # the leg is back, not merely alive
        assert len(codex.procs) == 2  # exactly one respawn
        transcript.close()

    asyncio.run(scenario())
    body = (tmp_path / "session.txt").read_text(encoding="utf-8").splitlines()
    events = [ln.split("] ", 1)[1] for ln in body]
    assert events[0] == "-- translation disabled: codex app-server exited"
    assert events[1].startswith("-- translation restored: ")
    assert events[2] == "EN 7: Hello."  # the caption that paid for the respawn


def test_the_respawned_thread_carries_the_glossary_learned_before_the_death(monkeypatch):
    # M14.2(b). A term learned mid-session reaches the model through ONE channel,
    # the thread's developerInstructions, and the thread died with the process.
    # A respawn that did not re-run _instructions would silently un-learn the
    # session while looking healthy.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        ctx = live_stt.SessionContext()
        t = live_stt.CodexTranslator(ctx)
        first = await _live_leg(t, codex)

        def opened(proc: _FakeProc) -> list[dict]:
            sent = [json.loads(w) for w in proc.stdin.writes]
            return [m for m in sent if m.get("method") == "thread/start"]

        # 標柱 is absent from TRANSLATOR_INSTRUCTIONS, so its presence below can
        # only come from the glossary (プレドニン rides the fixed drug-name line).
        assert "標柱" not in opened(first)[0]["params"]["developerInstructions"]

        for _ in range(live_stt.CONTEXT_TERM_SUPPORT):  # learned while the leg was up
            ctx.observe_ja("標柱が見えました")
        await _kill(t, first)

        respawn = asyncio.create_task(t._recover())
        second = await _await_spawn(codex, 2)
        await _serve_start(t, second)
        assert await asyncio.wait_for(respawn, timeout=3.0) is True

        assert len(opened(second)) == 1
        assert "標柱" in opened(second)[0]["params"]["developerInstructions"]
        assert t._brief == ctx.translator_brief()  # and the rotation check agrees

    asyncio.run(scenario())


def test_a_stale_wake_sentinel_cannot_fail_the_respawn(monkeypatch):
    # The EOF cleanup enqueues one {method:error} wake sentinel (T8.3) and it is
    # still in _notes when recovery starts, so the respawn's warm-up turn collects
    # THAT instead of its own turn/completed and raises. Measured against a real
    # codex app-server: `init failed (RuntimeError: {})` in 0.41 s, a healthy
    # server thrown away. _notes must be drained before the handshake.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        await _kill(t, await _live_leg(t, codex))
        assert t._notes.qsize() == 1  # the stale sentinel is the whole hazard

        respawn = asyncio.create_task(t._recover())
        await _serve_start(t, await _await_spawn(codex, 2))
        assert await asyncio.wait_for(respawn, timeout=3.0) is True
        assert t.enabled is True

    asyncio.run(scenario())


def test_a_dead_codex_costs_a_bounded_number_of_respawns(monkeypatch):
    # M14.2(c). A codex that can be spawned but never completes its handshake
    # must not be retried forever: the budget is per session, so a leg that keeps
    # dying stops costing captions their latency once it is plainly gone. submit()
    # stops queueing at the same moment, which is what keeps the backlog -- and
    # the tdrop= counter that means real backpressure -- clear of a dead leg.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        await _kill(t, await _live_leg(t, codex))

        for attempt in range(1, live_stt.TRANSLATE_MAX_RECOVERIES + 1):
            t._recover_at = 0.0  # the backoff itself is locked separately
            respawn = asyncio.create_task(t._recover())
            await _serve_start(t, await _await_spawn(codex, attempt + 1), warmup=False)
            assert await asyncio.wait_for(respawn, timeout=3.0) is False
            assert t.enabled is False

        assert t._recoverable() is False
        t._recover_at = 0.0
        assert await t._recover() is False  # budget spent: no further spawn
        assert len(codex.procs) == live_stt.TRANSLATE_MAX_RECOVERIES + 1

        t.submit(1, "こんにちは。")  # and captions stop entering the queue
        assert t.queue.empty()
        assert t.dropped_translations == 0

    asyncio.run(scenario())


def test_a_failed_respawn_backs_off_before_the_next(monkeypatch):
    # M14.2(c), the other half: without a cooldown every caption arriving behind a
    # dead codex pays a full handshake, so a broken leg would cost more latency
    # than the degrade it is repairing. The wait doubles, so a leg that stays down
    # is probed logarithmically rather than per caption.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        await _kill(t, await _live_leg(t, codex))
        assert t._recover_at == 0.0  # the first attempt is immediate

        respawn = asyncio.create_task(t._recover())
        await _serve_start(t, await _await_spawn(codex, 2), warmup=False)
        assert await asyncio.wait_for(respawn, timeout=3.0) is False

        assert t._recover_at > time.monotonic()  # a deadline, not a free retry
        assert t._recover_wait == live_stt.TRANSLATE_RECOVERY_WAIT_S * 2
        assert await t._recover() is False  # inside the cooldown
        assert len(codex.procs) == 2  # and it spawned nothing
        assert t._recoveries == 1  # a skipped attempt does not spend budget

    asyncio.run(scenario())


def test_a_missing_codex_binary_ends_recovery_after_one_attempt(monkeypatch, caplog):
    # M14.2(d). codex uninstalled or upgraded out from under a live session is not
    # a transient: there is nothing to come back to, so it must not spend the whole
    # respawn budget rediscovering that. start() assigns _proc only once the exec
    # succeeds, which is what separates a missing binary from a failed handshake.
    async def scenario():
        codex = _Codex(missing=True)
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        t._proc = _FakeProc(asyncio.StreamReader())  # type: ignore[assignment]
        with caplog.at_level(logging.ERROR, logger="live_stt"):
            assert await t._recover() is False
            assert t._recoverable() is False  # permanent, not one of the budget
            t._recover_at = 0.0
            assert await t._recover() is False

    asyncio.run(scenario())
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and "codex" in errors[0].getMessage()


def test_the_shutdown_sentinel_stops_recovery(monkeypatch):
    # M14.2(f). The drain window between submit_sentinel() and close() is the one
    # place run() is alive with recovery armed: a caption queued just before the
    # Ctrl+C would spend a full handshake on a session that is ending, and close()
    # would then kill the process it had just started.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        await _kill(t, await _live_leg(t, codex))

        t.submit(1, "こんにちは。")  # queued while recovery was still armed
        assert t.queue.qsize() == 1
        t.submit_sentinel()  # Ctrl+C / SIGHUP: no more captions are coming
        await asyncio.wait_for(t.run(), timeout=3.0)
        assert len(codex.procs) == 1  # the queued caption started no respawn

    asyncio.run(scenario())


def test_close_stops_recovery(monkeypatch):
    # M14.2(f), the other latch, proved on its own: close() is the teardown
    # primitive and a caller that reaches it without the sentinel -- any path but
    # run_session's -- must still leave nothing able to spawn codex afterwards.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        await _kill(t, await _live_leg(t, codex))
        assert t._recoverable() is True  # armed right up to the close

        await t.close()

        assert t._recoverable() is False
        t._recover_at = 0.0
        assert await t._recover() is False
        assert len(codex.procs) == 1

    asyncio.run(scenario())


def test_the_respawned_leg_starts_with_a_clean_strike_count(monkeypatch):
    # A leg that died carrying strikes would be one failure from a permanent
    # disable the moment it came back, and a carried turn count rotates the fresh
    # thread before it has served TRANSLATE_ROTATE_TURNS turns. A new process and
    # a new thread start their own counters.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        await _kill(t, await _live_leg(t, codex))
        t._failures = live_stt.TRANSLATE_MAX_FAILURES - 1
        t._turns = live_stt.TRANSLATE_ROTATE_TURNS - 3

        respawn = asyncio.create_task(t._recover())
        await _serve_start(t, await _await_spawn(codex, 2))
        assert await asyncio.wait_for(respawn, timeout=3.0) is True
        assert t._failures == 0
        assert t._turns == 0

    asyncio.run(scenario())


# --- M14.3: re-probe the EN leg after a transient-failure disable ------------


def _sent(proc: _FakeProc, method: str) -> list[dict]:
    """Every JSON-RPC message of one method the translator wrote to this proc."""
    return [m for m in (json.loads(w) for w in proc.stdin.writes) if m.get("method") == method]


async def _strike_out(t: live_stt.CodexTranslator, proc: _FakeProc):
    """The session-1 death: TRANSLATE_MAX_FAILURES turn failures take the leg
    down while the app-server itself keeps running -- the state a probe is for.

    Driven through the real _translate/_turn/error-note path, with _abort_turn
    stubbed to its no-op shape: its 1 s drain wait is real time in a fast suite,
    and what it drains is locked by the tests above.
    """

    async def no_wait():
        return

    t._abort_turn = no_wait  # type: ignore[assignment]
    for _ in range(live_stt.TRANSLATE_MAX_FAILURES):
        turn = asyncio.create_task(t._translate("失敗する。"))
        await _answer(t, proc.stdout, {}, 0)  # turn/start; nothing else is pending
        proc.stdout.feed_data(_rpc_note("error", {}))
        assert await asyncio.wait_for(turn, timeout=3.0) == ""
    assert t.enabled is False
    assert t._alive()  # the server outlived the turns: this is not the EOF path


async def _serve_probe(t: live_stt.CodexTranslator, proc: _FakeProc, healthy: bool = True) -> int:
    """Answer the two requests a probe issues; `healthy=False` fails its turn."""
    tier = live_stt.TRANSLATE_SERVICE_TIER
    thread = {"thread": {"id": "th-probe"}, "serviceTier": tier}
    rid = await _answer(t, proc.stdout, thread, 0)  # nothing else is pending
    rid = await _answer(t, proc.stdout, {}, rid)  # the probe turn/start
    proc.stdout.feed_data(_rpc_note("turn/completed" if healthy else "error", {}))
    return rid


def test_a_three_strike_disable_is_probed_on_the_surviving_server(tmp_path, monkeypatch):
    # M14.3(a)+(b)+(e). Session 1 lost translation at n=194 to three consecutive
    # runaway captions, yet single runaways at n=130 and n=138 translated fine --
    # so what disabled the leg was transient and the app-server is still there.
    # Recovery must therefore re-qualify THAT server: respawning would spend
    # 4.8 s throwing away a healthy one. The transcript records the re-enable in
    # order, or a saved session reads JA-only while EN lines resume below it.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        transcript = live_stt.TranscriptFile(tmp_path / "session.txt")
        t = live_stt.CodexTranslator(output_file=transcript)
        proc = await _live_leg(t, codex)
        await _strike_out(t, proc)

        run_task = asyncio.create_task(t.run())
        t.submit(9, "こんにちは。")  # the caption that finds the leg disabled
        rid = await _serve_probe(t, proc)
        rid = await _answer(t, proc.stdout, {}, rid)  # the caption's own turn
        item = {"item": {"type": "agentMessage", "text": "Hello."}}
        proc.stdout.feed_data(_rpc_note("item/completed", item))
        proc.stdout.feed_data(_rpc_note("turn/completed", {}))

        t.submit_sentinel()
        await asyncio.wait_for(run_task, timeout=3.0)
        assert t.enabled is True
        assert len(codex.procs) == 1  # the healthy app-server was not replaced
        transcript.close()

    asyncio.run(scenario())
    body = (tmp_path / "session.txt").read_text(encoding="utf-8").splitlines()
    events = [ln.split("] ", 1)[1] for ln in body]
    assert events[0] == "-- translation disabled: 3 consecutive failures (RuntimeError: {})"
    assert events[1].startswith("-- translation restored: codex app-server probed")
    assert events[2] == "EN 9: Hello."  # the caption that paid for the probe


def test_recovery_routes_on_the_surviving_process_not_on_the_trigger(monkeypatch):
    # M14.3(b). One entry point, two arms, and _alive() is the whole
    # discriminator -- which is what keeps the second trigger from growing a
    # second mechanism with its own budget, backoff and markers to keep in step.
    # Routing the live process to _respawn would not merely cost a comparable
    # 4.8 s: it drops _proc and overwrites _reader_task, leaving the surviving
    # server a reader no one owns, whose eventual EOF disables the healthy leg
    # this very recovery just produced.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        proc = await _live_leg(t, codex)
        reader = t._reader_task
        await _strike_out(t, proc)

        probe = asyncio.create_task(t._recover())
        await _serve_probe(t, proc)
        assert await asyncio.wait_for(probe, timeout=3.0) is True
        assert len(codex.procs) == 1  # alive -> probed
        assert t._proc is proc and t._reader_task is reader  # kept, not abandoned

        await _kill(t, proc)
        t._recover_at = 0.0
        respawn = asyncio.create_task(t._recover())
        await _serve_start(t, await _await_spawn(codex, 2))
        assert await asyncio.wait_for(respawn, timeout=3.0) is True
        assert len(codex.procs) == 2  # dead -> respawned

        assert t._recoveries == 2  # and both arms spend the one budget

    asyncio.run(scenario())


def test_the_probe_opens_a_fresh_thread_carrying_the_current_glossary(monkeypatch):
    # A stalled turn poisons its THREAD and interrupt-plus-drain does not clear
    # it (L-026: on one shared thread a stalled turn made a later real-speech
    # control hang, where a fresh thread measured 3.4 s). The three strikes are
    # exactly that class, so a probe reusing the thread would measure the wedge
    # and report a healthy server dead. The fresh thread is also what re-carries
    # the terms learned since startup, since the glossary rides only there.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        ctx = live_stt.SessionContext()
        t = live_stt.CodexTranslator(ctx)
        proc = await _live_leg(t, codex)
        wedged = t._thread_id
        await _strike_out(t, proc)
        for _ in range(live_stt.CONTEXT_TERM_SUPPORT):  # captions keep learning
            ctx.observe_ja("標柱が見えました")

        probe = asyncio.create_task(t._recover())
        await _serve_probe(t, proc)
        assert await asyncio.wait_for(probe, timeout=3.0) is True

        opened = _sent(proc, "thread/start")
        assert len(opened) == 2 and t._thread_id != wedged
        assert "標柱" in opened[1]["params"]["developerInstructions"]
        assert t._brief == ctx.translator_brief()  # and the rotation check agrees
        assert _sent(proc, "turn/start")[-1]["params"]["threadId"] == t._thread_id

    asyncio.run(scenario())


def test_a_failed_probe_backs_off_before_the_next(monkeypatch):
    # M14.3(a). A server that answers but cannot finish a turn must not be probed
    # once per caption: every attempt costs its caption a turn's latency, so the
    # wait doubles and a leg that stays down is re-probed logarithmically.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        proc = await _live_leg(t, codex)
        await _strike_out(t, proc)
        del t._abort_turn  # the real one back: this test is what locks its call
        assert t._recover_at == 0.0  # the first attempt is immediate

        probe = asyncio.create_task(t._recover())
        await _serve_probe(t, proc, healthy=False)
        assert await asyncio.wait_for(probe, timeout=3.0) is False
        assert t.enabled is False
        # A probe turn that failed is still running server-side, and an
        # un-interrupted one converts every later turn into a timeout (L-026) --
        # which would make the next probe fail for the previous probe's reason.
        assert len(_sent(proc, "turn/interrupt")) == 1

        assert t._recover_at > time.monotonic()  # a deadline, not a free retry
        assert t._recover_wait == live_stt.TRANSLATE_RECOVERY_WAIT_S * 2
        assert await t._recover() is False  # inside the cooldown
        assert len(_sent(proc, "thread/start")) == 2  # and it opened no thread
        assert t._recoveries == 1  # a skipped attempt does not spend budget

    asyncio.run(scenario())


def test_a_wedged_app_server_costs_a_bounded_number_of_probes(monkeypatch):
    # M14.3(c). A server that stays alive and stays broken is the case a probe
    # cannot distinguish from a transient one, so the budget is what ends it --
    # shared with the respawn arm, so a leg that dies both ways cannot spend
    # twice. submit() stops queueing at the same moment, keeping a permanently
    # dead leg clear of the backlog and of the tdrop= counter that means real
    # backpressure.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        proc = await _live_leg(t, codex)
        await _strike_out(t, proc)

        for _ in range(live_stt.TRANSLATE_MAX_RECOVERIES):
            t._recover_at = 0.0  # the backoff itself is locked separately
            probe = asyncio.create_task(t._recover())
            await _serve_probe(t, proc, healthy=False)
            assert await asyncio.wait_for(probe, timeout=3.0) is False

        assert t._recoverable() is False
        t._recover_at = 0.0
        assert await t._recover() is False  # budget spent: no further probe
        assert len(_sent(proc, "thread/start")) == 1 + live_stt.TRANSLATE_MAX_RECOVERIES

        t.submit(1, "こんにちは。")  # and captions stop entering the queue
        assert t.queue.empty()
        assert t.dropped_translations == 0

    asyncio.run(scenario())


def test_the_probed_leg_starts_with_a_clean_strike_count(monkeypatch):
    # M14.3(d). The probe arm reaches recovery carrying a FULL strike count by
    # construction -- three consecutive failures are what disabled the leg -- so
    # without the reset the very next failure would disable it again, permanently
    # this time in all but name, and each probe would cost a whole cooldown to
    # buy one caption. The fresh thread wants its own turn count for the same
    # reason a respawned one does: a carried count rotates it early.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        proc = await _live_leg(t, codex)
        t._turns = live_stt.TRANSLATE_ROTATE_TURNS - 3
        await _strike_out(t, proc)
        assert t._failures == live_stt.TRANSLATE_MAX_FAILURES

        probe = asyncio.create_task(t._recover())
        await _serve_probe(t, proc)
        assert await asyncio.wait_for(probe, timeout=3.0) is True
        assert t._failures == 0
        assert t._turns == 0

    asyncio.run(scenario())


def test_the_stalled_turns_late_output_cannot_fail_the_probe(monkeypatch):
    # A turn that STALLS rather than errors keeps running server-side: its
    # deltas and its eventual error arrive after _abort_turn has drained, so the
    # queue the probe turn reads from is not empty. The probe would collect that
    # tail instead of its own turn/completed and raise, failing a recovery that
    # had in fact succeeded -- the EOF arm's measured `init failed
    # (RuntimeError: {})` with a live server as the cause.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        proc = await _live_leg(t, codex)
        await _strike_out(t, proc)

        proc.stdout.feed_data(_rpc_note("item/agentMessage/delta", {"delta": "late"}))
        proc.stdout.feed_data(_rpc_note("error", {}))
        for _ in range(200):  # let _read_loop file both notes
            await asyncio.sleep(0)
            if t._notes.qsize() == 2:
                break
        assert t._notes.qsize() == 2  # the tail is the whole hazard

        probe = asyncio.create_task(t._recover())
        await _serve_probe(t, proc)
        assert await asyncio.wait_for(probe, timeout=3.0) is True
        assert t.enabled is True

    asyncio.run(scenario())


def test_a_server_that_dies_under_the_probe_is_respawned_next(monkeypatch):
    # The two triggers compose: a wedged server that then exits must not keep
    # being probed, and nothing tracks which trigger fired -- _alive() is read at
    # each attempt, so the arm follows the process's current state rather than
    # the one that opened recovery.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        proc = await _live_leg(t, codex)
        await _strike_out(t, proc)

        probe = asyncio.create_task(t._recover())
        rid = await _answer(t, proc.stdout, {}, 0)  # thread/start, answered by EOF
        proc.stdout.feed_eof()
        assert await asyncio.wait_for(probe, timeout=3.0) is False
        assert rid not in t._pending  # the EOF failed it, not a timeout
        assert len(codex.procs) == 1

        t._recover_at = 0.0
        respawn = asyncio.create_task(t._recover())
        await _serve_start(t, await _await_spawn(codex, 2))
        assert await asyncio.wait_for(respawn, timeout=3.0) is True
        assert t.enabled is True

    asyncio.run(scenario())


def test_a_server_that_dies_between_the_probe_turn_and_the_enable_stays_off(monkeypatch):
    # T8.6 on the probe path. The probe turn completes, its turn/completed is
    # consumed, and the next readline hits EOF -- so a probe that enabled on the
    # turn alone would hand every later caption to a server that is gone, each
    # one waiting out TRANSLATE_TIMEOUT_S on a request no one will resolve. The
    # enable is guarded by the same liveness read that routes the arms.
    async def scenario():
        codex = _Codex()
        monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", codex.exec)
        t = live_stt.CodexTranslator()
        proc = await _live_leg(t, codex)
        await _strike_out(t, proc)

        probe = asyncio.create_task(t._recover())
        tier = live_stt.TRANSLATE_SERVICE_TIER
        thread = {"thread": {"id": "th-probe"}, "serviceTier": tier}
        rid = await _answer(t, proc.stdout, thread, 0)
        rid = await _answer(t, proc.stdout, {}, rid)
        # One synchronous burst, so _read_loop has certainly processed the EOF
        # before _probe reads liveness: readline() returns a buffered line
        # without yielding (L-022's deterministic async-EOF technique).
        proc.stdout.feed_data(_rpc_note("turn/completed", {}))
        proc.stdout.feed_eof()

        assert await asyncio.wait_for(probe, timeout=3.0) is False
        assert t.enabled is False  # a completed turn is not a live server

    asyncio.run(scenario())
