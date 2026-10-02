"""Turn ownership: queue contract, Correlate each translation turn with its own server turn.

Real protocol shapes over fake stdio; generated IDs exercise equality rather than prefixes.
Implementation bodies stay unread. EOF/anonymous-note controls preserve existing fake protocols.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import pytest

import live_stt


class _FakeStdin:
    def __init__(self, reader: asyncio.StreamReader):
        self.reader = reader
        self.writes: list[bytes] = []

    def write(self, data: bytes):
        self.writes.append(data)
        request = json.loads(data)
        if request.get("method") == "turn/interrupt" and "id" in request:
            self.reader.feed_data(_rpc_result(request["id"], {}))

    def close(self):
        pass


class _FakeProc:
    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.stdin = _FakeStdin(self.stdout)
        self.returncode: int | None = None

    async def wait(self):
        self.returncode = 0
        return 0

    def kill(self):
        self.returncode = -9


def _rpc_result(rid: int, result: dict) -> bytes:
    return (json.dumps({"id": rid, "result": result}) + "\n").encode()


def _rpc_note(method: str, params: dict) -> bytes:
    return (json.dumps({"method": method, "params": params}) + "\n").encode()


async def _await_pending(t: live_stt.CodexTranslator, rid: int, spins: int = 1000):
    for _ in range(spins):
        await asyncio.sleep(0)
        if rid in t._pending:
            return
    raise AssertionError(f"request id {rid} was never issued")


async def _answer(t: live_stt.CodexTranslator, proc: _FakeProc, result: dict, after: int) -> int:
    for _ in range(1000):
        await asyncio.sleep(0)
        new = [rid for rid in t._pending if rid > after]
        if new:
            rid = min(new)
            proc.stdout.feed_data(_rpc_result(rid, result))
            return rid
    raise AssertionError(f"no request was issued after id {after}")


async def _answer_turn(
    t: live_stt.CodexTranslator, proc: _FakeProc, turn_id: str, after: int = 0
) -> tuple[int, str]:
    # A compliant failure policy may replace a thread before the next caption.
    for _ in range(1000):
        await asyncio.sleep(0)
        new = [rid for rid in t._pending if rid > after]
        if not new:
            continue
        rid = min(new)
        request = next(json.loads(w) for w in proc.stdin.writes if json.loads(w).get("id") == rid)
        if request["method"] == "thread/start":
            result = {
                "thread": {"id": f"fresh-thread-{rid}"},
                "serviceTier": live_stt.TRANSLATE_SERVICE_TIER,
            }
            after = await _answer(t, proc, result, after)
            continue
        assert request["method"] == "turn/start"
        await _answer(t, proc, {"turn": {"id": turn_id, "status": "inProgress"}}, after)
        return rid, request["params"]["threadId"]
    raise AssertionError("no turn/start was issued")


def _note(method: str, thread_id: str | None, turn_id: str | None, text: str) -> bytes:
    params: dict[str, Any]
    if method == "item/agentMessage/delta":
        params = {"itemId": "message", "delta": text}
    elif method == "item/completed":
        params = {"item": {"type": "agentMessage", "text": text}}
    elif method == "turn/completed":
        params = {"turn": {"status": "completed"}}
    else:
        assert method == "error"
        params = {"error": {"message": text}, "willRetry": False}
    if thread_id is not None:
        params["threadId"] = thread_id
    if turn_id is not None:
        if method == "turn/completed":
            params["turn"]["id"] = turn_id
        else:
            params["turnId"] = turn_id
    return _rpc_note(method, params)


def _open(
    output_file: live_stt.TranscriptFile | None = None,
) -> tuple[live_stt.CodexTranslator, _FakeProc, asyncio.Task]:
    t = live_stt.CodexTranslator(output_file=output_file, sources=("ja", "en"))
    proc = _FakeProc()
    t._proc = proc  # type: ignore[assignment]
    t.enabled = True
    t._legs["ja"].thread_id = "thread-ja"
    t._legs["en"].thread_id = "thread-en"
    reader_task = asyncio.create_task(t._read_loop())
    t._reader_task = reader_task
    return t, proc, reader_task


async def _stop_reader(task: asyncio.Task):
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def _result(task: asyncio.Task) -> str | Exception:
    try:
        return await asyncio.wait_for(task, timeout=0.5)
    except Exception as exc:
        return exc


def _fast_failures(monkeypatch: pytest.MonkeyPatch):
    sleep = asyncio.sleep

    async def no_delay(delay, result=None):
        await sleep(0)
        return result

    monkeypatch.setattr(live_stt.asyncio, "sleep", no_delay)
    monkeypatch.setattr(live_stt, "TRANSLATE_TIMEOUT_S", 0.01)


def _ids() -> list[tuple[str, str, str, str]]:
    rng = random.Random(1002)  # noqa: S311 — reproducible identity samples
    pairs = [
        ("thread-1", "thread-10", "turn-1", "turn-10"),
        ("thread-10", "thread-1", "turn-10", "turn-1"),
        ("会話🗣", "会話🗣x", "発話🗣", "発話🗣x"),
    ]
    for _ in range(9):
        ids = [f"id-{rng.getrandbits(128):032x}" for _ in range(4)]
        pairs.append((ids[0], ids[1], ids[2], ids[3]))
    return pairs


@pytest.mark.parametrize("source", ["ja", "en"])
@pytest.mark.parametrize(
    "identity",
    [
        "foreign-thread",
        "foreign-thread-own-turn",
        "foreign-both",
        "foreign-turn",
        "own-thread-foreign-turn",
    ],
)
@pytest.mark.parametrize(
    "method", ["item/agentMessage/delta", "item/completed", "turn/completed", "error"]
)
def test_a_foreign_note_never_changes_the_turn(source, identity, method):
    async def scenario():
        failures = []
        for thread, other_thread, turn, other_turn in _ids():
            t, proc, reader_task = _open()
            leg = t._legs[source]
            leg.thread_id = thread
            task = asyncio.create_task(t._turn("自分の発話。", leg))
            try:
                await _await_pending(t, 1)
                await _answer_turn(t, proc, turn)
                foreign = {
                    "foreign-thread": (other_thread, None),
                    "foreign-thread-own-turn": (other_thread, turn),
                    "foreign-both": (other_thread, other_turn),
                    "foreign-turn": (None, other_turn),
                    "own-thread-foreign-turn": (thread, other_turn),
                }[identity]
                proc.stdout.feed_data(_note(method, foreign[0], foreign[1], "foreign"))
                proc.stdout.feed_data(_note("item/agentMessage/delta", thread, turn, "own"))
                proc.stdout.feed_data(_note("turn/completed", thread, turn, ""))
                result = await _result(task)
                if result != "own":
                    failures.append((thread, turn, result))
            finally:
                await _stop_reader(reader_task)
        assert failures == [], (source, identity, method, failures)

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["ja", "en"])
@pytest.mark.parametrize("identity", ["anonymous", "thread-only", "turn-only", "both"])
@pytest.mark.parametrize(
    "method", ["item/agentMessage/delta", "item/completed", "turn/completed", "error"]
)
def test_matching_and_anonymous_notes_still_reach_the_turn(source, identity, method):
    async def scenario():
        t, proc, reader_task = _open()
        leg = t._legs[source]
        task = asyncio.create_task(t._turn("自分の発話。", leg))
        try:
            _, thread = await _answer_turn(t, proc, "own-turn")
            named_thread = thread if identity in {"thread-only", "both"} else None
            named_turn = "own-turn" if identity in {"turn-only", "both"} else None
            if method == "turn/completed":
                proc.stdout.feed_data(_note("item/agentMessage/delta", thread, "own-turn", "own"))
            proc.stdout.feed_data(_note(method, named_thread, named_turn, "own"))
            if method != "turn/completed":
                proc.stdout.feed_data(_note("turn/completed", thread, "own-turn", ""))
            result = await _result(task)
            if method == "error":
                assert isinstance(result, RuntimeError)
            else:
                assert result == "own"
        finally:
            await _stop_reader(reader_task)

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["ja", "en"])
def test_a_named_turn_wakes_on_the_anonymous_eof_sentinel(source):
    async def scenario():
        t, proc, reader_task = _open()
        task = asyncio.create_task(t._turn("自分の発話。", t._legs[source]))
        _, _ = await _answer_turn(t, proc, "own-turn")
        proc.stdout.feed_eof()
        result = await _result(task)
        assert isinstance(result, RuntimeError), result
        assert t.enabled is False
        await reader_task

    asyncio.run(scenario())


async def _timeout(t: live_stt.CodexTranslator, proc: _FakeProc, source: str) -> tuple[int, str]:
    task = asyncio.create_task(t._translate("応答の止まった発話。", source))
    rid, thread = await _answer_turn(t, proc, "stalled-turn")
    assert await _result(task) == ""
    assert t._failures == 1
    return rid, thread


@pytest.mark.parametrize("source", ["ja", "en"])
@pytest.mark.parametrize("arrival", ["before", "during"])
def test_a_timed_out_turn_never_publishes_under_the_next_caption(
    source, arrival, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _fast_failures(monkeypatch)

    async def scenario():
        transcript = live_stt.TranscriptFile(tmp_path / "session.txt")
        t, proc, reader_task = _open(transcript)
        try:
            rid, stale_thread = await _timeout(t, proc, source)

            def late_completion():
                proc.stdout.feed_data(
                    _note("item/agentMessage/delta", stale_thread, "stalled-turn", "stale")
                )
                proc.stdout.feed_data(
                    _note("item/completed", stale_thread, "stalled-turn", "stale")
                )
                proc.stdout.feed_data(_note("turn/completed", stale_thread, "stalled-turn", ""))

            if arrival == "before":
                late_completion()
                for _ in range(1000):
                    await asyncio.sleep(0)
                    if t._notes.qsize() == 3:
                        break
                assert t._notes.qsize() == 3

            t.submit(2, "次の発話です。", source)
            t.submit_sentinel()
            runner = asyncio.create_task(t.run())
            _, thread = await _answer_turn(t, proc, "next-turn", rid)
            if arrival == "during":
                late_completion()
            proc.stdout.feed_data(_note("item/completed", thread, "next-turn", "own"))
            proc.stdout.feed_data(_note("turn/completed", thread, "next-turn", ""))
            await asyncio.wait_for(runner, timeout=0.5)
            transcript.close()
            published = [
                line.split("] ", 1)[1]
                for line in (tmp_path / "session.txt").read_text().splitlines()
            ]
            assert published == ["TGT 2: own"]
        finally:
            transcript.close()
            await _stop_reader(reader_task)

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["ja", "en"])
def test_a_steered_start_fails_the_caption(source, monkeypatch: pytest.MonkeyPatch):
    _fast_failures(monkeypatch)

    async def scenario():
        t, proc, reader_task = _open()
        try:
            rid, thread = await _timeout(t, proc, source)
            task = asyncio.create_task(t._translate("次の発話です。", source))
            _, _ = await _answer_turn(t, proc, "stalled-turn", rid)
            proc.stdout.feed_data(_note("item/completed", thread, "stalled-turn", "stale"))
            proc.stdout.feed_data(_note("turn/completed", thread, "stalled-turn", ""))
            result = await _result(task)
            assert result == "", result
            assert t._failures == 2
            assert t.enabled is True
        finally:
            await _stop_reader(reader_task)

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["ja", "en"])
@pytest.mark.parametrize("failure", ["timeout", "error"])
def test_abort_interrupts_the_unfinished_turn(source, failure, monkeypatch: pytest.MonkeyPatch):
    _fast_failures(monkeypatch)

    async def scenario():
        t, proc, reader_task = _open()
        try:
            task = asyncio.create_task(t._translate("応答の止まった発話。", source))
            _, thread = await _answer_turn(t, proc, "unfinished-turn")
            if failure == "error":
                proc.stdout.feed_data(_note("error", thread, "unfinished-turn", "failed"))
            assert await _result(task) == ""
            requests = [json.loads(w) for w in proc.stdin.writes]
            interrupts = [r["params"] for r in requests if r.get("method") == "turn/interrupt"]
            assert interrupts == [{"threadId": thread, "turnId": "unfinished-turn"}]
            assert t._failures == 1
        finally:
            await _stop_reader(reader_task)

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["ja", "en"])
def test_a_new_turn_after_a_completed_turn_is_collected(source):
    async def scenario():
        t, proc, reader_task = _open()
        after = 0
        try:
            for turn_id in ["completed-turn", "next-turn"]:
                task = asyncio.create_task(t._translate("自分の発話。", source))
                after, thread = await _answer_turn(t, proc, turn_id, after)
                proc.stdout.feed_data(_note("item/completed", thread, turn_id, "own"))
                proc.stdout.feed_data(_note("turn/completed", thread, turn_id, ""))
                assert await _result(task) == "own"
                assert t._failures == 0
        finally:
            await _stop_reader(reader_task)

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["ja", "en"])
def test_a_turn_completed_before_the_server_exits_still_publishes(source):
    # reviewer-1: EOF cleanup clears the leg's thread before the turn reads what the
    # server sent ahead of its exit, so matching against the live leg dropped a caption
    # that had in fact completed.
    async def scenario():
        t, proc, reader_task = _open()
        task = asyncio.create_task(t._translate("自分の発話。", source))
        _, thread = await _answer_turn(t, proc, "own-turn")
        proc.stdout.feed_data(_note("item/agentMessage/delta", thread, "own-turn", "own"))
        proc.stdout.feed_data(_note("turn/completed", thread, "own-turn", ""))
        proc.stdout.feed_eof()
        assert await _result(task) == "own"
        await reader_task
        assert t._legs[source].thread_id is None  # the cleanup did run first

    asyncio.run(scenario())


@pytest.mark.parametrize("source", ["ja", "en"])
def test_an_interrupt_names_the_turn_on_its_own_thread(source, monkeypatch: pytest.MonkeyPatch):
    # reviewer-2: a turn that timed out on one thread stays named against that thread when
    # the next caption rotates the leg and its turn/start never answers.
    _fast_failures(monkeypatch)

    async def scenario():
        t, proc, reader_task = _open()
        try:
            first = asyncio.create_task(t._translate("応答の止まった発話。", source))
            rid, old_thread = await _answer_turn(t, proc, "old-turn")
            assert await _result(first) == ""
            t._legs[source].turns = live_stt.TRANSLATE_ROTATE_TURNS
            second = asyncio.create_task(t._translate("次の発話です。", source))
            fresh = {
                "thread": {"id": "fresh-thread"},
                "serviceTier": live_stt.TRANSLATE_SERVICE_TIER,
            }
            await _answer(t, proc, fresh, rid)
            assert await _result(second) == ""
            requests = [json.loads(w) for w in proc.stdin.writes]
            interrupts = [r["params"] for r in requests if r.get("method") == "turn/interrupt"]
            assert interrupts == [{"threadId": old_thread, "turnId": "old-turn"}] * 2
        finally:
            await _stop_reader(reader_task)

    asyncio.run(scenario())
