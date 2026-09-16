"""U4 reverse leg: which thread each turn runs on, and what a failure takes down with it.

The grading check for queue row 3 unit (4). Stub seeded by the lead: fill each body in
place, never rename a case and never delete one. Add a case with the next free name where
the contract needs more.

Contract, in the order the cases sit below:

- **The flag decides HOW MANY legs exist and nothing else.** One-way constructs exactly one
  leg, keyed on the session's source language, and every existing caller reaches it through
  defaulted arguments -- `CodexTranslator(..., sources=None)`, `submit(seq, text,
  source=None)`, `translator_brief(source=None)`. Under `--two-way` there are two legs and
  the settled label picks one per turn. No `--two-way` conditional belongs inside
  `CodexTranslator` at all.
- **Two immutable direction-specific threads in the ONE app-server**, never a bidirectional
  instruction and never a second process (`translation-leg.md`). Each leg holds its own
  `thread_id`, its own `developerInstructions`, its own glossary snapshot and its own turn
  count, because splitting threads also splits dialogue history. The reverse thread opens
  LAZILY, on the first utterance in that direction, so a session that never hears the other
  language pays nothing.
- **The translator never infers direction from the text.** The source language is settled by
  the audio LID upstream, and a wrong whisper token yields FLUENT text in the conditioned
  script, so a text-side re-decision is asked to recover evidence already erased.
- **The same canonical glossary rides both legs, rendered in each one's direction**: `term =
  rendering` toward English, `rendering = term` toward Japanese. Only a PAIRED term can
  appear in the reverse brief, an unpaired term having no English key to name it by. A
  changed brief rotates the leg whose brief moved and leaves the other leg's thread alone;
  so does `TRANSLATE_ROTATE_TURNS`, counted per leg.
- **Degrade scope follows ownership.** An app-server EOF disables BOTH directions and a
  respawn recreates a thread for each; a poisoned or stalled turn replaces ONLY its own
  direction's thread. `enabled`, the failure count, the recovery budget and the backoff stay
  session-scoped, there being one app-server behind both directions. On any failure the
  transcript stays source-only -- no `TGT` line is published for a turn that did not return.

Weights-free and network-free by construction: drive the shipped `CodexTranslator` against a
scripted transport and record the `threadId` every `turn/start` named. Never spawn `codex`.

Import `live_stt` as the filled bodies need it; the skeleton leaves it out so an all-unfilled
file still passes `ruff check`.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import live_stt
from streaming import Segment


@dataclass(frozen=True)
class _Turn:
    thread_id: str
    text: str


class _ScriptedStdin:
    def __init__(self, server: _ScriptedServer) -> None:
        self.server = server
        self.closed = False

    def write(self, data: bytes) -> None:
        self.server.receive(json.loads(data))

    def close(self) -> None:
        self.closed = True


class _ScriptedProc:
    def __init__(self, server: _ScriptedServer) -> None:
        self.stdout = server.reader
        self.stdin = _ScriptedStdin(server)
        self.returncode: int | None = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


class _ScriptedServer:
    def __init__(self, number: int) -> None:
        self.number = number
        self.reader = asyncio.StreamReader()
        self.proc = _ScriptedProc(self)
        self.thread_requests: dict[str, dict[str, Any]] = {}
        self.turns: list[_Turn] = []
        self.failures: dict[str, int] = {}

    def fail(self, text: str, times: int = 1) -> None:
        self.failures[text] = times

    def _feed(self, message: dict[str, Any]) -> None:
        self.reader.feed_data((json.dumps(message) + "\n").encode())

    def _result(self, rid: int, result: dict[str, Any]) -> None:
        self._feed({"id": rid, "result": result})

    def _note(self, method: str, params: dict[str, Any]) -> None:
        self._feed({"method": method, "params": params})

    def receive(self, message: dict[str, Any]) -> None:
        method = message.get("method")
        if "id" not in message:
            return
        rid = int(message["id"])
        if method == "initialize":
            self._result(rid, {})
        elif method == "thread/start":
            thread_id = f"server-{self.number}-thread-{len(self.thread_requests) + 1}"
            params = dict(message.get("params") or {})
            self.thread_requests[thread_id] = params
            self._result(
                rid,
                {
                    "thread": {"id": thread_id},
                    "serviceTier": live_stt.TRANSLATE_SERVICE_TIER,
                },
            )
        elif method == "turn/start":
            params = message.get("params") or {}
            text = str(params["input"][0]["text"])
            self.turns.append(_Turn(str(params["threadId"]), text))
            self._result(rid, {})
            if self.failures.get(text, 0):
                self.failures[text] -= 1
                self._note("error", {"error": {"message": "scripted poisoned turn"}})
            else:
                self._note(
                    "item/completed",
                    {"item": {"type": "agentMessage", "text": f"translated:{text}"}},
                )
                self._note("turn/completed", {})
        else:
            self._result(rid, {})

    def stop(self) -> None:
        self.proc.returncode = 0
        self.reader.feed_eof()


class _CodexFactory:
    def __init__(self) -> None:
        self.servers: list[_ScriptedServer] = []

    async def exec(self, *_args: object, **_kwargs: object) -> _ScriptedProc:
        server = _ScriptedServer(len(self.servers) + 1)
        self.servers.append(server)
        return server.proc

    @property
    def turns(self) -> list[_Turn]:
        return [turn for server in self.servers for turn in server.turns]

    @property
    def thread_requests(self) -> dict[str, dict[str, Any]]:
        return {
            thread_id: params
            for server in self.servers
            for thread_id, params in server.thread_requests.items()
        }


def _thread_for_text(factory: _CodexFactory, text: str) -> str:
    matches = [turn.thread_id for turn in factory.turns if turn.text == text]
    assert len(matches) == 1, f"expected one turn for {text!r}, got {matches}"
    return matches[0]


async def _start(
    monkeypatch: pytest.MonkeyPatch,
    *,
    context: live_stt.SessionContext | None = None,
    output_file: live_stt.TranscriptFile | None = None,
    sources: tuple[str, ...] | None = None,
) -> tuple[live_stt.CodexTranslator, _CodexFactory]:
    factory = _CodexFactory()
    monkeypatch.setattr(live_stt.asyncio, "create_subprocess_exec", factory.exec)
    translator = live_stt.CodexTranslator(context, output_file, sources=sources)
    assert await translator.start()
    return translator, factory


def test_one_way_opens_exactly_one_leg_and_translates_as_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja")
        translator, factory = await _start(monkeypatch)
        try:
            translator.submit(1, "one-way-default")
            translator.submit_sentinel()
            await translator.run()

            assert set(translator._legs) == {"ja"}
            assert len(factory.servers) == 1
            assert len(factory.thread_requests) == 1
            (thread_id,) = factory.thread_requests
            assert _thread_for_text(factory, "one-way-default") == thread_id
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_two_way_opens_one_leg_per_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))
        try:
            assert await translator._translate("two-way-ja", "ja")
            assert await translator._translate("two-way-en", "en")

            assert set(translator._legs) == {"ja", "en"}
            assert len(factory.servers) == 1
            assert len(factory.thread_requests) == 2
            assert _thread_for_text(factory, "two-way-ja") != _thread_for_text(
                factory, "two-way-en"
            )
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_the_source_language_selects_the_thread_the_turn_runs_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))
        try:
            assert await translator._translate("route-ja", "ja")
            assert await translator._translate("route-en", "en")
            ja_thread = _thread_for_text(factory, "route-ja")
            en_thread = _thread_for_text(factory, "route-en")

            assert translator._legs["ja"].thread_id == ja_thread
            assert translator._legs["en"].thread_id == en_thread
            assert ja_thread != en_thread
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_the_reverse_thread_opens_lazily_on_its_first_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))
        try:
            assert len(factory.thread_requests) == 1
            assert translator._legs["ja"].thread_id is not None
            assert translator._legs["en"].thread_id is None

            assert await translator._translate("first-reverse-turn", "en")

            assert len(factory.thread_requests) == 2
            assert translator._legs["en"].thread_id == _thread_for_text(
                factory, "first-reverse-turn"
            )
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_each_leg_carries_its_own_immutable_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))
        try:
            assert await translator._translate("open-reverse", "en")
            ja_thread = translator._legs["ja"].thread_id
            en_thread = translator._legs["en"].thread_id
            assert ja_thread is not None and en_thread is not None

            ja_instructions = factory.thread_requests[ja_thread]["developerInstructions"]
            en_instructions = factory.thread_requests[en_thread]["developerInstructions"]
            assert translator._legs["ja"].instructions == live_stt._LEG_INSTRUCTIONS["ja"]
            assert translator._legs["en"].instructions == live_stt._LEG_INSTRUCTIONS["en"]
            assert str(ja_instructions).startswith(translator._legs["ja"].instructions)
            assert str(en_instructions).startswith(translator._legs["en"].instructions)
            assert ja_instructions != en_instructions
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_the_reverse_instructions_name_the_english_to_japanese_direction() -> None:
    instructions = live_stt._LEG_INSTRUCTIONS["en"]

    assert "English→Japanese translator" in instructions
    assert "transcribed English speech" in instructions
    assert "Japanese translation" in instructions
    # The property, not one phrasing of it: the reverse leg must forbid adding what the
    # source did not say, and must name the sex case the JA→EN block measured (12.3 % ->
    # 0.9 % of turns). Which noun it uses for the source side is the author's call.
    lowered = instructions.lower()
    assert "never add information" in lowered
    assert "does not state" in lowered
    assert "sex" in lowered
    assert "Japanese→English translator" not in instructions


def _learn(ctx: live_stt.SessionContext, term: str) -> None:
    for _ in range(live_stt.CONTEXT_TERM_SUPPORT):
        ctx.observe_ja(f"{term}です")


def _pair(ctx: live_stt.SessionContext, term: str, rendering: str) -> None:
    for _ in range(live_stt.CONTEXT_EN_SUPPORT):
        ctx.observe_en(f"{term}です", f"We heard {rendering} again.")


def _paired_context() -> live_stt.SessionContext:
    context = live_stt.SessionContext()
    _learn(context, "ゴン")
    _pair(context, "ゴン", "Gon")
    return context


def test_the_glossary_renders_in_the_direction_it_is_asked_for() -> None:
    context = _paired_context()

    toward_english = context.translator_brief("ja")
    toward_japanese = context.translator_brief("en")

    assert "ゴン = Gon" in toward_english
    assert "Gon = ゴン" in toward_japanese
    assert "Gon = ゴン" not in toward_english
    assert "ゴン = Gon" not in toward_japanese


def test_the_reverse_brief_lists_only_terms_that_have_a_rendering() -> None:
    context = _paired_context()
    _learn(context, "カスケ")

    toward_english = context.translator_brief("ja")
    toward_japanese = context.translator_brief("en")

    assert "カスケ" in toward_english
    assert "Gon = ゴン" in toward_japanese
    assert "カスケ" not in toward_japanese


def test_a_glossary_change_rotates_only_the_leg_whose_brief_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        context = live_stt.SessionContext()
        translator, factory = await _start(monkeypatch, context=context, sources=("ja", "en"))
        try:
            assert await translator._translate("glossary-ja-before", "ja")
            assert await translator._translate("glossary-en-before", "en")
            ja_before = _thread_for_text(factory, "glossary-ja-before")
            en_before = _thread_for_text(factory, "glossary-en-before")

            _learn(context, "カスケ")  # unpaired: forward brief only
            assert await translator._translate("glossary-en-after", "en")
            assert await translator._translate("glossary-ja-after", "ja")
            en_after = _thread_for_text(factory, "glossary-en-after")
            ja_after = _thread_for_text(factory, "glossary-ja-after")

            assert en_after == en_before
            assert ja_after != ja_before
            assert translator._legs["en"].thread_id == en_after
            assert translator._legs["ja"].thread_id == ja_after
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_a_leg_rotates_on_its_own_turn_count(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(live_stt, "TRANSLATE_ROTATE_TURNS", 2)
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))
        try:
            for number in range(1, 4):
                assert await translator._translate(f"cadence-ja-{number}", "ja")
                assert await translator._translate(f"cadence-en-{number}", "en")

            ja_threads = [
                _thread_for_text(factory, f"cadence-ja-{number}") for number in range(1, 4)
            ]
            en_threads = [
                _thread_for_text(factory, f"cadence-en-{number}") for number in range(1, 4)
            ]

            assert ja_threads[0] == ja_threads[1]
            assert ja_threads[2] != ja_threads[1]
            assert en_threads[0] == en_threads[1]
            assert en_threads[2] != en_threads[1]
            assert set(ja_threads).isdisjoint(en_threads)
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_an_app_server_eof_disables_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))
        try:
            assert await translator._translate("eof-ja-before", "ja")
            assert await translator._translate("eof-en-before", "en")
            turns_before = len(factory.turns)

            factory.servers[0].stop()
            reader_task = translator._reader_task
            assert reader_task is not None
            await reader_task

            assert translator.enabled is False
            assert await translator._translate("eof-ja-after", "ja") == ""
            assert await translator._translate("eof-en-after", "en") == ""
            assert len(factory.turns) == turns_before
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_a_respawn_recreates_a_thread_for_each_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))
        try:
            assert await translator._translate("respawn-ja-before", "ja")
            assert await translator._translate("respawn-en-before", "en")
            old_threads = {
                _thread_for_text(factory, "respawn-ja-before"),
                _thread_for_text(factory, "respawn-en-before"),
            }

            factory.servers[0].stop()
            reader_task = translator._reader_task
            assert reader_task is not None
            await reader_task
            assert await translator._recover("ja")

            assert len(factory.servers) == 2
            # A respawn re-runs start(), which opens and probes the startup leg's thread.
            # The other direction reopens on its OWN next turn rather than eagerly here:
            # that reopen is free at the point it is needed and carries the CURRENT
            # glossary, while opening both inside the recovery would charge the reverse
            # direction's handshake to whichever caption happened to arrive in this one.
            # What must hold is that no leg keeps an id from the dead process.
            assert old_threads.isdisjoint(
                {leg.thread_id for leg in translator._legs.values() if leg.thread_id}
            )

            assert await translator._translate("respawn-ja-after", "ja")
            assert await translator._translate("respawn-en-after", "en")
            new_threads = {
                _thread_for_text(factory, "respawn-ja-after"),
                _thread_for_text(factory, "respawn-en-after"),
            }
            assert len(new_threads) == 2
            assert old_threads.isdisjoint(new_threads)
            assert new_threads <= set(factory.servers[1].thread_requests)
            assert {leg.thread_id for leg in translator._legs.values()} == new_threads
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_a_poisoned_turn_replaces_only_its_own_direction_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(live_stt, "TRANSLATE_MAX_FAILURES", 1)
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))

        async def no_wait() -> None:
            return

        translator._abort_turn = no_wait  # type: ignore[method-assign]
        try:
            assert await translator._translate("poison-ja-before", "ja")
            assert await translator._translate("poison-en-before", "en")
            ja_before = translator._legs["ja"].thread_id
            en_before = translator._legs["en"].thread_id
            assert ja_before is not None and en_before is not None

            factory.servers[0].fail("poisoned-reverse")
            assert await translator._translate("poisoned-reverse", "en") == ""
            assert translator.enabled is False
            assert await translator._recover("en")

            assert len(factory.servers) == 1
            assert translator._legs["ja"].thread_id == ja_before
            assert translator._legs["en"].thread_id != en_before
            assert translator.enabled is True
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_a_failed_turn_leaves_the_transcript_source_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        path = tmp_path / "failed-turn.txt"
        transcript = live_stt.TranscriptFile(path)
        translator, factory = await _start(
            monkeypatch, output_file=transcript, sources=("ja", "en")
        )

        async def no_wait() -> None:
            return

        translator._abort_turn = no_wait  # type: ignore[method-assign]
        monkeypatch.setattr(live_stt, "write_stdout", lambda _text: None)
        try:
            text = "source-only-on-failure"
            live_stt.emit_line("SRC", 7, text, transcript)
            factory.servers[0].fail(text)
            translator.submit(7, text, "en")
            translator.submit_sentinel()
            await translator.run()
        finally:
            await translator.close()
            transcript.close()

        events = [line.split("] ", 1)[1] for line in path.read_text(encoding="utf-8").splitlines()]
        assert events == [f"SRC 7: {text}"]

    asyncio.run(scenario())


class _VacVad:
    def __init__(self, script: Sequence[bool]) -> None:
        self.script = tuple(script)
        self.calls = 0
        self.queued = 0

    def accept_waveform(self, _block: np.ndarray) -> None:
        was_speech = self.is_speech_detected()
        self.calls += 1
        if was_speech and not self.is_speech_detected():
            self.queued += 1

    def is_speech_detected(self) -> bool:
        index = min(self.calls, len(self.script)) - 1
        return self.script[index] if index >= 0 else False

    def empty(self) -> bool:
        return self.queued == 0

    def pop(self) -> None:
        assert self.queued
        self.queued -= 1


class _VacRecognizer:
    def __init__(self, text: str) -> None:
        self.text = text

    def decode_segments(
        self, samples: np.ndarray, language: str | None = None
    ) -> tuple[str, list[Segment]]:
        duration = len(samples) / live_stt.SAMPLE_RATE
        return self.text, [Segment(0.0, duration, self.text)]


class _AcceptEnglish:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def decide(self, pcm: np.ndarray) -> str:
        self.calls.append(pcm.copy())
        return "en"


def test_the_settled_label_routes_the_published_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))
        text = "英語の発話"
        script = [True, True, True, True, False]
        window = live_stt.SAMPLE_RATE // 2
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        for value in range(len(script)):
            queue.put_nowait(np.full(window, value + 1, dtype=np.float32))
        queue.put_nowait(None)
        detector = _AcceptEnglish()
        emitted: list[tuple[str, int, str]] = []

        def capture(tag: str, seq: int, body: str, _output: object, **_kwargs: object) -> None:
            emitted.append((tag, seq, body))

        monkeypatch.setattr(live_stt, "VAD_PRE_PAD_S", 0.0)
        monkeypatch.setattr(live_stt, "emit_line", capture)
        try:
            await live_stt._vac_segments(
                _VacRecognizer(text),
                _VacVad(script),
                window,
                queue,
                live_stt.State(),
                None,
                translator=translator,
                detector=detector,
            )
            translator.submit_sentinel()
            await translator.run()

            assert detector.calls
            assert ("SRC", 1, text) in emitted
            assert ("TGT", 1, f"translated:{text}") in emitted
            routed = _thread_for_text(factory, text)
            assert routed == translator._legs["en"].thread_id
            assert routed != translator._legs["ja"].thread_id
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_the_flag_off_path_submits_and_translates_exactly_as_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja")
        context = _paired_context()
        translator, factory = await _start(monkeypatch, context=context)
        emitted: list[tuple[str, int, str]] = []
        monkeypatch.setattr(
            live_stt,
            "emit_line",
            lambda tag, seq, text, _output: emitted.append((tag, seq, text)),
        )
        try:
            translator.submit(23, "従来の呼び出し")
            translator.submit_sentinel()
            await translator.run()

            assert context.translator_brief() == context.translator_brief("ja")
            assert set(translator._legs) == {"ja"}
            assert len(factory.thread_requests) == 1
            assert emitted == [("TGT", 23, "translated:従来の呼び出し")]
            assert _thread_for_text(factory, "従来の呼び出し") in factory.thread_requests
        finally:
            await translator.close()

    asyncio.run(scenario())


def test_no_turn_is_ever_issued_on_a_thread_from_the_other_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr(live_stt, "TRANSLATE_ROTATE_TURNS", 1)
        translator, factory = await _start(monkeypatch, sources=("ja", "en"))
        routed: dict[str, list[str]] = {"ja": [], "en": []}
        turns = [
            ("ja", "This text looks English but the settled label is ja."),
            ("en", "これは日本語に見えますがラベルは英語です。"),
            ("ja", "English-looking text still follows the audio label."),
            ("en", "日本語らしい文字列でも音声ラベルに従います。"),
        ]
        try:
            for source, text in turns:
                assert await translator._translate(text, source)
                thread_id = _thread_for_text(factory, text)
                routed[source].append(thread_id)
                assert thread_id == translator._legs[source].thread_id
                other = "en" if source == "ja" else "ja"
                assert thread_id != translator._legs[other].thread_id

            assert len(set(routed["ja"])) == 2
            assert len(set(routed["en"])) == 2
            assert set(routed["ja"]).isdisjoint(routed["en"])
        finally:
            await translator.close()

    asyncio.run(scenario())
