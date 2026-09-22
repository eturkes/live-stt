"""Regression locks for bounded translation queue age and its attribution."""

from __future__ import annotations

import asyncio
import logging

import live_stt
import session_report


class _Clock:
    def __init__(self, now: float):
        self.now = now

    def monotonic(self) -> float:
        return self.now


def test_a_stale_caption_is_skipped_in_both_output_channels(tmp_path, monkeypatch, caplog):
    # The live cascade put targets beside unrelated speech tens of seconds later.
    # A stale turn must spend neither model time nor a target line, while both
    # durable channels name the deliberate source-only decision.
    async def scenario():
        clock = _Clock(100.0)
        monkeypatch.setattr(live_stt, "time", clock)
        transcript = live_stt.TranscriptFile(tmp_path / "session.txt")
        translator = live_stt.CodexTranslator(output_file=transcript)
        translator.enabled = True
        translated: list[str] = []

        async def fake_translate(text: str, source: str | None = None) -> str:
            translated.append(text)
            return "Too late."

        translator._translate = fake_translate  # type: ignore[assignment]
        translator.submit(17, "古い字幕")
        clock.now += 16.0
        translator.submit_sentinel()
        with caplog.at_level(logging.WARNING, logger="live_stt"):
            await translator.run()
        transcript.close()
        return translator, translated

    translator, translated = asyncio.run(scenario())
    body = (tmp_path / "session.txt").read_text(encoding="utf-8")
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]

    assert translated == []
    assert "TGT 17:" not in body
    assert "-- translation skipped (stale): 17" in body
    assert warnings == ["caption 17 not translated: queued 16 s, over TRANSLATE_MAX_STALENESS_S"]
    assert getattr(translator, "stale_translations", None) == 1
    assert translator.dropped_translations == 0
    assert getattr(live_stt, "TRANSLATE_MAX_STALENESS_S", None) == 15.0


def test_a_fresh_caption_is_translated_without_counting_a_skip(tmp_path, monkeypatch, caplog):
    # The bound protects temporal relevance, not ordinary slow-but-current turns.
    # A check against processing time or an inverted comparison would discard
    # captions that are still useful.
    async def scenario():
        clock = _Clock(200.0)
        monkeypatch.setattr(live_stt, "time", clock)
        transcript = live_stt.TranscriptFile(tmp_path / "fresh.txt")
        translator = live_stt.CodexTranslator(output_file=transcript)
        translator.enabled = True
        translated: list[str] = []

        async def fake_translate(text: str, source: str | None = None) -> str:
            translated.append(text)
            return "Still current."

        translator._translate = fake_translate  # type: ignore[assignment]
        translator.submit(18, "新しい字幕")
        clock.now += 14.0
        translator.submit_sentinel()
        with caplog.at_level(logging.WARNING, logger="live_stt"):
            await translator.run()
        transcript.close()
        return translator, translated

    translator, translated = asyncio.run(scenario())
    body = (tmp_path / "fresh.txt").read_text(encoding="utf-8")
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]

    assert translated == ["新しい字幕"]
    assert "TGT 18: Still current." in body
    assert "translation skipped (stale)" not in body
    assert warnings == []
    assert getattr(translator, "stale_translations", None) == 0


def test_a_stale_caption_recovers_the_leg_before_it_is_skipped(monkeypatch):
    # run() is the recovery driver's only caller. Checking age first appears to
    # skip the right caption, but it leaves a down leg down; the recovery event
    # must therefore precede the stale note, then serve the fresh caption behind it.
    async def scenario():
        clock = _Clock(300.0)
        monkeypatch.setattr(live_stt, "time", clock)
        translator = live_stt.CodexTranslator()
        translator.enabled = False
        events: list[str] = []
        fresh_done = asyncio.Event()

        async def fake_recover(source: str | None = None) -> bool:
            events.append("recover")
            translator.enabled = True
            return True

        async def fake_translate(text: str, source: str | None = None) -> str:
            events.append(f"translate:{text}")
            if text == "新しい字幕":
                fresh_done.set()
            return "Fresh target."

        def fake_note(text: str, output_file) -> None:
            events.append(f"note:{text}")

        def fake_emit(tag: str, seq: int, text: str, output_file, **_kwargs) -> None:
            events.append(f"emit:{tag}:{seq}:{text}")

        translator._recover = fake_recover  # type: ignore[assignment]
        translator._translate = fake_translate  # type: ignore[assignment]
        monkeypatch.setattr(live_stt, "emit_note", fake_note)
        monkeypatch.setattr(live_stt, "emit_line", fake_emit)

        translator.submit(30, "古い字幕")
        clock.now += 15.0
        translator.submit(31, "新しい字幕")
        clock.now += 1.0

        run_task = asyncio.create_task(translator.run())
        await asyncio.wait_for(fresh_done.wait(), timeout=1.0)
        translator.submit_sentinel()
        await asyncio.wait_for(run_task, timeout=1.0)
        return translator, events

    translator, events = asyncio.run(scenario())

    assert events == [
        "recover",
        "note:translation skipped (stale): 30",
        "translate:新しい字幕",
        "emit:TGT:31:Fresh target.",
    ]
    assert translator.enabled is True
    assert getattr(translator, "stale_translations", None) == 1


def test_the_four_tuple_drains_before_the_shutdown_close(monkeypatch):
    # Ctrl+C queues the sentinel before run() drains the final caption. The age
    # stamp must not disturb that ordering: one last target lands, run returns on
    # None, and only then may close tear down the app-server.
    async def scenario():
        clock = _Clock(400.0)
        monkeypatch.setattr(live_stt, "time", clock)
        translator = live_stt.CodexTranslator()
        translator.enabled = True
        events: list[str] = []

        async def fake_translate(text: str, source: str | None = None) -> str:
            events.append(f"translate:{text}")
            return "Drained target."

        async def fake_end_proc() -> None:
            events.append("close")

        def fake_emit(tag: str, seq: int, text: str, output_file, **_kwargs) -> None:
            events.append(f"emit:{tag}:{seq}:{text}")

        translator._translate = fake_translate  # type: ignore[assignment]
        translator._end_proc = fake_end_proc  # type: ignore[assignment]
        monkeypatch.setattr(live_stt, "emit_line", fake_emit)
        translator.queue.put_nowait((44, "最後の字幕", None, clock.now))
        translator.submit_sentinel()

        await translator.run()
        events.append("run returned")
        await translator.close()
        return translator, events

    translator, events = asyncio.run(scenario())

    assert events == [
        "translate:最後の字幕",
        "emit:TGT:44:Drained target.",
        "run returned",
        "close",
    ]
    assert translator.queue.empty()
    assert translator.enabled is False


def test_session_report_separates_stale_skips_from_screen_declines(tmp_path):
    # Both decisions share the `caption n not translated:` prefix, but only a
    # decline says the text screen refused the caption. A stale queue decision
    # must outrank that broad pattern or the post-session diagnosis is false.
    loop = "ねこ" * 30
    transcript = tmp_path / "2026-09-04T10-00-00.txt"
    transcript.write_text(
        "\n".join(
            [
                "[2026-09-04T10:00:01+09:00] SRC 1: こんにちは",
                f"[2026-09-04T10:00:20+09:00] SRC 2: {loop}",
                "[2026-09-04T10:00:30+09:00] SRC 3: おはようございます",
                "[2026-09-04T10:00:32+09:00] TGT 3: Good morning.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    log = tmp_path / "session.log"
    log.write_text(
        "\n".join(
            [
                "[2026-09-04 10:00:17,000] WARNING caption 1 not translated:"
                " queued 16 s, over TRANSLATE_MAX_STALENESS_S",
                "[2026-09-04 10:00:20,000] WARNING caption 2 not translated:"
                " 60 of 60 characters are one repeated unit",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    session = session_report.read_session(str(transcript))
    session.events = session_report.read_log(str(log))
    reasons = {row["n"]: row["why"] for row in session_report.explain_missing(session)}

    assert reasons == {1: "stale", 2: session_report.DECLINED}
    assert getattr(session_report, "STALE", None) == "stale"
