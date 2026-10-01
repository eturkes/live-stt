"""Queue: Two-way English captions pass the screen — decode language owns the latin rule."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from string import ascii_letters
from typing import cast

import pytest

import live_stt
import session_report
from tests.test_split_publication import BAD, CLEAN, GOOD, _drive, _texts

ENGLISH = ("We ", "will ", "review ", "the ", "new ", "design ", "after ", "lunch.")
LOOP = "wait" * 12


def _screen(text: str, language: str | None) -> str | None:
    # The contract extends the baseline signature; the red tree must still typecheck.
    defect = cast(Callable[[str, str | None], str | None], live_stt.caption_defect)
    return defect(text, language)


@pytest.mark.parametrize("close", [True, False], ids=["vad-close", "sentinel"])
@pytest.mark.parametrize("trim_s", [0.5, 60.0], ids=["trimmed", "final-only"])
@pytest.mark.parametrize("outcomes", [["en"], [None, "en"]], ids=["accepted", "delayed"])
def test_two_way_english_pieces_publish_under_the_accepted_decode_language(
    monkeypatch: pytest.MonkeyPatch, close: bool, trim_s: float, outcomes: list[str | None]
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja")
    run = _drive(monkeypatch, CLEAN, reverse=ENGLISH, outcomes=outcomes, trim_s=trim_s, close=close)

    assert run.detector is not None and run.detector.accepted_at is not None
    assert "en" in run.rec.languages, "the witness never decoded under the accepted en token"
    assert "".join(_texts(run)) == "".join(ENGLISH), "English decode disappeared at publication"
    assert all(not line.held and line.window >= run.detector.accepted_at for line in run.lines)
    assert [line.seq for line in run.lines] == list(range(1, len(run.lines) + 1))
    assert run.submitted == [(line.seq, line.text, "en") for line in run.lines]
    assert run.state.dropped_captions == 0
    if trim_s == 0.5:
        assert sum(processor.trims for processor in run.processors) > 0
        assert len(run.lines) > 1
    else:
        assert len(run.lines) == 1


def test_two_way_english_without_a_translator_still_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja")
    run = _drive(monkeypatch, CLEAN, reverse=ENGLISH, outcomes=["en"], translate=False)

    assert "en" in run.rec.languages
    assert "".join(_texts(run)) == "".join(ENGLISH)
    assert run.submitted == []
    assert run.state.dropped_captions == 0


def test_two_way_held_english_uses_its_decode_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja")
    labels = [*CLEAN[:4], "", "", *CLEAN[4:]]
    reverse = [*ENGLISH[:4], "", "", *ENGLISH[4:]]
    run = _drive(
        monkeypatch,
        labels,
        reverse=reverse,
        script=[True] * 4 + [False] * 2 + [True] * 4,
        outcomes=["en", None, None],
        trim_s=60.0,
    )

    assert run.detector is not None and len(run.detector.calls) == 3
    assert [(line.seq, line.text, line.held) for line in run.lines] == [
        (1, "".join(ENGLISH[:4]), False),
        (2, "".join(ENGLISH[4:]), True),
    ]
    assert run.submitted == [(line.seq, line.text, "en") for line in run.lines]
    assert run.state.dropped_captions == 0


@pytest.mark.parametrize("two_way", [False, True], ids=["one-way", "two-way"])
def test_control_latin_dominant_japanese_pieces_still_drop(
    monkeypatch: pytest.MonkeyPatch, two_way: bool
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja")
    run = _drive(monkeypatch, [BAD] + [""] * 3, outcomes=["ja"] if two_way else None)

    assert run.raw == [BAD]
    assert "ja" in run.rec.languages and "en" not in run.rec.languages
    assert run.lines == run.submitted == run.observed == []
    assert run.state.dropped_captions == 1


@pytest.mark.parametrize("two_way", [False, True], ids=["one-way", "two-way"])
def test_control_repetition_still_drops_an_english_decode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, two_way: bool
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja" if two_way else "en")
    run = _drive(
        monkeypatch,
        [LOOP] + [""] * 3,
        reverse=[LOOP] + [""] * 3,
        outcomes=["en"] if two_way else None,
    )

    assert run.raw == [LOOP]
    assert "en" in run.rec.languages
    assert run.lines == run.submitted == run.observed == []
    assert run.state.dropped_captions == 1
    assert "repeated unit" in caplog.text, "a latin drop hid a disabled repetition rule"


@pytest.mark.parametrize("close", [True, False], ids=["vad-close", "sentinel"])
def test_control_one_way_english_still_publishes(
    monkeypatch: pytest.MonkeyPatch, close: bool
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "en")
    run = _drive(monkeypatch, ENGLISH, close=close, translate=False)

    assert set(run.rec.languages) == {"en"}
    assert "".join(_texts(run)) == "".join(ENGLISH)
    assert run.submitted == []
    assert run.state.dropped_captions == 0


@pytest.mark.parametrize("language", ["ja", "en"])
def test_control_omitted_language_keeps_the_session_screen(
    monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    cases = [
        ("", False),
        (" 0123!?、。", False),
        (GOOD, False),
        ("ABCDあ", False),
        ("ABCDEあ", language == "ja"),
        (BAD, language == "ja"),
        ("あ" * 39, False),
        ("あ" * 40, True),
        (LOOP, True),
    ]
    for text, dropped in cases:
        assert (live_stt.caption_defect(text) is not None) == dropped, (language, text)


@pytest.mark.parametrize("session_language", ["ja", "en"])
@pytest.mark.parametrize("decode_language", ["ja", "en"])
def test_explicit_language_owns_the_latin_rule(
    monkeypatch: pytest.MonkeyPatch, session_language: str, decode_language: str
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", session_language)

    assert (_screen(BAD, decode_language) is not None) == (decode_language == "ja")
    assert _screen(GOOD, decode_language) is None


@pytest.mark.parametrize("language", ["ja", "en"])
def test_explicit_none_keeps_the_session_screen(
    monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    for text in ("", GOOD, BAD, LOOP, "ABCDあ", "ABCDEあ"):
        assert _screen(text, None) == live_stt.caption_defect(text), (language, text)


@pytest.mark.parametrize("session_language", ["ja", "en"])
def test_generated_latin_boundary_obeys_only_the_decode_language(
    monkeypatch: pytest.MonkeyPatch, session_language: str
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", session_language)
    for alphabet in ("あいうえおかきく", "アイウエオカキク", "今日会議新計画話"):
        for japanese in range(len(alphabet) + 1):
            for latin in sorted({max(4 * japanese - 1, 0), 4 * japanese, 4 * japanese + 1}):
                text = ascii_letters[:latin] + alphabet[:japanese] + " 0123!?、。"
                for decode_language in ("ja", "en"):
                    dropped = decode_language == "ja" and latin > 4 * japanese
                    assert (_screen(text, decode_language) is not None) == dropped, (
                        session_language,
                        decode_language,
                        japanese,
                        latin,
                        text,
                    )


@pytest.mark.parametrize("session_language", ["ja", "en"])
def test_generated_repetition_is_language_independent(
    monkeypatch: pytest.MonkeyPatch, session_language: str
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", session_language)
    for size in range(1, 14):
        for alphabet in (ascii_letters, "あいうえおかきくけこさしす"):
            phrase = alphabet[:size]
            repetitions = (40 + size - 1) // size
            text = phrase * repetitions
            assert len(text) >= 40
            for decode_language in ("ja", "en"):
                assert _screen(text, decode_language) is not None, (
                    session_language,
                    decode_language,
                    size,
                    text,
                )
    for decode_language in ("ja", "en"):
        assert _screen("あ" * 39, decode_language) is None
        assert _screen("あ" * 40, decode_language) is not None


@pytest.mark.parametrize("language", ["ja", "en"])
def test_control_session_report_keeps_the_one_argument_screen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, language: str
) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    original = live_stt.caption_defect
    calls: list[str] = []

    def screen(text: str) -> str | None:
        calls.append(text)
        return original(text)

    monkeypatch.setattr(live_stt, "caption_defect", screen)
    path = tmp_path / "screen.txt"
    path.write_text(
        "".join(
            f"[2026-09-04T10:00:0{n}+09:00] SRC {n}: {text}\n"
            for n, text in enumerate((GOOD, BAD, LOOP), 1)
        ),
        encoding="utf-8",
    )
    report = session_report.build([session_report.read_session(str(path))], [])
    result = report["screen"]

    assert {GOOD, BAD, LOOP} <= set(calls), "the report never exercised the shipped screen"
    assert result["source_lang"] == language
    assert result["latin_drops"] == (1 if language == "ja" else 0)
    assert result["repetition_drops"] == 1
    assert result["combined_drops"] == (2 if language == "ja" else 1)
