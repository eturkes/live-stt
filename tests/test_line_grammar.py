"""U0 line grammar: every published line is `SRC n:` or `TGT n:`, in every mode.

The grading check for queue row 3 unit (0). Red on `d5ec9b7` against unfixed production --
11 failed, 3 passed -- and green on the unit's own commit. The three that pass unfixed are
regression guards rather than witnesses: one number shared by a source line and its target,
legacy `JA`/`EN` transcripts still parsing, and `lag_s` staying TGT - SRC.
"""

from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path

import numpy as np
import pytest

import live_stt as app
import session_report as sr
from tests.test_session_report import event, report, write
from tests.test_streaming import _run_vac, _StubRec, _StubVad


def test_vac_source_line_uses_the_src_tag() -> None:
    lines, _, _ = _run_vac([True] * 40 + [False])

    assert [(tag, seq) for tag, seq, _ in lines] == [("SRC", 1)]


def test_translator_line_uses_the_tgt_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[tuple[str, int, str]] = []
    translator = app.CodexTranslator()
    translator.enabled = True

    async def translate(_source: str) -> str:
        return "Hello."

    monkeypatch.setattr(translator, "_translate", translate)
    monkeypatch.setattr(
        app,
        "emit_line",
        lambda tag, seq, text, _output: emitted.append((tag, seq, text)),
    )
    translator.queue.put_nowait((7, "こんにちは。"))
    translator.queue.put_nowait(None)

    asyncio.run(translator.run())

    assert emitted == [("TGT", 7, "Hello.")]


def test_sherpa_branch_source_line_uses_the_src_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    emitted: list[tuple[str, int, str]] = []
    samples = np.zeros(1600, dtype=np.float32)
    state = app.State()
    state.segment_queue_depth = 1

    monkeypatch.setattr(app, "_decode", lambda _rec, _samples: "こんにちは。")
    monkeypatch.setattr(
        app,
        "emit_line",
        lambda tag, seq, text, _output: emitted.append((tag, seq, text)),
    )

    async def decode() -> None:
        queue: asyncio.Queue = asyncio.Queue()
        queue.put_nowait((0, len(samples), samples))
        queue.put_nowait(None)
        await app._decode_segments(object(), queue, state, None)

    asyncio.run(decode())

    assert emitted == [("SRC", 1, "こんにちは。")]


def _publish_pair(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    transcript = app.TranscriptFile(path)
    translator = app.CodexTranslator(output_file=transcript)
    translator.enabled = True
    script = [True] * 40 + [False]

    async def translate(_source: str) -> str:
        return "Hello."

    monkeypatch.setattr(translator, "_translate", translate)

    async def publish() -> None:
        queue: asyncio.Queue = asyncio.Queue()
        for _ in script:
            queue.put_nowait(np.zeros(1600, dtype=np.float32))
        queue.put_nowait(None)
        await app._vac_segments(
            _StubRec(),
            _StubVad(script),
            1600,
            queue,
            app.State(),
            transcript,
            translator=translator,
        )
        translator.submit_sentinel()
        await translator.run()

    asyncio.run(publish())
    transcript.close()


def test_transcript_file_carries_the_same_grammar_as_the_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "session.txt"
    monkeypatch.setattr(app, "_STDOUT_TTY", False)

    _publish_pair(monkeypatch, path)

    screen = [line.removeprefix("  ") for line in capsys.readouterr().out.splitlines()]
    saved = [line.split("] ", 1)[1] for line in path.read_text(encoding="utf-8").splitlines()]
    assert screen == saved
    assert [line.partition(": ")[0] for line in saved] == ["SRC 1", "TGT 1"]


def test_one_number_is_shared_by_a_src_line_and_its_tgt_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "session.txt"
    monkeypatch.setattr(app, "_STDOUT_TTY", False)

    _publish_pair(monkeypatch, path)

    bodies = [line.split("] ", 1)[1] for line in path.read_text(encoding="utf-8").splitlines()]
    numbers = [int(body.partition(": ")[0].split()[1]) for body in bodies]
    assert numbers == [1, 1]


def test_session_report_reads_the_src_tgt_grammar(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "roles",
        [
            event("10:00:01", "SRC", 1, "こんにちは。"),
            event("10:00:03", "TGT", 1, "Hello."),
            event("10:00:10", "SRC", 2, "次の文です。"),
        ],
    )

    session = report([path])["sessions"][0]

    assert (session["captions"], session["translated"]) == (2, 1)
    assert [row["n"] for row in session["missing"]] == [2]


def test_session_report_reads_a_legacy_ja_en_transcript(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "legacy",
        [
            event("10:00:01", "JA", 1, "こんにちは。"),
            event("10:00:03", "EN", 1, "Hello."),
            event("10:00:10", "JA", 2, "次の文です。"),
        ],
    )

    session = report([path])["sessions"][0]

    assert (session["captions"], session["translated"]) == (2, 1)
    assert [row["n"] for row in session["missing"]] == [2]


def test_session_report_reads_a_file_holding_both_grammars(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "mixed",
        [
            event("10:00:01", "SRC", 1, "こんにちは。"),
            event("10:00:03", "TGT", 1, "Hello."),
            event("10:00:10", "JA", 2, "次の文です。"),
            event("10:00:14", "EN", 2, "The next sentence."),
        ],
    )

    session = report([path])["sessions"][0]

    assert (session["captions"], session["translated"]) == (2, 2)
    assert session["missing"] == []
    assert session["lag_s"]["pairs"] == 2


def test_reported_lag_is_tgt_minus_src(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "lag",
        [
            event("10:00:01", "JA", 9, "こんにちは。"),
            event("10:00:05", "EN", 9, "Hello."),
        ],
    )

    lag = report([path])["sessions"][0]["lag_s"]

    assert lag == {"pairs": 1, "p50": 4.0, "p90": 4.0, "max": 4.0}


def test_no_production_emit_site_still_writes_a_ja_or_en_tag() -> None:
    source = Path(app.__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    stale = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "emit_line"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in {"JA", "EN"}
        ):
            continue
        stale.append((node.lineno, node.args[0].value))

    assert stale == []


def test_the_shipped_docs_show_the_src_tgt_grammar() -> None:
    readme = (Path(app.__file__).parent / "README.md").read_text(encoding="utf-8")
    required = {"`SRC n:`", "`TGT n:`", "SRC 1:", "TGT 1:"}

    assert {text for text in required if text not in readme} == set()
    assert re.findall(r"\b(?:JA|EN) (?:n|\d+):", readme) == []
    assert "JA-only" not in readme


def _degraded_report(tmp_path: Path) -> dict:
    path = write(
        tmp_path,
        "degraded",
        [
            event("10:00:01", "JA", 1, "一。"),
            event("10:00:03", "EN", 1, "One."),
            event("10:00:10", "JA", 2, "二。"),
            event("10:00:20", "JA", 3, "三。"),
            event("10:00:30", "JA", 4, "四。"),
        ],
    )
    return report([path])


def test_session_report_degrade_json_uses_last_tgt_and_source_only_after(tmp_path: Path) -> None:
    degrade = _degraded_report(tmp_path)["sessions"][0]["degrade"]

    assert degrade is not None
    assert {"last_tgt", "source_only_after"} <= degrade.keys()
    assert {"last_en", "ja_only_after"}.isdisjoint(degrade)
    assert (degrade["last_tgt"], degrade["source_only_after"]) == (1, 3)


def test_session_report_renders_last_tgt_and_source_only_after(tmp_path: Path) -> None:
    rendered = sr.render(_degraded_report(tmp_path))

    assert "(last TGT n=1, 3 source-only after)" in rendered


def test_user_facing_degrade_strings_say_source_only() -> None:
    source = Path(app.__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    messages: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        is_print = isinstance(node.func, ast.Name) and node.func.id == "print"
        is_log = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logger"
        )
        first = node.args[0]
        if (
            (is_print or is_log)
            and isinstance(first, ast.Constant)
            and isinstance(first.value, str)
        ):
            messages.append((node.lineno, first.value))

    stale = [(line, text) for line, text in messages if "JA-only" in text]
    replacements = [(line, text) for line, text in messages if "source-only" in text]
    assert stale == []
    assert len(replacements) >= 6
