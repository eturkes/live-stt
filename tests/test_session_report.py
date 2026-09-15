"""Locks for `session_report.py`: attribution, log ownership, empty-tree exit.

Hermetic by construction -- every transcript here is written into `tmp_path`, so
the suite never reads the gitignored `transcripts/` the tool exists to explain.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import live_stt as app  # noqa: E402
import session_report as sr  # noqa: E402

LOOP = "ねこ" * 30  # 60 chars of one 2-char unit -> past CAPTION_REPEAT_MAX_CHARS
ENGLISH = "Do you have a time?"  # latin with no japanese -> past CAPTION_LATIN_RATIO
CLEAN = "こんにちは、おはようございます"


def write(tmp_path: Path, name: str, lines: list[str]) -> str:
    p = tmp_path / f"{name}.txt"
    p.write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
    return str(p)


def event(ts: str, tag: str, n: int, text: str) -> str:
    return f"[2026-09-04T{ts}+09:00] {tag} {n}: {text}"


def log_event(ts: str, body: str, level: str = "INFO") -> str:
    return f"[2026-09-04 {ts},000] {level} {body}"


def read_log(tmp_path: Path, lines: list[str]) -> list[sr.LogEvent]:
    lp = tmp_path / "run.log"
    lp.write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
    return sr.read_log(str(lp))


def report(paths: list[str], log: list[str] | None = None, tmp_path: Path | None = None) -> dict:
    sessions = sorted((sr.read_session(p) for p in paths), key=lambda s: s.name)
    events = [] if log is None or tmp_path is None else read_log(tmp_path, log)
    unclaimed = sr.attach_logs(sessions, sorted(events, key=lambda e: e.at))
    return sr.build(sessions, unclaimed)


def reasons(session: dict) -> dict[int, str]:
    return {r["n"]: r["why"] for r in session["missing"]}


def test_no_transcripts_exits_clean(tmp_path, monkeypatch, capsys):
    """A fresh clone has no transcripts/ and that is not a failure."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["session_report.py"])
    assert sr.main() == 0
    assert "no transcripts found" in capsys.readouterr().err


def test_declined_captions_name_their_own_defect(tmp_path):
    path = write(
        tmp_path,
        "2026-09-04T10-00-00",
        [
            event("10:00:01", "JA", 1, CLEAN),
            event("10:00:03", "EN", 1, "Hello."),
            event("10:00:10", "JA", 2, LOOP),
            event("10:00:20", "JA", 3, ENGLISH),
            event("10:00:30", "JA", 4, CLEAN),
            event("10:00:32", "EN", 4, "Hello again."),
        ],
    )
    rep = report([path])
    got = reasons(rep["sessions"][0])
    assert got == {2: sr.DECLINED, 3: sr.DECLINED}
    detail = {r["n"]: r["detail"] for r in rep["sessions"][0]["missing"]}
    assert "repeated unit" in detail[2]
    assert "latin letters" in detail[3]


def test_three_strike_degrade_is_inferred_without_a_marker(tmp_path):
    """Every session saved before the marker existed still shows its own death."""
    lines = [event("11:00:01", "JA", 1, CLEAN), event("11:00:03", "EN", 1, "One.")]
    for n in range(2, 8):
        lines.append(event(f"11:0{n}:00", "JA", n, CLEAN))
    rep = report([write(tmp_path, "2026-09-04T11-00-00", lines)])
    s = rep["sessions"][0]
    assert s["degrade"]["source"] == "inferred"
    assert s["degrade"]["last_en"] == 1
    # The strike budget is spent on the first three, the rest ride the degrade.
    assert reasons(s) == {
        2: sr.STRIKE,
        3: sr.STRIKE,
        4: sr.STRIKE,
        5: sr.DISABLED,
        6: sr.DISABLED,
        7: sr.DISABLED,
    }


def test_eof_after_the_last_caption_is_still_attributed(tmp_path):
    """The cause of death is logged AFTER the final caption, by construction."""
    path = write(
        tmp_path,
        "2026-09-04T12-00-00",
        [
            event("12:00:01", "JA", 1, CLEAN),
            event("12:00:03", "EN", 1, "One."),
            event("12:00:10", "JA", 2, CLEAN),
        ],
    )
    rep = report(
        [path],
        log=[
            "[2026-09-04 12:00:12,100] ERROR codex app-server exited;"
            " JA-only for the rest of the session"
        ],
        tmp_path=tmp_path,
    )
    s = rep["sessions"][0]
    assert s["degrade"]["source"] == "log"
    assert s["degrade"]["reason"] == "codex app-server exited"
    # An EOF kills the leg between turns, so no caption was ever a strike.
    assert reasons(s) == {2: sr.DISABLED}
    assert rep["totals"]["unclaimed_log_events"] == 0


def test_shutdown_loss_is_not_read_as_a_degrade(tmp_path):
    """One trailing caption is the drain losing its last EN, not the leg dying."""
    path = write(
        tmp_path,
        "2026-09-04T13-00-00",
        [
            event("13:00:01", "JA", 1, CLEAN),
            event("13:00:03", "EN", 1, "One."),
            event("13:00:10", "JA", 2, CLEAN),
        ],
    )
    s = report([path])["sessions"][0]
    assert s["degrade"] is None
    assert reasons(s) == {2: sr.SHUTDOWN}


def test_markers_outrank_inference(tmp_path):
    path = write(
        tmp_path,
        "2026-09-04T14-00-00",
        [
            event("14:00:01", "JA", 1, CLEAN),
            event("14:00:03", "EN", 1, "One."),
            "[2026-09-04T14:00:05+09:00] -- translation disabled: codex app-server exited",
            event("14:00:10", "JA", 2, CLEAN),
            "[2026-09-04T14:00:20+09:00] -- translation restored:"
            " codex app-server respawned (attempt 1)",
            event("14:00:30", "JA", 3, CLEAN),
            event("14:00:32", "EN", 3, "Three."),
        ],
    )
    s = report([path])["sessions"][0]
    assert s["degrade"]["source"] == "marker"
    assert s["degrade"]["reason"] == "codex app-server exited"
    assert s["restores"][0]["reason"] == "codex app-server respawned (attempt 1)"


def test_sweep_names_the_caption_a_wider_bound_catches(tmp_path):
    """The count alone cannot answer which loop was escaping."""
    unit = "いい音があるので、"  # 9 chars: invisible at bound 8, caught at 9
    path = write(
        tmp_path,
        "2026-09-04T15-00-00",
        [event("15:00:01", "JA", 1, unit * 8), event("15:00:03", "EN", 1, "One.")],
    )
    rows = report([path])["screen"]["sweep"]
    by_bound = {b: r for r in rows for b in range(r["from"], r["to"] + 1)}
    assert by_bound[8]["caught"] == 0
    assert by_bound[9]["caught"] == 1
    assert by_bound[9]["new"] == [{"session": "2026-09-04T15-00-00.txt", "n": 1}]


def test_repeat_span_at_restores_the_shipped_bound(monkeypatch):
    """The sweep rebinds a shipped constant, so it must put it back on any exit."""
    before = sr.app.CAPTION_REPEAT_UNIT_CHARS
    sr.repeat_span_at("ねこねこねこ", 2)
    assert sr.app.CAPTION_REPEAT_UNIT_CHARS == before

    def boom(_text: str) -> int:
        raise RuntimeError("decode failed")

    monkeypatch.setattr(sr.app, "repeat_span", boom)
    with pytest.raises(RuntimeError):
        sr.repeat_span_at("ねこ", 2)
    assert sr.app.CAPTION_REPEAT_UNIT_CHARS == before


def test_the_rotation_boundary_tag_names_the_turn_that_pays_it(tmp_path):
    """`_translate` tests the cadence BEFORE incrementing `_turns` ⇒ the 101st turn rotates.

    Tagging `n % 100` read the 100th, which is why f398818's "0 of 75 slow turns
    at a boundary" could not constrain the tax it was quoted against.
    """
    lines = []
    for n, at in ((100, "10:00"), (101, "10:01")):
        lines.append(event(f"{at}:00", "JA", n, CLEAN))
        lines.append(event(f"{at}:07", "EN", n, "Hello"))
    session = report([write(tmp_path, "s", lines)])["sessions"][0]
    tagged = {t["n"]: t["at_rotation"] for t in session["slow_turns"]}
    assert tagged == {100: False, 101: True}


def _screen(tmp_path: Path, name: str) -> dict:
    lines = [
        event("10:00:01", "JA", 1, CLEAN),
        event("10:00:03", "EN", 1, "Hello."),
        event("10:00:10", "JA", 2, LOOP),
        event("10:00:20", "JA", 3, ENGLISH),
    ]
    return report([write(tmp_path, name, lines)])


def test_an_english_session_is_not_re_derived_under_a_screen_it_never_ran(tmp_path, monkeypatch):
    """`--source-lang en` retires the latin rule live, so the report must retire it too.

    The report imports the shipped screen precisely so a rule change moves it,
    but the drop SPLIT restated the latin rule inline -- and that copy answered
    to nothing, reporting an English caption as screened in a session that
    published it. Derive the split from `caption_defect`'s own verdict instead.
    """
    monkeypatch.setattr(sr.app, "ASR_LANGUAGE", "ja")
    ja = _screen(tmp_path, "ja")["screen"]
    assert (ja["source_lang"], ja["latin_drops"], ja["combined_drops"]) == ("ja", 1, 2)

    monkeypatch.setattr(sr.app, "ASR_LANGUAGE", "en")
    en = _screen(tmp_path, "en")
    assert (en["screen"]["source_lang"], en["screen"]["latin_drops"]) == ("en", 0)
    assert en["screen"]["combined_drops"] == en["screen"]["repetition_drops"] == 1
    # The English caption has an EN of its own missing, but not for that reason.
    assert "latin letters" not in str(reasons(en["sessions"][0]))


def test_the_source_lang_flag_reaches_the_shipped_screen(tmp_path, monkeypatch, capsys):
    """The transcript records captions, not flags, so the language rides argv."""
    monkeypatch.setattr(sr.app, "ASR_LANGUAGE", "ja")
    path = write(tmp_path, "s", [event("10:00:20", "JA", 1, ENGLISH)])
    monkeypatch.setattr(sys, "argv", ["session_report.py", path, "--source-lang", "en", "--json"])
    assert sr.main() == 0
    assert json.loads(capsys.readouterr().out)["screen"]["latin_drops"] == 0


def test_a_drop_increase_names_the_captions_that_bracket_it(tmp_path):
    prev_text = "前" * 45
    next_text = "次" * 45
    screened_text = "ねこ" * 12
    path = write(
        tmp_path,
        "2026-09-04T10-00-00",
        [event("10:00:03", "JA", 41, prev_text), event("10:00:20", "JA", 42, next_text)],
    )

    session = report(
        [path],
        log=[
            log_event("10:00:02", "caption dropped (outside before): いぬいぬ…", "WARNING"),
            log_event("10:00:05", "backlog peak: q=0.25s"),
            log_event(
                "10:00:08",
                f"caption dropped (repetition (unit=ねこ)): {screened_text}…",
                "WARNING",
            ),
            log_event("10:00:10", "backlog peak: q=2.00s drop=5 skip=1"),
            log_event("10:00:12", "backlog peak: q=2.00s seg=1 drop=5 skip=2"),
            log_event("10:00:21", "caption dropped (outside after): とりとり…", "WARNING"),
        ],
        tmp_path=tmp_path,
    )["sessions"][0]

    assert session["drops"] == [
        {
            "at": "2026-09-04 10:00:10",
            "drop": 5,
            "delta": 5,
            "skip_delta": 1,
            "prev": {"n": 41, "at": "2026-09-04 10:00:03", "text": "前" * 40},
            "next": {"n": 42, "at": "2026-09-04 10:00:20", "text": "次" * 40},
            "gap_s": 17.0,
            "screened": [
                {
                    "at": "2026-09-04 10:00:08",
                    "defect": "repetition (unit=ねこ)",
                    "text": screened_text,
                }
            ],
        }
    ]


def test_a_peak_change_without_more_drops_creates_no_entry(tmp_path):
    path = write(tmp_path, "2026-09-04T11-00-00", [event("11:00:10", "JA", 1, CLEAN)])

    session = report(
        [path],
        log=[
            log_event("11:00:01", "backlog peak: q=0.25s"),
            log_event("11:00:02", "backlog peak: q=0.50s seg=1 skip=2"),
        ],
        tmp_path=tmp_path,
    )["sessions"][0]

    assert session["drops"] == []


def test_the_translation_counters_are_never_read_as_audio_drops(tmp_path):
    """`tdrop=`/`tskip=` count TURNS, not captured audio, and share a suffix with the
    audio pair. Dropping `\\b` from `_counter` reads `tdrop=9` as nine dropped blocks
    and the rest of this file stays green, so the boundary needs its own input."""
    path = write(tmp_path, "2026-09-04T11-30-00", [event("11:30:10", "JA", 1, CLEAN)])

    session = report(
        [path],
        log=[
            log_event("11:30:01", "backlog peak: q=0.25s tdrop=9 tskip=4"),
            log_event("11:30:02", "backlog peak: q=0.50s tdrop=11 tskip=6"),
        ],
        tmp_path=tmp_path,
    )["sessions"][0]

    assert session["drops"] == []


def test_the_first_peak_compares_its_drop_count_with_zero(tmp_path):
    next_text = "最初の公開字幕" * 8
    path = write(
        tmp_path,
        "2026-09-04T12-00-00",
        [event("12:00:20", "JA", 1, next_text)],
    )

    session = report(
        [path],
        log=[log_event("12:00:05", "backlog peak: drop=4 skip=2")],
        tmp_path=tmp_path,
    )["sessions"][0]

    assert session["drops"] == [
        {
            "at": "2026-09-04 12:00:05",
            "drop": 4,
            "delta": 4,
            "skip_delta": 2,
            "prev": None,
            "next": {"n": 1, "at": "2026-09-04 12:00:20", "text": next_text[:40]},
            "gap_s": None,
            "screened": [],
        }
    ]


def test_a_drop_after_the_last_caption_has_no_following_caption(tmp_path):
    prev_text = "最後の公開字幕" * 8
    screened_text = "ループ" * 8
    path = write(
        tmp_path,
        "2026-09-04T13-00-00",
        [event("13:00:01", "JA", 7, prev_text)],
    )

    session = report(
        [path],
        log=[
            log_event("13:00:05", "backlog peak: drop=3"),
            log_event(
                "13:00:06",
                f"caption dropped (repetition): {screened_text}…",
                "WARNING",
            ),
            log_event("13:00:07", "worker stopped"),
        ],
        tmp_path=tmp_path,
    )["sessions"][0]

    assert session["drops"] == [
        {
            "at": "2026-09-04 13:00:05",
            "drop": 3,
            "delta": 3,
            "skip_delta": 0,
            "prev": {"n": 7, "at": "2026-09-04 13:00:01", "text": prev_text[:40]},
            "next": None,
            "gap_s": None,
            "screened": [
                {
                    "at": "2026-09-04 13:00:06",
                    "defect": "repetition",
                    "text": screened_text,
                }
            ],
        }
    ]


def test_a_screened_head_keeps_an_ellipsis_the_caption_itself_carried(tmp_path):
    """Production logs `%.24s…`, so a head whose own 24th character is an ellipsis
    lands in the log as `……`. Exactly one of the two belongs to the logger."""
    screened_text = "きりのなか…"
    path = write(
        tmp_path,
        "2026-09-04T13-30-00",
        [event("13:30:01", "JA", 1, CLEAN)],
    )

    session = report(
        [path],
        log=[
            log_event("13:30:05", "backlog peak: drop=2"),
            log_event(
                "13:30:06",
                f"caption dropped (repetition): {screened_text}…",
                "WARNING",
            ),
        ],
        tmp_path=tmp_path,
    )["sessions"][0]

    assert [s["text"] for s in session["drops"][0]["screened"]] == [screened_text]


def test_render_names_drop_increases_in_blocks(tmp_path):
    screened_text = "ねこ" * 12
    path = write(
        tmp_path,
        "2026-09-04T14-00-00",
        [
            event("14:00:01", "JA", 1, "一"),
            event("14:00:20", "JA", 2, "二"),
            event("14:00:40", "JA", 3, "三"),
        ],
    )
    rep = report(
        [path],
        log=[
            log_event("14:00:10", "backlog peak: drop=2 skip=1"),
            log_event(
                "14:00:11",
                f"caption dropped (repetition): {screened_text}…",
                "WARNING",
            ),
            log_event("14:00:30", "backlog peak: q=0.50s drop=5 skip=1"),
        ],
        tmp_path=tmp_path,
    )

    rendered = sr.render(rep)
    assert "   drops: 2 increases, 5 blocks" in rendered
    assert "     14:00:10  +2 blocks (drop=2)  gap 19.0s  n=1 -> n=2  skip +1" in rendered
    assert f"       screened 14:00:11  repetition  {screened_text}" in rendered
    assert "     14:00:30  +3 blocks (drop=5)  gap 20.0s  n=2 -> n=3" in rendered


def test_a_pre_caption_drop_increase_on_an_o_path_session_names_its_screened_caption(tmp_path):
    screened = "ねこ" * 12
    path = write(tmp_path, "run", [event("10:00:20", "JA", 1, CLEAN)])
    session = sr.read_session(path)
    events = read_log(
        tmp_path,
        [
            log_event("10:00:00", f"session: {path}"),
            log_event(
                "10:00:05", f"caption dropped (repetition (unit=ねこ)): {screened}…", "WARNING"
            ),
            log_event("10:00:10", "backlog peak: q=2.00s drop=4 skip=1"),
        ],
    )
    unclaimed = sr.attach_logs([session], events)

    assert sr.attribute_drops(session) == [
        {
            "at": "2026-09-04 10:00:10",
            "drop": 4,
            "delta": 4,
            "skip_delta": 1,
            "prev": None,
            "next": {"n": 1, "at": "2026-09-04 10:00:20", "text": CLEAN},
            "gap_s": None,
            "screened": [
                {
                    "at": "2026-09-04 10:00:05",
                    "defect": "repetition (unit=ねこ)",
                    "text": screened,
                }
            ],
        }
    ]
    assert unclaimed == []


def test_a_second_sessions_startup_events_do_not_land_on_the_first(tmp_path):
    first_path = write(tmp_path, "2026-09-04T10-00-00", [event("10:00:10", "JA", 1, CLEAN)])
    second_path = write(tmp_path, "run", [event("11:00:20", "JA", 1, CLEAN)])
    first, second = sr.read_session(first_path), sr.read_session(second_path)
    first_marker, second_marker = f"session: {first_path}", f"session: {second_path}"
    first_eof = "codex app-server exited; JA-only for the rest of the session"
    second_peak = "backlog peak: q=1.00s drop=2"
    events = read_log(
        tmp_path,
        [
            log_event("10:00:00", first_marker),
            log_event("10:00:15", first_eof, "ERROR"),
            log_event("11:00:00", second_marker),
            log_event("11:00:05", second_peak),
        ],
    )
    unclaimed = sr.attach_logs([second, first], events)

    assert [e.body for e in second.events] == [second_marker, second_peak]
    assert [e.body for e in first.events] == [first_marker, first_eof]
    assert unclaimed == []


def test_an_unsaved_run_between_two_sessions_owns_its_own_events(tmp_path):
    first_path = write(tmp_path, "2026-09-04T10-00-00", [event("10:00:10", "JA", 1, CLEAN)])
    second_path = write(tmp_path, "2026-09-04T12-00-00", [event("12:00:10", "JA", 1, CLEAN)])
    first, second = sr.read_session(first_path), sr.read_session(second_path)
    first_marker, second_marker = f"session: {first_path}", f"session: {second_path}"
    unsaved_marker = "session: not saved (--no-save)"
    unsaved_peak = "backlog peak: q=2.00s drop=7"
    events = read_log(
        tmp_path,
        [
            log_event("10:00:00", first_marker),
            log_event("11:00:00", unsaved_marker),
            log_event("11:00:05", unsaved_peak),
            log_event("12:00:00", second_marker),
        ],
    )
    unclaimed = sr.attach_logs([first, second], events)

    assert [e.body for e in unclaimed] == [unsaved_marker, unsaved_peak]
    assert [e.body for e in first.events] == [first_marker]
    assert [e.body for e in second.events] == [second_marker]


def test_a_log_without_markers_keeps_the_filename_rule(tmp_path):
    # Characterization: the fallback the marker path must leave alone.
    default_path = write(tmp_path, "2026-09-04T09-00-00", [event("09:00:20", "JA", 1, CLEAN)])
    custom_path = write(tmp_path, "run", [event("10:00:20", "JA", 1, CLEAN)])
    default, custom = sr.read_session(default_path), sr.read_session(custom_path)

    assert default.start == datetime(2026, 9, 4, 9, 0, 0)
    assert custom.start == datetime(2026, 9, 4, 10, 0, 20)

    before_default = "before the filename boundary"
    default_at_start = "at the filename boundary"
    default_pre_caption = "default session before its first caption"
    custom_pre_caption = "custom session before its first caption"
    custom_at_start = "at the custom first-caption boundary"
    custom_after_caption = "custom session after its first caption"
    events = read_log(
        tmp_path,
        [
            log_event("08:59:59", before_default),
            log_event("09:00:00", default_at_start),
            log_event("09:00:05", default_pre_caption),
            log_event("10:00:05", custom_pre_caption),
            log_event("10:00:20", custom_at_start),
            log_event("10:00:25", custom_after_caption),
        ],
    )
    unclaimed = sr.attach_logs([custom, default], events)

    assert [e.body for e in unclaimed] == [before_default]
    assert [e.body for e in default.events] == [
        default_at_start,
        default_pre_caption,
        custom_pre_caption,
    ]
    assert [e.body for e in custom.events] == [custom_at_start, custom_after_caption]


def test_the_marker_event_is_inert_in_every_other_answer(tmp_path):
    path = write(
        tmp_path,
        "run",
        [
            event("10:00:10", "JA", 1, CLEAN),
            event("10:00:12", "EN", 1, "Hello."),
            event("10:00:20", "JA", 2, CLEAN),
        ],
    )
    session = sr.read_session(path)
    marker = f"session: {path}"
    unclaimed = sr.attach_logs([session], read_log(tmp_path, [log_event("10:00:00", marker)]))
    summary = sr.build([session], unclaimed)["sessions"][0]

    assert unclaimed == []
    assert [e.body for e in session.events] == [marker]
    assert summary["backlog_peak"] is None
    assert summary["drops"] == []
    assert summary["degrade"] is None
    assert summary["restores"] == []
    assert summary["translated"] == 1
    assert reasons(summary) == {2: sr.SHUTDOWN}


def test_live_stt_logs_the_marker_only_when_a_stream_is_redirected(monkeypatch):
    stream = io.StringIO()
    logger = logging.Logger("marker", level=logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))
    monkeypatch.setattr(app, "logger", logger)

    monkeypatch.setattr(app, "_STDOUT_TTY", True)
    monkeypatch.setattr(app, "_STDERR_TTY", True)
    app.log_session_marker("/tmp/session.txt")
    assert stream.getvalue() == ""

    monkeypatch.setattr(app, "_STDERR_TTY", False)
    app.log_session_marker("/tmp/session.txt")
    monkeypatch.setattr(app, "_STDOUT_TTY", False)
    monkeypatch.setattr(app, "_STDERR_TTY", True)
    app.log_session_marker(None)
    assert stream.getvalue().splitlines() == [
        "session: /tmp/session.txt",
        "session: not saved (--no-save)",
    ]


def test_the_marker_outranks_the_filename_for_a_default_named_session(tmp_path):
    path = write(tmp_path, "2026-09-04T10-00-00", [event("10:10:20", "JA", 1, CLEAN)])
    session = sr.read_session(path)
    before_marker = "backlog peak: q=0.25s drop=1"
    marker = f"session: {path}"
    after_marker = "backlog peak: q=0.50s drop=2"
    events = read_log(
        tmp_path,
        [
            log_event("10:05:00", before_marker),
            log_event("10:10:00", marker),
            log_event("10:10:05", after_marker),
        ],
    )
    unclaimed = sr.attach_logs([session], events)

    assert [e.body for e in unclaimed] == [before_marker]
    assert [e.body for e in session.events] == [marker, after_marker]
