"""Locks for `session_report.py`: attribution, log ownership, empty-tree exit.

Hermetic by construction -- every transcript here is written into `tmp_path`, so
the suite never reads the gitignored `transcripts/` the tool exists to explain.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
from datetime import datetime, timedelta
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


def test_read_session_accepts_legacy_and_mixed_grammars(tmp_path):
    path = write(
        tmp_path,
        "mixed",
        [
            event("10:00:01", "JA", 1, CLEAN),
            event("10:00:03", "EN", 1, "One."),
            event("10:00:10", "SRC", 2, CLEAN),
            event("10:00:12", "TGT", 2, "Two."),
        ],
    )
    session = sr.read_session(path)
    assert sorted(session.src) == [1, 2]
    assert sorted(session.tgt) == [1, 2]


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


def test_a_bare_run_reads_saved_logs_and_the_redirect_log(tmp_path, monkeypatch, capsys):
    """`--save-log` writes `transcripts/<stamp>.log`; `2> stt.log` stays the other default."""
    monkeypatch.chdir(tmp_path)
    saved = tmp_path / "transcripts"
    saved.mkdir()
    for name, ts, peak, log in (
        ("2026-09-04T10-00-00", "10", "q=0.25s", saved / "2026-09-04T10-00-00.log"),
        ("2026-09-04T11-00-00", "11", "q=0.50s", tmp_path / "stt.log"),
    ):
        path = write(saved, name, [event(f"{ts}:00:20", "SRC", 1, CLEAN)])
        log.write_text(
            f"{log_event(f'{ts}:00:00', f'session: {path}')}\n"
            f"{log_event(f'{ts}:00:05', f'backlog peak: {peak}')}\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(sys, "argv", ["session_report.py", "--json"])
    assert sr.main() == 0
    sessions = json.loads(capsys.readouterr().out)["sessions"]
    assert [s["backlog_peak"] for s in sessions] == [
        "backlog peak: q=0.25s",
        "backlog peak: q=0.50s",
    ]


def test_declined_captions_name_their_own_defect(tmp_path):
    path = write(
        tmp_path,
        "2026-09-04T10-00-00",
        [
            event("10:00:01", "SRC", 1, CLEAN),
            event("10:00:03", "TGT", 1, "Hello."),
            event("10:00:10", "SRC", 2, LOOP),
            event("10:00:20", "SRC", 3, ENGLISH),
            event("10:00:30", "SRC", 4, CLEAN),
            event("10:00:32", "TGT", 4, "Hello again."),
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
    lines = [event("11:00:01", "SRC", 1, CLEAN), event("11:00:03", "TGT", 1, "One.")]
    for n in range(2, 8):
        lines.append(event(f"11:0{n}:00", "SRC", n, CLEAN))
    rep = report([write(tmp_path, "2026-09-04T11-00-00", lines)])
    s = rep["sessions"][0]
    assert s["degrade"]["source"] == "inferred"
    assert s["degrade"]["last_tgt"] == 1
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
            event("12:00:01", "SRC", 1, CLEAN),
            event("12:00:03", "TGT", 1, "One."),
            event("12:00:10", "SRC", 2, CLEAN),
        ],
    )
    rep = report(
        [path],
        log=[
            "[2026-09-04 12:00:12,100] ERROR codex app-server exited;"
            " source-only for the rest of the session"
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
    """One trailing caption is the drain losing its last TGT, not the leg dying."""
    path = write(
        tmp_path,
        "2026-09-04T13-00-00",
        [
            event("13:00:01", "SRC", 1, CLEAN),
            event("13:00:03", "TGT", 1, "One."),
            event("13:00:10", "SRC", 2, CLEAN),
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
            event("14:00:01", "SRC", 1, CLEAN),
            event("14:00:03", "TGT", 1, "One."),
            "[2026-09-04T14:00:05+09:00] -- translation disabled: codex app-server exited",
            event("14:00:10", "SRC", 2, CLEAN),
            "[2026-09-04T14:00:20+09:00] -- translation restored:"
            " codex app-server respawned (attempt 1)",
            event("14:00:30", "SRC", 3, CLEAN),
            event("14:00:32", "TGT", 3, "Three."),
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
        [event("15:00:01", "SRC", 1, unit * 8), event("15:00:03", "TGT", 1, "One.")],
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


def test_caption_cadence_is_labelled_as_a_proxy_not_an_observed_rotation(tmp_path, monkeypatch):
    """The no-skip/no-reset caption-number proxy begins at 101, not 100."""
    monkeypatch.setattr(sr, "TWO_WAY", False)
    lines = []
    for n, at in ((100, "10:00"), (101, "10:01")):
        lines.append(event(f"{at}:00", "SRC", n, CLEAN))
        lines.append(event(f"{at}:07", "TGT", n, "Hello"))
    rep = report([write(tmp_path, "s", lines)])
    slow = rep["sessions"][0]["slow_turns"]
    assert all("at_rotation" not in row for row in slow)
    assert {row["n"]: row["caption_cadence_proxy"] for row in slow} == {100: False, 101: True}
    rendered = sr.render(rep)
    assert "caption-number proxy: matches=1 (period 100)" in rendered
    assert "not observed rotations" in rendered
    assert "Skipped turns and recovery resets" in rendered


def _screen(tmp_path: Path, name: str) -> dict:
    lines = [
        event("10:00:01", "SRC", 1, CLEAN),
        event("10:00:03", "TGT", 1, "Hello."),
        event("10:00:10", "SRC", 2, LOOP),
        event("10:00:20", "SRC", 3, ENGLISH),
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
    # The English caption has a TGT of its own missing, but not for that reason.
    assert "latin letters" not in str(reasons(en["sessions"][0]))


def test_the_source_lang_flag_reaches_the_shipped_screen(tmp_path, monkeypatch, capsys):
    """The transcript records captions, not flags, so the language rides argv."""
    monkeypatch.setattr(sr.app, "ASR_LANGUAGE", "ja")
    path = write(tmp_path, "s", [event("10:00:20", "SRC", 1, ENGLISH)])
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
        [event("10:00:03", "SRC", 41, prev_text), event("10:00:20", "SRC", 42, next_text)],
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
    path = write(tmp_path, "2026-09-04T11-00-00", [event("11:00:10", "SRC", 1, CLEAN)])

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
    path = write(tmp_path, "2026-09-04T11-30-00", [event("11:30:10", "SRC", 1, CLEAN)])

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
        [event("12:00:20", "SRC", 1, next_text)],
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
        [event("13:00:01", "SRC", 7, prev_text)],
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
        [event("13:30:01", "SRC", 1, CLEAN)],
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
            event("14:00:01", "SRC", 1, "一"),
            event("14:00:20", "SRC", 2, "二"),
            event("14:00:40", "SRC", 3, "三"),
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
    # The log body already carries its label; the renderer must not add a second.
    assert "   backlog peak: q=0.50s drop=5 skip=1\n" in rendered
    assert "backlog peak: backlog peak:" not in rendered


def test_a_pre_caption_drop_increase_on_an_o_path_session_names_its_screened_caption(tmp_path):
    screened = "ねこ" * 12
    path = write(tmp_path, "run", [event("10:00:20", "SRC", 1, CLEAN)])
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
    first_path = write(tmp_path, "2026-09-04T10-00-00", [event("10:00:10", "SRC", 1, CLEAN)])
    second_path = write(tmp_path, "run", [event("11:00:20", "SRC", 1, CLEAN)])
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
    first_path = write(tmp_path, "2026-09-04T10-00-00", [event("10:00:10", "SRC", 1, CLEAN)])
    second_path = write(tmp_path, "2026-09-04T12-00-00", [event("12:00:10", "SRC", 1, CLEAN)])
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
    default_path = write(tmp_path, "2026-09-04T09-00-00", [event("09:00:20", "SRC", 1, CLEAN)])
    custom_path = write(tmp_path, "run", [event("10:00:20", "SRC", 1, CLEAN)])
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
            event("10:00:10", "SRC", 1, CLEAN),
            event("10:00:12", "TGT", 1, "Hello."),
            event("10:00:20", "SRC", 2, CLEAN),
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
    path = write(tmp_path, "2026-09-04T10-00-00", [event("10:10:20", "SRC", 1, CLEAN)])
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


def test_a_two_way_sessions_english_lines_are_never_screen_declines(tmp_path, monkeypatch, capsys):
    """Queue: Session report reads two-way transcripts (reviewer-6's authored mixed transcript).

    The transcript records no per-line decode language. An English line a two-way session
    published was decoded under <|en|>, where the latin rule never applies, so the report
    re-screened it under ja and read it as a `declined` caption.
    """
    monkeypatch.setattr(sr.app, "ASR_LANGUAGE", "ja")
    # main() rebinds it; teardown restores it.
    monkeypatch.setattr(sr, "TWO_WAY", False, raising=False)
    path = write(
        tmp_path,
        "2026-09-04T10-00-00",
        [
            event("10:00:01", "SRC", 1, CLEAN),
            event("10:00:03", "TGT", 1, "Hello, good morning."),
            event("10:00:10", "SRC", 2, ENGLISH),
            event("10:00:20", "SRC", 3, LOOP),
            "[2026-09-04T10:00:30+09:00] SRC 4 <!>: こんにちは",
            event("10:00:32", "TGT", 4, "Hello."),
            event("10:00:40", "SRC", 5, "Ha ha ha, fine."),
            event("10:00:42", "TGT", 5, "はははは、いいよ。"),
        ],
    )
    monkeypatch.setattr(sys, "argv", ["session_report.py", path, "--json"])
    assert sr.main() == 0
    one_way = json.loads(capsys.readouterr().out)
    assert one_way["screen"]["latin_drops"] == 2, "the authored transcript lost its latin lines"

    monkeypatch.setattr(sys, "argv", ["session_report.py", path, "--two-way", "--json"])
    assert sr.main() == 0
    two_way = json.loads(capsys.readouterr().out)
    screen = two_way["screen"]
    assert (screen["source_lang"], screen["latin_drops"]) == ("two-way", 0)
    assert screen["combined_drops"] == screen["repetition_drops"] == 1
    missing = {r["n"]: r for r in two_way["sessions"][0]["missing"]}
    assert missing[2]["why"] != sr.DECLINED and not missing[2]["screened_now"]
    assert missing[3]["why"] == sr.DECLINED
    assert two_way["totals"]["held"] == 1
    assert "Ha ha ha, fine." in [r["text"] for r in screen["longest_surviving"]]


def test_two_way_and_source_lang_are_one_choice(tmp_path, monkeypatch, capsys):
    """live-stt rejects the pair at parse time; the report reads a session the same way."""
    path = write(tmp_path, "s", [event("10:00:20", "SRC", 1, ENGLISH)])
    monkeypatch.setattr(
        sys, "argv", ["session_report.py", path, "--two-way", "--source-lang", "en"]
    )
    with pytest.raises(SystemExit) as exit_:
        sr.main()
    assert exit_.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_a_session_that_never_translated_reads_as_untranslated(tmp_path):
    """--no-translate (or a leg unavailable at startup) leaves no TGT and, by design, no marker:
    nothing on record says a leg ran, so no caption reads as its transient failure."""
    path = write(
        tmp_path,
        "2026-09-04T15-00-00",
        [event("15:00:01", "SRC", 1, CLEAN), event("15:00:05", "SRC", 2, CLEAN),
         event("15:00:09", "SRC", 3, CLEAN)],
    )  # fmt: skip
    s = report([path])["sessions"][0]
    assert set(reasons(s).values()) == {"untranslated"}
    assert s["degrade"] is None


@pytest.mark.parametrize("with_log", [False, True])
def test_stale_transcript_note_is_sufficient_without_the_optional_log(tmp_path, with_log):
    path = write(
        tmp_path,
        "2026-09-04T10-00-00",
        [
            event("10:00:01", "SRC", 1, CLEAN),
            "[2026-09-04T10:00:17+09:00] -- translation skipped (stale): 1",
            event("10:00:20", "SRC", 2, CLEAN),
            event("10:00:22", "TGT", 2, "The next caption."),
        ],
    )
    log = (
        [
            log_event(
                "10:00:17",
                "caption 1 not translated: queued 16 s, over TRANSLATE_MAX_STALENESS_S",
                "WARNING",
            )
        ]
        if with_log
        else None
    )
    session = report([path], log, tmp_path)["sessions"][0]
    assert reasons(session) == {1: sr.STALE}
    assert len(session["missing"]) == 1


def test_a_session_without_tgt_but_with_a_degrade_marker_stays_disabled(tmp_path):
    path = write(
        tmp_path,
        "2026-09-04T16-00-00",
        [
            event("16:00:01", "SRC", 1, CLEAN),
            "[2026-09-04T16:00:02+09:00] -- translation disabled: codex app-server exited",
            event("16:00:05", "SRC", 2, CLEAN),
        ],
    )
    assert set(reasons(report([path])["sessions"][0]).values()) == {sr.DISABLED}


def _transcript(tmp_path: Path, *lines: str) -> sr.Session:
    return sr.read_session(write(tmp_path, "2026-09-04T10-00-00", list(lines)))


def test_missing_caption_inside_a_recovered_outage_is_disabled(tmp_path):
    # An unsuccessful recovery and the following cooldown can discard a caption;
    # a later caption restores the leg. No caption inside that outage spent a strike.
    session = _transcript(
        tmp_path,
        "[2026-09-04T10:00:01+09:00] SRC 1: 最初の字幕",
        "[2026-09-04T10:00:03+09:00] TGT 1: The first caption.",
        "[2026-09-04T10:00:05+09:00] -- translation disabled: codex app-server exited",
        "[2026-09-04T10:00:10+09:00] SRC 2: 復旧を待つ字幕",
        "[2026-09-04T10:00:30+09:00] SRC 3: 次の字幕",
        "[2026-09-04T10:00:35+09:00] -- translation restored: "
        "codex app-server respawned (attempt 2)",
        "[2026-09-04T10:00:37+09:00] TGT 3: The next caption.",
    )

    (missing,) = sr.explain_missing(session)

    assert missing["why"] == sr.DISABLED, missing


def test_a_recovered_outage_does_not_own_a_later_shutdown_loss(tmp_path):
    session = _transcript(
        tmp_path,
        "[2026-09-04T10:00:01+09:00] SRC 1: 最初の字幕",
        "[2026-09-04T10:00:03+09:00] TGT 1: The first caption.",
        "[2026-09-04T10:00:05+09:00] -- translation disabled: codex app-server exited",
        "[2026-09-04T10:00:10+09:00] SRC 2: 復旧する字幕",
        "[2026-09-04T10:00:15+09:00] -- translation restored: "
        "codex app-server respawned (attempt 1)",
        "[2026-09-04T10:00:17+09:00] TGT 2: The recovered caption.",
        "[2026-09-04T10:00:20+09:00] SRC 3: 最後の字幕",
    )

    (missing,) = sr.explain_missing(session)

    assert missing["why"] == sr.SHUTDOWN, missing


def test_one_restore_written_to_both_channels_is_one_report_event(tmp_path):
    reason = "codex app-server respawned (attempt 1)"
    session = _transcript(
        tmp_path,
        "[2026-09-04T10:00:01+09:00] SRC 1: 字幕",
        f"[2026-09-04T10:00:10+09:00] -- translation restored: {reason}",
        "[2026-09-04T10:00:12+09:00] TGT 1: Caption.",
    )
    log = tmp_path / "run.log"
    log.write_text(
        f"[2026-09-04 10:00:10,000] WARNING translation restored: {reason}\n",
        encoding="utf-8",
    )
    session.events = sr.read_log(str(log))

    restores = sr.build([session], [])["sessions"][0]["restores"]

    assert len(restores) == 1, restores


def test_explicit_stale_tail_does_not_infer_a_three_strike_degrade(tmp_path):
    session = _transcript(
        tmp_path,
        "[2026-09-04T10:00:01+09:00] SRC 1: 最初の字幕",
        "[2026-09-04T10:00:03+09:00] TGT 1: The first caption.",
        "[2026-09-04T10:00:04+09:00] SRC 2: 次の字幕",
        "[2026-09-04T10:00:05+09:00] SRC 3: その次の字幕",
        "[2026-09-04T10:00:06+09:00] SRC 4: 最後の字幕",
        "[2026-09-04T10:00:22+09:00] -- translation skipped (stale): 2",
        "[2026-09-04T10:00:22+09:00] -- translation skipped (stale): 3",
        "[2026-09-04T10:00:22+09:00] -- translation skipped (stale): 4",
    )
    for n in (2, 3, 4):
        session.events.append(
            sr.LogEvent(
                datetime(2026, 9, 4, 10, 0, 22),
                "WARNING",
                f"caption {n} not translated: queued 16 s, over TRANSLATE_MAX_STALENESS_S",
            )
        )

    assert sr.find_degrade(session) is None


def test_each_outage_owns_its_strikes_and_ends_at_its_restore(tmp_path):
    lines = [event("10:00:01", "SRC", 1, CLEAN), event("10:00:03", "TGT", 1, "One.")]
    for n in range(2, 6):
        lines.append(event(f"10:00:{n * 5:02}", "SRC", n, CLEAN))
    lines += [
        "[2026-09-04T10:00:21+09:00] -- translation disabled: "
        "3 consecutive failures (TimeoutError)",
        "[2026-09-04T10:00:31+09:00] -- translation restored: codex app-server probed (attempt 1)",
        event("10:00:32", "SRC", 6, CLEAN),
        event("10:00:34", "TGT", 6, "Six."),
        "[2026-09-04T10:00:35+09:00] -- translation disabled: codex app-server exited",
        event("10:00:36", "SRC", 7, CLEAN),
        "[2026-09-04T10:00:40+09:00] -- translation restored: "
        "codex app-server respawned (attempt 2)",
        event("10:00:41", "SRC", 8, CLEAN),
        event("10:00:43", "TGT", 8, "Eight."),
        event("10:00:44", "SRC", 9, CLEAN),
    ]
    session = report([write(tmp_path, "run", lines)])["sessions"][0]
    assert reasons(session) == {
        2: sr.STRIKE,
        3: sr.STRIKE,
        4: sr.STRIKE,
        5: sr.DISABLED,
        7: sr.DISABLED,
        9: sr.SHUTDOWN,
    }
    assert [(d["last_tgt"], d["source_only_after"]) for d in session["degrades"]] == [
        (1, 4),
        (6, 1),
    ]
    assert session["degrade"] == session["degrades"][0]


@pytest.mark.parametrize("skip_kind", ["stale", "declined", "screen"])
def test_known_declines_never_spend_inferred_strikes(tmp_path, skip_kind):
    lines = [event("10:00:01", "SRC", 1, CLEAN), event("10:00:03", "TGT", 1, "One.")]
    for n in range(2, 5):
        lines.append(event(f"10:00:{n * 5:02}", "SRC", n, LOOP if skip_kind == "screen" else CLEAN))
        if skip_kind == "stale":
            lines.append(
                f"[2026-09-04T10:00:{n * 5 + 1:02}+09:00] -- translation skipped (stale): {n}"
            )
    log = (
        [
            log_event("10:00:25", f"caption {n} not translated: refused", "WARNING")
            for n in range(2, 5)
        ]
        if skip_kind == "declined"
        else None
    )
    session = report([write(tmp_path, "run", lines)], log, tmp_path)["sessions"][0]
    assert session["degrade"] is None
    assert sr.STRIKE not in reasons(session).values()


def test_log_marker_uses_the_written_path_not_just_the_basename(tmp_path):
    first_path = tmp_path / "morning" / "session.txt"
    second_path = tmp_path / "afternoon" / "session.txt"
    for path in (first_path, second_path):
        path.parent.mkdir()
    first_path.write_text("[2026-09-04T10:00:20+09:00] SRC 1: 午前の字幕\n", encoding="utf-8")
    second_path.write_text("[2026-09-04T11:00:20+09:00] SRC 1: 午後の字幕\n", encoding="utf-8")
    first, second = sr.read_session(str(first_path)), sr.read_session(str(second_path))
    first_marker = sr.LogEvent(datetime(2026, 9, 4, 10), "INFO", f"session: {first_path}")
    first_peak = sr.LogEvent(datetime(2026, 9, 4, 10, 0, 10), "INFO", "backlog peak: drop=5")
    second_marker = sr.LogEvent(datetime(2026, 9, 4, 11), "INFO", f"session: {second_path}")

    sr.attach_logs([first, second], [first_marker, first_peak, second_marker])

    assert first_peak in first.events, (first.events, second.events)
    assert first_peak not in second.events, (first.events, second.events)


def test_an_ambiguous_marker_basename_leaves_its_window_unclaimed(tmp_path):
    sessions = []
    for hour, folder in ((10, "morning"), (11, "afternoon")):
        root = tmp_path / folder
        root.mkdir()
        sessions.append(
            sr.read_session(write(root, "session", [event(f"{hour}:00:20", "SRC", 1, CLEAN)]))
        )
    marker = sr.LogEvent(datetime(2026, 9, 4, 10), "INFO", "session: /moved/session.txt")
    peak = sr.LogEvent(datetime(2026, 9, 4, 10, 0, 10), "INFO", "backlog peak: drop=5")
    assert sr.attach_logs(sessions, [marker, peak]) == [marker, peak]
    assert all(not s.events for s in sessions)


def test_append_mode_output_does_not_pair_different_runs(tmp_path, monkeypatch, capsys):
    # Exercise the real writer twice with the sequence restart a new run uses.
    moments = iter(
        datetime.fromisoformat(at)
        for at in (
            "2026-09-04T10:00:01+09:00",
            "2026-09-04T10:00:03+09:00",
            "2026-09-04T11:00:01+09:00",
        )
    )

    class Clock:
        @staticmethod
        def now():
            return next(moments)

    monkeypatch.setattr(app, "datetime", Clock)
    monkeypatch.setattr(app, "write_stdout", lambda _text: None)
    monkeypatch.setattr(app, "ASR_LANGUAGE", "ja")
    monkeypatch.setattr(sr, "TWO_WAY", False)
    path = tmp_path / "custom.txt"
    first = app.TranscriptFile(path)
    app.emit_line("SRC", 1, "最初の実行", first)
    app.emit_line("TGT", 1, "The first run.", first)
    first.close()
    second = app.TranscriptFile(path)
    app.emit_line("SRC", 1, "次の実行", second)
    second.close()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["session_report.py", str(path), "--json"])

    assert sr.main() == 0
    report = json.loads(capsys.readouterr().out)

    assert report["totals"]["captions"] == 2, report["totals"]
    assert report["totals"]["missing"] == 1, report["totals"]
    assert report["lag_s"]["max"] >= 0, report["lag_s"]


@pytest.mark.parametrize("tags", [("SRC", "TGT"), ("JA", "EN")])
def test_appended_runs_own_their_log_windows_and_startup_notes(tmp_path, tags):
    src, tgt = tags
    path = write(
        tmp_path,
        "custom",
        [
            event("10:00:10", src, 1, CLEAN),
            event("10:00:12", tgt, 1, "One."),
            "[2026-09-04T11:00:02+09:00] -- translation disabled: codex app-server exited",
            event("11:00:10", src, 1, CLEAN),
        ],
    )
    sessions = sr.read_sessions(path)
    first_peak = log_event("10:00:05", "backlog peak: drop=2")
    second_peak = log_event("11:00:05", "backlog peak: drop=3")
    events = read_log(
        tmp_path,
        [
            log_event("10:00:00", f"session: {path}"),
            first_peak,
            log_event("11:00:00", f"session: {path}"),
            second_peak,
        ],
    )
    assert sr.attach_logs(sessions, events) == []
    summary = sr.build(sessions, [])
    assert summary["totals"]["captions"] == 2
    assert [s["session"] for s in summary["sessions"]] == ["custom.txt#1", "custom.txt#2"]
    assert [s["drops"][-1]["drop"] for s in summary["sessions"]] == [2, 3]
    assert sessions[0].notes == []
    assert len(sessions[1].notes) == 1
    assert reasons(summary["sessions"][1]) == {1: sr.DISABLED}
    assert sr.lags(sessions[0]) == [(1, 2.0)]
    assert sr.lags(sessions[1]) == []
    with pytest.raises(ValueError, match="multiple runs"):
        sr.read_session(path)


def test_same_named_transcripts_keep_distinct_caption_identities_in_the_sweep(tmp_path):
    paths = []
    for folder in ("morning", "afternoon"):
        root = tmp_path / folder
        root.mkdir()
        paths.append(write(root, "session", [event("10:00:01", "SRC", 1, LOOP)]))
    result = report(paths)
    assert result["screen"]["sweep"][-1]["caught"] == 2
    assert {s["session"] for s in result["sessions"]} == set(paths)


def test_disable_log_names_current_state_and_both_log_grammars_remain_readable(tmp_path, caplog):
    translator = app.CodexTranslator()
    translator.enabled = True
    translator._disable("codex app-server exited")
    body = next(r.getMessage() for r in caplog.records if "translation disabled:" in r.getMessage())
    assert "for the rest of the session" not in body
    assert not translator.enabled
    for line in [
        body,
        "codex app-server exited; source-only for the rest of the session",
        "codex app-server exited; JA-only for the rest of the session",
        "translation disabled: codex app-server exited; source-only for the rest of the session",
        "translation disabled: codex app-server exited; JA-only for the rest of the session",
    ]:
        session = _transcript(
            tmp_path,
            event("10:00:01", "SRC", 1, CLEAN),
            event("10:00:03", "TGT", 1, "One."),
            event("10:00:10", "SRC", 2, CLEAN),
        )
        session.events = [sr.LogEvent(datetime(2026, 9, 4, 10, 0, 12), "ERROR", line)]
        degrade = sr.find_degrade(session)
        assert degrade is not None
        assert degrade["reason"] == "codex app-server exited"
        assert {r["why"] for r in sr.explain_missing(session)} == {sr.DISABLED}


def test_a_silent_run_in_an_appended_path_does_not_claim_the_next_speaking_run(tmp_path):
    path = write(
        tmp_path,
        "custom",
        [
            event("10:00:10", "SRC", 1, CLEAN),
            event("12:00:10", "SRC", 1, CLEAN),
        ],
    )
    sessions = sr.read_sessions(path)
    events = read_log(
        tmp_path,
        [
            log_event("10:00:00", f"session: {path}"),
            log_event("11:00:00", f"session: {path}"),
            log_event("11:00:05", "backlog peak: drop=7"),
            log_event("12:00:00", f"session: {path}"),
            log_event("12:00:05", "backlog peak: drop=3"),
        ],
    )
    unclaimed = sr.attach_logs(sessions, events)
    assert sessions[1].marker_start == datetime(2026, 9, 4, 12)
    assert unclaimed == events[1:3]
    assert [d["drop"] for d in sr.attribute_drops(sessions[1])] == [3]


def test_lifecycle_twins_can_straddle_a_whole_second_without_merging_attempts(tmp_path):
    session = _transcript(
        tmp_path,
        event("10:00:01", "SRC", 1, CLEAN),
        "[2026-09-04T10:00:11+09:00] -- translation restored: probed (attempt 1)",
        "[2026-09-04T10:00:13+09:00] -- translation restored: probed (attempt 2)",
    )
    session.events = [
        sr.LogEvent(
            datetime(2026, 9, 4, 10, 0, 10), "WARNING", "translation restored: probed (attempt 1)"
        ),
        sr.LogEvent(
            datetime(2026, 9, 4, 10, 0, 12), "WARNING", "translation restored: probed (attempt 2)"
        ),
    ]
    restores = sr.find_restores(session)
    assert [(r["reason"], r["source"]) for r in restores] == [
        ("probed (attempt 1)", "marker"),
        ("probed (attempt 2)", "marker"),
    ]


def test_two_way_rotation_tag_does_not_claim_a_global_caption_cadence(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "ASR_LANGUAGE", "ja")
    monkeypatch.setattr(sr, "TWO_WAY", True)
    translator = app.CodexTranslator(sources=("ja", "en"))
    translator.enabled = True
    for leg in translator._legs.values():
        leg.thread_id = f"initial-{leg.source}"
    actual_rotations: list[int] = []
    current_caption = 0

    async def new_thread():
        actual_rotations.append(current_caption)
        return f"rotated-{current_caption}"

    async def turn(_text):
        return "Translated."

    monkeypatch.setattr(translator, "_new_thread", new_thread)
    monkeypatch.setattr(translator, "_turn", turn)
    lines: list[str] = []
    start = datetime.fromisoformat("2026-09-04T10:00:00+09:00")

    async def scenario():
        nonlocal current_caption
        for n in range(1, app.TRANSLATE_ROTATE_TURNS + 2):
            current_caption = n
            language = "ja" if n % 2 else "en"
            text = "日本語の字幕" if language == "ja" else "English speech."
            translated = await translator._translate(text, language)
            at = start + timedelta(seconds=10 * n)
            target_at = at + timedelta(seconds=7)
            lines.extend(
                [
                    f"[{at.isoformat()}] SRC {n}: {text}",
                    f"[{target_at.isoformat()}] TGT {n}: {translated}",
                ]
            )

    asyncio.run(scenario())
    assert actual_rotations == [], "each direction is below its 101st turn"
    session = _transcript(tmp_path, *lines)
    rep = sr.build([session], [])
    slow = rep["sessions"][0]["slow_turns"]
    reported_rotations = [row["n"] for row in slow if row.get("at_rotation")]

    assert reported_rotations == actual_rotations, slow[-1]
    assert all(row["caption_cadence_proxy"] is None for row in slow)
    assert all("at_rotation" not in row for row in slow)
    assert "caption-number proxy: unavailable (--two-way)" in sr.render(rep)
