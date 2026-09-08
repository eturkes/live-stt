"""Locks for `session_report.py`: attribution, log ownership, empty-tree exit.

Hermetic by construction -- every transcript here is written into `tmp_path`, so
the suite never reads the gitignored `transcripts/` the tool exists to explain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def report(paths: list[str], log: list[str] | None = None, tmp_path: Path | None = None) -> dict:
    sessions = sorted((sr.read_session(p) for p in paths), key=lambda s: s.name)
    events = []
    if log is not None and tmp_path is not None:
        lp = tmp_path / "run.log"
        lp.write_text("".join(f"{ln}\n" for ln in log), encoding="utf-8")
        events = sr.read_log(str(lp))
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
