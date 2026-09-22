"""Locks on the committed lag trace (contract: .scratch/row3/CONTRACT-unit1.md)."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import session_report as sr

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "tests" / "lag_sessions.json"
BUILDER = ROOT / "tests" / "build_lag_trace.py"
SESSION_IDS = ("2026-09-15T16-48-03", "2026-09-18T12-00-26")
TRANSCRIPTS = {session_id: ROOT / "transcripts" / f"{session_id}.txt" for session_id in SESSION_IDS}
TOP_KEYS = {"schema", "generator", "note", "sessions"}
SESSION_KEYS = {
    "id",
    "source_file",
    "source_sha256",
    "grammar",
    "start",
    "captions",
    "notes",
    "pairs",
}
NOTE_KEYS = {"at_s", "text"}
PAIR_KEYS = {"n", "src_s", "tgt_s", "src_chars", "tgt_chars", "held"}


def _load_trace() -> dict[str, Any]:
    assert TRACE.is_file(), "tests/lag_sessions.json is missing"
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    assert isinstance(trace, dict), "lag trace root must be an object"
    return trace


def test_schema_keys_are_exactly_the_contract() -> None:
    trace = _load_trace()
    assert set(trace) == TOP_KEYS, f"top-level keys: {sorted(trace)}"
    assert trace["schema"] == 1, f"schema: {trace['schema']!r}"
    assert trace["generator"] == "tests/build_lag_trace.py"

    sessions = trace["sessions"]
    assert isinstance(sessions, list), "sessions must be a list"
    assert len(sessions) == 2, f"sessions: {len(sessions)}"
    assert {session["id"] for session in sessions} == set(SESSION_IDS)
    for session in sessions:
        assert isinstance(session, dict), "session must be an object"
        assert set(session) == SESSION_KEYS, (
            f"session {session.get('id')!r} keys: {sorted(session)}"
        )
        assert session["captions"] == len(session["pairs"]), (
            f"session {session['id']}: captions={session['captions']}, "
            f"pairs={len(session['pairs'])}"
        )
        for note in session["notes"]:
            assert isinstance(note, dict), f"session {session['id']}: note must be an object"
            assert set(note) == NOTE_KEYS, f"session {session['id']} note keys: {sorted(note)}"
        for pair in session["pairs"]:
            assert isinstance(pair, dict), f"session {session['id']}: pair must be an object"
            assert set(pair) == PAIR_KEYS, f"session {session['id']} pair keys: {sorted(pair)}"


def test_the_trace_is_text_free() -> None:
    assert TRACE.is_file(), "tests/lag_sessions.json is missing"
    raw = TRACE.read_bytes()
    assert raw.isascii(), "tests/lag_sessions.json contains non-ASCII bytes"
    trace = json.loads(raw)
    for session in trace["sessions"]:
        for pair in session["pairs"]:
            extra = set(pair) - PAIR_KEYS
            assert not extra, (
                f"session {session['id']} pair {pair.get('n')}: extra keys {sorted(extra)}"
            )


def test_pairs_are_well_formed_and_monotonic() -> None:
    trace = _load_trace()
    for session in trace["sessions"]:
        pairs = session["pairs"]
        numbers = [pair["n"] for pair in pairs]
        assert numbers == list(range(1, len(pairs) + 1)), (
            f"session {session['id']}: n sequence is not 1..{len(pairs)}"
        )
        src_times = [pair["src_s"] for pair in pairs]
        assert all(a <= b for a, b in zip(src_times, src_times[1:], strict=False)), (
            f"session {session['id']}: src_s decreases"
        )
        for pair in pairs:
            label = f"session {session['id']} pair {pair['n']}"
            assert type(pair["n"]) is int, f"{label}: n is not an integer"
            assert type(pair["src_s"]) is int, f"{label}: src_s is not an integer"
            assert type(pair["src_chars"]) is int, f"{label}: src_chars is not an integer"
            assert type(pair["held"]) is bool, f"{label}: held is not a boolean"
            tgt_s = pair["tgt_s"]
            tgt_chars = pair["tgt_chars"]
            assert tgt_s is None or type(tgt_s) is int, f"{label}: tgt_s is not null or integer"
            assert tgt_s is None or tgt_s >= pair["src_s"], f"{label}: TGT precedes SRC"
            assert (tgt_chars is None) == (tgt_s is None), (
                f"{label}: tgt_chars nullness disagrees with tgt_s"
            )
            assert tgt_chars is None or type(tgt_chars) is int, (
                f"{label}: tgt_chars is not null or integer"
            )


def test_the_recorded_percentiles_re_derive() -> None:
    trace = _load_trace()
    expected = {
        "2026-09-15T16-48-03": (143, 2, 4, 11),
        "2026-09-18T12-00-26": (696, 2, 5, 42),
    }
    sessions = {session["id"]: session for session in trace["sessions"]}
    assert set(sessions) == set(expected), f"session ids: {sorted(sessions)}"
    for session_id, (pairs, p50, p90, maximum) in expected.items():
        lags = sorted(
            pair["tgt_s"] - pair["src_s"]
            for pair in sessions[session_id]["pairs"]
            if pair["tgt_s"] is not None
        )
        assert len(lags) == pairs, f"session {session_id}: pairs={len(lags)}, expected={pairs}"
        assert statistics.median(lags) == p50, (
            f"session {session_id}: p50={statistics.median(lags)}, expected={p50}"
        )
        actual_p90 = lags[int(len(lags) * 0.9)]
        assert actual_p90 == p90, f"session {session_id}: p90={actual_p90}, expected={p90}"
        assert max(lags) == maximum, f"session {session_id}: max={max(lags)}, expected={maximum}"


def test_the_trace_agrees_with_session_report() -> None:
    trace = _load_trace()
    missing = [path for path in TRANSCRIPTS.values() if not path.is_file()]
    if missing:
        pytest.skip("absent: " + ", ".join(str(path.relative_to(ROOT)) for path in missing))

    sessions = {session["id"]: session for session in trace["sessions"]}
    for session_id, path in TRANSCRIPTS.items():
        measured = sr.lags(sr.read_session(str(path)))
        recorded = [
            (pair["n"], pair["tgt_s"] - pair["src_s"])
            for pair in sessions[session_id]["pairs"]
            if pair["tgt_s"] is not None
        ]
        assert measured == recorded, f"session {session_id}: trace lags differ from session_report"


def test_the_trace_regenerates_byte_identically() -> None:
    assert BUILDER.is_file(), "tests/build_lag_trace.py is missing"
    assert TRACE.is_file(), "tests/lag_sessions.json is missing"
    missing = [path for path in TRANSCRIPTS.values() if not path.is_file()]
    if missing:
        pytest.skip("absent: " + ", ".join(str(path.relative_to(ROOT)) for path in missing))

    checked = subprocess.run(
        [sys.executable, str(BUILDER), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, (
        f"build_lag_trace.py --check exited {checked.returncode}\n"
        f"stdout:\n{checked.stdout}\nstderr:\n{checked.stderr}"
    )
