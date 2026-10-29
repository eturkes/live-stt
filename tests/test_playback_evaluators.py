"""Playback evaluator metrics + resource ownership, without audio or child processes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from tests import eval_playback_routing as routing
from tests import eval_playback_silence as silence
from tests.evaluator_fakes import hermetic_evaluator  # noqa: F401

pytestmark = pytest.mark.usefixtures("hermetic_evaluator")


def test_silence_cer_normalizes_both_operands(monkeypatch):
    async def rows(_samples, _rec):
        return [{"start": 0, "text": "aBこんにちは"}]

    monkeypatch.setattr(silence.replay, "_run_recognizer", rows)
    result = silence.run_arm(
        "Z",
        object(),
        np.zeros(441, dtype=np.float32),
        [("clip", 0.0, 1.0)],
        {"clip": "Ａ、b こんにちは。"},
    )
    row = result["clips"]["clip"]
    assert result["N"] == 7
    assert (row["S"], row["D"], row["I"]) == (0, 0, 0), result
    assert result["cer"] == 0.0
    assert row["text"] == "aBこんにちは"


def test_routing_e2e_normalizes_both_operands(monkeypatch, tmp_path):
    class Graph:
        dir = tmp_path
        env = {}
        sink = "sink_a"

        def isolated(self, argv, pidfile):
            pidfile.write_text("999999999")
            return argv

        def feeding_capture(self):
            return {f"{self.sink}:monitor_FL", f"{self.sink}:monitor_FR"}

        def default(self, sink):
            self.sink = sink

        def play(self, _):
            return 100.0, 101.0

    def popen(argv, **kw):
        kw["stdout"].write("Listening\nStopped.\n")
        kw["stdout"].flush()
        transcript = Path(argv[argv.index("-o") + 1])
        stamp = datetime.fromtimestamp(100).isoformat()
        transcript.write_text(f"[{stamp}] SRC 1: aBこんにちは\n", encoding="utf-8")
        return SimpleNamespace(poll=lambda: None, wait=lambda **_: 0, returncode=0)

    monkeypatch.setattr(routing.subprocess, "Popen", popen)
    monkeypatch.setattr(routing, "_until", lambda *_: None)
    monkeypatch.setattr(routing, "_kill_own", lambda *_: None)
    monkeypatch.setattr(routing.time, "sleep", lambda _: None)
    monkeypatch.setattr(routing, "ROUNDS", 2)
    monkeypatch.setattr(
        routing, "_clip_files", lambda *_: [("clip", tmp_path / "unused.wav", "Ａ、b こんにちは。")]
    )
    result = routing.e2e(cast("routing.Graph", Graph()), "pinned")
    assert all(result["pass"].values())
    row = result["plays"][0]
    assert (row["S"], row["D"], row["I"]) == (0, 0, 0), row
    assert row["text"] == "aBこんにちは"
