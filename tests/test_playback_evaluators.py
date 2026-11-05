"""Playback evaluator metrics + resource ownership, without audio or child processes."""

from __future__ import annotations

import contextlib
import hashlib
import io
import shutil
import signal
import subprocess
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


def _fake_e2e(monkeypatch, tmp_path, caption: str, clips: list) -> tuple[routing.Graph, list]:
    """e2e with every process, sleep and graph boundary faked: (graph, live-stt's argv).

    Play k spans [100 + 10k, 101 + 10k] and the transcript carries `caption` once inside each.
    """
    seen: list = []
    plays = 2 * len(clips)  # ROUNDS = 2

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

        played = 0

        def play(self, _):
            self.played += 1
            return 90.0 + 10 * self.played, 91.0 + 10 * self.played

    def popen(argv, **kw):
        seen[:] = argv
        kw["stdout"].write("Listening\nStopped.\n")
        kw["stdout"].flush()
        _stderr_peaks(Path(kw["stderr"].name))
        transcript = Path(argv[argv.index("-o") + 1])
        stamps = (datetime.fromtimestamp(100 + 10 * k).isoformat() for k in range(plays))
        transcript.write_text("".join(f"[{t}] {caption}\n" for t in stamps), encoding="utf-8")
        return SimpleNamespace(poll=lambda: None, wait=lambda **_: 0, returncode=0)

    monkeypatch.setattr(routing.subprocess, "Popen", popen)
    monkeypatch.setattr(routing, "_until", lambda *_: None)
    monkeypatch.setattr(routing, "_kill_own", lambda *_: None)
    monkeypatch.setattr(routing.time, "sleep", lambda _: None)
    monkeypatch.setattr(routing, "ROUNDS", 2)
    monkeypatch.setattr(routing, "_clip_files", lambda *_: clips)
    return cast("routing.Graph", Graph()), seen


def test_routing_e2e_normalizes_both_operands(monkeypatch, tmp_path):
    clips = [("clip", tmp_path / "unused.wav", "Ａ、b こんにちは。")]
    graph, _ = _fake_e2e(monkeypatch, tmp_path, "SRC 1: aBこんにちは", clips)
    result = routing.e2e(graph, "pinned")
    assert all(result["pass"].values())
    row = result["plays"][0]
    assert (row["S"], row["D"], row["I"]) == (0, 0, 0), row
    assert row["text"] == "aBこんにちは"


@pytest.mark.parametrize(
    ("which", "device", "two_way"), [("pinned", "NPU", False), ("two-way", "GPU", True)]
)
def test_routing_e2e_runs_the_requested_mode(monkeypatch, tmp_path, which, device, two_way):
    clips = [("clip", tmp_path / "unused.wav", "こんにちは。")]
    graph, argv = _fake_e2e(monkeypatch, tmp_path, "SRC 1: こんにちは", clips)
    result = routing.e2e(graph, which, device)
    assert argv[argv.index("--asr-device") + 1] == device
    assert ("--two-way" in argv) is two_way
    assert result["device"] == device and all(result["pass"].values())
    assert ("routed" in result["pass"]) is two_way
    assert result["backlog_peak"] == "q=1.75s drop=3"  # the LAST peak line, not the first


def _stderr_peaks(stderr_path):
    stderr_path.write_text(
        "INFO backlog peak: q=0.03s\nINFO caption dropped (x)\nINFO backlog peak: q=1.75s drop=3\n"
    )


def test_two_way_e2e_fails_a_play_decoded_under_the_other_language(monkeypatch, tmp_path):
    """JFK's English published as a Japanese hallucination is the misroute the set exists for."""
    clips = [("jfk", tmp_path / "unused.wav", routing.JFK_REF)]
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    graph, _ = _fake_e2e(
        monkeypatch, tmp_path / "a", "SRC 1 <!>: ご視聴ありがとうございました", clips
    )
    assert routing.e2e(graph, "two-way", "GPU")["pass"]["routed"] is False
    graph, _ = _fake_e2e(monkeypatch, tmp_path / "b", f"SRC 1: {routing.JFK_REF}", clips)
    assert routing.e2e(graph, "two-way", "GPU")["pass"]["routed"] is True


def test_a_pinned_download_refuses_other_bytes(monkeypatch, tmp_path):
    payload = b"fLaC not the pinned clip"

    @contextlib.contextmanager
    def urlopen(_url, timeout):
        yield io.BytesIO(payload)

    monkeypatch.setattr(routing.urllib.request, "urlopen", urlopen)
    dest = tmp_path / "jfk.flac"
    with pytest.raises(RuntimeError, match="sha256"):
        routing._fetch_pinned(routing.JFK_URL, routing.JFK_SHA256, dest)
    assert not dest.exists()
    good = hashlib.sha256(payload).hexdigest()
    assert routing._fetch_pinned(routing.JFK_URL, good, dest).read_bytes() == payload


def test_private_graph_unwinds_when_setup_fails(monkeypatch, tmp_path):
    """A failed __enter__ must reap the stack it already started."""
    directory = tmp_path / "graph"
    directory.mkdir()
    monkeypatch.setattr(routing.tempfile, "mkdtemp", lambda **_: str(directory))
    stack = SimpleNamespace(reaped=False)

    def wait(*, timeout):
        stack.reaped = True

    stack.wait = wait
    monkeypatch.setattr(routing.subprocess, "Popen", lambda *_, **__: stack)
    monkeypatch.setattr(routing.Graph, "isolated", lambda _, argv: argv)

    def fail_ready(*_):
        raise RuntimeError("pipewire-pulse failed to start")

    monkeypatch.setattr(routing, "_until", fail_ready)
    graph = routing.Graph()
    try:
        with pytest.raises(RuntimeError, match="pipewire-pulse failed"):
            with graph:
                pytest.fail("setup unexpectedly succeeded")
        assert stack.reaped, "failed Graph.__enter__ abandoned its started stack"
        assert not directory.exists(), "failed Graph.__enter__ left its private runtime behind"
    finally:
        graph.__exit__()
        shutil.rmtree(directory, ignore_errors=True)


def test_routing_e2e_reaps_the_app_when_playback_fails(monkeypatch, tmp_path):
    """A paplay/readiness error must not leave the live session running."""
    cleanup = []

    class Graph:
        dir = tmp_path
        env = {}

        def isolated(self, argv, pidfile):
            pidfile.write_text("999999999")
            return argv

        def feeding_capture(self):
            return {"sink_a:monitor_FL", "sink_a:monitor_FR"}

        def play(self, _path):
            raise RuntimeError("paplay failed")

    def popen(_argv, **kw):
        kw["stdout"].write("Listening\n")
        kw["stdout"].flush()
        return SimpleNamespace(
            poll=lambda: None,
            wait=lambda **_: cleanup.append("wait"),
            returncode=None,
        )

    monkeypatch.setattr(routing.subprocess, "Popen", popen)
    monkeypatch.setattr(routing, "_until", lambda *_: None)
    monkeypatch.setattr(routing, "_kill_own", lambda *_: cleanup.append("signal"))
    monkeypatch.setattr(routing.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        routing, "_clip_files", lambda *_: [("clip", tmp_path / "unused.wav", "日本語")]
    )
    with pytest.raises(RuntimeError, match="paplay failed"):
        routing.e2e(cast("routing.Graph", Graph()), "pinned")
    assert cleanup == ["signal", "wait"], "playback exception abandoned the live-stt session"


def test_routing_reaps_a_probe_when_its_output_fails(monkeypatch, tmp_path):
    cleanup = []

    class Graph:
        dir = tmp_path
        env = {}

        def isolated(self, argv, pidfile=None):
            if pidfile is not None:
                pidfile.write_text("999999999")
            return argv

    proc = SimpleNamespace(wait=lambda **_: cleanup.append("wait"), stdout=None)
    monkeypatch.setattr(routing.subprocess, "Popen", lambda *_args, **_kwargs: proc)
    monkeypatch.setattr(routing, "_kill_own", lambda *_: cleanup.append("signal"))

    def fail_line(*_):
        raise RuntimeError("probe printed nothing")

    monkeypatch.setattr(routing, "_line", fail_line)
    with pytest.raises(RuntimeError, match="probe printed nothing"):
        routing.routing(cast("routing.Graph", Graph()))
    assert cleanup == ["signal", "wait"]


def test_e2e_escalates_and_reaps_a_child_that_ignores_stop(monkeypatch, tmp_path):
    cleanup = []

    class Graph:
        dir = tmp_path
        env = {}

        def isolated(self, argv, pidfile):
            pidfile.write_text("999999999")
            return argv

        def feeding_capture(self):
            return set()

        def play(self, _path):
            raise RuntimeError("paplay failed")

    def wait(*, timeout):
        cleanup.append(("wait", timeout))
        if sum(event[0] == "wait" for event in cleanup) < 3:
            raise subprocess.TimeoutExpired("fake-live-stt", timeout)
        return 0

    def popen(_argv, **kw):
        kw["stdout"].write("Listening\n")
        kw["stdout"].flush()
        return SimpleNamespace(poll=lambda: None, wait=wait, kill=lambda: cleanup.append(("kill",)))

    monkeypatch.setattr(routing.subprocess, "Popen", popen)
    monkeypatch.setattr(routing, "_until", lambda *_: None)
    monkeypatch.setattr(
        routing, "_kill_own", lambda _pid, sig, _marker: cleanup.append(("signal", sig))
    )
    monkeypatch.setattr(routing.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        routing, "_clip_files", lambda *_: [("clip", tmp_path / "unused.wav", "日本語")]
    )
    with pytest.raises(RuntimeError, match="paplay failed"):
        routing.e2e(cast("routing.Graph", Graph()), "pinned")
    assert cleanup == [
        ("signal", signal.SIGINT),
        ("wait", 90),
        ("signal", signal.SIGTERM),
        ("wait", 10),
        ("signal", signal.SIGKILL),
        ("kill",),
        ("wait", 10),
    ]
