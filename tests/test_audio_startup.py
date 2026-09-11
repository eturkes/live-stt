"""Exercise the CLI against blocked audio bindings, without touching a device."""

from __future__ import annotations

import json
import os
import pty
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import live_stt

ROOT = Path(__file__).resolve().parent.parent
RUNNER = """
import sys
import live_stt
live_stt.AUDIO_OPERATION_TIMEOUT_S = 0.5
live_stt.SESSION_STOP_TIMEOUT_S = 1.0
live_stt.SESSION_KILL_WAIT_S = 0.2
sys.argv = ['live-stt', '--list-devices']
live_stt.main()
"""


def launch(tmp_path, backend, *, runner=RUNNER, uv=False, files=False):
    (tmp_path / "sounddevice.py").write_text(backend)
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}{os.pathsep}{ROOT}")
    env["XDG_RUNTIME_DIR"] = str(tmp_path / "runtime")
    env["UV_PROJECT_ENVIRONMENT"] = sys.prefix
    argv = [sys.executable, "-u", "-c", runner]
    if uv:
        argv = [shutil.which("uv") or str(Path.home() / ".local/bin/uv"), "run", "--no-sync", *argv]
    with (tmp_path / "stdout").open("w") as out, (tmp_path / "stderr").open("w") as err:
        return subprocess.Popen(
            argv,
            cwd=ROOT,
            env=env,
            stdout=out if files else subprocess.PIPE,
            stderr=err if files else subprocess.PIPE,
            text=True,
            start_new_session=True,
        )


def cleanup(proc, tmp_path):
    try:
        owner = json.loads((tmp_path / "runtime/live-stt/session.lock").read_text())
        pid = owner["pid"]
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        if b"from live_stt import _session_child;" in argv:
            os.killpg(pid, signal.SIGKILL)
    except (OSError, ValueError):
        pass
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait(timeout=2)


def finish(proc, tmp_path):
    try:
        return proc.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        cleanup(proc, tmp_path)
        proc.communicate(timeout=2)
        pytest.fail("CLI remained blocked in the audio binding")


@pytest.mark.parametrize("uv", [False, True])
def test_hung_audio_import_releases_cli_and_names_phase(tmp_path, uv):
    marker = tmp_path / "entered"
    proc = launch(
        tmp_path,
        f"from pathlib import Path\nimport time\nPath({str(marker)!r}).touch()\n"
        "while True: time.sleep(1)\n",
        uv=uv,
    )
    out, err = finish(proc, tmp_path)
    assert marker.exists(), "the real audio-import boundary was not reached"
    assert proc.returncode != 0
    assert "initializing audio" in err.lower()
    assert "timed out" in err.lower()
    assert "Stopped." not in out


def test_device_listing_still_prints_the_backend_result(tmp_path):
    proc = launch(tmp_path, "def query_devices(): return 'test microphone'\n")
    out, err = finish(proc, tmp_path)
    assert proc.returncode == 0, err
    assert "test microphone" in out


def test_device_listing_teardown_is_bounded_too(tmp_path):
    proc = launch(
        tmp_path,
        "import atexit, time\n"
        "def close():\n    while True: time.sleep(1)\n"
        "atexit.register(close)\n"
        "def query_devices(): return 'test microphone'\n",
    )
    out, err = finish(proc, tmp_path)
    assert "test microphone" in out
    assert proc.returncode != 0
    assert "closing audio backend" in err.lower()


def test_second_launch_is_refused_before_audio_initialization(tmp_path):
    marker = tmp_path / "entered"
    backend = (
        f"from pathlib import Path\nimport time\n"
        f"with Path({str(marker)!r}).open('a') as f: f.write('entered\\n')\n"
        "while True: time.sleep(1)\n"
    )
    first = launch(tmp_path, backend, runner=RUNNER.replace("TIMEOUT_S = 0.5", "TIMEOUT_S = 20.0"))
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        second = launch(tmp_path, backend)
        _, err = finish(second, tmp_path)
        assert second.returncode != 0
        assert "already running" in err.lower()
        assert marker.read_text() == "entered\n"
    finally:
        first.send_signal(signal.SIGINT)
        finish(first, tmp_path)


def test_interrupt_during_audio_import_returns_without_waiting_for_its_timeout(tmp_path):
    marker = tmp_path / "entered"
    proc = launch(
        tmp_path,
        "import signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(marker)!r}).touch()\nwhile True: time.sleep(1)\n",
        runner=RUNNER.replace("TIMEOUT_S = 0.5", "TIMEOUT_S = 20.0"),
    )
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker.exists()
    proc.send_signal(signal.SIGINT)
    _, err = finish(proc, tmp_path)
    assert proc.returncode != 0
    assert "interrupted" in err.lower()


def test_post_kill_wait_is_bounded_when_the_kernel_never_reaps_the_child(monkeypatch):
    waits = []

    class Unreapable:
        pid = 900123

        def wait(self, timeout=None):
            assert timeout is not None and timeout <= live_stt.SESSION_KILL_WAIT_S
            waits.append(timeout)
            raise subprocess.TimeoutExpired("blocked audio", timeout)

    monkeypatch.setattr(Path, "iterdir", lambda path: iter([Path("/proc/900123")]))
    monkeypatch.setattr(live_stt, "_session_process", lambda pid: (1, "original", "D"))
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    assert live_stt._stop_session(Unreapable()) == [900123]  # type: ignore[arg-type]
    assert sum(waits) <= 2 * live_stt.SESSION_KILL_WAIT_S


def test_surviving_child_keeps_the_lock_after_the_supervisor_exits(tmp_path):
    # An ordinary sleeping child stands in for D state. Suppress kill only in
    # this test's supervisor, then explicitly clean up the surviving child.
    runner = RUNNER.replace(
        "live_stt.main()", "live_stt._stop_session = lambda proc: [proc.pid]\nlive_stt.main()"
    )
    backend = "import time\nwhile True: time.sleep(1)\n"
    proc = launch(tmp_path, backend, runner=runner, files=True)
    try:
        proc.wait(timeout=8)
        assert proc.returncode != 0
        assert "still present after SIGKILL" in (tmp_path / "stderr").read_text()
        second = launch(tmp_path, backend)
        _, err = finish(second, tmp_path)
        assert second.returncode != 0
        assert "already running" in err.lower()
    finally:
        cleanup(proc, tmp_path)


def test_forced_cleanup_stops_a_detached_descendant_too(tmp_path):
    marker = tmp_path / "helper.pid"
    helper = (
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    proc = launch(
        tmp_path,
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {helper!r}], start_new_session=True)\n"
        "while True: time.sleep(1)\n",
    )
    try:
        _, err = finish(proc, tmp_path)
        assert marker.exists(), "detached descendant did not start"
        assert proc.returncode != 0
        assert "timed out" in err.lower()
        assert "still present" not in err.lower()
        info = live_stt._session_process(int(marker.read_text()))
        assert info is None or info[2] == "Z", "detached helper survived the supervisor"
    finally:
        cleanup(proc, tmp_path)
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
@pytest.mark.parametrize("interrupt", ["ctrl_c", "terminal_close"])
def test_supervisor_forwards_terminal_signals_and_preserves_the_drain(tmp_path, interrupt):
    transcript = tmp_path / "transcript.txt"
    (tmp_path / "sitecustomize.py").write_text(
        "import asyncio\nfrom pathlib import Path\nimport live_stt\n"
        "live_stt.check_models = lambda engine: None\n"
        "live_stt.check_device = lambda *args: None\n"
        "async def session(args):\n"
        "    state = live_stt.State()\n"
        "    state.stop_event = asyncio.Event()\n"
        "    live_stt._install_signal_handlers(state)\n"
        f"    f = live_stt.TranscriptFile(Path({str(transcript)!r}))\n"
        "    live_stt.emit_line('JA', 1, 'before stop', f)\n"
        "    await state.stop_event.wait()\n"
        "    live_stt.emit_line('EN', 1, 'after stop', f)\n"
        "    f.close()\n"
        "live_stt.run_session = session\n"
    )
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}{os.pathsep}{ROOT}")
    env["XDG_RUNTIME_DIR"] = str(tmp_path / "runtime")
    argv = [
        sys.executable,
        "-u",
        "-c",
        RUNNER.replace("['live-stt', '--list-devices']", "['live-stt']"),
    ]
    pid, master = pty.fork()
    if pid == 0:  # pragma: no cover - immediately replaced by the real CLI
        os.execve(sys.executable, argv, env)
    status = None
    try:
        deadline = time.monotonic() + 8
        while not transcript.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert transcript.exists(), "session never reached its signal handlers"
        if interrupt == "ctrl_c":
            os.write(master, b"\x03")
        else:
            os.close(master)
            master = -1
        while time.monotonic() < deadline:
            done, raw = os.waitpid(pid, os.WNOHANG)
            if done:
                status = raw
                break
            time.sleep(0.02)
        assert status is not None, "supervisor did not exit after the terminal signal"
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        lines = transcript.read_text().splitlines()
        assert len(lines) == 2
        assert lines[0].endswith("JA 1: before stop")
        assert lines[1].endswith("EN 1: after stop")
    finally:
        if master != -1:
            os.close(master)
        if status is None:
            try:
                owner = json.loads((tmp_path / "runtime/live-stt/session.lock").read_text())
                os.kill(owner["pid"], signal.SIGKILL)
            except (OSError, ValueError):
                pass
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
