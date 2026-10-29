"""--playback's routing, re-derived on a private PipeWire graph that reaches no audio hardware.

The claims `asr-pipeline.md`'s `--playback` bullet rests on: PortAudio lists PipeWire's ALSA
plugin as `PLAYBACK_DEVICE`; without `PLAYBACK_STREAM_PROPS` the capture stream links to the
default source; with it, to the default sink's monitor, carrying what plays there; a
default-sink switch moves the links. Every process that could open ALSA runs in a mount
namespace whose /dev/snd is an empty tmpfs -- the SoundWire driver behind this laptop's audio
has hung the kernel -- against a private PipeWire + WirePlumber (hardware monitors off, no
D-Bus) + pipewire-pulse under a temp XDG_RUNTIME_DIR, so the user's own graph stays untouched.

Arms (each a fresh PortAudio process opening the device the way `run_session` does):
  control  no property: the stream links to the default source (a remap-source virtual mic)
  monitor  the property: it links to sink_a's monitor, and a 440 Hz tone at 8000/32768 played
           into the default sink reads back at that peak
  follow   the default sink moves to sink_b mid-stream: the links follow
  --e2e SET  the real `live-stt --playback --no-translate` (whisper: source the accelerator
           prelude first) over SET = pinned | probe (`eval_playback_silence.py`'s two halves),
           ROUNDS rounds, GAP_S of silence after each clip, the default moved to sink_b after
           round 1: per-clip captions and the captions landing past a clip's TAIL_S.

Needs sudo, unshare, setpriv, pipewire, wireplumber, pipewire-pulse, pipewire-alsa,
pactl + paplay (pulseaudio-utils) and pw-link. Exit 1 when an arm misses its expectation.

  uv run python tests/eval_playback_routing.py [--e2e pinned|probe] [--json]
  (--e2e probe decodes CV8 rows: add --with soundfile==0.14.0 --with pyarrow==26.0.0)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import wave
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

import live_stt  # noqa: E402
from cer import align, normalize  # noqa: E402

TONE_HZ, TONE_PEAK = 440, 8000
GAP_S, ROUNDS, TAIL_S = 10.0, 3, 4.0
NO_HARDWARE = """wireplumber.profiles = {
  main = {
    monitor.alsa = disabled
    monitor.alsa-midi = disabled
    monitor.bluez = disabled
    monitor.bluez-midi = disabled
    monitor.v4l2 = disabled
    monitor.libcamera = disabled
  }
}
"""
STACK = """pipewire > "$R/pipewire.log" 2>&1 & echo $! > "$R/pids"
until [ -S "$R/pipewire-0" ]; do sleep 0.1; done
wireplumber > "$R/wireplumber.log" 2>&1 & echo $! >> "$R/pids"
pipewire-pulse > "$R/pipewire-pulse.log" 2>&1 & echo $! >> "$R/pids"
wait
"""
# sudo drops PATH (secure_path) and every LD_* variable even under --preserve-env.
ISOLATE = 'mount -t tmpfs none /dev/snd && exec setpriv --reuid="$SUDO_UID" --regid="$SUDO_GID" \
--init-groups -- "$@"'


def probe(mode: str, hold_s: float) -> None:
    """In the namespace: open the capture stream as `run_session` does, report the peak."""
    import numpy as np

    if mode == "playback":
        os.environ["PIPEWIRE_ALSA"] = live_stt.PLAYBACK_STREAM_PROPS
    import sounddevice as sd  # pyright: ignore[reportMissingImports]

    devices = sd.query_devices()
    device = live_stt.playback_device(devices) if mode == "playback" else None
    info = sd.query_devices(device, kind="input")
    peak = [0.0]

    def callback(indata, _frames, _time, _status):
        peak[0] = max(peak[0], float(np.abs(indata).max(initial=0.0)))

    rate = int(info["default_samplerate"])
    kw = {"channels": 1, "dtype": "float32", "blocksize": 0, "latency": "high"}
    with sd.InputStream(device=device, samplerate=rate, callback=callback, **kw):
        names = [d["name"] for d in devices]
        print(json.dumps({"open": True, "devices": names, "device": device}), flush=True)
        time.sleep(hold_s)
    print(json.dumps({"peak": peak[0]}), flush=True)


class Graph:
    """The private stack: sinks a + b, a virtual mic as the default source."""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="live-stt-pw-"))
        run = self.dir / "run"
        self.env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DBUS_SESSION_BUS_ADDRESS", "PIPEWIRE_REMOTE", "PULSE_SERVER")
            and k != "PIPEWIRE_ALSA"
        }
        self.env |= {
            "XDG_RUNTIME_DIR": str(run),
            "XDG_CONFIG_HOME": str(self.dir / "cfg"),
            "XDG_STATE_HOME": str(self.dir / "state"),
            "R": str(run),
        }
        self.stack: subprocess.Popen | None = None

    def isolated(self, argv: list[str], pidfile: Path | None = None) -> list[str]:
        carried = [f"{k}={v}" for k, v in self.env.items() if k == "PATH" or k.startswith("LD_")]
        inner = ["env", *carried, *argv]
        if pidfile is not None:  # the user-level process records its PID, then becomes argv
            inner = ["sh", "-c", 'echo $$ > "$0"; exec "$@"', str(pidfile), *inner]
        return ["sudo", "--preserve-env", "unshare", "-m", "--", "sh", "-c", ISOLATE, "sh", *inner]

    def run(self, *argv: str) -> str:
        done = subprocess.run(argv, env=self.env, capture_output=True, text=True, timeout=30)  # noqa: S603
        if done.returncode:
            raise RuntimeError(f"{argv}: {done.stderr.strip()}")
        return done.stdout

    def __enter__(self) -> Graph:
        (self.dir / "run").mkdir(mode=0o700)
        conf = self.dir / "cfg" / "wireplumber" / "wireplumber.conf.d"
        conf.mkdir(parents=True)
        (conf / "90-no-hardware.conf").write_text(NO_HARDWARE)
        (self.dir / "state").mkdir()
        self.stack = subprocess.Popen(  # noqa: S603
            self.isolated(["sh", "-c", STACK]),
            env=self.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _until(lambda: (self.dir / "run" / "pulse" / "native").exists(), 30, "pipewire-pulse")
        for name in ("sink_a", "sink_b", "mic_feed"):
            self.run("pactl", "load-module", "module-null-sink", f"sink_name={name}")
        self.run(
            "pactl",
            "load-module",
            "module-remap-source",
            "master=mic_feed.monitor",
            "source_name=mic_x",
        )
        self.run("pactl", "set-default-source", "mic_x")
        self.default("sink_a")
        return self

    def __exit__(self, *_exc) -> None:
        pids = self.dir / "run" / "pids"
        for pid in pids.read_text().split() if pids.exists() else []:
            _kill_own(int(pid), signal.SIGTERM, str(self.dir))
        if self.stack is not None:
            try:
                self.stack.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.stack.kill()
        shutil.rmtree(self.dir, ignore_errors=True)

    def default(self, sink: str) -> None:
        self.run("pactl", "set-default-sink", sink)
        _until(lambda: f"Default Sink: {sink}" in self.run("pactl", "info"), 5, sink)

    def feeding_capture(self) -> set[str]:
        """Nodes whose ports link into a PortAudio (alsa_capture.*) stream."""
        feeders, port = set(), ""
        for line in self.run("pw-link", "-l").splitlines():
            if not line.startswith(" "):
                port = line.strip()
            elif "|->" in line and line.split("|->")[1].strip().startswith("alsa_capture."):
                feeders.add(port)
        return feeders

    def play(self, path: Path) -> tuple[float, float]:
        start = time.time()
        self.run("paplay", str(path))
        return start, time.time()


def _until(ready, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not ready():
        if time.monotonic() > deadline:
            raise RuntimeError(f"timed out waiting for {what}")
        time.sleep(0.1)


def _kill_own(pid: int, sig: int, marker: str) -> None:
    """Signal a process only while its environment still names this run's temp dir."""
    try:
        environ = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return
    if marker.encode() in environ:
        os.kill(pid, sig)


def _line(proc: subprocess.Popen, timeout_s: float) -> dict:
    assert proc.stdout is not None
    if not select.select([proc.stdout], [], [], timeout_s)[0]:
        raise RuntimeError("probe printed nothing")
    return json.loads(proc.stdout.readline())


def _probe(graph: Graph, mode: str, hold_s: float) -> subprocess.Popen:
    argv = [sys.executable, "-I", __file__, "--probe", mode, str(hold_s)]
    return subprocess.Popen(  # noqa: S603
        graph.isolated(argv), env=graph.env, stdout=subprocess.PIPE, text=True
    )


def tone(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(
            b"".join(
                struct.pack("<h", round(TONE_PEAK * math.sin(2 * math.pi * TONE_HZ * i / 48000)))
                for i in range(48000)
            )
        )


def routing(graph: Graph) -> dict:
    control = _probe(graph, "mic", 3.0)
    opened = _line(control, 60)
    control_links = graph.feeding_capture()
    _line(control, 30)
    control.wait(timeout=30)

    monitor = _probe(graph, "playback", 8.0)
    opened_pb = _line(monitor, 60)
    monitor_links = graph.feeding_capture()
    tone(graph.dir / "tone.wav")
    graph.play(graph.dir / "tone.wav")
    graph.default("sink_b")
    want = {"sink_b:monitor_FL", "sink_b:monitor_FR"}
    try:
        _until(lambda: graph.feeding_capture() == want, 5, "the links to follow")
    except RuntimeError:
        pass
    follow_links = graph.feeding_capture()
    peak = _line(monitor, 30)["peak"]
    monitor.wait(timeout=30)
    graph.default("sink_a")
    arms = {
        "devices": opened["devices"],
        "control": sorted(control_links),
        "monitor": sorted(monitor_links),
        "follow": sorted(follow_links),
        "peak": round(peak, 4),
    }
    arms["pass"] = {
        "device": opened_pb["device"] is not None
        and opened_pb["devices"][opened_pb["device"]] == live_stt.PLAYBACK_DEVICE,
        "control": control_links == {"mic_x:capture_FL", "mic_x:capture_FR"},
        "monitor": monitor_links == {"sink_a:monitor_FL", "sink_a:monitor_FR"},
        "peak": abs(peak - TONE_PEAK / 32768) < 0.01,
        "follow": follow_links == want,
    }
    return arms


def _clip_files(graph: Graph, which: str) -> list[tuple[str, Path, str]]:
    refs = json.loads((TESTS / "real_clips.json").read_text(encoding="utf-8"))
    if which == "pinned":
        cache = ROOT / "spike" / "backends" / "cache"
        names = ("short", "med", "long", "kana", "xlong", "multi", "paused")
        return [(f"cv_{n}", cache / f"cv_{n}.wav", refs[f"cv_{n}"]["ja_ref"]) for n in names]
    import eval_playback_silence  # noqa: PLC0415 -- probe half: its pinned CV8 rows
    import soundfile as sf  # pyright: ignore[reportMissingImports]  (uv run --with soundfile)

    out = []
    for name, pcm, rate, ref in eval_playback_silence.clips():
        if name.startswith("cv_"):
            continue
        path = graph.dir / f"{name}.wav"
        sf.write(str(path), pcm, rate, subtype="FLOAT")
        out.append((name, path, ref))
    return out


def e2e(graph: Graph, which: str) -> dict:
    clips = _clip_files(graph, which)
    out, pidfile = graph.dir / "e2e", graph.dir / "live-stt.pid"
    out.mkdir()
    argv = [sys.executable, str(ROOT / "live_stt.py"), "--playback", "--no-translate"]
    argv += ["-o", str(out / "e2e.txt")]
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        app = subprocess.Popen(  # noqa: S603
            graph.isolated(argv, pidfile), env=graph.env, stdout=stdout, stderr=stderr
        )
        _until(lambda: "Listening" in (out / "stdout.txt").read_text() or app.poll() is not None,
               600, "live-stt to listen")  # fmt: skip
        if app.poll() is not None:
            raise RuntimeError(
                f"live-stt exited {app.returncode}: {(out / 'stderr.txt').read_text()}"
            )
        time.sleep(3)
        links = {"start": sorted(graph.feeding_capture())}
        plays = []
        for round_ in range(1, ROUNDS + 1):
            if round_ == 2:
                graph.default("sink_b")
                time.sleep(1)
                links["switched"] = sorted(graph.feeding_capture())
            for name, path, _ in clips:
                plays.append((name, *graph.play(path)))
                time.sleep(GAP_S)
        _kill_own(int(pidfile.read_text()), signal.SIGINT, str(graph.dir))
        code = app.wait(timeout=90)
    graph.default("sink_a")
    captions = []
    for line in (out / "e2e.txt").read_text(encoding="utf-8").splitlines():
        stamp, rest = line[1:].split("] ", 1)
        if rest.startswith("SRC "):
            captions.append((datetime.fromisoformat(stamp).timestamp(), rest.split(": ", 1)[1]))
    owned: list[list[str]] = [[] for _ in plays]
    late = []
    for t, text in captions:
        # Whole-second stamps: a caption up to TAIL_S past its clip's end is that clip's.
        hit = [i for i, (_, a, b) in enumerate(plays) if a - 1.0 <= t <= b + TAIL_S]
        if hit:
            owned[hit[0]].append(text)
        else:
            late.append(text)
    refs = {name: ref for name, _, ref in clips}
    per_play = []
    for (name, _, _), texts in zip(plays, owned, strict=True):
        s, d, i = align(normalize(refs[name]), normalize("".join(texts)))
        per_play.append({"clip": name, "lines": len(texts), "S": s, "D": d, "I": i,
                         "text": "".join(texts)})  # fmt: skip
    stopped = "Stopped." in (out / "stdout.txt").read_text()
    result = {"set": which, "exit": code, "stopped": stopped, "links": links, "plays": per_play,
              "captions": len(captions), "late": late}  # fmt: skip
    result["pass"] = {
        "exit": code == 0 and stopped,
        "links": links["start"] == ["sink_a:monitor_FL", "sink_a:monitor_FR"]
        and links["switched"] == ["sink_b:monitor_FL", "sink_b:monitor_FR"],
    }
    return result


def main() -> None:
    if sys.argv[1:2] == ["--probe"]:
        probe(sys.argv[2], float(sys.argv[3]))
        return
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--e2e", choices=("pinned", "probe"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    tools = ("sudo", "unshare", "setpriv", "pipewire", "wireplumber", "pipewire-pulse")
    missing = [b for b in (*tools, "pactl", "paplay", "pw-link") if not shutil.which(b)]
    if missing:
        raise SystemExit(f"absent: {', '.join(missing)}")
    with Graph() as graph:
        result = {"routing": routing(graph)}
        if args.e2e:
            result["e2e"] = e2e(graph, args.e2e)
    passed = all(all(arm["pass"].values()) for arm in result.values())
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
    else:
        r = result["routing"]
        print(f"devices {r['devices']}")
        for arm in ("control", "monitor", "follow"):
            print(f"{arm:8} {r[arm]}  {'ok' if r['pass'][arm] else 'MISS'}")
        verdict = "ok" if r["pass"]["peak"] else "MISS"
        print(f"peak     {r['peak']} (tone {TONE_PEAK / 32768:.4f})  {verdict}")
        if e := result.get("e2e"):
            print(f"e2e {e['set']}: exit {e['exit']}, Stopped. {e['stopped']}, links {e['links']}")
            for p in e["plays"]:
                print(
                    f"  {p['clip']:9} {p['lines']} lines S{p['S']} D{p['D']} I{p['I']} {p['text']}"
                )
            print(f"  captions {e['captions']}, past a clip's {TAIL_S:g} s tail: {e['late']}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
