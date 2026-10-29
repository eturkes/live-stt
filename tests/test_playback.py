"""--playback: caption what the computer plays (a Zoom call) instead of the microphone.

Contract: the flag opens PortAudio's exact-name `pipewire` input (PipeWire's ALSA
plugin) with `PIPEWIRE_ALSA` carrying `stream.capture.sink=true` and no target, so
WirePlumber links it to the default output's monitor and moves it when the default
output changes; everything downstream of the callback is the mic path's. A host
without that device fails before any model loads; `--device` contradicts the flag.
"""

from __future__ import annotations

import asyncio
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import live_stt
from tests.session_globals import keep_session_globals

PIPEWIRE = {"name": "pipewire", "max_input_channels": 64, "default_samplerate": 44100.0}
HARDWARE = {
    "name": "sof-soundwire: - (hw:0,0)",
    "max_input_channels": 2,
    "default_samplerate": 48000.0,
}
DEFAULT = {"name": "default", "max_input_channels": 64, "default_samplerate": 44100.0}


def _run_session(
    monkeypatch, tmp_path, devices, seen=None, *, blocks=1, late=False, caption=None, **flags
) -> dict[str, Any]:
    """run_session over a fake PortAudio holding `devices`, no models, no codex.

    `seen` is the caller's: a session that exits still leaves what it reached there. The
    fake stream resolves and validates its arguments as sounddevice does and feeds `blocks`
    441-frame blocks; the worker stops the session on its first block, or with `late` stops
    it at once and drains only after shutdown has queued the resampler's tail.
    """
    seen = {} if seen is None else seen
    seen.update(loaded=False, blocks=0, translators=[], refused_after_stop=0)
    rec, vad = object(), object()
    seen.update(rec=rec, vad=vad)

    def query_devices(device=None, kind=None):
        if kind not in (None, "input", "output"):  # as sounddevice.query_devices rejects it
            raise ValueError(f"Invalid kind: {kind!r}")
        if device is None and kind is None:
            return devices
        if device is None:
            return DEFAULT
        return devices[device]

    class Translator:
        """A leg that starts, drains and closes, recorded; never a codex process."""

        dropped_translations = degenerate_captions = stale_translations = 0

        def __init__(self, *_args, **_kwargs):
            self.events: list[str] = []
            self._done = asyncio.Event()
            seen["translators"].append(self)

        async def start(self):
            self.events.append("start")
            return True

        async def run(self):
            await self._done.wait()
            self.events.append("drained")

        def submit_sentinel(self):
            self._done.set()

        async def close(self):
            self.events.append("close")

    class InputStream:
        """sounddevice's argument resolution: None takes the device's or library's default."""

        def __init__(self, callback, device=None, samplerate=None, channels=None, dtype=None,
                     blocksize=None, latency=None):  # fmt: skip
            info = query_devices(device, kind="input")
            channels = info["max_input_channels"] if channels is None else channels
            dtype = "float32" if dtype is None else dtype
            if dtype not in ("float32", "int32", "int16", "int8", "uint8"):
                raise ValueError(f"Invalid dtype: {dtype!r}")
            latency = "high" if latency is None else latency
            if latency not in ("low", "high") and not isinstance(latency, (int, float)):
                raise ValueError(f"Invalid latency: {latency!r}")
            seen["stream"] = {
                "device": device,
                "samplerate": samplerate,
                "channels": channels,
                "dtype": dtype,
                "blocksize": blocksize or 0,
                "latency": latency,
            }
            seen["env"] = live_stt.os.environ.get("PIPEWIRE_ALSA")
            self.callback = callback

        def start(self):
            for _ in range(blocks):
                block = np.full((441, seen["stream"]["channels"]), 0.25, dtype=np.float32)
                self.callback(block, len(block), None, None)

        def stop(self):
            seen["stopped"] = True

        def close(self):
            pass

    class CountingQueue(live_stt.AudioQueue):
        def __init__(self):
            super().__init__(seen.get("headroom_s", live_stt.AUDIO_HEADROOM_S))
            seen["queue"] = self
            self.refused = 0

        def put_nowait(self, item):
            try:
                super().put_nowait(item)
            except asyncio.QueueFull:
                if item is not None:
                    self.refused += 1
                    seen["refused_after_stop"] += bool(seen.get("stopped"))
                raise

    class Detector:
        def __init__(self, *_args, **_kwargs):
            seen["detector"] = self

    def load_recognizer(*_args):
        seen["loaded"] = True
        return rec

    async def worker(rec_, vad_, window, audio_q, state, output_file, translator, *_a, **kw):
        seen["worker"] = {"rec": rec_, "vad": vad_, "window": window, **kw}
        seen["worker_translator"] = translator
        seen["state"] = state
        if caption is not None:
            live_stt.emit_line("SRC", 1, caption, output_file)
        if late:
            state.request_stop()
            await asyncio.sleep(0.05)
        while await audio_q.get() is not None:
            seen["blocks"] += 1
            state.request_stop()

    monkeypatch.delenv("PIPEWIRE_ALSA", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=InputStream, query_devices=query_devices),
    )
    monkeypatch.setattr(live_stt, "CodexTranslator", Translator)
    monkeypatch.setattr(live_stt, "LanguageDetector", Detector)
    monkeypatch.setattr(live_stt, "AudioQueue", CountingQueue)
    monkeypatch.setattr(live_stt, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr(live_stt, "load_recognizer", load_recognizer)
    monkeypatch.setattr(live_stt, "make_vad", lambda: (vad, 512))
    monkeypatch.setattr(live_stt, "worker", worker)
    monkeypatch.setattr(
        live_stt, "_install_signal_handlers", lambda state: seen.__setitem__("signal_state", state)
    )
    args = types.SimpleNamespace(
        **{
            "engine": "whisper",
            "asr_device": "NPU",
            "two_way": False,
            "device": None,
            "output": None,
            "no_save": True,
            "no_translate": True,
            "context": "",
            **flags,
        }
    )
    keep_session_globals(monkeypatch)
    asyncio.run(live_stt.run_session(args))
    return seen


def test_playback_opens_the_pipewire_device_on_the_default_output_monitor(
    monkeypatch, tmp_path, capsys
):
    seen = _run_session(monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=True)

    assert seen["stream"]["device"] == 1
    assert seen["stream"]["samplerate"] == 44100
    assert seen["env"] == "{ stream.capture.sink=true }"
    assert seen["blocks"] >= 1  # the callback feeds the same queue the mic does
    assert seen["translators"] == []  # --no-translate builds no leg, whatever the source
    out = capsys.readouterr().out
    assert "Playback: the default output, through pipewire @ 44100 Hz" in out
    assert "Mic:" not in out


def test_the_readme_shows_the_playback_banner_the_session_prints(monkeypatch, tmp_path, capsys):
    """L-021: a literal the README duplicates from the code stays in sync with it."""
    _run_session(monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=True)
    (banner,) = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("Playback:")
    ]
    readme = (Path(live_stt.__file__).resolve().parent / "README.md").read_text(encoding="utf-8")
    assert banner in readme.splitlines()


def test_a_mic_session_leaves_the_capture_property_unset(monkeypatch, tmp_path, capsys):
    seen = _run_session(monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=False)

    assert seen["stream"]["device"] is None
    assert seen["env"] is None
    assert seen["translators"] == []
    assert "Mic: default @ 44100 Hz" in capsys.readouterr().out


@pytest.mark.parametrize(
    "language,two_way,banner",
    [
        ("en", False, "Listening... Speak English. Press Ctrl+C to stop."),
        ("ja", True, "Listening... Speak Japanese or English. Press Ctrl+C to stop."),
    ],
    ids=["english-only", "two-way"],
)
def test_the_mic_prompt_names_the_languages_the_session_accepts(
    monkeypatch, tmp_path, capsys, language, two_way, banner
):
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    seen = _run_session(monkeypatch, tmp_path, [DEFAULT], two_way=two_way)

    assert seen["stream"]["channels"] == 1
    assert [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("Listening")
    ] == [banner]


def test_a_mono_pipewire_input_is_the_playback_device(monkeypatch, tmp_path):
    mono = {**PIPEWIRE, "max_input_channels": 1}
    seen = _run_session(monkeypatch, tmp_path, [HARDWARE, mono], playback=True)
    assert seen["stream"]["device"] == 1


def test_a_mic_session_opens_the_device_it_names_at_that_devices_rate(
    monkeypatch, tmp_path, capsys
):
    """--device 0 (falsy, and at its own 48 kHz rather than the default's 44.1) still wins."""
    seen = _run_session(monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], device=0)

    assert (seen["stream"]["device"], seen["stream"]["samplerate"]) == (0, 48000)
    assert seen["env"] is None
    assert "Mic: #0 sof-soundwire: - (hw:0,0) @ 48000 Hz" in capsys.readouterr().out


@pytest.mark.parametrize("playback", [False, True], ids=["mic", "playback"])
def test_the_translation_leg_runs_whichever_source_is_captured(monkeypatch, tmp_path, playback):
    seen = _run_session(
        monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=playback, no_translate=False
    )

    (leg,) = seen["translators"]
    assert seen["worker_translator"] is leg
    assert leg.events == ["start", "drained", "close"]


@pytest.mark.parametrize(
    "devices",
    [
        [HARDWARE, DEFAULT],
        # Exact name only: the plugin's description and a near name are not the device.
        [{**PIPEWIRE, "name": "PipeWire Sound Server"}, {**PIPEWIRE, "name": "pipewire-x"}],
        [{**PIPEWIRE, "max_input_channels": 0}],
    ],
    ids=["absent", "near-names", "output-only"],
)
def test_playback_without_the_pipewire_input_fails_before_any_model_loads(
    monkeypatch, tmp_path, capsys, devices
):
    seen: dict[str, Any] = {}
    with pytest.raises(SystemExit) as exit_info:
        _run_session(monkeypatch, tmp_path, devices, seen, playback=True)

    assert exit_info.value.code == 1
    assert seen["loaded"] is False
    assert "stream" not in seen
    err = capsys.readouterr().err
    assert "--playback needs PipeWire" in err
    assert "pipewire" in err


def _parse(monkeypatch, *argv):
    seen = {}
    monkeypatch.setattr(sys, "argv", ["live-stt", *argv])

    def supervise(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(live_stt, "_supervise_session", supervise)
    monkeypatch.setattr(live_stt, "_run_cli", supervise)
    keep_session_globals(monkeypatch)
    live_stt.main()
    return seen["args"]


def test_the_help_says_what_playback_captures(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _parse(monkeypatch, "--help")
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    entry = text[text.index("--playback Caption") :].split(" --device DEVICE")[0]
    for meaning in ("this computer plays", "instead of the microphone", "default output"):
        assert meaning in entry
    assert "PipeWire" in entry
    assert "--device" in entry


def test_playback_with_device_is_a_parse_error(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _parse(monkeypatch, "--playback", "--device", "3")
    assert exit_info.value.code == 2
    assert "--playback cannot be used with --device" in capsys.readouterr().err


def test_playback_is_off_by_default_and_parses_on(monkeypatch):
    assert _parse(monkeypatch).playback is False
    assert _parse(monkeypatch, "--playback").playback is True


# --- the mic path's session wiring, which --playback reuses unchanged (mutation-credit locks) ---


@pytest.mark.parametrize("playback", [False, True], ids=["mic", "playback"])
def test_every_audio_operation_runs_under_a_named_deadline(monkeypatch, tmp_path, playback):
    """The supervisor times a phase only while it is truthy: a None phase is no deadline."""
    phases: list[Any] = []
    monkeypatch.setattr(live_stt, "_SESSION_STATUS_FD", -1)
    monkeypatch.setattr(live_stt, "_write_session_status", lambda _fd, phase: phases.append(phase))
    _run_session(monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=playback)

    named = [p for p in phases if p != ""]
    # import, (playback: the device enumeration), the input lookup, open, start, stop, close
    assert len(named) == (7 if playback else 6)
    assert all(isinstance(p, str) and p for p in named)
    assert phases == [x for p in named for x in (p, "")]  # each cleared as it ends


@pytest.mark.parametrize("playback", [False, True], ids=["mic", "playback"])
def test_the_capture_stream_is_mono_float32_at_the_backends_block_size(
    monkeypatch, tmp_path, playback
):
    seen = _run_session(monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=playback)
    stream = {k: seen["stream"][k] for k in ("channels", "dtype", "blocksize", "latency")}
    assert stream == {"channels": 1, "dtype": "float32", "blocksize": 0, "latency": "high"}


def test_the_worker_gets_what_the_session_built(monkeypatch, tmp_path):
    seen = _run_session(
        monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=True, two_way=True,
        context="症例数のカンファレンス",
    )  # fmt: skip
    worker = seen["worker"]
    assert (worker["rec"], worker["vad"], worker["window"]) == (seen["rec"], seen["vad"], 512)
    assert worker["context"].seed == "症例数のカンファレンス"
    assert worker["detector"] is seen["detector"]
    assert seen["worker_translator"] is None  # --no-translate: no leg, not a stand-in
    assert seen["signal_state"] is seen["state"]  # Ctrl+C stops the session the worker drains


def test_an_absent_context_reaches_the_worker_empty(monkeypatch, tmp_path):
    seen = _run_session(monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=True)
    assert seen["worker"]["context"].seed == ""
    assert seen["worker"]["detector"] is None


def test_every_refused_block_counts_on_the_session_state_the_tail_included(monkeypatch, tmp_path):
    """A full queue drops blocks during capture and the resampler tail at shutdown alike."""
    seen: dict[str, Any] = {"headroom_s": 0.05}
    _run_session(
        monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], seen, playback=True, blocks=20,
        late=True,
    )  # fmt: skip
    assert seen["queue"].refused > 1
    assert seen["refused_after_stop"] == 1  # the flushed tail
    assert seen["state"].dropped == seen["queue"].refused


def test_a_saved_playback_session_names_its_files_by_the_session_stamp(monkeypatch, tmp_path):
    _run_session(
        monkeypatch, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=True, caption="症例数",
        no_save=False, save_audio=True,
    )  # fmt: skip
    (transcript,) = tmp_path.glob("*.txt")
    (recording,) = tmp_path.glob("*.wav")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d-\d\d-\d\d", transcript.stem)
    assert recording.stem == transcript.stem
    assert "SRC 1: 症例数" in transcript.read_text(encoding="utf-8")


def test_playback_composes_with_the_existing_flags(monkeypatch, tmp_path):
    out = str(tmp_path / "call.txt")
    for spelling in ("-o", "--output"):
        args = _parse(
            monkeypatch, "--playback", "--source-lang", "ja", spelling, out, "--save-audio",
            "--save-log", "--no-translate",
        )  # fmt: skip
        assert (args.playback, args.source_lang, args.output) == (True, "ja", out)
        assert (args.save_audio, args.save_log, args.no_translate) == (True, True, True)
    args = _parse(monkeypatch, "--playback", "--no-save", "--source-lang", "en")
    assert (args.no_save, args.source_lang, args.output) == (True, "en", None)
    assert _parse(monkeypatch, "--device", "0").device == 0  # an int, the falsy index included
