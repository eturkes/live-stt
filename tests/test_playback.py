"""--playback: caption what the computer plays (a Zoom call) instead of the microphone.

Contract: the flag opens PortAudio's exact-name `pipewire` input (PipeWire's ALSA
plugin) with `PIPEWIRE_ALSA` carrying `stream.capture.sink=true` and no target, so
WirePlumber links it to the default output's monitor and moves it when the default
output changes; everything downstream of the callback is the mic path's. A host
without that device fails before any model loads; `--device` contradicts the flag.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import live_stt

PIPEWIRE = {"name": "pipewire", "max_input_channels": 64, "default_samplerate": 44100.0}
HARDWARE = {
    "name": "sof-soundwire: - (hw:0,0)",
    "max_input_channels": 2,
    "default_samplerate": 48000.0,
}
DEFAULT = {"name": "default", "max_input_channels": 64, "default_samplerate": 44100.0}


def _run_session(monkeypatch, tmp_path, devices, seen=None, **flags) -> dict[str, Any]:
    """run_session over a fake PortAudio holding `devices`, one block, no models.

    `seen` is the caller's: a session that exits still leaves what it reached there.
    """
    seen = {} if seen is None else seen
    seen.update(loaded=False, blocks=0)

    def query_devices(device=None, kind=None):
        if device is None and kind is None:
            return devices
        if device is None:
            return DEFAULT
        return devices[device]

    class InputStream:
        def __init__(self, callback, **kwargs):
            seen["stream"] = kwargs
            seen["env"] = live_stt.os.environ.get("PIPEWIRE_ALSA")
            self.callback = callback

        def start(self):
            block = np.full((441, 1), 0.25, dtype=np.float32)
            self.callback(block, len(block), None, None)

        def stop(self):
            pass

        def close(self):
            pass

    def load_recognizer(*_args):
        seen["loaded"] = True
        return object()

    async def worker(_rec, _vad, _window, audio_q, state, *_args, **_kwargs):
        while await audio_q.get() is not None:
            seen["blocks"] += 1
            state.request_stop()

    monkeypatch.delenv("PIPEWIRE_ALSA", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=InputStream, query_devices=query_devices),
    )
    monkeypatch.setattr(live_stt, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr(live_stt, "load_recognizer", load_recognizer)
    monkeypatch.setattr(live_stt, "make_vad", lambda: (None, 512))
    monkeypatch.setattr(live_stt, "worker", worker)
    monkeypatch.setattr(live_stt, "_install_signal_handlers", lambda _state: None)
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
    assert "Mic: default @ 44100 Hz" in capsys.readouterr().out


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
    live_stt.main()
    return seen["args"]


def test_playback_with_device_is_a_parse_error(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _parse(monkeypatch, "--playback", "--device", "3")
    assert exit_info.value.code == 2
    assert "--playback cannot be used with --device" in capsys.readouterr().err


def test_playback_is_off_by_default_and_parses_on(monkeypatch):
    assert _parse(monkeypatch).playback is False
    assert _parse(monkeypatch, "--playback").playback is True
