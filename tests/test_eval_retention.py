"""Retention scoring holds the common accelerator lock through model construction + replay."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os

import pytest

from tests import eval_retention as retention
from tests.evaluator_fakes import hermetic_evaluator  # noqa: F401

pytestmark = pytest.mark.usefixtures("hermetic_evaluator")


@pytest.mark.parametrize("paced", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_retention_owns_the_accelerator_until_replay_ends(monkeypatch, tmp_path, paced, fail):
    wav = tmp_path / "probe.wav"
    wav.write_bytes(b"probe stand-in: never decoded")
    manifest = tmp_path / "retention.json"
    manifest.write_text(
        json.dumps(
            {
                "probe": {
                    "audio_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                    "ja_ref": "日本語",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retention, "WAV", wav)
    monkeypatch.setattr(retention, "MANIFEST", manifest)
    monkeypatch.setattr(retention, "NPU_LOCK", tmp_path / "npu.lock", raising=False)
    monkeypatch.setattr(retention, "check_models", lambda _engine: None)
    monkeypatch.setattr(retention, "check_device", lambda _engine, _device: None)
    locked = []

    def flock(fd, op):
        assert op == fcntl.LOCK_EX
        locked.append(fd.fileno())

    monkeypatch.setattr(fcntl, "flock", flock)

    def construct(_engine, _device):
        assert locked, "Whisper compile reached unlocked"
        os.fstat(locked[0])
        return object()

    def replay(*_):
        os.fstat(locked[0])
        if fail:
            raise RuntimeError("replay failed")
        return {
            "segments": [],
            "audio_s": 1.0,
            "n_segments": 0,
            "total_decode_s": 0.0,
            "overall_rtf": 0.0,
        }

    monkeypatch.setattr(retention, "load_recognizer", construct)
    monkeypatch.setattr(retention.replay, "replay_recognizer", replay)
    monkeypatch.setattr(retention, "_paced", replay)
    if fail:
        with pytest.raises(RuntimeError, match="replay failed"):
            retention.score("NPU", 1.75 if paced else None)
    else:
        row = retention.score("NPU", 1.75 if paced else None)
        assert row["D"] == row["N"] == 3
    assert len(locked) == 1
    with pytest.raises(OSError):
        os.fstat(locked[0])
