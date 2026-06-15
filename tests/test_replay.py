"""Tests for the deterministic WAV replay path (replay.py).

Two tiers:
- The WAV loader is model-independent and always runs.
- The golden regression replays the cached bench clips through the real
  pipeline and asserts the deterministic surface (segment count + per-segment
  transcript + boundary). It is gated on model weights AND the cached WAVs
  (both gitignored), so it skips cleanly on a fresh clone. Decode latency is
  never asserted — it is CPU-variable.

Regenerate the goldens after an intentional pipeline change:
    uv run python tests/gen_replay_goldens.py
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

import replay
from live_stt import check_models

ROOT = Path(__file__).resolve().parent.parent
GOLDENS = json.loads((ROOT / "tests" / "replay_goldens.json").read_text(encoding="utf-8"))
CACHE = ROOT / "spike" / "backends" / "cache"
# 0.1 s: guards against segmentation-boundary drift without flaking on the
# sub-sample float noise that ONNX ops can introduce across runs/machines.
START_TOL = 1600


def _write_wav(path: Path, data: np.ndarray, sr: int, nchan: int = 1):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(nchan)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.astype(np.int16).tobytes())


# ---- model-independent: the WAV loader ----

def test_load_wav_resamples_to_16k(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, np.zeros(8000), 8000)
    a = replay.load_wav_f32_16k(p)
    assert a.dtype == np.float32
    assert abs(len(a) - 16000) <= 2  # 8 kHz upsampled ~2x to 16 kHz


def test_load_wav_passthrough_16k(tmp_path):
    p = tmp_path / "b.wav"
    _write_wav(p, np.zeros(16000), 16000)
    a = replay.load_wav_f32_16k(p)
    assert a.dtype == np.float32 and len(a) == 16000


def test_load_wav_downmixes_stereo(tmp_path):
    p = tmp_path / "s.wav"
    inter = np.empty(2000)
    inter[0::2] = 1000  # L
    inter[1::2] = -1000  # R -> mono averages to 0
    _write_wav(p, inter, 16000, nchan=2)
    a = replay.load_wav_f32_16k(p)
    assert a.dtype == np.float32 and len(a) == 1000
    assert abs(float(a.mean())) < 1e-3


# ---- models + cached corpus gated: golden regression ----

def _resources_ready(clip_id: str) -> bool:
    return check_models("k2v2") is None and (CACHE / f"{clip_id}.wav").exists()


@pytest.mark.parametrize("clip_id", sorted(GOLDENS))
def test_replay_golden(clip_id):
    if not _resources_ready(clip_id):
        pytest.skip(f"models or cached WAV for {clip_id!r} absent")
    golden = GOLDENS[clip_id]
    report = replay.replay_wav(CACHE / f"{clip_id}.wav", golden["engine"])
    assert report["n_segments"] == golden["n_segments"]
    got, exp = report["segments"], golden["segments"]
    assert len(got) == len(exp)
    for g, e in zip(got, exp, strict=True):
        assert g["text"] == e["text"]
        assert abs(g["start"] - e["start"]) <= START_TOL
