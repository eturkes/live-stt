"""Tests for the deterministic WAV replay path (replay.py).

Two tiers:
- The WAV loader is model-independent and always runs.
- The golden regression replays the cached bench clips through the real
  pipeline and asserts the deterministic surface (segment count + per-segment
  transcript + boundary). It is gated on model weights, the cached WAVs (both
  gitignored) AND the accelerator, so it skips cleanly on a fresh clone and
  under `gate.py`, which sources no accel farm. Decode latency is never
  asserted — it is CPU-variable.

The whisper row is an `ASR_DEVICE` artifact rather than a portable one, so it
records the device it was produced on. That record is compared BEFORE any
readiness probe: a golden that disagrees with the shipped constant is a stale
committed artifact, which must fail on every box rather than skip on the ones
without the hardware to notice.

Regenerate the goldens after an intentional pipeline change:
    uv run python tests/gen_replay_goldens.py
"""

from __future__ import annotations

import asyncio
import json
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import replay
from live_stt import ASR_DEVICE, WHISPER_ENGINES, check_device, check_models

ROOT = Path(__file__).resolve().parent.parent
GOLDENS = json.loads((ROOT / "tests" / "replay_goldens.json").read_text(encoding="utf-8"))
# Flatten the engine-keyed goldens to (engine, clip_id) cases for parametrization.
GOLDEN_CASES = sorted((engine, clip_id) for engine, clips in GOLDENS.items() for clip_id in clips)
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


def test_run_feeds_long_audio_in_live_sized_blocks():
    async def fake_worker(_rec, _vad, _window, audio_q, _state, *_args):
        blocks = []
        while True:
            block = audio_q.get_nowait()
            if block is None:
                break
            blocks.append(block)
        seen.extend(blocks)

    seen = []
    samples = np.arange(2 * replay.SAMPLE_RATE + 7, dtype=np.float32)
    with (
        mock.patch.object(replay, "_recognizer", return_value=object()),
        mock.patch.object(replay, "make_vad", return_value=(object(), 512)),
        mock.patch.object(replay, "worker", fake_worker),
    ):
        assert asyncio.run(replay._run(samples, "k2v2")) == []

    assert [len(block) for block in seen] == [replay.SAMPLE_RATE, replay.SAMPLE_RATE, 7]
    assert np.array_equal(np.concatenate(seen), samples)


def test_run_surfaces_worker_shutdown_as_evaluator_failure():
    async def failed_worker(_rec, _vad, _window, _audio_q, state, *_args):
        state.request_stop()

    with (
        mock.patch.object(replay, "_recognizer", return_value=object()),
        mock.patch.object(replay, "make_vad", return_value=(object(), 512)),
        mock.patch.object(replay, "worker", failed_worker),
        pytest.raises(RuntimeError, match="worker failed during replay"),
    ):
        asyncio.run(replay._run(np.ones(1, dtype=np.float32), "k2v2"))


def test_the_cli_preflights_the_accelerator_before_loading_an_engine(monkeypatch, tmp_path, capsys):
    """Without this, `--engine whisper` on a farm-less box fails ten frames deep
    in openvino_genai on a missing NPU compiler loader. The engine stub proves
    the exit happens before any of that, so the test needs no accelerator."""

    def unreachable(*_args, **_kw):
        raise AssertionError("the engine must not load once the preflight has failed")

    wav = tmp_path / "clip.wav"
    _write_wav(wav, np.zeros(16000), 16000)
    monkeypatch.setattr(replay, "check_models", lambda engine: None)
    monkeypatch.setattr(replay, "check_device", lambda engine: f"{engine}: device unavailable")
    monkeypatch.setattr(replay, "replay_wav", unreachable)
    monkeypatch.setattr("sys.argv", ["replay.py", str(wav), "--engine", "whisper"])

    with pytest.raises(SystemExit) as exited:
        replay.main()

    assert exited.value.code == 1
    assert "Error: whisper: device unavailable" in capsys.readouterr().err


def test_the_cli_reports_a_missing_wav_without_probing_the_accelerator(
    monkeypatch, tmp_path, capsys
):
    """Ordering: the argument check is free, check_device imports OpenVINO."""

    def unreachable(*_args, **_kw):
        raise AssertionError("a bad path must not cost an OpenVINO import")

    monkeypatch.setattr(replay, "check_device", unreachable)
    monkeypatch.setattr(
        "sys.argv", ["replay.py", str(tmp_path / "absent.wav"), "--engine", "whisper"]
    )

    with pytest.raises(SystemExit) as exited:
        replay.main()

    assert exited.value.code == 1
    assert "no such WAV" in capsys.readouterr().err


# ---- models + cached corpus gated: golden regression ----


def _stale_device(engine: str, golden: dict) -> str | None:
    """Why this golden no longer describes what the engine would run, else None.

    Hardware-free on purpose: a missing, null or foreign `device` means the
    committed row was produced against a different target, which is wrong
    everywhere and must not hide behind a skip on a box without the accelerator.
    """
    if engine not in WHISPER_ENGINES:
        return None
    stored = golden.get("device")
    if stored != ASR_DEVICE:
        return (
            f"golden device {stored!r} != live_stt.ASR_DEVICE {ASR_DEVICE!r}: "
            "regenerate with tests/gen_replay_goldens.py"
        )
    return None


def _not_ready(engine: str, clip_id: str) -> str | None:
    """Why this case cannot run here, else None."""
    if err := check_models(engine):
        return err.splitlines()[0]
    if err := check_device(engine):
        return err
    if not (CACHE / f"{clip_id}.wav").exists():
        return f"cached WAV for {clip_id!r} absent"
    return None


@pytest.mark.parametrize("engine,clip_id", GOLDEN_CASES)
def test_replay_golden(engine, clip_id):
    golden = GOLDENS[engine][clip_id]
    stale = _stale_device(engine, golden)
    assert stale is None, stale
    if reason := _not_ready(engine, clip_id):
        pytest.skip(reason)
    report = replay.replay_wav(CACHE / f"{clip_id}.wav", engine)
    assert report["n_segments"] == golden["n_segments"]
    got, exp = report["segments"], golden["segments"]
    assert len(got) == len(exp)
    for g, e in zip(got, exp, strict=True):
        assert g["text"] == e["text"]
        assert abs(g["start"] - e["start"]) <= START_TOL
        # `n` is the VAC utterance length: the one committed end-boundary signal,
        # and the whole surface a trim/commit regression would move. The sherpa
        # rows keep their original start-only assertion.
        if engine in WHISPER_ENGINES:
            assert abs(g["n"] - e["n"]) <= START_TOL
