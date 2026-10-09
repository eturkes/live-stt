"""VAD-copy leveling: quiet speech opens; every downstream consumer keeps raw PCM."""

import asyncio
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import sherpa_onnx

import live_stt

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "spike/backends/cache"
WINDOW = 512
RATE = 16000
TARGET = 10 ** (-25 / 20)
FLOOR = 10 ** (-70 / 20)
MIN_WINDOWS = round(2 * RATE / WINDOW)
HISTORY_WINDOWS = round(10 * RATE / WINDOW)
PINNED_CLIPS = (
    "greet",
    "short",
    "medium",
    "long",
    "paused",
    "cv_short",
    "cv_med",
    "cv_long",
    "cv_kana",
    "cv_xlong",
    "cv_multi",
    "cv_paused",
    "stress_med",
    "stress_long",
    "retention_probe",
    *(f"gongitsune_{i:02d}" for i in range(1, 7)),
)


def _clip(name: str) -> np.ndarray:
    path = CACHE / f"{name}.wav"
    if not path.is_file():
        pytest.skip(f"absent: pinned WAV {path}")
    with wave.open(str(path), "rb") as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (RATE, 1, 2)
        return (
            np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
        )


def _quiet_clip() -> np.ndarray:
    samples = _clip("cv_xlong")
    complete = samples[: len(samples) // WINDOW * WINDOW].reshape(-1, WINDOW)
    rms = np.sqrt(np.mean(complete.astype(np.float64) ** 2, axis=1))
    target = 10 ** (-48 / 20)
    scale = target / np.percentile(rms[rms > FLOOR], 90)
    # Attenuation moves near-silence below the floor; normalize the admitted set again.
    scale = target / np.percentile(rms[rms * scale > FLOOR], 90)
    scaled_rms = rms * scale
    assert np.percentile(scaled_rms[scaled_rms > FLOOR], 90) == pytest.approx(target)
    return samples * np.float32(scale)


def _window(level: float, size: int = WINDOW) -> np.ndarray:
    return np.full(size, level, dtype=np.float32)


@dataclass
class _VadInput:
    samples: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    calls: int = 0
    initialized: list = field(default_factory=list)


def _fake_vad(monkeypatch, window: int = WINDOW):
    # Import here leaves the make_vad behavioural regression runnable on pre-fix HEAD.
    from live_stt import LeveledVad

    sink = _VadInput()

    def initialize(_self, cfg, buffer_size_in_seconds):
        sink.initialized.append((cfg, buffer_size_in_seconds))

    def accept(_self, samples):
        sink.samples = samples
        sink.calls += 1

    monkeypatch.setattr(sherpa_onnx.VoiceActivityDetector, "__init__", initialize)
    monkeypatch.setattr(sherpa_onnx.VoiceActivityDetector, "accept_waveform", accept)
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.window_size = window
    # pybind's __call__ requires native initialization; explicit allocation tests
    # the Python constructor against a spy without loading weights or native state.
    vad = LeveledVad.__new__(LeveledVad)
    LeveledVad.__init__(vad, cfg)
    assert isinstance(vad, sherpa_onnx.VoiceActivityDetector)
    assert sink.initialized == [(cfg, 60)]
    return vad, sink


def _prime(vad, amplitude: float = 0.01, count: int = MIN_WINDOWS):
    for _ in range(count):
        vad.accept_waveform(_window(amplitude))


def test_quiet_pinned_speech_opens_a_segment():
    if not (ROOT / "models/silero_vad.onnx").is_file():
        pytest.skip("absent: models/silero_vad.onnx")
    samples = _quiet_clip()
    unscaled, window = live_stt.make_vad()
    for start in range(0, len(samples) - window + 1, window):
        sherpa_onnx.VoiceActivityDetector.accept_waveform(unscaled, samples[start : start + window])
    unscaled.flush()
    assert unscaled.empty(), "the quiet fixture must still expose the unscaled VAD's miss"
    vad, window = live_stt.make_vad()
    for start in range(0, len(samples) - window + 1, window):
        vad.accept_waveform(samples[start : start + window])
    vad.flush()
    assert not vad.empty(), "a pinned speech clip at -48 dBFS p90 must open the VAD"


@pytest.mark.parametrize("name", PINNED_CLIPS)
def test_pinned_clip_keeps_unit_gain_on_every_window(monkeypatch, name):
    samples = _clip(name)
    vad, sink = _fake_vad(monkeypatch)
    for start in range(0, len(samples) - WINDOW + 1, WINDOW):
        block = samples[start : start + WINDOW]
        vad.accept_waveform(block)
        assert vad.gain == 1.0, (name, start / RATE, vad.gain)
        assert sink.samples is block
    assert sink.calls > 0


@pytest.mark.parametrize("readonly", [False, True])
def test_scaled_copy_never_mutates_the_callers_array(monkeypatch, readonly):
    vad, sink = _fake_vad(monkeypatch)
    _prime(vad)
    owner = np.linspace(-0.04, 0.04, 2 * WINDOW, dtype=np.float32)
    samples = owner[::2]
    before = owner.copy()
    samples.flags.writeable = not readonly
    vad.accept_waveform(samples)
    assert vad.gain > 1
    np.testing.assert_array_equal(owner, before)
    assert not np.shares_memory(sink.samples, samples)
    np.testing.assert_allclose(sink.samples, np.clip(samples * vad.gain, -1, 1))
    assert sink.samples.dtype == np.float32


def test_worker_vac_decodes_the_raw_ring_not_the_vad_copy(monkeypatch):
    vad, sink = _fake_vad(monkeypatch)
    monkeypatch.setattr(sherpa_onnx.VoiceActivityDetector, "is_speech_detected", lambda _: True)
    monkeypatch.setattr(sherpa_onnx.VoiceActivityDetector, "empty", lambda _: True)
    monkeypatch.setattr(sherpa_onnx.VoiceActivityDetector, "flush", lambda _: None)
    monkeypatch.setattr(live_stt, "emit_line", lambda *_: None)
    samples = np.linspace(-0.003, 0.003, 160 * WINDOW, dtype=np.float32)
    original = samples.copy()
    decoded: list[np.ndarray] = []

    class Recognizer:
        def decode_segments(self, audio, language=None):
            decoded.append(audio.copy())
            return "静かな声です。", []

    async def scenario():
        queue = asyncio.Queue()
        for start in range(0, len(samples), WINDOW):
            queue.put_nowait(samples[start : start + WINDOW])
        queue.put_nowait(None)
        state = live_stt.State()
        await live_stt.worker(Recognizer(), vad, WINDOW, queue, state, None)
        assert not state.stopping

    asyncio.run(scenario())
    assert sink.calls == len(samples) // WINDOW
    assert vad.gain > 1
    assert decoded, "worker must actually reach the VAC recogniser"
    for audio in decoded:
        np.testing.assert_array_equal(audio, original[: len(audio)])
    np.testing.assert_array_equal(decoded[-1], original)
    np.testing.assert_array_equal(samples, original)


def test_floor_is_strict_and_silence_does_not_advance_history(monkeypatch):
    vad, _ = _fake_vad(monkeypatch)
    at_floor = np.zeros(WINDOW, dtype=np.float64)
    at_floor[0] = FLOOR * math.sqrt(WINDOW)
    assert float(np.sqrt(np.mean(at_floor**2))) == FLOOR
    excluded = (_window(0), _window(FLOOR / 2), at_floor)
    for index in range(HISTORY_WINDOWS + MIN_WINDOWS):
        vad.accept_waveform(excluded[index % len(excluded)])
        assert vad.gain == 1.0
    _prime(vad, count=MIN_WINDOWS - 1)
    assert vad.gain == 1.0
    vad.accept_waveform(_window(0.01))
    assert vad.gain == pytest.approx(TARGET / 0.01)
    held = vad.gain
    for index in range(HISTORY_WINDOWS + 1):
        vad.accept_waveform(excluded[index % len(excluded)])
        assert vad.gain == held
    _prime(vad, amplitude=0.2, count=10)
    assert vad.gain == 1.0, "excluded windows must not evict the admitted history"


def test_minimum_history_uses_rounded_window_count(monkeypatch):
    vad, sink = _fake_vad(monkeypatch)
    assert vad.gain == 1.0
    for _ in range(MIN_WINDOWS - 1):
        samples = _window(0.01)
        vad.accept_waveform(samples)
        assert vad.gain == 1.0
        assert sink.samples is samples
    vad.accept_waveform(_window(0.01))
    assert vad.gain == pytest.approx(TARGET / 0.01)


@pytest.mark.parametrize("level, expected", [(1.0, 1.0), (0.001, 16.0), (0.01, TARGET / 0.01)])
def test_gain_is_bounded_and_never_attenuates(monkeypatch, level, expected):
    vad, _ = _fake_vad(monkeypatch)
    _prime(vad, amplitude=level)
    assert vad.gain == pytest.approx(expected)


def test_scaled_native_input_is_float32_and_clipped_at_both_limits(monkeypatch):
    vad, sink = _fake_vad(monkeypatch)
    _prime(vad, amplitude=0.001)
    samples = np.full(WINDOW, 0.001, dtype=np.float64)
    samples[:2] = [-0.2, 0.2]
    before = samples.copy()
    vad.accept_waveform(samples)
    assert vad.gain == 16
    assert sink.samples.dtype == np.float32
    assert sink.samples[:2].tolist() == [-1, 1]
    np.testing.assert_array_equal(sink.samples, np.clip(samples * 16, -1, 1).astype(np.float32))
    np.testing.assert_array_equal(samples, before)


def test_p90_uses_linear_interpolation_not_peak_or_mean(monkeypatch):
    vad, _ = _fake_vad(monkeypatch)
    _prime(vad, amplitude=0.01, count=55)
    _prime(vad, amplitude=0.03, count=7)
    # 62 samples: p90 lies 0.9 of the way from sorted index 54 to 55.
    assert vad.gain == pytest.approx(TARGET / (0.01 * 0.1 + 0.03 * 0.9))


def test_ten_seconds_holds_only_the_last_admitted_windows(monkeypatch):
    vad, _ = _fake_vad(monkeypatch)
    _prime(vad, amplitude=0.1, count=HISTORY_WINDOWS)
    assert vad.gain == 1
    for _ in range(280):
        vad.accept_waveform(_window(0.005))
        vad.accept_waveform(_window(0))
    assert vad.gain == 1
    vad.accept_waveform(_window(0.005))
    assert vad.gain == pytest.approx(TARGET / 0.005)
    _prime(vad, amplitude=0.005, count=HISTORY_WINDOWS)
    assert vad.gain == pytest.approx(TARGET / 0.005)


@dataclass
class _ReferenceLevel:
    history: list[float] = field(default_factory=list)
    gain: float = 1.0

    def accept(self, samples: np.ndarray) -> float:
        rms = math.sqrt(math.fsum(float(value) ** 2 for value in samples) / len(samples))
        if rms > FLOOR:
            self.history.append(rms)
            self.history = self.history[-HISTORY_WINDOWS:]
        if len(self.history) >= MIN_WINDOWS:
            ordered = sorted(self.history)
            location = (len(ordered) - 1) * 0.9
            lower = math.floor(location)
            fraction = location - lower
            level = ordered[lower] * (1 - fraction) + ordered[math.ceil(location)] * fraction
            self.gain = min(16.0, max(1.0, TARGET / level))
        return self.gain


@pytest.mark.parametrize("seed", [0, 73, 1009])
def test_generated_stream_matches_independent_window_oracle(monkeypatch, seed):
    vad, sink = _fake_vad(monkeypatch)
    oracle = _ReferenceLevel()
    rng = np.random.default_rng(seed)
    amplitudes = np.concatenate(
        (
            rng.uniform(0, FLOOR / 2, MIN_WINDOWS + 3),
            np.full(MIN_WINDOWS + 2, 0.001),
            np.full(HISTORY_WINDOWS + 2, 0.3),
            np.zeros(HISTORY_WINDOWS + 2),
            np.full(HISTORY_WINDOWS + 2, 0.004),
            10 ** rng.uniform(-5, -0.2, 500),
        )
    )
    gains = []
    for index, amplitude in enumerate(amplitudes):
        shape = rng.uniform(-1, 1, WINDOW)
        samples = (shape * amplitude / np.sqrt(np.mean(shape**2))).astype(np.float32)
        before = samples.copy()
        expected = oracle.accept(samples)
        vad.accept_waveform(samples)
        assert vad.gain == pytest.approx(expected, rel=2e-6), (seed, index)
        assert 1 <= vad.gain <= 16
        if expected == 1:
            assert sink.samples is samples
        else:
            np.testing.assert_allclose(
                sink.samples,
                np.clip(samples * expected, -1, 1).astype(np.float32),
                rtol=3e-6,
                atol=1e-7,
                err_msg=f"seed={seed}, window={index}",
            )
            assert sink.samples.dtype == np.float32
        np.testing.assert_array_equal(samples, before)
        gains.append(vad.gain)
    assert min(gains) == 1 and max(gains) == 16
