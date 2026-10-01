"""Band-limited resampling: duration, callback partition invariance + FFT oracle.

The oracle reconstructs the retained Fourier series, independently of any FIR or
production resampler. Periodic tones and seeded periodic mixtures make its edges
known; trimming 20 ms excludes the streaming filter's startup/shutdown transient.
"""

import sys
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import live_stt  # noqa: E402

TARGET = 16000
RATES = (44100, 48000, 22050)
SCHEDULES = (256, 441, 1024, "random")
EDGE = 320


class Stream(Protocol):
    def __call__(self, block: np.ndarray) -> np.ndarray: ...

    def flush(self) -> np.ndarray: ...


def new_stream(rate: int, target: int = TARGET) -> Stream:
    cls = getattr(live_stt, "Resampler", None)
    assert cls is not None, "needs Resampler: streaming state/flush API absent"
    return cast(Stream, cls(rate, target))


def checked(output: np.ndarray) -> np.ndarray:
    assert output.dtype == np.float32, f"output dtype = {output.dtype}"
    assert output.ndim == 1, f"output shape = {output.shape}"
    assert np.isfinite(output).all(), "finite input produced a non-finite output"
    return output


def streamed(samples: np.ndarray, rate: int, schedule: int | str) -> np.ndarray:
    converter = new_stream(rate)
    rng = np.random.default_rng(9271)
    pieces = [checked(converter(np.empty(0, dtype=np.float32))).copy()]
    at = 0
    while at < len(samples):
        size = int(rng.integers(64, 2049)) if schedule == "random" else int(schedule)
        pieces.append(checked(converter(samples[at : at + size])).copy())
        at += size
        # Empty callbacks must preserve phase and pending output, even midstream.
        pieces.append(checked(converter(np.empty(0, dtype=np.float32))).copy())
    pieces.append(checked(converter.flush()).copy())
    return np.concatenate(pieces)


def reference(samples: np.ndarray, rate: int) -> np.ndarray:
    """Ideal periodic low-pass → evaluate at TARGET, via retained Fourier bins."""
    length = round(len(samples) * TARGET / rate)
    if not length:
        return np.empty(0, dtype=np.float64)
    bins = np.fft.rfft(samples.astype(np.float64))[: length // 2 + 1].copy()
    if length % 2 == 0 and length < len(samples):
        # Positive/negative frequencies merge at the output Nyquist bin.
        bins[-1] = 2 * bins[-1].real
    return np.fft.irfft(bins, n=length) * (length / len(samples))


def tone(rate: int, frequency: int) -> np.ndarray:
    time = np.arange(rate // 5, dtype=np.float64) / rate
    return (0.7 * np.sin(2 * np.pi * frequency * time + 0.31)).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def assert_fidelity(actual: np.ndarray, expected: np.ndarray) -> None:
    assert abs(len(actual) - len(expected)) <= 1
    end = min(len(actual), len(expected)) - EDGE
    actual, expected = actual[EDGE:end], expected[EDGE:end]
    assert len(actual) > 0, "no steady-state samples remain"
    amplitude_db = 20 * np.log10(rms(actual) / rms(expected))
    error = rms(actual.astype(np.float64) - expected)
    snr_db = float("inf") if error == 0 else 20 * np.log10(rms(expected) / error)
    assert abs(amplitude_db) <= 0.5, f"in-band amplitude = {amplitude_db:.3f} dB"
    assert snr_db >= 40, f"reference SNR = {snr_db:.3f} dB < 40 dB"


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_stream_length(rate: int, schedule: int | str) -> None:
    rng = np.random.default_rng(2104)
    lengths = (0, 1, 2, 3, 63, 64, 255, 256, 257, 440, 441, 442, 1023, 1024, 1025)
    lengths += (rate - 1, rate, rate + 1, 2 * rate + 137)
    lengths += tuple(int(n) for n in rng.integers(4, 4097, size=8))
    for length in lengths:
        samples = rng.uniform(-1, 1, size=length).astype(np.float32)
        actual = streamed(samples, rate, schedule)
        ideal = round(length * TARGET / rate)
        assert abs(len(actual) - ideal) <= 1, (
            f"n_in={length}, rate={rate}, schedule={schedule}: n_out={len(actual)}, ideal={ideal}"
        )


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_callback_partition_invariance(rate: int, schedule: int | str) -> None:
    rng = np.random.default_rng(4021)
    samples = rng.uniform(-1, 1, size=rate // 2 + 137).astype(np.float32)
    # Abrupt impulses expose a per-callback filter/phase reset as well as tones do.
    samples[::257] = 1
    actual = streamed(samples, rate, schedule)
    expected = checked(live_stt.resample(samples, rate, TARGET))
    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("frequency", (1000, 3000))
def test_stream_tone_fidelity(rate: int, schedule: int | str, frequency: int) -> None:
    samples = tone(rate, frequency)
    assert_fidelity(streamed(samples, rate, schedule), reference(samples, rate))


@pytest.mark.parametrize("rate", RATES)
def test_one_shot_bandlimited(rate: int) -> None:
    for frequency in (1000, 3000):
        samples = tone(rate, frequency)
        actual = checked(live_stt.resample(samples, rate, TARGET))
        assert_fidelity(actual, reference(samples, rate))
    # Integer-ratio decimation preserves passband tones but still aliases.
    samples = tone(rate, 9500)
    actual = checked(live_stt.resample(samples, rate, TARGET))
    attenuation_db = 20 * np.log10(max(rms(actual[EDGE:-EDGE]), 1e-30) / rms(samples))
    assert attenuation_db <= -40, f"alias attenuation = {attenuation_db:.3f} dB > -40 dB"


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_stream_generated_reference(rate: int, schedule: int | str) -> None:
    rng = np.random.default_rng(5692)
    time = np.arange(rate // 5, dtype=np.float64) / rate
    samples = np.zeros(len(time), dtype=np.float64)
    for frequency, phase, amplitude in zip(
        rng.choice(np.arange(300, 4501, 5), size=24, replace=False),
        rng.uniform(-np.pi, np.pi, size=24),
        rng.uniform(0.01, 0.04, size=24),
        strict=True,
    ):
        samples += amplitude * np.sin(2 * np.pi * frequency * time + phase)
    samples = samples.astype(np.float32)
    assert_fidelity(streamed(samples, rate, schedule), reference(samples, rate))


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("frequency", (9500, 12000))
def test_stream_alias_rejection(schedule: int | str, frequency: int) -> None:
    samples = tone(44100, frequency)
    actual = streamed(samples, 44100, schedule)
    attenuation_db = 20 * np.log10(max(rms(actual[EDGE:-EDGE]), 1e-30) / rms(samples))
    assert attenuation_db <= -40, f"alias attenuation = {attenuation_db:.3f} dB > -40 dB"


@pytest.mark.parametrize("frequency", (9500, 12000))
def test_one_shot_alias_rejection(frequency: int) -> None:
    samples = tone(44100, frequency)
    actual = checked(live_stt.resample(samples, 44100, TARGET))
    attenuation_db = 20 * np.log10(max(rms(actual[EDGE:-EDGE]), 1e-30) / rms(samples))
    assert attenuation_db <= -40, f"alias attenuation = {attenuation_db:.3f} dB > -40 dB"


def test_identity_and_empty_flush() -> None:
    converter = new_stream(TARGET)
    rng = np.random.default_rng(8541)
    for length in (0, 1, 256, 441, 1024, 2049):
        samples = rng.uniform(-1, 1, size=length).astype(np.float32)
        expected = samples.copy()
        actual = checked(converter(samples))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(samples, expected)
        np.testing.assert_array_equal(checked(live_stt.resample(samples, TARGET, TARGET)), expected)
    assert checked(converter.flush()).size == 0
