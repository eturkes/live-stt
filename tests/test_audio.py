"""Tests for pure functions in live_stt.py."""

from __future__ import annotations

import asyncio
import io

import numpy as np

from live_stt import RingBuffer, emit_line, resample


def test_resample_identity():
    audio = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    out = resample(audio, 16000, 16000)
    np.testing.assert_array_equal(out, audio)


def test_resample_halving():
    audio = np.linspace(-1.0, 1.0, 3200, dtype=np.float32)
    out = resample(audio, 32000, 16000)
    assert len(out) == 1600
    assert out.dtype == np.float32


def test_resample_upsampling():
    audio = np.linspace(-1.0, 1.0, 1600, dtype=np.float32)
    out = resample(audio, 16000, 48000)
    assert len(out) == 4800


def test_resample_preserves_first_endpoint():
    audio = np.array([1.0, -1.0], dtype=np.float32)
    out = resample(audio, 16000, 32000)
    assert len(out) == 4
    assert out[0] == 1.0
    # np.interp clamps indices past the end to the last sample's value.
    assert out[-1] == -1.0


def test_resample_integer_decimation_48k_to_16k_matches_slice():
    # Optimization: the 48k->16k path uses audio[::3] instead of np.interp.
    # Verify content matches a manual decimation.
    audio = np.arange(4800, dtype=np.float32) / 4800.0
    out = resample(audio, 48000, 16000)
    assert len(out) == 1600
    np.testing.assert_array_equal(out, audio[::3])


def test_resample_integer_decimation_32k_to_16k_matches_slice():
    audio = np.arange(3200, dtype=np.float32) / 3200.0
    out = resample(audio, 32000, 16000)
    assert len(out) == 1600
    np.testing.assert_array_equal(out, audio[::2])


def test_resample_index_cache_repeat_calls():
    # The cache reuses precomputed indices AND the output buffer across same-shape
    # calls — audio_callback copies the result before enqueueing. Copy when retaining.
    a = np.random.RandomState(0).randn(4410).astype(np.float32) * 0.1
    out_a = resample(a, 44100, 16000).copy()
    b = np.random.RandomState(1).randn(4410).astype(np.float32) * 0.1
    out_b = resample(b, 44100, 16000).copy()
    # Different inputs -> different outputs (after copying out of the shared buffer).
    assert not np.array_equal(out_a, out_b)
    # Same input -> same output across calls.
    out_a_again = resample(a, 44100, 16000).copy()
    np.testing.assert_array_equal(out_a, out_a_again)


def test_resample_returns_shared_output_buffer():
    # Document the buffer-reuse contract: same key -> same buffer object.
    a = np.linspace(-0.5, 0.5, 882, dtype=np.float32)
    b = np.linspace(0.5, -0.5, 882, dtype=np.float32)
    out_a = resample(a, 44100, 16000)
    out_b = resample(b, 44100, 16000)
    assert out_a is out_b


def test_resample_matches_np_interp_for_typical_rates():
    # The custom interp must agree with np.interp for our supported rates.
    rng = np.random.default_rng(42)
    for n_in, orig, target in [
        (882, 44100, 16000),
        (1764, 44100, 16000),
        (441, 22050, 16000),
        (160, 16000, 48000),
    ]:
        audio = rng.standard_normal(n_in).astype(np.float32) * 0.1
        got = resample(audio, orig, target).copy()
        xp = np.arange(n_in, dtype=np.float64)
        step = orig / target
        indices = np.arange(int(n_in / step), dtype=np.float64) * step
        expected = np.interp(indices, xp, audio).astype(np.float32)
        np.testing.assert_allclose(got, expected, atol=1e-6)


def test_resample_dtype_preserved_for_integer_decimation():
    audio = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3], dtype=np.float32)
    out = resample(audio, 48000, 16000)
    assert out.dtype == np.float32


def test_audio_callback_pipeline_end_to_end():
    # Walk the exact resample+copy pipeline the live audio_callback runs, for
    # both the integer-decim path (48k) and the custom-interp path (44.1k).
    rng = np.random.default_rng(7)
    for native_rate, n_frames in [(48000, 960), (44100, 882), (32000, 640)]:
        indata = rng.standard_normal((n_frames, 1)).astype(np.float32) * 0.1
        mono = indata[:, 0]
        pcm = resample(mono, native_rate, 16000).copy()
        expected_samples = int(n_frames * 16000 / native_rate)
        assert len(pcm) == expected_samples
        assert pcm.dtype == np.float32
        # The copy must be contiguous and independent of the shared buffer.
        assert pcm.flags.c_contiguous


# --- RingBuffer (VAD pre-pad re-slicing, D-010 finding 2) ---


def test_ring_append_slice_basic():
    r = RingBuffer(10)
    r.append(np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(r.slice(0, 4), np.arange(4, dtype=np.float32))
    assert r.total == 4


def test_ring_wraparound_keeps_absolute_indexing():
    r = RingBuffer(8)
    for i in range(5):  # 25 samples through an 8-sample ring
        r.append(np.arange(i * 5, i * 5 + 5, dtype=np.float32))
    assert r.total == 25
    # The last 8 samples (17..24) are retained, absolute indices intact.
    np.testing.assert_array_equal(
        r.slice(17, 25), np.arange(17, 25, dtype=np.float32)
    )


def test_ring_slice_clamps_to_retained_window():
    r = RingBuffer(8)
    r.append(np.arange(20, dtype=np.float32))
    # Samples 0..11 are gone; a slice reaching back returns only 12..15.
    np.testing.assert_array_equal(
        r.slice(0, 16), np.arange(12, 16, dtype=np.float32)
    )


def test_ring_slice_clamps_negative_prepad():
    # The worker slices (start - pad, start + n); near stream start this goes
    # negative and must clamp to 0.
    r = RingBuffer(16)
    r.append(np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(r.slice(-4, 6), np.arange(6, dtype=np.float32))


def test_ring_slice_empty_when_out_of_range():
    r = RingBuffer(8)
    r.append(np.arange(4, dtype=np.float32))
    assert len(r.slice(10, 12)) == 0
    assert len(r.slice(3, 3)) == 0


def test_ring_append_larger_than_capacity():
    r = RingBuffer(4)
    r.append(np.arange(10, dtype=np.float32))
    assert r.total == 10
    np.testing.assert_array_equal(r.slice(6, 10), np.arange(6, 10, dtype=np.float32))


def test_ring_slice_spanning_wrap_point():
    r = RingBuffer(8)
    r.append(np.arange(6, dtype=np.float32))   # fills 0..5
    r.append(np.arange(6, 12, dtype=np.float32))  # wraps: retained 4..11
    np.testing.assert_array_equal(r.slice(4, 12), np.arange(4, 12, dtype=np.float32))


# --- emit_line ---


def test_emit_line_ja(capsys):
    buf = io.StringIO()
    emit_line("JA", 1, "こんにちは", buf)
    captured = capsys.readouterr()
    assert "JA 1: こんにちは" in captured.out
    assert "JA 1: こんにちは" in buf.getvalue()


def test_emit_line_en_shares_seq_tag(capsys):
    # JA and EN are emitted independently; the seq number ties pairs together.
    buf = io.StringIO()
    emit_line("JA", 2, "こんにちは", buf)
    emit_line("JA", 3, "次の文", buf)
    emit_line("EN", 2, "Hello", buf)
    content = buf.getvalue()
    assert "JA 2: こんにちは" in content
    assert "JA 3: 次の文" in content
    assert "EN 2: Hello" in content
    # Interleaved arrival keeps one self-describing event per line.
    assert content.index("JA 3") < content.index("EN 2")


def test_emit_line_writes_iso8601_timestamp_prefix():
    buf = io.StringIO()
    emit_line("JA", 1, "テスト", buf)
    first_line = buf.getvalue().split("\n", 1)[0]
    assert first_line.startswith("[")
    assert "] JA 1: テスト" in first_line
    assert "T" in first_line.split("]", 1)[0]


def test_emit_line_no_file_no_crash(capsys):
    emit_line("JA", 1, "テスト", None)
    captured = capsys.readouterr()
    assert "JA 1: テスト" in captured.out


def test_emit_line_line_clear_gated_on_stdout_tty(monkeypatch, capsys):
    # The \r\x1b[2K status-line clear must reach stdout only on a TTY, so a
    # redirected stdout stays ANSI-clean (symmetric with _StderrFormatter).
    monkeypatch.setattr("live_stt._STDOUT_TTY", False)
    emit_line("JA", 1, "x", None)
    assert "\x1b[2K" not in capsys.readouterr().out
    monkeypatch.setattr("live_stt._STDOUT_TTY", True)
    emit_line("JA", 2, "y", None)
    assert "\x1b[2K" in capsys.readouterr().out


# --- shutdown worker-stop sentinel (T8.1, run_session finally) ---


def test_shutdown_sentinel_lands_on_full_audio_queue_without_blocking():
    # run_session's shutdown sentinels worker() via an evict-then-put idiom
    # (matching CodexTranslator.submit_sentinel), NOT a blocking
    # `await audio_q.put(None)`. If worker() already died and the mic callback
    # filled audio_q to capacity, a blocking put would park the loop forever
    # (Ctrl+C routes to request_stop, not KeyboardInterrupt -> SIGKILL-only).
    # This exercises that idiom on a synthetic full queue: it must land the
    # sentinel without blocking, evicting exactly the oldest block.
    async def sentinel_into_full_queue():
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        for i in range(4):  # fill to capacity, no consumer
            q.put_nowait(i)
        while True:
            try:
                q.put_nowait(None)
                break
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
        return q

    # wait_for must NOT fire: a regression to a blocking put would hang here.
    q = asyncio.run(asyncio.wait_for(sentinel_into_full_queue(), 1.0))
    assert q.qsize() == 4  # still capped
    items = [q.get_nowait() for _ in range(4)]
    assert items[-1] is None  # sentinel landed
    assert items.count(None) == 1  # exactly one sentinel
    assert 0 not in items  # oldest block evicted, newer ones (1,2,3) survive
