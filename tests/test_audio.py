"""Tests for pure functions in live_stt.py."""

from __future__ import annotations

import io

import numpy as np

from live_stt import emit_block, pcm16_bytes, resample


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


def test_pcm16_roundtrip():
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    pcm = pcm16_bytes(audio)
    parsed = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    np.testing.assert_allclose(parsed, audio, atol=1e-4)


def test_pcm16_clipping():
    audio = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    parsed = np.frombuffer(pcm16_bytes(audio), dtype=np.int16)
    assert parsed[0] == 32767
    assert parsed[1] == -32767
    assert parsed[2] == 0


def test_pcm16_byte_length():
    audio = np.zeros(100, dtype=np.float32)
    assert len(pcm16_bytes(audio)) == 200


def test_emit_block_parses_ja_and_en(capsys):
    buf = io.StringIO()
    emit_block("JA: こんにちは\nEN: Hello", buf, expect_en=True)
    captured = capsys.readouterr()
    assert "JA: こんにちは" in captured.out
    assert "EN: Hello" in captured.out
    content = buf.getvalue()
    assert "JA: こんにちは" in content
    assert "EN: Hello" in content


def test_emit_block_suppresses_en_when_not_expected(capsys):
    buf = io.StringIO()
    emit_block("JA: こんにちは\nEN: Hello", buf, expect_en=False)
    captured = capsys.readouterr()
    assert "JA: こんにちは" in captured.out
    assert "EN:" not in captured.out


def test_emit_block_omits_en_when_model_only_sent_ja(capsys):
    buf = io.StringIO()
    emit_block("JA: only japanese", buf, expect_en=True)
    captured = capsys.readouterr()
    assert "JA: only japanese" in captured.out
    assert "EN:" not in captured.out


def test_emit_block_falls_back_on_unlabeled_text(capsys):
    buf = io.StringIO()
    emit_block("just some text without prefixes", buf, expect_en=True)
    captured = capsys.readouterr()
    assert "JA: just some text without prefixes" in captured.out


def test_emit_block_writes_iso8601_timestamp_prefix():
    buf = io.StringIO()
    emit_block("JA: テスト", buf, expect_en=True)
    first_line = buf.getvalue().split("\n", 1)[0]
    assert first_line.startswith("[") and first_line.endswith("]")
    assert "T" in first_line


def test_emit_block_no_file_no_crash(capsys):
    emit_block("JA: テスト", None, expect_en=True)
    captured = capsys.readouterr()
    assert "JA: テスト" in captured.out


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
    # calls — callers in audio_callback hand the result straight to pcm16_bytes,
    # so the buffer is consumed before the next call. Copy when retaining.
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


def test_pcm16_accepts_strided_view():
    # After the 48k->16k optimization, resample() returns a non-contiguous strided
    # view (audio[::3]). pcm16_bytes must handle that input shape correctly.
    audio = np.arange(4800, dtype=np.float32) / 5000.0  # 0..0.96
    decimated = audio[::3]
    assert not decimated.flags.c_contiguous
    pcm = pcm16_bytes(decimated)
    parsed = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    np.testing.assert_allclose(parsed, decimated, atol=1e-4)


def test_resample_dtype_preserved_for_integer_decimation():
    audio = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3], dtype=np.float32)
    out = resample(audio, 48000, 16000)
    assert out.dtype == np.float32


def test_audio_callback_pipeline_end_to_end():
    # Walk the exact resample+pcm16 pipeline the live audio_callback runs, for
    # both the integer-decim path (48k) and the custom-interp path (44.1k).
    rng = np.random.default_rng(7)
    for native_rate, n_frames in [(48000, 960), (44100, 882), (32000, 640)]:
        indata = (rng.standard_normal((n_frames, 1)).astype(np.float32) * 0.1)
        mono = indata[:, 0]
        # Mirror the audio_callback RMS step.
        ms = float(mono.dot(mono)) / len(mono)
        assert ms >= 0.0
        pcm = pcm16_bytes(resample(mono, native_rate, 16000))
        # 16k mono pcm16 => 16000/native_rate * n_frames samples * 2 bytes.
        expected_samples = int(n_frames * 16000 / native_rate)
        assert len(pcm) == expected_samples * 2
        # The bytes can be decoded back into int16 in the valid range.
        parsed = np.frombuffer(pcm, dtype=np.int16)
        assert parsed.min() >= -32767
        assert parsed.max() <= 32767


def test_pcm16_repeat_calls_share_scratch_buffers():
    # pcm16_bytes caches a float and int16 scratch buffer per output length to
    # skip per-call allocations. The returned bytes must still be independent of
    # subsequent calls — bytes are immutable, so this is automatic, but verify.
    from live_stt import _PCM16_FLOAT_BUF, _PCM16_INT16_BUF
    a = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    b = np.array([0.9, -0.9, 0.5, -0.5], dtype=np.float32)
    pa = pcm16_bytes(a)
    fa = _PCM16_FLOAT_BUF[len(a)]
    ia = _PCM16_INT16_BUF[len(a)]
    pb = pcm16_bytes(b)
    # Same key -> same cached buffers (not realloc'd).
    assert _PCM16_FLOAT_BUF[len(b)] is fa
    assert _PCM16_INT16_BUF[len(b)] is ia
    # Bytes are snapshots, so earlier results survive later calls.
    assert pa != pb
    parsed_a = np.frombuffer(pa, dtype=np.int16).astype(np.float32) / 32767.0
    np.testing.assert_allclose(parsed_a, a, atol=1e-4)


def test_pcm16_clears_cache_when_too_many_sizes():
    # The size-keyed cache evicts when it grows beyond 8 distinct sizes so a
    # jittery blocksize can't leak memory.
    from live_stt import _PCM16_FLOAT_BUF
    _PCM16_FLOAT_BUF.clear()
    for n in range(1, 9):
        pcm16_bytes(np.zeros(n, dtype=np.float32))
    assert len(_PCM16_FLOAT_BUF) == 8
    # 9th distinct size triggers a cache flush before insert.
    pcm16_bytes(np.zeros(100, dtype=np.float32))
    assert len(_PCM16_FLOAT_BUF) == 1
