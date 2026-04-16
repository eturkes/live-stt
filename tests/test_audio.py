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
