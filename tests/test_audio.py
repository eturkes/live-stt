"""Tests for pure functions in live_stt.py."""

from __future__ import annotations

import argparse
import asyncio
import errno
import io
import subprocess
import sys
import types
import wave
from datetime import datetime

import numpy as np

import live_stt
from live_stt import (
    DECODE_CHUNK_OVERLAP_S,
    DECODE_CHUNK_S,
    DECODE_SPLIT_RMS_WINDOW_S,
    DECODE_SPLIT_SEARCH_S,
    DECODE_SPLIT_TRIGGER_S,
    SAMPLE_RATE,
    TRANSCRIPT_DIR,
    AudioRecording,
    RingBuffer,
    TranscriptFile,
    _merge_chunk_text,
    _split_decode_segment,
    audio_path,
    emit_line,
    linear_resample,
    resample,
    session_stamp,
    submit_audio_sentinel,
    transcript_path,
)


def test_module_import_keeps_audio_backend_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import live_stt; assert 'sounddevice' not in sys.modules",
        ],
        check=True,
    )


def test_resample_identity():
    audio = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    out = resample(audio, 16000, 16000)
    np.testing.assert_array_equal(out, audio)


def test_linear_resample_halving():
    audio = np.linspace(-1.0, 1.0, 3200, dtype=np.float32)
    out = linear_resample(audio, 32000, 16000)
    assert len(out) == 1600
    assert out.dtype == np.float32


def test_linear_resample_upsampling():
    audio = np.linspace(-1.0, 1.0, 1600, dtype=np.float32)
    out = linear_resample(audio, 16000, 48000)
    assert len(out) == 4800


def test_linear_resample_preserves_first_endpoint():
    audio = np.array([1.0, -1.0], dtype=np.float32)
    out = linear_resample(audio, 16000, 32000)
    assert len(out) == 4
    assert out[0] == 1.0
    # np.interp clamps indices past the end to the last sample's value.
    assert out[-1] == -1.0


def test_linear_resample_integer_decimation_48k_to_16k_matches_slice():
    # Optimization: the 48k->16k path uses audio[::3] instead of np.interp.
    # Verify content matches a manual decimation.
    audio = np.arange(4800, dtype=np.float32) / 4800.0
    out = linear_resample(audio, 48000, 16000)
    assert len(out) == 1600
    np.testing.assert_array_equal(out, audio[::3])


def test_linear_resample_integer_decimation_32k_to_16k_matches_slice():
    audio = np.arange(3200, dtype=np.float32) / 3200.0
    out = linear_resample(audio, 32000, 16000)
    assert len(out) == 1600
    np.testing.assert_array_equal(out, audio[::2])


def test_linear_resample_index_cache_repeat_calls():
    # The cache reuses precomputed indices AND the output buffer across same-shape
    # calls — audio_callback copies the result before enqueueing. Copy when retaining.
    a = np.random.RandomState(0).randn(4410).astype(np.float32) * 0.1
    out_a = linear_resample(a, 44100, 16000).copy()
    b = np.random.RandomState(1).randn(4410).astype(np.float32) * 0.1
    out_b = linear_resample(b, 44100, 16000).copy()
    # Different inputs -> different outputs (after copying out of the shared buffer).
    assert not np.array_equal(out_a, out_b)
    # Same input -> same output across calls.
    out_a_again = linear_resample(a, 44100, 16000).copy()
    np.testing.assert_array_equal(out_a, out_a_again)


def test_linear_resample_returns_shared_output_buffer():
    # Document the buffer-reuse contract: same key -> same buffer object.
    a = np.linspace(-0.5, 0.5, 882, dtype=np.float32)
    b = np.linspace(0.5, -0.5, 882, dtype=np.float32)
    out_a = linear_resample(a, 44100, 16000)
    out_b = linear_resample(b, 44100, 16000)
    assert out_a is out_b


def test_linear_resample_matches_np_interp_for_typical_rates():
    # The custom interp must agree with np.interp for our supported rates.
    rng = np.random.default_rng(42)
    for n_in, orig, target in [
        (882, 44100, 16000),
        (1764, 44100, 16000),
        (441, 22050, 16000),
        (160, 16000, 48000),
    ]:
        audio = rng.standard_normal(n_in).astype(np.float32) * 0.1
        got = linear_resample(audio, orig, target).copy()
        xp = np.arange(n_in, dtype=np.float64)
        step = orig / target
        indices = np.arange(int(n_in / step), dtype=np.float64) * step
        expected = np.interp(indices, xp, audio).astype(np.float32)
        np.testing.assert_allclose(got, expected, atol=1e-6)


def test_linear_resample_dtype_preserved_for_integer_decimation():
    audio = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3], dtype=np.float32)
    out = linear_resample(audio, 48000, 16000)
    assert out.dtype == np.float32


def test_audio_callback_pipeline_end_to_end():
    # Walk the exact Resampler+copy pipeline the live audio_callback runs: one stream
    # per session, so a session's output length is the ideal one, never a per-block
    # floor. The filter holds its tail until flush, which run_session drains at stop.
    rng = np.random.default_rng(7)
    for native_rate, n_frames in [(48000, 960), (44100, 882), (44100, 256), (32000, 640)]:
        resampler = live_stt.Resampler(native_rate)
        total = 0
        for _ in range(50):
            indata = rng.standard_normal((n_frames, 1)).astype(np.float32) * 0.1
            pcm = resampler(indata[:, 0]).copy()
            assert pcm.dtype == np.float32
            assert pcm.flags.c_contiguous
            total += len(pcm)
        total += len(resampler.flush())
        assert abs(total - round(50 * n_frames * 16000 / native_rate)) <= 1


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
    np.testing.assert_array_equal(r.slice(17, 25), np.arange(17, 25, dtype=np.float32))


def test_ring_slice_clamps_to_retained_window():
    r = RingBuffer(8)
    r.append(np.arange(20, dtype=np.float32))
    # Samples 0..11 are gone; a slice reaching back returns only 12..15.
    np.testing.assert_array_equal(r.slice(0, 16), np.arange(12, 16, dtype=np.float32))


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
    r.append(np.arange(6, dtype=np.float32))  # fills 0..5
    r.append(np.arange(6, 12, dtype=np.float32))  # wraps: retained 4..11
    np.testing.assert_array_equal(r.slice(4, 12), np.arange(4, 12, dtype=np.float32))


# --- Long-segment decode chunking (M9.4) ---


def test_split_decode_segment_preserves_short_input_by_identity():
    segment = np.zeros(round(DECODE_SPLIT_TRIGGER_S * SAMPLE_RATE), dtype=np.float32)
    chunks = _split_decode_segment(segment)
    assert len(chunks) == 1
    assert chunks[0] is segment


def test_split_decode_segment_uses_balanced_low_rms_cuts_and_overlap():
    n = round((DECODE_SPLIT_TRIGGER_S + 0.1) * SAMPLE_RATE)
    segment = np.ones(n, dtype=np.float32)
    chunk_samples = round(DECODE_CHUNK_S * SAMPLE_RATE)
    count = (n + chunk_samples - 1) // chunk_samples
    rms_half = round(DECODE_SPLIT_RMS_WINDOW_S * SAMPLE_RATE) // 2
    offset = round(0.25 * SAMPLE_RATE)
    cuts = [round(i * n / count) + offset for i in range(1, count)]
    assert offset < round(DECODE_SPLIT_SEARCH_S * SAMPLE_RATE)
    for cut in cuts:
        segment[cut - rms_half : cut + rms_half] = 0.0

    chunks = _split_decode_segment(segment)
    overlap = round(DECODE_CHUNK_OVERLAP_S * SAMPLE_RATE)
    bounds = [0, *cuts, n]
    expected_lengths = [
        min(n, end + overlap) - max(0, start - overlap)
        for start, end in zip(bounds, bounds[1:], strict=False)
    ]
    assert [len(chunk) for chunk in chunks] == expected_lengths
    assert all(np.shares_memory(segment, chunk) for chunk in chunks)


def test_merge_chunk_text_removes_only_plausible_exact_overlap():
    assert _merge_chunk_text(["空が青い", "青いです"]) == "空が青いです"
    # One repeated character can be real speech at the cut; retain it rather
    # than guessing from too little evidence.
    assert _merge_chunk_text(["時", "時です"]) == "時時です"
    assert _merge_chunk_text(["そのまま"]) == "そのまま"


# --- emit_line ---


def test_emit_line_src(capsys):
    buf = io.StringIO()
    emit_line("SRC", 1, "こんにちは", buf)
    captured = capsys.readouterr()
    assert "SRC 1: こんにちは" in captured.out
    assert "SRC 1: こんにちは" in buf.getvalue()


def test_emit_line_tgt_shares_seq_tag(capsys):
    # SRC and TGT are emitted independently; the seq number ties pairs together.
    buf = io.StringIO()
    emit_line("SRC", 2, "こんにちは", buf)
    emit_line("SRC", 3, "次の文", buf)
    emit_line("TGT", 2, "Hello", buf)
    content = buf.getvalue()
    assert "SRC 2: こんにちは" in content
    assert "SRC 3: 次の文" in content
    assert "TGT 2: Hello" in content
    # Interleaved arrival keeps one self-describing event per line.
    assert content.index("SRC 3") < content.index("TGT 2")


def test_emit_line_writes_iso8601_timestamp_prefix():
    buf = io.StringIO()
    emit_line("SRC", 1, "テスト", buf)
    first_line = buf.getvalue().split("\n", 1)[0]
    assert first_line.startswith("[")
    assert "] SRC 1: テスト" in first_line
    assert "T" in first_line.split("]", 1)[0]


def test_emit_line_no_file_no_crash(capsys):
    emit_line("SRC", 1, "テスト", None)
    captured = capsys.readouterr()
    assert "SRC 1: テスト" in captured.out


class _DeadTerminal:
    """A pty slave whose master is gone: every write raises OSError errno 5."""

    def write(self, text):
        raise OSError(errno.EIO, "Input/output error")

    def flush(self):
        raise OSError(errno.EIO, "Input/output error")


def test_emit_line_persists_when_the_terminal_is_already_gone(monkeypatch):
    # Closing the terminal runs the shutdown drain against a dead pty. stdout used
    # to go first, so the line raised on the display instead of reaching the file
    # it was being drained into -- one lost target line per session, always the last.
    monkeypatch.setattr(live_stt, "_stdout_live", True)
    monkeypatch.setattr(sys, "stdout", _DeadTerminal())
    buf = io.StringIO()
    emit_line("SRC", 1, "こんにちは", buf)
    emit_line("TGT", 1, "Hello", buf)
    assert "SRC 1: こんにちは" in buf.getvalue()
    assert "TGT 1: Hello" in buf.getvalue()


def test_write_stdout_latches_off_after_the_first_refusal(monkeypatch):
    # One failed syscall per line for the rest of the session buys nothing, and a
    # latch is what lets every other stdout writer share this path unguarded.
    dead = _DeadTerminal()
    calls = []
    monkeypatch.setattr(live_stt, "_stdout_live", True)
    monkeypatch.setattr(sys, "stdout", dead)
    monkeypatch.setattr(dead, "write", lambda text: calls.append(text) or _raise_eio())
    live_stt.write_stdout("first")
    live_stt.write_stdout("second")
    assert calls == ["first"]
    assert live_stt._stdout_live is False


def _raise_eio():
    raise OSError(errno.EIO, "Input/output error")


def test_emit_line_line_clear_gated_on_stdout_tty(monkeypatch, capsys):
    # The \r\x1b[2K status-line clear must reach stdout only on a TTY, so a
    # redirected stdout stays ANSI-clean (symmetric with _StderrFormatter).
    monkeypatch.setattr("live_stt._STDOUT_TTY", False)
    emit_line("SRC", 1, "x", None)
    assert "\x1b[2K" not in capsys.readouterr().out
    monkeypatch.setattr("live_stt._STDOUT_TTY", True)
    emit_line("SRC", 2, "y", None)
    assert "\x1b[2K" in capsys.readouterr().out


# --- transcript persistence (saving is on by default) ---


def _save_args(**overrides):
    return argparse.Namespace(
        **{"output": None, "no_save": False, "save_audio": False, **overrides}
    )


def test_transcript_path_defaults_to_timestamped_session_file():
    path = transcript_path(_save_args(), session_stamp())
    assert path is not None
    assert path.parent == TRANSCRIPT_DIR
    assert path.suffix == ".txt"
    # Sortable start time, no colons (keeps the name shell- and tool-friendly).
    datetime.strptime(path.stem, "%Y-%m-%dT%H-%M-%S")


def test_transcript_path_output_flag_overrides_default(tmp_path):
    target = tmp_path / "sub" / "session.txt"
    assert transcript_path(_save_args(output=str(target)), session_stamp()) == target


def test_transcript_path_none_when_saving_disabled():
    assert transcript_path(_save_args(no_save=True), session_stamp()) is None


# --- opt-in session audio (--save-audio) ---


def test_audio_path_is_off_unless_asked_for():
    assert audio_path(_save_args(), session_stamp()) is None


def test_audio_shares_the_default_transcript_stem():
    stamp = session_stamp()
    transcript = transcript_path(_save_args(), stamp)
    audio = audio_path(_save_args(save_audio=True), stamp)
    assert transcript is not None and audio is not None
    assert (audio.parent, audio.stem, audio.suffix) == (transcript.parent, transcript.stem, ".wav")


def test_audio_stays_in_the_session_directory_whatever_the_transcript_flags(tmp_path):
    stamp = session_stamp()
    for flags in ({"output": str(tmp_path / "x.txt")}, {"no_save": True}):
        assert audio_path(_save_args(save_audio=True, **flags), stamp) == (
            TRANSCRIPT_DIR / f"{stamp}.wav"
        )


def test_the_recording_defers_creation_to_the_first_block(tmp_path):
    path = tmp_path / "nested" / "session.wav"
    recording = AudioRecording(path)
    assert path.parent.is_dir()
    assert not path.exists()
    recording.close()
    assert not path.exists()


def test_the_recording_is_readable_before_close_and_round_trips(tmp_path):
    path = tmp_path / "session.wav"
    blocks = [
        np.linspace(-1.0, 1.0, 85, dtype=np.float32),
        np.full(85, 0.25, dtype=np.float32),
        np.array([1.5, -1.5], dtype=np.float32),  # out-of-range PCM clips, never wraps
    ]
    recording = AudioRecording(path)
    # The header is patched on every write, so a session that dies after ANY block,
    # the first included, leaves a file every reader accepts at its true length.
    for written, block in enumerate(blocks, start=1):
        recording.write(block)
        with wave.open(str(path), "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SAMPLE_RATE)
            assert w.getnframes() == sum(len(b) for b in blocks[:written])
    recording.close()
    import replay  # noqa: PLC0415 -- the consumer this format exists for

    loaded = replay.load_wav_f32_16k(path)
    expected = np.clip(np.concatenate(blocks), -1.0, 32767 / 32768)
    assert np.max(np.abs(loaded - expected)) <= 0.5 / 32768


def _run_session_with_fake_mic(monkeypatch, tmp_path, blocks, rate=SAMPLE_RATE, **flags):
    """run_session end to end over an in-process fake mic + worker, no device."""
    stream_blocks = [b.reshape(-1, 1) for b in blocks]

    class InputStream:
        def __init__(self, callback, **_kwargs):
            self.callback = callback

        def start(self):
            for block in stream_blocks:
                self.callback(block, len(block), None, None)

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = types.SimpleNamespace(
        InputStream=InputStream,
        query_devices=lambda *_a, **_k: {"default_samplerate": rate, "name": "fake"},
    )
    seen = []

    async def worker(_rec, _vad, _window, audio_q, state, *_args, **_kwargs):
        while (chunk := await audio_q.get()) is not None:
            seen.append(chunk)
            if len(seen) == len(blocks):
                state.request_stop()  # every callback has landed; the drain follows

    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(live_stt, "TRANSCRIPT_DIR", tmp_path)
    monkeypatch.setattr(live_stt, "load_recognizer", lambda *_a: object())
    monkeypatch.setattr(live_stt, "make_vad", lambda: (None, 512))
    monkeypatch.setattr(live_stt, "worker", worker)
    monkeypatch.setattr(live_stt, "_install_signal_handlers", lambda _state: None)
    args = types.SimpleNamespace(
        engine="whisper",
        asr_device="NPU",
        two_way=False,
        device=None,
        output=None,
        no_save=True,
        no_translate=True,
        context="",
        **flags,
    )
    asyncio.run(live_stt.run_session(args))
    return seen


def test_save_audio_records_every_captured_block(monkeypatch, tmp_path):
    blocks = [np.full(160, 0.25, dtype=np.float32), np.linspace(-1, 1, 320, dtype=np.float32)]
    seen = _run_session_with_fake_mic(monkeypatch, tmp_path, blocks, save_audio=True)
    assert len(seen) == len(blocks)  # the recording rides beside the queue, not instead of it
    (path,) = tmp_path.glob("*.wav")
    with wave.open(str(path), "rb") as w:
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    expected = np.clip(np.round(np.concatenate(blocks) * 32768.0), -32768, 32767)
    assert np.array_equal(frames, expected.astype("<i2"))


def test_a_44k_session_delivers_its_whole_stream_to_the_worker_and_the_wav(monkeypatch, tmp_path):
    """The resampler's filter holds the last 6-40 ms; stopping must drain it.

    50 callbacks of 256 frames at 44.1 kHz carry 4644 samples at 16 kHz; without the
    shutdown flush the worker and the WAV both ended 327 short (reviewer-4's witness).
    """
    rng = np.random.default_rng(3)
    blocks = [(rng.standard_normal(256) * 0.1).astype(np.float32) for _ in range(50)]
    seen = _run_session_with_fake_mic(monkeypatch, tmp_path, blocks, rate=44100, save_audio=True)
    ideal = round(50 * 256 * SAMPLE_RATE / 44100)
    assert abs(sum(len(chunk) for chunk in seen) - ideal) <= 1
    (path,) = tmp_path.glob("*.wav")
    with wave.open(str(path), "rb") as w:
        assert abs(w.getnframes() - ideal) <= 1


def test_a_session_without_the_flag_writes_no_audio(monkeypatch, tmp_path):
    blocks = [np.full(160, 0.25, dtype=np.float32)]
    _run_session_with_fake_mic(monkeypatch, tmp_path, blocks, save_audio=False)
    assert list(tmp_path.glob("*.wav")) == []


def test_transcript_file_defers_creation_to_first_line(tmp_path):
    # Default-on saving must not leave an empty file behind a session that
    # decoded nothing; the parent directory is still made at construction.
    path = tmp_path / "nested" / "session.txt"
    f = TranscriptFile(path)
    assert path.parent.is_dir()
    assert not path.exists()
    f.close()
    assert not path.exists()


def test_transcript_file_appends_and_flushes_each_event(tmp_path):
    path = tmp_path / "session.txt"
    path.write_text("[t] SRC 1: 既存\n", encoding="utf-8")
    f = TranscriptFile(path)
    emit_line("SRC", 2, "テスト", f)
    # emit_line flushes per event, so a killed session keeps every landed line.
    content = path.read_text(encoding="utf-8")
    assert content.startswith("[t] SRC 1: 既存\n")
    assert "SRC 2: テスト" in content
    f.close()


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
        await submit_audio_sentinel(q)
        return q

    # wait_for must NOT fire: a regression to a blocking put would hang here.
    q = asyncio.run(asyncio.wait_for(sentinel_into_full_queue(), 1.0))
    assert q.qsize() == 4  # still capped
    items = [q.get_nowait() for _ in range(4)]
    assert items[-1] is None  # sentinel landed
    assert items.count(None) == 1  # exactly one sentinel
    assert 0 not in items  # oldest block evicted, newer ones (1,2,3) survive


def test_shutdown_sentinel_follows_already_scheduled_capture():
    async def scheduled_capture_then_sentinel():
        q: asyncio.Queue = asyncio.Queue()
        pcm = np.ones(16, dtype=np.float32)
        # Match audio_callback's cross-thread scheduling surface exactly.
        asyncio.get_running_loop().call_soon_threadsafe(q.put_nowait, pcm)
        await submit_audio_sentinel(q)
        return q, pcm

    q, pcm = asyncio.run(scheduled_capture_then_sentinel())
    assert q.get_nowait() is pcm
    assert q.get_nowait() is None
    assert q.empty()
