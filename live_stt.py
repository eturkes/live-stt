#!/usr/bin/env python3
"""Live Japanese speech-to-text, fully local: silero VAD + sherpa-onnx decode.

Captures microphone audio, segments utterances with silero VAD, and decodes
each completed segment on-CPU with a sherpa-onnx offline recognizer
(reazonspeech-k2-v2 default, parakeet-ja alternate). No network, no API keys.
English translation joins in T4.4 via the Codex subscription surface (D-011).
"""

import argparse
import asyncio
import logging
import math
import signal
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd

SAMPLE_RATE = 16000  # VAD + recognizer rate; mic native rate is resampled to this
METER_WIDTH = 40
METER_INTERVAL = 0.1
METER_FULL_SCALE_RMS = 0.05  # RMS that fills the bar
AUDIO_QUEUE_MAX = 100
NUM_THREADS = 4  # onnxruntime intra-op threads (8-core box; decode RTF ~0.05)

MODELS_DIR = Path(__file__).resolve().parent / "models"
VAD_MODEL = MODELS_DIR / "silero_vad.onnx"
ENGINE_DIRS = {
    "k2v2": MODELS_DIR / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01",
    "parakeet": MODELS_DIR / "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8",
}

# VAD tuning (T4.1 bench, D-010): 2 s pauses must split utterances, sub-second
# intra-sentence pauses must not.
VAD_MIN_SILENCE_S = 0.5
VAD_MIN_SPEECH_S = 0.25
# Silero opens segments 0.2-0.7 s after true speech onset and sherpa exposes
# no pad field, which clips leading syllables (こんにちは → はい). Re-slice
# each segment from the ring buffer with this much lead-in; reaching back
# into silence is harmless.
VAD_PRE_PAD_S = 0.4
RING_SECONDS = 60  # ring capacity; bounds VAD pre-pad re-slicing memory

# Prebuilt meter bars indexed by fill level — avoids two string allocations per tick.
_METER_BARS = tuple("#" * i + " " * (METER_WIDTH - i) for i in range(METER_WIDTH + 1))
# Constant-folded scale factor: rms * _METER_SCALE truncates to bar level.
_METER_SCALE = METER_WIDTH / METER_FULL_SCALE_RMS
# ANSI: carriage-return + erase-line. Lets block output overwrite the live meter.
_LINE_CLEAR = "\r\x1b[2K"

logger = logging.getLogger("live_stt")


class _StderrFormatter(logging.Formatter):
    # Prepend _LINE_CLEAR only when stderr is a TTY so log records erase the live
    # level meter (stdout) in place. When stderr is redirected to a file, the prefix
    # is omitted so the log stays free of ANSI escapes.
    def __init__(self):
        super().__init__(fmt="[%(asctime)s] %(levelname)s %(message)s")
        self._tty = sys.stderr.isatty()

    def format(self, record):
        msg = super().format(record)
        return _LINE_CLEAR + msg if self._tty else msg


def _configure_logging():
    if logger.handlers:
        return
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_StderrFormatter())
    logger.addHandler(handler)
    logger.propagate = False


# key -> (idx_floor, idx_ceil, frac, y0, y1). y0 doubles as the output buffer.
# Returned buffer is reused across calls — callers must consume (or copy) before
# the next call. audio_callback copies when enqueueing.
_RESAMPLE_CACHE: dict[
    tuple[int, int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
] = {}


def resample(audio, orig_rate, target_rate):
    if orig_rate == target_rate:
        return audio
    if orig_rate > target_rate:
        ratio = orig_rate / target_rate
        decim = int(ratio)
        if decim == ratio:
            # Integer downsample (e.g. 48k/32k -> 16k): a strided slice beats np.interp
            # by ~20x. Downstream copy handles contiguity.
            return audio[::decim]
    n_in = len(audio)
    key = (n_in, orig_rate, target_rate)
    cached = _RESAMPLE_CACHE.get(key)
    if cached is None:
        # xp is implicitly arange(n_in), so floor(index) is the lookup. Precomputing
        # idx_floor/idx_ceil/frac lets us skip np.interp's binary search: 1.3-2x faster.
        step = orig_rate / target_rate  # input samples per output sample
        n_out = int(n_in / step)
        indices = np.arange(n_out, dtype=np.float64) * step
        np.clip(indices, 0.0, n_in - 1, out=indices)  # match np.interp's edge behavior
        idx_floor = np.floor(indices).astype(np.intp)
        np.minimum(idx_floor, n_in - 2, out=idx_floor)  # keep idx+1 in bounds
        idx_ceil = idx_floor + 1
        frac = (indices - idx_floor).astype(np.float32)
        y0 = np.empty(n_out, dtype=np.float32)
        y1 = np.empty(n_out, dtype=np.float32)
        cached = (idx_floor, idx_ceil, frac, y0, y1)
        # Cap the cache: with a stable mic the key is constant; this bounds worst-case
        # memory if blocksize fluctuates.
        if len(_RESAMPLE_CACHE) >= 8:
            _RESAMPLE_CACHE.clear()
        _RESAMPLE_CACHE[key] = cached
    idx_floor, idx_ceil, frac, y0, y1 = cached
    # Fancy indexing then slice-assign beats np.take(out=) by ~40% on small arrays:
    # np.take's Python dispatch with out= is heavier than the fresh-alloc-then-memcpy
    # path inside fancy indexing. The y0/y1 buffers stay reused for the arithmetic.
    y0[:] = audio[idx_floor]
    y1[:] = audio[idx_ceil]
    np.subtract(y1, y0, out=y1)
    np.multiply(y1, frac, out=y1)
    np.add(y0, y1, out=y0)
    return y0


class RingBuffer:
    """Fixed-capacity float32 ring with absolute sample indexing.

    Retains the most recent `capacity` samples ever appended; slice(a, b)
    returns a copy of samples [a, b) clamped to the retained window. Absolute
    indices match silero's segment starts because everything fed to the VAD is
    appended here in the same order.
    """

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self.total = 0  # absolute count of samples ever appended

    def append(self, x: np.ndarray):
        n = len(x)
        if n >= self._cap:
            # Only the tail survives — but it must land phase-aligned so that
            # absolute index j still lives at position j % cap.
            tail = x[-self._cap :]
            pos = (self.total + n - self._cap) % self._cap
            k = self._cap - pos
            self._buf[pos:] = tail[:k]
            self._buf[:pos] = tail[k:]
            self.total += n
            return
        pos = self.total % self._cap
        end = pos + n
        if end <= self._cap:
            self._buf[pos:end] = x
        else:
            k = self._cap - pos
            self._buf[pos:] = x[:k]
            self._buf[: end - self._cap] = x[k:]
        self.total += n

    def slice(self, a: int, b: int) -> np.ndarray:
        lo = max(a, self.total - self._cap, 0)
        hi = min(b, self.total)
        if hi <= lo:
            return np.empty(0, dtype=np.float32)
        out = np.empty(hi - lo, dtype=np.float32)
        start = lo % self._cap
        end = start + (hi - lo)
        if end <= self._cap:
            out[:] = self._buf[start:end]
        else:
            k = self._cap - start
            out[:k] = self._buf[start:]
            out[k:] = self._buf[: end - self._cap]
        return out


def load_recognizer(engine: str) -> sherpa_onnx.OfflineRecognizer:
    d = ENGINE_DIRS[engine]
    if engine == "k2v2":
        # int8 encoder + fp32 decoder/joiner ≈ fp32 CER (HILab table), RTF 0.054.
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(d / "decoder-epoch-99-avg-1.onnx"),
            joiner=str(d / "joiner-epoch-99-avg-1.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=NUM_THREADS,
        )
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(d / "model.int8.onnx"),
        tokens=str(d / "tokens.txt"),
        num_threads=NUM_THREADS,
    )


def make_vad() -> tuple[sherpa_onnx.VoiceActivityDetector, int]:
    """Returns (vad, window_size_in_samples)."""
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(VAD_MODEL)
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = VAD_MIN_SILENCE_S
    cfg.silero_vad.min_speech_duration = VAD_MIN_SPEECH_S
    cfg.sample_rate = SAMPLE_RATE
    window = int(cfg.silero_vad.window_size)
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60), window


def _decode(rec: sherpa_onnx.OfflineRecognizer, samples: np.ndarray) -> str:
    s = rec.create_stream()
    s.accept_waveform(SAMPLE_RATE, samples)
    rec.decode_stream(s)
    return s.result.text.strip()


def check_models(engine: str) -> str | None:
    """Returns an error message if model files are missing, else None."""
    missing = []
    if not VAD_MODEL.exists():
        missing.append("silero_vad.onnx")
    d = ENGINE_DIRS[engine]
    if not (d / "tokens.txt").exists():
        missing.append(d.name + "/")
    if not missing:
        return None
    return (
        f"Missing model files under {MODELS_DIR}/: {', '.join(missing)}\n"
        "Download from https://github.com/k2-fsa/sherpa-onnx/releases "
        "(asr-models tag holds the model tarballs and silero_vad.onnx); "
        "see models/README.md."
    )


class State:
    def __init__(self):
        # Mean-square only; sqrt is deferred to the meter render path so the audio
        # thread doesn't pay for it 10-200x per second.
        self.latest_ms = 0.0
        self.dropped = 0
        self.stopping = False
        self.stop_event: asyncio.Event = None  # type: ignore[assignment]

    def request_stop(self):
        self.stopping = True
        if self.stop_event is not None:
            self.stop_event.set()


def emit_block(ja, en, output_file):
    """Display + persist one utterance block. en may be empty (JA-only mode)."""
    lines = [f"JA: {ja}"]
    if en:
        lines.append(f"EN: {en}")
    sys.stdout.write(_LINE_CLEAR)
    for line in lines:
        print(f"  {line}")
    print("-" * 60)
    if output_file:
        ts = datetime.now().astimezone().isoformat(timespec="seconds")
        output_file.write(f"[{ts}]\n")
        for line in lines:
            output_file.write(line + "\n")
        output_file.write("\n")
        output_file.flush()


async def worker(rec, vad, window, audio_q, state, output_file):
    """Drain mic queue -> feed VAD -> decode completed segments -> emit blocks.

    Decode runs in the default executor so queue draining (and the mic
    callback's enqueue) never stalls behind a long segment. Sequential decode
    preserves block order. A None sentinel flushes the VAD and exits.
    """
    loop = asyncio.get_running_loop()
    buf = np.empty(0, dtype=np.float32)
    ring = RingBuffer(RING_SECONDS * SAMPLE_RATE)
    pad = int(VAD_PRE_PAD_S * SAMPLE_RATE)
    try:
        while True:
            chunk = await audio_q.get()
            flush = chunk is None
            if not flush:
                buf = np.concatenate([buf, chunk]) if len(buf) else chunk
                while len(buf) >= window:
                    vad.accept_waveform(buf[:window])
                    ring.append(buf[:window])
                    buf = buf[window:]
            else:
                if len(buf):
                    vad.accept_waveform(buf)
                    ring.append(buf)
                vad.flush()
            while not vad.empty():
                start = int(vad.front.start)
                n = len(vad.front.samples)
                vad.pop()
                seg = ring.slice(start - pad, start + n)
                text = await loop.run_in_executor(None, _decode, rec, seg)
                if text:
                    emit_block(text, "", output_file)
            if flush:
                return
    except Exception:
        logger.exception("worker died")
        state.request_stop()


async def meter(state, audio_q):
    sleep = asyncio.sleep
    interval = METER_INTERVAL
    sqrt = math.sqrt
    width = METER_WIDTH
    scale = _METER_SCALE
    bars = _METER_BARS
    write = sys.stdout.write
    flush = sys.stdout.flush
    while not state.stopping:
        rms = sqrt(state.latest_ms)
        level = int(rms * scale)
        if level > width:
            level = width
        qsize = audio_q.qsize()
        pending = f" q={qsize}" if qsize > 0 else ""
        dropped = f" drop={state.dropped}" if state.dropped else ""
        write(f"\r  [{bars[level]}] {rms:.4f}{pending}{dropped}")
        flush()
        await sleep(interval)


async def run_session(args):
    print(f"Loading {args.engine} model...")
    rec = load_recognizer(args.engine)
    vad, window = make_vad()

    dev_info = sd.query_devices(args.device, kind="input")
    native_rate = int(dev_info["default_samplerate"])
    if args.device is not None:
        dev_label = f"#{args.device} {dev_info['name']}"
    else:
        dev_label = dev_info["name"]
    print(f"Mic: {dev_label} @ {native_rate} Hz (decoding locally at {SAMPLE_RATE} Hz)")

    state = State()
    state.stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAX)

    output_file = open(args.output, "a", encoding="utf-8") if args.output else None
    if output_file:
        print(f"Writing transcriptions to: {args.output}")

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning("audio: %s", status)
        # sounddevice always passes a 2-D (frames, channels) array for InputStream;
        # frames is the same as len(mono), so use it directly.
        mono = indata[:, 0]
        # Dot product avoids the (mono**2) temporary allocation; sqrt is deferred
        # to the meter render path so the audio thread doesn't pay for it.
        state.latest_ms = float(mono.dot(mono)) / frames
        # Copy: resample() returns a shared/reused (or strided-view) buffer, and
        # the queue defers consumption past the next callback.
        pcm = resample(mono, native_rate, SAMPLE_RATE).copy()
        loop.call_soon_threadsafe(_enqueue, pcm)

    def _enqueue(pcm):
        try:
            audio_q.put_nowait(pcm)
        except asyncio.QueueFull:
            state.dropped += 1

    _install_signal_handlers(state)

    print("\nListening... Speak Japanese. Press Ctrl+C to stop.\n")
    print("-" * 60)

    stream = sd.InputStream(
        device=args.device,
        samplerate=native_rate,
        channels=1,
        dtype="float32",
        blocksize=0,
        latency="high",
        callback=audio_callback,
    )

    meter_task = asyncio.create_task(meter(state, audio_q))
    worker_task = asyncio.create_task(
        worker(rec, vad, window, audio_q, state, output_file)
    )

    try:
        stream.start()
        await state.stop_event.wait()
    finally:
        # Order matters: stop the mic first so the queue stops growing, then
        # sentinel the worker and let it flush the VAD — a mid-utterance Ctrl+C
        # still decodes and persists what was already spoken (T1.4 behavior).
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        await audio_q.put(None)
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        meter_task.cancel()
        try:
            await meter_task
        except (asyncio.CancelledError, Exception):
            pass
        if output_file:
            output_file.close()
        sys.stdout.write("\n")


def _install_signal_handlers(state):
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, state.request_stop)
        except NotImplementedError:
            pass


def main():
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=sorted(ENGINE_DIRS),
        default="k2v2",
        help="Local STT engine (default: k2v2 = reazonspeech-k2-v2; see D-010).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Append transcriptions to a text file.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index (see --list-devices). Default: system default.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit.",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    err = check_models(args.engine)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"Engine: {args.engine} (local sherpa-onnx, no network)")

    try:
        asyncio.run(run_session(args))
    except KeyboardInterrupt:
        pass
    print("Stopped.")


if __name__ == "__main__":
    main()
