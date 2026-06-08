"""Local CPU STT prototype: sherpa-onnx offline recognizer + silero VAD.

Chunked-streaming pattern (no native frame-sync open JA model exists as of
June 2026): silero VAD segments utterances; each completed segment is decoded
by a sherpa-onnx offline model in a thread-pool executor. The frame reader
never blocks on decode — mirrors a live mic where capture can't stall.

Engines (select via `engine=` kwarg from bench.py BACKENDS extras):
    k2v2     — reazonspeech-k2-v2 zipformer transducer, int8 encoder
    parakeet — nvidia parakeet-tdt_ctc-0.6b-ja, int8, punctuated

`api_key` is accepted and ignored (harness contract). `translate` is ignored:
the translation leg moves to the Codex surface (T4.2/T4.4), so Blocks carry
ja only.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
from harness import Block, Err, Event, Info

MODELS = Path(__file__).resolve().parents[2] / "models"
K2V2_DIR = MODELS / "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
PARAKEET_DIR = MODELS / "sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8"
VAD_MODEL = MODELS / "silero_vad.onnx"

SR = 16000
NUM_THREADS = 4
# 2 s gap in the `paused` clip must split into two segments; sub-second
# pauses within a sentence must not.
VAD_MIN_SILENCE_S = 0.5
VAD_MIN_SPEECH_S = 0.25
# Silero opens segments 0.2-0.7 s after true speech onset (no pad field in
# sherpa's config) — clipped こんにちは to はい. Re-slice each segment from
# the fed-sample buffer with this much lead-in; reaching back into silence
# is harmless, and run-on speech wouldn't have split anyway.
VAD_PRE_PAD_S = 0.4

# Model load is seconds; bench.py creates a fresh stream() per clip, so cache
# recognizers per engine for the life of the process.
_RECOGNIZERS: dict[str, sherpa_onnx.OfflineRecognizer] = {}


def _load_recognizer(engine: str) -> sherpa_onnx.OfflineRecognizer:
    rec = _RECOGNIZERS.get(engine)
    if rec is not None:
        return rec
    if engine == "k2v2":
        # int8 encoder + fp32 decoder/joiner ≈ fp32 CER (HILab table).
        rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(K2V2_DIR / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(K2V2_DIR / "decoder-epoch-99-avg-1.onnx"),
            joiner=str(K2V2_DIR / "joiner-epoch-99-avg-1.onnx"),
            tokens=str(K2V2_DIR / "tokens.txt"),
            num_threads=NUM_THREADS,
        )
    elif engine == "parakeet":
        rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(PARAKEET_DIR / "model.int8.onnx"),
            tokens=str(PARAKEET_DIR / "tokens.txt"),
            num_threads=NUM_THREADS,
        )
    else:
        raise ValueError(f"unknown engine {engine!r}")
    _RECOGNIZERS[engine] = rec
    return rec


def _make_vad() -> tuple[sherpa_onnx.VoiceActivityDetector, int]:
    """Returns (vad, window_size_in_samples)."""
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(VAD_MODEL)
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = VAD_MIN_SILENCE_S
    cfg.silero_vad.min_speech_duration = VAD_MIN_SPEECH_S
    cfg.sample_rate = SR
    window = int(cfg.silero_vad.window_size)
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60), window


def _decode(rec: sherpa_onnx.OfflineRecognizer, samples: np.ndarray) -> str:
    s = rec.create_stream()
    s.accept_waveform(SR, samples)
    rec.decode_stream(s)
    return s.result.text.strip()


async def stream(pcm_frames, *, translate, api_key, engine="k2v2", **kwargs):
    """Harness contract: async generator of Events over 100 ms PCM16 frames."""
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    try:
        rec = await loop.run_in_executor(None, _load_recognizer, engine)
    except Exception as e:
        yield Err(f"model load: {type(e).__name__}: {e}", time.monotonic() - t0)
        return
    vad, window = _make_vad()
    yield Info("connected", f"engine={engine}", time.monotonic() - t0)

    raw_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    out_q: asyncio.Queue[Event | None] = asyncio.Queue()

    async def reader():
        # Pull frames at the feeder's pace; never blocked by decode.
        async for f in pcm_frames:
            raw_q.put_nowait(f)
        raw_q.put_nowait(None)

    async def worker():
        buf = np.empty(0, dtype=np.float32)
        fed: list[np.ndarray] = []   # everything pushed into the VAD, for pre-pad re-slicing
        fed_len = 0
        pad = int(VAD_PRE_PAD_S * SR)

        def fed_slice(a: int, b: int) -> np.ndarray:
            return np.concatenate(fed)[a:b] if fed else np.empty(0, dtype=np.float32)
            # O(total) per call — fine for ≤15 s bench clips; production
            # (T4.3) wants a bounded ring buffer instead.

        try:
            while True:
                chunk = await raw_q.get()
                flush = chunk is None
                if not flush:
                    f32 = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    buf = np.concatenate([buf, f32])
                    while len(buf) >= window:
                        vad.accept_waveform(buf[:window])
                        fed.append(buf[:window])
                        fed_len += window
                        buf = buf[window:]
                else:
                    if len(buf):
                        vad.accept_waveform(buf)
                        fed.append(buf)
                        fed_len += len(buf)
                    vad.flush()
                while not vad.empty():
                    start = int(vad.front.start)
                    n = len(vad.front.samples)
                    vad.pop()
                    seg = fed_slice(max(0, start - pad), start + n)
                    text = await loop.run_in_executor(None, _decode, rec, seg)
                    t = time.monotonic() - t0
                    if text:
                        out_q.put_nowait(Block(ja=text, en="", t_first=t, t_final=t))
                if flush:
                    break
        except Exception as e:
            out_q.put_nowait(Err(f"worker: {type(e).__name__}: {e}", time.monotonic() - t0))
        finally:
            out_q.put_nowait(None)

    tasks = [asyncio.create_task(reader()), asyncio.create_task(worker())]
    try:
        while (ev := await out_q.get()) is not None:
            yield ev
    finally:
        for t in tasks:
            t.cancel()
    yield Info("closed", "", time.monotonic() - t0)
