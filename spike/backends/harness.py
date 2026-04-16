"""Shared harness for the backends spike.

Each backend prototype exposes an async generator::

    async def stream(pcm_frames, *, translate, api_key, **kwargs) -> AsyncIterator[Event]

`pcm_frames` is an async iterator of 100 ms PCM16 @ 16 kHz mono chunks.
Events are one of {Partial, Block, Info, Err}.

This module provides:
- Event dataclasses + result aggregation.
- `feed_wav(path, speedup)` — pace a WAV file into 100 ms frames at wall-clock.
- `run_bench(...)` — drive a backend against one clip and produce a BenchResult.

The harness never assumes microphone. Replace the feeder in `live_stt.py` if
you adopt a backend; here we test the network half only.
"""

from __future__ import annotations

import asyncio
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEND_RATE = 16000
FRAME_MS = 100
BYTES_PER_FRAME = SEND_RATE * (FRAME_MS / 1000) * 2  # int16 mono


# -------- Events --------

@dataclass
class Partial:
    text: str
    is_ja: bool
    t: float


@dataclass
class Block:
    ja: str
    en: str
    t_first: float   # arrival time of first token for this turn
    t_final: float   # arrival time when the block was completed


@dataclass
class Info:
    kind: str        # "connected" | "go_away" | "reconnect" | "closed" | vendor-specific
    detail: str
    t: float


@dataclass
class Err:
    message: str
    t: float


Event = Partial | Block | Info | Err


# -------- Result --------

@dataclass
class BenchResult:
    backend: str
    clip_id: str
    clip_duration_s: float
    translate: bool
    # Timing (all seconds, monotonic relative to stream() invocation)
    connect_s: float | None = None
    audio_end_s: float | None = None
    first_block_s: float | None = None
    last_block_s: float | None = None
    # Derived
    ttft_s: float | None = None
    total_s: float | None = None
    n_blocks: int = 0
    # Transcripts
    ja_text: str = ""
    en_text: str = ""
    # Diagnostics
    infos: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Cost estimate (filled by bench.py from vendor pricing)
    est_cost_usd: float | None = None
    est_cost_per_hr_usd: float | None = None
    # Backend meta
    model: str = ""
    skipped_reason: str | None = None

    def derive(self):
        if self.audio_end_s is not None and self.first_block_s is not None:
            self.ttft_s = max(0.0, self.first_block_s - self.audio_end_s)
        if self.audio_end_s is not None and self.last_block_s is not None:
            self.total_s = max(0.0, self.last_block_s - self.audio_end_s)


# -------- WAV loader --------

def load_wav_16k_mono_pcm16(path: Path) -> tuple[bytes, float]:
    """Return (pcm_bytes, duration_s). Resamples to 16 kHz mono if needed."""
    with wave.open(str(path), "rb") as w:
        nchan = w.getnchannels()
        sr = w.getframerate()
        sw = w.getsampwidth()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
    else:
        raise ValueError(f"unsupported sample width {sw}")

    if nchan == 2:
        data = data.reshape(-1, 2).mean(axis=1)

    if sr != SEND_RATE:
        ratio = SEND_RATE / sr
        n_out = int(len(data) * ratio)
        data = np.interp(
            np.arange(n_out) / ratio,
            np.arange(len(data)),
            data,
        ).astype(np.float32)

    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    duration = len(pcm) / 2 / SEND_RATE
    return pcm, duration


# -------- Feeder --------

async def feed_wav(
    pcm: bytes,
    *,
    on_audio_end: Callable[[float], None] | None = None,
    speedup: float = 1.0,
    t0: float,
) -> AsyncIterator[bytes]:
    """Yield 100 ms PCM16 frames paced at wall-clock (or faster with speedup).

    Calls `on_audio_end(t_relative)` with the monotonic offset at which the
    last frame was yielded. This is the "stop speaking" timestamp the harness
    uses to compute TTFT.
    """
    frame_bytes = int(SEND_RATE * (FRAME_MS / 1000)) * 2
    n = len(pcm) // frame_bytes
    remainder = len(pcm) - n * frame_bytes
    sleep = (FRAME_MS / 1000) / max(speedup, 1e-6)

    for i in range(n):
        frame = pcm[i * frame_bytes:(i + 1) * frame_bytes]
        yield frame
        await asyncio.sleep(sleep)
    if remainder:
        # Pad the tail to a full frame so backends that require fixed-size
        # frames don't choke on a short one.
        last = pcm[n * frame_bytes:] + b"\x00" * (frame_bytes - remainder)
        yield last
        await asyncio.sleep(sleep)

    if on_audio_end is not None:
        on_audio_end(time.monotonic() - t0)


# -------- Runner --------

StreamFn = Callable[..., AsyncIterator[Event]]


async def run_bench(
    *,
    backend: str,
    stream_fn: StreamFn,
    clip_id: str,
    pcm: bytes,
    duration_s: float,
    translate: bool,
    api_key: str,
    model: str = "",
    speedup: float = 1.0,
    overall_timeout_s: float = 120.0,
    **backend_kwargs,
) -> BenchResult:
    """Drive one backend against one clip. Returns a populated BenchResult."""
    result = BenchResult(
        backend=backend,
        clip_id=clip_id,
        clip_duration_s=duration_s,
        translate=translate,
        model=model,
    )

    t0 = time.monotonic()
    audio_end_holder: dict[str, float] = {}

    def _on_end(t_rel: float):
        audio_end_holder["t"] = t_rel

    async def consume():
        frames = feed_wav(pcm, on_audio_end=_on_end, speedup=speedup, t0=t0)
        try:
            async for ev in stream_fn(
                frames,
                translate=translate,
                api_key=api_key,
                **backend_kwargs,
            ):
                t = ev.t if hasattr(ev, "t") else (time.monotonic() - t0)
                if isinstance(ev, Info):
                    if ev.kind == "connected" and result.connect_s is None:
                        result.connect_s = t
                    result.infos.append(f"{ev.kind}: {ev.detail}")
                elif isinstance(ev, Err):
                    result.errors.append(ev.message)
                elif isinstance(ev, Partial):
                    pass  # not used for timing; some vendors emit these
                elif isinstance(ev, Block):
                    if result.first_block_s is None:
                        result.first_block_s = ev.t_first
                    result.last_block_s = ev.t_final
                    result.n_blocks += 1
                    if ev.ja:
                        result.ja_text = (result.ja_text + "\n" + ev.ja).strip()
                    if ev.en:
                        result.en_text = (result.en_text + "\n" + ev.en).strip()
        except Exception as e:
            result.errors.append(f"runner: {type(e).__name__}: {e}")

    try:
        await asyncio.wait_for(consume(), timeout=overall_timeout_s)
    except (TimeoutError, asyncio.TimeoutError):
        result.errors.append(f"runner: timed out after {overall_timeout_s}s")

    result.audio_end_s = audio_end_holder.get("t")
    result.derive()
    return result


# -------- Helpers for prototypes --------

async def drain_frames(
    frames: AsyncIterator[bytes],
    consumer: Callable[[bytes], Awaitable[None]],
) -> None:
    """Pump frames from the harness into a backend-specific consumer.

    Useful for prototypes that want to split reading frames from processing
    them (e.g., push to a WebSocket in one task while receive runs in another).
    """
    async for f in frames:
        await consumer(f)


def summarize(result: BenchResult) -> str:
    lines = [
        f"backend={result.backend} clip={result.clip_id} "
        f"dur={result.clip_duration_s:.2f}s translate={result.translate}",
    ]
    if result.skipped_reason:
        lines.append(f"  SKIPPED: {result.skipped_reason}")
        return "\n".join(lines)
    if result.connect_s is not None:
        lines.append(f"  connect={result.connect_s:.2f}s")
    if result.ttft_s is not None:
        lines.append(f"  ttft={result.ttft_s:.2f}s total={result.total_s:.2f}s "
                     f"blocks={result.n_blocks}")
    if result.est_cost_per_hr_usd is not None:
        lines.append(f"  cost/hr=${result.est_cost_per_hr_usd:.3f}")
    if result.ja_text:
        lines.append(f"  JA: {result.ja_text.splitlines()[0][:80]}")
    if result.en_text:
        lines.append(f"  EN: {result.en_text.splitlines()[0][:80]}")
    if result.errors:
        lines.append(f"  errors: {'; '.join(result.errors[:3])}")
    return "\n".join(lines)
