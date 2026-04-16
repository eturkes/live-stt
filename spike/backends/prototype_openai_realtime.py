"""OpenAI Realtime prototype, conforming to the harness Backend protocol.

Uses `AsyncOpenAI().realtime.connect(model=...)` (stable path per the
research doc) with `output_modalities=["text"]` so we never pay for
audio-out. JA transcript comes from the dedicated `gpt-4o-transcribe`
attached to the input audio; EN translation comes from the Realtime
model's own text response, steered by the system instruction.

One session per clip. No reconnect logic here — `live_stt.py` owns that.

Emits:
  Info("connected", ...)     after session.update succeeds
  Block(ja, en, t_first, t_final) once per turn (JA from transcription,
                                  EN from response.output_text when translating)
  Info("closed", ...)        on session exit
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncIterator

import numpy as np

from harness import Block, Err, Event, Info, SEND_RATE

DEFAULT_MODEL = "gpt-realtime-mini"
TRANSCRIBE_MODEL = "gpt-4o-transcribe"
TARGET_RATE = 24000  # OpenAI Realtime requires 24 kHz PCM16 input.

SYS_TRANSLATE = (
    "You are a live Japanese interpreter. You will hear continuous Japanese speech.\n"
    "For every distinct utterance, respond with exactly two lines and nothing else:\n"
    "JA: <verbatim Japanese transcription>\n"
    "EN: <natural English translation>\n"
    "If audio is unclear or silent, respond with a single line: [inaudible]\n"
    "Never add commentary."
)

SYS_TRANSCRIBE = (
    "You are a live Japanese transcriber. You will hear continuous Japanese speech.\n"
    "For every distinct utterance, respond with exactly one line:\n"
    "JA: <verbatim Japanese transcription>\n"
    "If audio is unclear or silent, respond with a single line: [inaudible]\n"
    "Never add commentary."
)


def _upsample_16_to_24(frame_16k_pcm16: bytes) -> bytes:
    """Linear-interpolate 16 kHz PCM16 mono to 24 kHz PCM16 mono.

    Matches `live_stt.resample()`'s np.interp approach. 100 ms @ 16 kHz
    (1600 samples) becomes 100 ms @ 24 kHz (2400 samples).
    """
    if not frame_16k_pcm16:
        return frame_16k_pcm16
    src = np.frombuffer(frame_16k_pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    ratio = TARGET_RATE / SEND_RATE
    n_out = int(len(src) * ratio)
    if n_out <= 0:
        return b""
    dst = np.interp(
        np.arange(n_out) / ratio,
        np.arange(len(src)),
        src,
    ).astype(np.float32)
    return (np.clip(dst, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _split_ja_en(text: str) -> tuple[str, str]:
    ja = en = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("JA:") and not ja:
            ja = s[3:].strip()
        elif s.startswith("EN:") and not en:
            en = s[3:].strip()
    if not ja:
        ja = text.strip().replace("\n", " ")
    return ja, en


async def stream(
    pcm_frames: AsyncIterator[bytes],
    *,
    translate: bool,
    api_key: str,
    model: str = DEFAULT_MODEL,
    **_,
) -> AsyncIterator[Event]:
    # Imports inside the function so this module loads cleanly when `openai`
    # isn't installed (e.g., during static analysis or collection).
    from openai import AsyncOpenAI

    t0 = time.monotonic()
    client = AsyncOpenAI(api_key=api_key)
    sys_inst = SYS_TRANSLATE if translate else SYS_TRANSCRIBE

    session_config: dict = {
        "type": "realtime",
        "output_modalities": ["text"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": TARGET_RATE},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 700,
                    "prefix_padding_ms": 300,
                },
                "transcription": {
                    "model": TRANSCRIBE_MODEL,
                    "language": "ja",
                },
                "noise_reduction": {"type": "near_field"},
            },
            "output": {},
        },
        "instructions": sys_inst,
    }

    queue: asyncio.Queue[Event] = asyncio.Queue()
    sender_done = asyncio.Event()

    async def sender(conn):
        try:
            async for frame in pcm_frames:
                up = _upsample_16_to_24(frame)
                if not up:
                    continue
                b64 = base64.b64encode(up).decode("ascii")
                await conn.input_audio_buffer.append(audio=b64)
        except Exception as e:
            await queue.put(Err(
                f"openai sender: {type(e).__name__}: {e}",
                time.monotonic() - t0,
            ))
        finally:
            sender_done.set()

    async def receiver(conn):
        # JA arrives from the separate transcription model (input_audio_transcription.*).
        # EN (when translating) arrives in response.output_text.*. Turns finalize on
        # response.done when translating, else on transcription.completed.
        state = {"ja": "", "en": "", "t_first": None}

        def mark(t: float):
            if state["t_first"] is None:
                state["t_first"] = t

        def reset():
            state["ja"] = state["en"] = ""
            state["t_first"] = None

        try:
            async for ev in conn:
                et = getattr(ev, "type", "")
                now = time.monotonic() - t0

                if et == "conversation.item.input_audio_transcription.delta":
                    d = getattr(ev, "delta", "") or ""
                    if d:
                        mark(now)
                        state["ja"] += d
                elif et == "conversation.item.input_audio_transcription.completed":
                    tr = getattr(ev, "transcript", "") or ""
                    if tr:
                        mark(now)
                        state["ja"] = tr
                    if not translate:
                        await _emit_turn(queue, state["ja"], "", state["t_first"], now, translate)
                        reset()
                elif et == "response.output_text.delta":
                    d = getattr(ev, "delta", "") or ""
                    if d:
                        mark(now)
                        state["en"] += d
                elif et == "response.output_text.done":
                    state["en"] = getattr(ev, "text", "") or state["en"]
                elif et == "response.done":
                    if translate:
                        await _emit_turn(queue, state["ja"], state["en"], state["t_first"], now, translate)
                        reset()
                    else:
                        state["en"] = ""
                elif et == "error":
                    err = getattr(ev, "error", None)
                    msg = getattr(err, "message", None) or repr(err)
                    await queue.put(Err(f"openai error: {msg}", now))

                if sender_done.is_set() and not state["ja"] and not state["en"]:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await queue.put(Err(
                f"openai recv: {type(e).__name__}: {e}",
                time.monotonic() - t0,
            ))

    try:
        async with client.realtime.connect(model=model) as conn:
            await conn.session.update(session=session_config)
            await queue.put(Info("connected", model, time.monotonic() - t0))

            async with asyncio.TaskGroup() as tg:
                send_task = tg.create_task(sender(conn))
                recv_task = tg.create_task(receiver(conn))
                async for ev in _yield_from(queue, send_task, recv_task):
                    yield ev

            await queue.put(Info("closed", "session ended", time.monotonic() - t0))
            # Flush the closed event.
            while not queue.empty():
                yield queue.get_nowait()
    except Exception as e:
        yield Err(
            f"openai connect: {type(e).__name__}: {e}",
            time.monotonic() - t0,
        )


async def _emit_turn(queue, ja, en, t_first, t_final, translate):
    ja = (ja or "").strip()
    en = (en or "").strip()
    # Model may return both lines in the response blob instead of the
    # transcription stream — recover the split if so.
    if translate and en and "JA:" in en and "EN:" in en:
        split_ja, split_en = _split_ja_en(en)
        ja = ja or split_ja
        en = split_en
    if not ja or "[inaudible]" in ja.lower():
        return
    if translate and "[inaudible]" in en.lower():
        en = ""
    await queue.put(Block(
        ja=ja, en=en if translate else "",
        t_first=t_first if t_first is not None else t_final,
        t_final=t_final,
    ))


async def _yield_from(queue: asyncio.Queue, *tasks):
    while True:
        done = all(t.done() for t in tasks)
        try:
            ev = await asyncio.wait_for(queue.get(), timeout=0.2)
            yield ev
        except (TimeoutError, asyncio.TimeoutError):
            if done and queue.empty():
                return
