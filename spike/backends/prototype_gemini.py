"""Gemini Live baseline, conforming to the harness Backend protocol.

Strips the mic/meter/reconnect concerns from live_stt.py — those are live_stt.py's
job. Here we run one session per clip.

Shutdown: once the audio iterator is exhausted we send `audio_stream_end`, give
the server a grace period (GRACE_S) for any final turn to arrive, then cancel
the receiver. Without this, `session.receive()` blocks indefinitely on an idle
session and the harness hits its runner timeout.

Emits:
  Info("connected", model, t)
  Block(ja, en, t_first, t_final) per completed turn
  Info("closed", reason, t)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from harness import Block, Err, Event, Info, SEND_RATE

DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
GRACE_S = 3.0   # how long to wait for server to finalize turns after audio_stream_end

SYS_TRANSLATE = (
    "You are a live Japanese interpreter. You will hear continuous Japanese speech.\n"
    "For every distinct utterance, respond by speaking exactly two lines and nothing else:\n"
    "JA: <verbatim Japanese transcription>\n"
    "EN: <natural English translation>\n"
    "If audio is unclear or silent, respond with a single line: [inaudible]\n"
    "Never add commentary."
)

SYS_TRANSCRIBE = (
    "You are a live Japanese transcriber. You will hear continuous Japanese speech.\n"
    "For every distinct utterance, respond by speaking exactly one line:\n"
    "JA: <verbatim Japanese transcription>\n"
    "If audio is unclear or silent, respond with a single line: [inaudible]\n"
    "Never add commentary."
)


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
    t0 = time.monotonic()
    client = genai.Client(api_key=api_key)

    sys_inst = SYS_TRANSLATE if translate else SYS_TRANSCRIBE
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=sys_inst)]),
    )

    queue: asyncio.Queue[Event | None] = asyncio.Queue()
    shutdown = asyncio.Event()

    async def sender(session):
        try:
            async for frame in pcm_frames:
                await session.send_realtime_input(
                    audio=types.Blob(data=frame, mime_type=f"audio/pcm;rate={SEND_RATE}")
                )
            try:
                await session.send_realtime_input(audio_stream_end=True)
            except Exception:
                pass
            # Let the server finalize remaining turns, then trigger shutdown.
            await asyncio.sleep(GRACE_S)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await queue.put(Err(f"gemini sender: {e}", time.monotonic() - t0))
        finally:
            shutdown.set()

    async def receiver(session):
        buf = ""
        turn_t_first: float | None = None
        try:
            while not shutdown.is_set():
                try:
                    async for resp in session.receive():
                        if shutdown.is_set():
                            break
                        if resp.server_content is None:
                            continue
                        sc = resp.server_content
                        tr = sc.output_transcription
                        if tr and tr.text:
                            if turn_t_first is None:
                                turn_t_first = time.monotonic() - t0
                            buf += tr.text
                        if sc.turn_complete or sc.generation_complete:
                            text = buf.strip()
                            buf = ""
                            t_final = time.monotonic() - t0
                            if text and "[inaudible]" not in text.lower():
                                ja, en = _split_ja_en(text)
                                await queue.put(Block(
                                    ja=ja, en=en,
                                    t_first=turn_t_first or t_final,
                                    t_final=t_final,
                                ))
                            turn_t_first = None
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    if not shutdown.is_set():
                        await queue.put(Err(f"gemini recv: {e}", time.monotonic() - t0))
                    return
        except asyncio.CancelledError:
            return

    async with client.aio.live.connect(model=model, config=config) as session:
        await queue.put(Info("connected", model, time.monotonic() - t0))

        send_task = asyncio.create_task(sender(session))
        recv_task = asyncio.create_task(receiver(session))

        try:
            while True:
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if ev is None:
                        break
                    yield ev
                except (TimeoutError, asyncio.TimeoutError):
                    if shutdown.is_set() and queue.empty():
                        break
        finally:
            # Cancel receiver cleanly; sender should already be done.
            if not recv_task.done():
                recv_task.cancel()
            for t in (send_task, recv_task):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    yield Info("closed", "session ended", time.monotonic() - t0)
