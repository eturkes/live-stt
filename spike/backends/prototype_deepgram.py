"""Deepgram Nova-3 streaming backend, conforming to the harness Backend protocol.

One session per clip. Deepgram returns JA transcripts only; EN comes from the
shared cascade translator when `translate=True`.

Emits:
  Info("connected", ...)               on WS upgrade
  Block(ja, en, t_first, t_final)      per finalized utterance (speech_final)
  Info("closed", ...)                  on session exit
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from urllib.parse import urlencode

from harness import Block, Err, Event, Info, SEND_RATE

DEFAULT_MODEL = "nova-3"
DG_URL = "wss://api.deepgram.com/v1/listen"
KEEPALIVE_INTERVAL_S = 4.0  # Deepgram closes idle WS after ~10 s.


def _build_url(model: str) -> str:
    params = {
        "model": model,
        "language": "ja",
        "encoding": "linear16",
        "sample_rate": SEND_RATE,
        "channels": 1,
        "interim_results": "true",
        "endpointing": 300,
        "utterance_end_ms": 1000,
        "vad_events": "true",
        "smart_format": "true",
    }
    return f"{DG_URL}?{urlencode(params)}"


def _extract_transcript(msg: dict) -> str:
    """Pull the top alternative transcript out of a Results frame."""
    try:
        alt = msg["channel"]["alternatives"][0]
    except (KeyError, IndexError, TypeError):
        return ""
    return (alt.get("transcript") or "").strip()


async def stream(
    pcm_frames: AsyncIterator[bytes],
    *,
    translate: bool,
    api_key: str,
    model: str = DEFAULT_MODEL,
    **_,
) -> AsyncIterator[Event]:
    try:
        import websockets
    except ImportError as e:
        raise RuntimeError(f"websockets library required: {e}") from e

    t0 = time.monotonic()
    url = _build_url(model)
    headers = [("Authorization", f"Token {api_key}")]

    queue: asyncio.Queue[Event] = asyncio.Queue()
    sender_done = asyncio.Event()
    audio_sent = asyncio.Event()

    if translate:
        from translate import translate as mt_translate
    else:
        mt_translate = None

    async def sender(ws):
        """Pump PCM frames; send KeepAlive during silence gaps."""
        try:
            frames_iter = pcm_frames.__aiter__()
            while True:
                try:
                    frame = await asyncio.wait_for(
                        frames_iter.__anext__(), timeout=KEEPALIVE_INTERVAL_S
                    )
                except StopAsyncIteration:
                    break
                except (TimeoutError, asyncio.TimeoutError):
                    if audio_sent.is_set():
                        await ws.send(json.dumps({"type": "KeepAlive"}))
                    continue
                await ws.send(frame)
                audio_sent.set()
            # Tell Deepgram we're done; it flushes any final transcript.
            await ws.send(json.dumps({"type": "CloseStream"}))
        except Exception as e:
            await queue.put(Err(f"deepgram sender: {e}", time.monotonic() - t0))
        finally:
            sender_done.set()

    async def receiver(ws):
        """Parse Results frames; finalize on speech_final."""
        turn_t_first: float | None = None
        turn_text = ""
        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                kind = msg.get("type")
                if kind == "Results":
                    text = _extract_transcript(msg)
                    is_final = bool(msg.get("is_final"))
                    speech_final = bool(msg.get("speech_final"))

                    if text and turn_t_first is None:
                        turn_t_first = time.monotonic() - t0

                    if is_final and text:
                        turn_text = (turn_text + " " + text).strip() if turn_text else text

                    if speech_final and turn_text:
                        ja = turn_text
                        t_final = time.monotonic() - t0
                        t_first = turn_t_first if turn_t_first is not None else t_final
                        turn_text = ""
                        turn_t_first = None

                        en = ""
                        if mt_translate is not None:
                            try:
                                mt = await mt_translate(ja)
                                en = mt.en
                            except Exception as e:
                                await queue.put(
                                    Err(f"cascade: {e}", time.monotonic() - t0)
                                )

                        await queue.put(Block(
                            ja=ja, en=en, t_first=t_first, t_final=t_final,
                        ))

                elif kind == "UtteranceEnd":
                    # Fallback: endpoint silence gap without a speech_final flag.
                    if turn_text:
                        ja = turn_text
                        t_final = time.monotonic() - t0
                        t_first = turn_t_first if turn_t_first is not None else t_final
                        turn_text = ""
                        turn_t_first = None

                        en = ""
                        if mt_translate is not None:
                            try:
                                mt = await mt_translate(ja)
                                en = mt.en
                            except Exception as e:
                                await queue.put(
                                    Err(f"cascade: {e}", time.monotonic() - t0)
                                )

                        await queue.put(Block(
                            ja=ja, en=en, t_first=t_first, t_final=t_final,
                        ))

                elif kind == "Metadata":
                    await queue.put(Info(
                        "metadata",
                        str(msg.get("request_id", "")),
                        time.monotonic() - t0,
                    ))
                # SpeechStarted and other vendor events are ignored.
        except Exception as e:
            await queue.put(Err(f"deepgram recv: {e}", time.monotonic() - t0))

    async with websockets.connect(url, additional_headers=headers) as ws:
        await queue.put(Info("connected", model, time.monotonic() - t0))

        async with asyncio.TaskGroup() as tg:
            send_task = tg.create_task(sender(ws))
            recv_task = tg.create_task(receiver(ws))
            tg.create_task(_watcher(ws, send_task, sender_done))

            async for ev in _yield_from(queue, send_task, recv_task):
                yield ev

        await queue.put(Info("closed", "session ended", time.monotonic() - t0))


async def _watcher(ws, send_task: asyncio.Task, sender_done: asyncio.Event):
    """Close the WS once the sender has finished, so the receiver loop exits."""
    await sender_done.wait()
    # Give Deepgram a moment to flush the final Results frame.
    await asyncio.sleep(1.5)
    try:
        await ws.close()
    except Exception:
        pass


async def _yield_from(queue: asyncio.Queue, *tasks):
    while True:
        done = all(t.done() for t in tasks)
        try:
            ev = await asyncio.wait_for(queue.get(), timeout=0.2)
            yield ev
        except (TimeoutError, asyncio.TimeoutError):
            if done and queue.empty():
                return
