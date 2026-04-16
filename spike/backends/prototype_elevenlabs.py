"""ElevenLabs Scribe v2 Realtime backend, conforming to the harness protocol.

Uses raw `websockets` since elevenlabs-python has no first-class STT-realtime
helper (April 2026). ElevenLabs Scribe does not translate — we cascade to the
shared `translate.translate` helper when the caller asks for EN.

Commit strategy: `manual`. The harness feeds one clip per run, so we send a
single explicit commit after the frames iterator is exhausted. The research
doc flags VAD mode turn-boundary bugs and `committed_transcript` arriving many
seconds late; one-turn-per-clip sidesteps both.

Reconnect on mid-stream WS drops is a known issue (livekit/agents #4609) but
is out of scope for a prototype — the harness runs short clips.

Emits:
  Info("connected", ...)        on WS open
  Block(ja, en, t_first, t_final) on each committed_transcript
  Info("closed", ...)           on session exit
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator

try:
    import websockets
except ImportError:  # pragma: no cover — guarded so the module imports cleanly
    websockets = None  # type: ignore[assignment]

from harness import Block, Err, Event, Info
from translate import translate as mt_translate

DEFAULT_MODEL = "scribe-v2-realtime"
WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


def _api_model_id(model: str) -> str:
    # Research doc shows the API expects `scribe_v2_realtime` (underscores).
    # Accept the harness-style hyphenated default and normalize.
    return model.replace("-", "_")


async def stream(
    pcm_frames: AsyncIterator[bytes],
    *,
    translate: bool,
    api_key: str,
    model: str = DEFAULT_MODEL,
    **_,
) -> AsyncIterator[Event]:
    if websockets is None:
        yield Err("websockets not installed", 0.0)
        return

    t0 = time.monotonic()
    api_model = _api_model_id(model)

    # Query params per research §2.5.
    url = (
        f"{WS_URL}"
        f"?model_id={api_model}"
        f"&language_code=ja"
        f"&audio_format=pcm_16000"
        f"&commit_strategy=manual"
    )
    headers = {"xi-api-key": api_key}

    queue: asyncio.Queue[Event] = asyncio.Queue()
    sender_done = asyncio.Event()
    recv_done = asyncio.Event()

    async def sender(ws):
        # TODO: research doc shows JSON `input_audio_chunk` with base64 audio;
        # ElevenLabs may also accept raw binary frames. Using documented JSON
        # form to stay on the safe side — verify against current API docs.
        try:
            async for frame in pcm_frames:
                msg = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(frame).decode("ascii"),
                }
                await ws.send(json.dumps(msg))
            # Manual commit: tell the server this turn is done and force a
            # final `committed_transcript`.
            # TODO: exact commit message name not pinned in research doc.
            # `commit` is the most commonly documented form; confirm vs. API.
            await ws.send(json.dumps({"message_type": "commit"}))
        except Exception as e:
            await queue.put(Err(f"elevenlabs sender: {e}", time.monotonic() - t0))
        finally:
            sender_done.set()

    async def receiver(ws):
        turn_t_first: float | None = None
        try:
            async for raw in ws:
                # Server frames are JSON per research §2.5.
                try:
                    evt = json.loads(raw)
                except (TypeError, ValueError):
                    continue

                mtype = evt.get("message_type") or evt.get("type") or ""

                if mtype == "partial_transcript":
                    if turn_t_first is None and evt.get("text"):
                        turn_t_first = time.monotonic() - t0
                    # Harness ignores Partials for timing; skip emission to
                    # keep the stream quiet.
                    continue

                if mtype == "committed_transcript":
                    ja = (evt.get("text") or "").strip()
                    t_final = time.monotonic() - t0
                    if not ja:
                        turn_t_first = None
                        continue
                    en = ""
                    if translate:
                        try:
                            mt = await mt_translate(ja)
                            en = mt.en
                        except Exception as e:
                            await queue.put(Err(
                                f"elevenlabs translate: {e}",
                                time.monotonic() - t0,
                            ))
                    await queue.put(Block(
                        ja=ja, en=en,
                        t_first=turn_t_first or t_final,
                        t_final=t_final,
                    ))
                    turn_t_first = None
                    # One clip, one commit, one block — we're done.
                    if sender_done.is_set():
                        return
                    continue

                if mtype == "error":
                    await queue.put(Err(
                        f"elevenlabs server: {evt.get('error') or evt}",
                        time.monotonic() - t0,
                    ))
                    continue

                # Unknown but harmless; surface as Info for visibility.
                if mtype:
                    await queue.put(Info(
                        f"elevenlabs:{mtype}",
                        json.dumps(evt)[:120],
                        time.monotonic() - t0,
                    ))
        except Exception as e:
            await queue.put(Err(f"elevenlabs recv: {e}", time.monotonic() - t0))
        finally:
            recv_done.set()

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            await queue.put(Info("connected", model, time.monotonic() - t0))

            send_task = asyncio.create_task(sender(ws))
            recv_task = asyncio.create_task(receiver(ws))

            try:
                async for ev in _yield_from(queue, send_task, recv_task):
                    yield ev
            finally:
                for t in (send_task, recv_task):
                    if not t.done():
                        t.cancel()
                # Drain exceptions to avoid "never retrieved" warnings.
                for t in (send_task, recv_task):
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

        yield Info("closed", "session ended", time.monotonic() - t0)
    except Exception as e:
        yield Err(f"elevenlabs connect: {e}", time.monotonic() - t0)


async def _yield_from(
    queue: asyncio.Queue,
    *tasks: asyncio.Task,
) -> AsyncIterator[Event]:
    while True:
        done = all(t.done() for t in tasks)
        try:
            ev = await asyncio.wait_for(queue.get(), timeout=0.2)
            yield ev
        except (TimeoutError, asyncio.TimeoutError):
            if done and queue.empty():
                return
