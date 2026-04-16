"""Azure AI Speech backend, conforming to the harness Backend protocol.

Uses the Azure Speech SDK's integrated translation: one WebSocket carries
JA recognition + EN translation per turn. When `translate=False` we swap in
the plain `SpeechRecognizer` and emit JA only.

The SDK is callback-based on its own thread pool; we bridge to asyncio by
having the callbacks `loop.call_soon_threadsafe(queue.put_nowait, ...)` into
an asyncio.Queue that a consumer task drains.

Emits:
  Info("connected", ...)          on session_started
  Block(ja, en, t_first, t_final) on each recognized turn
  Info("closed", ...)             on session_stopped
  Err(...)                        on canceled / setup failure
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator

from harness import Block, Err, Event, Info, SEND_RATE

try:
    import azure.cognitiveservices.speech as speechsdk  # type: ignore
except Exception:  # pragma: no cover - guarded so the module still imports
    speechsdk = None  # type: ignore[assignment]


DEFAULT_MODEL = ""  # Azure doesn't take a model id; region + language is the knob.


async def stream(
    pcm_frames: AsyncIterator[bytes],
    *,
    translate: bool,
    api_key: str,
    region: str | None = None,
    model: str = DEFAULT_MODEL,
    **_,
) -> AsyncIterator[Event]:
    t0 = time.monotonic()

    if speechsdk is None:
        yield Err("azure-cognitiveservices-speech not installed", time.monotonic() - t0)
        return

    region = region or os.environ.get("AZURE_SPEECH_REGION")
    if not region:
        yield Err("no azure region configured", time.monotonic() - t0)
        return
    if not api_key:
        yield Err("no azure api key configured", time.monotonic() - t0)
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Event] = asyncio.Queue()

    # Sentinel to signal the yield loop to exit cleanly.
    DONE = object()

    def put(ev: Event | object) -> None:
        """Thread-safe enqueue from SDK callbacks."""
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    # ---- Audio input: 16 kHz mono PCM16 fed by our sender task ----
    audio_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=SEND_RATE, bits_per_sample=16, channels=1
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    # ---- Recognizer: translation vs. plain STT ----
    recognizer: "speechsdk.Recognizer"
    if translate:
        trans_cfg = speechsdk.translation.SpeechTranslationConfig(
            subscription=api_key, region=region
        )
        trans_cfg.speech_recognition_language = "ja-JP"
        trans_cfg.add_target_language("en")
        recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=trans_cfg, audio_config=audio_config
        )
    else:
        speech_cfg = speechsdk.SpeechConfig(subscription=api_key, region=region)
        speech_cfg.speech_recognition_language = "ja-JP"
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_cfg, audio_config=audio_config
        )

    # ---- Per-turn first-token timing ----
    # `recognizing` fires on each partial; track the first one per turn.
    turn_state: dict[str, float | None] = {"t_first": None}

    def on_recognizing(_evt):
        if turn_state["t_first"] is None:
            turn_state["t_first"] = time.monotonic() - t0

    def on_recognized(evt):
        result = evt.result
        reason = result.reason
        # Constants exist only when the SDK is installed; dodge attribute lookup on None.
        ok = reason in {
            speechsdk.ResultReason.RecognizedSpeech,
            speechsdk.ResultReason.TranslatedSpeech,
        }
        if not ok:
            return
        ja = result.text or ""
        en = ""
        if translate:
            try:
                en = result.translations.get("en", "") or ""
            except Exception:
                en = ""
        if not ja and not en:
            turn_state["t_first"] = None
            return
        t_final = time.monotonic() - t0
        t_first = turn_state["t_first"] if turn_state["t_first"] is not None else t_final
        turn_state["t_first"] = None
        put(Block(ja=ja, en=en, t_first=t_first, t_final=t_final))

    def on_session_started(evt):
        put(Info("connected", getattr(evt, "session_id", "") or "session", time.monotonic() - t0))

    def on_session_stopped(_evt):
        put(Info("closed", "session_stopped", time.monotonic() - t0))
        put(DONE)

    def on_canceled(evt):
        detail = getattr(evt, "error_details", "") or getattr(evt, "reason", "")
        put(Err(f"azure canceled: {detail}", time.monotonic() - t0))
        put(DONE)

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.canceled.connect(on_canceled)

    # ---- Sender: push frames from the async iterator into the SDK's buffer ----
    sender_done = asyncio.Event()

    async def sender():
        try:
            async for frame in pcm_frames:
                # push_stream.write is non-blocking (buffers internally).
                push_stream.write(frame)
        except Exception as e:
            put(Err(f"azure sender: {type(e).__name__}: {e}", time.monotonic() - t0))
        finally:
            # Signal end-of-audio to the SDK; it will emit a final `recognized`
            # then `session_stopped`.
            try:
                push_stream.close()
            except Exception:
                pass
            sender_done.set()

    # ---- Kick off continuous recognition ----
    # The `_async` variants return a Future-like object we can await via
    # run_in_executor. Using the sync form in an executor is equivalent and
    # more portable across SDK versions.
    try:
        await loop.run_in_executor(None, recognizer.start_continuous_recognition)
    except Exception as e:
        yield Err(f"azure start: {type(e).__name__}: {e}", time.monotonic() - t0)
        return

    sender_task = asyncio.create_task(sender())

    # ---- Yield loop: drain the queue until DONE or everything quiesces ----
    # Grace period after audio end for trailing recognitions to flush.
    GRACE_S = 10.0
    stop_requested = False
    grace_deadline: float | None = None

    try:
        while True:
            timeout = 0.2
            try:
                item = await asyncio.wait_for(queue.get(), timeout=timeout)
            except (TimeoutError, asyncio.TimeoutError):
                item = None

            if item is DONE:
                break
            if isinstance(item, (Info, Block, Err)):
                yield item

            # After the sender finishes, start a grace window. If session_stopped
            # doesn't arrive in time, force-stop the recognizer so we exit.
            if sender_done.is_set() and not stop_requested:
                stop_requested = True
                grace_deadline = time.monotonic() + GRACE_S
                # Ask the SDK to flush and stop; it will still emit session_stopped.
                asyncio.create_task(
                    loop.run_in_executor(None, recognizer.stop_continuous_recognition)
                )

            if (
                stop_requested
                and grace_deadline is not None
                and time.monotonic() > grace_deadline
                and queue.empty()
            ):
                yield Info("closed", "grace timeout", time.monotonic() - t0)
                break
    finally:
        if not sender_task.done():
            sender_task.cancel()
            try:
                await sender_task
            except (asyncio.CancelledError, Exception):
                pass
        # Best-effort stop if we broke out before the SDK ack'd.
        try:
            await loop.run_in_executor(None, recognizer.stop_continuous_recognition)
        except Exception:
            pass
