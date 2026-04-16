#!/usr/bin/env python3
"""Prototype A — baseline long-session support.

Live STT with session resumption + context-window compression + goAway-driven
reconnect. No client-side memory. This is the control for the T3.2 benchmark.

Differences from main live_stt.py:
- Receiver wraps session.receive() in an outer `while` to defeat python-genai#1224.
- Outer `while not state.stopping` loop in run_session opens a new Live session
  on every go_away or unexpected close.
- SessionResumptionConfig(transparent=True) + ContextWindowCompressionConfig
  enabled. Handle stored on State and passed to the next connect().
- Audio input stream stays open across reconnects; the bounded audio_q buffers
  across the gap (up to AUDIO_QUEUE_MAX blocks, ~10 s).
- For benchmark injection: pass client_factory= to run_session to swap in a mock.
"""

import argparse
import asyncio
import os
import signal
import sys

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

SEND_RATE = 16000
BLOCK_DURATION = 0.1
METER_WIDTH = 40
METER_INTERVAL = 0.1
AUDIO_QUEUE_MAX = 100
RECONNECT_BACKOFF_S = 1.0

SYSTEM_INSTRUCTION_TRANSLATE = (
    "You are a live Japanese interpreter. You will hear continuous Japanese speech.\n"
    "For every distinct utterance, respond by speaking exactly two lines and nothing else:\n"
    "JA: <verbatim Japanese transcription using appropriate kanji/hiragana/katakana>\n"
    "EN: <natural English translation>\n"
    "If audio is unclear or silent, respond with a single line: [inaudible]\n"
    "Never add commentary, greetings, or any other text."
)

SYSTEM_INSTRUCTION_TRANSCRIBE = (
    "You are a live Japanese transcriber. You will hear continuous Japanese speech.\n"
    "For every distinct utterance, respond by speaking exactly one line and nothing else:\n"
    "JA: <verbatim Japanese transcription using appropriate kanji/hiragana/katakana>\n"
    "If audio is unclear or silent, respond with a single line: [inaudible]\n"
    "Never add commentary, greetings, or any other text."
)


def resample(audio, orig_rate, target_rate):
    if orig_rate == target_rate:
        return audio
    ratio = target_rate / orig_rate
    n_samples = int(len(audio) * ratio)
    indices = np.arange(n_samples) / ratio
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def pcm16_bytes(audio_f32):
    return (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class State:
    def __init__(self):
        self.latest_rms = 0.0
        self.dropped = 0
        self.stopping = False
        self.connected = False
        self.handle: str | None = None
        self.should_reconnect = False
        self.reconnect_count = 0
        # Benchmark instrumentation (ignored in prod).
        self.transcripts_emitted = 0


def build_config(sys_inst: str, handle: str | None) -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=sys_inst)]),
        session_resumption=types.SessionResumptionConfig(handle=handle, transparent=True),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        ),
    )


async def sender(session, audio_q, state):
    while True:
        chunk = await audio_q.get()
        if chunk is None:
            break
        try:
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SEND_RATE}")
            )
        except Exception as e:
            if not state.stopping and not state.should_reconnect:
                sys.stderr.write(f"\n  [send error: {e}]\n")
            break


async def receiver(session, state, print_lock, output_file, expect_en):
    buf = ""
    while not state.stopping and not state.should_reconnect:
        try:
            async for response in session.receive():
                if response.go_away is not None:
                    state.should_reconnect = True
                    sys.stderr.write(
                        f"\n  [go_away, reconnecting (time_left={response.go_away.time_left})]\n"
                    )
                    return
                if response.session_resumption_update is not None:
                    u = response.session_resumption_update
                    if u.resumable and u.new_handle:
                        state.handle = u.new_handle
                if response.server_content is None:
                    continue
                sc = response.server_content
                if sc.output_transcription and sc.output_transcription.text:
                    buf += sc.output_transcription.text
                if sc.turn_complete or sc.generation_complete:
                    text = buf.strip()
                    buf = ""
                    if not text or "[inaudible]" in text.lower():
                        continue
                    emit_block(text, print_lock, output_file, expect_en, state)
        except Exception as e:
            if state.stopping:
                return
            sys.stderr.write(f"\n  [recv error: {e}]\n")
            state.should_reconnect = True
            return


def emit_block(text, print_lock, output_file, expect_en, state):
    ja_line = ""
    en_line = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("JA:") and not ja_line:
            ja_line = s
        elif s.startswith("EN:") and not en_line:
            en_line = s
    if not ja_line:
        ja_line = "JA: " + text.replace("\n", " ").strip()
    lines = [ja_line]
    if expect_en and en_line:
        lines.append(en_line)
    state.transcripts_emitted += 1
    sys.stdout.write("\r" + " " * 80 + "\r")
    for line in lines:
        print(f"  {line}")
    print("-" * 60)
    if output_file:
        for line in lines:
            output_file.write(line + "\n")
        output_file.write("\n")
        output_file.flush()


async def meter(state, print_lock, audio_q):
    while not state.stopping:
        level = min(int(state.latest_rms / 0.05 * METER_WIDTH), METER_WIDTH)
        bar = "#" * level + " " * (METER_WIDTH - level)
        qsize = audio_q.qsize()
        pending = f" q={qsize}" if qsize > 0 else ""
        dropped = f" drop={state.dropped}" if state.dropped else ""
        status = "LIVE" if state.connected else "RECONNECT"
        rc = f" rc={state.reconnect_count}" if state.reconnect_count else ""
        sys.stdout.write(
            f"\r  [{bar}] {state.latest_rms:.4f} * {status}{rc}{pending}{dropped}"
        )
        sys.stdout.flush()
        await asyncio.sleep(METER_INTERVAL)


async def _wait_for_stop_or_reconnect(state):
    while not state.stopping and not state.should_reconnect:
        await asyncio.sleep(0.05)


async def _install_signal_handlers(state):
    loop = asyncio.get_running_loop()

    def _handle():
        state.stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle)
        except NotImplementedError:
            pass


async def run_session(args, api_key, client_factory=None, stream_factory=None):
    """Run a live session loop. Factories allow test injection."""
    if client_factory is None:
        client = genai.Client(api_key=api_key)
    else:
        client = client_factory(api_key)

    expect_en = not args.no_translate
    sys_inst = SYSTEM_INSTRUCTION_TRANSLATE if expect_en else SYSTEM_INSTRUCTION_TRANSCRIBE

    state = State()
    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAX)
    print_lock = asyncio.Lock()

    output_file = open(args.output, "a", encoding="utf-8") if args.output else None
    if output_file:
        print(f"Writing transcriptions to: {args.output}")

    if stream_factory is None:
        dev_info = sd.query_devices(kind="input")
        native_rate = int(dev_info["default_samplerate"])
        block_size = int(native_rate * BLOCK_DURATION)
        print(f"Mic: {native_rate} Hz (streaming at {SEND_RATE} Hz to Live API)")

        def audio_callback(indata, frames, time_info, status):
            if status:
                sys.stderr.write(f"\n  audio: {status}\n")
            mono = indata[:, 0] if indata.ndim > 1 else indata
            state.latest_rms = float(np.sqrt(np.mean(mono**2)))
            down = resample(mono, native_rate, SEND_RATE)
            pcm = pcm16_bytes(down)

            def _put():
                try:
                    audio_q.put_nowait(pcm)
                except asyncio.QueueFull:
                    state.dropped += 1

            loop.call_soon_threadsafe(_put)

        stream = sd.InputStream(
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            latency="high",
            callback=audio_callback,
        )
    else:
        stream = stream_factory(state, audio_q, loop)

    await _install_signal_handlers(state)

    print("\nListening... Speak Japanese. Press Ctrl+C to stop.\n")
    print("-" * 60)

    try:
        stream.start()
        while not state.stopping:
            state.should_reconnect = False
            config = build_config(sys_inst, state.handle)
            try:
                async with client.aio.live.connect(
                    model=args.model, config=config
                ) as session:
                    state.connected = True
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(sender(session, audio_q, state))
                        tg.create_task(
                            receiver(session, state, print_lock, output_file, expect_en)
                        )
                        tg.create_task(meter(state, print_lock, audio_q))
                        await _wait_for_stop_or_reconnect(state)
                        try:
                            await session.send_realtime_input(audio_stream_end=True)
                        except Exception:
                            pass
                        try:
                            audio_q.put_nowait(None)
                        except asyncio.QueueFull:
                            pass
            except* Exception as eg:
                for e in eg.exceptions:
                    sys.stderr.write(f"\n  [session error: {type(e).__name__}: {e}]\n")
            state.connected = False
            if state.stopping:
                break
            state.reconnect_count += 1
            await asyncio.sleep(RECONNECT_BACKOFF_S)
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        if output_file:
            output_file.close()
        sys.stdout.write("\n")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-3.1-flash-live-preview")
    parser.add_argument("--no-translate", action="store_true")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set the GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model} (Live API, prototype A)")
    try:
        asyncio.run(run_session(args, api_key))
    except KeyboardInterrupt:
        pass
    print("Stopped.")


if __name__ == "__main__":
    main()
