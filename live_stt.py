#!/usr/bin/env python3
"""Live Japanese speech-to-text with English translation via the Gemini Live API.

Opens a persistent bidirectional streaming session, pipes microphone audio to Gemini
as raw PCM16, and prints the model's JA/EN transcription of what it says back.
"""

import argparse
import asyncio
import os
import signal
import sys
from datetime import datetime

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
AUDIO_QUEUE_MAX = 100  # ~10s of 100ms blocks
RECONNECT_BACKOFF_MIN_S = 1.0
RECONNECT_BACKOFF_MAX_S = 30.0
RECONNECT_RESET_AFTER_S = 10.0  # Session stable for this long resets backoff

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


def build_config(sys_inst: str, handle: str | None) -> types.LiveConnectConfig:
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
        system_instruction=types.Content(parts=[types.Part(text=sys_inst)]),
        session_resumption=types.SessionResumptionConfig(handle=handle),
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


async def receiver(session, state, output_file, expect_en):
    """Consume server messages, emit JA/EN blocks on turn boundaries.

    The outer while defeats python-genai#1224, where session.receive() exits
    its async iterator on turn_complete. go_away and unexpected closes set
    should_reconnect so the outer run_session loop opens a new session.
    """
    buf = ""
    try:
        while not state.stopping and not state.should_reconnect:
            try:
                async for response in session.receive():
                    if response.go_away is not None:
                        state.should_reconnect = True
                        sys.stderr.write(
                            f"\n  [go_away, reconnecting "
                            f"(time_left={response.go_away.time_left})]\n"
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
                        emit_block(text, output_file, expect_en)
            except Exception as e:
                if state.stopping:
                    return
                sys.stderr.write(f"\n  [recv error: {e}]\n")
                state.should_reconnect = True
                return
    finally:
        # Flush any in-flight partial turn on shutdown so a mid-utterance Ctrl+C
        # still persists what the model already transcribed. Skip on reconnect:
        # the resumed session may re-emit the same turn.
        if state.stopping:
            tail = buf.strip()
            if tail and "[inaudible]" not in tail.lower():
                try:
                    emit_block(tail, output_file, expect_en)
                except Exception:
                    pass


def emit_block(text, output_file, expect_en):
    """Extract JA: and EN: lines from model's spoken output, display + persist."""
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
    sys.stdout.write("\r" + " " * 80 + "\r")
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


async def meter(state, transcribe_q):
    while not state.stopping:
        level = min(int(state.latest_rms / 0.05 * METER_WIDTH), METER_WIDTH)
        bar = "#" * level + " " * (METER_WIDTH - level)
        qsize = transcribe_q.qsize()
        pending = f" q={qsize}" if qsize > 0 else ""
        dropped = f" drop={state.dropped}" if state.dropped else ""
        status = "LIVE" if state.connected else "RECONNECT"
        rc = f" rc={state.reconnect_count}" if state.reconnect_count else ""
        sys.stdout.write(
            f"\r  [{bar}] {state.latest_rms:.4f} * {status}{rc}{pending}{dropped}"
        )
        sys.stdout.flush()
        await asyncio.sleep(METER_INTERVAL)


async def run_session(args, api_key):
    client = genai.Client(api_key=api_key)

    dev_info = sd.query_devices(args.device, kind="input")
    native_rate = int(dev_info["default_samplerate"])
    block_size = int(native_rate * BLOCK_DURATION)
    if args.device is not None:
        dev_label = f"#{args.device} {dev_info['name']}"
    else:
        dev_label = dev_info["name"]
    print(f"Mic: {dev_label} @ {native_rate} Hz (streaming at {SEND_RATE} Hz to Live API)")

    expect_en = not args.no_translate
    sys_inst = SYSTEM_INSTRUCTION_TRANSLATE if expect_en else SYSTEM_INSTRUCTION_TRANSCRIBE

    state = State()
    loop = asyncio.get_running_loop()
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAX)

    output_file = open(args.output, "a", encoding="utf-8") if args.output else None
    if output_file:
        print(f"Writing transcriptions to: {args.output}")

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

    _install_signal_handlers(state)

    print("\nListening... Speak Japanese. Press Ctrl+C to stop.\n")
    print("-" * 60)

    stream = sd.InputStream(
        device=args.device,
        samplerate=native_rate,
        channels=1,
        dtype="float32",
        blocksize=block_size,
        latency="high",
        callback=audio_callback,
    )

    # Meter lives outside the per-session TaskGroup so it survives reconnects.
    meter_task = asyncio.create_task(meter(state, audio_q))

    try:
        stream.start()
        backoff = RECONNECT_BACKOFF_MIN_S
        while not state.stopping:
            state.should_reconnect = False
            config = build_config(sys_inst, state.handle)
            connected_at = None
            try:
                async with client.aio.live.connect(
                    model=args.model, config=config
                ) as session:
                    state.connected = True
                    connected_at = loop.time()
                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(sender(session, audio_q, state))
                        tg.create_task(
                            receiver(session, state, output_file, expect_en)
                        )
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
            if connected_at is not None and (loop.time() - connected_at) >= RECONNECT_RESET_AFTER_S:
                backoff = RECONNECT_BACKOFF_MIN_S
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)
    finally:
        state.stopping = True
        meter_task.cancel()
        try:
            await meter_task
        except (asyncio.CancelledError, Exception):
            pass
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        if output_file:
            output_file.close()
        sys.stdout.write("\n")


def _install_signal_handlers(state):
    loop = asyncio.get_running_loop()

    def _handle():
        state.stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle)
        except NotImplementedError:
            pass


async def _wait_for_stop_or_reconnect(state):
    while not state.stopping and not state.should_reconnect:
        await asyncio.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-live-preview",
        help="Gemini Live model (default: gemini-3.1-flash-live-preview).",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Transcribe only (no English translation).",
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

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set the GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {args.model} (Live API)")

    try:
        asyncio.run(run_session(args, api_key))
    except KeyboardInterrupt:
        pass
    print("Stopped.")


if __name__ == "__main__":
    main()
