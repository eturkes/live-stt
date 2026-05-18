#!/usr/bin/env python3
"""Live Japanese speech-to-text with English translation via the Gemini Live API.

Opens a persistent bidirectional streaming session, pipes microphone audio to Gemini
as raw PCM16, and prints the model's JA/EN transcription of what it says back.
"""

import argparse
import asyncio
import logging
import math
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
METER_WIDTH = 40
METER_INTERVAL = 0.1
METER_FULL_SCALE_RMS = 0.05  # RMS that fills the bar
AUDIO_QUEUE_MAX = 100
RECONNECT_BACKOFF_MIN_S = 1.0
RECONNECT_BACKOFF_MAX_S = 30.0
RECONNECT_RESET_AFTER_S = 10.0  # Session stable for this long resets backoff

# Prebuilt meter bars indexed by fill level — avoids two string allocations per tick.
_METER_BARS = tuple("#" * i + " " * (METER_WIDTH - i) for i in range(METER_WIDTH + 1))
# Constant-folded scale factor: rms * _METER_SCALE truncates to bar level.
_METER_SCALE = METER_WIDTH / METER_FULL_SCALE_RMS
# ANSI: carriage-return + erase-line. Replaces the 80-space repaint in emit_block.
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


# key -> (idx_floor, idx_ceil, frac, y0, y1). y0 doubles as the output buffer.
# Returned buffer is reused across calls — callers must consume before the next call
# (which is the only use pattern: audio_callback hands it straight to pcm16_bytes).
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
            # by ~20x. astype(int16) downstream will copy, so no need to ascontiguousarray.
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


# Size-keyed scratch buffers so pcm16_bytes can scale/clip/cast without per-call
# allocations. Bounded so blocksize jitter can't leak memory.
_PCM16_FLOAT_BUF: dict[int, np.ndarray] = {}
_PCM16_INT16_BUF: dict[int, np.ndarray] = {}

# Pre-typed scalars: passing Python floats forces numpy to upcast the working
# array to float64 (slow). float32 scalars stay in the float32 pipeline.
_PCM16_SCALE = np.float32(32767.0)
_PCM16_HI = np.float32(32767.0)
_PCM16_LO = np.float32(-32767.0)


def pcm16_bytes(audio_f32):
    n = len(audio_f32)
    float_buf = _PCM16_FLOAT_BUF.get(n)
    if float_buf is None:
        if len(_PCM16_FLOAT_BUF) >= 8:
            _PCM16_FLOAT_BUF.clear()
            _PCM16_INT16_BUF.clear()
        float_buf = np.empty(n, dtype=np.float32)
        _PCM16_FLOAT_BUF[n] = float_buf
        _PCM16_INT16_BUF[n] = np.empty(n, dtype=np.int16)
    int16_buf = _PCM16_INT16_BUF[n]
    # np.minimum + np.maximum is ~30% faster than np.clip on small arrays — clip
    # carries extra branching for the dual-bound case while min/max are single-op
    # ufuncs that share the same SIMD kernel.
    np.multiply(audio_f32, _PCM16_SCALE, out=float_buf)
    np.minimum(float_buf, _PCM16_HI, out=float_buf)
    np.maximum(float_buf, _PCM16_LO, out=float_buf)
    np.copyto(int16_buf, float_buf, casting="unsafe")
    return int16_buf.tobytes()


class State:
    def __init__(self):
        # Mean-square only; sqrt is deferred to the meter render path so the audio
        # thread doesn't pay for it 10-200x per second.
        self.latest_ms = 0.0
        self.dropped = 0
        self.stopping = False
        self.connected = False
        self.handle: str | None = None
        self.should_reconnect = False
        self.reconnect_count = 0
        # Events shadow the booleans so awaiters wake immediately instead of polling.
        # Constructed lazily once a running loop exists.
        self.stop_event: asyncio.Event = None  # type: ignore[assignment]
        self.reconnect_event: asyncio.Event = None  # type: ignore[assignment]

    def request_stop(self):
        self.stopping = True
        if self.stop_event is not None:
            self.stop_event.set()

    def request_reconnect(self):
        self.should_reconnect = True
        if self.reconnect_event is not None:
            self.reconnect_event.set()


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
    mime = f"audio/pcm;rate={SEND_RATE}"
    q_empty = audio_q.empty
    q_get_nowait = audio_q.get_nowait
    # model_construct skips Pydantic validation (~45% faster than Blob(...))
    # for a value we already know is well-formed.
    blob_construct = types.Blob.model_construct
    send = session.send_realtime_input
    while True:
        chunk = await audio_q.get()
        if chunk is None:
            break
        # Coalesce any already-queued chunks so we make 1 WebSocket round-trip
        # instead of N when the queue has backed up (e.g. just after a reconnect).
        # empty()-then-get is ~5x cheaper than catching QueueEmpty; we're the only
        # consumer, so the check-then-take is race-free.
        sentinel = False
        if q_empty():
            data = chunk
        else:
            chunks = [chunk]
            while not q_empty():
                more = q_get_nowait()
                if more is None:
                    sentinel = True
                    break
                chunks.append(more)
            data = b"".join(chunks) if len(chunks) > 1 else chunk
        try:
            await send(audio=blob_construct(data=data, mime_type=mime))
        except Exception as e:
            if not state.stopping and not state.should_reconnect:
                logger.error("[send error: %s]", e)
            break
        if sentinel:
            break


async def receiver(session, state, output_file, expect_en):
    """Consume server messages, emit JA/EN blocks on turn boundaries.

    The outer while defeats python-genai#1224, where session.receive() exits
    its async iterator on turn_complete. go_away and unexpected closes set
    should_reconnect so the outer run_session loop opens a new session.
    """
    # List + join avoids the O(n^2) cost of repeated str concatenation on long turns.
    buf: list[str] = []
    try:
        while not state.stopping and not state.should_reconnect:
            try:
                async for response in session.receive():
                    if response.go_away is not None:
                        state.request_reconnect()
                        logger.info(
                            "[go_away, reconnecting (time_left=%s)]",
                            response.go_away.time_left,
                        )
                        return
                    if response.session_resumption_update is not None:
                        u = response.session_resumption_update
                        if u.resumable and u.new_handle:
                            state.handle = u.new_handle
                    if response.server_content is None:
                        continue
                    sc = response.server_content
                    ot = sc.output_transcription
                    if ot is not None and ot.text:
                        buf.append(ot.text)
                    if sc.turn_complete or sc.generation_complete:
                        text = "".join(buf).strip()
                        buf.clear()
                        if not text or "[inaudible]" in text.lower():
                            continue
                        emit_block(text, output_file, expect_en)
            except Exception as e:
                if state.stopping:
                    return
                logger.error("[recv error: %s]", e)
                state.request_reconnect()
                return
    finally:
        # Flush any in-flight partial turn on shutdown so a mid-utterance Ctrl+C
        # still persists what the model already transcribed. Skip on reconnect:
        # the resumed session may re-emit the same turn.
        if state.stopping:
            tail = "".join(buf).strip()
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
        status = "LIVE" if state.connected else "RECONNECT"
        rc = f" rc={state.reconnect_count}" if state.reconnect_count else ""
        write(f"\r  [{bars[level]}] {rms:.4f} * {status}{rc}{pending}{dropped}")
        flush()
        await sleep(interval)


async def run_session(args, api_key):
    client = genai.Client(api_key=api_key)

    dev_info = sd.query_devices(args.device, kind="input")
    native_rate = int(dev_info["default_samplerate"])
    if args.device is not None:
        dev_label = f"#{args.device} {dev_info['name']}"
    else:
        dev_label = dev_info["name"]
    print(f"Mic: {dev_label} @ {native_rate} Hz (streaming at {SEND_RATE} Hz to Live API)")

    expect_en = not args.no_translate
    sys_inst = SYSTEM_INSTRUCTION_TRANSLATE if expect_en else SYSTEM_INSTRUCTION_TRANSCRIBE

    state = State()
    state.stop_event = asyncio.Event()
    state.reconnect_event = asyncio.Event()
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
        pcm = pcm16_bytes(resample(mono, native_rate, SEND_RATE))
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

    # Meter lives outside the per-session TaskGroup so it survives reconnects.
    meter_task = asyncio.create_task(meter(state, audio_q))

    try:
        stream.start()
        backoff = RECONNECT_BACKOFF_MIN_S
        while not state.stopping:
            state.should_reconnect = False
            state.reconnect_event.clear()
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
                    logger.error("[session error: %s: %s]", type(e).__name__, e)
            state.connected = False
            if state.stopping:
                break
            state.reconnect_count += 1
            if connected_at is not None and (loop.time() - connected_at) >= RECONNECT_RESET_AFTER_S:
                backoff = RECONNECT_BACKOFF_MIN_S
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX_S)
    finally:
        state.request_stop()
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
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, state.request_stop)
        except NotImplementedError:
            pass


async def _wait_for_stop_or_reconnect(state):
    stop_task = asyncio.create_task(state.stop_event.wait())
    rc_task = asyncio.create_task(state.reconnect_event.wait())
    try:
        await asyncio.wait(
            (stop_task, rc_task), return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        for t in (stop_task, rc_task):
            if not t.done():
                t.cancel()


def main():
    _configure_logging()
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
