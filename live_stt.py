#!/usr/bin/env python3
"""Live Japanese speech-to-text with English translation.

Listens to your microphone, transcribes Japanese speech using Gemini,
and displays the Japanese transcription with an English translation.
"""

import io
import os
import sys
import wave
import queue
import argparse
import threading

import numpy as np
import sounddevice as sd
from google import genai

SEND_RATE = 16000  # downsample to 16kHz before sending
BLOCK_DURATION = 0.1  # seconds per audio block
SILENCE_DURATION = 0.8  # seconds of silence to trigger transcription
MIN_SPEECH_DURATION = 0.3  # minimum speech length to transcribe
MAX_CHUNK_DURATION = 5.0  # force-send after this many seconds
OVERLAP_DURATION = 1.0  # overlap between force-cut chunks (seconds)
METER_WIDTH = 40
NUM_WORKERS = 5
CONTEXT_SIZE = 3  # number of previous transcriptions to include as context

PROMPT_TRANSLATE = (
    "You are a Japanese speech transcription and translation assistant.\n"
    "Listen to this audio clip of spoken Japanese.\n"
    "Transcribe EXACTLY what is said in Japanese (use kanji/hiragana/katakana as appropriate).\n"
    "Then provide a natural English translation.\n"
    "If the audio is unclear or silent, reply with: [inaudible]\n"
    "Reply ONLY in this format:\n"
    "JA: <exact Japanese transcription>\n"
    "EN: <natural English translation>"
)

PROMPT_TRANSCRIBE = (
    "You are a Japanese speech transcription assistant.\n"
    "Listen to this audio clip of spoken Japanese.\n"
    "Transcribe EXACTLY what is said in Japanese (use kanji/hiragana/katakana as appropriate).\n"
    "If the audio is unclear or silent, reply with: [inaudible]\n"
    "Reply ONLY in this format:\n"
    "JA: <exact Japanese transcription>"
)


def resample(audio, orig_rate, target_rate):
    """Resample audio using linear interpolation."""
    if orig_rate == target_rate:
        return audio
    ratio = target_rate / orig_rate
    n_samples = int(len(audio) * ratio)
    indices = np.arange(n_samples) / ratio
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def audio_to_wav_bytes(audio_data, sample_rate):
    """Convert float32 numpy audio to WAV bytes for the API."""
    pcm = (audio_data * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def transcription_worker(client, model_name, work_queue, print_lock, context, output_file, translate):
    """Background thread: sends audio to Gemini with rolling context."""
    base_prompt = PROMPT_TRANSLATE if translate else PROMPT_TRANSCRIBE
    continuation = "Now transcribe and translate the new audio clip above." if translate else "Now transcribe the new audio clip above."
    while True:
        job = work_queue.get()
        if job is None:
            break
        _, audio_data = job
        try:
            wav_bytes = audio_to_wav_bytes(audio_data, SEND_RATE)

            # Build prompt with prior context
            with context["lock"]:
                prev = list(context["history"])
            if prev:
                ctx = "\n".join(prev)
                full_prompt = (
                    f"{base_prompt}\n\n"
                    f"For continuity, here is what was said previously:\n{ctx}\n\n"
                    f"{continuation}"
                )
            else:
                full_prompt = base_prompt

            parts = []
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=[
                    {"inline_data": {"mime_type": "audio/wav", "data": wav_bytes}},
                    full_prompt,
                ],
            ):
                if chunk.text:
                    parts.append(chunk.text)
            full = "".join(parts).strip()

            if full and "[inaudible]" not in full:
                # Extract JA line for context history
                ja_line = ""
                for line in full.splitlines():
                    if line.startswith("JA:"):
                        ja_line = line
                        break
                if ja_line:
                    with context["lock"]:
                        context["history"].append(ja_line)
                        if len(context["history"]) > CONTEXT_SIZE:
                            context["history"].pop(0)

                with print_lock:
                    sys.stdout.write("\r" + " " * 70 + "\r")
                    for line in full.splitlines():
                        print(f"  {line}")
                    print("-" * 60)
                    if output_file:
                        for line in full.splitlines():
                            output_file.write(line + "\n")
                        output_file.write("\n")
                        output_file.flush()
        except Exception as e:
            with print_lock:
                sys.stdout.write("\r" + " " * 70 + "\r")
                print(f"  [error: {e}]")
                print("-" * 60)


def calibrate_threshold(native_rate):
    """Record 1 second of ambient noise and set threshold above it."""
    print("Calibrating... stay quiet for 1 second.")
    cal = sd.rec(int(native_rate * 1), samplerate=native_rate, channels=1, dtype="float32")
    sd.wait()
    ambient = np.sqrt(np.mean(cal**2))
    threshold = max(ambient * 3, 0.003)
    print(f"Ambient noise: {ambient:.4f} | Threshold: {threshold:.4f}")
    return threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-live-preview",
        help="Gemini model name (default: gemini-3.1-flash-live-preview).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="RMS silence threshold (default: auto-calibrate).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Concurrent API workers (default: {NUM_WORKERS}).",
    )
    parser.add_argument(
        "--max-chunk",
        type=float,
        default=MAX_CHUNK_DURATION,
        help=f"Max seconds before force-sending (default: {MAX_CHUNK_DURATION}).",
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
        help="Write transcriptions to this text file.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set the GEMINI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    print(f"Model: {args.model} | Workers: {args.workers} | Max chunk: {args.max_chunk}s\n")

    dev_info = sd.query_devices(kind="input")
    native_rate = int(dev_info["default_samplerate"])
    block_size = int(native_rate * BLOCK_DURATION)
    print(f"Mic: {native_rate} Hz (sending at {SEND_RATE} Hz)")

    threshold = args.threshold if args.threshold is not None else 0.0

    audio_queue = queue.Queue()
    transcribe_queue = queue.Queue(maxsize=args.workers * 2)
    print_lock = threading.Lock()
    silence_blocks_needed = int(SILENCE_DURATION / BLOCK_DURATION)
    min_speech_blocks = int(MIN_SPEECH_DURATION / BLOCK_DURATION)
    max_chunk_blocks = int(args.max_chunk / BLOCK_DURATION)
    overlap_blocks = int(OVERLAP_DURATION / BLOCK_DURATION)

    # Shared rolling context for continuity between chunks
    context = {"history": [], "lock": threading.Lock()}

    output_file = open(args.output, "a", encoding="utf-8") if args.output else None
    if output_file:
        print(f"Writing transcriptions to: {args.output}")

    translate = not args.no_translate

    for _ in range(args.workers):
        t = threading.Thread(
            target=transcription_worker,
            args=(client, args.model, transcribe_queue, print_lock, context, output_file, translate),
            daemon=True,
        )
        t.start()

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"  audio: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    print("\nListening... Speak Japanese. Press Ctrl+C to stop.\n")
    print("-" * 60)

    seq_counter = 0

    def send_chunk(buf):
        nonlocal seq_counter
        audio_data = np.concatenate(buf).flatten()
        audio_16k = resample(audio_data, native_rate, SEND_RATE)
        try:
            transcribe_queue.put_nowait((seq_counter, audio_16k))
            seq_counter += 1
        except queue.Full:
            pass

    try:
        with sd.InputStream(
            samplerate=native_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=audio_callback,
        ):
            audio_buffer = []
            silence_blocks = 0
            speech_detected = False
            speech_blocks = 0

            while True:
                block = audio_queue.get()
                rms = np.sqrt(np.mean(block**2))

                # Level meter
                level = min(int(rms / 0.05 * METER_WIDTH), METER_WIDTH)
                thresh_pos = min(int(threshold / 0.05 * METER_WIDTH), METER_WIDTH)
                bar = "#" * level + " " * (METER_WIDTH - level)
                bar_list = list(bar)
                if thresh_pos < METER_WIDTH:
                    bar_list[thresh_pos] = "|"
                indicator = "* REC" if speech_detected else "     "
                qsize = transcribe_queue.qsize()
                pending = f" q={qsize}" if qsize > 0 else ""
                with print_lock:
                    sys.stdout.write(
                        f"\r  [{''.join(bar_list)}] {rms:.4f} {indicator}{pending}"
                    )
                    sys.stdout.flush()

                if rms > threshold:
                    audio_buffer.append(block)
                    silence_blocks = 0
                    speech_detected = True
                    speech_blocks += 1

                    # Force-send if chunk is too long, keep overlap
                    if speech_blocks >= max_chunk_blocks:
                        if len(audio_buffer) >= min_speech_blocks:
                            send_chunk(audio_buffer)
                        # Keep last N blocks as overlap for next chunk
                        if len(audio_buffer) > overlap_blocks:
                            audio_buffer = audio_buffer[-overlap_blocks:]
                        else:
                            audio_buffer.clear()
                        speech_blocks = overlap_blocks

                elif speech_detected:
                    audio_buffer.append(block)
                    silence_blocks += 1

                    if silence_blocks >= silence_blocks_needed:
                        if len(audio_buffer) >= min_speech_blocks:
                            send_chunk(audio_buffer)
                        audio_buffer.clear()
                        silence_blocks = 0
                        speech_detected = False
                        speech_blocks = 0

    except KeyboardInterrupt:
        if output_file:
            output_file.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
