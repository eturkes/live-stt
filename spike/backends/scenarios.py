"""Canned JA audio clips for backend benchmarking.

Synthesizes via Gemini TTS on first run, caches WAVs to ./cache/.
Re-runs just load from disk — no re-synthesis unless the cache is cleared.

Clip shapes are chosen to exercise different latency and turn-detection
characteristics; see DESIGN.md for the rationale.

Usage
    >>> import asyncio
    >>> from scenarios import ensure_all, load_clip
    >>> asyncio.run(ensure_all())
    >>> pcm, dur = load_clip("medium")
"""

from __future__ import annotations

import asyncio
import os
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from harness import SEND_RATE, load_wav_16k_mono_pcm16

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

CACHE_DIR = Path(__file__).parent / "cache"
TTS_MODEL = "gemini-2.5-flash-preview-tts"
TTS_VOICE = "Kore"
TTS_SR = 24000  # Gemini TTS returns 24 kHz PCM16


@dataclass
class Clip:
    id: str
    ja_ref: str
    en_ref: str
    purpose: str
    # For composite clips, a list of (text, silence_after_s); for simple, silence_after=0.
    segments: list[tuple[str, float]]


CLIPS: list[Clip] = [
    Clip(
        id="greet",
        ja_ref="こんにちは。",
        en_ref="Hello.",
        purpose="Lower bound on TTFT",
        segments=[("こんにちは。", 0.0)],
    ),
    Clip(
        id="short",
        ja_ref="今日はライブAPIのテストをしています。",
        en_ref="Today I'm testing the Live API.",
        purpose="Single short sentence",
        segments=[("今日はライブAPIのテストをしています。", 0.0)],
    ),
    Clip(
        id="medium",
        ja_ref="こんにちは、今日はライブAPIのテストをしています。よろしくお願いします。",
        en_ref="Hello. Today I'm testing the Live API. Nice to meet you.",
        purpose="T3.1 cross-spike comparability clip",
        segments=[(
            "こんにちは、今日はライブAPIのテストをしています。"
            "よろしくお願いします。",
            0.0,
        )],
    ),
    Clip(
        id="long",
        ja_ref=(
            "このプロジェクトはリアルタイムの日本語音声認識ツールです。"
            "マイクから音声を取り込み、ジェミニAPIに送って、"
            "日本語の文字起こしと英語の翻訳を同時に表示します。"
        ),
        en_ref=(
            "This project is a real-time Japanese speech recognition tool. "
            "It captures audio from the microphone, sends it to the Gemini API, "
            "and displays the Japanese transcription and English translation simultaneously."
        ),
        purpose="Does TTFT scale with utterance length?",
        segments=[(
            "このプロジェクトはリアルタイムの日本語音声認識ツールです。"
            "マイクから音声を取り込み、ジェミニAPIに送って、"
            "日本語の文字起こしと英語の翻訳を同時に表示します。",
            0.0,
        )],
    ),
    Clip(
        id="paused",
        ja_ref="最初の文です。\n二つ目の文です。",
        en_ref="This is the first sentence.\nThis is the second sentence.",
        purpose="Does the backend emit one turn or two across a 2 s gap?",
        segments=[
            ("最初の文です。", 2.0),
            ("二つ目の文です。", 0.0),
        ],
    ),
]

CLIPS_BY_ID = {c.id: c for c in CLIPS}


def _silence_pcm16(seconds: float, sr: int) -> bytes:
    n = int(seconds * sr)
    return (np.zeros(n, dtype=np.int16)).tobytes()


def _write_wav(path: Path, pcm: bytes, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def _pcm_from_tts_response(response) -> bytes:
    """Extract PCM16 bytes from a Gemini TTS streaming / non-streaming response.

    Gemini TTS returns inline audio in one of:
        response.candidates[0].content.parts[0].inline_data.data    # base64-decoded bytes
    As of google-genai 1.x the SDK gives you raw bytes.
    """
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for p in parts:
            inline = getattr(p, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                data = inline.data
                if isinstance(data, str):
                    import base64
                    data = base64.b64decode(data)
                return bytes(data)
    raise RuntimeError("no inline audio data in TTS response")


async def _synth_segment(client, text: str) -> bytes:
    """Synthesize one JA segment. Returns 24 kHz mono PCM16 bytes."""
    loop = asyncio.get_running_loop()
    # Gemini TTS needs an explicit "say this" instruction wrapper, otherwise the
    # model sometimes tries to analyze the Japanese text instead of voicing it.
    prompt = f"Say this naturally in Japanese, without any introduction: {text}"
    def _call():
        return client.models.generate_content(
            model=TTS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                    )
                ),
            ),
        )
    resp = await loop.run_in_executor(None, _call)
    return _pcm_from_tts_response(resp)


def _downsample_24k_to_16k(pcm_24k: bytes) -> bytes:
    arr = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32) / 32768.0
    ratio = SEND_RATE / TTS_SR
    n_out = int(len(arr) * ratio)
    out = np.interp(
        np.arange(n_out) / ratio, np.arange(len(arr)), arr
    ).astype(np.float32)
    return (np.clip(out, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


async def ensure_clip(clip: Clip, force: bool = False) -> Path:
    """Synthesize a clip if not cached. Returns the cached WAV path (16 kHz mono)."""
    out_path = CACHE_DIR / f"{clip.id}.wav"
    if out_path.exists() and not force:
        return out_path
    if not HAS_GENAI:
        raise RuntimeError(
            "google-genai not available; cannot synthesize clips. "
            "Install deps or copy cached WAVs in manually."
        )
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set; cannot synthesize clips.")
    client = genai.Client(api_key=api_key)

    pieces_24k: list[bytes] = []
    for text, silence_after in clip.segments:
        seg = await _synth_segment(client, text)
        pieces_24k.append(seg)
        if silence_after > 0:
            pieces_24k.append(_silence_pcm16(silence_after, TTS_SR))
    full_24k = b"".join(pieces_24k)
    full_16k = _downsample_24k_to_16k(full_24k)
    _write_wav(out_path, full_16k, SEND_RATE)
    return out_path


async def ensure_all(force: bool = False) -> list[Path]:
    paths = []
    for clip in CLIPS:
        p = await ensure_clip(clip, force=force)
        paths.append(p)
    return paths


def load_clip(clip_id: str) -> tuple[bytes, float]:
    path = CACHE_DIR / f"{clip_id}.wav"
    if not path.exists():
        raise FileNotFoundError(
            f"clip {clip_id!r} not cached; run `python scenarios.py` first."
        )
    return load_wav_16k_mono_pcm16(path)


if __name__ == "__main__":
    async def main():
        print(f"Cache dir: {CACHE_DIR}")
        paths = await ensure_all()
        for p in paths:
            size_kb = p.stat().st_size / 1024
            with wave.open(str(p), "rb") as w:
                dur = w.getnframes() / w.getframerate()
            print(f"  {p.name}  {dur:.2f}s  {size_kb:.1f} KB")

    asyncio.run(main())
