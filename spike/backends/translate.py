"""Shared cascade translator for backends that only emit JA.

The "cascade" pattern: finalized JA transcript from Deepgram / ElevenLabs /
OpenAI-transcribe-text / Azure-without-translate goes out as a single HTTP
call to a cheap text LLM that returns the EN.

Defaults to Gemini Flash-Lite since GEMINI_API_KEY is already required by the
baseline. Swap to OpenAI gpt-5-nano via `MT_BACKEND=openai` env toggle.

Rough cost (April 2026):
- gemini-flash-lite: ~$0.10 / 1M input tokens, ~$0.40 / 1M output
- gpt-5-nano: $0.05 / 1M in, $0.40 / 1M out
At ~40 JA chars/turn and 20 turns/min, per-hour translation cost is ~$0.02-0.05.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a Japanese-to-English interpreter. "
    "Translate the given Japanese text into a single line of fluent English. "
    "Preserve proper nouns and numbers. "
    "Respond with only the English translation — no preamble, no quotes, no JA echo."
)


@dataclass
class MTResult:
    en: str
    latency_s: float
    prompt_chars: int
    completion_chars: int


async def translate(ja: str, *, backend: str | None = None, timeout_s: float = 5.0) -> MTResult:
    """One-shot JA→EN translation. Returns "" on failure rather than raising."""
    import time

    backend = backend or os.environ.get("MT_BACKEND", "gemini")
    t0 = time.monotonic()

    try:
        if backend == "openai":
            en = await _openai_translate(ja, timeout_s)
        else:
            en = await _gemini_translate(ja, timeout_s)
    except Exception as e:
        return MTResult(en=f"[translation error: {type(e).__name__}]",
                        latency_s=time.monotonic() - t0,
                        prompt_chars=len(ja),
                        completion_chars=0)

    return MTResult(
        en=en.strip(),
        latency_s=time.monotonic() - t0,
        prompt_chars=len(ja),
        completion_chars=len(en),
    )


async def _gemini_translate(ja: str, timeout_s: float) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=api_key)
    loop = asyncio.get_running_loop()

    def _call():
        return client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=ja,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                max_output_tokens=200,
            ),
        )

    resp = await asyncio.wait_for(loop.run_in_executor(None, _call), timeout=timeout_s)
    return getattr(resp, "text", "") or ""


async def _openai_translate(ja: str, timeout_s: float) -> str:
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = openai.AsyncOpenAI(api_key=api_key)
    resp = await asyncio.wait_for(
        client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ja},
            ],
            temperature=0.0,
            max_tokens=200,
        ),
        timeout=timeout_s,
    )
    return resp.choices[0].message.content or ""
