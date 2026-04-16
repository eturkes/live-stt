# Groq Audio API — Backend Research for `live-stt`

**Date.** 2026-04-16.
**Verdict up front.** **2/5.** Groq has gotten faster and cheaper, but as of April 2026 it is still HTTP batch transcription only — no true WebSocket/SSE streaming. For `live-stt` this is architecturally a regression to our pre-Gemini chunking loop. The $/hr advantage is real but only matters if we accept 1.5–3 s added latency and lose continuous JA+EN dual output.

## 1. Audio model lineup (April 2026)

Groq's public speech lineup is still just two Whisper variants, both exposed through the OpenAI-compatible `/openai/v1/audio/transcriptions` and `/openai/v1/audio/translations` endpoints:

- `whisper-large-v3` — full 1.55 B OpenAI model, 99+ languages, 217x real-time on LPU.
- `whisper-large-v3-turbo` — pruned/fine-tuned variant, 228x real-time, cheaper. **Translation-disabled** (see section 4).

`distil-whisper-large-v3-en` was available in 2024 but is no longer listed on the pricing page or the Speech-to-Text docs — it appears to have been deprecated in favor of `large-v3-turbo`. **There is no NLLB, no Seamless, no non-Whisper ASR, and no true streaming (WebSocket / SSE) mode** — Groq's docs and SDK only expose synchronous multipart HTTP uploads. They publish a chunking cookbook as the recommended workaround for anything over 25 MB (free) / 100 MB (dev).

Refs:
- https://console.groq.com/docs/speech-to-text
- https://console.groq.com/docs/model/whisper-large-v3-turbo
- https://groq.com/blog/whisper-large-v3-turbo-now-available-on-groq-combining-speed-quality-for-speech-recognition

## 2. JA streaming pattern (the "old architecture")

Without a streaming endpoint, the idiomatic pattern is exactly what `live-stt` used before the Gemini Live rewrite:

1. Local VAD (webrtcvad / silero) marks utterance boundaries on the 100 ms PCM16 @ 16 kHz blocks.
2. Flush each utterance (or a forced 3–5 s rolling window if VAD hasn't closed) to a WAV byte buffer.
3. POST to `audio/transcriptions` with `model=whisper-large-v3`, `language=ja`. Optionally a second POST to `audio/translations` for EN.
4. Print once the response lands. No partials.

**Latency math.** At 216–299x real-time, a 3 s clip is ~10–15 ms of compute. Network RTT and queue dominate: expect 300–800 ms end-to-end per clip on a warm connection. But TTFT is gated by the VAD endpoint — you can't emit until the utterance is closed, so user-perceived latency is `utterance_length + ~500 ms`. For a 4 s JA sentence that's ~4.5 s vs. Gemini Live's ~1 s partials. Forcing 2 s windows regardless of VAD caps latency but fragments JA sentences mid-mora and tanks WER. **This is fundamentally worse UX than Gemini Live for continuous dictation.**

## 3. Pricing (April 2026)

From https://groq.com/pricing:

| Model | Rate | $/hr audio | Notes |
|---|---|---|---|
| `whisper-large-v3` | $0.111 / audio-hour | **$0.111** | billed per audio-second, min 10 s/request |
| `whisper-large-v3-turbo` | $0.04 / audio-hour | **$0.04** | transcribe-only |

For hours-long continuous use: **$0.04–0.11 /hr for transcription alone**. If we also call `/audio/translations` (only available on `large-v3`, not turbo), double-billing applies — two requests per utterance — for an effective **~$0.22/hr** for transcribe+translate. Still ~6× cheaper than Gemini Live's $1.40/hr, but the savings shrink once you count the second request.

## 4. Whisper JA quality & the translate-task gotcha

- **JA CER.** Whisper large-v3 holds large-v2 parity on Japanese (CER, not WER, is the standard JA metric) with 10–20 % error reduction vs. large-v2 on CJK/Thai/Lao/Myanmar overall. Realistic CER ~7–12 % on clean studio audio, higher on spontaneous conversational speech. https://www.saytowords.com/blogs/Whisper-V3-Benchmarks/
- **The `translate` task.** Whisper's architecture supports transcribe-and-translate-to-English in a single forward pass. Groq exposes this via `/audio/translations`. **However, this is only supported on `large-v3`, NOT `large-v3-turbo`** — turbo was fine-tuned without translation data and OpenAI explicitly documents that it cannot translate. Users have reported turbo returning Japanese output when asked to translate JA→EN. https://github.com/ggml-org/whisper.cpp/issues/2476
- **Target-language limit.** `/audio/translations` only accepts `en` — no JA→ZH, JA→KO, etc. For `live-stt`'s JA+EN dual-output requirement this is fine, but note it's not a general MT path.
- **Known JA issue.** v3 occasionally inserts random English tokens mid-Japanese-sentence (a regression from v2 that persists). Rare but visible in long sessions. https://github.com/openai/whisper/discussions/1762

**Consequence for `live-stt`:** to get JA transcript + EN translation you need **two requests per utterance** on `large-v3` (not turbo) — doubling rate-limit burn and roughly doubling cost.

## 5. Sample Python snippet

Minimal chunk-based pattern using the `groq` SDK. Assumes an external VAD has already produced WAV-encoded utterance bytes.

```python
import asyncio, io, wave
from groq import AsyncGroq

client = AsyncGroq()  # reads GROQ_API_KEY

def pcm16_to_wav(pcm: bytes, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()

async def transcribe_and_translate(pcm: bytes) -> tuple[str, str]:
    wav = pcm16_to_wav(pcm)
    ja_task = client.audio.transcriptions.create(
        file=("utt.wav", wav, "audio/wav"),
        model="whisper-large-v3",
        language="ja",
        response_format="json",
        temperature=0.0,
    )
    en_task = client.audio.translations.create(
        file=("utt.wav", wav, "audio/wav"),
        model="whisper-large-v3",   # turbo cannot translate
        response_format="json",
        temperature=0.0,
    )
    ja, en = await asyncio.gather(ja_task, en_task)
    return ja.text, en.text
```

Reconnect loop is trivial (stateless HTTP — just retry with backoff on 429/5xx). No session resumption machinery needed, which is a genuine simplification over the Gemini Live codepath.

## 6. Session limits

N/A — stateless HTTP, no sessions, no 15-minute cap, no `goAway`. Rate limits instead (https://console.groq.com/docs/rate-limits, Developer tier):

- **whisper-large-v3 / -turbo**: 20 RPM, 2,000 RPD, **7,200 audio-seconds/hr (ASH)**, 28,800 ASD.
- No documented concurrency cap beyond RPM.

**ASH is the binding constraint for continuous use.** 7,200 audio-seconds/hr = 2 hours of audio per wall-clock hour. That covers a single live user trivially. But doubled by the JA+EN dual-call pattern (each call bills its own audio-seconds), effective capacity is **1 hour of live audio per wall-clock hour per API key** — fine for `live-stt`'s one-user case, dangerously tight for any fan-out. Enterprise tier lifts this.

## 7. Known issues

- **No streaming.** The decisive one. Every third-party wrapper (Pipecat's `GroqSTTService`, etc.) admits it buffers via VAD and emits final-only results — no partials possible.
- **Turbo can't translate.** Easy to misconfigure (section 4).
- **Min 10 s billing** per request. Short utterances (<10 s, which is most of them) are billed at the 10 s floor, inflating real-world $/hr above the headline rate by 1.5–2×.
- **Free-tier RPM (20)** is quickly hit by a dual-call pattern at conversational pace.
- **Cold-start.** Groq's LPU stays warm; no documented cold-start. But first-request TLS handshake + model routing is ~200–400 ms.
- **Whisper hallucinations** in silence/music segments are well-known; VAD gating mitigates.

## 8. Verdict — **2 / 5**

Groq is the fastest Whisper host on the planet and the cheapest at headline rate. But "fastest Whisper" is still a batch ASR: you endpoint-with-VAD, you upload a blob, you wait. For `live-stt`'s "print as the person speaks" UX that's a strict downgrade from Gemini Live's streaming partials.

- If we bring back local VAD and accept utterance-granularity output: **functional** at ~$0.22/hr (dual-call on `large-v3`) vs. Gemini's $1.40/hr. That's an 85 % cost cut.
- But we lose ~1 s TTFT, lose incremental partials, re-introduce every ASR pipeline bug (VAD tuning, WAV muxing, hallucinations on silence), and gain only modest code simplification (no session resumption).
- Only compelling if (a) cost dominates UX in the target deployment, or (b) offline batch transcription becomes a separate product requirement. For the current live-dictation use case Gemini Live wins on architecture fit.

**Worth prototyping as Tier-2 fallback** (cheap, simple, robust to network flaps) but not as primary. Re-evaluate immediately if Groq announces a streaming endpoint — they are the most likely vendor to ship one.

## Sources

- https://console.groq.com/docs/speech-to-text
- https://console.groq.com/docs/model/whisper-large-v3
- https://console.groq.com/docs/model/whisper-large-v3-turbo
- https://console.groq.com/docs/rate-limits
- https://console.groq.com/docs/api-reference
- https://groq.com/pricing
- https://groq.com/blog/whisper-large-v3-turbo-now-available-on-groq-combining-speed-quality-for-speech-recognition
- https://groq.com/blog/groq-runs-whisper-large-v3-at-a-164x-speed-factor-according-to-new-artificial-analysis-benchmark
- https://docs.pipecat.ai/server/services/stt/groq
- https://github.com/ggml-org/whisper.cpp/issues/2476
- https://github.com/openai/whisper/discussions/1762
- https://www.saytowords.com/blogs/Whisper-V3-Benchmarks/
- https://huggingface.co/openai/whisper-large-v3-turbo
- https://tokenmix.ai/blog/groq-free-tier-limits-2026
