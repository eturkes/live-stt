# OpenAI Realtime API — Backend Research for `live-stt`

Researched April 2026. All prices list-price USD.

## 1. Model lineup

| Model | Status | Audio in | Audio out | Text in/out | Notes |
|---|---|---|---|---|---|
| `gpt-realtime` (alias → `gpt-realtime-2025-08-28`) | **GA** (since 2025-08-28) | PCM16 @ 24 kHz, G.711 μ/A-law | same | yes | Speech-to-speech; GA flagship. 32 k ctx / 4 k out. |
| `gpt-realtime-mini` (alias → `gpt-realtime-mini-2025-12-15`) | **GA** | same | same | yes | Cheaper + faster. 128 k ctx / 4 k out. |
| `gpt-4o-realtime-preview-2025-06-03` | Preview (legacy) | same | same | yes | Superseded by `gpt-realtime`. |
| `gpt-4o-mini-realtime-preview-2024-12-17` | Preview (legacy) | same | same | yes | Superseded by `gpt-realtime-mini`. |
| `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `whisper-1` | Transcription-only (usable via Realtime in "transcription" session type or separate `/audio/transcriptions`) | — | none | — | Use for pure STT. |

JA is supported (Whisper was trained on ~680k h multilingual; `gpt-4o-transcribe` improves WER across FLEURS incl. JA). No JA-specific pricing tier. ([OpenAI GA post](https://openai.com/index/introducing-gpt-realtime/), [models docs](https://platform.openai.com/docs/models/gpt-realtime), [audio models announcement](https://openai.com/index/introducing-our-next-generation-audio-models/))

## 2. Pricing (April 2026)

Per 1 M tokens:

| | Audio in | Audio cached in | Audio out | Text in | Text out |
|---|---|---|---|---|---|
| `gpt-realtime` | $32.00 | $0.40 | $64.00 | $4.00 | $16.00 |
| `gpt-realtime-mini` | $10.00 | $0.30 | $20.00 | $0.60 | $2.40 |

Token rate (community-confirmed, not officially spec'd): user audio ≈ **1 token / 100 ms** → 600 tok/min; assistant audio ≈ 1 tok / 50 ms. ([pricing update thread](https://community.openai.com/t/gpt-realtime-gpt-realtime-mini-pricing-update/1372904), [pricing guide 2026](https://rahulkolekar.com/openai-api-pricing-in-2026-a-practical-guide-models-tokens-tiers-tools/))

**Text-output-only is fully supported** — set `session.output_modalities = ["text"]` → no audio-out tokens billed. ([forum](https://community.openai.com/t/openai-realtime-api-for-audio-input-text-output-only/1124284)) This is the decisive win over Gemini Live (which bills audio-out regardless).

**$/hr estimate — continuous JA speech, text-out only:**

- **Naïve (no context trimming)**: conversation context accumulates; every turn re-bills prior audio → ballooning cost. Community reports a 10-min, 70%-talk session = ~$2.68 on `gpt-realtime` = $16/hr. Unusable for hours.
- **With context summarization / truncation** (OpenAI [cookbook pattern](https://developers.openai.com/cookbook/examples/context_summarization_with_realtime_api)):
  - `gpt-realtime`: ~600 audio-in tok/min × $32/1M + ~400 text-out tok/min × $16/1M ≈ **$0.026/min ≈ $1.55/hr**.
  - `gpt-realtime-mini`: ~600 × $10/1M + ~400 × $2.40/1M ≈ **$0.007/min ≈ $0.42/hr**.
- Cached-input discount (80% off) applies to repeated context; with aggressive truncation, caching rarely fires for continuous streaming.

`gpt-realtime-mini` at ~$0.42/hr is **~3.3× cheaper than current Gemini Live ($1.40/hr)**. `gpt-realtime` is competitive but not a win.

## 3. API shape

- **Transport**: WebSocket (server-to-server, recommended for `live-stt`), WebRTC (browser/mobile), SIP (telephony). WS endpoint: `wss://api.openai.com/v1/realtime?model=gpt-realtime-mini`. Bearer-auth with API key.
- **SDK**: `openai>=1.x` has first-class support. `AsyncOpenAI().realtime.connect(model=...)` is the stable path; `client.beta.realtime.connect(...)` still works. ([README](https://github.com/openai/openai-python))
- **Session config** (one `session.update` at start):
  ```json
  {"type": "realtime",
   "output_modalities": ["text"],
   "audio": {
     "input": {"format": {"type": "audio/pcm", "rate": 24000},
               "turn_detection": {"type": "server_vad", "threshold": 0.5,
                                  "silence_duration_ms": 500, "prefix_padding_ms": 300},
               "transcription": {"model": "gpt-4o-transcribe", "language": "ja"},
               "noise_reduction": {"type": "near_field"}},
     "output": {}},
   "instructions": "You receive Japanese speech. For each utterance, output two lines:\nJA: <verbatim Japanese>\nEN: <natural English translation>"}
  ```
- **Per-turn events to listen for**:
  - `conversation.item.input_audio_transcription.delta` / `.completed` — JA transcript (from the dedicated transcription model).
  - `response.output_text.delta` / `.done` — model-generated output (EN translation when driven by `instructions`).
- VAD modes: `server_vad` (energy-based, tunable) or `semantic_vad` (LLM-classified end-of-turn). Set `turn_detection: null` for manual `input_audio_buffer.commit`. ([transcription guide](https://developers.openai.com/api/docs/guides/realtime-transcription), [VAD guide](https://developers.openai.com/api/docs/guides/realtime-vad))
- **Important**: input sample rate is **24 kHz** (not 16 kHz like Gemini). `live-stt` must upsample or reopen its mic at 24 kHz. PCM16, mono, little-endian, base64 in `input_audio_buffer.append`.

## 4. JA STT quality

- `gpt-4o-transcribe`: OpenAI claims lower WER than Whisper v2/v3 on FLEURS (incl. JA) via RL + scaled training. ([audio models post](https://openai.com/index/introducing-our-next-generation-audio-models/)) No published per-language JA WER number.
- **Community caveat**: multiple reports that `gpt-4o-transcribe` **drops words at utterance boundaries**, struggles with mumbled/paused speech, and occasionally produces output in wrong languages under ambiguous audio. Some users prefer `whisper-1` for edge cases. ([feedback thread](https://community.openai.com/t/gpt-4o-mini-transcribe-and-gpt-4o-transcribe-not-as-good-as-whisper/1153905))
- **Language drift** is documented for Realtime: model switches language based on proper nouns in prompt/audio (e.g. Italian on English input mentioning "Alina"). Mitigate with explicit `language: "ja"` on the transcription model and an ironclad system instruction. ([language-switching thread](https://community.openai.com/t/realtime-api-language-switching/1366289))
- **Hallucinations during silence**: documented — if the model waits 7–10 s with no input, it may emit unrelated speech not present in the transcript. ([hallucination bug](https://community.openai.com/t/hallucination-from-realtime-audio-api/1113538)) `output_modalities=["text"]` plus treating the Realtime model's output as translator-only (and relying on the separate transcription model for JA text) mitigates this.

## 5. Session limits

- **Hard cap: 60 min per WebSocket** (OpenAI direct; Azure EU still 30 min). Up from 30 min at GA. ([session-limit Q&A](https://learn.microsoft.com/en-us/answers/questions/5741275/gpt-realtime-maximum-session-length-30-minutes), [GA blog](https://developers.openai.com/blog/realtime-api))
- Context: 32 k (`gpt-realtime`) or 128 k (`gpt-realtime-mini`). Conversation accumulates; must truncate or summarize. OpenAI ships [context_summarization_with_realtime_api cookbook](https://developers.openai.com/cookbook/examples/context_summarization_with_realtime_api) as canonical pattern — summarize older turns into a system message, delete superseded items.
- **No native session resumption** (unlike Gemini Live's `SessionResumptionConfig`). Reconnect = new session. Strategy: on `close`/timeout, open new WS, replay a short summary as `instructions` or `conversation.item.create`. Users report ~15-min quality degradation even within the cap — so ~10-min proactive rollover is common. ([workaround thread](https://community.openai.com/t/realtime-api-hows-everyone-managing-longer-than-30min-sessions/1144295))
- For `live-stt` (transcribe + translate, no multi-turn state needed), rollover is cheap — just drop buffer on reconnect, no summary required.

## 6. Sample Python snippet

```python
import asyncio, base64, sounddevice as sd
from openai import AsyncOpenAI

INSTR = ("You receive Japanese speech utterances. For each, output exactly two lines:\n"
         "JA: <verbatim Japanese>\nEN: <natural English translation>\nThen stop.")

async def main():
    client = AsyncOpenAI()
    async with client.realtime.connect(model="gpt-realtime-mini") as conn:
        await conn.session.update(session={
            "type": "realtime",
            "output_modalities": ["text"],
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 600},
                "transcription": {"model": "gpt-4o-transcribe", "language": "ja"}}},
            "instructions": INSTR})

        loop = asyncio.get_running_loop()
        def cb(indata, *_):
            b64 = base64.b64encode(bytes(indata)).decode()
            asyncio.run_coroutine_threadsafe(
                conn.input_audio_buffer.append(audio=b64), loop)
        stream = sd.RawInputStream(samplerate=24000, channels=1, dtype="int16",
                                   blocksize=2400, callback=cb)  # 100 ms blocks
        stream.start()

        async for ev in conn:
            t = ev.type
            if t == "conversation.item.input_audio_transcription.completed":
                print(f"JA(raw): {ev.transcript}")
            elif t == "response.output_text.delta":
                print(ev.delta, end="", flush=True)
            elif t == "response.output_text.done":
                print("\n------")
            elif t == "error":
                print("ERR:", ev.error); break

asyncio.run(main())
```

SDK API surface per [openai-python README](https://github.com/openai/openai-python). Event names per [Realtime API reference](https://platform.openai.com/docs/api-reference/realtime).

## 7. Known issues

- **openai-python #2927** — empty `OPENAI_BASE_URL` breaks fallback. ([issue](https://github.com/openai/openai-python/issues/2927))
- **Hallucination during silence** — model emits canned phrases ("I got promoted...", "With great power...") not in transcript; ongoing since 2025. ([#1113538](https://community.openai.com/t/hallucination-from-realtime-audio-api/1113538))
- **Audio-delta/transcript desync** — transcripts and audio output occasionally don't align; can't align text to speech timing. ([latent.space missing manual](https://www.latent.space/p/realtime-api))
- **VAD false triggers under rapid speech / interruptions**; default 500 ms silence often too short for natural JA pauses — tune to 700–1000 ms. ([VAD forum](https://community.openai.com/t/realtime-api-server-turn-detection-limitations-suggestion-help-request/966610))
- **Cost-surprise** — context accumulation silently bills prior audio on every turn; "Much higher costs after updating to gpt-realtime-mini-2025-12-15" thread (Feb 2026). ([pricing update](https://community.openai.com/t/gpt-realtime-gpt-realtime-mini-pricing-update/1372904))
- **Language drift** — switches language on proper nouns. ([#1366289](https://community.openai.com/t/realtime-api-language-switching/1366289))
- **`gpt-4o-transcribe` drops boundary words** vs. Whisper. ([feedback](https://community.openai.com/t/gpt-4o-mini-transcribe-and-gpt-4o-transcribe-not-as-good-as-whisper/1153905))

## 8. Verdict

**Fit: 3.5 / 5. Prototype: yes.**

Pros:
- `gpt-realtime-mini` at **~$0.42/hr text-out is a ~3× cost win** vs Gemini Live.
- Mature Python SDK (`AsyncOpenAI().realtime.connect`) with async-iterator event loop — architecturally close to Gemini Live code path.
- Separate transcription model (`gpt-4o-transcribe`) means JA transcript isn't subject to Realtime-model hallucinations.
- Text-out modality is a first-class flag, not a workaround.

Cons / risks:
- **No session resumption.** Must build rollover logic; acceptable for `live-stt` since no persistent state.
- **60-min cap** + degradation reports at 15 min → implement proactive 10–15 min rollover.
- **24 kHz audio** requirement (project currently captures 16 kHz) — trivial upsample, but changes block size math.
- **Context accumulation cost trap** — must actively truncate `conversation.item.delete` or risk $16/hr+ bills. Gemini Live's sliding-window compression handled this automatically.
- **Silence hallucinations** and **language drift** are real and documented; treat Realtime-model text output as translator only, never as JA transcript source of truth.
- JA WER unpublished; community is split on `gpt-4o-transcribe` vs `whisper-1`. Benchmark on real `live-stt` audio before committing.

Decisive risks for this project: (1) cost-regression if context-trim logic is subtly wrong, (2) JA transcription quality degradation vs. current Gemini baseline. Both discoverable in a 1–2-day spike.
