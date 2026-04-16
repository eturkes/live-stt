# AssemblyAI Universal-Streaming vs. ElevenLabs Scribe v2 Realtime

Candidate backends for `live-stt` (JA transcription + EN translation, continuous mic, hours-long sessions). Research date: **April 2026**. Current baseline: Gemini Live (~1 s TTFT, ~$1.40/hr).

---

## 1. AssemblyAI — Universal-Streaming / Universal-3 Pro Streaming

### 1.1 Model lineup & status
- **Universal-Streaming** (`speech_model="universal-streaming"`) — GA. Speed/cost-optimized. English-only at launch; as of Oct 2025 release it added Spanish, French, German, Italian, Portuguese.
- **Universal-3 Pro Streaming** (`speech_model="u3-rt-pro"`) — GA. Highest accuracy, sub-300 ms latency, "native multilingual code switching", 1,000-word context prompting. Recommended for voice agents.
- **Slam-1** — deprecated in favor of Universal-3 Pro.
- **Whisper Streaming** (`whisper-rt`) — 99 languages including Japanese, but a separate, lower-priority pipeline.

### 1.2 Japanese in streaming
**Not supported in Universal-Streaming or Universal-3 Pro Streaming.** The multilingual streaming model lists exactly six languages (en/es/fr/de/it/pt) and explicitly excludes Japanese ([docs](https://www.assemblyai.com/docs/streaming/universal-streaming/multilingual-transcription)). Japanese on AssemblyAI is only available via (a) the batch `universal` model, or (b) `whisper-rt` streaming — the latter is the only realistic option for `live-stt`, but it is a different codepath from the flagship streaming product and lacks the Turn/immutable-transcription features.

### 1.3 Pricing (April 2026)
| Product | $/hr | $/min |
|---|---|---|
| Universal-Streaming | $0.15 | $0.0025 |
| Universal-3 Pro Streaming | $0.45 | $0.0075 |
| Keyterm Prompting add-on (US only) | +$0.04/hr | — |

**Caveat:** billed on session duration, not audio duration — a session kept open idle still accrues charges. No per-character surcharge ([pricing](https://www.assemblyai.com/pricing)).

### 1.4 Translation story
**No built-in translation on streaming transcripts.** AssemblyAI documents a cascade pattern: on `end_of_turn`, POST the final text to their LLM Gateway (`https://llm-gateway.assemblyai.com/v1/chat/completions`, e.g. `gemini-2.5-flash-lite`) for translation ([guide](https://www.assemblyai.com/docs/guides/real_time_translation)). Same architectural pattern as Deepgram.

### 1.5 API shape
- WebSocket: `wss://streaming.assemblyai.com/v3/ws`
- Auth: `Authorization: <api_key>` header
- Config: `sample_rate`, `speech_model`, `format_turns`; dynamic `UpdateConfiguration` messages without reconnect
- Turn detection: punctuation-based; `min_turn_silence` (100 ms default), `max_turn_silence` (1000 ms), `end_of_turn_confidence_threshold`
- Keep-alive: explicit `KeepAlive` message; optional `inactivity_timeout`

### 1.6 Session limits
**3-hour hard max per session.** Error 3005 on expiry; also fires if audio is sent faster than real-time ([common session errors](https://www.assemblyai.com/docs/streaming/common-session-errors-and-closures)). Reconnect loop is mandatory for `live-stt`.

### 1.7 Python snippet
```python
import assemblyai as aai
from assemblyai.streaming.v3 import (
    StreamingClient, StreamingClientOptions, StreamingEvents, StreamingParameters,
)

def on_turn(client, event):
    if event.end_of_turn:
        print(f"[FINAL] {event.transcript}")

client = StreamingClient(StreamingClientOptions(api_key=aai.settings.api_key))
client.on(StreamingEvents.Turn, on_turn)
client.connect(StreamingParameters(
    speech_model="u3-rt-pro", sample_rate=16000, format_turns=True,
))
client.stream(aai.extras.MicrophoneStream(sample_rate=16000))
```

### 1.8 Known issues
- Session-duration billing punishes always-on apps that aren't talking
- 3-hour hard cap forces reconnect churn
- Whisper-rt (the only JA option) is not the flagship pipeline — slower iteration, different semantics

### 1.9 Verdict: **1/5**
Japanese in the real streaming product is absent. `whisper-rt` exists but defeats the reason to pick AssemblyAI. **Do not prototype.**

---

## 2. ElevenLabs — Scribe v2 Realtime

### 2.1 Model lineup & status
- **Scribe v2 Realtime** (`model_id=scribe_v2_realtime`) — GA as of late 2025. ~150 ms TTFT, 90+ languages with auto language detection and mid-conversation switching.
- **Scribe v1 / v2 Batch** — 99-language batch transcription; separate endpoint.

### 2.2 Japanese in streaming
**Supported.** ElevenLabs lists Japanese (`jpn`) in the "Excellent Accuracy (≤ 5% WER)" tier for both Scribe v1 and v2, and Scribe v2 Realtime explicitly inherits the 90+ language set ([realtime page](https://elevenlabs.io/realtime-speech-to-text)). Spacing metrics are marked N/A for JA but this doesn't affect accuracy.

### 2.3 Pricing (April 2026)
| Product | $/hr | $/min |
|---|---|---|
| Scribe v2 Realtime | $0.39 | $0.0065 |
| Scribe v1/v2 Batch | $0.22 | $0.0037 |
| Entity detection add-on | +$0.07/hr | — |
| Keyterm prompting add-on | +$0.05/hr | — |

Billed per audio minute, no char surcharge ([pricing](https://elevenlabs.io/pricing/api)). **~3.6× cheaper than Gemini Live.**

### 2.4 Translation story
**No built-in translation on streaming transcripts.** ElevenLabs docs point to Chrome AI Translator API or external post-processing — same cascade-through-an-LLM pattern as AssemblyAI/Deepgram.

### 2.5 API shape
- WebSocket: `wss://api.elevenlabs.io/v1/speech-to-text/realtime` (plus EU/US/IN residency variants)
- Auth: `xi-api-key` header **or** single-use `token` query param
- Config (query params): `model_id`, `language_code` (ISO 639), `audio_format=pcm_16000`, `include_timestamps`, `include_language_detection`
- Turn detection: `commit_strategy` (`manual` | `vad`), `vad_threshold=0.4`, `vad_silence_threshold_secs=1.5`, `min_speech_duration_ms`, `min_silence_duration_ms`
- Output: `partial_transcript` (unstable) and `committed_transcript` (final); optional word-level timestamps
- Audio input: JSON `input_audio_chunk` with base64 PCM16

### 2.6 Session limits
No explicit duration advertised, but a `session_time_limit_exceeded` error type exists — reconnect loop still required. Concurrency shared across the ElevenLabs account (STT + TTS contend for the same pool).

### 2.7 Python snippet
No first-class STT-realtime helper in `elevenlabs-python` as of April 2026; use raw `websockets`:
```python
import asyncio, base64, json, os, websockets

async def stream_mic(mic_queue):
    url = (
        "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
        "?model_id=scribe_v2_realtime&language_code=ja"
        "&audio_format=pcm_16000&commit_strategy=vad"
    )
    headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}
    async with websockets.connect(url, additional_headers=headers) as ws:
        async def send():
            while True:
                pcm = await mic_queue.get()  # 100 ms PCM16 @ 16 kHz
                await ws.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(pcm).decode(),
                }))
        async def recv():
            async for msg in ws:
                evt = json.loads(msg)
                if evt.get("message_type") == "committed_transcript":
                    print(f"[FINAL] {evt['text']}")
        await asyncio.gather(send(), recv())
```

### 2.8 Known issues
- [livekit/agents #4255](https://github.com/livekit/agents/issues/4255): Scribe v2 Realtime via LiveKit produces zero transcriptions (Deepgram Nova-3 works on identical setup).
- [livekit/agents #4609](https://github.com/livekit/agents/issues/4609): SpeechStream does not auto-reconnect on mid-stream WS drop; subsequent transcripts come back as mojibake — **directly relevant to `live-stt`'s hours-long use case.**
- [livekit/agents #4087](https://github.com/livekit/agents/issues/4087): `committed_transcript` fires many seconds after user stops — 10–15 s end-of-turn latency in practice.
- [livekit/agents #3881](https://github.com/livekit/agents/issues/3881) / [pipecat #1307](https://github.com/pipecat-ai/pipecat/issues/1307): ecosystem integration gaps.
- No realtime speaker diarization; concurrency shared with TTS.

### 2.9 Verdict: **3/5**
Japanese streaming works, latency spec and pricing beat Gemini Live, but reconnect/commit reliability complaints are serious for a "must run for hours" tool. Worth prototyping with aggressive reconnect + `commit_strategy=manual` driven off local VAD.

---

## Cross-comparison

| Vendor | JA streaming | $/hr | TTFT | Session cap | Translation | Fit |
|---|---|---|---|---|---|---|
| AssemblyAI U-3 Pro | No (whisper-rt only) | $0.45 | ~300 ms | 3 h hard | Cascade via LLM Gateway | 1/5 |
| ElevenLabs Scribe v2 RT | **Yes** (90+ langs) | **$0.39** | **~150 ms** | Soft | External cascade | 3/5 |

**Recommendation: prototype ElevenLabs Scribe v2 Realtime.** AssemblyAI is disqualified on Japanese support in its flagship streaming path.
