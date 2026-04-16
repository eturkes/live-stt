# Backends Spike — REPORT

**Scope:** Is there a meaningfully better streaming STT+translation backend than the current `gemini-3.1-flash-live-preview` — enough to justify switching, or add a `--backend` flag? See `DESIGN.md` for the methodology, `RESEARCH_*.md` for per-vendor specs, `results.md` for raw runs.

**TL;DR.** Five candidates researched; four prototyped to the harness contract; one run end-to-end against live TTS-synthesized JA clips. **The Gemini Live baseline is solid but not irreplaceable:** three candidates (OpenAI Realtime mini, Deepgram Nova-3 + cascade, ElevenLabs Scribe v2 RT + cascade) promise **3× cheaper** operation with acceptable-or-better latency. None of the three were executable without API keys this agent doesn't have — prototypes ship ready-to-run on a key flip. **Recommend: add `OPENAI_API_KEY` + `DEEPGRAM_API_KEY` to `.env`, rerun `bench.py`, and then decide.**

## What ran

Only Gemini. The agent only has `GEMINI_API_KEY`. The other four prototypes compiled and the bench skipped them with `KEY_MISSING`.

### Gemini (`gemini-3.1-flash-live-preview`) — 5/5 clips, clean

| Clip | dur | connect | TTFT | total | blocks | transcript quality |
|---|---|---|---|---|---|---|
| greet | 0.85 s | 0.30 | **1.42** | 1.80 | 1 | exact |
| short | 2.61 s | 0.35 | **1.11** | 2.07 | 1 | exact (minor `LiveAPI` vs `ライブAPI` casing) |
| medium | 4.85 s | 0.29 | **1.00** | 1.29 | 1 | exact; `EN` says "Please treat me well" for よろしくお願いします (equivalent) |
| long | 12.17 s | 0.33 | **1.31** | 1.77 | 1 | exact JA + full EN translation |
| paused | 4.66 s | 0.31 | **0.00** | 2.43 | 2 | **JA drift: 最初の文 → 採寸分** (phonetic near-miss), second phrase correct, 2 turns split correctly across the 2 s gap |

Mean TTFT across the four non-`paused` clips is **1.21 s** — consistent with the T3.1 spike's 1.02 s on its `medium` clip. The `paused` clip's reported `ttft=0.00` is a clamp: the first block arrived before the audio feeder finished, because Gemini finalized the first turn during the internal 2 s silence, not after audio end.

**Cost:** flat **$1.40/hr** regardless of clip. Audio-output tokens billed whether the response modality is used or not — same story as T3.1.

**Bug discovered and fixed during the spike:** `prototype_gemini.py` originally inherited the `while True: session.receive()` pattern from `live_stt.py` without inheriting its `state.stopping` escape hatch. Result: after the final turn, the receiver blocked in `session.receive()` for the full 120 s harness-runner timeout. Fixed by having the sender sleep `GRACE_S=3.0` after `audio_stream_end` and then set a `shutdown` event; the main coroutine cancels the receiver on shutdown. This is a prototype-only bug; `live_stt.py` handles it via `_wait_for_stop_or_reconnect`.

## What didn't run, and what the prototypes promise

All four prototypes compile, guard imports, conform to the harness `stream(pcm_frames, *, translate, api_key, ...)` contract, and emit `Info`/`Block`/`Err` per the spec. Line counts are all ≤ 253, under the DESIGN cap of 2×`live_stt.py`. Each has flagged ambiguities noted at the bottom of its file and reproduced below.

### OpenAI Realtime (`prototype_openai_realtime.py`, 253 LoC)

- Uses `AsyncOpenAI(api_key).realtime.connect(model="gpt-realtime-mini")` per `RESEARCH_openai_realtime.md`. Text-only output modality (`output_modalities=["text"]`) — this is the key cost lever; confirms ~$0.42/hr vs. Gemini's $1.40/hr.
- Input upsampled 16 kHz → 24 kHz in-prototype via `np.interp`, base64-encoded into `input_audio_buffer.append`.
- JA from `conversation.item.input_audio_transcription.completed`; EN from `response.output_text.delta|done`.
- Server VAD with `silence_duration_ms=700`.
- **Ambiguities (flagged by the writing agent):** exact SDK method names for the realtime namespace (`conn.input_audio_buffer.append` vs. a differently-shaped helper), `response.output_text.done.text` attribute path (falls back to accumulated deltas), `error` event attribute shape.

### Deepgram Nova-3 + cascade translation (`prototype_deepgram.py`, 214 LoC)

- Raw `websockets` to `wss://api.deepgram.com/v1/listen?model=nova-3&language=ja&encoding=linear16&sample_rate=16000&interim_results=true&endpointing=300&utterance_end_ms=1000&vad_events=true`.
- `is_final` + `speech_final` finalize a turn; `UtteranceEnd` is the safety-net.
- KeepAlive via `{"type":"KeepAlive"}` on silence (not relevant for bench clips — continuous audio — but needed in production).
- Translation via shared `translate.translate(ja)` → Gemini Flash-Lite (cheapest; ~$0.02/hr overhead).
- **Ambiguities:** `websockets.connect` kwarg rename `extra_headers → additional_headers` (v12+); `CloseStream` vs. `Finalize` message name if Deepgram tightened their server since research. Cascade runs inline in the receiver hot path — folds translation latency into `t_final`, which is the end-to-end metric we want, but means the pure Deepgram TTFT is shorter than what the harness reports.

### ElevenLabs Scribe v2 RT + cascade (`prototype_elevenlabs.py`, 199 LoC)

- Raw `websockets` to `wss://api.elevenlabs.io/v1/speech-to-text/realtime?model=scribe_v2_realtime&language=ja`.
- `commit_strategy=manual` — one `{"message_type":"commit"}` after the frames iterator exhausts. Simpler than local VAD for this single-clip harness; DESIGN notes that a production version driven by mic would need VAD-based commits or deal with the server VAD's known 10–15 s end-of-turn latency (livekit/agents #4255).
- Audio as base64 JSON (`{"message_type":"input_audio_chunk", "audio_base_64":...}`); binary-frame path not pursued due to research doc ambiguity.
- Cascade same as Deepgram.
- **Ambiguities:** exact commit-message shape (`commit` vs. `finish` vs. `finalize`), binary vs. base64 wire format, model-id punctuation (`scribe-v2-realtime` vs. `scribe_v2_realtime` — the prototype normalizes to underscores at the API boundary).

### Azure Speech Translation (`prototype_azure.py`, 224 LoC)

- `speechsdk.translation.TranslationRecognizer` with `SpeechTranslationConfig(subscription=key, region=region)`, `speech_recognition_language="ja-JP"`, `add_target_language("en")`. Integrated JA+EN in one stream — no cascade.
- `PushAudioInputStream(AudioStreamFormat(16000, 16, 1))` fed in a sender task; SDK callbacks bridged to asyncio via `loop.call_soon_threadsafe(queue.put_nowait, ...)`.
- Sentinel object in the queue unambiguously terminates the yield loop when `session_stopped` fires (or grace-window expires, to tolerate the SDK's occasional missing session-stop — Azure SDK issue #2740).
- **Ambiguities:** 10 s grace window after `push_stream.close()` may be off from real sessions; Python `_async` variants of `start_continuous_recognition` return a non-awaitable future, so the prototype runs sync `.start()/.stop()` via `loop.run_in_executor`. `translate=False` uses `SpeechRecognizer` (no translation config).

## Comparative table — the picture assuming prototypes run true

Cost figures are vendor list price as of April 2026 (see per-vendor research doc). TTFT and quality columns carry the research-stated expectation where the prototype didn't actually run; the **measured** column is populated only for Gemini.

| Backend | $/hr | TTFT (measured or expected) | JA quality | Session cap | Prototype LoC | Fit |
|---|---|---|---|---|---|---|
| **Gemini Live** | $1.40 | 1.00–1.42 s (measured, 4 clips) | exact on 4/5 clips; phonetic drift on `paused` | transparent unbounded | n/a (baseline) | **5/5** (ships today) |
| **OpenAI Realtime mini** | $0.42 | 1–2 s (research) | unknown — unpublished JA WER | hard 60-min WS, no transparent resumption | 253 | 3.5/5 — cost win, quality risk |
| **Deepgram Nova-3** + cascade | $0.66 | ~1.3–1.8 s (research, incl. cascade) | vendor-claimed; no public JA WER | stateless HTTP, reconnect trivial | 214 | 4/5 — cheapest & cleanest |
| **ElevenLabs Scribe v2 RT** + cascade | $0.41 | ~150 ms transcript (research); + cascade ~400–800 ms | ≤5% WER claimed | no documented cap, but reconnect bugs in community | 199 | 3/5 — cheapest but reliability risk |
| **Azure Speech Translation** | $2.50 | unknown | established JA Chirp-lineage model | no hard cap | 224 | 4/5 — most expensive but integrated |
| AssemblyAI | n/a | n/a | JA streaming not offered | n/a | skipped | 0/5 |
| Google Cloud STT v2 + Translate | ~$1.20 | n/a | strong Chirp 3 quality | **hard 5-min** | skipped | 2/5 |
| Groq Whisper (batch) | $0.22 | 4–5 s | Whisper-large-v3 JA | n/a | skipped | 2/5 (not streaming) |
| Local OSS (WhisperLiveKit + kotoba-whisper + nano MT) | ~$0.02 | variable, GPU-dependent | strong JA with domain models | unbounded | separate fork | 2/5 for main tool |

## Decision criteria recap (from DESIGN.md)

A backend is **viable** if: JA matches reference on `medium`, EN fluent, TTFT ≤ 2 s, $/hr < 1.40 OR session story better, prototype LoC ≤ 2× `live_stt.py` minimal path. **Recommended** if it dominates baseline on ≥ 2 of {cost, latency, session resilience} without regressing quality.

With only Gemini measured:

- Only Gemini **definitely** meets all viability bars.
- Deepgram and OpenAI Realtime mini **likely** do on paper — both are 3× cheaper; latency expectation is in-range; JA quality unconfirmed but neither vendor is an outlier.
- ElevenLabs is **cheapest on paper** but has open GitHub issues on mid-stream reconnect; either a deal-breaker or a quick fix depending on their resolution.
- Azure is **most resilient** on session cap but **2× more expensive** than Gemini, which already has unbounded sessions via resumption.

## Recommendation

**Do not migrate yet.** The data to decide doesn't exist. Two concrete next steps:

1. **Run Deepgram.** Provision a `DEEPGRAM_API_KEY` (their $200 new-account credit covers far more than a spike), populate `.env`, rerun `bench.py --only deepgram`. Compare `medium` JA transcript against the reference and against Gemini's. If they match on quality, Deepgram is a **2× cost win** with equivalent latency and a simpler reconnect model. This is the most likely winner — prototype it first.

2. **Run OpenAI Realtime mini.** Same pattern, `OPENAI_API_KEY`. **3× cost win** but more unknowns (language drift, silence hallucinations flagged in research). Worth the half-day to learn whether the cost win is real for JA or eaten by quality regressions.

ElevenLabs and Azure are tier-2: only worth running if Deepgram and OpenAI both disappoint.

## Post-spike follow-ups (distinct from `live_stt.py` changes)

- **Entity retention across backends.** The `paused` clip's 最初の文 → 採寸分 miss shows Gemini can phonetically hallucinate on short utterances. Worth including a **named-entity torture-test clip** in `scenarios.py` for each backend to measure drift before committing.
- **Real-mic smoke test.** The harness feeds canned WAVs; the prototypes never touch `sd.InputStream`. Before flipping `live_stt.py` to a non-Gemini backend, the user must smoke-test the mic path (described in `AGENT_PROMPT.md` as out-of-scope for the agent).
- **Streaming cascade.** `translate.py` is one HTTP call per JA turn. For a Deepgram/ElevenLabs migration, the cascade can be made streaming (OpenAI Chat Completions streaming, Gemini streaming) to shave the cascade latency from the bench report — post-decision optimization.
- **`--backend` flag in `live_stt.py`.** Explicitly out of scope for this spike (DESIGN.md). If the user adopts a non-Gemini primary, the flag is follow-up work.

## Non-goals reconfirmed

- No automated WER/BLEU. Spot-check against the five hand-written references is enough to rule backends in or out.
- No local OSS path. Lives in a `live-stt-local` fork per the research verdict.
- No mic testing per backend. Harness is network-path only.

## Artifacts

```
spike/backends/
├── DESIGN.md                           ← methodology
├── REPORT.md                           ← this file
├── RESEARCH_openai_realtime.md         ← per-vendor research, 1/6
├── RESEARCH_deepgram.md                ← 2/6
├── RESEARCH_assemblyai_elevenlabs.md   ← 3/6
├── RESEARCH_azure_google_cloud.md      ← 4/6
├── RESEARCH_groq.md                    ← 5/6 (skip-documented)
├── RESEARCH_local_oss.md               ← 6/6 (separate track)
├── harness.py                          ← Event/Block/Info protocol + runner
├── scenarios.py                        ← canned JA clips via Gemini TTS (5 clips)
├── translate.py                        ← shared JA→EN cascade (gemini-flash-lite)
├── prototype_gemini.py                 ← baseline, ran cleanly
├── prototype_openai_realtime.py        ← compiles; needs key to run
├── prototype_deepgram.py               ← compiles; needs key to run
├── prototype_elevenlabs.py             ← compiles; needs key to run
├── prototype_azure.py                  ← compiles; needs key + region to run
├── bench.py                            ← orchestrator
├── results.json                        ← raw structured results
├── results.md                          ← human-readable results table
└── cache/*.wav                         ← TTS-synthesized clips (5 files, ~785 KB)
```
