# Backends Spike — Alternative Streaming STT + Translation

**Scope:** Is there a meaningfully better backend (cheaper, faster, comparable JA quality) than the current Gemini Live path — enough to justify switching, or at least adding a `--backend` flag? Produce the empirical evidence a migration decision would need.

**Baseline** (from `SPIKE_REPORT.md`, T3.1):
- `gemini-3.1-flash-live-preview` via `client.aio.live.connect`
- TTFT ≈ 1.02 s after utterance end (benchmark clip)
- Total post-audio latency ≈ 2.9 s
- List price: ~**$1.40 / hr** continuous speech, dominated by unavoidable audio-output tokens
- Session duration: unbounded (resumption handle + sliding-window compression)

## Candidates and disposition

Research notes live in `RESEARCH_*.md` in this directory. One agent per vendor, each read current April 2026 pricing off the vendor's site.

| Vendor | Disposition | Key reason |
|---|---|---|
| Gemini Live (baseline) | **Prototype** | Control |
| OpenAI Realtime (`gpt-realtime-mini`) | **Prototype** | Architecturally similar (bidi WS), ~$0.42/hr |
| Deepgram Nova-3 + cascade | **Prototype** | Cheap ($0.66/hr), stateless, clean API |
| ElevenLabs Scribe v2 RT + cascade | **Prototype** | Cheapest ($0.39/hr), ≤5% JA WER reputed |
| Azure Speech Translation | **Prototype** | Integrated STT+MT in one stream, no session cap ($2.50/hr) |
| AssemblyAI | Skip | Streaming is English/EU-only; `whisper-rt` fallback defeats the point |
| Groq Whisper | Skip (documented) | Still batch-only, reverts to pre-T3.1 architecture |
| Google Cloud STT v2 + Translate | Skip (documented) | Hard 5-min streaming cap, reconnect churn |
| Local OSS (WhisperLiveKit + kotoba-whisper) | Separate track | Full rewrite; belongs in a `live-stt-local` fork, not this spike |

Five prototypes total. Skipped vendors get a paragraph in `REPORT.md` so the reasoning is preserved.

## Common interface

Each prototype exports an async generator function:

```python
async def stream(
    pcm_frames: AsyncIterator[bytes],   # 100 ms @ 16 kHz PCM16 mono blocks
    translate: bool,                    # True = emit EN; False = JA only
    api_key: str,                       # vendor-specific
    **kwargs,                           # per-vendor extras (region, model, etc.)
) -> AsyncIterator[TranscriptEvent]:
    ...
```

`TranscriptEvent` is one of:
- `Partial(text, is_ja, t)` — interim transcript delta
- `Block(ja, en, t_first, t_final)` — finalized utterance turn
- `Info(kind, detail, t)` — session lifecycle (connected, go_away, reconnect, closed)
- `Err(message, t)` — non-fatal diagnostic

The harness collects events into a `BenchResult` and ignores partials for the latency calculation — only `Block` events count.

Backends that don't natively emit English (all non-Gemini) internally run the cascade translator. The prototype is responsible for invoking it; the harness sees `Block(ja, en, …)` regardless of whether `en` came from one call or two. This keeps the harness backend-agnostic.

## Audio feeder

Real-time pacing is the whole point of "streaming" STT, so benches feed WAV files at 1× wall-clock by default:

- Canned clips in `scenarios.py`, cached to `spike/backends/cache/*.wav`.
- `harness.feed(clip, frames_per_second=10)` yields 100 ms PCM16 frames, sleeping `0.1 s` between them. Optional `speedup` param for quicker smoke-tests.
- Mic capture is out of scope; the user smoke-tests the real mic path separately.

## Scenarios

Five clips, synthesized via `gemini-2.5-flash-preview-tts` (same as T3.1):

| Clip | JA text | Duration | Purpose |
|---|---|---|---|
| `greet` | こんにちは。 | ~1 s | Lower bound on TTFT |
| `short` | 今日はライブAPIのテストをしています。 | ~3 s | Single sentence |
| `medium` | T3.1 benchmark clip verbatim | ~5 s | Cross-spike comparability |
| `long` | ~3-sentence paragraph about the project | ~15 s | Does TTFT scale with utterance length? |
| `paused` | Two short phrases with a 2 s silence between them | ~7 s | Does the backend emit two turns or one? |

Each clip has a hand-authored reference `{ja, en}` for qualitative comparison (not automated WER/BLEU — that's follow-up work).

## Metrics per (backend, clip)

| Metric | Definition |
|---|---|
| `connect_s` | Wall-clock from `stream()` invocation to first backend-level "ready" signal |
| `ttft_s` | Audio-end-of-clip → first `Block` event |
| `total_s` | Audio-end-of-clip → last `Block` event |
| `n_blocks` | Count of `Block` events |
| `ja_text` / `en_text` | Concatenated transcript/translation |
| `est_cost_usd` | Duration × vendor list price (+ cascade if applicable) |
| `loc` | Prototype file line count |
| `bugs` | Any `Err` events or exceptions captured |

`bench.py` writes one JSON row per (backend, clip) run plus a `results.md` aggregation.

## Decision criteria

A backend is **viable** for recommendation if ALL of:

1. JA text matches the reference on the `medium` clip on spot check (allow spacing/punct drift).
2. EN text is fluent and on-topic.
3. `ttft_s ≤ 2.0 s` on the `medium` clip.
4. `est_cost_usd/hr < 1.40` OR session-duration story better than Gemini's.
5. Prototype LoC ≤ 2× `live_stt.py` minimal path (~300 lines).

A backend is **recommended** if it dominates the baseline on ≥ 2 of {cost, latency, session resilience} without regressing quality.

## Runnability caveat

The agent running this spike has `GEMINI_API_KEY` (baseline can actually run). The other four keys may or may not be present in `.env`. `bench.py` probes for each key and emits `SKIPPED (KEY_MISSING: X)` when absent. The prototypes are written so they're verifiable-on-paper and runnable-on-a-key-flip — the user can populate `.env` later and rerun.

## Explicit non-goals

- A `--backend` flag in `live_stt.py`. That ships after this spike lands a decision.
- Offline evaluation against a large JA ASR dataset. Five hand-picked clips is enough to decide whether to keep going.
- Mic-path integration tests per backend. The shape of each prototype's `stream()` matches the current audio pipeline; the harness exercises the network path and leaves mic plumbing to `live_stt.py`.
- A production-quality translation cascade. The prototypes use `gpt-5-nano` (or Gemini Flash-Lite) in simple per-turn HTTP mode. A batched/streaming cascade is optimization, not decision-making.
