# Backends Spike Report — Alternative STT Providers

**Scope:** Is there a meaningfully better backend than `gemini-3.1-flash-live-preview` — enough to justify switching, or adding a `--backend` flag? Follow-on from T3.1 (which adopted Gemini Live) now that the ecosystem has moved a year.

**Outcome:** No backend swap recommended yet — **data to decide doesn't exist** because four of five candidate backends require API keys the spike environment didn't have. But three candidates are **3× cheaper on paper** with plausibly equivalent latency, so the next step is obvious: populate keys, rerun `spike/backends/bench.py`, decide.

## Results at a glance

Five backends researched, four prototyped, one run:

| Backend | $/hr | TTFT | JA quality | Session cap | Verdict |
|---|---|---|---|---|---|
| **Gemini Live** (baseline) | $1.40 | **1.21 s mean (measured)** | **exact on 4/5 clips; phonetic drift on `paused`** | unbounded | ships today, not beaten on evidence |
| OpenAI Realtime mini | $0.42 | ~1–2 s (research) | unknown | 60-min hard WS | ran prototype; awaits key |
| Deepgram Nova-3 + cascade | $0.66 | ~1.3–1.8 s (research) | vendor-claimed | stateless | ran prototype; awaits key |
| ElevenLabs Scribe v2 RT + cascade | $0.41 | ~400–800 ms w/ cascade (research) | ≤5% WER claimed | no cap | ran prototype; awaits key |
| Azure Speech Translation | $2.50 | unknown | established | no cap | ran prototype; awaits key + region |

Skipped (documented in `spike/backends/REPORT.md`):

- AssemblyAI — no Japanese in streaming.
- Google Cloud STT v2 — 5-minute hard cap on `StreamingRecognize`.
- Groq Whisper — batch-only, reverts to pre-T3.1 chunked architecture.
- Local OSS (WhisperLiveKit + kotoba-whisper) — full rewrite; belongs in a `live-stt-local` fork.

## Recommendation

**Add `DEEPGRAM_API_KEY` and `OPENAI_API_KEY` to `.env` and rerun:**

```sh
uv run python spike/backends/bench.py --only deepgram
uv run python spike/backends/bench.py --only openai
```

Deepgram first — it's the most likely winner (2× cost win, simplest architecture, stateless reconnect). OpenAI second — bigger cost win (3×) but bigger unknowns on Japanese transcription quality and context-bill behavior.

ElevenLabs and Azure are tier-2; only pursue if both above disappoint.

## What landed in `spike/backends/`

- **Research** — six `RESEARCH_*.md` files (one per vendor family) produced by parallel agents from current April 2026 docs. Each is ~500–1000 words with pricing, API shape, JA support, known issues, verdict.
- **Design** — `DESIGN.md` lays out the harness contract, the five canned scenarios (greet → long → paused), the decision criteria.
- **Harness** — `harness.py`, `scenarios.py`, `translate.py`: common `stream()` async-generator protocol, Gemini-TTS-synthesized clips cached on disk, shared cheap-LLM cascade for backends that don't translate natively.
- **Prototypes** — five files `prototype_{gemini,openai_realtime,deepgram,elevenlabs,azure}.py`, 155–253 LoC each, all compile-clean with guarded SDK imports so they load without their respective deps installed.
- **Bench** — `bench.py` probes `.env` for each vendor's key, runs present ones against all clips, writes `results.json` + `results.md`. Skipped backends show `KEY_MISSING: VAR_NAME`.
- **Full analysis** — `spike/backends/REPORT.md` (longer than this file; covers methodology, measured numbers, per-prototype ambiguities, decision criteria).

## Measured baseline (Gemini)

5 clips × 1 backend = 5 runs, all clean:

| Clip | dur | TTFT | total | blocks | notes |
|---|---|---|---|---|---|
| greet | 0.85 s | 1.42 s | 1.80 s | 1 | exact |
| short | 2.61 s | 1.11 s | 2.07 s | 1 | exact |
| medium | 4.85 s | 1.00 s | 1.29 s | 1 | exact, matches T3.1 |
| long | 12.17 s | 1.31 s | 1.77 s | 1 | exact, full EN translation |
| paused | 4.66 s | 0.00 s¹ | 2.43 s | 2 | JA drift: 最初の文 → 採寸分 (phonetic near-miss); 2-turn split correct |

¹ First block arrived before audio feeder finished (during internal 2 s silence); clamped to 0 by the harness.

The `paused` result is the most instructive data point: Gemini's JA ASR **can** hallucinate near-phonetic substitutions on short utterances. Any replacement backend should be spot-checked against at least this clip before it's trusted.

## Bug found and fixed during the spike

`prototype_gemini.py` initially inherited the `while True: session.receive()` pattern from `live_stt.py` without the state-driven escape path — the receiver blocked on an idle session for 120 s after the last turn. Fixed with a `GRACE_S=3.0` sleep in the sender post-`audio_stream_end`, then `shutdown.set()` and receiver-cancel. **Does not affect `live_stt.py`.** Prototype-only bug surfaced by the synthetic-clip harness (which ends the audio stream, unlike a live mic).

## Follow-ups (if a migration is chosen)

- Streaming cascade in `translate.py` (currently one-shot per turn).
- Named-entity torture-test clip in `scenarios.py` for drift comparison.
- Real-mic smoke test for the chosen backend (out of this agent's scope).
- `--backend` flag in `live_stt.py`.

## Non-decisions (explicit)

- No code change to `live_stt.py`.
- No new deps added to `pyproject.toml`.
- No README or PLAN changes — this is a spike; `live_stt.py` continues on Gemini Live.
