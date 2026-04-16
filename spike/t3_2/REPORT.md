# T3.2 Spike Report — Long-session memory

**Scope.** PLAN.md T3.2. The original framing ("`CONTEXT_SIZE=3` causes name drift past ~10 minutes") is obsolete — that client-side ring was deleted in the T3.1 rewrite. What remains is the Gemini Live 15-minute audio-only session cap and the underlying ~10-minute WebSocket timeout. Beyond that cliff, all server-side conversation state is lost.

**Outcome.** Adopt **Approach A** (resumption + compression + reconnect loop + the `#1224` receiver fix). It's the minimum change that lifts the 15-min cap and survives unexpected disconnects. Approaches B and C add real machinery for narrower failure modes that we have no data to justify yet.

## Inputs

- **Research** (summarized in `DESIGN.md`): three parallel sub-agents covered Gemini SDK internals, competitor STT behavior, and rolling-summary patterns.
- **SDK verification** against installed `google-genai==1.70.0`: all fields used in prototypes are present (`SessionResumptionConfig`, `ContextWindowCompressionConfig`, `SlidingWindow`, `LiveServerGoAway`, `LiveServerSessionResumptionUpdate`).
- **Two benchmarks** against a scripted mock of `client.aio.live.connect`:
  1. 45-min session with `goAway` + unexpected close (tests within-process reconnect).
  2. Cold start with pre-loaded `recent_blocks` (simulates process-restart recovery).

## Key finding separate from the three approaches

**`live_stt.py:85` has a latent bug (`python-genai#1224`).** `async for response in session.receive():` exits its async iterator on the first `turn_complete` event, silently dropping every subsequent transcription. Wrap with `while not state.stopping:` to defeat it. All three prototypes apply this fix. It should be ported to `live_stt.py` regardless of which T3.2 approach ships.

## The three prototypes

All three share a common baseline:

- Outer reconnect loop wrapping `client.aio.live.connect`.
- `SessionResumptionConfig(transparent=True)` on the config.
- `ContextWindowCompressionConfig(sliding_window=SlidingWindow())` — **required** to lift the 15-min cap; resumption alone only handles the 10-min WebSocket timeout.
- `goAway` handling: set `should_reconnect`, exit TaskGroup cleanly, open next session.
- The receiver-loop bug fix.
- `state.handle` carried across reconnects.

**Approach A** — baseline only. 327 lines. No client-side memory.

**Approach B** — A + `state.recent_blocks` deque (maxlen=20) + `send_client_content` seed on every cold connect (`state.handle is None`). 356 lines (+29 vs. A).

**Approach C** — A + `EntityMemory` dict with LRU eviction + background `gemini-2.5-flash` JSON-mode extractor every ~30 s + glossary injected into `system_instruction` on every (re)connect. 505 lines (+178 vs. A).

## Benchmark results

### Benchmark 1: 45-min normal session

3 scripted sessions, `goAway` between 1–2, unexpected close between 2–3, resumption handle issued early in each session and carried across.

| proto | blocks | reconnects | sessions | seed_events | extractor_runs | entities | wall_s |
|---|---|---|---|---|---|---|---|
| A | 29/29 | 2 | 3 | 0 | 0 | 0 | 17.70 |
| B | 29/29 | 2 | 3 | **0** | 0 | 0 | 17.71 |
| C | 29/29 | 2 | 3 | 0 | 58 | 8/8 (P=R=1.00) | 17.66 |

Headline: **A, B, C are functionally equivalent for within-process recovery.** No data loss. The resumption handle carries in-memory across reconnects, so B's cold-seed path (`state.handle is None`) never fires. C extracts all 8 ground-truth entities at 100% precision/recall — valuable as insurance, but not exercised by any user-visible failure in this benchmark.

### Benchmark 2: cold start with prior history

Single session, fresh `State` with `handle=None`, but `recent_blocks` pre-populated with 4 prior JA/EN pairs (simulating restore-from-disk after a process restart).

| proto | blocks | seed_events | seed_bytes | client_content_calls | entities |
|---|---|---|---|---|---|
| A | 3/3 | 0 | 0 | 0 | 0 |
| B | 3/3 | **1** | **373** | **1** | 0 |
| C | 3/3 | 0 | 0 | 0 | 4/8 (P=1.00, R=0.50) |

Headline: **B's differentiation only materializes if you first implement disk persistence of `recent_blocks`.** Without that, a process restart begins with empty history and B behaves identically to A. C's extractor picks up 4 of 8 entities — all four that appeared in the pre-populated history — demonstrating that the glossary survives restart if persisted, but the spike did not implement persistence for entities either.

## Analysis

### A is the floor

Without A-level changes, the tool has two latent failure modes:

1. **15-min hard cliff** on audio-only Live sessions. Context compression lifts it; nothing else does.
2. **Silent transcription loss** after the first turn (`#1224`). This is already a user-facing bug today.

A is not optional — it fixes a real bug and unblocks sessions longer than 15 minutes. Every reasonable version of T3.2 ships A's changes.

### B's value hinges on persistence we haven't built

The premise of B was "seed on cold reconnect." But within a running process, `state.handle` gets set during the first session and carries forward — so "cold reconnect" in the live-process sense doesn't exist. B's path only fires at process restart, and only if `recent_blocks` was persisted externally.

To make B genuinely useful would require:

- Persist `recent_blocks` to disk (JSON file beside `--output`, say).
- Restore `recent_blocks` on process start.
- Optionally detect "handle rejected by server" and clear `state.handle` so a subsequent connect hits the cold path. The SDK doesn't currently surface this signal distinctly, so we'd infer from lack of resumption update within N seconds — brittle.

That's a separate feature (persist transcription history across process restarts) that's probably worth doing, but can't justify B's code in `live_stt.py` on its own.

### C pays for insurance we don't know we need

C's entity extractor genuinely works: precision=recall=1.00 in Benchmark 1. The glossary is injected into `system_instruction` on every connect, so it survives:

- Sliding-window compression (system instructions are exempt).
- Resumption handle expiry.
- Process restart (if `entities` dict is persisted, which the spike did not implement).

The costs:

- **+178 LoC** vs. A. 55% larger file.
- **~58 extractor calls per 18 s** in the benchmark ≈ ~3.2 Hz. At production `EXTRACTOR_INTERVAL_S=30s`, that's 2 calls/min, or ~120 calls/hr at `gemini-2.5-flash` pricing ≈ $0.25/hr extra. On top of Live's ~$1.40/hr → ~18% cost bump. (A naive optimization — skip extraction when `recent_blocks` hasn't advanced since last run — cuts this roughly in half.)
- **A second model in the pipeline**, with its own failure modes (rate limits, JSON-mode parse errors, rare hallucinated entities). The prototype handles extractor errors by returning `[]` but they're still noise.
- **Entity extraction doesn't fix model recognition.** The glossary tells the model how to *write* 田中先生, not how to *hear* it. ASR misrecognition of rare proper nouns still happens upstream of the glossary.

The spike research (Agent 3) flagged entity drift as the dominant failure mode for this kind of tool in theory — but we have no bug reports from real use yet. This is insurance against a theoretical problem.

## Recommendation

**Ship A. Defer B and C.**

Concrete next PR for `live_stt.py`:

- Wrap `session.receive()` in `while not state.stopping:` (fix `#1224`).
- Add the outer reconnect loop to `run_session`.
- Add `state.handle` and `state.should_reconnect`, capture new handle from resumption updates.
- Enable `SessionResumptionConfig(transparent=True)` and `ContextWindowCompressionConfig(sliding_window=SlidingWindow())` on the connect config.
- Handle `go_away` in the receiver (log it, trigger reconnect).
- Update README: note the 15-min cap is lifted, document the reconnect behavior in the "How It Works" section.

Estimated delta: ~60 lines of additional code vs. current `live_stt.py`. Single file, no new deps.

## Follow-ups that could justify B or C later

- **If the user reports topic drift or name drift in actual long sessions:** revisit C. Start with the simplest variant — extract entities only from the last ~2 minutes, not the full window; run extractor only when `recent_blocks` has changed.
- **If the user wants transcripts to survive process restarts:** implement persistence of `recent_blocks` to a JSON sidecar file. Once that exists, B becomes a ~20-line follow-up (seed on cold connect using the restored history).
- **If Google ships a half-cascade Live model with text-only output modality:** revisit the pricing analysis in SPIKE_REPORT.md and the cost overhead of C's extractor.

## What this spike did not verify

- Behavior against a real Gemini Live server. The mock covers the protocol surface we use but cannot reproduce: actual handle rejection semantics, `goAway` timing variance per model, real VAD turn boundaries, and actual context-compression behavior when the window is exceeded.
- End-to-end audio with a microphone. All benchmarks used `FakeStream` (no audio) and injected transcription events directly.
- Long sessions (hours). The benchmark is time-scaled 100× — 45 sim-min = 18 wall-sec. No real clock-drift or network-jitter effects.
- The actual user-facing quality of A vs. C. We measured client-side correctness, not model output quality.

A real-mic smoke test after A ships is the next step to catch anything the mock hid.
