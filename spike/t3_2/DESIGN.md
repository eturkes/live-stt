# T3.2 Spike — Long-session memory

Scope: PLAN.md T3.2 ("CONTEXT_SIZE=3 causes topic/name drift past ~10 minutes"). The premise of that task statement is **obsolete** post-T3.1 — the client-side `CONTEXT_SIZE` ring buffer no longer exists; the Live API holds conversation state server-side for the life of one session. The surviving problem is different: the Live API audio-only session hard-caps at 15 min, and the underlying WebSocket times out at ~10 min. We lose all server-side state at the cliff. Beyond 15 min, coherence of names/topics/ongoing references depends on what we do at each reconnect.

## Findings from research (see `../../research/` if kept, otherwise task comments)

### What Gemini gives us natively (google-genai 1.70.0, verified against installed SDK)

- `SessionResumptionConfig(handle, transparent)` — server returns `LiveServerSessionResumptionUpdate.new_handle` once state is established. TTL is 2h (one doc hit says 24h; treat 2h as authoritative). Resuming restores server-side conversation context.
- `ContextWindowCompressionConfig(trigger_tokens, sliding_window=SlidingWindow(target_tokens))` — sliding window truncates oldest USER turns. **System instruction + prefix turns are exempt.** Per the capabilities doc this is the mechanism that lifts the 15-min cap, not resumption.
- `LiveServerGoAway.time_left` — proto duration string (`"60s"`). Sent ~60 s before disconnect.
- `send_client_content(turns=..., turn_complete=False)` pre-seeds conversation history. Intended to be called **before the first realtime audio chunk**, not interleaved.

### Current-code bug (unrelated to this spike but surfaced)

`live_stt.py:85`:

```python
async for response in session.receive():
```

`session.receive()` ends its async iterator on `turn_complete`. The receiver task then exits silently and all further transcriptions are lost (github `python-genai#1224`). Every prototype here fixes this with a `while not state.stopping:` wrapper.

### Prior-art signal

Providers split into two camps. Speechmatics / Google Meet / Otter run one long server-side stream and don't expose reconnect state to clients (they just work). Everyone else (OpenAI Realtime, Deepgram, AssemblyAI, ElevenLabs) makes the client responsible for re-seeding context via either `conversation.item.create`, keyterm prompts, or custom dictionaries. Gemini Live sits uniquely in the middle: it gives us *both* server-side resumption (Speechmatics-style) *and* `send_client_content` for explicit seed (OpenAI-style).

### Drift failure modes (in order of user-visible severity for this tool)

1. **Named-entity drift** — "Dr. Tanaka" in minute 2 becomes "the doctor" in minute 40. The single biggest quality issue for a Japanese live transcription tool where proper nouns (names, place names, company names) are the hardest tokens to recognize to begin with.
2. **Topic coherence loss** — the model forgets what you were talking about, affects translation word choice.
3. **Style drift** — the model stops following the `JA:` / `EN:` format. Low impact since our parser tolerates missing prefixes.

## Three approaches worth prototyping

Every prototype includes the common baseline:

- Fix the `#1224` bug (wrap `session.receive()` in `while`).
- Enable `SessionResumptionConfig(transparent=True)`.
- Enable `ContextWindowCompressionConfig(sliding_window=SlidingWindow())`.
- Handle `go_away` with a graceful reconnect loop: finish in-flight transcriptions, send `audio_stream_end`, reconnect with stored handle, resume sender/receiver tasks against the new session.
- Keep a small audio ring buffer across the reconnect gap (~200 ms) so no words are lost in the swap.

### Approach A — Baseline only (native features, no client-side memory)

Resumption handles the normal case; compression lifts the hard cap. If the handle expires (2h) or is unavailable mid-generation, reconnect with a cold session and accept the context loss. This is the "do the least" candidate and the control against which B and C are measured.

**Fails when:** handle expires, model crashes mid-generation, user explicitly restarts the tool.

### Approach B — Seed-on-reconnect (transcript replay)

A plus: maintain a client-side deque of the last ~20 emitted JA/EN blocks. On every **cold** reconnect (not transparent resume), issue `send_client_content` with those blocks as alternating user/model turns before starting the audio stream. Cost: one extra round-trip per cold reconnect, ~2k tokens of prompt bloat.

**Targets:** recovery after handle expiry, handle-less cold starts, user restarts. Bets that recent *verbatim* dialogue is worth more to the model than any summary would be.

### Approach C — Entity-dict + rolling summary

A plus: every ~30 s, a background `gemini-2.5-flash` JSON-mode call extracts proper nouns from recent transcript into a structured dict `{surface: {en, type}}`. Maintained client-side, merged monotonically (entities accumulate; optional capacity cap with LRU-by-last-mention). At session boundaries (cold reconnect **and** transparent resume boundary), the dict is serialized into the `system_instruction` as a glossary block. Optional: also generate a one-paragraph summary of the last ~5 min at each cold reconnect.

**Targets:** named-entity drift specifically, which the research identifies as the dominant quality failure for this use case. Names live outside the lossy compression and never get paraphrased.

**Failure modes:** extractor hallucinates false entities, glossary grows unboundedly over multi-hour sessions, background task adds latency spike around its run, extra API cost (~$0.002 per extraction call at `gemini-2.5-flash` pricing).

## Benchmark plan

We cannot run real audio end-to-end from the agent. The honest approach is two-layer testing:

1. **Mock-harness integration tests** — in-memory fake of `client.aio.live.connect` that:
   - Accepts `send_realtime_input` / `send_client_content` silently.
   - Emits canned `LiveServerMessage` sequences from a scripted scenario file.
   - Can inject `go_away` and force-close the connection to trigger reconnect paths.
   - Issues `session_resumption_update` messages with fake handles.
   - Records everything the client sent (for assertions).

2. **Text-level entity-extraction benchmark** — canned Japanese transcript with a known set of proper nouns introduced in the first 5 minutes and referenced later. Run approach C's extractor against it offline (live or mocked). Measure precision / recall of extracted entities.

### Metrics

Per approach, across a 45-min simulated session with 2 `go_away` events and 1 cold reconnect:

| Metric | Definition |
|---|---|
| `reconnects_ok` | reconnect loop fires, receiver resumes, no dropped tasks | 
| `transcripts_lost` | JA/EN blocks that were emitted by mock server but never reached `emit_block` |
| `seed_bytes` | total bytes sent via `send_client_content` across session |
| `extra_api_calls` | `gemini-2.5-flash` calls made outside the Live session |
| `extra_ms_p50` / `p95` | wall-clock overhead added by the approach (background tasks, seed round-trips) |
| `state_size_bytes` | resident size of client-side memory structures at session end |
| `code_delta` | LoC added vs. current `live_stt.py` |

Plus the offline entity-extraction precision/recall for approach C.

## Decision criteria

Adopt the approach with:

- `transcripts_lost == 0` across the benchmark.
- `code_delta <= 100 lines` (respect the single-file ethos).
- Justifiable cost overhead (< 20% above baseline Live API cost).
- Clearly better entity retention on the extraction benchmark *if* it adds complexity beyond A.

A is the default. B must beat A on recovery quality to justify ~30 extra LoC. C must beat both on entity retention to justify ~80 extra LoC and the extra API calls.
