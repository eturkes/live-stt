# Development Plan: live-stt

Single-file Python tool (`live_stt.py`). The trajectory favors simplicity: no new abstractions, frameworks, or enterprise-y patterns. Items are sized for one focused pass each.

## Shipped

- **T1.1 — Exponential reconnect backoff.** Outer reconnect loop now doubles on each failure (1s → 30s cap) and resets to the minimum once a session stays up for `RECONNECT_RESET_AFTER_S`. The original framing targeted the pre-Live-API REST worker and was moot after T3.1; this is the equivalent resilience guarantee for the persistent session.
- **T1.2 — Timestamped output file.** Each block written via `-o` is prefixed with an ISO-8601 local timestamp. Terminal display unchanged.
- **T1.3 — Audio device selection.** `--list-devices` prints `sd.query_devices()` output and exits; `--device N` selects the input by index and is threaded through `sd.query_devices()` and `sd.InputStream`.
- **T1.4 — Graceful shutdown.** Receiver flushes its partial-turn buffer on stop via a `finally`, so a mid-utterance Ctrl+C still persists what the model already transcribed. Output file close and stream close remain in the outer `finally`.
- **T2.1 — Tests for pure functions.** `tests/test_audio.py` covers `resample()` (identity, halving, upsampling, endpoint handling), `pcm16_bytes()` (round-trip, clipping, length), and `emit_block()` parsing (JA-only, JA+EN, fallback on unlabeled text, timestamp prefix). Run with `uv run pytest`.
- **T3.1 — Gemini Live API.** Full rewrite onto `client.aio.live.connect` with `send_realtime_input`. Removed the REST chunking/overlap machinery.
- **T3.2 — Long-session memory.** Outer reconnect loop + `SessionResumptionConfig(handle=handle)` + `ContextWindowCompressionConfig(sliding_window=SlidingWindow())` + fix for python-genai#1224. Sessions run indefinitely with preserved context across reconnects (2h resumption handle TTL). Deferred sub-tasks: client-side transcript replay (Approach B, needs disk persistence) and entity-dict glossary injection (Approach C, 18% cost overhead for insurance against drift modes we haven't observed).

## Open

### T2.2 — Parameterize source language
Add `--language` (default `japanese`) and template it into both system prompts. Decide whether to keep the literal `JA:` label or generalize to a dynamic prefix (`KO:`, `ZH:`, …).

**Acceptance:** `live-stt --language korean` produces Korean transcription with matching output label.

### T2.3 — Structured logging for errors
Replace the ad-hoc `sys.stderr.write(...)` calls (send/recv/session errors) with `logging` to stderr so users can pipe `-o` output cleanly while still seeing diagnostics.

**Acceptance:** Stdout/stderr cleanly separable; existing console UX unchanged when run in a terminal.

## Out of scope (explicit)

Config files, multi-mic mixing, VAD reintroduction, speaker diarization, web UI, auth, metrics beyond the current meter.

## Decisions needed before executing

1. **Language scope.** If only Japanese, skip T2.2. Otherwise do it before hardcoding more JA-isms.
2. **Test discipline.** T2.1 tests only pay off if they actually run — wire `uv run pytest` into a pre-commit hook or accept it's aspirational.
