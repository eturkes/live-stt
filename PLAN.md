# Development Plan: live-stt

Single-file Python tool (`live_stt.py`, ~270 lines). The trajectory — especially commit `3930a8e` which stripped VAD and calibration — favors simplicity. This plan respects that: no new abstractions, frameworks, or enterprise-y patterns. Items are sized for one focused pass each.

## Tier 1 — Quick wins

### T1.1 — Retry on API errors with backoff
Worker at `live_stt.py:133` silently drops chunks on exception. Wrap `generate_content_stream` in a retry loop (3 attempts, 1s/2s/4s backoff). Preserves continuity when Gemini throttles or returns transient 5xx.

**Acceptance:** Transient API errors no longer drop audio; permanent errors still surface via the existing error print path after retries are exhausted.

### T1.2 — Timestamped output file
`live_stt.py:128-132` writes bare JA/EN lines. Prepend ISO-8601 timestamp to each block so the file is useful as a post-session log.

**Acceptance:** Output file has `[2026-04-16T14:22:07]` (or similar) prefix per block. Terminal display unchanged.

### T1.3 — Audio device selection
Add `--device` flag; pass to `sd.InputStream` at `live_stt.py:226`. Include `--list-devices` mode that prints `sd.query_devices()` and exits.

**Acceptance:** `live-stt --list-devices` prints a numbered list; `live-stt --device 3` uses that device.

### T1.4 — Graceful shutdown
On Ctrl+C (`live_stt.py:263`), push sentinel `None` per worker and `join()` with a bounded timeout so in-flight transcriptions finish. Move `output_file.close()` into a `finally` so crashes also flush.

**Acceptance:** In-flight API requests complete before the process exits; output file always closes cleanly.

## Tier 2 — Quality / correctness

### T2.1 — Tests for pure functions
`resample()` and `audio_to_wav_bytes()` are pure and trivially testable. Add `tests/test_audio.py`, wire `pytest` into a dependency group in `pyproject.toml`.

**Acceptance:** `uv run pytest` passes. At minimum: resample identity (same rate in/out), resample halving (32k→16k length), WAV round-trip (bytes parse back to same samples).

### T2.2 — Parameterize source language
README line 135 already flags this. Add `--language` (default `japanese`) and template it into both prompts. Decide whether to keep literal `JA:` label or generalize.

**Acceptance:** `live-stt --language korean` produces Korean transcription with matching output label.

### T2.3 — Structured logging for errors
Replace `print(f"  [error: {e}]")` at `live_stt.py:136` with `logging` to stderr — lets users pipe `-o` output cleanly.

**Acceptance:** Stdout/stderr cleanly separable; existing console UX unchanged when run in a terminal.

## Tier 3 — Bigger bets (decide before starting)

### T3.1 — Gemini Live API (bidirectional streaming)
`list_live_models.py` hints this was anticipated. Likely cuts latency 2–5× and removes the chunking/overlap machinery entirely. Substantial rewrite (REST worker pool → single persistent session), different error surface, possibly different pricing. Run a timeboxed 4-hour spike before committing.

### T3.2 — Long-session memory
`CONTEXT_SIZE=3` causes topic/name drift past ~10 minutes. Could summarize older history into a single rolling "so far" line. Only worth it if long sessions matter in practice.

## Out of scope (explicit)

Config files, multi-mic mixing, VAD reintroduction, speaker diarization, web UI, auth, metrics beyond the current meter.

## Decisions needed before executing

1. **Live API or not?** T3.1 reshapes the project. If going there, T1.1 and overlap code become moot — do the spike first.
2. **Language scope.** If only Japanese, skip T2.2. Otherwise do it before hardcoding more JA-isms.
3. **Test discipline.** T2.1 only pays off if tests actually run — add `uv run pytest` to a pre-commit hook or accept it's aspirational.

## Suggested first PR

Bundle **T1.1 + T1.4** — shared worker code path, both prevent lost transcriptions, both small. One PR, shippable in an afternoon.
