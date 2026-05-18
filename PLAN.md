# PLAN — `live-stt`

Single-file Python tool. Trajectory: simplicity over completeness. No frameworks, no config, no premature abstractions. Each task is sized for one focused pass.

Status legend: `SHIPPED` `OPEN` `BLOCKED` `DEFERRED` `OUT-OF-SCOPE`.

---

## Shipped

| ID | Title | Notes |
|---|---|---|
| T1.1 | Exponential reconnect backoff | Outer loop doubles delay 1s → 30s; resets to min after `RECONNECT_RESET_AFTER_S` of session stability. Replaces pre-Live-API REST-worker equivalent (made moot by T3.1). |
| T1.2 | Timestamped output file | `-o` writes each block prefixed with ISO-8601 local timestamp. Terminal display unchanged. |
| T1.3 | Audio device selection | `--list-devices` prints `sd.query_devices()` and exits; `--device N` threads through `sd.InputStream`. |
| T1.4 | Graceful shutdown | Receiver flushes partial-turn buffer on stop via `finally`. Mid-utterance Ctrl+C still persists what was transcribed. |
| T2.1 | Tests for pure functions | `tests/test_audio.py`: `resample()` (identity/halving/upsampling/endpoint), `pcm16_bytes()` (round-trip/clipping/length), `emit_block()` (JA-only/JA+EN/fallback/timestamp). Run: `uv run pytest`. |
| T3.1 | Gemini Live API rewrite | Replaced REST chunking pipeline with `client.aio.live.connect` + `send_realtime_input`. Decision record: `SPIKE_REPORT.md`. ADR: `.agent/decisions.md` D-001. |
| T3.2 | Long-session memory | Outer reconnect loop + `SessionResumptionConfig(handle=handle)` + `ContextWindowCompressionConfig(sliding_window=SlidingWindow())`. Fix for `python-genai#1224` (see `.agent/lessons.md` L-002). Sessions run indefinitely with preserved context across reconnects (2 h handle TTL). ADR: D-003. Deferred sub-tasks: client-side transcript replay (needs disk persistence), entity-dict glossary injection (~18% cost overhead, insurance against unobserved drift). |
| T-CLEANUP-001 | Spike artifact audit | `spike/backends/cache/` and `spike/backends/results.json` gitignored in commit `0b5a6b0`. Empty `spike/t3_2/` directory removed 2026-05-18. |
| T-HOOK-001 | `uv run pytest` pre-commit hook | `.githooks/pre-commit` runs `uv run pytest -q` and aborts commit on failure. One-time setup: `git config --local core.hooksPath .githooks`. Resolves PLAN pending decision #2. |
| T2.3 | Structured logging for errors | 5 runtime `sys.stderr.write` sites → `logger` at INFO (go_away) / WARNING (audio status) / ERROR (send/recv/session). Custom `_StderrFormatter` prepends `_LINE_CLEAR` only when `sys.stderr.isatty()` so terminal coexists with the level meter and redirected stderr stays free of ANSI. Format: `[%(asctime)s] %(levelname)s %(message)s`. The GEMINI_API_KEY preflight `print` stays — outside PLAN scope. |

---

## Open

### T-BACKENDS-001 — Finish backends spike (blocked on API keys)

**Status:** BLOCKED.

**Source:** `SPIKE_REPORT_BACKENDS.md`.

**Acceptance:** `spike/backends/bench.py` produces measured rows for Deepgram and OpenAI Realtime. Decision recorded in a new ADR.

**Blocked on:** User adding `DEEPGRAM_API_KEY` and `OPENAI_API_KEY` to `.env`.

---

## Deferred

### T2.2 — Parameterize source language

**Status:** DEFERRED 2026-05-18 — user confirmed tool is Japanese-only by design. `SYSTEM_INSTRUCTION_TRANSLATE`/`TRANSCRIBE` and the `JA:` prefix in `emit_block()` stay hardcoded.

**Revisit if:** Use-case expands to other source languages. The original approach (add `--language` flag, template into system instructions, dynamic 2-letter ISO prefix in `emit_block()`, extend `test_emit_block_*`) remains valid; it just doesn't pay rent today.

---

## Out of scope (explicit)

Listed so future agents don't redebate them:

- Config files / YAML / TOML for tunables. Constants at the top of `live_stt.py` are the config surface.
- Multi-mic mixing.
- VAD reintroduction (Gemini's native VAD is sufficient).
- Speaker diarization.
- Web UI.
- Auth / multi-user.
- Metrics dashboards beyond the existing level meter.
- Package split / multi-module layout (see `.agent/decisions.md` D-002).

---

## Decisions still needed from user

_None pending._ (Last resolved 2026-05-18: language scope = Japanese-only; pre-commit hook = yes; prefix style = N/A.)
