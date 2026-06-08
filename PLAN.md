# PLAN — `live-stt`

Single-file Python tool. Trajectory: simplicity over completeness. No frameworks, no config, no premature abstractions. Each task is sized for one focused pass.

Status legend: `SHIPPED` `OPEN` `BLOCKED` `DEFERRED` `OUT-OF-SCOPE` `SUPERSEDED`.

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

## Open — T4 series: no-API-key re-architecture (ADR: D-009)

User directive 2026-06-08: zero metered API keys. New architecture: **local STT (JA) + Codex-subscription translation**. Gemini Live replaced outright. Plan detail: `.agent/scratch/2026-06-08_T4-rearch.md`.

### T4.1 — Pick local JA STT engine via spike bench

**Status:** OPEN (research agents dispatched).

**Acceptance:** ≥1 local engine prototyped against `spike/backends/harness.py` `stream()` protocol; `bench.py` rows on the 5 cached clips; TTFT + JA-quality compared to Gemini baseline (1.21 s mean, exact 4/5); engine choice recorded in ADR. CPU-only (8-core/30 GB).

### T4.2 — Codex CLI install + auth + translation-leg bench

**Status:** OPEN (research agent dispatched).

**Acceptance:** codex CLI installed; user completed OAuth (interactive — agent must prompt); per-call latency measured on short JA→EN prompts via the chosen persistent surface; instruction control verified (translation-only output, agentic loop off); quota accounting on Pro tier understood.

### T4.3 — Rewrite `live_stt.py` around local STT

**Status:** OPEN. Blocked by T4.1.

**Acceptance:** Gemini session machinery removed; chosen engine streams mic audio with endpointing (engine-native or silero-vad); mic capture/level meter/`-o`/signal handling/`_StderrFormatter` preserved; `uv run pytest` green (pure-fn tests adapted); smoke-test items flagged per L-004.

### T4.4 — Wire Codex translation into live loop

**Status:** OPEN. Blocked by T4.2 + T4.3.

**Acceptance:** per-block async JA→EN with ordered display; graceful JA-only degradation on translation failure/quota exhaustion; sustained-session viability (latency + quota) demonstrated on synthetic clips.

### T4.5 — Cleanup

**Status:** OPEN. Blocked by T4.4.

**Acceptance:** `google-genai` dep + `GEMINI_API_KEY` preflight + `list_live_models.py` removed; README/orientation rewritten for new architecture; D-001/D-003 marked superseded-by-D-009; L-002/L-003 marked historical; SPIKE_REPORT*.md kept as history.

---

## Superseded

### T-BACKENDS-001 — Finish backends spike

**Status:** SUPERSEDED 2026-06-08 by the T4 series — user ruled out API keys entirely, which removes both blocked candidates (Deepgram, OpenAI Realtime metered). The spike's harness, clips, and Gemini baseline rows carry forward into T4.1. `SPIKE_REPORT_BACKENDS.md` stands as the historical record; its "populate keys" recommendation is void.

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
- Speaker diarization.
- Web UI.
- Auth / multi-user.
- Metrics dashboards beyond the existing level meter.
- Package split / multi-module layout (see `.agent/decisions.md` D-002).

_Removed from this list 2026-06-08:_ "VAD reintroduction" — Gemini's native VAD is gone with D-009; local pipeline needs endpointing (engine-native preferred, silero-vad fallback). See T4.3.

---

## Decisions still needed from user

_None pending._ (Last resolved 2026-05-18: language scope = Japanese-only; pre-commit hook = yes; prefix style = N/A.)
