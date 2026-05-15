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

---

## Open

### T2.2 — Parameterize source language

**Acceptance:** `live-stt --language korean` produces Korean transcription with matching output label (e.g. `KO:` instead of `JA:`).

**Approach:**
- Add `--language LANG` (default `japanese`).
- Template the language name into `SYSTEM_INSTRUCTION_TRANSLATE` and `SYSTEM_INSTRUCTION_TRANSCRIBE`.
- Decide: keep literal `JA:` label or generalize to dynamic prefix (`KO:`, `ZH:`, …). Recommendation: dynamic prefix — strips one source of model drift (model is more likely to emit a prefix that matches the source language than to translate the label).
- Update `emit_block()` parser to use the dynamic prefix instead of hardcoded `JA:`.
- Tests: extend `test_emit_block_*` to cover the dynamic prefix.

**Blockers:** None. Decision needed: should the prefix be a 2-letter ISO code (`KO:`) or full name (`Korean:`)? Default to 2-letter ISO if the user has no preference.

---

### T2.3 — Structured logging for errors

**Acceptance:** Stdout/stderr cleanly separable when running with `-o`. Existing terminal UX unchanged. `live-stt > /dev/null 2> errors.log` shows only diagnostics in `errors.log`.

**Approach:**
- Replace `sys.stderr.write(...)` calls (send error, recv error, session error, `go_away`, audio status) with `logging` to stderr.
- Configure a stderr handler with format `[%(asctime)s] %(levelname)s %(message)s`.
- Keep the level meter on stdout; it must still overwrite cleanly with the existing `\r\x1b[2K` clear sequence.

**Blockers:** None.

---

### T-CLEANUP-001 — Audit untracked spike artifacts

**Acceptance:** `git status` clean of confusing untracked files. Bench artifacts gitignored if appropriate.

**Approach:**
- `spike/backends/cache/` — TTS-clip cache, regenerable. Gitignore.
- `spike/backends/results.json` — bench output, regenerable from `spike/backends/bench.py`. Gitignore unless we want a check-in baseline (decide).
- Empty `spike/t3_2/` directory — was supposedly removed in commit `ae2f706` but still on disk. `rmdir` it.

---

### T-BACKENDS-001 — Finish backends spike (blocked on API keys)

**Status:** BLOCKED.

**Source:** `SPIKE_REPORT_BACKENDS.md`.

**Acceptance:** `spike/backends/bench.py` produces measured rows for Deepgram and OpenAI Realtime. Decision recorded in a new ADR.

**Blocked on:** User adding `DEEPGRAM_API_KEY` and `OPENAI_API_KEY` to `.env`.

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

| # | Question | Default if no answer |
|---|---|---|
| 1 | Language scope: only Japanese (skip T2.2) or general (do T2.2 before more JA-isms accrete)? | Do T2.2 — generalize is cheap now, expensive later. |
| 2 | T2.1 test discipline: wire `uv run pytest` into a pre-commit hook, or accept tests as aspirational? | Pre-commit hook — tests only pay off if they actually run. |
| 3 | T2.2 prefix style: 2-letter ISO (`KO:`) or full name (`Korean:`)? | 2-letter ISO. |
