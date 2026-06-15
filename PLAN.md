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

## T4 series (shipped) — no-API-key re-architecture (ADR: D-009)

User directive 2026-06-08: zero metered API keys. New architecture: **local STT (JA) + Codex-subscription translation**. Gemini Live replaced outright. Plan detail: `.agent/scratch/2026-06-08_T4-rearch.md`.

### T4.1 — Pick local JA STT engine via spike bench

**Status:** SHIPPED 2026-06-08. Engine: **reazonspeech-k2-v2** primary, parakeet-ja A/B (ADR: D-010). Both via sherpa-onnx + silero VAD; TTFT ≤0.10 s vs Gemini 1.21 s; near-exact JA. `prototype_local.py` carries the pattern (VAD pre-pad 0.4 s, per-sentence bench clips — see D-010 findings). Models in `models/` (gitignored; URLs in `.agent/scratch/2026-06-08_T4-research-notes.md`).

### T4.2 — Codex CLI install + auth + translation-leg bench

**Status:** SHIPPED 2026-06-08 (ADR: D-011). codex 0.137.0 + user OAuth (plan=`prolite`; Spark entitled). Persistent `codex app-server` surface: **Spark+low, tool features off, `developerInstructions`** → p50 0.99 s/turn, 8/8 clean JA→EN, 4/4 injection-resistant, ~180 uncached tokens/turn marginal (quota burn ≈0 %). `codex_client.py` carries the T4.4 pattern. Fallback: mini+none (p50 1.18 s).

### T4.3 — Rewrite `live_stt.py` around local STT

**Status:** SHIPPED 2026-06-08. Gemini machinery (session/reconnect/resumption/sender/receiver, `pcm16_bytes`) removed; mic → `resample` → silero VAD → `RingBuffer` pre-pad re-slice → executor decode → `emit_block(ja, en, file)`. Meter/`-o`/signals/`_StderrFormatter` kept; `--engine k2v2|parakeet`; `models/README.md` documents weights. 22 tests green (ring-buffer tests added; pcm16/parse tests dropped with their code). Synthetic end-to-end reproduces T4.1 bench. **User smoke test pending (L-004): live mic, `--device`, Ctrl+C flush.**

### T4.4 — Wire Codex translation into live loop

**Status:** SHIPPED 2026-06-08. `CodexTranslator` in `live_stt.py` (app-server JSON-RPC per D-011): sequential turns preserve EN order; numbered `JA n:`/`EN n:` lines keep interleaved pairs unambiguous; startup warm-up turn absorbs the ~3 s uncached-prompt cost (first real block JA+EN ≈1.05 s). Degradation: missing CLI/failed init → JA-only at startup; 3 consecutive turn failures → JA-only for session; backlog >50 drops oldest; `--no-translate` skips. Synthetic 9-block session: 9/9 EN ordered, ~1 s cadence, 0 failures.

### T4.5 — Cleanup

**Status:** SHIPPED 2026-06-08. `google-genai` + `python-dotenv` removed (25 pkgs uninstalled; spike `load_dotenv` lines dropped — export keys manually if re-running historical metered benches); `list_live_models.py` + `.env` deleted; README/orientation rewritten for local-STT+Codex architecture; D-001/D-003 superseded-by-D-009; L-002/L-003 historical, L-004 rescoped; SPIKE_REPORT*.md kept as history.

---

## T5 series — regression-testability (ADR: D-014)

Make the local STT pipeline deterministically replayable so agents can catch
segmentation/transcript regressions without a mic. No new app features; the live-mic
path behavior is unchanged.

### T5.1 — Deterministic WAV replay / evaluation path

**Status:** SHIPPED 2026-06-15. `replay.py` drives the **exact** production loop
(`live_stt.worker` via a new optional `on_segment` hook — mic path behavior unchanged)
over a WAV and reports per-segment segmentation (start/length), decode latency + RTF,
and transcript: `uv run python replay.py WAV [--engine k2v2|parakeet] [--json]`. Golden
regression `tests/test_replay.py` vs `tests/replay_goldens.json` (characterization
snapshot of the real pipeline; regenerate via `tests/gen_replay_goldens.py`) asserts
segment count + per-segment text + boundary (±0.1 s), never the CPU-variable latency;
skips when models/WAVs absent. Reproduces D-010 quirks (ジェミニ→ゼミニ, 文→分) and the
0.7 s-silence splits. Retired the now-superseded bench harness (`prototype_local.py` was
a drifted copy of this loop). 30 tests green.

### T5.2 — Parakeet-engine goldens

**Status:** SHIPPED 2026-06-15. `replay_goldens.json` is now engine-first
(`engine → clip_id → {n_segments, segments, …}`; the redundant per-clip `engine` field
dropped); `gen_replay_goldens.py` regenerates both `k2v2` + `parakeet`, skipping (with a
warning) any engine whose weights are absent; `tests/test_replay.py` is parametrized over
`(engine, clip_id)` pairs with per-engine model gating. 35 tests green (was 30: +5 parakeet
goldens). Parakeet snapshot reproduces the D-010 quirks (ジェミニ→`jeミinapi`, numeral
`2つ目`, lowercase `api`) and wins the `文` homophone (`最初の文です` vs k2v2's
`最初の分です`). Tooling-only; no `live_stt.py`/CLI change → no new smoke surface.

### T5.3 — Real-recorded JA corpus

**Status:** SHIPPED 2026-06-15. The gate ("user records/sources clips") was
dissolved by web-fetch: L-004 blocks only *mic capture*, not network download
(CLAUDE.md grants network access), so the agent fetched the clips itself. 7 real
Common Voice 8.0 JA clips (CC0) added to the gitignored cache: 5 single utterances
+ 2 concatenations of independent real utterances joined by real silence (0.7 s ->
3 seg, 2.0 s -> 2 seg; mirrors synthetic medium/paused, no continuous-render
artifact per D-010). Sourced via the HF datasets-server `/rows` API on the ungated
Parquet mirror `japanese-asr/ja_asr.common_voice_8_0` (a few labeled samples, tiny
download; MP3 decoded by soundfile/libsndfile -> no ffmpeg). `tests/fetch_real_clips.py`
(committed; pins dataset revision + row indices) writes the WAVs (internal path,
L-016) + a `tests/real_clips.json` manifest, which `gen_replay_goldens.py` merges
into its clip list. 49 tests green (was 35: +14 = 7 real clips x 2 engines).
Real-acoustic signal the synthetic corpus lacked: katakana フィリピン decoded
correctly; engines diverge on proper nouns / voicing (松井 k2v2 / 松居 parakeet;
バック k2v2 / パック parakeet; 午後七時 / 午後7時) — captured as characterization
goldens. Tooling-only; no `live_stt.py`/CLI change -> no new smoke surface.

### Coverage split — agent-verifiable (replay) vs user-only (smoke)

| Behavior | Replay covers (agent, no mic) | User-only smoke |
|---|---|---|
| WAV → resample → VAD segmentation (count + boundaries) | ✅ | |
| RingBuffer pre-pad re-slice → sherpa decode → transcript | ✅ | |
| Per-segment decode latency / RTF | ✅ (report-only) | |
| JA-only path (translator absent) | ✅ | |
| Mic capture (`sd.InputStream`, `audio_callback`, `call_soon_threadsafe`) | | ✅ |
| Device enumeration / `--device` selection | | ✅ |
| Real-time latency feel + VAD endpointing on live speech | | ✅ |
| Ctrl+C / signal mid-utterance flush + translator drain | | ✅ |
| Multi-hour session (quota burn, thread rotation, memory) | | ✅ |
| Live Codex translation cadence/interleave | | ✅ (leg correctness itself is agent-verifiable via synthetic turns) |

Authoritative user-only list: `.agent/orientation.md` § "Smoke-test constraints".

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
