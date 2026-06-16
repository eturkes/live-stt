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

## T6 — Maintenance + security pass

Periodic upkeep per CLAUDE.md (security audits + keep software current); no new
features. **Status: SHIPPED 2026-06-15.**

- **Deps + CVEs:** runtime deps already latest (numpy 2.4.6, sherpa-onnx[-core] 1.13.2
  = floor *and* newest, sounddevice 0.5.5); dev tooling bumped pytest 9.0.3->9.1.0 +
  ruff 0.15.16->0.15.17 (lock + venv). `pip-audit` (`uv export | uvx pip-audit -r`) =
  no known vulnerabilities. Suite + ruff + pyright green post-bump.
- **Security review of the Codex leg** (the only non-local input surface): clean —
  `create_subprocess_exec` (no shell), sandbox read-only + approvals never +
  server-requests auto-denied, malformed-line skip, EOF/turn-failure -> JA-only
  (D-009); no eval/os.system/shell/pickle/network-listener; `--device` int-typed,
  `--engine` choices-bound, `-o` user-owned. No remotely-exploitable surface. One
  low-sev gap fixed: `_read_loop`'s `readline()` wrapped so an oversized-line (>64 KiB
  asyncio limit) / broken-transport error routes into the existing EOF cleanup
  (immediate clean JA-only vs timeout-delayed).
- **Codex drift re-verify:** CLI 0.137.0 (D-011 bench) -> 0.139.0; a synthetic turn
  through the real `CodexTranslator` confirms `gpt-5.3-codex-spark`+low+features-off+
  `developerInstructions` valid, clean JA->EN, 0 failures. Non-breaking.

---

## T7 — Proactive refactor pass

Code-health pass per CLAUDE.md (periodic proactive refactor); no new features, no
behavior change. **Status: SHIPPED 2026-06-15.**

- **Applied (live_stt.py):** C1 `CodexTranslator.close()` `except (TimeoutError,
  Exception)` -> `except Exception` (TimeoutError is an Exception subclass -> the
  tuple was redundant; the lookalike `(CancelledError, Exception)` forms elsewhere
  ARE necessary and were left). C9 named the control-plane RPC timeout
  `CODEX_CONTROL_TIMEOUT_S = 10`, replacing 3 bare `10`s (initialize +
  thread/start x2), matching the `TRANSLATE_TIMEOUT_S` convention. C2 removed
  `meter()`'s 8-line global->local hoisting (inert at the 10 Hz meter cadence).
- **Rejected (so they aren't re-litigated):** merge `submit`/`submit_sentinel`
  (deliberate drop-vs-must-land split); cross-file dedup of the WAV loaders/writers
  (different formats/sources); centralize the triplicated `CACHE` constant; prune
  overlapping resample tests. Audit detail: `.agent/scratch/2026-06-15_T7.md`.
- **Verified:** 49 tests green, ruff clean, pyright 0 errors, import OK. Changes are
  behavior-preserving (C1/C9 provably inert, C2 logic-identical) -> codex leg not
  re-benched (not a CLI drift, L-018); no new smoke surface.

---

## T8 series — Hardening & quality pass (workflow-generated 2026-06-15)

Generated by a dynamic multi-agent workflow (6 hardening lenses → adversarial
philosophy screen → synthesis): 11 candidates, 6 survived, merged to **5 items**.
Scope: **hardening/quality only, no new features** (user directive). Claims re-verified
against live `live_stt.py` before landing; tasks written by symbol/behavior, not line
number (agent line refs drift). Honest yield is modest — correct for a mature codebase
(L-019). Priority = task order (T8.1 highest).

### T8.1 — Non-blocking worker-stop sentinel (fix shutdown deadlock)

**Status:** SHIPPED 2026-06-16. **Effort:** S. **Verification:** agent.

**Shipped:** `run_session`'s shutdown `finally` no longer blocks on
`await audio_q.put(None)` — replaced with the in-file evict-then-put idiom
(`put_nowait`; on `QueueFull`, `get_nowait` one stale block, retry), the same
pattern as `submit_sentinel`. Normal Ctrl+C path is effect-identical (a put with
spare capacity lands immediately, then `break`); only the pathological
dead-worker + full-queue case changes (drops one queued block instead of hanging
to SIGKILL). No new symbol. Test `test_shutdown_sentinel_lands_on_full_audio_queue_without_blocking`
(tests/test_audio.py) exercises the idiom on a synthetic full
`asyncio.Queue(maxsize=4)` under a 1 s `wait_for` (must not fire) and asserts the
sentinel lands while the oldest block is evicted (1,2,3 survive). 50 tests green
(+1), ruff + pyright clean. Live Ctrl+C-in-terminal flush stays user-smoke (L-004,
captured in T8.2) but is structurally unchanged.

**Problem (named failure mode):** `run_session`'s shutdown `finally` does a *blocking*
`await audio_q.put(None)` on the bounded `audio_q` (`AUDIO_QUEUE_MAX`). `worker()` is the
queue's only consumer; on an in-worker exception it calls `state.request_stop()` and
**returns** (consumer gone). The mic callback can have filled the queue to capacity
before `stream.stop()`. The blocking put on a full queue with a dead consumer parks the
loop forever — signal handlers route Ctrl+C to `request_stop` (not `KeyboardInterrupt`),
so escape is SIGKILL-only and the `-o` file is left unclosed. A recoverable worker death
becomes a permanent hang.

**Acceptance criteria:**
- Shutdown tail no longer contains a bare blocking `await audio_q.put(None)`; the
  sentinel uses the in-file evict-then-put idiom (`put_nowait`; on `QueueFull`,
  `get_nowait` one stale item then retry).
- Agent test (no mic): a synthetic `asyncio.Queue(maxsize=N)` filled to capacity with no
  consumer accepts the sentinel without blocking (assert completion under a short
  `wait_for` that must not fire).
- Normal Ctrl+C path unchanged: sentinel lands, `worker()` flushes the VAD and exits;
  replay tests stay green; `import live_stt` clean. ruff + pyright clean; no new symbol.

**Approach:** Transplant `submit_sentinel`'s evict-then-put to the one blocking put in
`run_session`'s `finally`. Dropping one queued audio block during shutdown is harmless.

### T8.2 — Repeatable live-mic smoke + multi-hour soak checklist

**Status:** OPEN. **Effort:** S. **Verification:** user-smoke.

**Problem:** The one acknowledged standing debt — the live path has had no user smoke
since the 2026-06-08 re-architecture (re-derived ad-hoc in 3 journal entries). The
user-only behaviors exist as a coverage-split table but **no runnable procedure**; the
multi-hour soak row has zero steps and no named observable, so a real
mic/latency/Ctrl+C/leak/quota regression can't conclude pass/fail.

**Acceptance criteria:**
- A short markdown checklist (~40 lines): (a) a numbered live-mic pass, (b) a soak
  section; each item states its pass criterion **and** the `live_stt.py` observable
  backing it.
- Soak observables are the real in-code ones: meter `q=`/`drop=`, thread rotation near
  `TRANSLATE_ROTATE_TURNS`, quota via `account/rateLimits/read`. No invented metric, no
  new code.
- Live-mic items cover exactly the user-only coverage-split rows: mic capture,
  `--device`/`--list-devices`, latency feel, Ctrl+C mid-utterance flush+persist, `-o`
  persistence.
- PLAN T4.3 + orientation "Smoke-test constraints" link to the new file so the recurring
  "Did not verify (L-004)" disclaimer points at a fixed list. No code/CLI/output change.

**Approach:** Add `.agent/smoke.md`; link from PLAN + orientation. Doc only.

### T8.3 — Wake the turn collect-loop on codex EOF (prompt mid-turn degrade)

**Status:** OPEN. **Effort:** S. **Verification:** agent. **Synergy:** locked by T8.4's
read-loop test.

**Problem (named failure mode):** `_turn` collects via `while True: await
self._notes.get()`. `_read_loop`'s EOF cleanup fails pending *requests* but never
sentinels `_notes`. If codex dies **after** `turn/start` resolves but **before**
`turn/completed` (turn already collecting, no pending request to fail), the `get()`
blocks until the outer `wait_for(_turn, TRANSLATE_TIMEOUT_S)` fires a full 15 s before
JA-only — contradicting the D-009 "degrade promptly" contract at the one boundary the
T6 readline-hardening did not cover.

**Acceptance criteria:**
- `_read_loop`'s EOF cleanup enqueues one sentinel note onto `_notes` that `_turn`'s
  existing `error`/no-`willRetry` branch raises on.
- Agent test (no subprocess/mic): fake `_proc`/stdout `StreamReader`, resolve
  `turn/start`, start `_turn` as a task, `feed_eof`; the in-flight `_turn` raises well
  under `TRANSLATE_TIMEOUT_S`.
- Fires only on genuine death: `close()` cancels `_reader_task` (CancelledError escapes
  the `(ValueError, OSError)` catch → cleanup not reached), so graceful close enqueues no
  sentinel (assert no spurious raise). Output identical JA-only, just prompt; ruff/pyright
  clean.

**Approach:** One line in the EOF cleanup —
`self._notes.put_nowait({"method": "error", "params": {}})` — reusing `_turn`'s error
branch → `_translate`'s per-block fallback.

### T8.4 — `tests/test_translator.py`: lock degradation + backlog + read-loop branches

**Status:** OPEN. **Effort:** M. **Verification:** agent.

**Problem:** Zero regression coverage on the documented graceful-degradation contract
(D-009 hard requirement, D-011) and the sole non-local input boundary T6 hardened. Tests
reference only `check_models`/`RingBuffer`/`emit_line`/`resample`; the entire
`CodexTranslator` is untested. A refactor can silently break degradation (3-strike never
disables → hang every block; `submit` eviction inverts → drops newest caption;
`_read_loop` stops auto-denying server requests) with nothing to catch it pre-session.

**Acceptance criteria:**
- New `tests/test_translator.py`, no new dependency (`asyncio.run` per test, no
  pytest-asyncio), no subprocess/mic, no `live_stt.py` change.
- 3-strike disable: `enabled=True`, `_proc=None` (so `_abort_turn` early-returns), `_turn`
  monkeypatched to raise; 3× `_translate` flips `enabled` False on the 3rd with
  `_failures==3`; a passing `_turn` resets `_failures`.
- Backlog eviction: filling `queue` to `TRANSLATE_QUEUE_MAX` then `submit`-ing keeps
  `qsize==cap`, proves oldest evicted + newest survives; `submit_sentinel` on a full queue
  lands `None` at `size==cap`.
- `_read_loop` dispatch via `StreamReader` + `FakeProc`: malformed line skipped, id+result
  resolves a pending future, a notification lands in `_notes`, `feed_eof` sets
  `enabled==False` and fails pending futures (also locks T8.3/T8.5 once they land).

**Approach:** Single pytest file, in-memory fakes (`StreamReader` + tiny `FakeProc`),
`asyncio.run` per test. Cover only named-failure branches, not the happy-path live turn
(stays user-smoke per L-019).

### T8.5 — Surface the two silent translator-degradation events

**Status:** OPEN. **Effort:** S. **Verification:** both. **Merged:** the standalone
EOF-log + tdrop-meter observ candidates.

**Problem (named failure mode):** Two silent EN-leg failures, asymmetric with the audio
side. (1) `_read_loop`'s EOF cleanup flips `enabled=False` with **no log**, while both
sibling degradations (startup, 3-strike) log — so a codex death in an idle gap silently
becomes permanent JA-only, indistinguishable from "operator stopped speaking." (2)
`submit`'s `QueueFull` handler evicts the oldest caption with no counter/log, while the
audio side surfaces its analogue (`state.dropped` → `drop=N` meter field). The operator
cannot diagnose a dead or load-shedding translation leg.

**Acceptance criteria:**
- `_read_loop` logs the EOF degradation exactly once (e.g. `logger.error("codex
  app-server exited; JA-only for the rest of the session")`) on the existing TTY-aware
  logger, guarded against repeat.
- `submit` increments an eviction counter on `QueueFull` (init beside the other counters)
  and surfaces it: default a `tdrop=N` meter field (shown only when >0, mirroring
  `drop=`); acceptable fallback if meter-creep is judged too much = a throttled
  `logger.warning`.
- Agent test (fold into `test_translator.py`): EOF path asserts the one error record +
  `enabled` False via caplog; `submit` past `TRANSLATE_QUEUE_MAX` asserts the counter
  increments while `qsize` stays capped.
- No new output mode/stream/config/abstraction; only `live_stt.py`; ruff/pyright clean;
  the two existing degradation logs now symmetric with the EOF case.

**Approach:** (a) one guarded `logger.error` at the EOF break; (b) `dropped_translations`
counter + conditional `tdrop=` on the existing meter (the one diagnostic surface PLAN
Out-of-scope permits; `drop=` is precedent), throttled-log fallback documented.

### Rejected by the screen (recorded so they aren't re-litigated)

- **GitHub Actions CI mirroring the local hook** — ~zero marginal catch for a
  single-user/AI-only repo where the user owns all remote ops; the one novel hazard
  (L-009 moved-venv) is structurally absent from CI's fresh-clone env; D-007 already
  chose the minimal local hook (L-019 productivity theater).
- **Drain residual codex notes at turn entry** — self-contradicting: the next turn's
  collect loop already drains leaked notes; worst real impact is a single self-healing
  dropped EN line, not persistent desync.
- **Assert `check_models` missing-file branch** — tautological change-detector (pins a
  hardcoded literal against itself); a regression fails loudly on the next manual run.
- **Bound `_notes` queue (DoS)** — incoherent threat model: the only unbounded sub-case
  needs the locally-spawned, user-authenticated codex CLI to flood its own client, a
  process already trusted for translation correctness.
- **CodexTranslator thread-rotation unit test** — tautological (asserts the modulo in the
  test body, not against the live `_translate` code it claims to guard).

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
