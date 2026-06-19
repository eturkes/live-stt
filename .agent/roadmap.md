# Roadmap — live-stt

Canonical plan + status. Pick the lowest-numbered OPEN task; restate its acceptance criteria before coding. Trajectory: single-file tool, simplicity over completeness — no frameworks, no config, no premature abstraction. Status legend: OPEN · SHIPPED · DEFERRED · SUPERSEDED · OUT-OF-SCOPE · REJECTED.

## Status
- Phase: **T8 hardening/quality pass** (no new features — user directive). Shipped through **T8.5**; T8's coding tasks are closed — only the standing live-mic user smoke remains (L-004).
- Open: none actionable by an agent (see § Open). Standing debt: the live-mic user smoke + soak, user-only.
- Architecture stable since the 2026-06-08 re-arch (D-009): mic → resample → silero VAD → RingBuffer pre-pad → sherpa-onnx decode (JA) → `CodexTranslator` (JA→EN via persistent `codex app-server`, D-011) → `emit_line`; degrades to JA-only when codex is absent/failing.
- Standing debt: the live-mic path has had no real user smoke since the re-arch (L-004). Runnable procedure: `.agent/memory.md` § Smoke. T8.2 made it runnable, not closed — it still needs an actual user pass.

## Open (do these; lowest ID first)

None actionable by an agent. The sole remaining T8 item is the live-mic user smoke + soak (L-004; procedure in `.agent/memory.md` § Smoke) — user-only, unverifiable in the sandbox. A fresh session handed an empty task should surface this to the user rather than invent work; new coding tasks come from the user or a roadmap-generation pass (L-020).

## Shipped (ID · outcome · decisions/lessons produced)
- T1.1 ✓ Exponential reconnect backoff (1s→30s, resets after stability) — pre-Live-API; moot since D-009.
- T1.2 ✓ Timestamped `-o` output file (ISO-8601 local prefix per block).
- T1.3 ✓ Audio device selection — `--list-devices`, `--device N`.
- T1.4 ✓ Graceful shutdown — partial-turn flush in `finally`; Ctrl+C mid-utterance persists.
- T2.1 ✓ Pure-function tests (`tests/test_audio.py`: resample/pcm16/emit).
- T2.3 ✓ Structured logging — 5 stderr sites → `logger`; `_StderrFormatter` clears the meter line only on a TTY.
- T3.1 ✓ Gemini Live API rewrite → D-001 (superseded by D-009). History: `SPIKE_REPORT.md`.
- T3.2 ✓ Long-session memory (resumption + context compression) → D-003 (superseded by D-009), L-002.
- T-CLEANUP-001 ✓ Spike artifact audit (cache gitignored).
- T-HOOK-001 ✓ `uv run pytest -q` pre-commit hook (`.githooks/`; `core.hooksPath`) → D-007.
- T4.1 ✓ Local JA STT engine via spike bench — k2v2 primary, parakeet A/B; sherpa-onnx + silero VAD; TTFT ≤0.10s → D-010.
- T4.2 ✓ Codex CLI install+auth+bench — Spark+low, tool features off, `developerInstructions`; p50 0.99s/turn → D-011.
- T4.3 ✓ Rewrite `live_stt.py` around local STT (Gemini machinery removed) → L-004 smoke debt opened.
- T4.4 ✓ Wire Codex translation into live loop (`CodexTranslator`, sequential turns, warm-up, JA-only degrade) → D-011.
- T4.5 ✓ Cleanup — google-genai/python-dotenv removed; D-001/D-003 superseded by D-009; SPIKE_REPORT*.md kept as history.
- T5.1 ✓ Deterministic WAV replay (`replay.py` drives the production `worker` via `on_segment`) + golden regression → D-014, reproduces D-010 quirks; retired the drifted bench harness.
- T5.2 ✓ Parakeet goldens — `replay_goldens.json` engine-first; tests parametrized per (engine, clip).
- T5.3 ✓ Real-recorded JA corpus — 7 Common Voice 8.0 clips fetched over network (L-004 blocks mic, not download) → L-016, characterization goldens; 49 tests.
- T6 ✓ Maintenance + security pass — deps current + `pip-audit` clean; codex leg reviewed (no remote-exploit surface, D-009); oversized-readline gap fixed; codex drift re-verified.
- T7 ✓ Proactive refactor — C1/C9/C2 behavior-preserving; rejections recorded; → L-018 (no re-bench when no CLI drift).
- T8.1 ✓ Non-blocking worker-stop sentinel — shutdown `finally` uses the evict-then-put idiom; fixes dead-worker + full-queue hang; test `test_shutdown_sentinel_lands_on_full_audio_queue_without_blocking`.
- T8.2 ✓ Live-mic smoke + soak checklist (`.agent/memory.md` § Smoke) — 7-item live pass + soak watching only in-code observables (meter `q=`/`drop=`, `TRANSLATE_ROTATE_TURNS`, `account/rateLimits/read` quota) → L-020. Runnable, not yet user-passed.
- T8.3 ✓ Wake the turn collect-loop on codex EOF — EOF cleanup enqueues one error sentinel onto `_notes`; a turn parked mid-collect (turn/start resolved, no pending request to fail) raises via the error branch and degrades in <2 s vs waiting out TRANSLATE_TIMEOUT_S (D-009); graceful close() cancels mid-readline → stays silent. Tests `test_turn_wakes_on_eof_under_timeout`, `test_graceful_close_enqueues_no_sentinel`.
- T8.4 ✓ `tests/test_translator.py` — first CodexTranslator regression net: 3-strike disable+reset, backlog evict-oldest, sentinel-on-full-queue, `_read_loop` dispatch/EOF branches. In-memory StreamReader+FakeProc, `asyncio.run` per test, no new dep.
- T8.5 ✓ Surface the two silent translator degradations — `_read_loop` logs the EOF→JA-only flip once (guarded on `enabled`); `submit` counts evictions in `dropped_translations` → meter `tdrop=` (>0 only, mirrors `drop=`). Tests `test_eof_logs_once_and_disables`, `test_submit_evicts_oldest_and_counts`; meter render is user-smoke (L-004).

## Rejected (recorded so they are not re-litigated)
T8 screen:
- GitHub Actions CI mirroring the local hook — ~zero marginal catch for a single-user/AI-only repo; the one novel hazard (L-009 moved-venv) is absent from CI's fresh-clone env; D-007 already chose the minimal local hook (L-019 productivity theater).
- Drain residual codex notes at turn entry — self-contradicting: the next turn's collect loop already drains leaked notes; worst case is one self-healing dropped EN line.
- Assert `check_models` missing-file branch — tautological change-detector against a hardcoded literal; a regression fails loudly on next manual run.
- Bound `_notes` queue (DoS) — incoherent threat model: flooding needs the locally-spawned, user-authenticated codex CLI to attack its own client, already trusted.
- CodexTranslator thread-rotation unit test — tautological (asserts the modulo in the test body, not the live code).

T7 screen: merge `submit`/`submit_sentinel` (deliberate drop-vs-must-land split); cross-file dedup of WAV loaders/writers (different formats/sources); centralize the triplicated `CACHE` constant; prune overlapping resample tests.

## Deferred
- T2.2 — Parameterize source language. Tool is Japanese-only by design (user-confirmed). Revisit if the use-case expands: add `--language`, template into the system instructions, dynamic 2-letter ISO prefix in `emit_line`, extend `test_emit_*`. Approach stays valid; doesn't pay rent today.

## Superseded
- T-BACKENDS-001 — Backends spike, superseded by the T4 series (user ruled out metered API keys, removing Deepgram + OpenAI Realtime candidates). `SPIKE_REPORT_BACKENDS.md` stands as history; its "populate keys" recommendation is void.

## Out of scope (do not redebate)
Config files / YAML / TOML for tunables (constants at the top of `live_stt.py` are the config surface) · multi-mic mixing · speaker diarization · web UI · auth / multi-user · metrics dashboards beyond the level meter · package split / multi-module layout (`.agent/memory.md` D-002).

## Decisions pending from user
None. (Last resolved 2026-05-18: language scope = Japanese-only; pre-commit hook = yes.)
