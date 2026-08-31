# Archive — T1–T8 pre-milestone build-out

Closed-work detail lifted out of `.agent/roadmap.md` under the attached-state rule (`CLAUDE.md` § Claude Code); the roadmap keeps the summary that names this record. Rows below are verbatim. Range `8ec8482..e3a654c`: Gemini-era CLI → local-STT rewrite → translator hardening. Predates the milestone/unit vocabulary and context records.

- T1.1 ✓ Exponential reconnect backoff (1s→30s, resets after stability) — pre-Live-API; moot since D-009.
- T1.2 ✓ Timestamped `-o` output file (ISO-8601 local prefix per block).
- T1.3 ✓ Audio device selection — `--list-devices`, `--device N`.
- T1.4 ✓ Graceful shutdown — partial-turn flush in `finally`; Ctrl+C mid-utterance persists.
- T2.1 ✓ Pure-function tests (`tests/test_audio.py`: resample/pcm16/emit).
- T2.3 ✓ Structured logging — 5 stderr sites → `logger`; `_StderrFormatter` clears the meter line only on a TTY.
- T3.1 ✓ Gemini Live API rewrite — superseded by D-009; history in git (D-005).
- T3.2 ✓ Long-session memory (resumption + context compression) — Gemini-era, superseded by D-009; history in git (D-005).
- T-CLEANUP-001 ✓ Spike artifact audit (cache gitignored).
- T-HOOK-001 ✓ `uv run pytest -q` pre-commit hook (`.githooks/`; `core.hooksPath`) → D-007.
- T4.1 ✓ Local JA STT engine via spike bench — k2v2 primary, parakeet A/B; sherpa-onnx + silero VAD; TTFT ≤0.10s → D-010.
- T4.2 ✓ Codex CLI install+auth+bench — Spark+low, tool features off, `developerInstructions`; p50 0.99s/turn → D-011.
- T4.3 ✓ Rewrite `live_stt.py` around local STT (Gemini machinery removed) → L-004 smoke debt opened.
- T4.4 ✓ Wire Codex translation into live loop (`CodexTranslator`, sequential turns, warm-up, JA-only degrade) → D-011.
- T4.5 ✓ Cleanup — google-genai/python-dotenv removed; Gemini backend superseded by D-009; spike reports since pruned (git history, D-005).
- T5.1 ✓ Deterministic WAV replay (`replay.py` drives the production `worker` via `on_segment`) + golden regression → D-014, reproduces D-010 quirks; retired the drifted bench harness.
- T5.2 ✓ Parakeet goldens — `replay_goldens.json` engine-first; tests parametrized per (engine, clip).
- T5.3 ✓ Real-recorded JA corpus — 7 Common Voice 8.0 clips fetched over network (L-004 blocks mic, not download) → L-017, characterization goldens; 49 tests.
- T6 ✓ Maintenance + security pass — deps current + `pip-audit` clean; codex leg reviewed (no remote-exploit surface, D-009); oversized-readline gap fixed; codex drift re-verified.
- T7 ✓ Proactive refactor — C1/C9/C2 behavior-preserving; rejections recorded; → L-018 (no re-bench when no CLI drift).
- T8.1 ✓ Non-blocking worker-stop sentinel — shutdown `finally` uses the evict-then-put idiom; fixes dead-worker + full-queue hang; test `test_shutdown_sentinel_lands_on_full_audio_queue_without_blocking`.
- T8.2 ✓ Live-mic smoke + soak checklist (`.agent/memory.md` § Smoke) — 7-item live pass + soak watching only in-code observables (meter `q=`/`drop=`, `TRANSLATE_ROTATE_TURNS`, `account/rateLimits/read` quota) → L-020. Runnable, not yet user-passed.
- T8.3 ✓ Wake the turn collect-loop on codex EOF — EOF cleanup enqueues one error sentinel onto `_notes`; a turn parked mid-collect (turn/start resolved, no pending request to fail) raises via the error branch and degrades in <2 s vs waiting out TRANSLATE_TIMEOUT_S (D-009); graceful close() cancels mid-readline → stays silent. Tests `test_turn_wakes_on_eof_under_timeout`, `test_graceful_close_enqueues_no_sentinel`.
- T8.4 ✓ `tests/test_translator.py` — first CodexTranslator regression net: 3-strike disable+reset, backlog evict-oldest, sentinel-on-full-queue, `_read_loop` dispatch/EOF branches. In-memory StreamReader+FakeProc, `asyncio.run` per test, no new dep.
- T8.5 ✓ Surface the two silent translator degradations — `_read_loop` logs the EOF→JA-only flip once (guarded on `enabled`); `submit` counts evictions in `dropped_translations` → meter `tdrop=` (>0 only, mirrors `drop=`). Tests `test_eof_logs_once_and_disables`, `test_submit_evicts_oldest_and_counts`; meter render is user-smoke (L-004).
- T8.6 ✓ Refuse to enable a dead app-server after warm-up — adversarial-review Finding 1 (HIGH): a warm-up turn that completes seconds before the server dies left `start()` flipping `enabled=True` over a finished reader, stranding every later turn on an unresolved turn/start until TRANSLATE_TIMEOUT_S (the silent-degrade class T8.5 targeted, reopened in the warm-up window). `start()` now checks `_reader_task.done()`/`returncode` before enabling. Tests `test_start_refuses_dead_server_after_warmup` (guard-disabled run confirms it fails — non-vacuous), `test_translate_degrades_to_ja_only_on_eof_under_timeout` (Finding 3: end-to-end of T8.3 through `_translate`, not just `_turn`). Findings 2 (EOF cleanup logs once; the 3-strike second log is `_translate`'s, distinct+correct) and 4 (close() runs only after `translator_task` ends — no mid-collect turn) rejected as correctly-scoped.
