# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

---

## 2026-06-15 — T8: hardening roadmap generated via dynamic workflow

**Trigger:** User (`/session-prompt` override): "The current roadmap is exhausted. Use a
dynamic workflow to create new roadmap items." Bootstrap confirmed T1-T7 all SHIPPED, no
OPEN task. User chose direction = **hardening & quality only (no new features)** +
**write-through** to PLAN.

**Method:** Ran a Workflow (multi-agent): 6 hardening lenses (error-paths, perf/
multi-hour, test-coverage/CI, security, operational-diagnostics, documented-debt critic),
each grounded by reading the real `live_stt.py` + `.agent/` memory, fanned out to
candidates -> per-candidate adversarial philosophy screen (L-001 name-the-failure, L-005
no-abstraction, L-019 no-padding, D-002, Out-of-scope, no-features gate) -> synthesis. 11
candidates, 6 survived, merged to **5 items**. Re-verified the highest-stakes claims
against live code before write-through; wrote tasks by symbol, not line number (refs were
approximate).

**Shipped (docs only):** PLAN.md +T8 series (T8.1-T8.5, all OPEN, priority-ranked) with
acceptance criteria + the 5 screen-rejected candidates recorded. T8.1 = real shutdown
deadlock (blocking `audio_q.put(None)` on a full queue after worker death -> SIGKILL-only,
`-o` unclosed); T8.2 = the standing live-mic/soak smoke debt as a runnable checklist (the
one acknowledged debt since 2026-06-08); T8.3 = wake the turn collect-loop on codex EOF
(prompt JA-only vs 15 s timeout); T8.4 = `tests/test_translator.py` locking the
degradation/backlog/read-loop branches (`CodexTranslator` is wholly untested); T8.5 =
surface the 2 silent EN-leg degradations (EOF log + `tdrop` counter). No `live_stt.py`
change this session.

**Verified (agent-checkable):** claims cross-checked against `live_stt.py` (the blocking
put in `run_session`'s finally; `submit_sentinel`'s evict-then-put idiom; `_turn`'s
`_notes.get()` collect loop; `_read_loop` EOF cleanup has no log/sentinel; `submit`'s
silent `QueueFull` evict vs the audio `drop=` precedent). No code changed -> 49 tests
still green (pre-commit hook re-runs on commit).

**Did not verify (user smoke, L-004):** none newly affected (docs only). The standing
live-mic / `--device` / latency / Ctrl+C / multi-hour debt is unchanged and is now itself
captured as OPEN task T8.2.

**Memory:** PLAN +T8 series; L-020 (workflow recipe for roadmap generation: the
adversarial screen carries the value; ground every finder in real code); journal pruned
oldest (T5.2). No new ADR (the workflow is a process, not an architecture choice; the
items earn ADRs if/when they ship).

---

## 2026-06-15 — T7: proactive refactor pass (minimal, evidence-driven)

**Trigger:** User (`/session-prompt`, blank override). Bootstrap found the roadmap
fully shipped (T1-T6); reported that and asked direction -> user chose a proactive
refactor pass. Scoped as T7.

**Method:** Read `live_stt.py` + `replay.py` + tests + tooling in full; cataloged
refactor candidates and screened each against L-001 (name the failure mode the edit
prevents), L-005 (less code, no abstraction), D-006 (don't churn `live_stt.py`).
Presented the screened set; user picked C1+C9+C2.

**Shipped (`live_stt.py` only):** C1 `close()` `except (TimeoutError, Exception)` ->
`except Exception` (redundant: TimeoutError is a subclass of Exception; the lookalike
`(CancelledError, Exception)` forms ARE necessary and were left untouched). C9 named
the control-RPC timeout `CODEX_CONTROL_TIMEOUT_S = 10` (3 bare `10`s: initialize +
thread/start x2; matches `TRANSLATE_TIMEOUT_S` house style; wrapped the one line that
crossed 100 cols). C2 removed `meter()`'s 8-line global->local hoisting (inert at the
10 Hz meter cadence, unlike the commented hot-path opts).

**Rejected (detail in scratch):** submit/submit_sentinel merge, cross-file WAV
loader/writer dedup, CACHE-constant centralization, resample-test pruning. Headline:
the code is mature; an honest refactor pass yields little, and that is the correct
outcome rather than a failure (-> L-019).

**Verified (agent-checkable):** 49 tests green (unchanged), ruff clean, pyright 0
errors, import OK. Behavior-preserving -> codex leg not re-benched (not a CLI drift,
L-018).

**Did not verify (user smoke, L-004):** the edits are inert at runtime; the standing
pre-2026-06-08-rearch live smoke set (mic / `--device` / latency feel / Ctrl+C flush /
multi-hour) is unaffected and still pending. C1 sits in the translator-teardown path
and C2 in the meter coroutine, but both are behavior/logic-identical.

**Memory:** PLAN +T7 (SHIPPED); L-019 (refactor-pass = audit; minimal honest output on
mature code is valid); journal pruned oldest (T5.1). No new ADR (changes too minor).

---

## 2026-06-15 — T6: maintenance + security pass (deps, codex-leg audit, drift re-verify)

**Trigger:** User (`/session-prompt`, blank override). Bootstrap found the PLAN roadmap
fully shipped (T1–T5 done, no OPEN task); I reported that and asked for direction — user
chose a maintenance + security pass. Scoped as T6.

**Shipped:** (1) Deps — runtime all at latest (sherpa-onnx 1.13.2 is floor+newest);
bumped dev tooling pytest 9.0.3→9.1.0 + ruff 0.15.16→0.15.17; `pip-audit` clean (no
known vulns). (2) Security-reviewed the Codex leg (only non-local input): strong posture
(no shell, sandbox/approvals locked, server-requests auto-denied, graceful JA-only
degradation) — no remotely-exploitable surface; one low-sev gap fixed (unwrapped
`_read_loop` `readline()` → wrapped, user-approved over document-only, so an
oversized-line/broken-transport error routes into the EOF cleanup). (3) Re-verified the
codex leg against CLI 0.139.0 (D-011 benched 0.137.0) via a synthetic `CodexTranslator`
turn — config valid, clean JA→EN, non-breaking.

**Verified (agent-checkable):** 49 tests green (unchanged count; the codex leg isn't in
pytest, so re-verified separately via synthetic `CodexTranslator` turns — 3 clean JA→EN
incl. one post-edit); ruff clean; pyright 0 errors; pip-audit clean.

**Did not verify (user smoke, L-004):** the `_read_loop` edit is in the codex path, not
the mic path. Still pending from the T4 re-arch and NOT closed here: live mic /
`--device` / real-time latency / Ctrl+C flush / multi-hour — the live path has had no
user smoke-test since 2026-06-08.

**Memory:** PLAN +T6 (SHIPPED); D-011 amendment (re-verified at 0.139.0 + reader-loop
hardening); L-018 (maintenance-pass recipe); journal pruned oldest (CLAUDE.md sync).

---

## 2026-06-15 — T5.3: real-recorded JA corpus via web-fetch (gate dissolved)

**Trigger:** User (`/session-prompt` override): "To unlock the current gate, can you
fetch Japanese recordings from the web?" T5.3 was OPEN/user-gated. Reframe: L-004
blocks only mic capture, not network fetch (CLAUDE.md network access) — so the agent
sources real clips itself. Confirmed source (Common Voice JA) + shape (singles +
concatenated) with the user before fetching.

**Shipped:** 7 real CV8.0-JA clips (CC0) in the gitignored cache — 5 single utterances
+ 2 concatenations of independent real utterances joined by real silence (0.7 s -> 3
seg, 2.0 s -> 2 seg; D-010 method, real voices). Fetched via the HF datasets-server
`/rows` API on the ungated Parquet mirror `japanese-asr/ja_asr.common_voice_8_0` (few
labeled samples; MP3 decoded by soundfile/libsndfile, no ffmpeg). `tests/fetch_real_clips.py`
(committed; pinned revision + row indices) writes WAVs (internal path, L-016) +
`tests/real_clips.json` manifest; `gen_replay_goldens.py` merges the manifest with the
inline synthetic CLIPS.

**Verified (agent-checkable):** 49 tests green (was 35: +14 = 7 clips x 2 engines); ruff
clean; pyright 0 errors (soundfile import scoped-ignored — a `uv run --with` dep, not a
project dep). Real-acoustic characterizations: katakana フィリピン correct; engine
divergence 松井/松居, バック/パック, 午後七時/午後7時; cv_multi -> 3 seg + cv_paused -> 2
seg confirm real-acoustic endpointing.

**Did not verify (user smoke, L-004):** none newly affected — tooling-only (new dev
fetch tool + manifest + goldens); `live_stt.py` untouched, no mic/`--device`/latency/
Ctrl+C/multi-hour surface.

**Memory:** PLAN T5.3 -> SHIPPED; D-014 amendment (2nd revisit-if resolved; web-fetch
substitutes mic-record); orientation file-map (+fetch_real_clips.py, +real_clips.json;
goldens row now bench+real); L-017 (HF rows-API fetch technique). Pruned oldest entry
(2026-06-08 compaction).
