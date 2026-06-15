# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

---

## 2026-06-15 — T5.2: parakeet-engine goldens (engine-first regression coverage)

**Trigger:** User (`/session-prompt`, blank override → roadmap). T5.2 was the lowest-numbered agent-actionable open task; T5.3 is user-gated (needs real mic recordings, L-004). Goal: extend the T5.1 golden regression from `k2v2`-only to both engines. Confirmed direction before coding.

**Shipped:** `replay_goldens.json` restructured **engine-first** (`engine → clip_id → {n_segments, segments, ja_ref, purpose}`); per-clip `engine` field dropped as redundant (engine is the key). `gen_replay_goldens.py` loops `ENGINES = [k2v2, parakeet]`, guarding each with `check_models` (skip+warn on absent weights, mirroring the existing clip-skip). `tests/test_replay.py` flattens goldens to `(engine, clip_id)` cases and parametrizes over them; `_resources_ready` gates per-engine. No `live_stt.py`/`replay.py` change — `replay.py` already accepted `--engine parakeet`.

**Verified (agent-checkable):** regenerated both engines; **35 tests green** (was 30: +5 parakeet); ruff clean (fixed one E501 — the `{engine}/` prefix pushed the progress print to 102 > 100 at line-length 100; wrapped it); `uvx pyright@1.1.410` 0 errors; JSON top-level `[k2v2, parakeet]`, 10 cases. Parakeet snapshot reproduces D-010 quirks (ジェミニ→`jeミinapi`, `2つ目` numeral, lowercase `api`) and wins the `文` homophone (`最初の文です` vs k2v2 `最初の分です`) — characterization values, not asserted against idealized refs.

**Did not verify (user smoke, L-004):** none newly affected — tooling-only (goldens + generator + test); no mic/`--device`/latency/Ctrl+C/multi-hour surface touched; `live_stt.py` untouched.

**Memory:** PLAN T5.2 → SHIPPED; D-014 amendment (engine-first shape; first revisit-if resolved); orientation goldens row `(k2v2)` → engine-keyed `(k2v2 + parakeet)`. No new lesson (ruff line-length=100 is project config, not generalizable).

---

## 2026-06-15 — T5.1: deterministic WAV replay regression path; bench harness retired

**Trigger:** User (`/session-prompt` override) — make live-stt regression-testable (no new features): a deterministic WAV replay/eval path through the **exact** VAD + RingBuffer + sherpa decode loop; reuse/retire the spike harness; minimal CLI/docs/tests; PLAN/.agent split of agent-verifiable replay vs user-only smoke. Confirmed 3 design choices before coding: real-`worker` hook (not a copy) / retire the whole bench harness / gitignored corpus + skip-if-absent.

**Shipped (D-014):** `replay.py` drives `live_stt.worker` over a WAV via a new optional `on_segment(start, n, seg_len, decode_s, text)` hook — the mic path passes `on_segment=None`, so live behavior is unchanged (only `live_stt.py` edits: `import time` + the guarded hook + a docstring note). It reports per-segment segmentation + decode latency/RTF + transcript; `--json` for machine read (worker's `emit_line` stdout captured via `redirect_stdout` so the JSON stays valid). Golden regression `tests/test_replay.py` = 3 always-run WAV-loader tests + 5 characterization goldens (`replay_goldens.json`, k2v2) asserting segment count + per-segment text + boundary (±0.1 s), never the CPU-variable latency; skips when models/clips absent. Reproduces D-010 quirks (ジェミニ→ゼミニ, 文→分) + the 0.7 s-silence splits → it re-tests the real pipeline, not an idealized one.

**Retired:** all 11 runnable `spike/backends/*.py` (`prototype_local.py` was a drifted copy of `worker`). Kept: `cache/*.wav` (gitignored+deny-listed replay corpus), `*.md` history, `codex_ws/AGENTS.md`. Memory/docs: PLAN T5 section + "Coverage split" table; README "Regression testing"; orientation file-map + replay-covered smoke pointer; D-014; L-016.

**Verified (agent-checkable):** 30 tests green (22 + 3 loader + 5 golden); ruff clean; `uvx pyright@1.1.410` 0 errors; `import live_stt, replay`; synthetic-WAV CLI smoke (human + `--json`, 0-segment branch).

**Did not verify (user smoke, L-004):** live mic / `--device` / real-time latency feel / Ctrl+C mid-utterance flush / multi-hour. None changed behaviorally (hook defaults None → mic path unchanged), but the `worker` signature did change, so flagged per policy.

**Spawned L-016:** the deny-list blocks a path on a tool/Bash command line but not a script's own runtime `open()` — gen/test construct the `spike/backends/cache` path internally to read the corpus.

---

## 2026-06-15 — CLAUDE.md sync: Headroom `.serena/` tracking + deny-list; scoped-commit doc

**Trigger:** User (`/session-prompt` override) — "I updated CLAUDE.md; do any work it implies." Diff (+3/−2): dropped the `# CLAUDE.md` H1; added a preferred-tooling bullet (`uv`/`pnpm`/`chromiumfish`); added the Headroom/`.serena/` bullet; changed the commit rule to require a scoped commit (scopedcommits.com).

**Triage:** Heading drop + tooling bullet → no project work (`uv` already used; `pnpm`/`chromiumfish` N/A to a Python/no-browser tool). Scoped-commit rule already met by history (`Tooling:`/`Maintenance:`/`Settings:` + co-author trailer match the `Scope: desc` spec) — documented only. All actionable work came from the `.serena/` bullet, which was out of sync.

**Shipped (D-013):** `.serena/` was entirely untracked; its nested `.gitignore` ignored `cache`+`project.local.yml` but **not** `memories`. Added `/memories` to `.serena/.gitignore`; git-tracked `project.yml` + `.serena/.gitignore` (`project.yml` verified portable — stock Serena config, no secrets/abs paths); deny-listed `Read()` on `.serena/{cache,memories,project.local.yml}` to honor "ignored by you." Orientation gains a `.serena/` row + deny enumeration + scoped-commit term (step 8).

**Verified (agent-checkable):** settings.json valid JSON; `git check-ignore` → all three subpaths ignored, `project.yml` tracked; `git add -n` stages exactly `project.yml` + `.serena/.gitignore`. No `live_stt.py`/runtime change → no smoke-test surface this session.

**Spawned L-015:** Headroom compresses tool reads, so long exact-match Edit `old_string`s can silently miss (this session: a `git log`-backtick diff invisible in my compressed view failed an Edit). Re-anchor on raw bytes when an Edit fails.

---

## 2026-06-08 — `compaction.sh` simplified to single-mode (verify + commit)

**Trigger:** User (`/session-prompt` override) — verify the user's edit to `compaction.sh` (made it token-efficient, stripped unneeded functionality); commit if it works.

**Verified (all green):** The edit drops the statusline (stdin-JSON) branch + ANSI coloring, leaving only the manual transcript-read path; the `c` color flag and the now-redundant `[ "$w" -gt 0 ]` guard (w is always set by the `case`) are gone. Exercised: normal → `23% 45K/200K`; `CLAUDE_CODE_DISABLE_1M_CONTEXT=1` (set in this env — explains the 200K window) → same; unset `CLAUDE_CODE_SESSION_ID` → falls back to newest `*.jsonl`, same; empty-usage awk branch → `? ?/1M`; `h()` 1M/1.5M/45K formatting correct; `sh -n` clean. Repo copy **byte-identical** to shared `$HOME/.claude/compaction.sh` → L-008 vendoring invariant holds (user updated both).

**Memory:** L-008 "Current state" re-synced dual-mode → single-mode. orientation.md row (`prints PCT USED/WINDOW`, needs `jq`) still accurate, untouched.

**Consequence to note (user-side):** if `compaction.sh` was ever wired as a Claude Code statusline command, it no longer consumes stdin JSON or emits color — it now reads the transcript (works as long as `CLAUDE_CODE_SESSION_ID` is set, or the newest transcript is the active session).

