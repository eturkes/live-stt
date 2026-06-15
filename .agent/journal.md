# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

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
