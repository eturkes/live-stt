# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

---

## 2026-06-16 — CLAUDE.md sync #3: cosmetic UI/UX-rule prose polish (no-op)

**Trigger:** User (`/session-prompt` override): "I updated the `CLAUDE.md`. If there is
any work you need to do in response, use this session to do so." Third consecutive
CLAUDE.md-sync session today; unlike the prior two syncs (substantive `.serena`/gitignore),
this diff (1+/1−) is a single-sentence **cosmetic prose polish** of the UI/UX rule: clause
reorder ("As it is largely human-authored" fronted) + de-parenthesized "...forced
optimizations present for LLM steering...". No new directive/constraint/fact/structural
requirement.

**Checked every surface the edit could touch (→ no work):** code/config — none (the UI/UX
rule has no code/config surface); live memory docs — no orientation/lessons/decisions entry
quotes the changed sentence (L-001 quotes a *different* CLAUDE.md rule, unchanged; journal
#2's "largely human-authored" is a historical diff description, left as-is); user-facing
README/CLI — unaffected (what compliance means is unchanged).

**Action:** committed the user's prose polish alone + this journal note. Confirmed scope
with the user first (chose "commit polish, end here" over recover-from-siblings / pivot-to-
T8.1). No code/config/doc change.

**Did not verify (user smoke, L-004):** none — no mic/`--device`/latency/Ctrl+C/multi-hour
surface touched. Standing live-mic/soak debt (T8.2) unchanged.

**Memory:** journal pruned oldest (T7 → git). No new lesson/ADR — a cosmetic doc tweak is
not a new failure mode or architecture choice (L-008 spirit: record state, not an "always").
Roadmap untouched — **T8.1** remains the lowest open task next session.

---

## 2026-06-16 — CLAUDE.md sync #2: `.serena/` committed-as-is, memories un-ignored

**Trigger:** User (`/session-prompt` override): "I updated the `CLAUDE.md`. If there is any
work you need to do in response, use this session to do so." Diff (8+/7−) is mostly cosmetic
— `# CLAUDE.md` title, backticks, Git-rule relocated higher, "Casual"→"Wasteful" file-dumps,
"largely human-authored", "`ignored_paths` in `.serena/project.yml`" made explicit. **One
substantive change** reverses last session: the `.serena/` rule dropped "home
`.serena/memories/` in the repo-root `.gitignore`" for "`.serena/` comes with its own
gitignore file and can be committed as is" + the new fact "Serena … memory system … disabled
globally".

**Verified the new claim:** `~/.serena/serena_config.yml` `excluded_tools:` lists all six
memory tools → Serena never writes `.serena/memories/`. Confirmed `.serena/memories/` is empty
and was protected only by the root reach-in (root `.gitignore:28`).

**User chose "fully as is"** (over self-contain / docs-only):
- Root `.gitignore`: removed `.serena/memories/`; left a one-line breadcrumb comment so the
  entry stops oscillating nested↔root across sessions. Nested `.serena/.gitignore` untouched
  (`/cache` + `/project.local.yml`) — now the only `.serena/` gitignore.
- `.serena/memories/` is now un-ignored by git (empty; memory disabled → stays empty; `.agent/`
  is the sole store). Still **Read-denied** in `.claude/settings.json` — the sanctioned
  do-not-read ≠ gitignore split.
- Docs: D-013 amendment #2 (reversal + verified disabled-globally); orientation `.serena/` row
  rewritten.

**Verified (agent):** `git check-ignore` confirms memories/ un-ignored; `project.local.yml`
still caught by nested; only 4 files modified (no stray `.serena/` staging — empty dir); 49
tests green (no code touched).

**Did not verify (user smoke, L-004):** none — config + docs only; no
mic/`--device`/latency/Ctrl+C/multi-hour surface. The standing live-mic/soak debt (T8.2) is
unchanged.

**Memory:** D-013 amendment #2; orientation `.serena/` row; journal pruned oldest (T6). No new
lesson (CLAUDE.md already states the sync rule; this is a one-line reversal, not a new failure
mode — L-008 spirit: record state, not an "always"). Roadmap untouched — **T8.1** remains the
lowest open task next session.

---

## 2026-06-16 — CLAUDE.md sync: Serena/gitignore + `ignored_paths` alignment

**Trigger:** User (`/session-prompt` override): "I updated the CLAUDE.md. Do the work
necessary to bring the project into alignment with it." Diff (19+/17−) is mostly
behavioral (concise responses, multi-step reasoning, code-review-report-everything,
fuzzing/PBT, UI/UX language, inform-on-rewrite, URL→local-`~/agents/docs` refs) → no
files. Three edits encode structural requirements; confirmed scope with the user (chose
root-only memories home).

**Shipped (config + docs; `live_stt.py` untouched):**
- Root `.gitignore` +`.serena/memories/`; removed `/memories` from the nested
  `.serena/.gitignore` so root is the sole, durable home (Serena regenerates the nested
  file and once dropped `memories` — the original D-013 trigger). Nested keeps `/cache`
  + `/project.local.yml`.
- `.serena/project.yml` `ignored_paths: [uv.lock, LICENSE]` — the deny-listed paths git
  does NOT ignore, synced per the new rule (Serena honors `.gitignore` but not the Claude
  `permissions.deny`).
- Docs: D-013 amendment (relocation + 3-surface sync rule), orientation `.serena/` +
  `.claude/settings.json` rows, L-015 (lead recovery with `headroom_retrieve`; replaced
  the now-globally-denied `cat -A` with `sed -n 'Nl'`).

**Verified (agent):** root `.gitignore:28` matches `.serena/memories/` (nested removal
didn't regress); nested still catches cache + project.local.yml; `.serena/` tracking
unchanged (`.gitignore` + `project.yml`). project.yml parses → `['uv.lock','LICENSE']`.
Global `~/.claude/settings.json` confirmed denying `cat`/`less`/`od`/… (CLAUDE.md claim
true). 49 tests green (no code touched).

**Did not verify (user smoke, L-004):** none — config + docs only, no
mic/`--device`/latency/Ctrl+C/multi-hour surface. The standing live-mic/soak debt (T8.2)
is unchanged.

**Already aligned (no action):** `/session-prompt` already does blank→roadmap /
arg→override; deny-list already embodies "do-not-read distinct from gitignore". Roadmap
untouched — T8.1 remains the lowest open task next session.

**Memory:** D-013 amendment; orientation 2 rows; L-015 updated; journal pruned oldest
(T5.3). No new ADR/lesson — CLAUDE.md now states the sync rule and D-013 records the
project state.

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
