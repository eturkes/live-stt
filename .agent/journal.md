# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

---

## 2026-06-16 — T8.1: non-blocking worker-stop sentinel (shutdown deadlock fix)

**Trigger:** `/session-prompt` (no override) → roadmap. T8.1 was the lowest open
task; user confirmed "Proceed with T8.1" over redirecting to another T8 item.

**Shipped (live_stt.py + tests):** `run_session`'s shutdown `finally` replaced its
blocking `await audio_q.put(None)` with the evict-then-put idiom (`put_nowait`; on
`QueueFull`, `get_nowait` one stale block, retry) — a verbatim transplant of
`submit_sentinel`. Fixes the named deadlock: `worker()` is audio_q's only consumer
and returns on an in-worker exception (after `request_stop`); if the mic callback
filled audio_q to `AUDIO_QUEUE_MAX`, the blocking put parked the loop forever, and
since Ctrl+C routes to `request_stop` (not `KeyboardInterrupt`) the only escape was
SIGKILL with `-o` left unclosed. New `test_shutdown_sentinel_lands_on_full_audio_queue_without_blocking`
(tests/test_audio.py) drives the idiom on a synthetic full `asyncio.Queue(maxsize=4)`
under a 1 s `wait_for` (must not fire) + asserts the sentinel lands while the oldest
block is evicted (1,2,3 survive). No new symbol.

**Verified (agent):** 50 tests green (+1 over 49), ruff clean, `uvx pyright@1.1.410`
0 errors/warnings/informations. Normal Ctrl+C path is effect-identical (a put with
spare capacity lands immediately then breaks); only the dead-worker + full-queue case
changes (drops one block vs hangs to SIGKILL). Replay path structurally untouched —
replay.py drives `worker` directly, never `run_session`. (In-session LSP flagged
stale `capsys`/`time_info` "unused" notes from line shifts; the pyright CLI gate is
authoritative and clean.)

**Did not verify (user smoke, L-004):** live Ctrl+C-in-terminal mid-utterance flush
+ `-o` persistence — structurally unchanged but on the shutdown/signal path, so it
stays user-smoke (now itself the open T8.2 checklist task).

**Memory:** PLAN T8.1 → SHIPPED with a shipped-summary; journal pruned oldest (T8
roadmap entry → git). No new lesson/ADR — a clean transplant of an existing,
already-reasoned idiom is not a new failure mode or architecture choice (L-008 spirit:
the idiom + its rationale already live in code comments + the submit_sentinel
precedent; nothing generalizable beyond L-005/L-020). **T8.2** is the lowest open task
next session.

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
