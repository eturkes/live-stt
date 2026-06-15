# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

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

---

## 2026-06-08 — Renamed `/session` slash command → `/session-prompt`

**Trigger:** User (`/session` override) — rename the bootstrap slash command `/session` → `/session-prompt`.

**Shipped:** `git mv .claude/commands/session.md → session-prompt.md` (history preserved; the command name derives from its filename). Body self-mention repointed (`text typed after /session-prompt`); frontmatter untouched, re-verified parsing (L-014). Live refs repointed: `.agent/README.md` intro pointer; `decisions.md` D-004 amendment + D-012 (title/body/verify → new name, plus a rename amendment recording the original `/session` for trajectory). Per user choice: historical `journal.md` entries and L-014's dated `session.md` context left intact (accurate-at-date). Grep-verified no stale *live* ref remains — remaining old-name hits are two false positives (`send/recv/session`, `thread/session`), the journal/L-014 history, and the amendment's own rename note. (Grep gotcha: `/session\b` matches `/session-prompt` because `-` is a word boundary; used `/session(?!-prompt)`.)

**Smoke-test (user-side, agent cannot verify):** in a fresh Claude Code session `/session-prompt` (roadmap) and `/session-prompt <TASK>` (override) must register in the menu and expand correctly; old `/session` should no longer appear. Slash-command discovery/expansion is outside agent reach.

---

## 2026-06-08 — Session bootstrap prompt → `/session [TASK]` slash command

**Trigger:** User — turn the reusable `SESSION_PROMPT.md` into a native slash command; a `<TASK>` arg overrides the roadmap for that session, blank follows `PLAN.md`.

**Shipped:** `.claude/commands/session.md` (`/session [TASK]`). Body = old prompt minus the human-paste preamble; the trailing "USER STEERING" HTML comment is replaced by an `$ARGUMENTS` slot in § "What to do right now" (non-empty → override + still run bootstrap reads; empty → lowest-numbered open `PLAN.md` task). `.agent/SESSION_PROMPT.md` deleted. Refs repointed: `.agent/README.md` (intro pointer added, stale table row dropped), `decisions.md` D-004 amended + **D-012** added. Historical journal mentions of `SESSION_PROMPT` left intact (accurate at their dates).

**Gotcha (→ L-014):** `argument-hint: [TASK] …` is invalid YAML — a leading `[` opens a flow sequence and the trailing prose raises `ParserError`; quoted the value. Verified the frontmatter parses via `uv run --with pyyaml`.

**Smoke-test (user-side, agent cannot verify):** in a fresh Claude Code session, `/session` (roadmap) and `/session <TASK>` (override) must register in the menu and expand correctly — slash-command discovery/expansion is outside agent reach.
