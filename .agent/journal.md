# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

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

---

## 2026-06-08 — Maintenance pass: pyright venv config + Optional fixes, deps bumped, codex already latest

**Trigger:** No open PLAN tasks (T4 series complete; live-mic smoke test pending on user) — user picked "maintenance pass" from session-start options.

**Pyright (closes last session's verify item):** plugin attached ✓ — `documentSymbol` returns full live_stt.py tree. Its first diagnostics exposed missing venv resolution → added `[tool.pyright] venvPath="." venv=".venv"` to pyproject.toml. Two real Optional-access fixes in `live_stt.py` (using the file's existing `assert` idiom): `close()` asserts `_proc.stdin` before closing; shutdown guard widened to `translator is not None and translator_task is not None` (logically equivalent — task exists iff translator). `time_info` unused-param hint left as-is (sounddevice callback signature). CLI verify: **0 errors / 0 warnings**. **Caveat learned:** the LSP server reads `[tool.pyright]` at session start — in-session diagnostics stay stale after config edits; the pyright CLI (now in orientation build/test cmds) is the immediate check. Full-project run then surfaced 28 errors, all `spike/` (prototypes import SDKs removed at T4.5 by design) → excluded. **Gotcha:** pyright `exclude` REPLACES its defaults — `.venv`/`models`/`__pycache__` must be re-listed or it walks ~1 GB and appears to hang (first attempt did; pyproject comment documents it). Cleanup of that hang spawned **L-013** (pgrep/pkill -f self-match via the harness Bash wrapper).

**Deps:** numpy 2.4.4→2.4.6, packaging 26.1→26.2, ruff 0.15.10→0.15.16; sherpa-onnx unchanged. Verified: import smoke, ruff clean, 22 tests green, `bench.py --only local-k2` reproduces T4.1 (TTFT ≤0.01 s, totals 0.04–0.15 s, known 文→分 quirk intact). codex CLI 0.137.0 **is** the latest GitHub release (2026-06-04) — no update. `uv cache prune` reclaimed 28 MiB.

**Hygiene:** line-count fact ~750→~800 in orientation + SESSION_PROMPT (795 actual); pyright CLI line added to build/test commands; scratch pruned — `2026-05-16_code-audit.md`, `2026-05-18_T2.3.md` deleted (pre-rearchitecture, unreferenced; git archives).

**Open for user (unchanged):** T4.3/T4.4 live-mic smoke test (L-004 list); ckc stale plugin record + redundant env block.
