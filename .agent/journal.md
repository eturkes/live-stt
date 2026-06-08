# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

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

---

## 2026-06-08 — Env-block question closed (already global); pyright-lsp enabled project-scope

**Trigger:** User picked "resolve env-settings question" from session-start options, then clarified: "I thought those settings were set globally?" — correct. Global `~/.claude/settings.json` (container HOME; symlink → `~/agents/claude/settings.json`) already carries `CLAUDE_CODE_SUBAGENT_MODEL=opus` + `CLAUDE_CODE_EFFORT_LEVEL=max` (plus MAX_OUTPUT_TOKENS=128000, agent-teams flag, etc.). Two sessions of "import ckc's env block?" flagging were moot — nobody had checked the global file. CLOSED → D-008 amendment (b). ckc's project copy = redundant no-op.

**pyright-lsp (user opted Enable):** `enabledPlugins.pyright-lsp@claude-plugins-official` now in project `.claude/settings.json`. First attempt used the CLI's default **user** scope and silently edited the GLOBAL settings → reverted (`uninstall -s user`, then `install -s project`; leftover empty key hand-cleaned at the symlink target) → **L-012**. Server dep already satisfied user-level: `~/.local/bin/pyright-langserver` (pyright 1.1.410, pnpm tree from ckc setup). Global file diff vs pre-session: key order only (CLI rewrites reorder).

**Verify next session:** pyright LSP tools attach + give diagnostics on `live_stt.py` (plugins load at session start — unverifiable from the session that flips the flag).

**Flagged for user:** (1) ckc's `installed_plugins.json` record still points at pre-move `~/Documents/pro/ckc` — ckc may need `install -s project` re-run from its new path. (2) ckc's project env block duplicates the global one; prune at will. (3) T4.3/T4.4 live-mic smoke test still pending (L-004).

---

## 2026-06-08 — T4.2–T4.5 shipped: re-architecture complete, Gemini fully removed

**Trigger:** User: "OK I authenticated Codex" — unblocked T4.2; ran the remaining T4 series to completion in one session.

**T4.2 (→ D-011):** `codex app-server` JSON-RPC/stdio surface benched via `spike/backends/codex_client.py`. Binding config: Spark+`low` (`minimal` rejected), tool-features off at spawn (`web_search="disabled"` + 5 `features.*=false`) — THE latency lever, p50 3.15→**0.99 s**/turn (Gemini baseline 1.21 s); instructions via `developerInstructions` (4/4 injection-resistant; AGENTS.md-in-cwd mode REJECTED — obeyed "delete all files" as a request). Marginal ~180 uncached in + 7–60 out tok/turn; ~50 turns moved the 5 h window 0→0%. Plan reports `prolite` (user said Pro); Spark entitled regardless. Fallback: mini+`none`, p50 1.18 s.

**T4.3 + T4.4:** `live_stt.py` rewritten — mic → resample → silero VAD + 60 s `RingBuffer` (absolute indexing; pre-pad 0.4 s re-slice) → executor decode → numbered `JA n:`/`EN n:` lines; `CodexTranslator` (warm-up turn absorbs ~3 s uncached cost; thread rotation @100 turns; degradation: missing CLI → JA-only at start, 3 consecutive failures → JA-only for session, backlog >50 drops oldest). RingBuffer phase bug caught by tests (oversized append ignored ring phase; fixed with phase-aligned two-segment write). 22 tests green; synthetic E2E: STT reproduces T4.1 bench exactly, 9/9 EN ordered @~1 s cadence.

**T4.5:** `google-genai` + `python-dotenv` removed (25 pkgs; spike `load_dotenv` lines dropped), `list_live_models.py` + `.env` deleted, README/orientation/SESSION_PROMPT rewritten for the new architecture, D-001/D-003 superseded, L-002/L-003 historical, L-004 rescoped.

**Did not verify (user smoke-test, L-004):** live mic capture, `--device`/`--list-devices`, Ctrl+C mid-utterance flush + translator drain, real-time latency feel, multi-hour quota burn. Note: secondary weekly Codex window already at 54% from user's other usage.

**Carried open question:** import `ckc`'s `env` settings (`CLAUDE_CODE_SUBAGENT_MODEL=opus`, `CLAUDE_CODE_EFFORT_LEVEL=max`)? Still unanswered.

