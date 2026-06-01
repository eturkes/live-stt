# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

Append-only by convention. Pruning happens periodically — old entries move to `archive/journal-pre-YYYY-MM.md` rather than being deleted outright.

---

## 2026-06-01 — CLAUDE.md update: propagate edits, fix PortAudio regression

**Trigger:** User updated CLAUDE.md (uncommitted `M`) and asked for any work needed in response.

**CLAUDE.md changes (diff vs `c1300fa`):** sandbox now names the Distrobox/openSUSE host + endorses LSP/REPL tooling; "plugins" added to the self-modify list; NEW home-dir-cleanliness/maintenance rule; NEW prose-style line (dry/direct/concise; assume technical proficiency); memory-system rule restated as a per-entry **value bar** (value beyond docs/codebase/git history; no superfluous info like package versions); overengineering bullet reframed positively (modularity/tight-scope/dedup); NEW rule endorsing TDD + multi-agent councils/teams + subagents-on-most-advanced-model; NEW agent-oriented-languages pointer.

**Propagation decision:** Most new rules are meta-instructions that live in CLAUDE.md and are read every session — per the memory system's own "do not duplicate CLAUDE.md" convention and the new value bar, they are NOT copied into `.agent/`. Only two edits earned their place:
- `.agent/README.md` Conventions: added the per-entry value test (operationalizes the strengthened memory-system rule at the point of writing).
- `.agent/orientation.md` Build/test: documented that the runtime needs system `libportaudio2` (sounddevice dlopens PortAudio; `uv sync` does not provide it).

**Environment fix (triggered by the new cleanliness/maintenance mandate):**
- Recon found the import smoke-check broken: `OSError: PortAudio library not found`. System `libportaudio2` was absent from the container, so `uv run python -c "import live_stt"`, `uv run pytest`, AND the pre-commit hook all failed — commits were blocked.
- Installed `libportaudio2` via apt (pulled libopus0, libsamplerate0, libjack-jackd2-0). Import + 23 tests now pass.
- Hygiene per new rule: `sudo apt-get clean`; `uv cache prune` (removed 14 unused files). The `.venv` "dangling" symlinks `find -xtype l` reported were a false positive — uv's multi-hop python links resolve to a real 3.14.5 interpreter.

**Verified:**
- `uv run python -c "import live_stt"` → import OK.
- `uv run pytest -q` → 23 passed.

**Did not verify (user smoke-test needed):**
- No audio code changed, but PortAudio was just (re)installed in this container. The next real `uv run live-stt --device <N>` against a mic is the end-to-end confirmation that capture works here (PortAudio present ≠ a working capture device inside Distrobox).

**Flagged for user (out of project scope — not touched):**
- `~/.cache/R` (352M) and `~/.cache/pnpm` (37M) are other toolchains' caches; prune at will. Sibling `~/` project dirs (`lobotomized-claude-code`, `tweakcc-*`) left alone.

**Findings / lessons:** No new `lessons.md` entry — the PortAudio gotcha lives as a single orientation setup note (avoids duplication per the value bar). Skipped reframing negatively-titled lessons (L-001, L-005): pink-elephant matters for imperatives the agent must follow, and those lesson bodies are already positively phrased where it counts; a title-only reframe would be value-free churn the new bar discourages.

---

## 2026-05-25 — CLAUDE.md update: sync .agent/ memory system

**Trigger:** User updated CLAUDE.md with expanded/reordered rules. Agent session to propagate changes through `.agent/` files and fix drift.

**Key CLAUDE.md changes absorbed:**
- CLAUDE.md edit permission: restriction removed → agent may rewrite at any time.
- Directory constraint added: development must stay within launch directory and children.
- Memory system: added explicit "keep up-to-date to avoid drift" mandate.
- Reusable prompt: merged into long-time-horizon rule; must write to file and inform user on updates.
- Commit policy: "only when user asks" → "at end-of-turn closing cohesive work; defer if mid-iteration."
- Security: added periodic audit scheduling and software update verification.
- Two new rules: test-suite philosophy (permissible if meaningful feedback loop) and overengineering guardrails (KISS, UNIX, periodic refactors).
- Objectivity: expanded with reasoning methodology (deductive, first principles, scientific/Socratic method, benchmarking).
- ChatGPT added alongside Gemini for non-conversational tasks.

**Changes made:**
- `orientation.md`: updated CLAUDE.md file-map entry (removed approval constraint); updated commit step in how-to-work loop.
- `SESSION_PROMPT.md`: rewrote CLAUDE.md modification and commit policy lines.
- `README.md`: switched rule-reference convention from positional (#N) to descriptive (topic-based).
- `decisions.md`: updated 3 positional references (D-004 ×2, D-007 ×1) to descriptive; fixed D-004 alternatives wording.
- `lessons.md`: updated 2 positional references (L-001, L-004) to descriptive.
- `scratch/2026-05-16_code-audit.md`: updated 1 positional reference to descriptive.
- Journal entries left as-is (historical, append-only convention).

**Rationale for reference convention change:** CLAUDE.md bullets are unnumbered and the user reorders them across updates. Positional references (#8, #15, etc.) were already off by the time the `.agent/` system was one session old. Descriptive references (e.g. "CLAUDE.md memory-system rule") are stable across reorderings and self-documenting.

**Verified:**
- All `grep -rn 'CLAUDE\.md #[0-9]' .agent/` matches are now in journal.md (historical, untouched) only.
- No remaining "user approval" constraints outside journal.md.

**Did not verify (user smoke-test needed):**
- None. Only documentation/memory-system files touched. No runtime behavior changed.

---

## 2026-05-18 — Proactive audit: doc-drift, spike lint, latent risks

**Trigger:** No actionable PLAN tasks remained after T2.3 (only blocked T-BACKENDS-001). User opted for a proactive audit. Punch list produced; user approved P1 (all four), P2 (clean the spike), and two P3 items.

**Changes:**
- P1 doc-drift (4 single-line edits):
  - `.agent/lessons.md` L-002: anchor `live_stt.py:222-262` → `live_stt.py:248-300` (outer while at 258).
  - `.agent/orientation.md`: anchor `222-232` → `248-300`; "~520 lines" → "~545 lines".
  - `.agent/SESSION_PROMPT.md`: "~520 lines" → "~545 lines".
  - `spike/backends/REPORT.md:102`: stale `AGENT_PROMPT.md` reference → `.agent/orientation.md` § "Smoke-test constraints".
- P2 spike ruff (15 errors → 0):
  - 13 auto-fixed via `uv run ruff check spike/ --fix` (`I001` import-order, `UP041` collapsed `(TimeoutError, asyncio.TimeoutError)` tuples on py311+, `UP037` quoted annotation).
  - 2 hand-fixed `E501`s: `bench.py:55` (wrapped Azure dict to continuation line preserving column alignment); `prototype_openai_realtime.py:183` (wrapped `_emit_turn(...)` call).
- P3:
  - `.gitignore`: added `spike/backends/results.md` — auto-generated by `bench.py`, mirrors `results.json` treatment.
  - Untracked the previously committed copy with `git rm --cached spike/backends/results.md`.
  - `live_stt.py:440-442`: added 3-line comment documenting the sentinel-on-full-queue recovery path (sender breaks out on the next `send_realtime_input` against the closed session, so a dropped sentinel is recoverable).

**Verified:**
- `uv run ruff check .` → clean across the whole repo (was 15 errors in spike, now 0).
- `uv run pytest -q` → 23 passed, no regressions.
- `uv run python -c "import live_stt"` → clean.
- `git check-ignore -v spike/backends/results.md` confirms the new `.gitignore` entry matches.

**Did not verify (user smoke-test needed):**
- None for this session. Only docs, lint, gitignore, and a descriptive comment touched. No runtime behavior changed.
- Previously-flagged T2.3 smoke-tests (level-meter clear on live log records; `2> errors.log` cleanliness with mic active) remain outstanding.

**Findings / lessons:**
- No new generalizable lesson worth promoting. L-001 ("don't over-edit already-optimized code") and L-005 ("avoid abstractions") together already explain why the audit found so little to change in `live_stt.py`.
- Confirmed: the `.agent/` memory system holds up under a second-look audit. The only durable drift was line-anchor + line-count — both predictable consequences of T2.3 adding ~22 lines. Cheap to mop up; not worth a process change.

**Open follow-ups:**
- T-BACKENDS-001 still blocked on `DEEPGRAM_API_KEY` / `OPENAI_API_KEY`. Spike code is now lint-clean, so the unblock-then-run path is one `.env` edit away.
- T2.3 smoke-tests still unverified.
- User has not requested a commit; the working tree carries 14 modified files + 1 staged deletion awaiting decision.

---

## 2026-05-18 — T2.3 shipped: structured logging for errors

**Trigger:** Bootstrap selected T2.3 as the lowest-numbered open task with no blockers; user said "Proceed."

**Changes:**
- `live_stt.py`:
  - Added `import logging` and a module-level `logger = logging.getLogger("live_stt")`.
  - New `_StderrFormatter` (subclass of `logging.Formatter`): format string `[%(asctime)s] %(levelname)s %(message)s`; prepends `_LINE_CLEAR` only when `sys.stderr.isatty()` so the live level meter is erased in place on terminals while redirected stderr stays free of ANSI escapes.
  - New `_configure_logging()`: idempotent (guards on `logger.handlers`), sets level INFO, attaches stderr `StreamHandler`, disables propagation. Called once at the top of `main()`.
  - Replaced 5 runtime stderr writes:
    - `[send error: %s]` → `logger.error` (was line 215)
    - `[go_away, reconnecting (time_left=%s)]` → `logger.info` (was line 236) — graceful, server-initiated reconnect; not an error
    - `[recv error: %s]` → `logger.error` (was line 260)
    - `audio: %s` → `logger.warning` (was line 354) — fires from the PortAudio callback thread; `logging` is thread-safe via internal RLock
    - `[session error: %s: %s]` → `logger.error` (was line 418)
  - Left the `GEMINI_API_KEY` preflight `print(..., file=sys.stderr)` untouched — PLAN T2.3 scope explicitly lists the 5 runtime sites only.
- `PLAN.md`: T2.3 → Shipped row in the table; removed the Open subsection.
- `README.md`: added a "Diagnostics" subsection under "How it works" describing the TTY vs redirected stderr behavior.
- `.agent/scratch/2026-05-18_T2.3.md`: planning note (call-site → level map, format rationale, meter-coexistence trace, thread-safety note, out-of-scope list).

**Verified:**
- `uv run python -c "import live_stt"` → clean import.
- `uv run pytest` → 23 passed in 0.58s (same suite, no regressions).
- Hand-test of `_StderrFormatter`: non-TTY branch returns plain `[ts] ERROR hello`; TTY branch returns `\r\x1b[2K[ts] ERROR hello`. Confirmed both produce no extra ANSI for the non-TTY path (acceptance for `2> errors.log`).

**Did not verify (user smoke-test needed):**
- Level meter clears cleanly when a real log record fires mid-session (the formatter writes `\r\x1b[2K` to stderr; stdout/stderr share the terminal cursor so this should erase the meter, but I cannot exercise it without a live mic). Per L-004, suggested user run: `uv run live-stt --device <N>` against a real mic, speak briefly to start the meter, then trigger a network blip (toggle Wi-Fi or unplug Ethernet) to fire a `[send error: …]` or `[go_away, …]` line and confirm the meter resumes on a fresh line below the message.
- `live-stt > /dev/null 2> errors.log` cleanliness with a live mic-driven session.
- `audio: <flags>` logging from the PortAudio thread under real underrun/overrun conditions.

**Findings / lessons:**
- One generalizable rule worth promoting: TTY-aware formatters are the right home for ANSI prefixes when a single stream has to coexist with a continuously-redrawn UI element on another stream. Each call site stays format-agnostic; the formatter owns the cursor protocol. Adding to `lessons.md` as L-006.

**Open follow-ups:**
- T-BACKENDS-001 still blocked on `DEEPGRAM_API_KEY` / `OPENAI_API_KEY`.
- PLAN.md "Open" section is now empty save for the blocked spike. Next agent session may want to either unblock backends or proactively scope new work (none currently identified).

---

## 2026-05-18 — T-CLEANUP-001 residue, T2.2 deferred, pytest pre-commit hook (T-HOOK-001)

**Trigger:** Bootstrap loaded; user selected `T-CLEANUP-001 first`, declared the tool Japanese-only (defer T2.2), and approved wiring `uv run pytest` into a pre-commit hook.

**Changes:**
- Removed empty `spike/t3_2/` directory (last residue from `ae2f706`). `spike/` now only contains `backends/`.
- `PLAN.md`:
  - T-CLEANUP-001 → Shipped (one-line note pointing at gitignore commit `0b5a6b0` + the rmdir today).
  - T-HOOK-001 → Shipped (new ID for the pre-commit hook work).
  - T2.2 → moved to a new "Deferred" section with rationale + "revisit if" trigger.
  - Pending-decisions table emptied (all 3 questions resolved).
- `.githooks/pre-commit`: 5-line `sh` script, `set -e; exec uv run pytest -q`. Executable bit set. Smoke-checked directly — 23 tests pass in 0.91s.
- Repo local config: `core.hooksPath` → `.githooks` (per-clone, not committed; each new clone reruns the one-liner).
- `README.md`: renamed "Tests" subsection to "Development", documented the one-time `core.hooksPath` step, added `.githooks/` to the project-structure tree, rewrote the T2.2 mention to reflect the Japanese-only design intent.
- `.agent/orientation.md`: added `.githooks/pre-commit` to file map; added the one-time setup line to build/test commands with a one-liner explaining it's not auto-applied by `uv sync`.
- `.agent/decisions.md`: new entry **D-007** documenting why we chose a project-local shell hook over the `pre-commit` framework.

**Findings / lessons:**
- No new generalizable lesson worth promoting. The D-007 rationale (avoid frameworks when one shell line suffices) is already captured by `L-005` ("avoid abstractions") and CLAUDE.md #3 ("opt for installation/configuration local to the scope of the project") — promoting it would duplicate.
- Re-read of L-001 before touching `README.md` paid off: kept the `live_stt.py` Notes line at the bottom of README short and factual rather than rewriting the code's existing rationale comments.

**Did not verify (user smoke-test needed):**
- None for the runtime path — this session touched only docs, scaffolding, and a git hook. No audio/network code changed.
- The hook itself executes correctly (proven by direct invocation), but the first real `git commit` after this session is the proper end-to-end test. If pytest somehow fails in the commit context (e.g., env vars stripped by some shell), the user will see it as an aborted commit and we'll fix it.

**Open follow-ups:**
- T2.3 is now the lowest-numbered open task. Acceptance is straightforward; can be done in a single session without user input.
- T-BACKENDS-001 still blocked on API keys (`DEEPGRAM_API_KEY`, `OPENAI_API_KEY`).
- Consider whether the hook should also run `ruff check .` once we have evidence it stays green. Not adding today because the user asked specifically for `pytest`.

---

## 2026-05-16 — Adopt new `CLAUDE.md`, scaffold `.agent/` memory system

**Trigger:** User added a new `CLAUDE.md` at repo root and asked the project to be updated to use it.

**Changes:**
- Created `.agent/` directory with structured files (`README.md`, `orientation.md`, `journal.md`, `lessons.md`, `decisions.md`, `scratch/`, `archive/`).
- Merged `AGENT_PROMPT.md` project-specific orientation into `.agent/orientation.md` and `.agent/SESSION_PROMPT.md`.
- Deleted `AGENT_PROMPT.md`.
- Rewrote `PLAN.md` for LLM-density (table-first task records, explicit acceptance criteria).
- Rewrote `README.md` keeping it human-usable on GitHub but trimming prose and structuring sections more tightly.
- Audited `live_stt.py` for LLM-readability changes — recommendation: no changes (code already optimized with rationale comments; further densification would hurt clarity). Decision logged in `decisions.md`.
- Fixed `list_live_models.py` line-19 length lint issue flagged in `SPIKE_REPORT.md`.
- Spike reports (`SPIKE_REPORT.md`, `SPIKE_REPORT_BACKENDS.md`) kept at root as historical records; indexed from `orientation.md`.

**Findings / lessons:**
- See `lessons.md` for: "Don't over-edit already-optimized code in the name of LLM-readability."
- See `decisions.md` for: "Memory system structure (`.agent/` shape)" and "Spike reports stay at root, not archived."

**Did not verify (user smoke-test needed):**
- None — this session touched only docs and scaffolding. No code logic changed.

**Open follow-ups:**
- Output `SESSION_PROMPT.md` content to chat at end of session (CLAUDE.md #12).
- `PLAN.md` open items remain: T2.2 (`--language`), T2.3 (structured logging). Backends spike awaits API keys.
- `spike/backends/cache/` and `spike/backends/results.json` are untracked — decide if they should be gitignored (likely yes; bench artifacts).
- Empty `spike/t3_2/` directory still on disk despite commit `ae2f706` claiming removal — investigate next session.
