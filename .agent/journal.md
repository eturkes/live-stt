# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

Append-only by convention. Pruning happens periodically — old entries move to `archive/journal-pre-YYYY-MM.md` rather than being deleted outright.

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
