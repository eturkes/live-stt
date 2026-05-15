# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

Append-only by convention. Pruning happens periodically — old entries move to `archive/journal-pre-YYYY-MM.md` rather than being deleted outright.

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
