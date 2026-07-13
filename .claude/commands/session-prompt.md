Continue this project (fresh session). Non-empty task below ⇒ your sole task: do exactly it, editing `.agent/roadmap.md` only if it directs you to. Empty ⇒ run the MODE from the roadmap's active milestone (first yet to reach DONE/REVIEWED).

Load `.agent/roadmap.md` (milestone ledger + active-milestone detail), then `.agent/memory.md` (lessons + decisions); CLAUDE.md (imports `AGENTS.md`) is auto-injected. Read only what the step implicates. Navigate via tokensave or LSP where available, else grep.

MODE ← active-milestone status (each mode advances it, then closes on a scoped commit; convention below):
- UNPLANNED (incl. a still-unsplit future milestone) → PLANNING
- IN-PROGRESS (has an OPEN unit) → WORK-UNIT (lowest OPEN unit)
- IMPLEMENTED (units all DONE, unreviewed) → MILESTONE-REVIEW

After each mode's commit I compact and run `/codex-review`; you fix accepted findings in a follow-up commit. MILESTONE-REVIEW is the exception — its `/codex-review` runs on the uncompacted session. Record context-usage in WORK-UNIT only.

PLANNING — split the scope into milestones if still unsplit, then plan only the next milestone.
- Read the prior milestone's commit range, especially its recorded context-usage (it right-sizes units); for the first planned milestone, the scope-seed commit(s) the roadmap names.
- Gate first: a milestone gated on an unmet precondition stops here — record the standing block. Confirm the precondition functionally (resolve it through the project's pipeline/tooling); deny-listed inputs stay off-limits.
- Plan (once unblocked): always a dynamic workflow (standing opt-in) + web search; finders read-only (`Explore`), then `git status`-reconcile. Break the milestone into ~200K-token units (soft — finish even over-budget); sequence gate-independent prep first; flag any still-gated unit BLOCKED (planned, awaiting its gate).
- Close: set the milestone IN-PROGRESS (units enumerated), commit `roadmap (M<m> plan): …`.

WORK-UNIT.
- Read the last completed unit's commit(s) — or the planning commit(s) if this is the milestone's first unit.
- Do: (1) restate the unit + its acceptance in one line; (2) SIZE-CHECK before writing code — score the unit against memory's sizing rules + the read-cost axis (modules its gates must read for exact shapes); a projection well past the ~200K aim ⇒ respec-split at a confirmed seam FIRST into fresh self-contained units (memory's retired-salvage rule: bank prose decisions + confirmed facts + reading pointers only; delete any session wip file before the closing commit), commit `roadmap (M<m>.<u> respec): …`; then implement the first half same-session — the 1M window absorbs the seam-confirmation reads + an occasional overshoot; close + start it fresh next session when it alone still projects well over the aim; (3) implement, reusing modules, matching surrounding style; (4) GATE — a gated unit needs its precondition met; confirm functionally (resolve through the pipeline/tooling), deny-listed inputs off-limits; unmet ⇒ stop and report, so every result traces to real inputs; (5) VERIFY the project's quality gates pass (`.agent/memory.md` § Commands — pytest, pyright, ruff, import-smoke); touched scripts exit clean; (6) record durable lessons/decisions in `.agent/memory.md`.
- Close (implemented unit): record the unit's context-usage (`.agent/context.sh`, full `pct used/window`) into the roadmap; set the unit DONE — and the milestone IMPLEMENTED once every unit is DONE; commit `<scope> (M<m>.<u>): …`. A respec-only session instead ends at its respec commit — replacement units stay OPEN, none set DONE.

MILESTONE-REVIEW — token-unbounded: hold it all in-context, undivided.
- Read every commit of the milestone, planning commits included.
- Adversarially review the milestone's whole body — AGENTS.md's review criteria + cross-unit consistency, conformance to scope/AGENTS.md/memory, token-efficiency, obsolescence — and fix what you find; revise the scope source on a better design (requirements changes reach me first).
- Close: set the milestone REVIEWED, commit `<scope> (M<m> review): …`. The next session plans the next milestone.

Commit convention — scoped (`<scope>: …`), trace key in parens: unit `(M<m>.<u>)`, plan `(M<m> plan)`, review `(M<m> review)`. Codex-review follow-ups keep the key and add a `Codex-Review: <accepted findings>` trailer. Grep a milestone's history: `git log --grep "(M<m>[. ]"`.

Task (may be empty): $ARGUMENTS
