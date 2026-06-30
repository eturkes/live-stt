Continue this project (fresh session). Non-empty task below ⇒ your sole task: do exactly it, editing `.agent/roadmap.md` only if it directs you to. Empty ⇒ run the MODE for the active phase (`.agent/roadmap.md` § Status names it).

Load `.agent/roadmap.md` (status + task ledger; legend OPEN · SHIPPED · DEFERRED · SUPERSEDED · OUT-OF-SCOPE · REJECTED), then `.agent/memory.md` (orientation, decisions, lessons). CLAUDE.md (it imports `AGENTS.md`) is auto-injected. Read only what the step implicates. Navigate via tokensave or LSP where available, else grep.

MODE ← active-phase state (each mode advances it, then closes on a scoped commit; convention at the end):
- scope not yet split into tasks → PLANNING
- a task is OPEN and agent-actionable → WORK-TASK (lowest-ID OPEN)
- nothing agent-actionable (only user-only or gated debt) → surface it to me and stop; never invent work

I launch PHASE-REVIEW myself at 1M context (see below). After each mode's commit I compact and run `/codex-review`; you fix accepted findings in a follow-up commit. PHASE-REVIEW is the exception — its `/codex-review` runs without compacting. Record context-usage in WORK-TASK only.

PLANNING — split the scope into phases if not yet split, then plan only the next phase.
- Read the prior phase's commit range, especially its recorded context-usage (it right-sizes tasks); for the first planned phase, the scope-seed commit(s) the roadmap names.
- Gate first: a phase gated on an unmet precondition stops here — record the standing block. Confirm the precondition functionally (resolve it through the project's pipeline/tooling); deny-listed inputs stay off-limits.
- Plan (once unblocked): always a dynamic workflow (standing opt-in) + web search; finders read-only (`Explore`), then `git status`-reconcile. Break the phase into tasks each completable within a 200K window; sequence gate-independent prep first; flag any still-gated task BLOCKED (planned, not yet runnable).
- Close: enumerate the new tasks OPEN in § Open, commit `roadmap (T<p> plan): …`.

WORK-TASK.
- Read the last SHIPPED task's commit(s) — or the planning commit(s) if this is the phase's first task.
- Do: (1) restate the task + its acceptance in one line; (2) implement, reusing modules, matching surrounding style; (3) GATE — a gated task needs its precondition met; confirm functionally (resolve through the pipeline/tooling), deny-listed inputs off-limits; unmet ⇒ stop and report, so every result traces to real inputs; (4) VERIFY the project's quality gates pass (`.agent/memory.md` § Commands — pytest, pyright, import-smoke); touched scripts exit clean; (5) record durable decisions/lessons in `.agent/memory.md`.
- Close: record the task's context-usage (`.agent/context.sh`, full `pct used/window`) into its § Shipped line; set the task SHIPPED; commit `<scope> (T<p>.<t>): …`.

PHASE-REVIEW — I launch this with 1M context (ideally the only 1M session): hold it all in-context, undivided.
- Read every commit of the phase, planning commits included.
- Adversarially review the phase's whole body — AGENTS.md's review criteria + cross-task consistency, conformance to scope/AGENTS.md/memory, token-efficiency, obsolescence — and fix what you find; revise the scope source on a better design (requirements changes reach me first).
- Close: note the phase reviewed in § Status, commit `<scope> (T<p> review): …`. The next session plans the next phase.

Commit convention — scoped (`<scope>: …`), trace key in parens: task `(T<p>.<t>)`, plan `(T<p> plan)`, review `(T<p> review)`. Codex-review follow-ups keep the key and add a `Codex-Review: <accepted findings>` trailer. Grep a phase's history: `git log --grep "(T<p>[. ]"`.

Task (may be empty): $ARGUMENTS
