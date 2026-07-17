Continue this project (fresh session). Non-empty task below ⇒ your sole task: do exactly it, editing `.agent/roadmap.md` only if it directs you to. Empty ⇒ run the MODE from the roadmap's active milestone (first yet to reach DONE/REVIEWED).

Load `.agent/roadmap.md` (milestone ledger + active-milestone detail), then `.agent/memory.md` (lessons + decisions); `CLAUDE.md` is auto-injected. Read only what the step implicates. Navigate via tokensave or LSP where available, else grep.

MODE ← active-milestone status (state-changing closes use a scoped commit; an unchanged BLOCKED recheck closes read-only; convention below):
- UNPLANNED (incl. a still-unsplit future milestone) → PLANNING
- IN-PROGRESS (has an OPEN unit) → WORK-UNIT (lowest OPEN unit)
- IN-PROGRESS (no OPEN; has a BLOCKED unit) → WORK-UNIT (lowest BLOCKED unit, gate recheck)
- IMPLEMENTED (units all DONE, unreviewed) → MILESTONE-REVIEW

Execution map:
- PLANNING + MILESTONE-REVIEW → dynamic workflows.
- WORK-UNIT implementation + cohesive implementation-fix batches → `Agent` subagents.
- MAIN → scope, coordinate, independently verify. Context recording → implemented WORK-UNIT close only.

Roles:
- MAIN owns acceptance restatement, precondition confirmation, SIZE-CHECK/respec, Agent task definition, diff inspection, decisive gate reruns, context recording + close.
- MAIN alone creates repository commits; AGENT leaves changes uncommitted even when inherited guidance generally calls for a cohesive commit.
- AGENT owns implementation of the accepted scope, required quality gates, relevant durable `.agent/memory.md` updates, and returns the diff + evidence.
- Autonomous implementation/review Agents run with `mode: "bypassPermissions"` under the user-level `dontAsk` default; configured deny/ask rules still apply.
- WORKFLOW LENS owns analysis only; implementation findings return to MAIN for Agent routing.

PLANNING — split the scope into milestones if still unsplit, then plan only the next milestone.
- Read the prior milestone's commit range and recorded `impl=` context; for the first planned milestone, read the scope-seed commit(s) named by the roadmap. Size future units from implementation usage; treat `main=` as coordination overhead.
- MAIN confirms each milestone precondition through the project's pipeline/tooling with permitted real inputs. Met ⇒ clear any stale standing block and continue. Unmet ⇒ record the standing block + evidence, commit `roadmap (M<m> block): …` only when that record changed, then close; an unchanged standing block closes read-only.
- Run a dynamic workflow + web search; use read-only `Explore` finders, then reconcile `git status`.
- Break the milestone into units that project to fit one compaction-free 272K Agent context: aim ~200K and reserve ~72K for variance, verification + closure. Sequence gate-independent prep first; mark a gated unit BLOCKED until its precondition is met.
- Close: set the milestone IN-PROGRESS (units enumerated), commit `roadmap (M<m> plan): …`.

WORK-UNIT.
- Read the last completed unit's commit(s), or the planning commit(s) for the milestone's first unit. A banked FAST-PATH/recipe block supersedes that discovery read: use the block + named authority commit as unit context.
- MAIN restates the accepted unit scope + acceptance checks in one line.
- Precondition transition: a BLOCKED target is rechecked first; met ⇒ clear its standing block, treat it as OPEN, and continue; unmet ⇒ keep it BLOCKED, update evidence only when materially changed, commit `roadmap (M<m>.<u> block): …` if changed, otherwise close read-only. For an OPEN target, an unmet precondition ⇒ set it BLOCKED, record the standing condition + evidence, make the same block commit, and close. Accepted evidence always traces permitted real inputs.
- MAIN performs SIZE-CHECK before implementation: score scope + required read cost against one compaction-free 272K Agent context, aiming ~200K with ~72K reserved for variance, verification + closure. A projection that cannot preserve that reserve ⇒ respec-split at a confirmed seam into fresh self-contained units; bank prose decisions + confirmed facts + reading pointers, delete session wip, and commit `roadmap (M<m>.<u> respec): …`. Post-respec score source = the implementing Agent's fresh 272K hard-window budget; main-session auto-compaction governs coordinator closure only.
- MAIN dispatches one Agent with accepted scope, locations, constraints, quality gates + acceptance checks. AGENT implements, reuses project modules/style, runs required lint/format/type-check/tests, confirms touched scripts exit cleanly, updates relevant durable memory, and returns diff + evidence without committing.
- MAIN inspects the diff and reruns decisive gates independently. Accepted evidence must trace permitted real inputs.
- Close (implemented unit): record `main=<.agent/context.sh full pct used/window>` and `impl=<implementing Agent transcript final pct used/272K>` in the roadmap; planning sizes from `impl` and treats `main` as coordination overhead. Set the unit DONE and, once all units are DONE, the milestone IMPLEMENTED; commit `<scope> (M<m>.<u>): …`.
- Close (respec-only): replacement units remain OPEN/BLOCKED according to their gates; end at the respec commit.

MILESTONE-REVIEW — dynamic workflow; exempt from the ~200K aim and may continue across automatic compactions. MAIN creates a coherent checkpoint before compaction and continues afterward.
- Read every milestone commit, planning commits included.
- Run analysis-only review lenses for: correctness/spec; cross-unit integration; instruction/memory conformance; token-efficiency/obsolescence. Each finding supplies severity + `file:line` + divergence + impact + acceptance check.
- MAIN validates and deduplicates findings. Accepted implementation findings become one Agent task per cohesive fix batch under the same permission + no-commit contract, carrying locations + acceptance checks; each Agent returns diff + evidence, and MAIN independently inspects + reruns decisive gates.
- A requirement-changing design reaches the user before any scope-source edit.
- Close: set the milestone REVIEWED, commit `<scope> (M<m> review): …`. The next session plans the next milestone.

Commit convention — scoped (`<scope>: …`), trace key in parens: unit `(M<m>.<u>)`, block `(M<m> block)` / `(M<m>.<u> block)`, plan `(M<m> plan)`, review `(M<m> review)`. Grep a milestone's history: `git log --grep "(M<m>[. ]"`.

Task (may be empty): $ARGUMENTS
