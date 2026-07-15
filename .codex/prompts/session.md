# live-stt session workflow

Continue this project in a fresh Codex session.

`TASK` = text supplied with `$session-prompt`. A non-empty task is the sole task: do exactly it,
editing `.agent/roadmap.md` only when the task directs you to. An empty task selects the MODE from
the roadmap's active milestone (the first milestone yet to reach REVIEWED).

Load `.agent/roadmap.md` (ledger + active-milestone detail), then `.agent/memory.md` (live lessons +
decisions); `AGENTS.md` is already active. Read only what the selected work implicates. Start with
tracked files + `git status`; navigate with `rg` or available language tooling.

MODE ← active-milestone status:
- UNPLANNED, including a still-unsplit future milestone → PLANNING
- IN-PROGRESS with an OPEN unit → WORK-UNIT (lowest OPEN unit)
- IMPLEMENTED with every unit DONE → MILESTONE-REVIEW

For a non-empty `TASK`, follow its requested workflow instead of selecting a mode. Every path ends
with an adversarial review per `AGENTS.md`, accepted fixes, proportionate verification,
roadmap/memory synchronization when implicated, and at most one scoped commit.

## PLANNING

- Read the prior milestone's commit range; for the first planned milestone, read the scope-seed
  commits named by the roadmap.
- Resolve gates through the project's real pipeline/tooling. An unmet precondition stops the mode;
  record the standing block and preserve real-input traceability.
- Plan only the next milestone. Split it into self-contained units sized for focused sessions;
  sequence gate-independent preparation first and mark gated units explicitly.
- Use current primary sources when a decision is time-sensitive or selects tooling. Reserve Codex
  subagents for independent, bounded exploration that materially improves the plan, then
  reconcile their results against `git status` and the live tree.
- Close: set the milestone IN-PROGRESS with units enumerated; commit
  `roadmap (M<m> plan): <summary>`.

## WORK-UNIT

- Read the last completed unit's commits, or the planning commits for the milestone's first unit.
- Restate the unit + acceptance in one line before editing.
- Size-check at a confirmed seam. If the unit cannot fit one focused session, respec it into fresh
  self-contained units; preserve only durable decisions, confirmed facts, and reading pointers.
  Close with `roadmap (M<m>.<u> respec): <summary>`; start replacement work in a fresh session.
- Implement against the live code, reusing modules and matching surrounding style. Confirm every
  gated input functionally through the project's pipeline/tooling.
- Run the quality gates listed in `.agent/memory.md` § Commands; touched scripts must exit clean.
- Record only durable new lessons/decisions in `.agent/memory.md`.
- Close: set the unit DONE and, when every unit is DONE, the milestone IMPLEMENTED; commit
  `<scope> (M<m>.<u>): <summary>`.

## MILESTONE-REVIEW

- Read every milestone commit, including planning commits.
- Adversarially review the milestone as one body: correctness, cross-unit consistency, conformance
  to scope/instructions/decisions, verification gaps, obsolete text, and guarantee-vs-claim gaps.
- Fix accepted findings and record rejected findings only when doing so prevents re-litigation.
- Close: set the milestone REVIEWED; commit `<scope> (M<m> review): <summary>`.

Trace lookup: `git log --grep "(M<m>[. ]"`.
