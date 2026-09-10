# Upstream sync

`CLAUDE.md` is refreshed **byte-for-byte** from `~/agents/claude/CLAUDE.project.md` → a sync is a pure
`cp` and any delta written into it is lost at the next one. **Every project-specific override lives in
`.claude/rules/`**, which upstream does not ship. Where the template and a rules file disagree, the
rules file wins.

After a refresh, re-read this table and `assurance-posture.md` before acting on the template's words.

| template clause | repo ruling |
| --- | --- |
| IMPLEMENT ships "CI from the first commit" | **CI still REJECTED.** Single-user repo: ~zero marginal catch over the `.githooks/` pre-commit hook (D-007), and the one novel hazard — moved-dir venv shebangs — cannot occur in CI's fresh-clone env (L-019). **Update automation ADOPTED** (user ruling): `.github/dependabot.yml`, `package-ecosystem: uv`, weekly, grouped. It is the only file under `.github/`, because an advisory feed between commits is the one thing a local hook structurally cannot do. |
| "adversarial review ledgered in `.agent/review.md`" | **RETIRED** (D-012, L-032). A unit's check set is fixed and adjudicated inside its own session; no ledger file, no cross-session review state, no `rev2`/`audit` roles. `rev` itself is live, row below. |
| `Session flow` Dispatch → global `Subagents` b1, "a closing diff is always its own stretch, every other stretch dispatches" | **ADOPTED** — b1 owns the trigger and the role map, and the template now draws the closing `rev` in every phase itself ⇒ this row adds bindings only. The six-licence solo list is closed, so ordinary code authorship is on none of it. `rev` runs INSIDE the unit that authored the diff, docs and law units included, and the last unit owns any phase-closing diff ⇒ **phase close adds no review of its own**. What L-032 cut was apparatus, not a second reader ⇒ the ledger + separate review pass stay retired (row above) while `rev` stays live. |
| "security scanning + update automation in gate + CI" | **Split by hermeticity** (user ruling). IN the gate, because both run offline: static analysis = ruff's `S` (flake8-bandit) inside the existing `ruff-check` step, and a `secrets` step = `detect-secrets` over the walked tree. OUT of the gate: `pip-audit` needs an advisory feed, so it stays in the L-018 recipe (`packaging-deps.md`) as a queued unit (`.agent/deferred.md` → *Maintenance + security pass*) — `gate.py` is hermetic and a network step would break that. Update automation = Dependabot, row above. |
| "contracts + tiers" per unit | **The acceptance contract IS the unit's `.agent/deferred.md` row**; its outcome is the commit body. No `.agent/contracts/`, no contract fingerprints, no claim registry, no mutation matrix. |
| "`prototype/` retires at close" | **Inapplicable.** This repo never had a PROTOTYPE phase — the CLI itself has been the inspectable artifact since its first commit. Verification integrity's `prototype/` carve-out is inert for the same reason: every check here binds, and no shortcut licence exists anywhere in the tree. |
| "A skipped, xfailed, deleted or tier-demoted case" earns a queue row + approval | **Narrowed to demotions, and made executable.** Every skip here gates on an absent RESOURCE — weights, corpus or accelerator — so its case stays live and earns no row. The `pytest` step decides which is which: a skip reason must open `absent: `, an xfail fails outright, and a SKIPPED or XFAILED case therefore cannot ride a green gate short of in-tree pytest configuration, whose limits `toolchain.md` names. Deleting a case and demoting a tier leave no such trace, so those two stay under the approval law alone. A real demotion of any kind still earns the row and the approval first. |

The template's own live clauses bind as written: assurance tiers · verification integrity, whose
red-first witness here is L-022 neutralization and whose skip rule the gate decides
(`assurance-posture.md`) · units = shortest path to the artifact, off-path work →
`.agent/deferred.md` rows · worktree isolation · two-tier delegated reports
· the review-termination rule (fixed check set, adjudicate every row, evidence bar → global
`Subagents`) applied *within* a unit · `Close order` · the commit convention, whose body carries the
unit's dispatch line — which every ad-hoc request states in the turn that opens the work, this repo
having no ITERATE phase · `Deferred` = the queue pointer + the unfinished units, so `spec.md`'s spine
line and `.agent/deferred.md` move in one commit · a purpose-built check shipping with the input that
fires it, recorded in `toolchain.md` beside the gate invocation.
