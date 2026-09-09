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
| `Session flow` Dispatch → global `Subagents` b1, "a closing diff is always its own stretch, every other stretch dispatches" | **ADOPTED.** b1 owns the trigger, the role map and the closed six-licence solo list — a solo stretch names its licence, and ordinary code authorship is on none of them. What L-032 cut was apparatus, not a second reader, so `rev` runs on every unit's closing diff, docs and law units included. Repo binding: the last unit owns any phase-closing diff ⇒ **phase close adds no review of its own**, and the ledger + separate review pass stay retired (row above). |
| "security scanning + update automation in gate + CI" | **Split by hermeticity** (user ruling). IN the gate, because both run offline: static analysis = ruff's `S` (flake8-bandit) inside the existing `ruff-check` step, and a `secrets` step = `detect-secrets` over the walked tree. OUT of the gate: `pip-audit` needs an advisory feed, so it stays in the L-018 recipe (`packaging-deps.md`) as a queued unit (`.agent/deferred.md` → *Maintenance + security pass*) — `gate.py` is hermetic and a network step would break that. Update automation = Dependabot, row above. |
| "contracts + tiers" per unit | **The acceptance contract IS the unit's `.agent/deferred.md` row**; its outcome is the commit body. No `.agent/contracts/`, no contract fingerprints, no claim registry, no mutation matrix. |
| "`prototype/` retires at close" | **Inapplicable.** This repo never had a PROTOTYPE phase — the CLI itself has been the inspectable artifact since its first commit. |
| Estimate calibration by an actual/estimate multiplier | **Superseded by L-031** — size by decisions ORIGINATED, not by lines or by a ratio. |

The template's own live clauses bind as written: assurance tiers · units = shortest path to the
artifact, off-path work → `.agent/deferred.md` rows · worktree isolation · two-tier delegated reports
· the review-termination rule (fixed check set, adjudicate every row, evidence bar → global
`Subagents`) applied *within* a unit · `Close order` · the commit convention, whose body carries the
unit's dispatch line — which every ad-hoc request states in the turn that opens the work, this repo
having no ITERATE phase · `Deferred` = the queue pointer + the unfinished units, so `spec.md`'s spine
line and `.agent/deferred.md` move in one commit · a purpose-built check shipping with the input that
fires it, recorded in `toolchain.md` beside the gate invocation.
