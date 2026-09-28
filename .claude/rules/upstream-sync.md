---
paths: ["CLAUDE.md"]
---

# Upstream sync

`CLAUDE.md` is refreshed **byte-for-byte** from `~/.local/app/agents/claude/CLAUDE.project.md` → a
sync is a pure `cp` and any delta written into it is lost at the next one. **Every project-specific
override lives in `.claude/rules/`**, which upstream does not ship. Where the template and a rules
file disagree, the rules file wins.

`last-sync = agents@8fc2e19`. A refresh = one migration-only session on
`~/.local/app/agents/claude/prompts/refresh.md`: `cp` the template, re-derive last-sync as the
upstream commit whose template `cmp`-equals `git show HEAD:CLAUDE.md` (the recorded value may be
stale), read `git -C ~/.local/app/agents diff <last-sync> HEAD -- claude/CLAUDE.project.md` plus that
range's commit bodies, then re-read this table and `assurance-posture.md` before acting on the
template's words.

| template clause | repo ruling |
| --- | --- |
| IMPLEMENT ships "CI from the first commit" | **CI still REJECTED.** Single-user repo: ~zero marginal catch over the `.githooks/` pre-commit hook (D-007), and the one novel hazard — moved-dir venv shebangs — cannot occur in CI's fresh-clone env (L-019). **Update automation ADOPTED** (user ruling): `.github/dependabot.yml`, `package-ecosystem: uv`, weekly, grouped. It is the only file under `.github/`, because an advisory feed between commits is the one thing a local hook structurally cannot do. |
| "adversarial review ledgered in `.agent/review.md`" + "its own pass after implementation" | **RETIRED** (D-012, L-032). A unit's check set is fixed and adjudicated inside its own session; no ledger file, no separate review pass, no cross-session review state. `reviewer` itself is live, row below. |
| `Session flow` Teammates, "on every closing diff — one per lens in IMPLEMENT, one covering every lens elsewhere" | **ADOPTED** — global `Subagents` owns the triggers and the role map, and the template draws the closing `reviewer` in every phase ⇒ this row adds bindings only. `reviewer` runs INSIDE the unit that authored the diff, docs and law units included, and the last unit owns any phase-closing diff ⇒ **phase close adds no review of its own**. What L-032 cut was apparatus, not a second reader ⇒ the ledger + separate review pass stay retired (row above) while `reviewer` stays live. |
| `Tasks`, "on-path finds appended, ticked rows cleared at phase close" | **MAINTAIN has no phase close ⇒ each request session is its own** (user ruling, `.agent/spec.md` `Decisions`): its closing commit deletes the funded row from `.agent/deferred.md` and `Tasks` together, and a `- [x] <sha>` row lives only between one session's commits. Every open `Tasks` row is one queue row ⇒ a find on the unit's path lands as a queue row with its acceptance check plus a `Tasks` row, in one commit; `tests/test_law_consistency.py` locks the pairing. |
| "security scanning + update automation in gate + CI" | **Split by hermeticity** (user ruling). IN the gate, because both run offline: static analysis = ruff's `S` (flake8-bandit) inside the existing `ruff-check` step, and a `secrets` step = `detect-secrets` over the walked tree. OUT of the gate: `pip-audit` needs an advisory feed, so it stays in the L-018 recipe (`packaging-deps.md`), which every MAINTAIN security pass runs — `gate.py` is hermetic and a network step would break that. Update automation = Dependabot, row above. |
| "contracts + tiers" per unit | **The acceptance contract IS the unit's `.agent/deferred.md` row**; its outcome is the commit body. No `.agent/contracts/`, no contract fingerprints, no claim registry, no mutation matrix. |
| "`prototype/` retires at close" | **Inapplicable.** This repo never had a PROTOTYPE phase — the CLI itself has been the inspectable artifact since its first commit. Verification integrity's `prototype/` carve-out is inert for the same reason: every check here binds, and no shortcut licence exists anywhere in the tree. |
| "A skipped, xfailed, deleted or tier-demoted case" earns a queue row + approval | **Narrowed to demotions, and made executable.** Every skip here gates on an absent RESOURCE — weights, corpus or accelerator — so its case stays live and earns no row. The `pytest` step decides which is which: a skip reason must open `absent: `, an xfail fails outright, and a SKIPPED or XFAILED case therefore cannot ride a green gate short of in-tree pytest configuration, whose limits `toolchain.md` names. Deleting a case and demoting a tier leave no such trace, so those two stay under the approval law alone. A real demotion of any kind still earns the row and the approval first. |

The template's own live clauses bind as written: assurance tiers · verification integrity, whose
red-first witness here is L-022 neutralization and whose skip rule the gate decides
(`assurance-posture.md`) · units = shortest path to the artifact, off-path work →
`.agent/deferred.md` rows · two-tier delegated reports · the review-termination rule (fixed check
set, adjudicate every row, evidence bar → global `Subagents`) applied *within* a unit · the commit
convention, whose body names each teammate the unit used (name, role, verdict) · thinking depth =
the session's `--effort`, set at launch · `Tasks` = the open units in queue order + the
`.agent/deferred.md` pointer, so a `Tasks` row and its queue row move in one commit.
