---
paths: ["CLAUDE.md", ".agent/spec-lint"]
---

# Upstream sync

`CLAUDE.md` = `~/.local/app/agents/claude/CLAUDE.project.md` and `.agent/spec-lint` =
`~/.local/app/agents/claude/spec-lint`, each **byte for byte** → a sync is a pure `cp` and any delta
written into either dies at the next one. The template sets defaults; `.claude/rules/`,
which upstream does not ship, holds this repo's rulings on them. Each ruling keys on the clause it
overrides, adapts, retires or marks inapplicable that structure, and names its replacement or the
user's waiver; every phase body follows it, and a retired structure stays retired. Where the
template and a rules file disagree, the rules file wins. A clause with no row below binds as written.

`last-sync = agents@8bc2c15`. A refresh = one migration-only session on
`~/.local/app/agents/claude/prompts/{auto,steered}/refresh.md`, which derives the upstream delta
from that value; re-read this table and `assurance-posture.md` before acting on the template's words.

The table = the index of every ruling, keyed on quoted template text (project or global
`CLAUDE.md`); `tests/test_law_consistency.py` holds each key to a live anchor.

| template clause | repo ruling |
| --- | --- |
| IMPLEMENT's CI, "gates + CI from its first commit" + "update automation in gate + CI" | **RETIRED** (user waiver: `.agent/spec.md` `Decisions` rules CI mirroring the local hook out of scope). Replacement = the `.githooks/` pre-commit hook (D-007): single-user repo ⇒ ~zero marginal catch, and the one novel hazard — moved-dir venv shebangs — cannot occur in CI's fresh-clone env (L-019). **Update automation ADOPTED** (user ruling): `.github/dependabot.yml`, `package-ecosystem: uv`, weekly, grouped — the only file under `.github/`, since an advisory feed between commits is the one thing a local hook structurally cannot do. |
| "adversarial review ledgered in `.agent/review.md`" + "its own pass after implementation" | **RETIRED** (D-012, L-032; user ruling: personal tool). Replacement = the unit's check set, fixed before reading and adjudicated inside the session that implements it; outcome = the commit body. No ledger file, no separate review pass, no cross-session review state. Owner: `assurance-posture.md`. |
| `Session flow` Teammates, "Closing-diff `reviewer` → one per lens in IMPLEMENT (each lens its full attention), one covering every lens elsewhere" + "MAINTAIN's closing-diff `reviewer` = where needed" | **ADAPTED**: review = a fit per case, never a repo mandate (user ruling); where a diff earns it, the reviewer runs INSIDE the unit that authored the diff, at the template's count. The last unit owns any phase-closing diff ⇒ phase close adds no review of its own. Each report = one table of compound risk-ranked rows. Lenses → `assurance-posture.md`. |
| `Tasks`, "the phase checklist: `- [ ]` open units in order" + "ticked rows cleared at each phase close + each MAINTAIN request's close" | **ADAPTED** (user rulings, `.agent/spec.md` `Decisions`): MAINTAIN has no phase close ⇒ each request session is its own; the commit closing a funded row deletes it from `.agent/deferred.md` and `Tasks` together, and a `- [x] <sha>` row lives only between one session's commits. `Tasks` = the open queue rows 1:1, rank + title verbatim, in queue order, then the queue pointer ⇒ an on-path find lands as a queue row + a `Tasks` row in one commit; a paused session adds its one `- [ ] RESUME: …` note (`pause.md` \| `pause-now.md`), a routing session its one `- [ ] ROUTE: …` row (`route.md`). Locked by `tests/test_law_consistency.py`. |
| `Session flow` Routing, "Routed state: `Phase` line 1 = `<PHASE> — <scope>`" | **ADAPTED** at the `Tasks` lock alone: a routed phase's own checklist rows, beside a suspended MAINTAIN's queue rows, fail its 1:1 pairing ⇒ the routing session asks the user how the lock follows before it writes them. Routing otherwise binds as written. |
| global "Multi-step run → its checklist on disk" | **ADAPTED**: `Tasks` carries queue rows alone (row above) ⇒ a session's step checklist lives in `.scratch/tasks.md`, gitignored; durable progress = commits + the paired `Tasks`/queue rows. |
| "security scanning + update automation in gate + CI" | **ADAPTED — split by hermeticity** (user ruling). IN the gate, both offline: ruff's `S` (flake8-bandit) inside the `ruff-check` step + a `secrets` step = `detect-secrets` over the walked tree. OUT: `pip-audit` needs an advisory feed ⇒ the L-018 recipe (`packaging-deps.md`), which every MAINTAIN security pass runs; a network step would break `gate.py`'s hermeticity. Scan scoping + coverage limit → `toolchain.md`. |
| "contracts + tiers" | **ADAPTED**: the acceptance contract IS the unit's `.agent/deferred.md` row, its outcome the commit body. RETIRED (L-032): `.agent/contracts/`, contract fingerprints, claim registries, mutation matrices. Owner: `assurance-posture.md`. |
| "it lives where `Artifacts` records it" + "a disposable prototype retires at close" + "A prototype runs under PROTOTYPE law" | **INAPPLICABLE**: no prototype exists — the CLI itself has been the inspectable artifact since its first commit ⇒ `Artifacts` records no prototype entry, nothing retires, and verification integrity's carve-out licenses nothing: every check here binds. A scope routed later to PROTOTYPE or ITERATE takes the template default. |
| "A skipped, xfailed, deleted or tier-demoted case" | **ADAPTED — narrowed to demotions, made executable.** A skip gated on an absent RESOURCE (weights, corpus, accelerator) keeps its case live and earns no row; the `pytest` step decides which is which — a skip reason must open `absent: `, an xfail fails outright, limits → `gate.md`. Deleting a case and demoting a tier leave no such trace ⇒ both stay under the approval law alone. A real demotion of any kind still earns the row and the approval first. |
| "A test counts once seen red on the unfixed revision" | **ADAPTED**: the unfixed revision = the working tree with the fix neutralized, restored from a `cp` snapshot, never `git checkout` (L-022); the commit body records the mutation + the command that reddened. A test pinning behavior that already passes follows the global mutation-tool clause as written (user ruling) ⇒ L-022 hand mutants never credit one. Owner: `assurance-posture.md`. |
| "a threshold, case or gate changes only in a unit I approve" | **ADAPTED**: the funded `.agent/deferred.md` row = the approval ⇒ a grader moves only where that row's acceptance names the move; a mid-unit wish to move one = a question for the user. The graders held → `assurance-posture.md`. |
| "detail → `docs/` or `.agent/archive/`" | **ADAPTED**: no `docs/`. Decision detail → the `.claude/rules/` file each `D-###` names; closed records → `.agent/archive/`; incident records → `.agent/postmortems/`. |
