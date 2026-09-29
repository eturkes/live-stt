# Assurance posture

This file owns what assurance the repo actually runs; `upstream-sync.md` tables the `CLAUDE.md`
clauses it overrides. Where the template and a rules file disagree, the rules file wins.

**User ruling: personal tool, not an industrial product** — the apparatus reached ~11,100 lines
around a 2,108-line tool and was cut with zero production change (`becc22b`, L-032). Verification =
`python gate.py` (7 blocking steps, invocation → `toolchain.md`) + `replay.py` goldens +
`tests/eval_cer.py` on demand. A unit closes when its acceptance holds under the gate and, where the
unit touches decode quality, a CER number the commit body records.

**Verification integrity binds every check here**; its "a prototype runs under PROTOTYPE law"
carve-out is inert, this repo having no prototype. Three things it fixes locally:

- **A test counts once seen RED, and L-022 neutralization is the witness.** The fix is still
  uncommitted when the lock is written, so "the unfixed revision" is the working tree with the fix
  neutralized and restored from a `cp` snapshot — never `git checkout`, which reverts the fix itself
  and re-runs the un-fixed baseline for every later mutant. The commit body records the mutation and
  the command that reddened.
- **The grading check is FIXED for the unit running under it.** The funded `.agent/deferred.md` row
  is the approval, so a threshold, case or gate moves only where that row's acceptance names the
  move, and the change records the original check's firing. A mid-unit wish to move one is a question
  for the user. The graders this holds: the `replay.py` goldens, `gate.py`'s step inventory,
  `test_backpressure.py`'s `AUDIO_HEADROOM_S` and `SEGMENT_QUEUE_MAX` bounds, the
  `CAPTION_REPEAT_*` locks, the retention CER 0.0609 a decode change must re-derive, and every
  `eval_*.py` threshold that exits or raises — `eval_cer.py`'s `MAX_CER` 0.15 and
  `MAX_EXCESS_DEL_RATE` 0.04, `eval_long_form.py`'s `SURFACE_BUDGET` 0.10, `eval_backpressure.py`'s
  bounds. Read that list as a starting point and check the script: where an evaluator only REPORTS,
  moving its number is a claim to re-measure rather than a grader to escape.
- **A skip is a demotion unless a resource is absent, and `gate.py` decides which** — a resource gate
  opens its reason with `absent: `, an xfail fails outright (`toolchain.md`). Writing that prefix is
  the declaration itself, not an L-032 escape; the escape is a demotion, and that is exactly the
  moment the queue row and the user's approval are owed. `green` names the checks run AND passed,
  with skipped, not-run and missing ones named or `none`.

**L-004 — mic and real-terminal paths are agent-unverifiable.** No mic, no interactive TTY here. A
change touching `sd.InputStream`, `audio_callback`, real-time latency, Ctrl+C or multi-hour behaviour
closes with a "**Did not verify (L-004)**" list naming each path for the user; never claim "done"
without it. The procedure those items point at is `live-smoke.md`.

**D-012 — judgment review is retired; a unit's check set closes inside the session that implements
it.** No review ledger, no separate review pass, no milestone review state. The unit's row in
`.agent/deferred.md` is its acceptance contract and the commit body is its outcome.

**L-031 — size work by the decisions it forces, not by its line count.** Measured over six closed
M11 units: Spearman **-0.395** between insertions and context cost, and the extremes invert — the
smallest unit by lines overran its window, the largest was the cheapest. The predictor is decisions
ORIGINATED inside the work: a unit whose whole surface was enumerated before it started is cheap; one
that hits unmapped collisions at the gate is not. Rule: one independently closable deliverable at a
time, split when a piece would force more than a couple of its own design rulings, and never write a
"close on partial acceptance if you run out of room" clause — that relabels a known overrun as a plan.

**L-032 — keep assurance proportional to the artifact; measure the ratio periodically.** Ask what
re-checks a defect for free and at what cost (`CLAUDE.md` assurance tiers): for a single-user tool a
fast suite, a replay golden answering "did the output change", and an on-demand scoring script cover
it, while provenance machinery costs more than the defects it catches. The tell was mechanical and
visible three units before it was acted on — **the contract fingerprint needed a hand-written,
individually-pinned migration clause in three consecutive units**, each waving through a legitimate
code change. *A guard that must be escaped every time it fires is not a guard.* The cut deleted
21,038 lines and took the suite from 466 tests / 48 s to 207 / 14.6 s with zero production change.

Retired — never reintroduce:

- `.agent/contracts/` ⇒ a unit writes no contract file: **the acceptance contract IS its row in
  `.agent/deferred.md`**, its outcome the commit body. Close appends no verdict table and tags no
  `archive/…` ref; a `tester` or `reviewer` brief cites the queue row as its contract.
- A separate review pass and `.agent/review*.md` (D-012). **`reviewer` itself is live INSIDE the
  unit that implements the diff** — what L-032 cut was apparatus, not a second reader. Every other
  teammate role global `Subagents` names stays live.
- Contract fingerprints · claim registries · mutation matrices.
- A separate project-memory file under `.agent/` ⇒ its law lives in these rules files, which reach
  MAIN and every teammate on their own. The attached set is **`.agent/spec.md` alone**: a MAIN-owned
  mutable ledger written mid-session, so it stays attached rather than moving here, where a frozen
  snapshot would read as current. Closed-milestone detail lives in `.agent/archive/` and the
  deferral queue in `.agent/deferred.md` — both committed, outside the attached set, read on demand.
  Neither is project memory: the queue is a monotonic funding list, so attaching it would make it a
  permanent growth term on every context.

Global `Subagents` owns the triggers and the roles; the bindings this repo adds are these. A
`reviewer` runs inside the unit that authored the diff, docs and law units included — D-012 retired
the separate pass and the ledger, never the second reader. Phase close adds no review of its own; it
is the last unit's. A diff whose vocabulary is security-flavoured (the `secrets` step, credential
handling) stays MAIN-side (global `Subagents`), because astra's request classifier ends any context
holding that material, every successor that reads it included.

Its report (`CLAUDE.md` review-termination rule, evidence bar → global `Subagents`) fixes the check
set before reading the diff, then folds it into **≤20 COMPOUND risk-ranked rows**, each carrying its
subchecks: adjudicate every subcheck, and route whatever a tool can decide into `gate.py` rather than
a review row. The cap bounds presentation, never coverage — a check set that will not fit means the
unit is oversized (L-031) ⇒ report that and let MAIN split it. The report is the whole record; no
ledger carries rows between sessions.

`<window>` in a gauge record = what `context-gauge` prints (mechanics → global `CLAUDE.md`): the raw
**1M** for MAIN, **272K** for a teammate; `/240K` in every gauge M11-M13 recorded, `/273K` in the
later ones. Compare units by absolute K; the percentage is denominator-relative. **Size by WORK
above the fresh-session baseline, never by a `main=` total**, because the baseline moves with the
attached state while the work does not. The analogs = the last three closed M-series units:
**106K / 121K / 130K** of work (`main=` 181K / 196K / 205K less a ≈75K baseline that carried 182 KB
attached) against bottom-up estimates of 92K / 100K / 40K ⇒ ratios 1.15 / 1.21 / 3.25, the outlier
being the SMALLEST unit by estimate, where the small denominator makes the ratio noise and the
absolute 130K is the signal. Size a new unit bottom-up against those work figures, never against a
literal written here. **A fresh MAIN session now opens at 47K** over 89 KB attached — global
`CLAUDE.md` 27.0 KB, `CLAUDE.md` 14.4 KB, `CLAUDE.local.md` 5.1 KB, `.agent/spec.md` 14.2 KB, the
three bare rules files 28.7 KB. MAIN at 1M is collapse-managed and has no compaction window a unit
must fit, so the budget that binds is a teammate's: `context-alert` hands it off at 205K, so a
brief's bottom-up work plus that teammate's own opening baseline (unmeasured here) must land under
205K.
