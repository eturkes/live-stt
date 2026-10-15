# Assurance posture

This file owns what assurance the repo actually runs. Each ruling on a template clause below is a
ruling tag keyed on that clause's text (`orientation.md`); where the template and a rules file
disagree, the rules file wins.

**User ruling: personal tool, not an industrial product** — the apparatus reached ~11,100 lines
around a 2,108-line tool and was cut with zero production change (`becc22b`, L-032). Verification =
`python gate.py` (8 blocking steps, invocation → `toolchain.md`) + `replay.py` goldens +
`tests/eval_cer.py` on demand. A unit closes when its acceptance holds under the gate and, where the
unit touches decode quality, a CER number the commit body records.

**Verification integrity binds every check here.** Template ruling: "it lives where `Artifacts`
records it" + "a disposable prototype retires at close" + "a prototype runs under PROTOTYPE law" →
**INAPPLICABLE** — no prototype exists, the CLI itself having been the inspectable artifact since
its first commit ⇒ `Artifacts` records no prototype entry, nothing retires, and the carve-out
licenses nothing; a scope routed later to PROTOTYPE or ITERATE takes the template default. Three
things verification integrity fixes locally:

- Template ruling: "A test counts once seen red on the unfixed revision" → **ADAPTED** — **a test
  counts once seen RED, and L-022 neutralization is the witness.** The fix is still
  uncommitted when the lock is written, so "the unfixed revision" is the working tree with the fix
  neutralized and restored from a `cp` snapshot — never `git checkout`, which reverts the fix itself
  and re-runs the un-fixed baseline for every later mutant. The commit body records the mutation and
  the command that reddened. A test pinning behavior that already passes has no unfixed revision ⇒
  `CLAUDE.md`'s `Engineering` credits it by the stack's mutation tool, as written (user ruling):
  tool-picked mutants over the files the test imports or calls, scope + command + every survivor in
  the commit body. Hand-picked L-022 mutants never credit such a pin. Each survivor takes a ruling
  against the unit's queue row from someone other than the test's author — a `reviewer` inside the
  unit, or the user; one still unruled at close becomes a `.agent/deferred.md` row in every phase
  (D-012 below).
- Template ruling: "a threshold, case or gate changes only in a unit I approve" → **ADAPTED** —
  **the grading check is FIXED for the unit running under it.** The funded `.agent/deferred.md` row
  is the approval, so a threshold, case or gate moves only where that row's acceptance names the
  move, and the change records the original check's firing. A mid-unit wish to move one is a question
  for the user. The graders this holds: the `replay.py` goldens, `gate.py`'s step inventory,
  `test_backpressure.py`'s `AUDIO_HEADROOM_S` and `SEGMENT_QUEUE_MAX` bounds, the
  `CAPTION_REPEAT_*` locks, the retention CER 0.0532 (whisper-ja-760M; turbo read 0.0609) a decode change must re-derive, and every
  `eval_*.py` threshold that exits or raises — `eval_cer.py`'s `MAX_CER` 0.15 and
  `MAX_EXCESS_DEL_RATE` 0.04, `eval_long_form.py`'s `SURFACE_BUDGET` 0.10, `eval_backpressure.py`'s
  bounds. Read that list as a starting point and check the script: where an evaluator only REPORTS,
  moving its number is a claim to re-measure rather than a grader to escape.
- Template ruling: "A skipped, xfailed, deleted or tier-demoted case" → **ADAPTED** — **a skip is a
  demotion unless a resource is absent, and `gate.py` decides which** — a resource gate opens its
  reason with `absent: ` and keeps its case live with no row, an xfail fails outright
  (`toolchain.md`). Writing that prefix is the declaration itself, not an L-032 escape; the escape
  is a demotion, and that is exactly the moment the queue row and the user's approval are owed. A
  deleted case and a demoted tier leave no trace the gate reads ⇒ both stay under the approval law
  alone. `green` names the checks run AND passed, with skipped, not-run and missing ones named or
  `none`.

**L-004 — mic and real-terminal paths are agent-unverifiable.** No mic, no interactive TTY here. A
change touching `sd.InputStream`, `audio_callback`, real-time latency, Ctrl+C or multi-hour behaviour
closes with a "**Did not verify (L-004)**" list naming each path for the user; never claim "done"
without it. The procedure those items point at is `live-smoke.md`. Template ruling: "Runnable change
→ run the real app by its recorded command" → **ADAPTED** — a change reaching the live audio path is
driven through `replay.py` (the real `worker` on a WAV, D-014) plus the CLI invocations that need no
mic (`--list-devices`, parse-time errors); the mic and real-terminal paths go on that list.

**D-012 — judgment review is retired; a unit's check set closes inside the session that implements
it.** Template ruling: "adversarial review ledgered in `.agent/review.md`" + "its own pass after
implementation" + "an open `.agent/review.md` row (IMPLEMENT: the closing-diff `reviewer` rules)" →
**RETIRED** (D-012, L-032; user ruling: personal tool) — no review ledger, no separate review pass,
no milestone review state; a mutation survivor still unruled at close lands as a
`.agent/deferred.md` row in every phase. The unit's row in `.agent/deferred.md` is its acceptance
contract and the commit body is its outcome.

**L-031 — size work by the decisions it forces, not by its line count.** Measured over six closed
M11 units: Spearman **-0.395** between insertions and context cost, and the extremes invert — the
smallest unit by lines overran its window, the largest was the cheapest. The predictor is decisions
ORIGINATED inside the work: a unit whose whole surface was enumerated before it started is cheap; one
that hits unmapped collisions at the gate is not. Rule: one independently closable deliverable at a
time, split when a piece would force more than a couple of its own design rulings, and never write a
"close on partial acceptance if you run out of room" clause — that relabels a known overrun as a plan.

**L-032 — keep assurance proportional to the artifact; measure the ratio periodically.** Ask what
re-checks a defect for free and at what cost (`CLAUDE.md`'s assurance tiers): for a single-user
tool a fast suite, a replay golden answering "did the output change", and an on-demand scoring
script cover it, while provenance machinery costs more than the defects it catches. The tell was
mechanical and visible three units before it was acted on — **the contract fingerprint needed a
hand-written, individually-pinned migration clause in three consecutive units**, each waving through
a legitimate code change. *A guard that must be escaped every time it fires is not a guard.* The cut
deleted 21,038 lines and took the suite from 466 tests / 48 s to 207 / 14.6 s with zero production
change.

Retired — never reintroduce:

- Template ruling: "contracts + tiers" → **ADAPTED** — `.agent/contracts/` retired ⇒ a unit writes
  no contract file: **the acceptance contract IS its row in `.agent/deferred.md`**, its tier
  declared there, its outcome the commit body. Close appends no verdict table and tags no
  `archive/…` ref; a `tester` or `reviewer` brief cites the queue row as its contract.
- A separate review pass and `.agent/review*.md` (D-012). **`reviewer` itself is live INSIDE the
  unit that implements the diff** — what L-032 cut was apparatus, not a second reader. Every other
  role `.claude/agents/` ships stays live.
- Contract fingerprints · claim registries · mutation matrices. A mutation-tool run the template
  clause asks for is no matrix: its record is the commit body.
- A separate project-memory file under `.agent/` ⇒ its law lives in these rules files, which reach
  MAIN and every teammate on their own. The attached set is **`.agent/spec.md` alone**: a MAIN-owned
  mutable ledger written mid-session, so it stays attached rather than moving here, where a frozen
  snapshot would read as current. Closed-milestone detail lives in `.agent/archive/` and the
  deferral queue in `.agent/deferred.md` — both committed, outside the attached set, read on demand.
  Neither is project memory: the queue is a monotonic funding list, so attaching it would make it a
  permanent growth term on every context.

`CLAUDE.md`'s `Teammates` + the `.claude/agents/` roles own the fits; this repo adds no teammate
mandate (user ruling) — `scientist` and `reviewer` included, each dispatched per its fit. A brief
whose teammate runs whisper names that teammate as the NPU holder (`toolchain.md`: two compiles on
one cache corrupt it). Template ruling: "Closing-diff `reviewer` → one per lens in IMPLEMENT (each
lens its full attention), one covering every lens elsewhere" + "MAINTAIN's closing-diff `reviewer` =
where needed" → **ADAPTED** — review = a fit per case, never a repo mandate (user ruling): where a
diff earns it, the `reviewer` runs inside the unit that authored it, at the template's count. The
lenses = `CLAUDE.md`'s code diff lenses (`Engineering`) for code, plus correctness/spec, claim
soundness, guarantee-vs-claim gaps, verification integrity and `CLAUDE.md` conformance. D-012
retired the separate pass and the ledger, never the second reader. Phase close adds no review of its
own; it is the last unit's.

Template ruling: "one row per check × target" → **ADAPTED** — each report (`CLAUDE.md`'s
review-termination rule in `Engineering`, evidence bar in `Teammates`) fixes the check set before
reading the diff, then folds it into **one table of COMPOUND risk-ranked rows**, each carrying its
subchecks: adjudicate every subcheck, and route whatever a tool can decide into
`gate.py` rather than a review row. The table's size follows the check set — a check set too large
to read as one table means the unit is oversized (L-031) ⇒ report that and let MAIN split it. The
report is the whole record; no ledger carries rows between sessions.

`<window>` in a recorded gauge reading = the loop's compaction trigger, its occupancy = the latest
assistant `usage` sum; `/240K` in every gauge M11-M13 recorded, `/273K` in the later ones, and the
raw `/1M` for MAIN before the trigger became the denominator. Compare units by absolute K; the
percentage is denominator-relative. **Size by WORK above the fresh-session baseline, never by a
`main=` total**, because the baseline moves with the attached state while the work does not. The
analogs = the last three closed M-series units: **106K / 121K / 130K** of work (`main=` 181K / 196K
/ 205K less a ≈75K baseline that carried 182 KB attached) against bottom-up estimates of 92K / 100K
/ 40K ⇒ ratios 1.15 / 1.21 / 3.25, the outlier being the SMALLEST unit by estimate, where the small
denominator makes the ratio noise and the absolute 130K is the signal. Size a new unit bottom-up
against those work figures, never against a literal written here. **A fresh MAIN session opened at
47K** over 89 KB attached — the user-level instruction file 27.0 KB, `CLAUDE.md` 14.4 KB,
`CLAUDE.local.md` 5.1 KB, `.agent/spec.md` 14.2 KB, the three bare rules files 28.7 KB — a literal
the template refresh outdated (`.agent/deferred.md` → *Re-measure the fresh-session baseline*).
MAIN's trigger sits far past any unit here, so the budget that binds is a teammate's: a brief's
bottom-up work plus that teammate's own opening baseline (unmeasured here) must land under that
teammate's context hand-off point.
