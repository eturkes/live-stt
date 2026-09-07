# Assurance posture

`CLAUDE.md` + `.claude/commands/*.md` are refreshed byte-for-byte from upstream `~/agents/claude/`
(`CLAUDE.project.md` + `slash-commands/`) → a sync is a pure `cp` and a delta written into either is
lost at the next one. **Every project-specific override lives in this file**, which upstream does not
ship; it loads beside `CLAUDE.md` for MAIN + teammates alike. Where the two disagree, this file wins.

**User ruling: personal tool, not an industrial product** — the apparatus reached ~11,100 lines
around a 2,108-line tool and was cut with zero production change (`becc22b`, L-032). Verification =
`python gate.py` (6 blocking steps, invocation → `toolchain.md`) + `replay.py` goldens +
`tests/eval_cer.py` on demand. A unit closes when its acceptance holds under the gate and, where the
unit touches decode quality, a CER number the commit body records.

**L-004 — mic and real-terminal paths are agent-unverifiable.** No mic, no interactive TTY here. A
change touching `sd.InputStream`, `audio_callback`, real-time latency, Ctrl+C or multi-hour behaviour
closes with a "**Did not verify (L-004)**" list naming each path for the user; never claim "done"
without it. The procedure those items point at is `live-smoke.md`.

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

Retired — never reintroduce (roadmap `## Out of scope`):

- `.agent/contracts/` ⇒ WORK-UNIT wave 1 writes no contract file: **the acceptance contract IS the
  unit's roadmap entry**, its outcome the commit body. Close appends no verdict table and tags no
  `archive/m<m>u<u>-<role>`; a `test`/`orc`/`diff` brief cites the roadmap entry as its contract.
- MILESTONE-REVIEW + `.agent/review-m<m>.md` + the `rev`/`rev2`/`audit` roles ⇒ **IMPLEMENTED is the
  terminal milestone state**. MODE dispatch: all units DONE ⇒ PLANNING for the next milestone.
  `REVIEWED` is historical (M9, M10) and no milestone earns it again. A unit's own check set belongs
  to its WORK-UNIT session, so `/session-polish` routes one back there rather than to a review mode.
- Contract fingerprints · claim registries · mutation matrices.
- A separate project-memory file under `.agent/` ⇒ its law lives in these rules files, which reach
  MAIN and every teammate on their own. Where the upstream command files name **memory** — as a scope
  source beside the roadmap, as the gate toolchain-env recipe's home, or as a conformance target —
  read `.claude/rules/`. The
  attached set is `.agent/roadmap.md` + `.agent/polish.md` alone: both are MAIN-owned mutable ledgers
  written mid-session, so they stay attached rather than moving here, where a frozen snapshot would
  read as current.

Live, per the command files: assurance tiers · MVP-spine units · `.agent/polish.md` · worktree
isolation · every role the command file names except the retired three · two-tier reports ·
Close order · commit convention.

Adversarial review (`CLAUDE.md` review-termination rule) fixes its check set before reading the diff,
then folds it into **≤20 COMPOUND risk-ranked rows**, each carrying its subchecks: adjudicate every
subcheck, and route whatever a tool can decide into `gate.py` rather than a review row. The cap
bounds presentation, never coverage — a check set that will not fit means the unit is oversized
(L-031) ⇒ report that and let MAIN split it, which is also PLANNING's calibration input. The report
is the whole record; no ledger carries rows between sessions.

`<window>` in a gauge record = what `context-gauge` prints (mechanics → global `CLAUDE.md`): **273K**
now, `/240K` in every gauge M11-M13 recorded. Compare units by absolute K; the percentage is
denominator-relative. Those recorded `main=` actuals stay live sizing analogs ⇒ PLANNING sizes new
units bottom-up against them plus the global reserve, never against a literal written here.
