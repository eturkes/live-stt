# Assurance posture

`CLAUDE.md` + `.claude/commands/*.md` are refreshed byte-for-byte from upstream `~/agents/claude/`
(`CLAUDE.project.md` + `slash-commands/`) → a sync is a pure `cp` and a delta written into either is
lost at the next one. **Every project-specific override lives in this file**, which upstream does not
ship; it loads beside `CLAUDE.md` for MAIN + teammates alike. Where the two disagree, this file wins.

**User ruling: personal tool, not an industrial product** — the apparatus reached ~11,100 lines
around a 2,108-line tool and was cut with zero production change (`becc22b`, L-032). Verification =
`python gate.py` (6 blocking steps) + `replay.py` goldens + `tests/eval_cer.py` on demand. A unit
closes when its acceptance holds under the gate and, where the unit touches decode quality, a CER
number the commit body records.

Retired — never reintroduce (roadmap `## Out of scope`):

- `.agent/contracts/` ⇒ WORK-UNIT wave 1 writes no contract file: **the acceptance contract IS the
  unit's roadmap entry**, its outcome the commit body. Close appends no verdict table and tags no
  `archive/m<m>u<u>-<role>`; a `test`/`orc`/`diff` brief cites the roadmap entry as its contract.
- MILESTONE-REVIEW + `.agent/review-m<m>.md` + the `rev`/`rev2`/`audit` roles ⇒ **IMPLEMENTED is the
  terminal milestone state**. MODE dispatch: all units DONE ⇒ PLANNING for the next milestone.
  `REVIEWED` is historical (M9, M10) and no milestone earns it again. A unit's own check set belongs
  to its WORK-UNIT session, so `/session-polish` routes one back there rather than to a review mode.
- Contract fingerprints · claim registries · mutation matrices.

Live, per the command files: assurance tiers · MVP-spine units · `.agent/polish.md` · worktree
isolation · evidence roles (`map` `res` `spike` `test` `orc` `prod` `diff` `triage` `scout` `gate`) ·
two-tier reports · Close order · commit convention.

Adversarial review (`CLAUDE.md` review-termination rule) fixes its check set before reading the diff,
then folds it into **≤20 COMPOUND risk-ranked rows**, each carrying its subchecks: adjudicate every
subcheck, and route whatever a tool can decide into `gate.py` rather than a review row. The cap
bounds presentation, never coverage — a check set that will not fit means the unit is oversized
(L-031) ⇒ report that and let MAIN split it, which is also PLANNING's calibration input. The report
is the whole record; no ledger carries rows between sessions.

Context window = **1M**. Every gauge recorded through M13 reads `N% NK/240K`: absolute K and the
`main=`/`est` ratios stay comparable across the change, the percentage does not, and the one-window
aim rose ~4× ⇒ size new units against 1M.
