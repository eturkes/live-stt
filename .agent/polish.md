# Polish — live-stt

Deferred-perfection register. `/session-polish` is its sole consumer: pick by `pri`, size-fit the remaining window, run one item at a time under the gate identity in `memory.md`, prune the row in the commit that lands it. Off-spine improvements are born here at deferral time with the acceptance check written while the evidence is fresh (`CLAUDE.md` Engineering). Milestone/unit state stays in `roadmap.md`; nothing here gates a milestone. Scope = out-of-contract items alone: a unit's own check-set rows are review debt and belong to MILESTONE-REVIEW's committed ledger `.agent/review-m<m>.md`, which gates REVIEWED, so they are never parked here.

Row shape: `P-<n>` monotonic and never reused (pruning leaves gaps) · `pri` 1 = do first … 3 = whenever · `size` S ≈ ≤15 % of a window, M ≈ ≤35 %, L = the session · `tier` = the assurance tier the item runs inside, which it never raises. An item whose evidence pointer or acceptance check stops holding takes `stale(<why>)` in place and waits for the next `/session-roadmap` session to re-rule it. A finding that implies spine work goes under Spine flags and to the user instead of running here.

## Open

- **P-005 — Port the per-session context census out of `.scratch/`.** `pri=3` `size=S` `tier=docs`
  - why: `.scratch/cost_archaeology.py` produced the measured basis for L-031 and for M11's whole unit split, and `.scratch/` is gitignored, so the tool backing a durable sizing claim cannot rerun from committed state (`CLAUDE.md` Engineering). It reads Claude Code transcripts under `~/.claude/projects/<project>/`, computes per-session turns, peak usage, output tokens and tool mix, and lists each session's subagents. Nothing else in the repo measures this, and every future PLANNING session wants it.
  - evidence: `.scratch/cost_archaeology.py`; `.scratch/m11-cost-model.md` holds its M11 output; `.agent/memory.md` L-031 cites both.
  - acceptance: the script lives at a committed path outside `tests/` (it tests nothing), runs with no argument to census the current project, and exits nonzero with a clear message when the transcript directory is absent; `.agent/memory.md` L-031 and this row's pointer are updated to the committed path; `python gate.py` stays green.

- **P-003 — Converge the ten remaining `ruff format` files so the step can go repo-wide.** `pri=2` `size=S` `tier=docs`
  - why: `gate.py`'s format step is per-touched-file because repo-wide is red. M11.1 measured the red set and the recorded justification did not hold: all 19 pre-existing hunks across 11 files are ruff joining wrapped expressions under `line-length = 100`, plus two blank-line fixes. No comment content and no aligned table is at stake, so D-006/L-001 do not defend them. `live_stt.py` converged in M11.1; ten files remain (`cer.py`, `tests/build_stressor.py`, `tests/eval_long_form.py`, `tests/gen_replay_goldens.py`, `tests/test_audio.py`, `tests/test_backpressure.py`, `tests/test_cer.py`, `tests/test_context.py`, `tests/test_replay.py`, `tests/test_streaming.py`). Per-touched-file works but leaves the gate unable to answer "is the repo clean?" in one command.
  - evidence: `python -m ruff format --diff .` at M11.1 = 19 hunks, all rewrapping; `.agent/memory.md` § Gate identity carries the corrected rationale.
  - acceptance: `python -m ruff format --check .` exits 0 repo-wide; the full gate stays green; `gate.py`'s format step becomes `[*RUFF, "format", "--check", "."]` with the touched-file machinery and its two tests deleted; no comment text changes in the diff (`git diff -U0 | grep '^[-+].*#'` shows only rewrapped code lines).

- **P-002 — Decide whether the session context should learn EN renderings, not just JA terms.** `pri=3` `size=M` `tier=kernel`
  - why: `SessionContext` (D-015) holds a JA term list and tells the translator to render each one consistently, but it never learns *which* EN rendering a term received. A first-draft `observe_en` that paired JA terms with capitalized EN runs was cut before it shipped: aligning a rendering out of unaligned JA/EN caption pairs is guesswork, the capitalization and title heuristics it needed were untestable, and no concrete failure mode was named that the plain term list does not already prevent. Repeating the same term list every turn already pins a name to one spelling across thread rotations, so the learner has to beat that, not beat nothing.
  - evidence: `live_stt.py` `SessionContext.translator_brief` carries the term list and the consistency instruction; the cut is why it lists terms unpaired.
  - acceptance: on a JA corpus with a proper noun recurring across ≥10 turns, count distinct EN renderings of that noun per session with and without the learner. The learner ships only if it strictly reduces the distinct-rendering count on a corpus it was not tuned on, AND `tests/eval_translation.py` adequacy shows no regression against the current brief judged paired on the same items. Either arm failing keeps the term list unpaired and closes this row.

## Spine flags

(none)

- **P-006 — Unify the resumable-JSONL writer across evaluators.** `tests/eval_vac.py` ports `tests/eval_streaming.py:656-846` (three journals, prefix reconciliation, paired fsync) with its own schema/validators, so the mechanism now lives in two places. `pri=med`. Acceptance: one shared writer parameterized by validators + row identity, both evaluators calling it, `streaming_baseline.json` and `vac_baseline.json` each rebuilding byte-identically under `--aggregate-only`, and the full gate green.
