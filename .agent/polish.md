# Polish — live-stt

Deferred-perfection register. `/session-polish` is its sole consumer: pick by `pri`, size-fit the
remaining window, run one item at a time under the gate identity in `memory.md`, prune the row in the
commit that lands it. Off-spine improvements are born here at deferral time with the acceptance check
written while the evidence is fresh. Milestone/unit state stays in `roadmap.md`; nothing here gates a
milestone.

Row shape: `P-<n>` monotonic and never reused (pruning leaves gaps) · `pri` 1 = do first … 3 =
whenever · `size` S ≈ ≤15 % of a window, M ≈ ≤35 %, L = the session. An item whose evidence pointer
or acceptance check stops holding takes `stale(<why>)` in place. A finding that implies spine work
goes under Spine flags and to the user instead of running here.

## Open

- **P-010 — Pin the meter's status-line width arithmetic.** `pri=2` `size=S`
  - why: `live_stt.py`'s `meter` reserves 3 columns for the separator between the status counters and
    the settling caption (`room = max(0, (columns - 1) - len(status) - 3)`, then `'   ' + partial`).
    Changing that 3 to a 4 leaves the whole suite green, so the exact remaining-width arithmetic is
    unpinned and an off-by-one that silently eats a caption character would ship. `tests/
    test_shipped_path.py` proves only that a fitting partial is retained and a long one is
    tail-truncated. Harvested from M11.2's review table (row R25) before that ledger was deleted; it
    is the one finding in 119 enumerated rows that named live code with no committed fix.
  - acceptance: a test asserts the exact rendered body and the maximal tail length for a fixed
    terminal width and status string, plus that an empty partial adds no separator while the status
    is nonempty; mutating the reserved width by one turns it red.

- **P-003 — Converge the ten remaining `ruff format` files so the step can go repo-wide.** `pri=2`
  `size=S`
  - why: `gate.py`'s format step is per-touched-file because repo-wide is red. M11.1 measured the red
    set and the recorded justification did not hold: all 19 pre-existing hunks across 11 files are
    ruff joining wrapped expressions under `line-length = 100`, plus two blank-line fixes. No comment
    content and no aligned table is at stake, so D-006/L-001 do not defend them. `live_stt.py`
    converged in M11.1. Re-measure the file list first — the 2026-09-02 cut deleted several of the
    originally-named files.
  - acceptance: `python -m ruff format --check .` exits 0 repo-wide; the full gate stays green;
    `gate.py`'s format step becomes `[*RUFF, "format", "--check", "."]` with the touched-file
    machinery and its two tests deleted; no comment text changes in the diff.

- **P-008 — Make `replay.py --engine whisper` fail cleanly on a farm-less box.** `pri=2` `size=S`
  - why: `main()` preflights with `check_models` + `check_device`, but the CLI still reaches OpenVINO
    and aborts inside it on a box with no accel farm sourced — the process dies in the native loader
    rather than returning the preflight's message. The test path is already clean (`_not_ready`
    skips), so this is the CLI surface alone, and it is the surface a human runs by hand.
  - acceptance: `uv run python replay.py <wav> --engine whisper` on a farm-less box exits nonzero
    with the `check_device` message on stderr and no native abort; the same command with the farm
    sourced still replays; a test asserts the farm-less exit path without requiring hardware.

- **P-009 — Rule on the VAC repetition artifact pinned in the `whisper/long` golden.** `pri=3`
  `size=M`
  - why: the committed row's second utterance carries a real LocalAgreement-2 repetition —
    `…ジェミニAPIに送ってに送って、…`. M11.3c characterized it and deliberately did not fix it (D-014
    makes goldens characterization snapshots; pinning is what makes a future commit-policy change
    visible), so the golden asserts a known-wrong transcript as correct. Right for a snapshot, wrong
    as an end state: nothing distinguishes "this artifact is still here" from "this artifact is
    intended". `forced_trims == 0` on the probe clips, so it is not a trim effect.
  - evidence: `tests/replay_goldens.json` `whisper/long` segment 2; `streaming.py`'s
    `common_prefix`/`_trim` commit path.
  - acceptance: a named root cause for the duplication in the LocalAgreement-2 commit path, then
    either a fix that removes it with `tests/eval_cer.py` showing no CER regression and the golden
    regenerated, or a written ruling that it is inherent to the policy, recorded beside the golden
    row so the next reader does not re-derive it. Deciding either way closes the row.

- **P-002 — Decide whether the session context should learn EN renderings, not just JA terms.**
  `pri=3` `size=M`
  - why: `SessionContext` (D-015) holds a JA term list and tells the translator to render each one
    consistently, but never learns *which* EN rendering a term received. A first-draft `observe_en`
    pairing JA terms with capitalized EN runs was cut before shipping: aligning a rendering out of
    unaligned JA/EN caption pairs is guesswork, its capitalization and title heuristics were
    untestable, and no failure mode was named that the plain term list does not already prevent.
    Repeating the term list every turn already pins a name to one spelling across thread rotations,
    so the learner has to beat that, not beat nothing.
  - evidence: `live_stt.py` `SessionContext.translator_brief` carries the term list and the
    consistency instruction; the cut is why it lists terms unpaired.
  - acceptance: on a JA corpus with a proper noun recurring across ≥10 turns, count distinct EN
    renderings of that noun per session with and without the learner. The learner ships only if it
    strictly reduces the distinct-rendering count on a corpus it was not tuned on, and a paired
    judged comparison on the same items shows no adequacy regression against the current brief.
    Either arm failing keeps the term list unpaired and closes this row.

## Spine flags

(none)
