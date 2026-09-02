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
