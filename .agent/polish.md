# Polish — live-stt

Deferred-perfection register. `/session-polish` is its sole consumer: pick by `pri`, size-fit the remaining window, run one item at a time under the gate identity in `memory.md`, prune the row in the commit that lands it. Off-spine improvements are born here at deferral time with the acceptance check written while the evidence is fresh (`CLAUDE.md` Engineering). Milestone/unit state stays in `roadmap.md`; nothing here gates a milestone.

Row shape: `P-<n>` monotonic and never reused (pruning leaves gaps) · `pri` 1 = do first … 3 = whenever · `size` S ≈ ≤15 % of a window, M ≈ ≤35 %, L = the session · `tier` = the assurance tier the item runs inside, which it never raises. An item whose evidence pointer or acceptance check stops holding takes `stale(<why>)` in place and waits for the next `/session-roadmap` session to re-rule it. A finding that implies spine work goes under Spine flags and to the user instead of running here.

## Open

- **P-002 — Decide whether the session context should learn EN renderings, not just JA terms.** `pri=3` `size=M` `tier=kernel`
  - why: `SessionContext` (D-015) holds a JA term list and tells the translator to render each one consistently, but it never learns *which* EN rendering a term received. A first-draft `observe_en` that paired JA terms with capitalized EN runs was cut before it shipped: aligning a rendering out of unaligned JA/EN caption pairs is guesswork, the capitalization and title heuristics it needed were untestable, and no concrete failure mode was named that the plain term list does not already prevent. Repeating the same term list every turn already pins a name to one spelling across thread rotations, so the learner has to beat that, not beat nothing.
  - evidence: `live_stt.py` `SessionContext.translator_brief` carries the term list and the consistency instruction; the cut is why it lists terms unpaired.
  - acceptance: on a JA corpus with a proper noun recurring across ≥10 turns, count distinct EN renderings of that noun per session with and without the learner. The learner ships only if it strictly reduces the distinct-rendering count on a corpus it was not tuned on, AND `tests/eval_translation.py` adequacy shows no regression against the current brief judged paired on the same items. Either arm failing keeps the term list unpaired and closes this row.

## Spine flags

(none)
