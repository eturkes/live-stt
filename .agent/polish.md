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

- **P-012 — Re-measure the EN rendering learner on ASR output, not clean text.** `pri=3` `size=M`
  - why: `observe_en` (D-015, shipped by P-002) keys a learned English spelling on the JA term the
    recogniser produced. Every P-002 arm ran on clean Aozora text, which is the learner's BEST case:
    live, a mis-recognised name pairs a correct spelling to a wrong key, and the key never recurs, so
    the pairing is dead weight — or worse, the recogniser alternates two spellings of one name and
    each keeps its own rendering. Neither was measured.
  - evidence: `live_stt.py` `SessionContext.observe_en` pairs on `t in ja` against the caption text;
    `CodexTranslator.run` feeds it the accepted caption, recognition errors included.
  - acceptance: replay a WAV corpus through the shipped whisper/VAC path so the JA carries real
    recognition errors, then count both distinct EN renderings per session (with/without the learner,
    ≥3 sessions each) and pairings whose key never recurs. The learner stands as shipped if it still
    reduces the rendering count and dead pairings stay near zero; otherwise gate pairing on a term
    seen un-prompted since it was trusted.
  - prerequisite (first cost of the row): no committed corpus has both audio and a proper noun in
    ≥10 utterances — `tests/long_form.json`'s chapter-一 WAV puts 兵十 in 8. Under the NPU default the
    recogniser ignores hotwords, so ASR output does not depend on the arm: one replay yields the
    caption stream both arms then translate.

## Spine flags

None open.
