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
  - **prerequisite, now measured against tree — this row is corpus acquisition first, and that is
    why it outgrew `size=M`.** Neither committed WAV can carry it, and the shortfall is in CAPTIONS,
    not sentences: VAC emits one caption per VAD utterance, so a pause-free clip collapses many
    sentences into one. `retention_probe` is 8 utterances over 182.482 s (D-016(e), committed), so
    its 松井/森永/フィリピン runs of 13/12/12 sentences land in at most 8 captions. `gongitsune_01.wav`
    is paused narration whose captions are sentence-sized or larger, and 兵十 occupies 8 clean
    sentences, so it cannot exceed 8 either (inference from the reference text, not a replay).
    Acquiring 「ごん狐」 sections 二+ means new pinned audio + Kokoro alignment per section under
    L-017, then one NPU replay. That replay is arm-independent: under the NPU default the recogniser
    ignores hotwords, so one caption stream serves every arm.
  - the honest cheap alternative is NOT available: injecting synthetic recognition noise into the
    clean corpus measures the noise model, and realism is the whole question.

## Spine flags

None open.
