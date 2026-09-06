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

*(empty)*

P-014 was CLOSED on committed data (user ruling): its exit-2 evidence pointer was wrong — a caption's
`decode_s` is the SUM of that utterance's VAC update decodes, so its 7.420 s max is not a blockage
and never was comparable to `AUDIO_HEADROOM_S`. The comparable instrument is CARRY, and
`tests/test_backpressure.py` now gates it with no corpus and no skip. `memory.md` D-016(d) owns the
numbers whole (worst carry 0.017 s of 2.000 s over 215 captions, knee ×1.541, burst 77.231 s).

P-012 was PROMOTED, not pruned: re-sizing it against tree showed a milestone wearing a `size=M`
label, and the user funded it on 2026-09-02 as **M12** in `roadmap.md`, which now owns its
why/evidence/acceptance whole. Do not re-file it here.

## Spine flags

- **The EN leg died permanently 194 turns into the first real-world session.** CAUSE FOUND and it is
  upstream: n=195/196/197 are three consecutive M13 runaway captions (341/444/86 chars) = exactly
  `TRANSLATE_MAX_FAILURES`=3, so **`roadmap.md` M13 owns the trigger** and no diagnosis unit is
  needed here. What survives as an open policy question, and only as one: a 3-strike disable that is
  permanent for the session costs every later turn on a 1-3 h soak target (`memory.md` § Smoke), and
  single runaways at n=130 and n=138 translated fine, so the failures that trip it can be transient.
  **User ruled it to `/session-roadmap`**: PLANNING sizes it as a unit with its own acceptance rather
  than a polish item. The two shapes already priced are "keep permanent" (M13.2 removed the trigger
  that fired it, so the flag closes as fixed upstream) and a cooldown re-probe that retries once per
  window, re-enabling on a healthy turn and doubling the wait on a failure, at one stalled turn per
  probe against a genuinely dead codex.

- **The translator's unbounded generation was FLAGGED and is now FUNDED — it left this register the
  same day.** Session 2 measured it as a second, independent defect (`"あ" + "は"*(N-1)` >120 s at
  N≥160 while 890 real characters cost 8.1 s), the user ruled on 2026-09-03 that the mitigation goes
  first, and it is now **`roadmap.md` M13.1**, which owns its seam, calibration probe, corpus check
  and acceptance whole. Do not re-file it here.

- **P-019 SHIPPED, and its rejected shape must not be re-proposed.** `observe_en` learned
  `標柱 = I`; the fix is English-side and lives in `live_stt.py:663-677` — a 5-entry stop set for
  the pronoun and its contractions, plus `_EN_SENTENCE` treating a quote-opening word as
  sentence-initial. The JA-side plausibility test (key must look like a name) was offered, chosen
  by the user, and **refuted by arms over the committed pairing trace**: it drops the CORRECT
  `神様 = God` and removes `標柱 = I` only because 標柱 is kanji, so it blocks the entire kanji-name
  class (兵十 / 加助 are kanji names in this story). No key-shape test can separate the cases,
  because the defect's key IS name-shaped. `memory.md` D-015 carries the measurement and the one
  residue that stayed unfixed: a hallucinated but genuine proper noun (`Okkawa`, `Anke`) is
  unreachable by any lexical or positional rule, and `CONTEXT_EN_SUPPORT` is the only lever there.

- **Two M13.2 alternatives were MEASURED and ruled out by the user — do not re-propose either.**
  (1) **A per-utterance whisper LID gate.** Feasible and cheap: LID on the NPU is reliable from 1 s
  of audio (EN → `en` at every duration 0.5-8 s, JA → `ja` from 1 s), and a fresh pipeline costs
  0.46 s p50 / 0.60 s max to construct plus 0.54 s to detect, RSS flat at 201 MB over 40 constructs.
  It is also the only shape that works, because **`WhisperPipeline` latches its language**: after a
  `generate(language=…)` call, or after an auto-detect call, that language persists into every later
  call on the instance, and neither `language=None`/`''` (both raise) nor `set_generation_config()`
  nor a positional config clears it. So an LID gate costs a fresh pipeline per utterance. The user
  ruled the text-side rule sufficient. Note the gate would ALSO have killed the hallucination
  phrases, since silence and −30 dB noise both detect as `en`.
  (2) **An utterance-length hard cap for pace.** Clean-caption p99 is 136-312 chars ≈ 18-40 s of
  speech and the live max is 664 chars ≈ 88 s, because a caption publishes only at utterance end and
  `VAD_MAX_SPEECH_S`=20 is a soft silero cap (L-023). The user ruled utterances stay UNCAPPED: one
  utterance is one line and one turn, whatever its length.

- **P-018 CLOSED by M13.2, by construction rather than by a guard.** The flagged path was
  `observe_ja` running before `translator.submit`, so a runaway briefed the translator on a decode
  artifact. M13.2 moved the screen to PUBLICATION, upstream of every consumer, which is the
  disposition the row asked the deciding session to rule on first. `observe_ja` cannot see a
  defective caption now; the lock is `test_an_invented_caption_reaches_no_consumer_at_all`.
