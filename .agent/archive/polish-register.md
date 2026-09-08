# Archived — polish register

**CLOSED historical record; superseded by `.agent/spec.md` § Deferred.** Its two live rows moved
there: **P-021** (make a live session self-evidencing) is `Deferred` 1 and **P-022**
(`CAPTION_REPEAT_MAX_CHARS` false negatives) is `Deferred` 3. The `P-<n>` numbering, the
`pri`/`size` grades and the register's own consumer are retired. Read this for the evidence and the
refutations behind a closed row — above all the shapes marked "do not re-propose".

Row shape: `P-<n>` monotonic and never reused (pruning leaves gaps) · `pri` 1 = do first … 3 =
whenever · `size` S ≈ ≤15 % of a window, M ≈ ≤35 %, L = the session. An item whose evidence pointer
or acceptance check stops holding takes `stale(<why>)` in place. A finding that implies spine work
goes under Spine flags and to the user instead of running here.

## Open

- **P-021 · make a live session self-evidencing** · `pri 2` · `size M` · off-spine for M14, which
  fixes the leg rather than watching it.
  **Why:** M14's whole scope came from reading `stt.log` and six `transcripts/*.txt` by hand — the
  9-character screen escape, both permanent-disable triggers, and the 12 declines that validated
  M13.1. That reading is repeatable and currently costs a session's attention every time, while
  M13.2 and four polish fixes sit live-unvalidated because no artifact reports on them.
  **Shape:** one committed script over `transcripts/*.txt` + a stderr log, no hardware and no
  gitignored input beyond the session's own files, reporting the observables `.claude/rules/live-smoke.md`
  already names: captions with no EN and why (declined / timeout / disabled / shutdown), degrade +
  re-enable markers with timestamps, `backlog peak:` high-water lines, caption length + repetition
  distributions with the longest surviving repetition, thread-rotation bumps, EN-behind-JA lag.
  **Acceptance:** run against the six saved sessions it re-derives, without hand-reading, the
  numbers M14's plan asserts — 1073 captions, 26 caught at unit bound 8, session 1 dying at n≈194 on
  3 strikes, session 6 on `codex app-server exited` at 14:38:49, and caption 263 escaping the screen
  — and a fresh clone with no `transcripts/` exits clean rather than failing.

- **P-022 · rule on `CAPTION_REPEAT_MAX_CHARS`=40's known false negatives** · `pri 3` · `size S` ·
  out of contract for M14.1, which held the threshold fixed at 40 by acceptance.
  **Why:** widening the unit bound to 13 made the threshold's margin measurable on the phrase side
  for the first time, and it is no longer the empty gap M13.2 recorded. Three live captions repeat a
  phrase exactly 4× and survive at 36 / 30 / 28 chars — `イメージの質問は、`×4 (session 1 n=213),
  `つもい`×10 (session 6 n=103), `翌日は翌日です`×4 (session 1 n=197) — all decode loops, none
  dropped. Against them the largest repetition a SPEAKER produced is 20 (`リソース?`×4), so the
  usable range is 21..36 and the current 40 sits above all three.
  **Cost of acting:** the margin over genuine speech falls from 2× to ~1.4× at a threshold of 28,
  and M13.1's measurement says nothing about it — 40 was picked against the TRANSLATOR's stall
  floor (shortest measured stall 120 chars), not against publication. A caption of 36 repeated
  characters does not stall a turn, so this trades reader noise against dropping real speech.
  **Acceptance:** re-derive the live population at the candidate threshold, adjudicate every newly
  dropped caption as loop or speech, and keep a ≥1.5× margin over the largest genuine repetition —
  or record the refusal with that margin as the reason. `tests/test_translator.py`'s corpus and
  boundary locks plus the pass-list in `test_shipped_path.py` are what must move with it.

P-014 was CLOSED on committed data (user ruling): its exit-2 evidence pointer was wrong — a caption's
`decode_s` is the SUM of that utterance's VAC update decodes, so its 7.420 s max is not a blockage
and never was comparable to `AUDIO_HEADROOM_S`. The comparable instrument is CARRY, and
`tests/test_backpressure.py` now gates it with no corpus and no skip. `.claude/rules/asr-pipeline.md` D-016(d) owns the
numbers whole (worst carry 0.017 s of 2.000 s over 215 captions, knee ×1.541, burst 77.231 s).

P-012 was PROMOTED, not pruned: re-sizing it against tree showed a milestone wearing a `size=M`
label, and the user funded it on 2026-09-02 as **M12** in `roadmap.md`, which now owns its
why/evidence/acceptance whole. Do not re-file it here.

## Spine flags

- **The permanent EN disable was FLAGGED and is now FUNDED — it left this register on 2026-09-06.**
  It is **`roadmap.md` M14**, which owns both triggers, the design fork and the acceptance whole:
  M14.2 respawns after `codex app-server exited`, M14.3 re-probes after a 3-strike disable. Do not
  re-file it here. **One shape this register priced is refuted and must not be re-proposed:** "keep
  permanent, because M13.2 removed the trigger that fired it" — session 6 died on an app-server EOF,
  which M13.2 does not touch, so the flag never closed as fixed upstream. Its companion insight
  survives and is why the shape is respawn+cooldown rather than cooldown alone: **a re-probe cannot
  revive an exited process.**

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
  because the defect's key IS name-shaped. `.claude/rules/translation-leg.md` D-015 carries the measurement and the one
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
