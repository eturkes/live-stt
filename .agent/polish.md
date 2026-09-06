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

- **P-014 · NPU decode stalls in bursts that no committed trace sampled.** `pri 1` · `size M` ·
  **`stale(the acceptance prices a paced replay of a stall the committed trace now refutes; which
  of the row's own two exits to take is a scope ruling, not a polish call)`** — a `/session-roadmap`
  session re-rules it: either fund reproducing the elevated-cost state and characterizing its cause,
  or re-scope to "worst observed decode vs `AUDIO_HEADROOM_S`" and close it on committed data
  (`caption_trace.json` @ n=215 + M11.4's `SCALE_LADDER` margin), which needs no accelerator run.
  Two NPU replays of `gongitsune_01.wav` (M12.1) produced byte-identical captions and boundaries but
  not identical cost: each run carried ONE cluster of 3-4 consecutive captions decoding at 1.9-11.7×
  real time — worst **55.74 s of decode on a 5.36 s utterance** — worth 77.0 s of 284.7 s and 64.9 s
  of 275.7 s. Onset moved between runs (caption 13 @ wall 26 s vs caption 34 @ wall 94 s), so it is
  machine state, not clip content. Clean captions are steady either side: per-utterance RTF mean
  0.789 / 0.796, p50 0.79. Live, a 42-56 s decode overruns `AUDIO_HEADROOM_S`=2 s by an order of
  magnitude, and M11.4's drop-free result cannot speak to it: that trace was built from the two
  pause-free clips and its max per-update cost was ~1 s, so this tail was never in the sample.
  Qualifier to carry into any measurement: replay decodes back-to-back with no pacing, so sustained
  load may provoke the event that live 1 s cadence would not.
  Acceptance: build per-update traces for this clip over ≥3 runs (`tests/build_vac_trace.py`
  pattern), replay them through `eval_backpressure.py`'s VAC arm at 20 ms pacing, and report
  `dropped` / `forced_trims` / audio-queue high-water per run. Drop-free ⇒ close with the numbers; a
  drop ⇒ this is spine work, so it moves to Spine flags and to the user rather than being fixed here.
  Evidence: `tests/caption_trace.json` (run A, committed); run B was scratch-only and is reproduced
  by rerunning `tests/build_caption_trace.py`.
  **M12.3's free sample says the burst is neither per-run nor per-unit-time — it did not happen.**
  One continuous 6-section pass (215 captions, 848.350 s of audio, ~4.4× the wall clock of one M12.1
  replay) carried **zero stalls**: max decode 7.42 s on a 9.84 s utterance, only 4 captions over 5 s.
  Section 01 alone reproduces the comparison exactly, same clip and device: decode sum **284.7 s →
  135.3 s**, p50 **2.890 → 1.702 s**, max **55.74 → 7.42 s**, RTF **0.99 → 0.469**. So M12.1's run
  was not a clean baseline plus a burst — its ordinary captions were ~1.7× slow too, which points at
  contention/thermal state across the whole replay rather than at an NPU scheduling event, and puts
  M12.1's RTF far outside D-016's measured 0.48-0.61 band while this run sits inside it.
  Consequence for the acceptance: the paced replay cannot be built on a stall that will not
  reproduce on demand. Either reproduce the elevated-cost state first and characterize what causes
  it, or re-scope this row to "worst observed decode vs `AUDIO_HEADROOM_S`" and close it on the
  `SCALE_LADDER` margin M11.4 already measured. Evidence: `tests/caption_trace.json` @ M12.3 (n=215,
  clean) vs the same file at `f25cfb5` (n=67, one burst).

P-012 was PROMOTED, not pruned: re-sizing it against tree showed a milestone wearing a `size=M`
label, and the user funded it on 2026-09-02 as **M12** in `roadmap.md`, which now owns its
why/evidence/acceptance whole. Do not re-file it here.

## Spine flags

- **spine? Closing the terminal kills live-stt outright and the last EN dies with it | why: P-017's
  drain is now PROVEN correct, so only process death explains the two short runs.** The drain lock
  (`test_the_final_utterance_keeps_its_en_line_through_shutdown`) replays shutdown's order over the
  real `_vac_segments` + `CodexTranslator` + `TranscriptFile`, with one turn in flight and one
  caption queued behind it, and both EN lines land; a mutant that clears the backlog at the sentinel
  reproduces the live symptom exactly (`JA 1, JA 2, EN 1`). `_install_signal_handlers` covers
  `(SIGINT, SIGTERM)` only, and SIGHUP's default action terminates — measured: a Python child with
  asyncio SIGINT/SIGTERM handlers installed exits `-1` (= −SIGHUP) with no cleanup. JA flushes as it
  lands while EN needs the drain ⇒ exactly one missing EN, the last, in both runs (8 JA/7 EN, 7 JA/6
  EN, final line `JA n` in each). **The fix is not one token.** Handling SIGHUP runs the drain
  against a dead pty: writing to a pty slave after its master closes raises `OSError` errno 5
  (measured), and `emit_line` prints to stdout BEFORE `output_file.write`, so the EN line would
  raise exactly where it is meant to be saved. A real fix = catch SIGHUP **and** make `emit_line`'s
  stdout write survive a dead terminal — shipped-behaviour policy, so it needs a user ruling.

- **The EN leg died permanently 194 turns into the first real-world session.** CAUSE FOUND and it is
  upstream: n=195/196/197 are three consecutive M13 runaway captions (341/444/86 chars) = exactly
  `TRANSLATE_MAX_FAILURES`=3, so **`roadmap.md` M13 owns the trigger** and no diagnosis unit is
  needed here. What survives as an open policy question, and only as one: a 3-strike disable that is
  permanent for the session costs every later turn on a 1-3 h soak target (`memory.md` § Smoke), and
  single runaways at n=130 and n=138 translated fine, so the failures that trip it can be transient.

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
