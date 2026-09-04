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

- **P-014 · NPU decode stalls in bursts that no committed trace sampled.** `pri 1` · `size M`.
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

- **P-015 · The saved transcript records a translation degrade only as absence.** `pri 2` · `size S`.
  Both degrade paths log to stderr alone — `translation disabled after %d consecutive failures`
  (`live_stt.py:1034`) and `codex app-server exited; JA-only for the rest of the session`
  (`live_stt.py:943`) — while `TranscriptFile` writes `JA`/`EN` lines only. The durable artifact
  therefore carries the loss with no cause, and a session cannot be diagnosed after the terminal
  scrollback is gone. Evidence: `transcripts/2026-09-03T14-03-43.txt` (gitignored, 434 lines) =
  **241 `JA` lines against 193 `EN`**; `EN` stops after n=194 and never returns (47 consecutive
  JA-only turns to the end), plus one isolated gap at n=144. Which path fired was recovered only by
  reading the JA side — n=195/196/197 are three consecutive M13 runaway captions, so it was the
  3-strike branch. That inference needed the caption shapes and does not generalize; a marker line
  would have said so directly.
  **Second evidence point, and it widens the row by one line.** Session 2
  (`transcripts/2026-09-03T16-07-24.txt`, 195 JA / 194 EN) lost only n=43, and its captured stderr
  carries the per-block warning the transcript omits: `translation failed (); JA-only for this
  block`. **The reason is empty** — `logger.warning("translation failed (%s); …", e)`
  (`live_stt.py:1039`) formats a `TimeoutError`, whose `str()` is `''`, so not even the stderr line
  says "timeout". `%s` → `%r`, or an explicit `type(e).__name__`, is the whole fix.
  Acceptance: a run that trips either path writes one marker line into the transcript naming the
  path, and a per-block failure logs a non-empty cause; `tests/test_translator.py`'s in-memory
  `FakeProc`/`StreamReader` locks prove the marker lands exactly once on the 3-strike path, once on
  the EOF path, never repeats after the flip, and that a timed-out turn logs a line naming
  `TimeoutError`. Neutralize each write (L-022) to prove the locks are non-vacuous.

- **P-016 · Backlog counters cannot be captured to a file.** `pri 2` · `size S`.
  `meter` returns immediately when `not _STDOUT_TTY` (`live_stt.py:1365`, symmetric with
  `_StderrFormatter`, L-006), so `q=` / `seg=` / `drop=` / `tdrop=` exist only on a live terminal.
  A user asked for "the output of the run" can therefore supply stderr (warnings only) or a
  redirected stdout (JA/EN lines, meter silenced) but never the backlog evidence, which is exactly
  what `memory.md` § Soak asks a soak run to watch. Evidence: session 2's `stt.log` = **one line for
  37 minutes**, and that session's drop counters are unrecoverable.
  Acceptance: off a TTY the counters reach the log at a bounded cadence — one stderr line only when
  a counter is nonzero or has changed, so a clean session still costs ~nothing — and the TTY path is
  byte-unchanged. Lock the off-TTY emission and the no-change silence in `tests/`; neutralize the
  gate (L-022) to prove both are non-vacuous.

- **P-017 · The last caption's EN went missing on both short runs, and the drain looks correct.**
  `pri 3` · `size S`.
  `transcripts/2026-09-03T16-01-04.txt` (8 JA / 7 EN) and `2026-09-03T16-04-25.txt` (7 JA / 6 EN)
  each lost the FINAL turn's EN; the 37-minute run between them lost only its runaway and
  translated n=195 two seconds before exit. Code reading does not explain it: `finalize()` submits
  the flushed tail (`live_stt.py:1249`) before `worker_task` is awaited, and shutdown then lands
  `submit_sentinel()` and waits `TRANSLATE_TIMEOUT_S + 5` on `translator_task`
  (`live_stt.py:1485-1489`). The 16:04 run had ~50 s of slack before the next run started, so a
  timeout is not it either. Most likely the process died on a signal outside the handled
  `(SIGINT, SIGTERM)` — `_install_signal_handlers` does not cover `SIGHUP`, so closing the terminal
  terminates immediately with no drain. Not established: those runs' stderr was not captured.
  Acceptance: reproduce off-mic through an in-memory harness — submit a final turn, land the
  sentinel, assert the EN line reaches the transcript; then rule on `SIGHUP` with that in hand.
  Close as no-defect if the drain holds and terminal death explains it, recording that.

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

- **The translator's unbounded generation was FLAGGED and is now FUNDED — it left this register the
  same day.** Session 2 measured it as a second, independent defect (`"あ" + "は"*(N-1)` >120 s at
  N≥160 while 890 real characters cost 8.1 s), the user ruled on 2026-09-03 that the mitigation goes
  first, and it is now **`roadmap.md` M13.1**, which owns its seam, calibration probe, corpus check
  and acceptance whole. Do not re-file it here.

- **P-019 · `observe_en` learned the English pronoun "I" as a name, on the shipped path.** `pri 1` ·
  `size S`. Found by M12.5, out of its contract. `_en_names` accepts any capitalized run that is not
  sentence-initial (`live_stt.py:673`), and English capitalizes "I" everywhere — so the rule that
  needs no lexicon needs exactly one entry.
  **Measured over 215 real translator turns** (`tests/eval_en_pairing.py` + the committed
  `tests/en_pairing_trace.json`): **"I" is the single most common sole proper noun in the whole
  English stream — 20 of the 63 turns that carry exactly one, ahead of the protagonist's real name
  `Gon` at 17.** Two of them (captions 77 and 127) landed while 標柱 was the only unpaired trusted
  term, meeting `CONTEXT_EN_SUPPORT`=2, so the session ended briefing the translator
  **`標柱 = I`** — carried through the last 13 of that term's 26 sightings — beside the correct
  `ゴン = Gon` and `神様 = God`. 標柱 is itself a mis-recognition of 兵十, so the learner pinned a
  wrong key to a pronoun and told the translator to render it that way every time.
  **The fix is measured and one token wide:** dropping `"I"` from `_en_names` removes the pairing and
  leaves both correct renderings identical (`ゴン = Gon` @31, `神様 = God` @194). Adding `It`/`Ah`
  changes nothing more here — both occur only inside quotes (captions 33, 83) after `thought, "`,
  which `_EN_SENTENCE`'s `[.!?]\s+` split cannot see, so they are the same class and did not land.
  **Why D-015 missed it:** its 90-word function-word lexicon comparison ran on 1,260 turns of clean
  third-person Aozora narration, which never puts a mid-sentence "I" in a caption carrying exactly
  one unpaired term. "No lexicon ships" holds for that corpus and fails on this one.
  Wider exposure, not fixed by the stop set: the same rule pins hallucinated names too — `Okkawa`
  (caption 106) and `Anke` (123) are proper nouns the translator invented, and either would have
  paired had it agreed with itself twice.
  Acceptance: over the committed pairing trace the learner acquires `ゴン = Gon` and `神様 = God` and
  nothing else, 標柱 unpaired; a model-free test locks that `_en_names` rejects a lone "I" while
  still accepting a real mid-sentence name; neutralizing the exclusion reds it (L-022).
  **The JA-side plausibility test was offered, chosen by the user, and then REFUTED by measurement
  — do not re-propose it.** The user picked it on 2026-09-04 conditional on it being the more robust
  long-term shape; it is not. Arms over the committed pairing trace, JA-side gate = "the key is a
  katakana run": shipped `{ゴン: Gon, 標柱: I, 神様: God}` → katakana-only `{ゴン: Gon}`. It drops the
  **correct** `神様 = God`, and it removes `標柱 = I` only by accident — 標柱 is kanji, so the same
  rule blocks the **entire kanji-name class**, and 兵十 / 加助 are kanji names in this very story.
  A key-shape test cannot separate these cases because the shipped defect's key IS name-shaped
  (標柱 is a mis-recognition of the name 兵十); what separates them is purely English-side, where
  `Gon` and `God` are names and `I` is a pronoun.
  **So the robust shape is English-side and has two mechanical classes plus one irreducible one.**
  (a) **Lexical** — "I" and its contractions, a closed set fixed by the language; `_EN_NAME`'s
  character class swallows the apostrophe, so `I’m`/`I’d`/`I’ll`/`I’ve` match as single tokens and
  the test must fold on the apostrophe. (b) **Positional** — `_EN_SENTENCE` splits on `[.!?]\s+`
  only, so the first word inside a quotation reads as mid-sentence: measured `It` (caption 33,
  `thought, “It’s a marker post.”`) and `Ah` (83). Neither landed here, but they are the same defect
  one quote earlier. (c) **Semantic** — a hallucinated but genuine proper noun (`Okkawa`, `Anke`) is
  unreachable by any lexical or positional rule; the defence that actually worked is
  `CONTEXT_EN_SUPPORT`=2 agreeing turns, which is what stopped `Anke`/`Hyōjū` pairing to イワシ.
  Raising that threshold is the tunable, not a key-shape test.
  Revised acceptance: (a) + (b) shipped together, `ゴン = Gon` and `神様 = God` preserved
  byte-identically on the committed trace and 標柱 unpaired; model-free locks that a lone `I` and
  each contraction are rejected while a real mid-sentence name is kept, and that a quote-opening
  word is treated as sentence-initial; neutralize each (L-022). Leave (c) recorded, unfixed.

- **P-018 · A declined caption still teaches the recogniser-side learner.** `pri 3` · `size S`.
  M13.1's screen sits at the TRANSLATOR seam (`live_stt.py:1005`), and both producers fold the
  caption into `SessionContext` first: `observe_ja` at `live_stt.py:1296` (VAC) and `:1222`
  (sherpa) run before `translator.submit` at `:1298` / `:1224`. A runaway therefore contributes one
  sighting of every `_TERM_RUN` candidate it contains (`dict.fromkeys` dedupes within a caption), so
  `CONTEXT_TERM_SUPPORT`=3 runaways repeating one unit would promote that unit and brief the
  translator on a decode artifact. Latent, not measured: the observed runaways repeat DIFFERENT
  units, so none promoted. Evidence: session 2 n=22 = `中央の`×111 (中央 is a valid 2-character kanji
  candidate); session 1 carries 次は / 私は / 副部 / クラブ / アーメン across five captions.
  Acceptance: a caption the screen declines contributes no candidate — `repeat_span(text) >=
  TRANSLATE_REPEAT_MAX_CHARS` short-circuits `observe_ja` at both call sites; lock with
  `CONTEXT_TERM_SUPPORT` identical runaways leaving `terms()` empty while an ordinary caption still
  promotes, and neutralize each guard (L-022). Decide first whether the screen belongs in the
  producers rather than in `submit`, since M13's recogniser units may want the same predicate.
