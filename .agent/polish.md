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
  JA-only turns to the end), plus one isolated gap at n=144 consistent with a single failed or
  evicted turn. Which of the two paths fired is unrecoverable.
  Acceptance: a run that trips either path writes one marker line into the transcript naming the
  path; `tests/test_translator.py`'s in-memory `FakeProc`/`StreamReader` locks prove the marker
  lands exactly once on the 3-strike path, once on the EOF path, and never repeats after the flip.
  Neutralize each write (L-022) to prove the locks are non-vacuous.

P-012 was PROMOTED, not pruned: re-sizing it against tree showed a milestone wearing a `size=M`
label, and the user funded it on 2026-09-02 as **M12** in `roadmap.md`, which now owns its
why/evidence/acceptance whole. Do not re-file it here.

## Spine flags

- **The EN leg died permanently 194 turns into the first real-world session and never recovered.**
  Same evidence as P-015: 47 consecutive JA-only turns closed a 41-minute run, ~20 % of the session's
  captions. D-009 makes JA-only a hard degrade guarantee, so the tool behaved as designed; what is
  unmeasured is whether a 3-strike disable that is permanent for the session is the right policy on
  a multi-hour target (`memory.md` § Smoke soaks for 1-3 h). Blocked on P-015: the cause is
  stderr-only and this session's scrollback is gone, so the next live run must capture stderr or
  land the marker first. User decision — fund a diagnosis unit, or leave it until it recurs.
