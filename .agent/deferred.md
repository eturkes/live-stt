# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). The session body the user pastes names the
row it funds; closing a row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and M14's `_respawn` arm have never met a mic (`_probe` has); standing
   debt = latency feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.
2. **Explain the live audio drops.** A 26-minute live session ended at
   `backlog peak: q=2.00s drop=9033 skip=20` — 9033 discarded callback blocks of captured audio, in
   12 counter increases, every one inside a publication gap of ≥17 s, each of the 4 largest with a
   `skip=` increase within 11 s (separations 0, 1, 1, 11 s; quote separations, a ratio here is an
   artifact of the window chosen). **The obvious hypothesis is already refuted: duration alone does
   not cause it.** 16 of the 20 gaps >20 s dropped nothing, including the longest (81 s), and
   `tests/eval_backpressure.py`'s VAC arm paces 182 s of pause-free speech drop-free at a queue peak
   of 1.060 s of 2.000 s ⇒ an arm that merely paces a long utterance would pass vacuously. The
   correlate is a SCREENED caption and no further: `caption_defect` has two arms and only the
   repetition runaway costs more than real time (RTF 1.106), plain English caught by the latin rule
   costing nothing extra.
   **(a) LANDED** — `session_report.py`'s `attribute_drops` places every `backlog peak:` drop increase
   against the captions bracketing it, the publication gap between them, the `skip=` step across the
   same peak line, and every `caption dropped (…)` line inside that gap with the defect it named;
   locked over a synthetic transcript+log pair in `tests/test_session_report.py`, proven red by
   neutralization. **The production choice this row held open is MADE, not refused** — the peak log
   gates on the stream PAIR (`not (_STDOUT_TTY and _STDERR_TTY)`) instead of on `_STDOUT_TTY` alone, so
   `live-stt 2> stt.log` keeps the live status line AND records the drop timeline; `live-smoke.md`
   item 2 and the README read that one-redirect form.
   **(b) LANDED** — one live session run as `live-stt 2>>stt-0924.log` reproduced a nonzero `drop=`
   with the log kept: 158 `backlog peak:` lines, the last `q=2.00s drop=1949 skip=1`, not yet
   analyzed. Evidence, all gitignored: `stt-0924.log`, its `script` typescript `screen-0924.log` +
   `screen-0924.timing.log`, transcript `transcripts/2026-09-24T14-18-40.txt`, and an out-of-process
   sampler trace (NPU busy, per-thread schedstat, PSI, power profile) with its analysis plan in
   `.scratch/live-monitor/NOTES.md`.
   **(c) mechanism NAMED — the contract this row closes on.** `_vac_segments` awaits every decode,
   update and final alike, inside the coroutine draining `audio_q`, so capture stacks at 1 s/s for the
   whole call. The 09-24 capture ran on battery under `low-power` (NPU at 950 MHz in 366 of 617 active
   samples); its 10 Hz `q=` meter reads 863 blocking spans, p50 0.60 / p90 0.92 / max 3.53 s, and
   those spans hold 1759 of the 1949 dropped blocks (188.9 blocks/s saturated ⇒ ~5.3 ms each, ~10 s
   of speech lost). Three shapes: **S** sustained 1.0-1.4 s updates against the 1 s cadence — each
   update consumes exactly `VAC_CHUNK_S`, so the backlog grows by `decode_s − 1` per update; **C** an
   update decode followed by the final decode with no drain between, 2.4-2.8 s; **R** a runaway
   444-character update decode, 3.53 s.
   **Fix — user ruling, which is also this row's approval of both grader moves.** A non-final update
   that falls due while `audio_q` holds more than `VAC_BACKLOG_S`=0.5 s of capture waits for the drain
   and fires on the first block that leaves the backlog ≤ 0.5 s, covering all pending audio, so the
   cadence stretches to the decode time instead of stacking; a queue without `queued_samples`
   (replay's) never waits, and the final decode never waits. `AUDIO_HEADROOM_S` 2 → **8** s, covering
   C, R and two back-to-back 448-token decodes at the measured low-power cost. `SCALE_LADDER` gains
   ×6 / ×8 / ×12, because the fix absorbs every rung up to ×4 and the non-vacuity locks must keep a
   dropping rung.
   **Accept:** a `tests/eval_backpressure.py` `live` arm — each traced clip at ×1.75 (the first rung
   at or above the live/trace ratios 1.42 and 1.64) with its middle update charged the measured
   3.53 s — drops on the unfixed code (stress_long 97, retention_probe 1323 blocks) and none after;
   the measured-cost arms keep `divergences == 0`; retention CER ≤ 0.0609 re-derived on the NPU; gate
   green; the closing commit names the live paths left unverified (L-004).
