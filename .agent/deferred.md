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
   **Still owed, agent-side: (c)** either the mechanism is named and reproduced as a
   `tests/eval_backpressure.py` arm proven RED against today's code and fixed green with retention
   CER ≤ 0.0609 re-derived, or the row records a refusal naming the measured headroom shortfall and
   what the user loses. A capture that cannot decide (c) sends the row back to the mic (L-004).
