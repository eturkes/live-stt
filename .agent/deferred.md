# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). A `steered/maintain.md` Queue body names
the rows it funds, and the `auto/maintain.md` Queue body funds every row in rank order; closing a
row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **A quiet mic never opens the VAD** — silero v4 (threshold 0.5) is level-sensitive and whisper
   is not: the 10-09 `--save-audio` meeting (`transcripts/2026-10-09T08-30-12.{txt,wav}`, 44.5 min,
   speech window-RMS p90 −43.8 dBFS against −11..−24 on every pinned clip) published 59 lines /
   1,103 characters, while whisper-ja-760M over the raw audio in fixed 30 s windows, no VAD, hears
   12,319 (at ×4 gain 12,282). The VAD opened on 124 s of 2,672: char-weighted coverage of that
   oracle's segments 0.046, and 0.570 on 10-08 (p90 −35.3). Measured at the VAD, 10-09 / 10-08:
   threshold 0.3 0.151; silero v5 0.364 / 0.712, v6 0.129 / 0.682, TEN 0.048 / 0.496; fixed ×8
   0.820 / 0.894 but it re-segments every pinned corpus (gongitsune_01 67 → 44); AGC on the VAD's
   copy alone — gain = clamp(10^(−25/20) / p90 window RMS over the last 10 s, 1, 16), windows
   ≤ −70 dBFS kept out of that history, gain 1 until 2 s of it — 0.839 / 0.892, gain exactly 1.0 on
   every window of all 21 pinned clips. Kernel, funded (user ruling 10-09: AGC on the VAD input,
   −25 dBFS).
   **Accept:** `make_vad()` returns a VAD whose silero hears each window scaled by that rule while
   the caller's samples — whisper, ring, LID, recording — stay raw; a lock where a pinned clip
   scaled to the 10-09 level opens no segment unleveled and opens with the rule, red before; a lock
   that every pinned clip present keeps gain 1.0 on every window; the 10-09 and 10-08 WAVs replayed
   through the shipped NPU path each publish more characters than the HEAD replay, 10-09 at least
   5×, each arm's screen drops and silence-hallucination captions (`ご視聴ありがとうございました`
   family) counted in the commit body; retention CER 0.0532, long-form §01+§03 0.2153 and the
   whisper NPU golden unchanged; the gate green.

2. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and M14's `_respawn` arm have never met a mic (`_probe` has); standing
   debt = latency feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.

3. **Reconcile the human-facing doc set** — `.agent/spec.md` `Artifacts` and `orientation.md` call
   `README.md` the only human-facing doc, while `human-docs.md`, L-021's owner, names `README.md`,
   `models/README.md` and the CLI strings ⇒ an agent reading the first two can write
   `models/README.md` in agent register. Docs tier.
   **Accept:** `rg -n --hidden 'human-facing' .agent/spec.md .claude/rules/` → every hit names the
   same surface set as `human-docs.md` or points at it; the gate stays green.

4. **Screen each segment inside a released piece** — one trim can release several whisper segments
   as one piece, and the screen drops that piece whole, so a loop segment takes its clean
   neighbours with it (tester-1 trace: `繰り返し`×12 + one clean sentence, both dropped). Kernel.
   **Accept:** a piece is screened per released segment (segment counts from the trimming decode,
   never hypothesis text), the surviving segments publish as one line, a lock shows a loop segment
   dropped alone with its clean neighbour published, red on the unfixed worker; the gate stays
   green.

5. **Session report reads two-way transcripts** — a transcript records no per-line decode language,
   so `session_report.py` re-screens every source line under one `--source-lang`: an English line a
   two-way session published reads as a latin `declined` caption (reviewer-6, authored mixed
   transcript: `latin_drops=1`). Data tier.
   **Accept:** a two-way session's English source lines are never reported as screen declines — the
   transcript carries what the report needs, or the report takes a flag that skips the latin rule
   for two-way sessions; a lock over an authored mixed transcript, red before; the gate stays green.

6. **Re-measure the fresh-session baseline** — `assurance-posture.md` states a fresh MAIN session
   opens at 47K over 89 KB attached (global `CLAUDE.md` 27.0 KB, `.agent/spec.md` 14.2 KB, three
   bare rules files 28.7 KB); `wc -c` now reads 98.3 KB (28.0 / 21.9 / 28.9 KB), so the literal the
   teammate-budget paragraph quotes no longer describes a fresh session. Docs tier.
   **Accept:** a fresh session on a clean tree, measured with `context-gauge`, plus `wc -c` over
   the attached set → the paragraph carries those numbers, or drops the literal for the recipe;
   the gate stays green.
