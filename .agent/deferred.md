# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). A `steered/maintain.md` Queue body names
the rows it funds, and the `auto/maintain.md` Queue body funds every row in rank order; closing a
row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Anchor residuals re-commit published text** — the 10-08 replay (290 utterances, 408
   lines) re-commits 3-8 characters 6 times outside the doubled-trim shape, by three mechanisms: a
   decode re-spells the published end and the next reverts to the original spelling (replay SRC
   199 `やりやすくっていうていう`, live SRC 186; 216 `ちょっといょっとい`), whisper restores a head
   ahead of the published text (313 `そういうそういう`), the count fallback re-derives the record
   from a garbage decode (123 `入れるっていうていう`). Live shows the same family (SRC 6 `から。ら。`,
   316 `感じ正してる感じ`), unclassifiable without its commit stream. Measured candidate: keeping the
   published record on a count fallback wins 85 scenario-harness scripts and loses 0
   (`--scripts 30000 --seed 7`), yet on the replay's real decodes changes 9 utterances — 5 better,
   2 worse where the count had tracked a same-length re-spelling (`その時点前`, `いました。`), 2
   even. Kernel, funded (user ruling 10-08: attempt now).
   **Accept:** a boundary change whose re-run over the 10-08 replay's logged decodes (scratch
   trace, regenerable from the WAV) removes more re-commits than it adds, each changed utterance
   adjudicated in the commit body; locks from the replay's shapes red before; no scenario-harness
   script lost at seed 11, losses at `--scripts 30000 --seed 7` reported for a user ruling;
   retention CER and the long-form replay no worse than recorded; the gate green — or the attempt
   finds no such change and the row closes with the classes and measured trades in
   `asr-pipeline.md`.

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
