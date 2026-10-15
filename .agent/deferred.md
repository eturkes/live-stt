# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). A `steered/maintain.md` Queue body names
the rows it funds, and the `auto/maintain.md` Queue body funds every row in rank order; closing a
row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and M14's `_respawn` arm have never met a mic (`_probe` has); standing
   debt = latency feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.

2. **Reconcile the human-facing doc set** — `.agent/spec.md` `Artifacts` and `orientation.md` call
   `README.md` the only human-facing doc, while `human-docs.md`, L-021's owner, names `README.md`,
   `models/README.md` and the CLI strings ⇒ an agent reading the first two can write
   `models/README.md` in agent register. Docs tier.
   **Accept:** `rg -n --hidden 'human-facing' .agent/spec.md .claude/rules/` → every hit names the
   same surface set as `human-docs.md` or points at it; the gate stays green.

3. **Screen each segment inside a released piece** — one trim can release several whisper segments
   as one piece, and the screen drops that piece whole, so a loop segment takes its clean
   neighbours with it (tester-1 trace: `繰り返し`×12 + one clean sentence, both dropped). Kernel.
   **Accept:** a piece is screened per released segment (segment counts from the trimming decode,
   never hypothesis text), the surviving segments publish as one line, a lock shows a loop segment
   dropped alone with its clean neighbour published, red on the unfixed worker; the gate stays
   green.

4. **Session report reads two-way transcripts** — a transcript records no per-line decode language,
   so `session_report.py` re-screens every source line under one `--source-lang`: an English line a
   two-way session published reads as a latin `declined` caption (reviewer-6, authored mixed
   transcript: `latin_drops=1`). Data tier.
   **Accept:** a two-way session's English source lines are never reported as screen declines — the
   transcript carries what the report needs, or the report takes a flag that skips the latin rule
   for two-way sessions; a lock over an authored mixed transcript, red before; the gate stays green.

5. **Re-measure the fresh-session baseline** — `assurance-posture.md` states a fresh MAIN session
   opened at 47K over 89 KB attached (the user-level instruction file 27.0 KB, `CLAUDE.md` 14.4 KB,
   `.agent/spec.md` 14.2 KB, three bare rules files 28.7 KB); the template refresh moved project
   practice back into `CLAUDE.md`, and `wc -c` now reads 108.9 KB (user-level 23.6, `CLAUDE.md`
   29.2, `CLAUDE.local.md` 5.0, spec 20.4, bare rules 30.7 KB), so the literal the teammate-budget
   paragraph quotes no longer describes a fresh session. Docs tier.
   **Accept:** a fresh session on a clean tree, its occupancy read as the latest assistant `usage`
   sum, plus `wc -c` over the attached set → the paragraph carries those numbers, or drops the
   literal for the recipe; the gate stays green.

6. **Name the shipped recogniser in two evaluator docstrings** — `tests/eval_retention.py` and
   `tests/build_caption_trace.py` describe the path they replay as "whisper large-v3-turbo int8",
   while `load_recognizer("whisper")` decodes Japanese on whisper-ja-760M (`live_stt.py`
   `ENGINE_DIRS["whisper"]`, D-016) ⇒ a reader takes their numbers for turbo's. Docs tier.
   **Accept:** both docstrings name whisper-ja-760M as the Japanese recogniser they replay;
   `rg -n 'large-v3-turbo int8' tests/eval_retention.py tests/build_caption_trace.py` → 0 hits; the
   gate stays green.

7. **Speed the pytest gate step** — the gate's slowest step: 48.7 s of ≈ 96 s sequential on this
   host (`gate.py --only` per step), the suite running serially; the gate's slowest-tests report
   leads with `test_lid_census.py::test_bounded_views_match_regeneration` at 10.4 s, then replay
   goldens + backpressure arms at 1.4-2.0 s each. Kernel tier: a speedup that drops a check leaves
   no downstream re-check.
   **Accept:** the step's wall-clock on this host before → after in the commit body (`gate.py`
   prints both), the same tests passing, and every `test_gate.py` seed still reddening its step
   (`gate FAILED: <step>`); the full gate stays green.

8. **Speed the secrets gate step** — 36.5 s, the gate's second-slowest step: `scan_files()` walks
   the filesystem, so gitignored live-session logs at the repo root (`screen-*.log`, 16.5 MB of the
   18.5 MB scanned) dominate, though no commit can carry them. Kernel tier.
   **Accept:** the step's wall-clock before → after in the commit body; the scan still reads every
   file a commit can carry (tracked + untracked-unignored); the seeded AWS-shaped key still reddens
   (`gate FAILED: secrets`) and `test_secret_scan_reaches_the_real_tree` still pins a production, a
   nested and a dotdir file; the full gate stays green.
