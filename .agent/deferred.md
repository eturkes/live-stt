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

2. **Speed the pytest gate step** — the gate's slowest step: 48.7 s of ≈ 96 s sequential on this
   host (`gate.py --only` per step), the suite running serially; the gate's slowest-tests report
   leads with `test_lid_census.py::test_bounded_views_match_regeneration` at 10.4 s, then replay
   goldens + backpressure arms at 1.4-2.0 s each. Kernel tier: a speedup that drops a check leaves
   no downstream re-check.
   **Accept:** the step's wall-clock on this host before → after in the commit body (`gate.py`
   prints both), the same tests passing, and every `test_gate.py` seed still reddening its step
   (`gate FAILED: <step>`); the full gate stays green.

3. **Speed the secrets gate step** — 36.5 s, the gate's second-slowest step: `scan_files()` walks
   the filesystem, so gitignored live-session logs at the repo root (`screen-*.log`, 16.5 MB of the
   18.5 MB scanned) dominate, though no commit can carry them. Kernel tier.
   **Accept:** the step's wall-clock before → after in the commit body; the scan still reads every
   file a commit can carry (tracked + untracked-unignored); the seeded AWS-shaped key still reddens
   (`gate FAILED: secrets`) and `test_secret_scan_reaches_the_real_tree` still pins a production, a
   nested and a dotdir file; the full gate stays green.

4. **Screen each segment of the speech-end remainder** — a trim-released piece is screened per
   segment, but the remainder `finalize()` publishes at speech end — every utterance shorter than
   the trim threshold, plus each long one's last piece — is still screened whole, so a loop in the
   final decode's last segment (where a runaway ends, at the token cap) takes the clean segments
   before it. `finish()` holds no segments today. Kernel.
   **Accept:** the remainder splits at the final decode's segment counts (the decode `finish()`
   releases; never hypothesis text) and screens like a trim piece — per segment, then the surviving
   line whole; a decode whose segments do not partition its text screens whole; a lock with an
   untrimmed utterance `[clean, loop]` publishes the clean segment, red on the unfixed worker; the
   gate stays green.
