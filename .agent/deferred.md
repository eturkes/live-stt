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

2. **Speed the secrets gate step** — 36.5 s, the gate's second-slowest step: `scan_files()` walks
   the filesystem, so gitignored live-session logs at the repo root (`screen-*.log`, 16.5 MB of the
   18.5 MB scanned) dominate, though no commit can carry them. Kernel tier.
   **Accept:** the step's wall-clock before → after in the commit body; the scan still reads every
   file a commit can carry (tracked + untracked-unignored); the seeded AWS-shaped key still reddens
   (`gate FAILED: secrets`) and `test_secret_scan_reaches_the_real_tree` still pins a production, a
   nested and a dotdir file; the full gate stays green.

3. **Screen each segment of the speech-end remainder** — a trim-released piece is screened per
   segment, but the remainder `finalize()` publishes at speech end — every utterance shorter than
   the trim threshold, plus each long one's last piece — is still screened whole, so a loop in the
   final decode's last segment (where a runaway ends, at the token cap) takes the clean segments
   before it. `finish()` holds no segments today. Kernel.
   **Accept:** the remainder splits at the final decode's segment counts (the decode `finish()`
   releases; never hypothesis text) and screens like a trim piece — per segment, then the surviving
   line whole; a decode whose segments do not partition its text screens whole; a lock with an
   untrimmed utterance `[clean, loop]` publishes the clean segment, red on the unfixed worker; the
   gate stays green.

4. **Run the pre-commit hook's suite in parallel** — the gate's pytest step runs 6 pytest-xdist
   workers (≈23 s), while the D-007 hook still runs `uv run pytest -q` serially (≈39 s), on the
   critical path of every commit. Data tier: the gate re-checks every commit.
   **Accept:** the hook stays one line and runs the suite on the gate's worker count; its
   wall-clock before → after in the commit body; a seeded failing test still blocks a commit
   (hook exit ≠ 0); the same tests pass; the gate stays green.
