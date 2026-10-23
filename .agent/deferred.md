# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). A `steered/maintain.md` Queue body names
the rows it funds, and the `auto/maintain.md` Queue body funds every row in rank order; closing a
row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Live-mic validation pass** — user-only (L-004): M14's `_respawn` arm, the degrade marker and
   the SIGHUP shutdown have never met a mic (`_probe` has); standing debt = `live-smoke.md` items 1,
   3, 4 (latency feel), 7 (Ctrl+C mid-decode), 8 (`-o`), 9 (quiet far speech + empty room), 10 (GPU
   mode) and a multi-hour soak.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.

2. **Re-measure the fresh-session baseline** — `assurance-posture.md` states a fresh MAIN session
   opens at ≈51K over 109.1 KB attached, `CLAUDE.local.md` 5.0 KB among it; that file is gone, its
   OpenVINO memo now in the user-level instruction file, and the template refresh grew `CLAUDE.md`,
   so the literal no longer describes the attached set its own paragraph says to re-measure
   whenever it moves. Docs tier.
   **Accept:** the paragraph's recipe on a clean tree at the repo root (`claude -p "Reply with the
   single word OK." --output-format stream-json --verbose` → the assistant `usage` sum, plus `wc -c`
   over the attached set) → the paragraph carries those numbers and names only files that exist;
   the gate stays green.
