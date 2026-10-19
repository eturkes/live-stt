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

2. **Startup locks time out under host load** — `tests/test_audio_startup.py` drives real child
   processes inside fixed wall-clock deadlines (5 s marker polls, 8 s `communicate`). With the box
   at load ≈ 20 on 8 cores (other sessions' suites beside a GPU screen run), the D-007 hook failed
   `test_the_session_child_resolves_openvino_from_the_venv_alone`,
   `test_second_launch_is_refused_before_audio_initialization` and
   `test_interrupt_during_audio_import_returns_without_waiting_for_its_timeout`, then
   `test_supervisor_forwards_terminal_signals_and_preserves_the_drain[ctrl_c]` on the retry, while
   the file passes alone (14 passed) and on a quiet box. Reproducer that loads no other core:
   `taskset -c 0 uv run --no-sync python -m pytest -q -n 6 -p no:randomly tests/test_audio_startup.py`
   (one CPU under the gate's worker count) failed 11, then 1, of 14. Tier `data`.
   **Accept:** that reproducer passes 5 consecutive runs, and every changed deadline still reddens
   its case on the hang that case exists to catch (L-022), the original deadline's firing recorded.
