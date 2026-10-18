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

2. **Recogniser term channel is dead plumbing** — no device takes the term list since the GPU
   measurement (`asr-pipeline.md`: reading-level CER 0.0596 → 0.0791 with a user `--context` on
   whisper-ja-760M), so `ASR_HOTWORDS_DEVICES`, `WhisperEngine.set_hotwords`/`hotwords`, the
   worker's `biased` handoff and `SessionContext.asr_hotwords` + the `prompted` exclusion carry no
   live path. Tier `docs` (behaviour-preserving: `prompted` is always empty today).
   **Accept:** either the channel is deleted — suite green, `translator_brief` output byte-identical
   over `tests/caption_trace.json`, `eval_term_census.py` and `eval_en_pairing.py` verdicts unchanged
   — or a recogniser is measured where term biasing lowers reading-level CER on the long-form
   one-session arm, and the gate admits that device for it.
