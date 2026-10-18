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

3. **GPU-mode screen lives in scratch** — the GPU-mode decisions (`asr-pipeline.md` `## GPU mode`:
   760M device parity, the 1.5B / large-v3 / Qwen3-ASR / Cohere refusals for Japanese, large-v3's
   VAC-path English refusal, the 0.5 s cadence and term-biasing refusals) rest on `.scratch/gpu/`
   harnesses (`eval.py`, `kana_cer.py`, `vadseg.py`, `hot_eval.py`, `cohere_eval.py`,
   `cap_time.py`) and the `.scratch/quality-manifest.json` + `.scratch/gpu/cv300/` inputs, all
   gitignored. Tier `data`.
   **Accept:** one committed `tests/eval_*.py` reproduces the table's whisper rows (char + reading
   level CER, FLEURS-ja / retention / long-form / CV300, plus the English VAC arm) on a named device
   from committed manifests (`short_corpus.json`, `real_clips.json`, `en_clips.json`,
   `long_form.json`, `retention_probe.json`; the FLEURS legs = the 150 + 150 clips
   `.scratch/build_quality_manifest.py` draws from the first two indexes, `Random(20260916)`, and the
   CV300 leg = `.scratch/gpu/prep_cv.py`'s seed-20261001 sample of the CV8 test parquet) within
   ±0.002, plus the term-biasing arm, documented in
   `evidence-artifacts.md`; reading-level scoring's dependency rides `--with`, never the project
   deps.
