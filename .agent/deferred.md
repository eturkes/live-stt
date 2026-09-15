# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
`Deferred` there = the pointer + one title per row below, and that list is the spine. Rank = funding
order. Acceptance is written at deferral time while the evidence is fresh, and the funded row is that
unit's whole contract (`assurance-posture.md`). The `/goal` body the user pastes names the row it
funds; closing a row deletes its title from this file and from `.agent/spec.md`'s spine in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and BOTH M14 recovery arms have never met a mic; standing debt = latency
   feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence. **Accept:** the
   user runs `live-smoke.md` and reports; each item lands verified or defective.
2. **Probe the two open NPU constructor properties.** The fixed decode term (0.417 s, 65 % of a p50
   update) is what floors `VAC_CHUNK_S`, and shorter-encoder / speculative / KV-reuse are all closed
   at the genai source (`asr-pipeline.md`). Two reach this build unmeasured: `NPU_TURBO=true`
   (max frequency/bandwidth) and `NPUW_LLM_GENERATE_HINT="BEST_PERF"` (stateful NPUW defaults to
   `FAST_COMPILE`, trading run speed for compile speed). **Accept:** `tests/eval_latency.py`
   `decode_s` p50 on `retention_probe` re-derived per arm against the committed 0.645 s, adopting only
   a cut that holds retention CER ≤ 0.0609 — or a recorded refusal carrying both measured deltas.
   Expect tens of ms, not the whole fixed term; a null result closes the row.
3. **Prove EN rendering consistency on the raw stream.** The rotation-tax row's acceptance cited
   `eval_en_pairing.py` "distinct spellings 1/1/1" and no such metric exists there. `SessionContext`
   keeps ONE rendering per term by construction, so every check reading `renderings` is tautological:
   mutating post-pairing EN (`Gon` → `Gawn`/`Ghone`) leaves both the learned map and the M12.5
   verdict green while a one-spelling assertion reddens. Consistency is a property of the RAW EN
   stream, which is where `translation-leg.md`'s 9-of-9 figure was counted. **Accept:**
   `eval_en_pairing.py` grows a raw-stream metric counting distinct spellings of each learned term's
   rendering across `turns[*].en`, reported per run and locked by a test proven red under that exact
   mutation; the committed trace then re-derives one spelling per paired term, or the divergence is
   recorded as the real number.
4. **M10 candidate-screen remainder** — zipformer + SenseVoice lack current JA evidence,
   Moonshine-JA's license is unclear, ReazonSpeech-k2-v2 adds PyTorch/Transformers + remote custom
   model code. **Accept:** re-open only if the shipped path fails AND the added runtime surface buys
   a materially different hypothesis. Tournament record → `.agent/archive/m10-asr-tournament.md`.
