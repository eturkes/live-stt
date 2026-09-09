# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
`Deferred` there keeps the pointer plus whatever blocks the spine. Rank = funding order. Acceptance
is written at deferral time while the evidence is fresh, and the funded row is that unit's whole
contract (`assurance-posture.md`). The `/goal` body the user pastes names the row it funds; a row
dies in the commit that closes it.

1. **Cut the EN-leg thread-rotation tax.** A glossary change opens a fresh codex thread inline, and
   the committed 215-caption trace pays it on 39 turns: 3.900 s p50 against 2.085 s steady, 18 % of
   the stream. **Accept:** rotations below 15 per 215 captions with rendering consistency unchanged
   (`eval_en_pairing.py` distinct spellings 1/1/1) and turn p90 under 3.5 s — or a recorded refusal
   naming the glossary-staleness cost. Budget + producer → `asr-pipeline.md`, `tests/eval_latency.py`.
2. **Rule on `CAPTION_REPEAT_MAX_CHARS`=40's known false negatives.** Three live captions repeat a
   phrase exactly 4× and survive at 36 / 30 / 28 chars; the largest repetition a SPEAKER produced is
   20, so the usable range is 21..36. **Accept:** re-derive the live population at the candidate
   threshold, adjudicate every newly dropped caption as loop or speech, keep a ≥1.5× margin over the
   largest genuine repetition — or record the refusal with that margin as the reason.
   `tests/test_translator.py`'s corpus + boundary locks and `test_shipped_path.py`'s pass-list move
   with it. Evidence + the 1073-caption sweep → `asr-pipeline.md`; history → `polish-register.md`
   P-022.
3. **Maintenance + security pass** (L-018 recipe, `packaging-deps.md`). `sherpa-onnx` 1.13.4 →
   1.13.7, `sounddevice` 0.5.5 → 0.5.6, `ruff` 0.15.21 → 0.16.6; `numpy`, `openvino`,
   `openvino-genai`, `pytest` current. **Accept:** pip-audit clean, full gate green, codex leg
   re-verified against a real app-server, `uv.lock` committed.
4. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and BOTH M14 recovery arms have never met a mic; standing debt = latency
   feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence. **Accept:** the
   user runs `live-smoke.md` and reports; each item lands verified or defective.
5. **Probe the two open NPU constructor properties.** The fixed decode term (0.417 s, 65 % of a p50
   update) is what floors `VAC_CHUNK_S`, and shorter-encoder / speculative / KV-reuse are all closed
   at the genai source (`asr-pipeline.md`). Two reach this build unmeasured: `NPU_TURBO=true`
   (max frequency/bandwidth) and `NPUW_LLM_GENERATE_HINT="BEST_PERF"` (stateful NPUW defaults to
   `FAST_COMPILE`, trading run speed for compile speed). **Accept:** `tests/eval_latency.py`
   `decode_s` p50 on `retention_probe` re-derived per arm against the committed 0.645 s, adopting only
   a cut that holds retention CER ≤ 0.0609 — or a recorded refusal carrying both measured deltas.
   Expect tens of ms, not the whole fixed term; a null result closes the row.
6. **Parameterize source language** (T2.2) — Japanese-only by design. **Accept:** re-open only if the
   use-case expands.
7. **M10 candidate-screen remainder** — zipformer + SenseVoice lack current JA evidence,
   Moonshine-JA's license is unclear, ReazonSpeech-k2-v2 adds PyTorch/Transformers + remote custom
   model code. **Accept:** re-open only if the shipped path fails AND the added runtime surface buys
   a materially different hypothesis. Tournament record → `.agent/archive/m10-asr-tournament.md`.
