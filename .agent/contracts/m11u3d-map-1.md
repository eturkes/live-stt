# map-m11u3d-1 — VAC production arms: surface map

Fill `answer` + `evidence` per row. Grade: `python3 .scratch/check_map.py .scratch/agents/map-m11u3d-1.md`.
`evidence` = `file:line`, a command + its real output, or a commit SHA. Never a claim without a pointer.

Unit under map: **M11.3d** — wire the arm x corpus matrix onto production replay and implement every
`measured` registry selector in `tests/d016_claims.json`. Kernel already shipped by M11.3b
(`tests/eval_vac.py`): schema, `arm_id/corpus_id` row identity, VAC contract fingerprint, resume,
content-addressed detail install, atomic aggregate, CLI. This unit adds arms, corpora and metrics.

Registry facts (already derived by MAIN, do not re-derive): 40 rows, 31 `disposition=measured`.
Distinct `arms.<id>` tokens = 10: `gpu_prompt`, `lf_vac_gpu`, `long_form_vac_npu`,
`long_form_vad_k2v2`, `ret_vac_gpu`, `ret_vad_gpu`, `ret_vad_gpu_prev`, `retention_vac_npu`,
`retention_vad_k2v2`, `vad_gpu`. Distinct `derived.<key>` tokens = 7: `clips_per_corpus`,
`prev_text_paused_cer_delta`, `stream_lag_median_s_range`, `term_list_paused_cer_delta`,
`vac_npu_realtime_headroom_range`, `vac_npu_rtf_range`, `vac_vs_k2v2_decode_cost_ratio`.
Corpora = `long_form`, `retention`.

## M1 — Arm census

### M1
Question: For each of the 10 `arms.<id>` tokens, report one line as
`id | corpus | policy (vad|vac|stream|vad_prev) | device | engine | conditioning | legacy artifact | metrics that artifact stores`.
Then rule: is the token's identity `(corpus x policy x device)` or `(policy x device)`?
- answer: unknown
- evidence: unknown

### M2
Question: Manifest shape. The kernel's row identity is `arm_id/corpus_id`
(`tests/eval_vac.py:107-118,255-268`) and `_key_path(manifest, key_path, corpus_id)`
(`tests/eval_vac.py:211-218`) already takes a corpus. Enumerate every manifest shape under which all
31 `measured` key paths resolve. Per shape: the published dict layout, the code consequence in
`build_vac_manifest`, and whether it forces an edit to `evaluator_key_path` in
`tests/d016_claims.json`. Recommend one and say why.
- answer: unknown
- evidence: unknown

### M3
Question: For each of the 7 `derived.<key>` tokens, give the exact formula, its input key paths
inside the same manifest, the claim value it must reproduce (from `claim_value` in the registry),
and whether the inputs exist in the arm set M1 enumerates.
- answer: unknown
- evidence: unknown

### M4
Question: Corpus definitions. For `long_form` and `retention`: committed manifest file, WAV path,
reference-text source, expected `reference_chars`, expected `duration_s`, and whether
`tests/long_form.json` / `tests/retention_probe.json` already carry each field or it must be derived.
Name the loader the evaluator should call.
- answer: unknown
- evidence: unknown

### M5
Question: Production-path feasibility per arm. Which of M1's arms are reachable today through
production `replay.replay_recognizer` / `live_stt.load_recognizer` with no copied scratch policy?
Rule specifically on `gpu_prompt` and `ret_vad_gpu_prev`: D-016 deleted `CONTEXT_PREV_CHARS` and the
carry buffer from production, so state whether prev-text conditioning still has any production seam,
and if not, what evaluator-side seam supplies it without reintroducing deleted production code.
- answer: unknown
- evidence: unknown

### M6
Question: `term_list_paused_cer_delta` (registry row C31, arm `vad_hotwords`) has no `arms.<id>` key
path of its own. Name its two input arms, where their values come from, and whether a hotwords arm
must therefore exist in the matrix. Report what `WhisperEngine.set_hotwords` does per device
(`live_stt.py`) and which device an arm needs to carry hotwords at all.
- answer: unknown
- evidence: unknown

### M7
Question: Lag metric. Give the exact per-character lag definition the legacy producers compute, the
inputs a run must retain to compute it, and whether production replay exposes them today. Anchor on
`replay.py`'s `on_segment` hook and on the legacy producer that computes lag. State what the
evaluator must capture per update that a plain `replay_wav` call currently discards.
- answer: unknown
- evidence: unknown

### M8
Question: k2v2 comparator arm. How does `.scratch/policy_baseline.py` run the k2v2 VAD arm, what is
the production seam that replaces it, and what differs between that call and the whisper arm's call
(engine load, device, chunking, hotwords)? Name every production symbol the comparator arm needs.
- answer: unknown
- evidence: unknown

### M9
Question: `rtf` and `decode_cost_ratio`. Where does each come from, does it belong under
`measurements` (timing, `excluded_from_deterministic_equality=true`) or `deterministic`, and how can
a synthetic no-hardware run make `measurements.arms.*.rtf` and `derived.vac_vs_k2v2_decode_cost_ratio`
resolve at all given that timing is not deterministic? Anchor on the house rule in
`tests/eval_models.py:1494-1503`.
- answer: unknown
- evidence: unknown

### M10
Question: Synthetic-corpus feasibility for the FULL matrix with no model, no NPU, no OpenVINO, no
sherpa weights. The whisper arms have the M11.2 fake-pipeline seam
(`tests/test_shipped_path.py:51-98`). Report the equivalent seam for a k2v2/sherpa arm: what object
`load_recognizer("k2v2")` returns, what `worker` calls on it, and whether a fake is injectable the
same way. Name any hard blocker.
- answer: unknown
- evidence: unknown

### M11
Question: Bottom-up size estimate for M11.3d in shipped lines, split evaluator delta / test delta /
registry-or-other delta, each sized against a named committed analog with its `wc -l`. State the
credible range.
- answer: unknown
- evidence: unknown
