# map-m11u3-2 — `.scratch/` D-016 evidence inventory (per-source rows)

One row = one lookup, answerable on its own. Replace every `unknown`. Flush (write this whole file) every 4 rows.
Section C rows are answerable by grepping `.scratch/*.json` DIRECTLY — do not wait for A or B.

## A. Producer scripts — one row each

For each: what it writes (exact output paths it constructs), which evidence layer it belongs to, and whether it
still imports at HEAD. Import check: `cd /run/host/home/eturkes/Projects/live-stt && UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python -c "import ast,sys; ast.parse(open('.scratch/X.py').read())"` for parse, plus a real
`python -c "import importlib.util,sys; ..."` load attempt with `PYTHONPATH` cleared. Report the exact error if it fails.

Layers (keep separate — the port must not merge them):
`L1` = VAD/VAC policy runs (the shipped comparison) · `L2` = prompt-conditioning matrices (prev-text carry + oracle hotwords) ·
`L3` = earlier whole-file Whisper/Kotoba device probes (superseded harness behaviour the port must NOT canonize).

| ID | Script | Writes (exact paths) | Layer | Imports at HEAD? (error if not) | What it computes, in 2 lines |
|---|---|---|---|---|---|
| A1 | `.scratch/streaming_policy.py` | result: caller-supplied `--out` via `Path(args.out).write_text`; otherwise stdout only; OpenVINO cache `models/openvino/cache` | L1 | YES: AST parse OK; registered clean-env module load OK | Implements character LocalAgreement-2 over repeated Whisper decodes, segment-anchored/forced trims, and computationally-aware replay. Emits transcript/events, decode cost, caption/pipeline lag, CER, and config. |
| A2 | `.scratch/policy_vac.py` | result: caller-supplied `--out` via `Path(args.out).write_text`; otherwise stdout only; OpenVINO cache `models/openvino/cache` | L1 | NO: AST parse OK; import exits `SystemExit: 2`; exact argparse error: `the following arguments are required: device` | Drives `StreamingProcessor` from real silero speech-start/end: cadence decodes during speech, immediate tail flush at close. Scores transcript/CER plus lag, decode, utterance, and device/config fields. |
| A3 | `.scratch/policy_baseline.py` | result: caller-supplied `--out` via `Path(args.out).write_text`; otherwise stdout only; OpenVINO cache `models/openvino/cache` for Whisper | L1 | NO: AST parse OK; import exits `SystemExit: 2`; exact argparse error: `the following arguments are required: device` | Replays the shipped VAD-segment path with Whisper override or sherpa engine and models emit time as end+silence+decode. Scores CER and preserves segment/event/config evidence. |
| A4 | `.scratch/policy_corpus.py` | none; read-only helper resolves `tests/long_form.json` WAV path or fixed `spike/backends/cache/retention_probe.wav` | L1 | YES: AST parse OK; registered clean-env module load OK | Selects the pinned long_form/retention WAV and reference text. Centralizes the two-corpus policy-arm input contract. |
| A5 | `.scratch/compare_policies.py` | none; stdout arm table plus JSON array; reads `.scratch/policy_*.json` | L1 | YES: AST parse OK; registered clean-env module load OK | Recomputes per-character lag by interpolating each emission span, then summarizes CER, lag, RTF, updates, and forced trims across every policy artifact. |
| A6 | `.scratch/decode_cost_matrix.py` | none; stdout JSON only; OpenVINO cache fixed at `models/openvino/cache` | L1 | YES: AST parse OK; registered clean-env module load OK | Benchmarks warm 2/5/10/20/30 s decodes across WhisperPipeline/ASRPipeline and selected devices. Reports compile, median/min decode, and RTF or bounded errors. |
| A7 | `.scratch/long_form_whisper_eval.py` | none; stdout JSON only; optional caller-supplied `--cache-dir` passed to OpenVINO | L3 | NO: AST parse OK; import exits `SystemExit: 2`; exact argparse error: `the following arguments are required: model, device` | Earlier production-frontend/whole-file Whisper/Kotoba probe with VAD, prev-text/context, and whole-file arms. Scores CER/decode/RTF against committed long_form comparators; superseded as shipped-path evidence. |
| A8 | `.scratch/whisper_eval.py` | unknown | unknown | unknown | unknown |
| A9 | `.scratch/whisper_probe.py` | unknown | unknown | unknown | unknown |
| A10 | `.scratch/vad_real_compare.py` | unknown | unknown | unknown | unknown |
| A11 | `.scratch/prompt_hotword_matrix.py` | unknown | unknown | unknown | unknown |
| A12 | `.scratch/npu_asr_prompt_hotword_matrix.py` | unknown | unknown | unknown | unknown |

## B. Result artifacts — one row per family

Enumerate with `ls .scratch/*.json .scratch/*.md .scratch/*.log`. Group into families by filename prefix.
For each family: member list, producer (from A), and the exact top-level key path shape of one member
(`python -c "import json;d=json.load(open(...));print(json.dumps(d,ensure_ascii=False)[:800])"`).

| ID | Family (glob) | Members | Producer | Top-level key shape | Metric keys it carries |
|---|---|---|---|---|---|
| B1 | `policy_*.json` | 19: `policy_gpu_t05.json`, `policy_gpu_t08.json`, `policy_gpu_t12.json`, `policy_lf_vac_gpu.json`, `policy_lf_vac_hot.json`, `policy_lf_vad_gpu_prev.json`, `policy_lf_vad_hot.json`, `policy_long_form_vac_npu.json`, `policy_long_form_vad_k2v2.json`, `policy_ret_stream_gpu_t8.json`, `policy_ret_vac_gpu.json`, `policy_ret_vac_hot.json`, `policy_ret_vad_gpu.json`, `policy_ret_vad_gpu_prev.json`, `policy_ret_vad_hot.json`, `policy_retention_vac_npu.json`, `policy_retention_vad_k2v2.json`, `policy_vad_gpu.json`, `policy_vad_npu.json` | A1/A2/A3 (`streaming_policy.py`, `policy_vac.py`, `policy_baseline.py`); shell redirection names outputs | representative `policy_gpu_t05.json`: top-level `transcript, events, updates, trims, forced_trims, decode_median_s, decode_p90_s, decode_total_s, audio_s, wall_s, realtime_factor, min_chunk_s, buffer_trim_s, use_prompt, caption_lag_median_s, caption_lag_p90_s, caption_lag_max_s, lag_events, pipeline_delay_median_s, pipeline_delay_p90_s, S, D, I, N, cer, device, compile_s, model` | CER counts; decode/audio/wall/RTF; caption/pipeline lag; event/trim traces; config/device/model |
| B2 | `lf_*.json` | 4: `lf_seeded.json`, `lf_ship_noseed.json`, `lf_ship_seeded.json`, `lf_tail.json` | A7 `long_form_whisper_eval.py` stdout redirected by one-off runs | representative `lf_seeded.json`: top-level `model, device_arg, compile_s, prev_chars, k2v2_committed_baseline, parakeet_committed_baseline, vad_ctx, vad_ctx_tail`; each result = `S,D,I,N,cer,segments,decode_s,wall_s,learned_terms,hyp` | CER counts; decode/wall; segments; learned terms; hypotheses; committed comparator baselines |
| B3 | `long_form_*.json` | 4: `long_form_a_gpu.json`, `long_form_ctx_gpu.json`, `long_form_kotoba_gpu.json`, `long_form_whisper_gpu.json` | A7 `long_form_whisper_eval.py` stdout redirected by whole-file/device/context runs | representative `long_form_a_gpu.json`: top-level `model, device_arg, compile_s, prev_chars, k2v2_committed_baseline, parakeet_committed_baseline, vad, vad_prev, whole_file`; `vad*` = `S,D,I,N,cer,segments,decode_s,rtf,wall_s,hyp` | CER counts; segment/decode/RTF/wall; hypotheses; comparator baselines |
| B4 | `model_{a,b}_*_eval.json` | 4: `model_a_cpu_eval.json`, `model_a_npu_eval.json`, `model_b_cpu_eval.json`, `model_b_npu_eval.json` | A8 `whisper_eval.py` stdout redirected per model/device | representative `model_a_cpu_eval.json`: top-level `clips, compile_s, device_arg, micro, model, rows, warm_cv_short`; `micro={D,I,N,S,cer}`; `rows[]={D,I,N,S,audio_s,cer,clip_id,elapsed_s,hyp,metrics,ref}` | micro/per-row CER counts; audio/elapsed; OpenVINO perf metrics (TTFT/TPOT/generate/encode/decode); hypotheses |
| B5 | `decode_cost_matrix.json` | `decode_cost_matrix.json` | A6 `decode_cost_matrix.py` stdout redirected | top-level `model, genai, repeats, runs`; `runs[]={pipeline,device,compile_s,by_length_s}`; `by_length_s.<2.0|5.0|10.0|20.0|30.0>={median_s,min_s,rtf}` | compile/decode latency and RTF by pipeline/device/audio length |
| B6 | `policy_results.md` | `policy_results.md` | A5 `compare_policies.py` supplies the arm table/JSON; findings/limits were manually authored around it | Markdown metadata; `long_form` and `retention` arm tables; `F1–F6`; Limits | CER; median/p90/max lag; RTF; derived comparisons; limitations |
| B7 | `*.log` (`policy_arms.log`, `hotwords.log`, others) | 16: `baseline_arms.log`, `cloud_latency_r1.log`, `decisive.log`, `download_model_a.log`, `download_model_b.log`, `export_deps_install.log`, `export_model_b.log`, `export_model_b_stateless.log`, `export_model_b_with_past.log`, `export_model_b_with_past_stateful.log`, `hotwords.log`, `policy_arms.log`, `policy_sweep.log`, `preflight_m10_5e.log`, `sherpa_base.log`, `vac_npu.log` | D-016 logs: A1/A2/A3 via `run_*` shell wrappers; remaining logs belong to unrelated M10/cloud/export work | representative `policy_arms.log`: repeated `=== arm ===` blocks containing flat JSON fields, then `=== arm rc=N ===`; no single top-level JSON value | D-016 blocks carry decode/audio/wall/RTF, lag, CER counts, device/model/config; unrelated logs carry command diagnostics |
| B8 | anything under `.scratch/` matching none of the above that carries a D-016 number | `prompt_hotword_matrix.txt`, `npu_asr_prompt_hotword_matrix.txt` (partial raw evidence for C23); no other top-level unmatched result artifact carries a D-016 metric (`m10_block.md` numeric substring was unrelated) | A11/A12 | line-oriented `<device> <case> OK|ERROR <seconds> <result>` records; cases = base/empty/1-char/2-char × initial/hotword | NPU/CPU/GPU success/error behavior, elapsed seconds, transcripts/exceptions; only the summary MD preserves the 4–200-char probes |

## C. D-016 claim → artifact map

Every numeric/behavioural claim in D-016 (`.agent/memory.md`), seeded verbatim below. For each, find the artifact
that carries it. Fill: `artifact path` + `JSON key path or line number` + `exact stored value` + `status`.
`status` = `EXACT` (stored value equals the claim) · `ROUNDED` (claim is a rounding of the stored value — give both) ·
`DERIVED` (claim is computed from stored values — give the formula) · `MISSING` (no artifact carries it).
Search method: `/usr/bin/rg -n '<number-fragment>' .scratch/` plus `python -c` key walks. Report `MISSING` honestly.

| ID | D-016 claim (verbatim) | Artifact | Key path / line | Stored value | Status |
|---|---|---|---|---|---|
| C1 | long_form corpus N=1383 reference characters | `.scratch/policy_long_form_vac_npu.json` | top-level `N` | `1383` | EXACT |
| C2 | retention (pause-free 182 s) corpus N=1166 reference characters | `.scratch/policy_retention_vac_npu.json` | top-level `N`; `audio_s` | `1166`; `182.482` s | ROUNDED duration (`182.482` → 182 s); N EXACT |
| C3 | On pause-free audio silero yields 8 segments of 20-33 s | `.scratch/policy_results.md` | line 45 | `yields 8 segments of 20–33 s` | EXACT (hyphen normalized to en dash) |
| C4 | VAD-close lag on pause-free: median 15.5 s | `.scratch/policy_results.md` | line 32, `ret_vad_gpu` med column | `15.504` s | ROUNDED (`15.504` → 15.5) |
| C5 | VAD-close lag on pause-free: max 36.6 s | `.scratch/policy_results.md` | line 32, `ret_vad_gpu` max column | `36.587` s | ROUNDED (`36.587` → 36.6) |
| C6 | VAC lag on pause-free: median 2.5 s | `.scratch/policy_results.md` | line 28, `retention_vac_npu` med column | `2.483` s | ROUNDED (`2.483` → 2.5) |
| C7 | VAC lag on pause-free: max 8.1 s | `.scratch/policy_results.md` | line 28, `retention_vac_npu` max column | `8.129` s | ROUNDED (`8.129` → 8.1) |
| C8 | Pure streaming (no VAD controller) on paused audio: median 3.18-3.74 s | `.scratch/policy_results.md` | lines 17, 19, 21, `gpu_t05/t08/t12` med columns | `3.176`, `3.306`, `3.742` s | ROUNDED range (`3.176–3.742` → `3.18–3.74`) |
| C9 | Plain VAD on paused audio: median 2.43 s | `.scratch/policy_results.md` | line 19, `vad_gpu` med column | `2.430` s | EXACT (trailing zero omitted) |
| C10 | long_form CER 0.2538 (shipped k2v2+VAD, before) | `.scratch/policy_long_form_vad_k2v2.json` | top-level `cer` | `0.25379609544468545` | ROUNDED (`0.25379609544468545` → 0.2538) |
| C11 | long_form CER 0.2321 (whisper+VAC NPU, after) | `.scratch/policy_long_form_vac_npu.json` | top-level `cer` | `0.23210412147505424` | ROUNDED (`0.23210412147505424` → 0.2321) |
| C12 | retention CER 0.1587 (before) | `.scratch/policy_retention_vad_k2v2.json` | top-level `cer` | `0.15866209262435677` | ROUNDED (`0.15866209262435677` → 0.1587) |
| C13 | retention CER 0.0583 (after) — KNOWN SUSPECT: roadmap says no artifact carries it; `policy_retention_vac_npu.json` and `policy_results.md` retain I=12 / CER 0.0686106. Confirm or refute by exhaustive search of every `.scratch/` file including logs. | MISSING: exhaustive literal `0.0583` search found no producer result JSON/MD/log; only `.scratch/agents/` planning reports repeat the unsupported claim | searched all `.scratch/` plus explicit `.scratch/policy_*.json`, `policy_results.md`, and top-level `*.log`; positive control found `0.0686106346483705` | closest artifact: `.scratch/policy_retention_vac_npu.json` top-level `I=12`, `cer=0.0686106346483705` | MISSING (confirmed; no `0.0583` artifact) |
| C14 | lag median 14.239 s (before) | `.scratch/policy_results.md` | line 34, `retention_vad_k2v2` med column | `14.239` s | EXACT |
| C15 | lag median 2.483 s (after) | `.scratch/policy_results.md` | line 28, `retention_vac_npu` med column | `2.483` s | EXACT |
| C16 | RTF 0.041 (k2v2) | `.scratch/policy_results.md` | line 22, `long_form_vad_k2v2` rtf column | `0.041` | EXACT |
| C17 | RTF 0.48-0.60 (whisper VAC) | `.scratch/policy_results.md` | lines 16 and 28, NPU VAC rtf columns | long_form `0.481`; retention `0.595` | ROUNDED range (`0.481–0.595` → `0.48–0.60`) |
| C18 | 1.7-2.1x real-time headroom | `.scratch/policy_results.md` | line 72 | `NPU RTF 0.481 / 0.595 = 1.7–2.1×` | DERIVED (`1/0.595=1.681`; `1/0.481=2.079`; rounded to `1.7–2.1×`) |
| C19 | NPU unconditioned VAC CER long_form 0.2321 | `.scratch/policy_long_form_vac_npu.json` | top-level `cer` | `0.23210412147505424` | ROUNDED (`0.23210412147505424` → 0.2321) |
| C20 | NPU unconditioned VAC CER retention 0.0686 | `.scratch/policy_retention_vac_npu.json` | top-level `cer` | `0.0686106346483705` | ROUNDED (`0.0686106346483705` → 0.0686) |
| C21 | GPU unconditioned VAC CER long_form 0.2292 | `.scratch/policy_lf_vac_gpu.json` | top-level `cer` | `0.2292118582791034` | ROUNDED (`0.2292118582791034` → 0.2292) |
| C22 | GPU unconditioned VAC CER retention 0.0695 | `.scratch/policy_ret_vac_gpu.json` | top-level `cer` | `0.06946826758147513` | ROUNDED (`0.06946826758147513` → 0.0695) |
| C23 | NPU `initial_prompt`/`hotwords` refusal: probed lengths 1/2/4/8/16/32/64/128/200 all raise, 0 passes | `.scratch/policy_results.md` | lines 73–77 | lengths `1, 2, 4, 8, 16, 32, 64, 128, 200` all raise; omitted/length-0 prompt passes | EXACT textual result; raw `.scratch/npu_asr_prompt_hotword_matrix.txt` preserves only omitted, empty, 1-, and 2-char probes |
| C24 | prev-text conditioning on pause-free: CER 1.8919 | `.scratch/policy_ret_vad_gpu_prev.json` | top-level `cer` | `1.8919382504288165` | ROUNDED (`1.8919382504288165` → 1.8919) |
| C25 | unconditioned comparison figure for C24: CER 0.1278 | `.scratch/policy_ret_vad_gpu.json` | top-level `cer` | `0.12778730703259006` | ROUNDED (`0.12778730703259006` → 0.1278) |
| C26 | prev-text conditioning: 2,126 insertions on 1,166 reference characters | `.scratch/policy_ret_vad_gpu_prev.json` | top-level `I`; `N` | `2126`; `1166` | EXACT |
| C27 | prev-text tail repeating one clause 7x | `.scratch/policy_results.md` | line 57 | `tail repeats 牛乳瓶をあしらったデザインのパック 7×` | EXACT textual summary; raw `.scratch/policy_ret_vad_gpu_prev.json` transcript contains 131 total occurrences, so the summarized tail-window boundary is not encoded |
| C28 | prior loop reproduction: streaming buffers CER 0.9328 | `.scratch/policy_arms.log` | line 21, `gpu_prompt.cer` | `0.9327548806941431` | ROUNDED (`0.9327548806941431` → 0.9328) |
| C29 | prior loop reproduction: Kotoba CER 0.8402 | `.scratch/long_form_kotoba_gpu.json` | `metrics.cer` | `0.8402024584237165` | ROUNDED (`0.8402024584237165` → 0.8402) |
| C30 | prev-text helps the paused case: 0.2408 -> 0.2054 | `.scratch/policy_vad_gpu.json`; `.scratch/policy_lf_vad_gpu_prev.json` | each top-level `cer` | `0.24078091106290672` → `0.20535068691250905` | ROUNDED pair (→ `0.2408` → `0.2054`) |
| C31 | bounded term list on the same slot: 0.2408 -> 0.1873 | `.scratch/policy_vad_gpu.json`; `.scratch/policy_lf_vad_hot.json` | each top-level `cer` | `0.24078091106290672` → `0.18727404193781635` | ROUNDED pair (→ `0.2408` → `0.1873`) |
| C32 | `emitted` append-only fix on retention: I 12 -> 0 | MISSING: no post-fix retention artifact; closest `.scratch/policy_retention_vac_npu.json` | top-level `I` in closest pre-fix artifact | pre-fix `12`; post-fix `0` absent from all policy JSON/MD/log artifacts | MISSING (before EXACT; after is commit-body-only) |
| C33 | `emitted` append-only fix on retention: CER 0.0686 -> 0.0583 | MISSING: no post-fix retention artifact; closest `.scratch/policy_retention_vac_npu.json` | top-level `cer` in closest pre-fix artifact | pre-fix `0.0686106346483705`; post-fix `0.0583` absent from every producer JSON/MD/log | MISSING (before ROUNDED to 0.0686; after commit-body-only) |
| C34 | Any additional numeric claim in D-016 not listed above (enumerate; add rows C35+) | `.agent/memory.md` D-016 inventory audit | C35–C41 below | 7 additional numeric claims: 4:48 duration; ~10× decode; one forfeited feature; third reproduction; 0 unseeded terms; one clip/corpus; mode 400 | EXACT enumeration; each mapped below |
| C35 | long_form is one 4:48 paused narration | `.scratch/policy_results.md`; `.scratch/policy_long_form_vac_npu.json` | line 3; top-level `audio_s` | `4:48`; `288.521` s | ROUNDED duration (`288.521` s → 4:48) |
| C36 | Whisper+VAC decode cost is ~10× the shipped k2v2+VAD path | `.scratch/policy_results.md` | lines 16/22 and 28/34, rtf columns | long_form `0.481/0.041=11.73×`; retention `0.595/0.058=10.26×` | DERIVED; `~10×` is coarse rounding of `10.26–11.73×` |
| C37 | NPU forfeits exactly one feature: ASR text conditioning | `.scratch/policy_results.md` | lines 70 and 73–75 | one capability class; both `initial_prompt` and `hotwords` assert unsupported | EXACT behavioral count |
| C38 | prev-text collapse is the third independent prompt-loop reproduction | `.scratch/policy_results.md` | lines 57–58 | current retention collapse plus prior streaming `0.9328` and Kotoba `0.8402` = 3 reproductions | DERIVED count (`1+2=3`) |
| C39 | Passive learning has contributed 0 terms unseeded | `.scratch/policy_results.md` | line 87 | `0 terms unseeded to date` | EXACT |
| C40 | Evidence uses one clip per corpus | `.scratch/policy_results.md` | line 98 | one long_form clip; one retention clip | EXACT |
| C41 | RAPL `energy_uj` is mode 400 | `.scratch/policy_results.md` | line 99 | mode `400`, owner `nobody:nogroup`; energy not measured | EXACT |

## D. Port hazards

| ID | Question | Answer |
|---|---|---|
| D1 | Which producers share code by copy-paste (same function defined in >1 script)? Name each duplicated function + its homes. | unknown |
| D2 | Which producers hardcode a path, device, or model that a committed evaluator must parameterize? Give `file:line`. | unknown |
| D3 | Which artifacts were written by a producer version that no longer exists on disk (i.e. the `.json` records a config the current script cannot reproduce)? Evidence = a config key in the JSON with no corresponding code path. | unknown |
| D4 | Do any two artifacts disagree on the same metric for the same arm? List each disagreement with both values and both paths. | unknown |

## Register

(none yet)
