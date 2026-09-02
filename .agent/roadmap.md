# Roadmap — live-stt

Canonical plan + status. Pick the lowest-numbered OPEN unit; restate its acceptance before coding.
Trajectory: single-file personal tool, simplicity over completeness — no frameworks, no config, no
premature abstraction. Milestone states: UNPLANNED · IN-PROGRESS · IMPLEMENTED · REVIEWED · PARKED.
Unit states: OPEN · BLOCKED · DONE.

**Assurance posture, set by user ruling 2026-09-02 (see `## Decisions pending from user`).** This is a
personal tool, not an industrial product. Verification = the fast test suite + replay goldens + a
plain CER script run on demand. Provenance machinery — contract fingerprints, claim registries,
mutation matrices, per-unit acceptance contracts, review ledgers — was deleted, not serviced. Do not
reintroduce any of it. A unit closes when its acceptance holds under `python gate.py` and, where the
unit touches decode quality, a CER number the commit body records.

## Status

- Milestone: **M12 does the EN rendering learner hold up on real ASR output?** — **IN-PROGRESS**
  (M12.1 DONE, M12.2 OPEN). Opened by user decision on the P-012 register row, which outgrew polish.
  `observe_en` (D-015, shipped by P-002 in `16a842b`) keys a learned English spelling on the JA
  string the RECOGNISER produced, and every P-002 arm ran on clean Aozora text — the learner's best
  case. Live the key is a hypothesis, so two modes were unmeasured: **(a) dead pairing** — a
  mis-recognised name pins a correct spelling to a key that never recurs; **(b) split key** — the
  recogniser alternates JA forms of one name and each form carries its own rendering. **M12.1
  answered both on real NPU output and neither fires:** (b) is refuted (7 of 8 occurrences take one
  form) and (a) is unreachable on a 67-caption clip (the key recurs to caption 65; a 60-segment lease
  cannot expire). What the census found instead is a **third mode P-012 never named — the learner
  stabilises a WRONG key**: both terms it trusts all session are mis-recognitions, while the
  correctly-recognised protagonist ゴン sits below `_TERM_RUN`'s 3-character katakana floor and is
  invisible to it. M12.2 now measures whether that helps or harms the English.
- Previous milestone: **M11 production-qualify the shipped whisper+VAC+NPU path** — **IMPLEMENTED**
  (all units DONE). Both questions M11 opened are answered and both answers are green: the VAC
  branch does not drop audio in real time (M11.4, with a 1.25-1.75× reserve), and D-016's retention
  CER is real (M11.5, 0.0583 reproduced exactly). Judgment-review sessions were retired on
  2026-09-02 (L-032, D-012), so IMPLEMENTED is this project's terminal milestone state.
- **M11's apparatus was cut on 2026-09-02** — 21,038 lines deleted across 20 files. Gone: the
  tournament stack (`eval_models`, `eval_streaming`, `fetch_eval_*`, their tests and baselines), the
  D-016 claim registry + validator + fixtures, the VAC evaluator (`eval_vac`, never produced an
  artifact), the mutation matrix, `eval_translation.py`, and the AST-closure contract fingerprints
  with their three pinned migration clauses. `gate.py` went 7 steps → **6** (the `aggregate-only`
  step died with its evaluator). Suite went 466 tests / 48 s → **207 tests / 14.6 s**. The tool
  itself is untouched: no production file changed.
- Kept because it earns its place: `live_stt.py` + `streaming.py` + `replay.py` + `cer.py` +
  `gate.py` (2,108 L), the fast unit tests, `replay.py` + `tests/replay_goldens.json` (the "did the
  output change" harness, D-014), `tests/test_shipped_path.py`, and `tests/eval_cer.py` /
  `tests/eval_long_form.py` / `tests/eval_backpressure.py` as on-demand scripts.
- Current architecture (D-009 + D-016): mic → resample → 2 s `AudioQueue` → silero VAD → VAC
  controller (speech-start opens a LocalAgreement-2 buffer in `streaming.py`; every `VAC_CHUNK_S`=1 s
  re-decodes the open buffer and commits what two decodes agree on; speech-end flushes the tail) →
  whisper large-v3-turbo int8 on OpenVINO NPU → `CodexTranslator` (JA→EN via persistent
  `codex app-server`, D-011) → `emit_line`; partials render on the meter status line, one utterance =
  one numbered JA line = one translation turn. `--engine k2v2|parakeet` retains the M9 sherpa
  fallback (D-010). Degrades to JA-only when codex is absent/failing.
- Agent-covered: VAC buffer mechanics + the VAC loop over stubbed VAD/recogniser; `SessionContext`
  learning rules; the shipped CLI/routing surface; two-engine short goldens + CER stressors + 4:48
  narration **on the sherpa fallback only**, plus the shipped path's one accuracy gate (M11.5
  `eval_retention.py`, 182 s pause-free, CER 0.0583 on NPU); deterministic paced backpressure on
  BOTH branches —
  44.722 s at RTF 0.20 on the sherpa fallback, 44.722 s + 182.482 s at real per-update NPU cost on
  the shipped VAC path (M11.4); shutdown/stage-failure/translation-degradation mechanics.
- User-only debt: latency feel, `-o`, soak, sustained live cadence, Ctrl+C-mid-decode, and the whole
  VAC partial-caption cadence (`.agent/memory.md` § Smoke, L-004).

## Open (do these; lowest ID first)

**M12 sizing fact, measured from tree, then confirmed by the M12.1 run — read it before M12.2.**
`tests/caption_trace.json` records `hotwords_reachable: false`. `ASR_DEVICE = "NPU"` and
`ASR_HOTWORDS_DEVICES = frozenset({"GPU", "CPU"})`, so `WhisperEngine.set_hotwords` drops the list
and `live_stt.py:1272` computes `biased = frozenset()` on every segment of a default run. Three
consequences size this whole milestone. **The caption stream is arm-independent** — `SessionContext`
cannot reach the recogniser on the shipped device, so ONE NPU replay serves every arm and every
downstream arm is a CPU-only offline replay of that trace (the unlock M11.4 got from
`vac_decode_trace.json`, applied again). **D-015's anti-feedback half is inert** — no sighting is
ever prompted, so the lease never expires by prompting and degenerates to "expire 60 segments after
the last sighting". **The learner's blast radius is the translator brief alone**, never the decode,
so both failure modes are translation-quality questions and D-016's CER numbers cannot move.

- **M12.2 — Rule on the learner against that trace. [OPEN — needs M12.1: MET]** M12.1's census picks
  the **third** branch (標柱 reaches support in 7 captions, inside the 4-9 band): run P-012's arm
  matrix offline off `tests/caption_trace.json` (learner on/off, 3 sessions each, real
  `CodexTranslator`, reps interleaved across arms per L-026), report BOTH statistics, and mark the
  rendering-count arm UNDERPOWERED — do not report its null as a pass (P-002 already spent one null
  that way on 「走れメロス」).
  - **Two facts the branch text did not anticipate, both from the census, both binding on scope.**
    (1) **Neither named failure mode can fire on this clip.** Mode (b) split key is refuted — 7 of 8
    occurrences take one form. Mode (a) dead pairing needs a key that is trusted, paired, then
    abandoned; 標柱 recurs to caption 65 of 67, and with `CONTEXT_TERM_LEASE`=60 against 29 captions
    remaining after first trust, no lease can expire in a session this short. So the dead-pairing
    arm is underpowered too, and BOTH arms are now corpus-limited.
    (2) **The learner's key is a mis-recognition, and the name it most needed is invisible.** The
    real question this clip poses is not consistency but what consistency is bought on: whether
    briefing `標柱 = <spelling>` helps or harms EN when the underlying name is 兵十. That is a
    translation-quality question the arm matrix can still answer, and it is the one worth its cost.
  - corpus acquisition — 「ごん狐」 sections 二+, meaning new pinned audio + Kokoro alignment per
    section under L-017 — was conditional on the first branch and stays unfunded; note that it is now
    also what a POWERED dead-pairing arm would need. Injecting synthetic recognition noise into clean
    text is not a substitute: it measures the noise model, and realism is the whole question.

## Done (ID · outcome · decisions/lessons produced)

- **M12.1 — What does the shipped path actually give the learner as a key? [DONE] — a
  mis-recognition, and it hides the one name that matters.** `tests/build_caption_trace.py` →
  **`tests/caption_trace.json`**: 67 captions over the pinned 288.521 s 「ごん狐」 narration, hash-gated
  against `long_form.json` `build.wav_sha256` before decoding, real whisper/VAC on NPU.
  `tests/eval_term_census.py` derives the census from that file alone — no model, no accelerator, no
  audio, <1 s, so a fresh clone reruns it (the `eval_vac_lag.py` shape). It locates the term by
  ALIGNING reference to captions with the shipped scorer, never by searching for guessed spellings,
  which is why `cer.py` now exposes `alignment()` (the pairs `align()` already counted; one DP, one
  tie order, `align`'s vectors unchanged).
  **Census of 兵十 — 8 reference occurrences, 8 landed as candidates, 0 dropped.** Two forms:
  **標柱** in 7 captions [33, 34, 38, 44, 46, 58, 65] → REACHES `CONTEXT_TERM_SUPPORT`=3, and 標準
  once [48] → below support, so it is never keyed. Over the whole session the learner trusts
  **exactly 2 terms, and both are errors**: 標柱 (兵十, "marker post") first trusted at caption 38,
  鼻腔 (びく, the fish creel → "nasal cavity") at 48. Pairing openings — captions where exactly one
  trusted term is unpaired, which is `observe_en`'s JA-side gate — 標柱 n=5, 鼻腔 n=2, both ≥
  `CONTEXT_EN_SUPPORT`=2, so renderings WILL be acquired.
  **The control is the sharper finding.** ごん, the protagonist, occurs 16× and whisper recognises it
  correctly as ゴン in 15 of them — and the learner sees it in NONE, because `_TERM_RUN`'s katakana
  floor is 3 characters and ゴン is 2. Only the compound ゴギツネ matched, once. So on real ASR output
  the learner is briefing the translator about a phantom while the name a reader would actually
  notice never enters the picture. `--term <any>` censuses any other name.
  **Determinism, measured not assumed:** two full NPU replays produced byte-identical caption text
  and byte-identical VAD boundaries (67/67), which is what licenses M12.2 to replay this file offline
  instead of paying for its own accelerator run. Decode cost did NOT reproduce — 284.7 s vs 275.7 s,
  each run carrying one burst of 3-4 stalled captions → `polish.md` **P-014**, out of contract here.
  Suite 221 → **228 passed / 1 skipped / 22.7 s** (`tests/test_term_census.py`, 7 model-free locks on
  the two rules that decide the census: alignment-located occurrences, and a rendering below
  `_TERM_RUN`'s floor being invisible however often it recurs). Gate 6/6. Cost: 2 NPU replays ≈ 5 min
  each. `main=77% 184K/240K`; no teammates funded — the census is script-derivable.

- **M11.1 — Installable, packaged, gated default path [DONE].** `openvino`/`openvino-genai` promoted
  from an optional extra to hard dependencies (the extra bought no saving for anyone running the
  default engine, only a way to reach a broken one), so plain `uv sync` produces a runnable default;
  `streaming.py` added to the wheel `only-include` and the sdist `include`, every sdist pattern
  root-anchored; `pyproject.toml` description realigned onto Whisper/OpenVINO; **`gate.py`** +
  `tests/test_gate.py` created; `replay.py`'s stale "matches live-stt default" claim corrected. Two
  packaging traps are now locked and must not return (`.agent/memory.md` § File map). Pyright was
  red for four commits while being reported green — that is why the gate is an executable script.
  Limit: the in-container NPU needs `source ~/.local/app/intel-accel/env.sh` **and** a cleared
  `PYTHONPATH`; the host needs neither.
- **M11.2 — Shipped-path unit coverage [DONE].** `tests/test_shipped_path.py` (45 tests). **No
  production change** — every named surface proved testable as shipped. The unlock: `WhisperEngine.
  __init__` runs a function-local `import openvino_genai`, so `monkeypatch.setitem(sys.modules, …)`
  substitutes the binding at call time and the REAL engine class runs end to end. The fake also
  reproduces the binding's one production-relevant refusal — `hotwords` on NPU raises — so a
  regression that leaks the parameter fails loudly. Review found six real gaps, all fixed: the
  `sorted(ENGINE_DIRS)` help order, `generate`'s language/task pin checked only on the untimestamped
  call (VAC could lose its Japanese pin silently), `hotwords` omission proved only on the incapable
  device, a fake deriving output from sample count alone while every test fed zeros (so
  `np.zeros_like(samples)` in production survived), marker readiness proved for one sherpa engine
  rather than each, and the both-missing message untested. Limits: `--no-translate`'s JA-only degrade
  needs a live session; `--device`/`--list-devices` stay user-only; `run_session` imports
  `sounddevice` at entry, so its wiring of `args.context`/`args.asr_device`/`args.engine` is
  unreachable in-process.
- **M11.3c — Whisper/VAC replay golden [DONE].** `784bc35`+`b2328b0`. Explicit `matrix()` replacing
  `ENGINES × all_clips()` with `WHISPER_CLIPS = ["long"]`, skip-preserving merge, per-leaf drop
  reporting, unknown/duplicate-id preflight; `tests/replay_goldens.json` 24 → **25 rows**, the
  whisper row recording `device: "NPU"`; `_stale_device` (hardware-free, before every readiness
  probe) + `_not_ready` + a whisper-only `n` assertion within `START_TOL`; `check_device()` beside
  `check_models`. Clip is `long`, not the planned `greet`: measurement showed `greet` = 1 VAC process
  call / **0 committed chars** (speech-end flush only) vs `long` = 13 calls / 51 chars. The golden
  pins a real LocalAgreement-2 repetition artifact — `polish.md` P-009 owns ruling on it.
- **M11.3, M11.3a, M11.3b — shipped, then deleted by the 2026-09-02 scope cut.** M11.3 replaced
  whole-file byte hashing with an AST-closure ASR contract fingerprint; M11.3a built the 40-row D-016
  claim registry + omission validator; M11.3b built the VAC evaluator kernel. All three were
  provenance machinery for a personal tool and all three are gone. M11.3b never produced an artifact
  (`vac_baseline.json` was never written). The fingerprint's own record indicts it: three units in a
  row each had to hand-write a pinned migration clause to make a legitimate code change, and M11.3d
  was blocked outright because `replay.py` was hashed whole. Git holds the code.
- **M11.4 — Does the VAC path drop audio in real time? [DONE] — no, with a 1.25-1.75× reserve.**
  Three pieces, one commit. (1) `on_update` seam through `_vac_segments`/`worker`/`replay.py`: one
  call per `StreamingProcessor.process`, in order, carrying `(buffer_s, buffer_end_s,
  commit_audio_s, text, final, decode_s)` — five deterministic fields plus the one measured one, and
  it is what recovers the commit audio endpoint `live_stt.py` discarded at `commit, _ = await …`.
  (2) `tests/build_vac_trace.py` → `tests/vac_decode_trace.json`: real per-update NPU decode cost for
  both pause-free clips, plus the hypothesis behind each. Storing the hypotheses is the unlock —
  `StreamingProcessor` is a pure function of decode outputs and buffer lengths, so replaying them
  reproduces the measured commit/trim trajectory with no model. (3) `eval_backpressure.py` VAC arm +
  4 tests: a `decode_segments` trace recogniser, real silero, production cadence/trim, executor
  interception charging each decode its recorded cost, and `max_segment_depth == 0` in place of the
  legacy `0 < depth <= 8`. Five tests: the seam's exactly-once/in-order contract over the stubbed
  VAD, both clips drop-free, and the two guards below.
  **Result — 44.722 s stressor / 182.482 s retention, paced at 20 ms:** `dropped == 0`, `divergences
  == 0` (44/44 and 180/180 updates on-trajectory), `forced_trims == 0`, segment queue 0, audio-queue
  high-water **0.760 s / 1.060 s** of 2.000 s, longest contiguous decode **0.764 s / 1.006 s**, duty
  0.544 / 0.642, buffer capped at 11.248 s by the trim rule. Contiguous == max single decode in both,
  which proves one update fires per queue drain and updates never bunch.
  **Two guards make that non-vacuous**, both committed tests: the same replay at ×4 decode cost
  drops, and a series shifted by one update reports 44/44 divergences while still dropping nothing —
  so the divergence counter, not the drop counter, is what certifies each cost was charged to the
  buffer it was measured on. `SCALE_LADDER` reports the margin: dropping starts at ×1.5 (retention)
  and ×2.0 (stressor).
  **Honest limits.** Decode cost varies ~20 % run to run (max 0.949 → 0.764 s across two builds), so
  the committed trace is one sample and the ladder is the margin around it. The retention queue
  high-water drifts 0.84 → 1.06 s across the clip as mean decode rises 0.606 → 0.674 s; duty < 1 and
  the 11.248 s buffer cap bound it, but the trend is real. Lag was not measured — the derivation is
  parked as `polish.md` P-011. `README.md`'s "Capture and VAD feeding continue during decode" was
  false for the shipped branch and is corrected to what was measured. Suite 207 → **212 tests /
  14.8 s**; `main=94% 225K/240K`.
- **M11.5 — Is D-016's retention CER 0.0583 real? [DONE] — yes, exactly.** The number reproduced to
  four decimals from committed inputs, so nothing in D-016 is superseded. **CER 0.0583** (S=33, D=35,
  **I=0**, N=1166, hyp 1131 chars) over the 182.482 s pause-free probe on the shipped whisper/VAC
  path. Both claims that rested on it hold: the headline retention improvement (0.1587 → 0.0583) and
  the append-only-`emitted` fix's `I` 12 → 0.
  The fix for "unsourced" is a producing artifact, not another loose figure: **`tests/eval_retention.py`**
  (on-demand, never a gate step, ~4 min warm) hash-gates the probe WAV against `retention_probe.json`
  before decoding — so a published number is always bound to the pinned input, not to whatever sits
  in the ignored cache — then replays through the real VAC path and scores with `cer.py`.
  **Placement is an exact-target inference, not a readback**, as the acceptance required:
  `openvino_genai.WhisperPipeline` wraps its compiled model without exposing it, so there is no
  `EXECUTION_DEVICES` to query; the run establishes `requested_device="NPU"` plus a successful decode,
  and `check_device` admits exact names only (an `AUTO:` spelling would relocate silently).
  **One honest divergence, recorded not smoothed:** decode 111.9 s / 182.482 s = **RTF 0.613**, just
  above D-016's recorded 0.48-0.60. It sits inside M11.4's measured ~20 % run-to-run decode variance,
  so the band is widened to **0.48-0.61** (headroom 1.6-2.1×) in D-016(e) and in `README.md` rather
  than left to read as a miss. 8 VAD utterances over the pause-free clip, matching D-016(a)'s 8.
  Docs realigned onto the shipped path: `models/README.md` rewritten (it still named k2v2 the default
  and never mentioned whisper or OpenVINO at all) with the verified `hf download` acquisition,
  per-engine sizes, and the regenerable compile cache; `README.md` lost its retired sherpa-cadence
  claims — the headline "~0.1 s after endpointing" is the VAD-segment rule, replaced by the VAC
  cadence + partial captions — and gained the status-line/partial-caption description, the
  `VAC_CHUNK_S`/`VAC_TRIM_S`/`ASR_DEVICE` constants, `gate.py` + the four evaluators, and correct
  sherpa-only scoping on the 8-segment queue and the >10 s chunker. `grep -nP '[\x{2013}\x{2014}]'
  README.md` clean (rc=1, positive control fires). Suite unchanged at **212 passed / 1 skipped /
  14.9 s** — the evaluator adds no tests by design. Gate 6/6. `main=65% 155K/240K`; no teammates
  funded (the measurement is script-derivable, the docs judgment-bearing), so `mate=` n/a.
- **T1–T8 · pre-milestone build-out [CLOSED].** `8ec8482..e3a654c`. Timestamped `-o` output,
  `--list-devices`/`--device`, partial-turn shutdown flush, structured logging, the `.githooks/`
  pytest hook (D-007), the sherpa-onnx + silero local STT leg replacing Gemini (D-009/D-010),
  `CodexTranslator` (D-011), deterministic `replay.py` + engine-first goldens (D-014), the 7-clip
  Common Voice corpus (L-017), T8's translator degradation net (L-022). Detail →
  `.agent/archive/t-series-buildout.md`.
- **M9 — agent-runnable accuracy + long-form completeness [REVIEWED].** `e74294c..b2428bd`. `cer.py`
  NFKC/S-D-I scorer + crossfade stressor builder, `tests/eval_cer.py` + `cer_baseline.json`,
  virtual-clock `tests/eval_backpressure.py`, low-RMS decode chunking for >10 s segments, the
  seconds-bounded two-stage `AudioQueue`/`TaskGroup` worker, the pinned 4:48 long-form corpus.
  k2v2 excess deletion 17.02 %/28.03 % → 0.0 %; 4:48 strict CER k2v2 25.38 %, parakeet 23.50 %;
  paced backpressure drop 139 → 0. Produced L-023/L-024. Detail →
  `.agent/archive/m9-accuracy-longform.md`.
- **M10 — Japanese ASR model/architecture tournament [REVIEWED].** `6c20bd6^..bf7e39c`. Closed **by
  events on user decision (2026-09-01)**: D-016 shipped whisper large-v3-turbo int8 on OpenVINO NPU
  under a VAC policy as the default engine, answering M10's question outside its machinery and
  breaking M10's own "preserve k2v2 as default" constraint. Measured conclusions that still stand:
  parakeet displaced k2v2 as the accuracy comparator (Common Voice micro CER 8.426 % vs 8.953 %);
  Qwen3-ASR and Cohere Transcribe failed content **and** CPU-resource gates; no Nemotron streaming
  variant qualified. M10.6/M10.7a/M10.7b retired unrun. **The evaluator stack that produced these
  numbers was deleted on 2026-09-02**; the numbers survive here and in the archive as history, and
  re-deriving one means writing a throwaway script. Detail → `.agent/archive/m10-asr-tournament.md`.
- **T-XLAT-ATTRIB-001 · cascade attribution [CLOSED] — speech→JA is the fault, not JA→EN.** `984fea5`.
  5 arms × 321 FLEURS sentences through the real `CodexTranslator`, 2 judges, human FLEURS EN as an
  unlabelled 6th candidate. Attribution of total adequacy loss: **ASR content 78 %, VAD fragmentation
  16 %, punctuation 5 %**; live-arm severe failures 13.4 % vs 0.0 % oracle. Produced L-028. Its
  harness (`tests/eval_translation.py`) was deleted on 2026-09-02 — its input was the tournament
  detail cache, which is also gone. Detail → `.agent/archive/t-xlat-attrib-001-cascade.md`.
- **T-SAVE-DEFAULT-001 · transcripts persist by default [CLOSED].** `TRANSCRIPT_DIR` = gitignored
  `transcripts/`, one file per run named by local start time; `-o PATH` overrides, `--no-save` opts
  out, argparse makes them mutually exclusive; `TranscriptFile` mkdirs the parent eagerly but defers
  file creation to the first line, so a silent session leaves no empty file.

## Rejected (recorded so they are not re-litigated)

- GitHub Actions CI mirroring the local hook — ~zero marginal catch for a single-user repo; the one
  novel hazard (moved-dir venv shebangs) is absent from CI's fresh-clone env (D-007, L-019).
- Drain residual codex notes at turn entry — the next turn's collect loop already drains leaked
  notes; worst case is one self-healing dropped EN line.
- Assert `check_models` missing-file branch as a literal — tautological change-detector. **Partly
  superseded by M11.2**: engine→marker *routing* has a real failure mode and is now covered.
- Bound `_notes` queue (DoS) — incoherent threat model: flooding needs the locally-spawned,
  user-authenticated codex CLI to attack its own client, already trusted.
- CodexTranslator thread-rotation unit test — tautological (asserts the modulo in the test body).
- Merge `submit`/`submit_sentinel` (deliberate drop-vs-must-land split); cross-file dedup of WAV
  loaders/writers (different formats/sources); centralize the triplicated `CACHE` constant.

## Deferred

- T2.2 — Parameterize source language. Tool is Japanese-only by design (user-confirmed). Revisit if
  the use-case expands.
- M10 candidate-screen remainder: old multilingual streaming zipformer + SenseVoice lack current JA
  evidence; Moonshine-JA has an unclear license; ReazonSpeech-k2-v2 adds PyTorch/Transformers +
  remote custom model code. Re-open only if the shipped path fails and the added runtime surface can
  buy a materially different hypothesis.

## Out of scope (do not redebate)

Config files / YAML / TOML for tunables (constants at the top of `live_stt.py` are the config
surface) · multi-mic mixing · speaker diarization · web UI · auth / multi-user · metrics dashboards
beyond the backlog/drop status counters · package split beyond `streaming.py` (D-002) · reintroducing
contract fingerprints, claim registries, mutation matrices or per-unit review ledgers.

Carried out of M11's standing scope boundary, still closed: engine/model selection (fixed by D-016) ·
NPU applicability for the sherpa fallbacks · energy per watt (RAPL `energy_uj` is mode 400
`nobody:nogroup`, unreadable in-container even under sudo, so NPU-vs-GPU perf/W is a host
measurement). **EN-rendering learning left this list by shipping** — it was out of M11's scope, not
rejected, and `16a842b` landed `observe_en` (D-015); M12 now owns the follow-up question. D-016's declared limits also stand and are not
re-litigated: the hotwords gain used an ORACLE term list drawn from the reference; long_form absolute
CER is inflated by period-vs-modern orthography in the 「ごん狐」 reference; one clip per corpus.

## Decisions pending from user

**One open, and it is not blocking: is M12.2 still worth its cost?** M12.1's census answered M12's
two named failure modes without the arm matrix — (b) refuted, (a) unreachable on this corpus — so
what M12.2 would now buy is the answer to the third mode the census exposed: does briefing
`標柱 = <spelling>` help or harm the English when the name is really 兵十? Options: (a) **run M12.2 as
scoped** — 6 real `CodexTranslator` sessions off the committed trace, both statistics reported, the
rendering-count arm marked underpowered; (b) **close M12 on the census** and rule on P-012's named
fix on its merits (gating pairing on recurrence-since-trust, which on NPU is what "un-prompted since
trusted" degenerates to, blocks a dead pairing at one extra sighting per pairing) — cheaper, and it
leaves the third mode unmeasured; (c) **fund the corpus first** (「ごん狐」 sections 二+ under L-017),
which is what a POWERED dead-pairing arm needs and what would let a rendering-count arm clear the
floor. Absent a ruling the next session runs (a), which is what the roadmap plans. The polish register
holds one row again (**P-014**, the NPU decode-stall bursts M12.1 measured). Standing options for
after M12, none of them blocking:
(a) **A live-mic validation pass** — the user-only debt is the largest untested surface and only the
    user can run it: latency feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, and the whole
    VAC partial-caption cadence (`memory.md` § Smoke). Agent coverage cannot substitute here.
(b) **A maintenance pass** (L-018) — dependency/CVE sweep, lock bumps, full gate, re-verify the codex
    leg.
(c) **A new capability milestone** — needs a direction from the user, since `## Out of scope` closes
    most of the obvious ones.

(Last resolved: **the P-012 disposition** — the row asked whether `observe_en` survives real ASR
output, and re-sizing it against tree showed it was a milestone wearing a `size=M` label. Offered
(a) close it the way P-009 closed, accepting the learner with D-015's recorded clean-text limit, or
(b) fund it as a roadmap unit; the user chose **(b)** on 2026-09-02, to be carried out in the next
session, and authorized the two stale-line corrections in this file at the same time. Before that:
**assurance-apparatus scope** — measurement showed ~11,100 lines of apparatus
guarding a 2,108-line personal tool, and the session was blocked writing a third fingerprint-migration
clause. Offered (a) cut provenance machinery, (b) additionally retire the M10 tournament stack and the
VAC evaluator, or (c) unblock only and continue the 9-unit tail; the user chose **(b)** on 2026-09-02,
with the standing judgment that this tool is "simple, but high performant, for personal use" and that
its development "needn't be so rigid". Before that: **M11.3d evidence scope + legacy-byte retention** —
the user chose 2 arms strict-NPU on 2026-09-02, a ruling now moot because the arm matrix is deleted.
Before that: **M10 disposition** — close M10 and open a fresh milestone (2026-09-01). Before that:
**ASR device** — NPU over GPU on 2026-08-31, shipping D-016 and forfeiting `hotwords`. Before that:
**how the streaming leg closes** — close on existing evidence (2026-08-03). Before that: the two
`TRANSLATOR_INSTRUCTIONS` defect lines — generic drug names + never invent a patient's sex, approved
2026-08-03. Before that: translation leg → `gpt-5.6-luna`+`low`.)
