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

- Milestone: **M11 production-qualify the shipped whisper+VAC+NPU path** — IN-PROGRESS.
  **2 units remain, both OPEN**: M11.4 (VAC real-time drop-freedom) and M11.5 (the unsourced
  retention CER). Next = **M11.4**.
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
  narration **on the sherpa fallback only**; deterministic 44.722 s paced backpressure **on the
  sherpa fallback only**; shutdown/stage-failure/translation-degradation mechanics.
- User-only debt: latency feel, `-o`, soak, sustained live cadence, Ctrl+C-mid-decode, and the whole
  VAC partial-caption cadence (`.agent/memory.md` § Smoke, L-004).

## Open (do these; lowest ID first)

### M11 — production-qualify the shipped whisper+VAC+NPU path

Outcome: answer the two questions that decide whether the shipped default path is trustworthy. M10
chose the engine; D-016 shipped it. What is missing is (a) proof the VAC branch does not drop audio
in real time, and (b) a real number behind D-016's headline retention CER.

Standing scope boundary — do not reopen: engine/model selection (fixed by D-016); NPU applicability
for the sherpa fallbacks; EN-rendering learning (`polish.md` P-002); energy per watt (RAPL
`energy_uj` is mode 400 `nobody:nogroup`, unreadable in-container even under sudo).

D-016's declared limits stand and are not re-litigated: the hotwords gain used an ORACLE term list
drawn from the reference; long_form absolute CER is inflated by period-vs-modern orthography in the
「ごん狐」 reference; one clip per corpus.

- **M11.4 — Does the VAC path drop audio in real time? [OPEN]**

  The real-time guarantee does not cover the shipped branch. `tests/eval_backpressure.py:116-196`
  forces legacy dispatch (`worker(object(), …)`), charges each decode `closed_segment × RTF` at
  default RTF 0.20 (`:60-61`), and asserts a nonzero segment queue (`:253-268`). VAC has no segment
  queue: it re-decodes an open, growing, trimmed buffer every 1 s and **awaits each decode inside the
  coroutine draining `audio_q`** (`live_stt.py:1122`, `:1149-1187`, `:1196-1223`), so VAD feeding
  pauses during decode while capture buffers into 2 s only (`live_stt.py:37`, `:492-518`).
  `README.md:80` claims the opposite — "Capture and VAD feeding continue during decode." Aggregate
  RTF 0.48-0.60 proves mean compute < real time, never bounded maximum blockage. Drop-freedom on the
  shipped path is **unmeasured, not disproved**.

  Three pieces, one unit, one commit:
  1. **Observation seam.** Per-`StreamingProcessor.process` hook on `_vac_segments` carrying the
     committed text, the commit audio endpoint that `live_stt.py:1155` (`commit, _ = await
     loop.run_in_executor(...)`) discards, the buffer end, and the decode duration; passed through
     `worker` and `replay.py` beside `on_segment` (D-014's sanctioned observation-hook precedent).
     Fires exactly once per call, in order. Keep the deterministic fields separate from the measured
     duration. This is now ~25 lines: the fingerprint that made it expensive is deleted.
  2. **Real per-update decode costs.** Replay both pinned corpora through the shipped NPU path with
     the seam attached and keep the ordered `(buffer_s, decode_s)` pairs. `.scratch/` holds only
     three-repeat medians at 2/5/10/20/30 s, which cannot pace a 1 s cadence.
  3. **Paced VAC scenario** in `tests/eval_backpressure.py`: a `decode_segments` recogniser, real
     silero, production cadence and trim, executor interception selecting cost by current buffer
     duration, and segment-queue depth asserted **exactly zero** in place of the legacy
     `0 < depth <= SEGMENT_QUEUE_MAX`.

  Acceptance: the 44.722 s stressor and the 182 s retention clip both run paced at 20 ms with
  `state.dropped == 0`; a deliberately overloading trace produces a nonzero drop, proving the check
  is not vacuously green; committed queue high-water and maximum-contiguous-decode bounds; report
  `StreamingProcessor.forced_trims` (it already exists — a nonzero count means the trim rule failed
  and discarded un-emitted audio, so surface it rather than asserting it away). Close by correcting
  `README.md:80` to what was measured and updating § Smoke's VAC observables.

  Red result — any dropped block, maximum contiguous decode reaching the 2.0 s queue headroom, or
  backlog repeated sub-2 s updates cannot clear — **stops and reaches the user**: decoupling decode
  from `audio_q` draining changes concurrency, buffer ownership, failure coupling and shutdown order,
  and that redesign is the user's call, not a branch inside this unit.

  Per-character lag definition, if lag is wanted alongside drop-freedom (derived, do not re-derive):
  for each update set `end=commit_audio_s`, `start` = previous committed endpoint, spread
  `len(text)` characters uniformly at midpoints `at_i = start + (end-start)*(i+0.5)/len(text)`, and
  record `lag_i = emit_s - at_i`; on replay derive `emit_s` on the virtual audio clock as
  `now = max(now, buffer_end_s) + decode_s`. A final update uses the utterance end as
  `commit_audio_s`. Never estimate lag from final segments — that collapses every early VAC commit
  into one utterance-close event.

- **M11.5 — Is D-016's retention CER 0.0583 real? [OPEN; after M11.4]**

  **The number is confirmed unsourced.** Every `.scratch/` JSON, MD and log was searched; nothing
  produces it, and only the planning reports repeat it. The search is credited by its positive
  control, which did find `0.0686106346483705` in `.scratch/policy_retention_vac_npu.json` (`I=12`).
  Both the headline retention improvement (0.1587 → 0.0583) and the append-only-`emitted` fix's
  claimed effect (`I` 12 → 0, CER 0.0686 → 0.0583) rest on it.

  Re-derive it: replay `retention_probe.wav` through the shipped NPU/VAC path and score with `cer.py`
  against `tests/retention_probe.json`'s `probe.ja_ref`. Prior warm-cache timing puts one pass near
  4.6 min. M11.4 precedes it because a decode/ingest redesign there would obsolete a number measured
  before it.

  Acceptance: the retention CER and its `I` are re-measured from committed inputs and recorded in the
  commit body; D-016 in `.agent/memory.md` either keeps 0.0583 with the reproduction recorded beside
  it, or is corrected in place naming the superseded value. A divergence is an explicit correction,
  never a silent rewrite. Placement evidence is recorded as an exact-target inference, not
  `EXECUTION_DEVICES`: the shipped `openvino_genai.WhisperPipeline` exposes no compiled model, so
  record `requested_device="NPU"` plus the successful exact-target decode and the API limitation.
  Close by realigning `README.md` and `models/README.md` onto the shipped path — install command,
  whisper model acquisition, VAC cadence, partial captions, and removal of every retired k2v2/VAD
  default claim; `grep -nP '[\x{2013}\x{2014}]' README.md` stays clean (L-021).

## Done (ID · outcome · decisions/lessons produced)

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

## Decisions pending from user

**None open.** M11.4 raises the next one if it comes back red: whether to decouple VAC decode from
`audio_q` draining.

(Last resolved: **assurance-apparatus scope** — measurement showed ~11,100 lines of apparatus
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
