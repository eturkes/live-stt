# rev2-m11u2 — M11.2 adversarial review (claim soundness / guarantee gaps / CLAUDE.md)

## Phase 1 — check set

Fixed before reading MAIN's diff. Verdicts move from `unknown` to `pass` / `fail` only; rows never move.

| ID | Verdict | Predicate | Evidence |
|---|---|---|---|
| R01 | unknown | P1: whisper `load_recognizer(engine, device)` routes through `WhisperEngine(ENGINE_DIRS[engine], device)`. | pending |
| R02 | unknown | P1: every non-whisper `ENGINE_DIRS` key routes through its sherpa constructor using that key's own directory. | pending |
| R03 | unknown | P1 / live CLI: `--engine` exposes `sorted(ENGINE_DIRS)` and defaults behaviorally to `whisper`. | pending |
| R04 | unknown | P1 / live CLI: `--asr-device` defaults to `ASR_DEVICE` and the parsed value reaches recognizer loading. | pending |
| R05 | unknown | Roadmap live CLI: `--context` defaults empty and a supplied value reaches session-context construction. | pending |
| R06 | unknown | P2: `WhisperEngine` creates the cache directory and constructs `WhisperPipeline(str(model_dir), device, CACHE_DIR=str(OPENVINO_CACHE_DIR))`. | pending |
| R07 | unknown | P2: `supports_hotwords == (device in ASR_HOTWORDS_DEVICES)` and a capable device stores the offered list. | pending |
| R08 | unknown | P2: an incapable device drops the offered hotwords to `""`. | pending |
| R09 | unknown | P2: `generate` always sends Japanese language, transcribe task, and the requested timestamp flag. | pending |
| R10 | unknown | P2: `generate` includes `hotwords` iff the stored value is non-empty. | pending |
| R11 | unknown | P2: `decode` joins all result texts and strips the joined transcript. | pending |
| R12 | unknown | P2: `decode_segments` maps chunks into exact `Segment(start, end, text)` spans and returns the transcript. | pending |
| R13 | unknown | P2: `decode_segments` returns `[]` spans when concatenated segment text differs from the transcript. | pending |
| R14 | unknown | P3: speech-open accepted conditioning sets `biased` to the exact offered term set and publishes that set to `observe_ja`. | pending |
| R15 | unknown | P3: speech-open dropped conditioning (the NPU case) sets `biased=frozenset()` and publishes that empty set to `observe_ja`. | pending |
| R16 | unknown | P4: `check_models` routes whisper engines to `openvino_encoder_model.xml` and every other engine to `tokens.txt` under its own directory. | pending |
| R17 | unknown | P4: missing VAD and marker are named independently/together; both present returns `None`. | pending |
| R18 | unknown | P5 / L-006: off-TTY `meter` returns without any stdout write. | pending |
| R19 | unknown | P5: on-TTY `meter` writes `_LINE_CLEAR` and only nonzero backlog/drop counters. | pending |
| R20 | unknown | P5: partial text is tail-truncated to remaining width; empty partial adds no separator. | pending |
| R21 | unknown | P5: `_vac_segments` updates `state.partial` on each commit and clears it after every finalize. | pending |
| R22 | unknown | P6: a recognizer with `decode_segments` takes `_vac_segments` and creates no segment-queue work. | pending |
| R23 | unknown | P6: a recognizer without `decode_segments` takes the `_feed_segments` + `_decode_segments` TaskGroup. | pending |
| R24 | unknown | P7: one non-empty utterance emits JA, submits translation, and observes context exactly once with identical text/sequence semantics. | pending |
| R25 | unknown | P7: intermediate `update(final=False)` ticks publish nothing and sequence advances only on publication. | pending |
| R26 | unknown | P7: an empty utterance publishes nothing but invokes `on_segment` exactly once. | pending |
| R27 | unknown | M1: changing the default engine `whisper→k2v2` kills a committed behavior-level test, not a literal echo. | pending |
| R28 | unknown | M2: changing `ASR_DEVICE` `NPU→CPU` kills a committed behavior-level test, not a literal echo. | pending |
| R29 | unknown | M3: adding `NPU` to `ASR_HOTWORDS_DEVICES` kills a committed behavior-level test, not a literal echo. | pending |
| R30 | unknown | M4: changing the whisper marker filename kills a committed behavior-level routing test, not a literal echo. | pending |
| R31 | unknown | M5: changing the duck predicate to `hasattr(rec, "decode")` kills a committed behavior-level dispatch test, not a literal echo. | pending |
| R32 | unknown | Mutation harness integrity: each arm runs committed tests against isolated mutated production, and an unmutated positive control stays green. | pending |
| R33 | unknown | Hard constraint: no new test imports/opens a microphone or requires an audio device/backend. | pending |
| R34 | unknown | Hard constraint: no new test requires `models/`, downloads a model, or depends on model payloads. | pending |
| R35 | unknown | Hard constraint: no new test needs working OpenVINO, constructs a real `WhisperPipeline`, or touches accelerator devices. | pending |
| R36 | unknown | Hard constraint by execution: both new files pass in the model-free worktree under the prescribed shared environment. | pending |
| R37 | unknown | Scope: durable changes are tests only; no production change or unrelated artifact enters M11.2. | pending |
| R38 | unknown | Scope: no M11.3 evaluator/fingerprint/Whisper-golden work is performed. | pending |
| R39 | unknown | Scope: no M11.4 VAC timing/backpressure/force-trim qualification work is performed. | pending |
| R40 | unknown | Scope: no M11.5 evidence-run/baseline/constant-binding work is performed. | pending |
| R41 | unknown | Scope: no M11.6 docs/smoke work or off-spine `polish.md` work is performed. | pending |
| R42 | unknown | Honesty: `test_shipped_path.py`'s module docstring claims exactly what its tests prove. | pending |
| R43 | unknown | Honesty: `test_mutations.py`'s module docstring claims exactly what its tests prove. | pending |
| R44 | unknown | CLAUDE Authoring / Engineering: `test_shipped_path.py` is dense and agent-optimized; comments/docstrings explain `why`, omit provenance, and avoid redundant `what`. | pending |
| R45 | unknown | CLAUDE Authoring / Engineering: `test_mutations.py` is dense and agent-optimized; comments/docstrings explain `why`, omit provenance, and avoid redundant `what`. | pending |
| R46 | unknown | CLAUDE Engineering / memory L-005: helpers and fixtures stay tightly scoped, deduplicated, KISS, and free of premature abstraction. | pending |
| R47 | unknown | House idiom: shipped-path test layout, helper naming, monkeypatching, assertions, and long test names fit `test_streaming.py`, `test_audio.py`, `test_context.py`, and `test_gate.py`, or divergence is justified. | pending |
| R48 | unknown | House idiom: mutation test layout, helper naming, subprocess use, assertions, and long test names fit the existing corpus, or divergence is justified. | pending |
| R49 | unknown | CLAUDE deterministic-check rule: tool-decidable style/type/test properties are owned by the executable gate from committed-equivalent state. | pending |
| R50 | unknown | CLAUDE assurance rule: `tier=kernel` rigor is a full adversarial battery for every named shipped-surface decision and failure path. | pending |

## Phase 2 — adjudication

Rows are written in batches of four. A `fail` row carries severity, `file:line`, divergence, breached predicate, impact, and acceptance check.

## Register

Observations outside the acceptance contract; each carries an evidence pointer and acceptance check.
