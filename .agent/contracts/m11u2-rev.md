# rev-m11u2 — M11.2 adversarial review (lens: correctness / spec / vacuity)

## Phase 1 — check set

Fixed deliverable: **44 rows**. Every row starts `unknown`; adjudication proceeds in batches of four.

| ID | Verdict | Predicate | Evidence |
|---|---|---|---|
| R01 | pass | Scope = exactly the two named test files; production remains byte-identical to `b01a332`; collected test count = 46. | Latest primary snapshot: 40 shipped-path cases + 6 mutation/control cases; `pytest --collect-only` = 46; `cmp` confirms both production files unchanged. |
| R02 | pass | Hard constraint: tests use no microphone, model download, real `WhisperPipeline`, accelerator device, or `models/` tree. | Fake binding enters `sys.modules`; model paths/cache are `tmp_path`; focused suite passes with no `models/` tree. |
| R03 | pass | P1: CLI parsing makes omitted `--engine` select behaviorally observed engine `whisper`. | Bare CLI test observes `check_models("whisper")` and the Whisper/OpenVINO banner; M1 kills it. |
| R04 | fail | P1: `--engine` accepts exactly `sorted(ENGINE_DIRS)` through argparse behavior. | **Low** — `tests/test_shipped_path.py:674`: acceptance/rejection proves the set, not sorted help order. Reversing `choices` at `live_stt.py:1385` leaves all 40 shipped-path tests green. Breach=P1 exact `sorted(ENGINE_DIRS)` clause; impact=reordered/nondeterministic CLI help can regress unnoticed. Acceptance: capture `main --help` and assert the rendered choice sequence equals sorted keys. |
| R05 | pass | P1: omitted `--asr-device` behaviorally reaches `ASR_DEVICE`. | Default construction and bare CLI banner both observe NPU; M2 makes the suite fail. |
| R06 | pass | P1: every `WHISPER_ENGINES` key routes to `WhisperEngine(ENGINE_DIRS[e], dev)`. | Parameterized over the complete set; observes class, model path, and requested GPU. |
| R07 | pass | P1: every remaining `ENGINE_DIRS` key routes to a sherpa constructor that reads that engine’s own directory. | Current non-Whisper set is exactly `k2v2` + `parakeet`; both constructor families and own-directory arguments are observed. |
| R08 | pass | P2: construction mkdirs the cache and passes string model path, exact device, and `CACHE_DIR=str(OPENVINO_CACHE_DIR)`. | Cache test observes directory + exact kwargs; routing test observes string model path and exact device. |
| R09 | pass | P2: `supports_hotwords == (device in ASR_HOTWORDS_DEVICES)` for capable and incapable devices. | CPU/GPU are parameterized true; shipped NPU is asserted false; M3 reverses the important branch. |
| R10 | pass | P2: `set_hotwords` stores the list when capable and stores `""` when incapable. | CPU/GPU retain the exact string; NPU drops it to the exact empty string. |
| R11 | fail | P2: `generate` always passes Japanese language, transcribe task, and the requested timestamp flag. | **Medium** — `tests/test_shipped_path.py:187-211` checks language/task only for `timestamps=False`; timestamped VAC only checks the flag. Making `live_stt.py:337` pass `language=None` when timestamps are requested leaves 60 shipped+streaming tests green. Breach=P2 “always”; impact=VAC can lose its Japanese pin while tests stay green. Acceptance: assert language + task on a direct timestamped generate/decode-segments call; survivor mutation must die. |
| R12 | fail | P2: `generate` passes non-empty hotwords and omits the keyword when empty. | **Low** — `tests/test_shipped_path.py:166-195` proves omission only on incapable NPU. Sending `hotwords=""` on capable CPU/GPU at `live_stt.py:331` leaves 60 shipped+streaming tests green. Breach=P2 “only when non-empty”; impact=binding-call contract can drift unnoticed. Acceptance: decode once on a capable device before `set_hotwords` and assert the keyword is absent. |
| R13 | pass | P2: `decode` joins and strips generated text fragments. | Multi-fragment result with outer/inter-fragment spaces asserts the exact joined+stripped string. |
| R14 | pass | P2: `decode_segments` returns the generated transcript plus timestamp spans. | Exact transcript, two converted spans, and timestamp request are asserted. |
| R15 | pass | P2: `decode_segments` returns empty spans when concatenated segment text differs from the transcript. | Both mismatching chunks and absent chunks return the transcript with `[]`. |
| R16 | fail | Fake fidelity: `_FakePipeline` matches the real binding contract used by `WhisperEngine` closely enough that invalid production arguments/results cannot pass only because the fake is permissive. | **High** — `_FakePipeline.generate` at `tests/test_shipped_path.py:67` derives output only from sample count; every test supplies zeros and none checks forwarded values/dtype. Replacing the real input at `live_stt.py:336` with `np.zeros_like(samples)` leaves all 60 shipped+streaming tests green. Breach=fake fidelity/P2 decode path; impact=real speech can become silence while the suite certifies `WhisperEngine`. Acceptance: use a nonzero float32 sentinel and assert the fake receives byte/value-identical one-dimensional samples; add the zero-audio survivor to the mutation matrix. |
| R17 | pass | P3: speech-open with context obtains offered ASR terms and calls a recognizer’s `set_hotwords`. | GPU integration starts empty and finishes with the exact offered string, proving the call and payload at speech-open. |
| R18 | pass | P3: truthy post-set `rec.hotwords` makes `biased` exactly the offered frozen term set. | GPU integration observes exactly `frozenset({"東京", "タワー"})`. |
| R19 | pass | P3: dropped/falsy post-set `rec.hotwords` makes `biased=frozenset()`. | Real default engine drops the offer; integration observes the exact empty frozen set. |
| R20 | pass | P3: `context.observe_ja` receives exactly the branch-selected `biased` set. | Both NPU-empty and GPU-offered branches assert `(published_text, exact_set)`. |
| R21 | fail | P4: `check_models` routes Whisper keys to `openvino_encoder_model.xml` and every other engine to `tokens.txt` under its own directory. | **Medium** — `tests/test_shipped_path.py:258-281` proves readiness for Whisper + `k2v2`, but only proves that a Whisper marker fails for `parakeet`. Making `parakeet` require nonexistent `token.txt` at `live_stt.py:468` leaves 60 tests green. Breach=P4 every-other-engine clause; impact=a complete parakeet install can be rejected. Acceptance: parameterize successful marker routing over every `ENGINE_DIRS` key. |
| R22 | fail | P4: missing VAD and marker are each named; both present returns `None`. | **Low** — `tests/test_shipped_path.py:258-291` tests each missing item only in isolation. Suppressing the marker whenever VAD is already missing at `live_stt.py:469` leaves 60 tests green. Breach=P4 “and/or”; impact=the preflight can hide one of two actionable missing assets. Acceptance: leave both absent and assert both `silero_vad.onnx` and the engine directory occur in one message. |
| R23 | pass | P5/L-006: off-TTY `meter` performs no stdout write. | A nonempty partial + dropped counter still produces zero writes when `_STDOUT_TTY=False`. |
| R24 | pass | P5: on-TTY `meter` begins with `_LINE_CLEAR` and renders only nonzero counters. | Idle output is exactly `_LINE_CLEAR`; each nonzero queue/drop counter appears; zero translator drops stay absent. |
| R25 | fail | P5: partial text is tail-truncated to remaining width; empty partial adds no separator. | **Low** — `tests/test_shipped_path.py:640` proves only “fits” + tail retention. Changing the reserved separator width at `live_stt.py:1251` from 3 to 4 leaves 60 tests green. Breach=P5 exact remaining-width arithmetic; impact=avoidable caption loss/off-by-one regressions pass. Acceptance: assert the exact rendered body and maximal tail length for a fixed width/status, plus no separator for an empty partial with nonempty status. |
| R26 | fail | P5: each VAC commit updates `state.partial` to the running utterance and finalize clears it. | **Medium** — `tests/test_shipped_path.py:542` accepts a partial that becomes nonempty once and never grows. Replacing `live_stt.py:1134` with `state.partial = state.partial or utterance` leaves 60 tests green. Breach=P5 each-commit clause; impact=the live caption can freeze after its first commit while final publication still passes. Acceptance: record every nonempty commit and assert `state.partial` equals each cumulative utterance; preserve final clear. |
| R27 | pass | P6: presence of `decode_segments` selects `_vac_segments` and creates no segment-queue work. | Route spy observes only `vac`; feed/decode are never called; worker remains healthy. |
| R28 | pass | P6: absence of `decode_segments` selects the `_feed_segments`/`_decode_segments` TaskGroup. | Text-only and sherpa-shaped recognizers each observe both feed + decode tasks and no VAC call. |
| R29 | unknown | P7: one non-empty utterance emits JA once, submits translation once, and observes context once with identical sequence/text. | pending |
| R30 | unknown | P7: every intermediate `update(final=False)` tick publishes nothing. | pending |
| R31 | unknown | P7: sequence advances only on publication; empty text publishes nothing but calls `on_segment` once. | pending |
| R32 | unknown | VAC arithmetic: `[True]*40 + [False]` at `window=1600` produces at least two intermediate update ticks before finalize under production constants. | pending |
| R33 | unknown | M1: changing default engine `whisper→k2v2` fails a behavioral test, not a literal-constant echo. | pending |
| R34 | unknown | M2: changing `ASR_DEVICE NPU→CPU` fails a behavioral test. | pending |
| R35 | unknown | M3: adding `NPU` to `ASR_HOTWORDS_DEVICES` fails a hotword/drop integration assertion. | pending |
| R36 | unknown | M4: changing the Whisper marker filename fails an engine→marker routing assertion. | pending |
| R37 | unknown | M5: changing the duck predicate to `hasattr(rec, "decode")` fails a dispatch assertion. | pending |
| R38 | unknown | Mutation harness isolates each edit and its unmutated positive control passes. | pending |
| R39 | unknown | M1–M5 mutations probe the material decision at each surface rather than only the easiest equivalent edit. | pending |
| R40 | unknown | A sixth reasonable shipped-path production decision cannot break while all new tests remain green; prove any gap with an added worktree mutation. | pending |
| R41 | unknown | Vacuity inventory: every one of 46 tests has a named production change it detects; any pure change-detector or tautology is identified. | pending |
| R42 | unknown | Isolation: no leak through globals, `sys.modules`, stdout/TTY state, interval constants, or monkeypatched class attributes; full suite passes in both relevant orders. | pending |
| R43 | unknown | Gate identity: `gate.py` reports 6/6 blocking green and the known `aggregate-only` step non-blocking-red. | pending |
| R44 | unknown | Evidence quality: each fail names severity, source line, divergence, breached predicate, impact, acceptance check, and a red worktree test where expressible. | pending |

## Phase 2 — adjudication

Replace `unknown` with `pass` or `fail` in batches of four. Failure evidence carries the complete acceptance record.

## Register

Observations outside the acceptance contract go here with an evidence pointer and acceptance check.
