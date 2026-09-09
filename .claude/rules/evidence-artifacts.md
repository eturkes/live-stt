---
paths:
  - "tests/*.py"
  - "tests/*.json"
---

# Evidence artifacts — corpora, traces, evaluators

Fast locks run in the gate. **The eight `eval_*.py` scripts are ON-DEMAND and never gate steps**: four
need gitignored weights and minutes of compute, so run one when a decode change raises an accuracy
question. `eval_latency.py`, `eval_term_census.py` and `eval_en_pairing.py` (default mode) need
neither, because they replay committed traces — a fresh clone runs them in under a second each.

## Committed artifacts and what each certifies

- `replay_goldens.json` + `gen_replay_goldens.py` — characterization snapshots, engine-first keying,
  asserting segment count + per-segment text + boundary (±0.1 s); CPU-variable decode latency is
  reported only. A cell the local machine cannot run (absent weights, WAV or accelerator) carries its
  committed row forward instead of vanishing, so a whisper regeneration on a box with no NPU cannot
  silently delete the committed whisper row.
- `short_corpus.json` + `fetch_real_clips.py` + `real_clips.json` + `test_corpus.py` — pinned Common
  Voice / FLEURS PCM corpus, compact evidence plus fail-closed locks. The real-recorded clips expose
  engine divergence the synthetic TTS corpus could not: 松井/松居, バック/パック, 午後七時/午後7時.
- `stressor_clips.json` + `build_stressor.py` — the 44.7 s genuinely-continuous stressor: silero-trimmed
  speech extents joined by ~10 ms equal-power crossfades (gap-concat leaves clip-edge quiet the VAD
  rightly splits on). Prove continuity honestly — every crossfade-join offset sits inside a cap-OFF VAD
  segment, and cap-ON yields more segments than the cap-OFF control; `n_seg < n_comp` is gameable.
- `retention_probe.json` + `build_retention_probe.py` + `test_retention_probe.py` — the 182 s pause-free
  state-retention probe (WAV gitignored, model-free locks).
- `long_form.json` + `eval_long_form.py` — the pinned LibriVox/Kokoro/Aozora narration: all six sections
  of 「ごん狐」, 848.351 s / 213 VAD segments, schema `{source, sections{"01".."06"}}`. **No row number
  is written down** — a section's range is discovered by grouping the alignment on its own audio-file
  column and reading the aligned text for its title/heading rows, so a re-released alignment fails a
  check instead of mis-cropping. Kokoro leaves 4-10 s of narration unaligned INSIDE four sections; the
  crop stays one continuous span (a hole is unaligned audio, never a splice), so `alignment_check`
  spends the unaligned fraction of the span as budget over the flat 0.10 surface allowance. Sherpa
  scoring is `--score`-only and a section keeps recorded rows while its WAV + reference hashes hold, so
  acquisition reruns without re-decoding. Downstream consumers resolve their section from the artifact
  (`source.wav`), never from a constant.
- `vac_decode_trace.json` + `build_vac_trace.py` + `eval_latency.py` — per-update `(buffer_s, decode_s)`
  **plus the hypothesis that produced each one**. Storing the hypotheses is what makes the trace
  replayable: `StreamingProcessor` is a pure function of decode outputs + buffer lengths, so replaying
  them reproduces the measured commit/trim trajectory with no model, and `divergences == 0` certifies
  each cost was charged to the buffer it was measured on. 122 KB ⇒ read the per-clip summary keys, not
  `series`. Rebuild needs the NPU + the whisper prelude.
- `caption_trace.json` + `build_caption_trace.py` — the shipped path's caption stream over the whole
  pinned narration, every section hash-checked before any of it is decoded: 215 captions / 848.350 s on
  NPU in one ~8 min pass, numbered continuously, `sections[k].offset_s` placing section-local times on
  the continuous timeline. One decode per section is faithful because nothing on the shipped path
  carries state between utterances. **Caption text + VAD boundaries reproduce byte-identically across
  NPU runs; decode cost does not** — that determinism is what lets every downstream arm replay this
  file instead of paying for an accelerator run. A caption's `decode_s` is a per-utterance SUM, so read
  real-time risk as carry (`asr-pipeline.md`).
- `en_pairing_trace.json` + `eval_en_pairing.py` + `test_en_pairing.py` — what a REAL translator hands
  the learner. `--live` puts all 215 committed captions through `SessionContext` + `CodexTranslator` in
  production's order (`observe_ja` → `_translate` → `observe_en`); the default mode re-derives every
  verdict from the trace offline. **Deviation from production, reported per run:** a leg disabled by
  `TRANSLATE_MAX_FAILURES` is restarted against the same context, since this measures the learner and
  not the degradation path — `restarts: 0` = production-identical. The trace is ONE sample of sampled
  model output ⇒ its verdicts are structural, never rates.
- `eval_term_census.py` + `test_term_census.py` — what the recogniser gives `SessionContext` as a key,
  and whether a key dies with a rendering on it. Occurrences are located by `cer.alignment` against the
  reference — never by searching for guessed spellings — one alignment per SECTION, then widened to the
  covering `_TERM_RUN` candidate because a form below its floor is invisible to the learner. `--floor N`
  rewrites the shipped `_TERM_RUN` rather than restating it, so an arm differs from production in the
  katakana floor and nothing else, and it refuses a pattern it cannot locate exactly one floor in.
- `cer_baseline.json` + `eval_cer.py` — model-gated two-engine CER / stressor / RTF evaluator.
  `eval_retention.py` — the shipped path's one agent-rerunnable ACCURACY gate: hash-gates the probe WAV,
  replays it through the whisper/VAC path on an exact-named device, scores with `cer.py` (~4 min warm).
  `eval_translate_repeat.py` — the producer of `CAPTION_REPEAT_MAX_CHARS`, through the real
  `codex app-server`; probe inputs are literals, so reruns need no artifact.
- `eval_backpressure.py` + `test_backpressure.py` — the production two-stage worker paced on a virtual
  sample clock with real silero and seconds-bounded queues, only decode cost replaced. Arms: the sherpa
  two-stage path, the VAC arm on recorded real NPU costs (`max_segment_depth == 0`), the `SCALE_LADDER`
  margin, and the carry arm over `caption_trace.json` — which needs no corpus and no skip, so the
  shipped path's real-time reserve stays checkable in a fresh clone.
- `session_report.py` + `test_session_report.py` — what a LIVE session did, re-derived from the two
  files that session already wrote (`transcripts/*.txt` + redirected stderr). No hardware, no
  weights, no network, and it imports the shipped `repeat_span`/`caption_defect` rather than
  restating them, so a threshold change moves the report with it. Answers what `live-smoke.md`
  names: captions with no EN and **why** (`declined`/`strike`/`disabled`/`shutdown`/`failed`),
  degrade + restore markers with timestamps, `backlog peak:` high-water lines, caption length +
  repetition distributions, EN-behind-JA lag, and slow turns tagged with whether they sit on a
  `TRANSLATE_ROTATE_TURNS` boundary. Two rules the live corpus forced: **a session owns log time
  from its own start until the next session starts**, never to its last caption, because
  `codex app-server exited` fires after the final caption by construction; and **once the leg is
  down, `disabled` outranks the text screen**, since a screen verdict behind a degrade is a
  counterfactual and rides `screened_now` instead of explaining the loss. Over the six saved
  sessions: 1073 captions, 1001 translated, 72 without EN; unit-bound sweep 26 caught at 6-8, 27 at
  9-11 (the newly-caught one being the 9-character loop that escaped bound 8), 29 at 13-21, 30 at
  22+; session 1 inferring a 3-strike degrade at last EN n=194, session 6 reading
  `codex app-server exited` off the log at 14:38:49.
- `test_gate.py` locks `gate.py`'s step inventory with one seeded defect per blocking step, in a
  throwaway tree, plus the secret scan's real-tree scope, which that throwaway tree cannot see
  shrink (`toolchain.md`).

## Rules that keep the evidence honest

- **D-014 — deterministic WAV replay is the regression harness.** `replay.py` drives the real
  `live_stt.worker` through an optional observation-only `on_segment` hook, chosen over
  freeze-and-reimplement precisely because a reimplemented loop is what drifted before. It feeds 1 s
  views, within the live `AudioQueue`'s 2 s headroom: a whole-file block larger than the 60 s ring
  delays VAD popping until early samples are evicted — an evaluator-only artifact. Live `worker` turns
  a stage failure into `state.request_stop()`; replay RE-RAISES that signal, so a partial or empty
  transcript cannot become a golden. `replay.py` defaults to `k2v2` because the goldens do, not because
  live-stt does.
- **L-024 — backpressure capacity is TIME; prove it with paced production replay.** Callback block
  sizes are backend-selected, so a block-count queue cannot express audio headroom — bound queued PCM
  seconds directly. Fast-as-possible file replay cannot expose real-time serialization: pace arrivals
  on a virtual sample clock, replace decode cost alone, and keep the production worker/VAD/ring/queue/
  drop policy. Keep feeding separate from sequential decode, copy ring slices before queueing them,
  bound both stages, and failure-couple siblings. Modeled timing proves the known workload, never
  arbitrary host scheduling ⇒ pair it with the live meter/soak checks.
- **L-025 — evaluator completeness is input/event accounting, not nonempty text.** On the complete
  short corpus both offline controls consumed every sample and EOF exactly once yet shared 10 identical
  empty hypotheses with `segments=[]` (quiet/short material the production VAD rejects). Those are
  completed ASR misses ⇒ retain the case ids and score D=N. A failed or partial run means missing,
  duplicate or reordered ids, unconsumed audio, wrong EOF/finalization, or a worker exception — never
  one completed clip emitting no text. Backend-declared generation/context truncation is a third state:
  capture native diagnostics per phase and fail a separate deterministic content gate instead of
  relabeling completion.
- **L-017 — corpus evidence = pinned payloads → canonical PCM → content-addressed index.** Dataset
  viewers lack revision selectors and can disappear: acquire revision-addressed files directly, verify
  size + SHA-256 on fresh AND cached paths, parse the declared schema, stage every decoded row, then
  install an index-addressed cache directory and commit its compact fingerprint last. Pin transient
  decoders and the expected whole-index hash, so package drift becomes an explicit requalification
  rather than silent evidence churn. Count repeated references and audio; uniqueness is a result, not
  an assumption.
- **L-026 — a model bench is evidence only if the harness mirrors production failure handling and the
  judge sees raw candidates.** Import the real config/instructions/thread options rather than retyping
  them. A `codex app-server` turn that times out keeps running server-side, so without
  `turn/interrupt` + note drain one stall silently converts every later turn into a timeout. That is
  not enough when a turn STALLS rather than errors: open a fresh thread per measured turn and put a
  real-input CANARY after every risky one — a row is credible only if the canary behind it is healthy,
  so carry `canary_s` in the matrix and restart the server when it is not. Present judge candidates
  between explicit `[Cn]`/`[/Cn]` markers, never through `repr()` or quotes, which caps the
  format-contract score for the whole field at once. Interleave repetitions across configs, run configs
  sequentially so latency stays uncontended, and compare tiny gaps paired on the same (item, rep,
  judge) grading with a bootstrap CI.
- **L-027 — a deterministic content probe measures your regex until you read the misses.** Nearly every
  early miss was the pattern: spelled-out numbers, hyphens, synonyms, analyte variants, reversed
  negation order, legitimate alternate renderings. Two habits make the layer trustworthy — validate
  every pattern against an authored reference at corpus-build time and fail the build on mismatch, and
  treat each first-run miss as a suspect regex until you read the actual output. Keep every loosening
  synonym-only, so a wrong dose, side or negation still fails. **The mirror trap is worse: a PASSING
  pattern can hide a real defect** — `prednis|predonine` passes for two different molecules, so the
  declared probe and the aggregate scores both scored the wrong-drug turn as clean. Once a defect is
  named, write a check that SEPARATES it from the correct output and A/B against that check.
- **L-028 — a parallel-corpus reference is a judge-validity probe, never a translation ceiling.**
  FLEURS EN and JA are independent renderings of a shared source, so the JA routinely carries
  specificity the EN drops; scored blind, the human EN reached adequacy 4.255/5 against MT-over-perfect-
  transcript at 4.880. Report that gap as a corpus property, never as "MT beats human". The low human
  score is the payoff: a judge that docks fluent human English for diverging from the source is grading
  content, not fluency. String metrics against such a reference are bounded far below their nominal
  maximum for correct output (oracle chrF2++ 57.75) ⇒ use them to rank arms sharing one reference,
  never as an absolute level. When a "gold" candidate scores near the system under test, suspect the
  gold's provenance first.
- **L-022 — a claimed fix ships with a NON-VACUOUS lock, proven by neutralization.** Neutralize the
  fix (`if False and …`), watch the new test fail with the exact bug, restore. Two reusable
  techniques: **deterministic async-EOF** — buffer every response line AND `feed_eof()` in one
  synchronous burst before the read loop drains, since `StreamReader.readline()` returns without
  yielding when a line is already buffered, so EOF is guaranteed processed before the code under test
  reads liveness; and answer control requests reactively through `_read_loop` (`_await_pending(rid)` →
  feed the result line) rather than reaching past it with `set_result`, which exercises the real
  dispatch path. **A scripted neutralization matrix must drop bytecode between mutants** (`-B` +
  `PYTHONDONTWRITEBYTECODE=1` + delete `__pycache__`): CPython invalidates on (size, mtime-to-the-
  second), so two mutants a second apart that change a module by the same byte count silently reuse the
  earlier `.pyc` and certify untested predicates. **A second cause produces the same tell and the
  baseline does not catch it: restoring each mutant with `git checkout -- <file>` reverts the
  UNCOMMITTED fix too**, so mutant 2 onward mutates a line that no longer exists (`sed` matches
  nothing, silently) and every later row re-runs the un-fixed baseline. Snapshot the fixed file with
  `cp` after implementing, restore from that copy, make the mutation ASSERT that it applied, print a
  baseline and a post-restore run, and re-derive any surprising row by hand. Tells of a broken matrix:
  identical red lists across unrelated mutants, and a red set that contradicts reading the code. Record
  rejected findings with their reason so they are not re-litigated.
- **Per-character caption lag, never per-commit.** A commit carries several characters over an audio
  span and the reader waited longest for its first ⇒ `end` = `commit_audio_s`, `start` = the previous
  commit's endpoint, `at_i = start + (end-start)(i+0.5)/len(text)`, `lag_i = emit_s - at_i`, with
  `emit_s` on the virtual clock `now = max(now, buffer_end_s) + decode_s`. Two rules the trace's shape
  forces: a FINAL update ends at `buffer_end_s`, not its recorded `commit_audio_s`, because `update()`
  appends `processor.finish()` AFTER `process()` returned that timestamp; and **never derive lag from
  final updates alone** — that collapses every early in-speech commit into one utterance-close event and
  re-measures the VAD policy VAC exists to beat (finals-only reads 4.6× inflated). Qualifier:
  `commit_audio_s` moves BACKWARD on 6 of 157 commits (max 0.452 s), so those characters are placed at
  the previous endpoint and their lag is understated by that much.
- **L-016 — gitignored bulk corpora stay reachable by a script's runtime `open()`** even where the
  agent must keep them out of context. Construct such a path INSIDE the script rather than naming it on
  a command line, and reach the data through the existing manifests and tooling.

## On-demand commands

```sh
uv run python session_report.py [--log F] [--json]      # re-derive a live session from its own files
uv run python tests/eval_latency.py                     # per-stage latency budget, traces only, <1 s
uv run python tests/eval_term_census.py [--term T] [--floor N]   # term census + arms, no hardware
uv run python tests/eval_en_pairing.py [--live]         # what a real translator pairs; default <1 s
uv run python tests/eval_backpressure.py                # virtual-clock bounded/drop-free (silero+corpus)
uv run python tests/eval_cer.py                         # 2-engine CER + RTF baseline (models + corpus)
uv run python tests/eval_retention.py [--device D]      # retention CER on the shipped NPU/VAC path
uv run python tests/build_caption_trace.py              # caption stream over all six sections, ~8 min
uv run python tests/build_retention_probe.py            # rebuild the pause-free retention probe
uv run python tests/gen_replay_goldens.py               # regenerate goldens after a pipeline change
uv run python tests/eval_translate_repeat.py [--json F] # what repetition stalls a real turn (~15 min)
uv run --with soundfile python tests/eval_long_form.py [--sections 1,3] [--score]
uv run --with soundfile==0.14.0 --with pyarrow==25.0.0 python tests/fetch_real_clips.py
```

Whisper, NPU and GPU runs need the prelude first (`openvino-accel.md`).
