---
paths:
  - "tests/*.py"
  - "tests/*.json"
---

# Evidence artifacts — corpora, traces, evaluators

Fast locks run in the gate. **The ten `eval_*.py` scripts are ON-DEMAND and never gate steps**: five
need gitignored weights, a corpus or the real translator plus minutes of compute, so run one when a
decode change raises an accuracy question. `eval_latency.py`, `eval_two_way_settled.py`,
`eval_term_census.py`, `eval_en_pairing.py` (default mode) and `eval_lag.py` need none of that,
because they replay committed traces — a fresh clone runs them in under a second each.

## Committed artifacts and what each certifies

- `replay_goldens.json` + `gen_replay_goldens.py` — characterization snapshots, engine-first keying,
  asserting segment count + per-segment text + boundary (±0.1 s); CPU-variable decode latency is
  reported only. A cell the local machine cannot run (absent weights, WAV or accelerator) carries its
  committed row forward instead of vanishing, so a whisper regeneration on a box with no NPU cannot
  silently delete the committed whisper row.
- `short_corpus.json` + `fetch_real_clips.py` + `real_clips.json` + `test_corpus.py` — pinned Common
  Voice / FLEURS PCM corpus, compact evidence plus fail-closed locks. The real-recorded clips expose
  engine divergence the synthetic TTS corpus could not: 松井/松居, バック/パック, 午後七時/午後7時.
- `en_clips.json` + `fetch_real_clips.py`'s `en_us` arm + `test_en_corpus.py` — the pinned FLEURS
  **English** corpus, 647 clips / 6387.900 s of 16 kHz mono PCM16 under
  `spike/backends/cache/en_clips-v1-1b13a64fdc119f6c/`, `EN_EXPECTED_INDEX_SHA256` pinning the whole
  index. It exists because every other corpus here is Japanese, so an English-pinned decode had no
  scoreable input; FLEURS pairs it with the `ja_jp` side L-028 characterized, which is what lets a
  JA/EN arm differ in the language token and nothing else. One script and a second config, never a
  second copy — and `test_japanese_corpus_identity_is_unchanged` pins `EXPECTED_INDEX_SHA256`
  (`98e0d8a4…`) plus both JA manifests, so an English-side edit that moved the Japanese corpus turns
  red instead of re-qualifying it silently.
- `lid_census.json` + `build_lid_census.py` + `test_lid_census.py` — the LID decision table, reduced
  to what the three-part decision rule reads and nothing else: `view_fields` `[spoken, utterance,
  bucket, argmax, argmax_score, ja, en]` over 8,628 `views` (buckets `1s` 1925, `2s` 1516, `3s` 1453,
  `5s` 1188, `8s` 620, `VADfin` 1926) for 1,926 utterances cut by the production VAD from the two
  committed FLEURS corpora, plus 25 `synthetic` probes under `synthetic_fields`. Probabilities are
  the RAW 107-way softmax and are never renormalized over `{ja, en}` — renormalizing is the thing the
  rule rejects, so a renormalized census could not fire it. It is what lets the decision table run
  weights-free in a fresh clone, and its consumer is `tests/test_language_detector.py`, which replays
  the whole table through `lid_accept` and reproduces 1,338 correct / 178 abstain / 0 false at 2 s —
  compare that Counter to a Counter, since one that never incremented `false` compares unequal to a
  dict carrying an explicit zero for it.
  **It regenerates from committed state, and a green run IS the credit**:
  `uv run python tests/build_lid_census.py` re-cuts every buffer with the shipped `make_vad()`,
  re-scores every view through the shipped `LanguageDetector`, and compares bytes; `--clips N` is the
  bounded form the resource-gated lock rides, `--write` the deliberate rewrite. **A bounded run slices
  its oracle by UTTERANCE count, never by `len(actual)`** — the second spelling compares a truncated
  rebuild against its own prefix, so it passes on the regression it exists to catch — and the guard
  behind it is `corpus_clips` refusing a corpus shorter than the requested clip count. Full mode is
  weights-gated, so `test_lid_census.py` reaches it through an INJECTED two-clip corpus and a fake
  detector: without that lock every other lock in the file stays green with `census` replaced by a
  raise. The spike's own
  outputs (`models/lid/results/*.json.gz`, `models/lid/analysis.json`) are gitignored and no longer
  load-bearing. What they alone recorded was the ORDER, and the corpora define it: languages `ja`
  then `en`, clips `sorted()` by filename, buffers in cut order, buckets prefix-closed then `VADfin`,
  `utterance` dense over the buffers that exist — 5 EN clips open none. Credited against that dying
  analysis while it lasts: 12 of 12 bucket×language cells match `.d2.threshold_counts`, 0
  mismatches, and rounding to 6 dp moves no decision.
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
- `eval_two_way_settled.py` — the same trace under two-way's commit rule, which withholds every commit
  until the LID accepts a label. Four arms over both clips; the table and its refutation live in
  `asr-pipeline.md`. **Its own positive control is the no-withholding arm**, which must reproduce
  `eval_latency.py`'s `commit_lag_s` exactly (2.357 / 4.112 / 8.098 over n=277, 2.535 / 4.600 / 8.157
  over n=1135) — same clock, same per-character placement, same `_quantiles`, so a drift there means
  the replay diverged rather than the rule. The `never accepted` arm forces a path abstention does not
  take and is an upper bound only.
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
  model output ⇒ its verdicts are structural, never rates — a per-turn rate, a p90 or a latency
  ranking read off it describes that sample and binds nothing.
  **Rendering consistency is measured on the RAW stream, never off `renderings`.** `SessionContext`
  stores ONE rendering per term by construction, so any check reading it is tautological and stays
  green while the EN stream spells the term three ways. `raw_spellings` walks the recorded turns
  through a fresh context and reads `turns[*].en`, attributing on `observe_en`'s own gate MINUS the
  `t not in self.renderings` clause, which is what lets a paired term keep being read. On this trace
  no paired term carries more than one SUPPORTED spelling — `multi_spelled == []`, and ゴン
  `Gon`×8, 標柱 `Heijū`×9 +
  `Gon`×1, カスケ `Gon`×1 + `Kasuke`×1, 神様 `God`×2 over 8/38, 10/26, 2/4 and 2/3 attributable
  readings. **Read `readings` against `turns` as COVERAGE, never as disagreement**: a turn naming two
  trusted terms, or whose English carries two names, is unattributable, so the census is a lower
  bound on evidence. **And read a lone stray as NOISE**: the gate is a substring test on the JA, so a
  標柱-only caption whose English names `Gon` reads as one stray spelling of 標柱 — which is why the
  histogram ships with counts and why `supported` (`CONTEXT_EN_SUPPORT` agreeing turns) exists
  alongside `distinct`. `turns` counts a term only while it is already trusted, the population
  `readings` is drawn from; the trust-blind count is two higher for seven of this trace's eight terms
  and THREE higher for 二人, whose first sighting aged out of `CONTEXT_TERM_MEMORY` before the next
  one — read that delta as a per-term measurement, never as a constant.
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
  names: captions with no TGT and **why** (`declined`/`strike`/`disabled`/`shutdown`/`failed`),
  degrade + restore markers with timestamps, `backlog peak:` high-water lines and **drop
  attribution** — one row per `backlog peak:` line whose `drop=` grew, carrying the captions that
  bracket it, the publication gap between them, the `skip=` step, and every `caption dropped (…)`
  line inside that gap with its defect, which is the only record a screened caption leaves — caption
  length + repetition distributions, TGT-behind-SRC lag, and slow turns tagged with whether they sit on
  a `TRANSLATE_ROTATE_TURNS` boundary. `drop=` counts callback BLOCKS, so every drop figure here is
  blocks. Feeding it a drop timeline costs one redirect: the peak log gates on the stream PAIR, so
  `2> stt.log` keeps the status line too (L-006, `asr-pipeline.md`). **A transcript records captions, not the flags that produced
  them**, so the session's language rides `--source-lang`, which rebinds `live_stt.ASR_LANGUAGE`
  before anything is derived: re-running the shipped screen is the point, and an `en` session read
  under the default reported a latin screen the live run never applied. The drop split is derived
  from `caption_defect`'s own verdict (combined − repetition) rather than restating the latin rule,
  which is how it follows that flag. Three rules the live corpus forced: **a session owns log time
  from its own start until the next session starts**, never to its last caption, because
  `codex app-server exited` fires after the final caption by construction; **the run stamps that
  start itself** — `live_stt.log_session_marker` logs `session: <path>` under L-006's pair gate, so
  a marked log opens the window at the real process start instead of at the first caption, which is
  what `-o PATH` sessions lost (a free name carries no time), and a marker naming a transcript the
  report was not given owns a window with NO session, keeping `--no-save` events off the run before
  it; and **once the leg is
  down, `disabled` outranks the text screen**, since a screen verdict behind a degrade is a
  counterfactual and rides `screened_now` instead of explaining the loss. Over the six saved
  sessions: 1073 captions, 1001 translated, 72 without TGT; unit-bound sweep 26 caught at 6-8, 27 at
  9-11 (the newly-caught one being the 9-character loop that escaped bound 8), 29 at 13-21, 30 at
  22+; session 1 inferring a 3-strike degrade at last TGT n=194, session 6 reading
  `codex app-server exited` off the log at 14:38:49.
- `test_gate.py` locks `gate.py`'s step inventory with one seeded defect per blocking step, in a
  throwaway tree, plus the secret scan's real-tree scope, which that throwaway tree cannot see
  shrink (`toolchain.md`).

- `lag_sessions.json` + `build_lag_trace.py` + `eval_lag.py` + `test_lag_trace.py` — how far behind
  its source line a translation actually lands, on a real mic. `transcripts/` is gitignored ⇒ the two
  live sessions the claim rests on cannot be re-read once the recordings are gone, and this is the
  reduction that outlives them: per caption, integer seconds from session start for the source and
  target lines, the source and target character counts, and the session's `--` notes. **It is
  text-free by contract and `test_lag_trace.py` enforces that with an ASCII-only assertion over the
  whole file plus a closed key set per pair** — the user's own speech must not enter git, and a
  future edit adding a `text` field has to redden rather than leak.
  `build_lag_trace.py` imports `session_report.py`'s parser instead of restating the line grammar,
  and `--check` re-derives the committed bytes; both it and the agreement lock are transcript-gated
  with an `absent: ` reason, so a fresh clone runs the other four locks on committed bytes alone.
  Percentiles use `session_report.py`'s own convention (`statistics.median` for p50,
  `sorted(x)[int(len(x) * 0.9)]` for p90) so the numbers compose with the shipped report's.
  **Whole-second timestamps ⇒ every lag is quantised to whole seconds**; read a p50 of 2 s as a
  bucket, never as 2.0.
  Two limits the file states rather than hides. A caption with no target is EXCLUDED from the
  outstanding-work existential — it was abandoned, not left pending — because counting it marks every
  later caption backlogged for the rest of the session (session 2 reads 430 backlogged that way
  against 94 honestly). And the transcript cannot see that the leg was still retrying those captions,
  so caption 297 reads standalone while the leg was in fact wedged.
  `eval_lag.py` also carries the **staleness counterfactual that sized `TRANSLATE_MAX_STALENESS_S`**
  (`translation-leg.md`). Turns are sequential and FIFO, so queue wait replays from the trace:
  `turn_start(n)` is the later of caption `n`'s own source time and the target time of its
  predecessor-with-a-target, raised again by any note inside `n`'s own source-to-target interval.
  Two columns, never one — `arithmetic` uses queue arithmetic alone, `notes` admits the degrade
  markers as evidence that a turn could not predate them, and the gap between them at the shipped
  bound is exactly caption 297 (7 against 8 of 696). It prints THREE limits with the numbers, and a
  reader who drops any of them is over-claiming: whole-second timestamps; a wait reconstructed from
  publication times rather than measured in-process; and **no feedback modelled** — a skip moves
  every later turn start, and caption 296 likely reached a 23 s wait before spending the third
  strike, so the shipped bound may prevent the very cascade it is sized on. Sizing evidence, not a
  prediction.

## Rules that keep the evidence honest

- **A resource gate declares itself: `absent: <what>`.** Every skip in this suite gates on absent
  weights, corpus or accelerator, and the reason carries that prefix so `gate.py` can tell a live
  case from a demotion — a bare `pytest.mark.skip` or any xfail fails the pytest step
  (`assurance-posture.md`). Write the prefix into the `skipif` reason or the `pytest.skip()` call
  when adding a gate; demoting a real case needs a `.agent/deferred.md` row and the user's approval
  first.
- **The goldens and traces ARE contract-owned expected output, not a table smuggled past a contract.**
  `replay_goldens.json` is D-014's characterization snapshot and the committed traces are recorded
  measurements, so both answer "did the output change" for a contract that owns them. What the
  template's verification-integrity clause forbids is the opposite direction: an implementation that
  returns a fixture's expected value, detects the gate, or is graded against a table written to match
  whatever it already emits.

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
uv run python session_report.py [--log F] [--json] [--source-lang L]   # re-derive a live session
uv run python tests/eval_latency.py                     # per-stage latency budget, traces only, <1 s
uv run python tests/eval_two_way_settled.py [--json]    # two-way's commit rule vs time-to-SETTLED
uv run python tests/eval_term_census.py [--term T] [--floor N]   # term census + arms, no hardware
uv run python tests/eval_en_pairing.py [--live]         # what a real translator pairs; default <1 s
uv run python tests/eval_lag.py [--json]                # live TGT-behind-SRC lag, trace only, <1 s
uv run python tests/build_lag_trace.py [--check]        # rebuild that trace from the transcripts
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
