---
paths:
  - "live_stt.py"
  - "streaming.py"
  - "replay.py"
---

# ASR pipeline — capture → VAD → VAC → whisper → publication

## Capture, VAD, ring

- **silero opens a segment 0.2-0.7 s LATE and exposes no pad field. This binds EVERY engine, VAC
  included** ⇒ each segment is re-sliced from the fed-sample ring with `VAD_PRE_PAD_S`=0.4, on the
  legacy VAD-segment path and on the VAC buffer alike. Every VAD or `worker` edit preserves it
  (D-010).
- **`VoiceActivityDetector` owns two buffers with different drain rules; every feeder must pop.**
  Closed segments queue in `segments_`, each owning a copy of its audio, and `pop()` is the only
  drain the Python binding exposes (no `clear()`) ⇒ a consumer that never pops retains **~3.8 MB per
  speech-minute** (214 segments over 848 s = 39.4 MB, RSS +49.7 MB). `pop()` touches that queue alone
  — model, `start_` and sample buffer untouched — so `is_speech_detected()` stays bit-identical with
  a drain in place, which is why draining costs the VAC path nothing. Its OTHER buffer (the
  `buffer_size_in_seconds`=60 sample ring) pops on a NON-speech window only, so >60 s of unbroken
  speech logs `circular-buffer.cc:Push … Overflow!` and doubles it — lossless, bounded by the longest
  speech run, unrelated to the segment queue, and no drain prevents that line.
- `RING_SECONDS`=60 bounds the tested envelope: a VAD segment outliving the ring loses its unretained
  head.
- silero's `max_speech_duration` (20 s) is a SOFT cap — it raises thresholds (0.5→0.9,
  min_silence→0.1 s) instead of hard-cutting, so dip-less speech grows unboundedly and VAD tuning
  alone cannot bound segment length (L-023). The value × 16 kHz is an int32 sample count, so a huge
  sentinel overflows into spurious splits; the honest "cap off" control is a cap just above the audio
  length.

## VAC hypothesis buffer (`streaming.py`)

- `StreamingProcessor` = LocalAgreement-2 over repeated decodes of ONE open utterance buffer: commit
  the common prefix of two consecutive decodes, trim on wholly-covered segments, hold `emitted`
  **append-only**. The commit unit is the CHARACTER, because openvino.genai applies Whisper
  space-splitting to every language.
- Append-only is load-bearing: a shrinking `emitted` re-commits on-screen characters. Holding it
  append-only removed every insertion on the retention clip (I 12→0, CER 0.0686→0.0583). Committed
  characters must never be rewritten or duplicated once shown.
- **`whisper/long`'s golden pins a real duplication (`…に送ってに送って…`) — CLOSED by user ruling after
  two measured attempts. Do not re-derive or re-fix it.** Cause: `emitted` is a character COUNT
  re-derived from the latest hypothesis where it must denote what was PUBLISHED — `process()` pins
  `stable = max(agreed, len(self.emitted))` then assigns `self.emitted = text[:stable]`, so a decode
  that RE-SPELLS an already-published prefix at the same length re-anchors the boundary to earlier
  audio and `finish()` flushes `previous[len(emitted):]` a second time. Instrumented NPU replay of
  `long.wav`: 13 updates, **0 trims** ⇒ not a trim effect, and append-only covers the SHRINKING case
  alone. The policy is fine; the bookkeeping is wrong. Both fixes cost more than the artifact — a
  `startswith` commit guard starves `_trim` (max_buffer 11.248 → 23.9/25.5 s, **RTF 1.15**, above real
  time), and an audio-time cut at `finish()` traded 4 duplicated characters for 6 dropped ones
  (retention CER 0.0583 → **0.0635**, D 35 → 41; preserved whole on branch
  `attempt/p009-audio-time-cut` @ `bd37bf7`). A third attempt must first separate a genuine
  re-spelling from ordinary span jitter — 6 of 157 commits move backward, median 0.084 s / max
  0.452 s, against the ~0.7 s real re-spelling — and is a funded roadmap unit, never polish.
- The `on_update` seam (`worker` / `_vac_segments` / `replay.py`) is what makes commit timing
  observable at all; `commit_audio_s` is otherwise discarded at `commit, _ = await …`.

## Whisper on the NPU (D-016)

- **Policy.** Waiting for silero to close a segment bounds the first character's latency by the last:
  pause-free audio yields 8 segments of 20-33 s, lag median 15.5 s / max 36.6 s. VAC measured 2.5 s /
  8.1 s on the same audio and won CER on both corpora and both devices. Pure streaming without the
  VAD controller LOSES to plain VAD at the median on paused audio (3.18-3.74 s vs 2.43 s) — it bounds
  the tail, it does not beat pause-triggered emission; VAC is the synthesis. Shipped k2v2+VAD →
  whisper+VAC on NPU: long_form CER 0.2538→0.2321, retention 0.1587→0.0583, lag median
  14.239→2.483 s, for ~10× decode (RTF 0.041→0.48-0.61) and 1.6-2.1× real-time headroom.
- **Device.** The NPU costs nothing in accuracy (VAC CER 0.2321/0.0686 vs GPU 0.2292/0.0695) and
  forfeits exactly one feature: `StaticWhisperPipeline` raises `'initial_prompt' parameter is not
  supported on NPU device`, and the same for `hotwords`. Probed prompt lengths 1/2/4/8/16/32/64/128/
  200 all raise and 0 passes ⇒ a capability gap, not a length budget. `WhisperEngine.set_hotwords`
  drops the list per device; `--asr-device GPU` unlocks it. The NPU default is the user's efficiency
  choice.
- **Never feed prose back to the recogniser.** Recent transcript as prev-text measured CER **1.8919**
  against 0.1278 unconditioned on the pause-free clip — 2,126 insertions on 1,166 reference
  characters, the tail repeating one clause 7× — the third reproduction of prompt-induced looping.
  `CONTEXT_PREV_CHARS` and the carry buffer are deleted. A bounded TERM LIST on the same slot is
  loop-free (0.2408→0.1873), so terms ride `hotwords`; under the NPU default session context reaches
  the model through the translator glossary alone (`translation-leg.md`). Hotword figures used an
  ORACLE list drawn from the reference = an upper bound.
- **`repetition_penalty` is the ONLY repetition knob that reaches this build.**
  `no_repeat_ngram_size` is accepted and silently ignored (sizes 2/3/4/5/8 return byte-identical text
  on a clip repeating a 5-char unit ~140×); `num_beams=2` raises (`zero_remote_tensor.cpp:259`). The
  knee is sharp — 1.15 leaves a 510-char loop intact, 1.2 takes it to 0, 1.25 buys nothing more.
  `return_timestamps=True` markedly reduces (never eliminates) looping. Shipped
  `ASR_REPETITION_PENALTY`=1.2 costs 3 substitutions in 1,166 characters ⇒ retention CER **0.0609**
  shipped, 0.0583 penalty-free.
- **`WhisperPipeline` LATCHES its language for the life of the instance.** `generate(language=…)`
  persists into every later call, and an auto-detect call latches the detected language too. Neither
  `language=None` nor `''` clears it (both raise), nor `set_generation_config()`, nor a positional
  config. **Only a fresh pipeline re-detects**, so any per-utterance LID gate costs one construct per
  utterance (0.46 s p50 / 0.60 s max construct + 0.54 s detect, RSS flat at 201 MB over 40). LID is
  reliable on the NPU from 1 s of audio, but silence and −30 dB noise both detect as `en`. **Measured
  and ruled out by the user — do not re-propose.**
- `ASRDecodedResults` fields: `chunks, language, perf_metrics, scores, texts, words`. There is no
  `og.WhisperDecodedResults`.
- **The 0.35-0.42 s fixed decode term is structural in genai 2026.3.1 — three levers are closed at
  the source, not untried.** The feature extractor always pads to 30 s = 3000 mel frames and the NPU
  reshape fixes batch/features while leaving the model's time axis alone ⇒ **a shorter encoder window
  is unreachable through public config**, which is why the encoder measures FLAT at 0.31 s from a
  1.0 s buffer to a 28.0 s one. Whisper validation asserts `!is_assisting_generation()` ⇒ **no
  speculative/draft decoding**. And `"NPU"` now defaults to the **stateful** implementation
  (`STATIC_PIPELINE=false`), yet `whisper_generate` calls `decoder->reset_state()` after every audio
  chunk ⇒ **no KV reuse across the growing buffer's repeated `generate()` calls**. Do not re-derive
  these; the open candidates are the two constructor properties in `.agent/deferred.md` → *Probe the
  two open NPU constructor properties*.
- **Repetition-loop cause + free repro.** The recogniser is pinned to Japanese, so audio it cannot
  account for is emitted as Japanese tokens until `max_length`=448 — the 444-char live captions. The
  trigger is neither laughter nor room tone: synthetic non-speech (digital silence, −60 dB noise,
  60 Hz hum) reproduces the hallucination PHRASES (`ご視聴ありがとうございました`) but not the loop,
  while **English speech loops it every time**. Repro without an artifact: `curl -sSL -o
  .scratch/jfk.flac https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac` (11 s, public
  domain), resample 44.1k→16k, pad 1 s of silence each end, `replay.py --engine whisper` ⇒ segment 3 is
  a 528-char loop at **RTF 1.106** — a runaway costs more than real time. `soundfile` is absent from
  `.venv`; add `--with soundfile` to read the FLAC.

## Publication screen — `caption_defect()`

- The screen runs at PUBLICATION, upstream of every consumer, so `observe_ja`, the translator queue,
  `_turn` and `_failures` cannot see a defective caption by construction and a runaway streak of any
  length costs no strike. `CodexTranslator.submit`'s identical screen is the BACKSTOP ⇒ on the
  shipped path `tskip=` stays 0. **`tskip=N` is a CONTENT decision, never backpressure.**
- Thresholds, corpus-picked against 1073 live JA captions over 6 sessions (`transcripts/*.txt` is
  gitignored, so these numbers are the durable record; `session_report.py` re-derives every one):
  `CAPTION_REPEAT_UNIT_CHARS`=13 · `CAPTION_REPEAT_MAX_CHARS`=40 · `CAPTION_LATIN_RATIO`=4. Combined
  drop rate **4.3 %** there, **3.26 %** re-derived over the 1409 captions of 7 sessions — the same 46
  drops, because the 336-caption session that followed `ASR_REPETITION_PENALTY`=1.2 contributed none
  (largest span in it 14, one caption ≥13).
- **The unit bound is bounded by the REPEAT COUNT, not by the phrase length.** A drop takes
  `ceil(40/size)` repeats ⇒ 13 is the last size needing FOUR (3×13=39) and 14 lets a TRIPLED phrase
  drop; at 20-22 a mere doubling drops, and a live caption pays for it (a 22-char sentence doubled,
  then a unique third one). In tree the longest repetition is 18 over 215 NPU captions + every golden
  + the Aozora reference (10.7 K chars): `、うなぎが食べたい`×2, a speaker saying a phrase twice — a
  9-character unit, so it is visible from bound 9. Sweep over the 1073: 26 caught at 8, 27 at 9, 28 at
  12, **29 at 13**, flat to 21, 30 at 22. The largest repetition any SPEAKER produced is **20**
  (`リソース?`×4).
- **`CAPTION_REPEAT_MAX_CHARS`=40 is CLOSED — measured, then REFUSED.** Re-derived over 1409 captions
  / 7 sessions, the entire adjudication surface is 4 captions with 21 ≤ span < 40. Three are drops
  worth making: `イメージの質問は、`×4 (s1 n=213, span 36) and `翌日は翌日です`×4 (s1 n=197, 28) are
  wholly degenerate captions, and `3 months`×4 (s1 n=206, 32) the latin rule already drops. The
  fourth refutes the row's premise — `つもい`×10 (s6 n=103, 30) is a 30-character loop sitting inside
  **301 characters of genuine speech**, and the screen drops WHOLE (never truncates), so any bound
  reaching it costs 271 real characters. P-022 read that caption as a decode loop; the loop is real
  but the CAPTION is not, and "is the repetition a loop" is therefore the wrong question to adjudicate.
  Admissible band = **31..36**: ≥31 to spare n=103, ≤36 to catch n=213. **The tripling invariant
  floors the bound at 40** — a drop needs `ceil(bound/size)` repeats, so `unit*3 < bound` is what
  keeps a 13-character phrase said three times out of the screen
  (`test_a_phrase_said_three_times_survives_at_every_size_the_screen_scans`), demanding ≥40.
  31..36 ∩ [40,∞) = ∅ ⇒ no bound satisfies both. Forgone by refusing: exactly ONE caption, n=213.
  Re-open only by redesigning the screen to count REPEATS per unit size rather than characters, which
  is a different screen, not a threshold move.
- Latin ratio: the 23 latin-dominant live captions split cleanly — 17 true English at ≤0.15
  Japanese-per-character, 6 Japanese-carrying-loanwords at ≥0.27, nothing between. A 1:1 rule
  (`latin > japanese`) drops **6 genuine Japanese captions**, because a Latin letter is one phoneme
  where a Japanese character is a whole syllable, so one loanword outnumbers the kana around it.
  `CAPTION_LATIN_RATIO`=4 cuts the gap at 0.20.
- **Utterances stay UNCAPPED (user ruling).** One utterance = one line = one turn, whatever its
  length: a caption publishes only at utterance end and `VAD_MAX_SPEECH_S`=20 is a soft cap, so
  clean-caption p99 is 136-312 characters ≈ 18-40 s of speech and the live max is 664 ≈ 88 s.

## Sherpa fallback path (D-010)

- Both engines TTFT ≤0.10 s post-audio, $0/hr. **k2v2 is the default WITHIN the pair** (RTF 0.054 vs
  0.106, 148 MB vs 625 MB, Apache-2.0, JA-specialist, kanji-rich output). **Prefer `--engine parakeet`
  when a fallback run is chosen for accuracy** — Common Voice CER 8.426 % vs 8.953 %, FLEURS 10.445 %
  vs 13.664 %, and judged EN adequacy +0.126 [+0.019, +0.227] with severe failures nearly halved;
  counter-signal, parakeet's `confident_error` rate is higher (0.260 vs 0.229) — fewer wrecks, subtler
  substitutions. The k2v2↔parakeet DEFAULT question is retired, not pending: D-016 made both engines
  non-default fallbacks.
- **Raw long segments collapse offline decode — this path only.** Segment LENGTH (>~15-20 s
  pause-less), not synthetic-vs-real audio, causes wholesale deletions (L-023), engine-dependent
  (k2v2 collapses ~3-4× harder than parakeet). Fix: keep 20 s endpointing, preserve ≤10 s decode input
  by identity, and split only longer slices into balanced ~2 s low-RMS views with small overlap and a
  conservative exact-text merge. Gate total CER alongside deletion, or overlap duplication games the
  deletion target. Lowering the soft cap to 5-12 s still leaves 9-16 s blocks and adds harmful reset
  boundaries. VAC sidesteps the whole failure structurally by re-decoding a bounded open buffer.
- sherpa online gotchas: `from_transducer(enable_endpoint_detection=…)` surfaces as config
  `.enable_endpoint`; `OnlineRecognizer.reset(stream)` resets by side effect and returns `None`
  despite its bool annotation — never branch on that return.

## Shutdown, streams, meter

- **Closing the terminal is a shipped shutdown path and cost three separate fixes.** SIGHUP's default
  action is termination, so it must join `(SIGINT, SIGTERM)` or the drain never runs and exactly one
  EN line dies, the last (JA flushes as it lands; EN needs the drain). The drain then executes against
  a pty whose master is gone, where a write raises `OSError` errno 5 ⇒ `emit_line` persists to the
  transcript BEFORE stdout, and `write_stdout` latches the stream off on `(OSError, ValueError)`.
  Third and least guessable: **CPython flushes `sys.stdout` during finalization and that flush fails
  the same way, exiting 120** on an otherwise clean shutdown ⇒ the latch also swaps in `os.devnull`.
  stderr is not implicated: its handler flushes per record, so finalization finds nothing buffered.
- **L-006 — TTY-gate the cursor-clear (`\r\x1b[2K`) protocol per stream, evaluated once.** stderr: the
  `logging.Formatter` prepends it gated on `sys.stderr.isatty()`, the rare ~10-line subclass that
  beats inlining. stdout: the meter status line and `emit_line` gate on module-level `_STDOUT_TTY`, so
  redirected stdout stays ANSI-clean and the meter DRAWS nothing off-TTY; any new stdout status writer
  gates the same way. Gating the writer is not gating the information — the meter's counters exist
  only there, so off-TTY they move to stderr as high-water marks (`backlog peak: …`, sampled every
  `METER_LOG_INTERVAL`=1 s, logged only when a peak moves) rather than being dropped.
- **L-010 — keep device-backend imports at device entry points.** PortAudio probes host audio controls
  during `sounddevice` import, so an eager app-module import can hang model-only evaluators on an
  unhealthy device; `run_session` + `--list-devices` own those imports and offline replay/test stays
  hardware-independent.

## Latency budget — per stage, `tests/eval_latency.py`

Every figure re-derives from `vac_decode_trace.json` + `en_pairing_trace.json` in under a second, no
hardware. `retention_probe` (182 s pause-free) is the demanding clip; `stress_long` (44.7 s) runs
0.1-0.2 s cheaper on every row.

| stage | p50 | p90 | max | what it is |
| --- | --- | --- | --- | --- |
| update decode | 0.645 | 0.814 | 1.006 | one `process()` |
| commit lag | 2.535 | 4.600 | 8.157 | voice → committed character on the meter |
| provisional lag | 1.187 | 1.615 | 2.385 | voice → the same character shown UNCONFIRMED |
| redraw bound | 2.114 | 5.192 | 9.131 | upper bound: every redraw of a slot recharged as a fresh wait |
| publication | 1.173 | 1.444 | 1.444 | speech end → `JA n:` = `VAD_MIN_SILENCE_S` + final decode |
| translate turn | 2.330 | 4.190 | 6.200 | `JA n:` → `EN n:` (steady 2.085, rotating 3.900) |

- **Decode cost is `0.417 s fixed + 7.15 ms/char`** (`stress_long`: 0.360 + 5.77). The fixed term is
  Whisper's encoder over a 30 s window, measured FLAT at 0.31 s for buffers of 1.0 s through 28.0 s
  via `perf_metrics.get_encode_inference_duration`; feature extraction adds ~1.8 ms per buffer second
  and `return_timestamps=True` is free (`get_word_level_timestamps_processing_duration` = 0). So
  shortening the buffer touches the MARGINAL half only — 11.25 s → 5 s buys ~0.15 s — and 0.35 s is
  the floor under any update cadence.
- **The commit lag is a DISPLAY POLICY cost, not a compute cost.** `lag = audio-time holdback +
  decode_s`, and the holdback (p50 1.207 s) is LocalAgreement-2 withholding text until a second decode
  confirms it. The same decode already held that text: showing its unconfirmed tail costs nothing and
  takes p50 to 1.187 s, max to 2.385 s. `provisional_lag_s` is that arm, run on the same virtual clock
  and the same per-character placement as the committed arm, so their difference is the policy alone.
- **Both arms measure FIRST APPEARANCE ⇒ the gap is NOT a settled-text speedup.** Settled text still
  lands at 2.535 s p50 — `emitted` is append-only and the published line is unchanged — and the gap
  buys an earlier UNSETTLED rendering of the same character. `redraws` carries that cost: **105 of
  180 updates** on `retention_probe` (20 of 44 on `stress_long`) diverge from their predecessor
  before it ended, i.e. rewrite already-visible characters, which land in the dimmed tail by
  construction. `redraw_bound_s` is that cost charged as a fresh wait per redraw — a loose UPPER
  bound, double-counting by construction — and reads p50 2.114 / p90 5.192 / max 9.131. Quote 1.187 s
  for time-to-first-glimpse and 2.535 s for time-to-settled; neither alone describes the screen.
- **`VAC_CHUNK_S` is floored by decode cost, not by taste.** Work rate = `decode_s / VAC_CHUNK_S`:
  0.645 today, 0.86 at 0.75 s, **1.29 at 0.5 s** — past real time, so the audio queue never drains.
  Every candidate below 0.75 s needs the fixed 0.35 s term cut first.
- Translation is **2.330 s p50, not the ~1 s the tournament's 1.38 s implied** — that figure was a
  bare turn, this one is production order over 215 captions. 39 of those turns opened a fresh thread
  on a glossary change and cost 3.900 s against 2.085 s steady (`.agent/deferred.md` → *Cut the
  EN-leg thread-rotation tax*).

## Real-time cost — the instrument is CARRY (D-016(d))

VAC awaits each decode inside the coroutine draining `audio_q`, so unlike the sherpa two-stage worker
it does NOT feed VAD during decode — capture buffers into `AUDIO_HEADROOM_S`=2 s instead. Survivable,
not free: per-update NPU decode measured p50 0.552/0.645 s and max 0.764/1.006 s on the two pause-free
clips against a 1 s update cadence, with the trim rule capping the buffer at 11.248 s.

- **Aggregate RTF is the wrong instrument** — it shows mean compute below real time and says nothing
  about maximum blockage, which is what the headroom is spent against.
- **Carry is the right one:** a caption costing more wall time than its own audio hands the difference
  to its successor, and only that accumulation can outgrow a bounded queue (silence between captions
  drains further, so ignoring gaps is conservative). Over 215 captions of narration the worst carry is
  **0.017 s of the 2.000 s headroom** ⇒ the queue empties inside every utterance and its peak is one
  update decode. Knee: carry reaches the headroom at **×1.541** decode cost, reproducing the ×1.5
  ladder reserve on 4.4× the audio.
- **A caption's `decode_s` is a SUM of that utterance's update decodes, never one blocking call** —
  reading its 7.420 s max against `AUDIO_HEADROOM_S` reports a stall that did not happen. Never
  re-derive real-time risk from per-caption sums.
- Rerun cost varies ~20 % run to run, and the burst is machine state rather than a path property: the
  same clip/section/device at RTF 1.098 (`git show f25cfb5:tests/caption_trace.json`) carries
  **77.231 s** where the current trace carries 0.000 s.

## Known caveats

- Models are a runtime prerequisite — `check_models()` preflights and points at `models/README.md`.
- **The translator is NOT the quality bottleneck — the recogniser is.** Over 30 consecutive live
  JA/EN pairs the English is fluent and faithful to whatever Japanese it is handed, and every defect
  is ASR (`パーツ`→`パンツ`, `左肩甲骨`→`左肩骨`, `コロナル`→`セコロナルタ`, a stray `おやすみなさい。`
  hallucination mid-meeting). Benign default-engine quirks: ジェミニ→ゼミニ, 文→分 homophone — the EN
  leg translates through them.
