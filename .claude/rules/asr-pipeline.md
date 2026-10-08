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
- **Capture resamples through ONE stateful band-limited stream (`Resampler`, soxr HQ).** The mic
  runs at 44.1 kHz; the old per-block `resample()` (now `linear_resample`) had no anti-alias filter
  (12 kHz came through 2.1 dB down, folded to 4 kHz; 48 kHz took a bare `[::3]` stride) and floored
  every block's output (256 frames → 92 of 92.88 samples: −0.95 % duration, a phase jump per
  ~5 ms block). Cost 6.6 µs per 256-frame block. HQ holds a stop-position-dependent 6-40 ms in
  its filter (first output after ~40 ms); `run_session` drains it into the recording and the queue
  after `stream.stop()` joins the callback thread, ahead of the sentinel — without that drain a
  50 × 256-frame session reached the worker and the WAV 327 samples short
  (`test_a_44k_session_delivers_its_whole_stream_to_the_worker_and_the_wav`).
  `linear_resample` stays for the pinned corpora alone — `fetch_real_clips.py` and
  `eval_long_form.py` cut their SHA-256-pinned PCM through it — so every corpus CER here measures
  audio that passed the old recipe. `tests/test_resampler.py` (74 cases, tester-2's FFT oracle) owns
  length, block-size invariance, passband and ≥40 dB stopband. CER A/B through the NPU VAC path, 300 Common Voice clips (48 kHz mp3 → 44.1 kHz,
  −60 dBFS floor), N=6364, paired: old per-block 256-frame path **0.1131**, `Resampler`
  **0.1113**, 441-frame linear (no remainder loss) 0.1109; old − new = +0.0019, 95 % CI
  [−0.0055, +0.0094], 209 of 300 hypotheses identical (226 with equal edit counts) ⇒ no
  measurable CER change on clean read
  speech, shipped as the correctness fix it is (the old path delivered 29,473,319 of 29,755,200
  samples). Live meeting audio — far-field, fricative-rich, noisy — is unmeasured
  (`.scratch/perf/resample_ab.py`).
  **Never pad an evaluator clip with digital silence when comparing frontends**: the band-limited
  arm leaves ~1e-8 RMS filter residue there, silero opened a spurious ~0.6 s segment on it, and
  whisper hallucinated `ご視聴ありがとうございました` into it on 6 of 82 Common Voice clips (the
  per-block linear arm on 2, the 441-frame one on 0). Ordinary analog-mic capture is expected to
  carry a noise floor (a digital mute or gate could still deliver zeros); a −60 dBFS floor
  removed the spurious segment on all 5 affected probe clips (4 hallucinations, 1 emptied clip)
  and the 300-clip rerun shows 0 in every arm — evidence for that floor, not a proof that no live
  input can open such a segment.
- **`--save-audio` records at CAPTURE** (`AudioRecording`, opt-in): every resampled block lands in
  `transcripts/<start>.wav` (16 kHz mono int16, `× 32768` = `replay.load_wav_f32_16k`'s inverse) BEFORE
  the queue, so blocks backpressure later drops stay on disk — a superset of what the recogniser saw ⇒
  `replay.py` over it is never the live trajectory (replay never waits on `queued_samples`, so
  catch-up folds differ). Lazy; `wave.writeframes` re-patches the header per block (5.9 µs per
  85-sample block) ⇒ a crash leaves a playable file — once the zero-length priming write exists, since
  wave sizes its header to the FIRST write and skips that write's patch (first block buffered, file
  empty on disk; locked by an independent reopen after every block). Always `TRANSCRIPT_DIR`, never beside `-o`,
  which APPENDS across runs where a WAV cannot. Lock: `test_save_audio_records_every_captured_block`
  (fake mic through the real `run_session`).
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
- **The published boundary is ALIGNED, never counted (`StreamingProcessor._anchor`).** `emitted` must
  denote what was PUBLISHED, yet `process()` re-derived it as `text[:len(emitted)]`, so a decode that
  re-spells the published prefix — an inserted 、, a dropped particle, パック→バック — moved the boundary
  a character early (re-commit: `…に送ってに送って…`, らら/のの) or late (lost character). Mechanism
  witnessed on whisper-ja-760M's retention trace: all 10 of its boundary doublings sat one update
  after a silent re-anchor. The fix aligns the last `ANCHOR_TAIL`=24 published characters by edit
  distance against the text from where they should start (text end free, window ±`ANCHOR_DRIFT`=4);
  equal-cost ends — an edit at the very end, dropped vs inserted vs re-spelled — go to the end after
  which the text continues as the PREVIOUS decode did right after the published end, else the end
  nearest the old one; evidence never outbids cost, because a repeated short phrase (そうそう) puts a
  confirming continuation after a LATER repeat and taking it swallowed new speech. Below half
  agreement `_thin` decides (next bullet); a decode stopping short of the published end commits
  nothing.
  **A published mark the decode drops gives back the character it spent.** Where the published
  tail ends in a mark (punctuation, space), the last character actually published
  (`published_last`) is a mark too, the chosen end (evidenced first, then nearest) spends a word
  character on that mark, the end before it ties, and the tail minus the mark aligns one cheaper there, the boundary moves back one: a mark
  carries no audio, so a word character in its slot is new speech. Whisper drops or moves a
  published `。` between decodes, and the nearer end ate the next word's first mora — 10-02 live,
  whisper-ja-760M: 25 of 441 sentence joins plus 9 line starts (`。れどおりに` for `。それどおりに`,
  `。ゃあ`, `。ースの3人`); the count rule cut at the same place. Three guards (reviewers 3 + 4):
  `published_last`, because the record may re-spell a published word character as a mark (`ABC` →
  `AB。`) and giving back there re-committed `C`; the evidence pool, since a continuation evidences
  both tied ends of a repeated mora (`。こ` → `ここから`); a lone published mark skips `_thin`, whose
  count spent the next character (`。` → `それ` published `。れ`). **Scope (user ruling):** below half
  agreement over a longer tail `_thin`'s count stands ⇒ `AB。` → `XBそれ` still publishes `AB。れ`;
  giving back there lost 197 scenario scripts against 18 at seed 11, total drops and garbage no
  longer self-healing. Scenario harness (`markdrop` = the live shape, `markdrop_edit` = it plus one
  earlier re-spelled character), `--scripts 30000 --seed 7` against the prior processor: 47,508 won
  (24,156 of 24,156 + 23,352) against 201 lost — 197 `flip1`, where one decode re-spells the
  published mark as a word character and the next reverts, so the given-back character re-commits
  the mark (`ました。。兵十`), plus 1 `headdrop1`, 1 `totaldrop1`, 2 `garbage1`; 200 of the 201 repeat
  one mark (`。` 127, `、` 68, `?` 5), 1 two characters. Seed 11: 18 lost, 4,782 won. Refused variants:
  filtering every tied end before the evidence check (39 lost at seed 11); giving back at every tied
  end (re-committed whole phrases, `ていました。`). Retention CER 0.0532 and the whisper NPU golden
  unmoved. Locks: `tests/test_dropped_mark.py` (tester-2's 24 + 4 reviewer reproducers: 16 of 28 red
  on the prior processor, 4 on the reviewed shape).
  Trims fire on a commit, where count slicing WOULD have committed (so the committed trace keeps its
  whole trim schedule: 0 offset divergences, 1 + 3 commits changed on turbo's pre-swap trace), or past `buffer_trim_s` +
  `ANCHOR_STALL_S`=4 (12 s; traced buffers never pass 11.25 s) — without the last two an aligned
  empty commit starved `_trim` into a forced trim at 29 s, the old `startswith` guard's failure.
  Span whitespace is normalized in `process()` (`' ABC'` counted 4: a cut there retained D yet
  marked it trimmed → ABCDDEFG). Measured on the NPU through the shipped VAC path: turbo retention
  **0.0609 → 0.0592** (S36 D35 I0 → S37 D32 I0), long-form §01+§03 0.2546 → 0.2486; whisper-ja-760M
  retention **0.0626 → 0.0532** (I 21 → 11; the rest is one 11-char hallucination), long-form
  0.2213 → 0.2153; the `whisper/long` golden lost its `に送ってに送って` (that golden moved by the
  row's acceptance). The two earlier attempts stay binding evidence: a `startswith` commit guard
  starved `_trim` (max buffer 11.248 → 23.9/25.5 s, RTF 1.15), and an audio-time cut at `finish()`
  traded 4 duplicated characters for 6 dropped ones (retention CER 0.0583 → 0.0635; branch
  `attempt/p009-audio-time-cut` @ `bd37bf7`). **Ambiguous by construction, documented, not bugs:**
  a 2-3 character insertion right before the last published character reads as a 1-character
  re-spelling (edit distance prefers it); a terminal deletion whose continuation ALSO changed reads
  as a re-spelling. A re-spelling
  inside the RETAINED segment shifts the next piece boundary (`settled_boundary`) by its length —
  no published character lost or repeated. Locks: `tests/test_boundary_anchor.py` (tester-4, 36
  cases incl. a seeded 576-edit property) + six `test_streaming.py` cases, red on the old processor.
- **Below half agreement the boundary moves only once two decodes agree (`_thin`, user ruling).**
  The count alone dropped the speech after a re-spelling of most of the tail (reviewer-7: ABCDEFGH
  → XXGHIJK lost IJK; turbo's retention trace counted `があった森永の美味しい牛乳` 4 early, cancelled
  only by the next decode). Evidence = the ≤2 characters that followed the published end in
  `found_in`, the decode the record was last FOUND in — never a held or counted decode, whose
  continuation proves nothing; a counted record clears `found_in`, which keeps garbage plus a
  repeated phrase (チンチロリン、チンチロリン) from confirming a wrong end. It may sit at any end the
  tail reaches with ≥1 matching character (`cost < max(len(tail), j)`: a restored head costs
  insertions past `ANCHOR_DRIFT`); cost, then nearness, picks. The first evidenced decode is HELD —
  no commit, record kept, no ordinary trim (the `HARD_TRIM_S` force trim still fires) — and the
  next adopts the end by proposing the same re-spelled
  prefix, else the count stands; `finish()` adopts unconfirmed, no decode being left to confirm it.
  **Taking the end at once is REFUSED** (consultant-1 + consultant-2 BLOCK): it moves the record
  into a one-decode spelling, so the reversion, head drop or total drop after it re-commits
  published text. `tests/eval_anchor_scenarios.py --baseline <old streaming.py>`, then 11 scenarios ×
  ~2,870 scripts, seed 11: 0 scripts the old processor outputs exactly and this one does not —
  sampling-bounded: seed 0 finds 1, `--scripts 30000 --seed 7` 18 of 315,805 against ~64,000 the
  other way (user ruling: shipped, the contract's "no script" restated as these numbers); exact
  output old → new: persistent re-spelling 23.7 → 74.6 %, two-decode re-spelling 24.0 → 74.6 %,
  one-decode head drop 63.6 → 92.9 %, final-decode drop 51.7 → 91.2 %, total drop 99.9 → 99.9 %
  (the at-once rule: 50.6 %). **A synthetic ranking only**: the committed whisper-ja-760M clips
  reach the thin branch 0 times in 224 updates, frozen turbo 3 on the old processor and 2 now,
  and every recorded clip (760M + frozen turbo) replays byte-identical commits. Limits: with no
  located continuation, or a second decode proposing another prefix, the count stands (loss or
  repeat up to the re-spelled span); a held decode's dim tail is count-sliced for that one
  update. **Where the old count wins, documented, not bugs:** a same-length re-spelling holding
  the continuation earlier adopts the earlier end (ABCDEFGH, ABCDEFGHIJ, XGIJXXYYIJK ×2 →
  re-commits IJXXYY, reviewer-2), and so does an unconfirmed final decode (XGHIJXXXIJ;
  `1。これは私が小` → final `雑が小さい雑雑雑さい` publishes `さい雑雑雑さい`, reviewer-1), the count
  being right there only because the length held; a two-decode head drop can confirm a short
  re-spelled prefix and drop speech on restoration (`…カン、カンと鐘…` loses `カン、`, reviewer-1,
  seed 0). Locks: `tests/test_thin_rewrite.py`
  (tester-1, 49 cases: 24 locks red on the old processor, 25 controls green on both).
- **A decode re-telling the published end past the located boundary skips that run
  (`_past_retelling`, user ruling 10-08: attempt).** `_anchor` locates `emitted`, a record later
  decodes re-spell, so where record and screen part ways the located end can sit before text that
  re-tells the SHOWN end — 10-08 replay, 6 re-commits of 3-8 characters in 290 utterances: a
  re-spelling the next decode reverts (`やりやすくっていうていう`, live SRC 186; `ちょっといょっとい`),
  a head restored ahead of the published text (`そういうそういう`), a count off a garbage decode
  (`入れるっていうていう`). `shown` = what the buffer published, as published (sliced with `emitted`
  at `_trim`, cleared at `_force_trim`); `heard` = what the publishing decode heard past it. Where
  `_anchor` LOCATED the end — `_thin`'s confirmed or final end included, never a held, stopped-short
  or count-fallback one — the longest run of
  ≥ `RETELL_MIN`=3 characters ending `shown` that the text continues with is skipped, in `process()`
  and `finish()` alike, `emitted` keeping `text[:stable]`; it stays where the text spells it twice
  or a nonempty `heard` agrees with it over their overlap, the real-repeat evidence (`ごちゃごちゃ`
  with the first copy re-spelled for good; `ごち` heard for `ごちゃ`). Over the logged decodes
  (`.scratch/s1008/trace_replay.py`): 10-08, 5 of 290 utterances change — 4 re-commits gone (−14
  spurious characters) and 1 loss of 2 real characters where a count-drifted record had already
  mis-published (u139, `…あってるんですよね。人のワーク…` → `…。ワーク…`); every pinned clip
  unchanged (229 utterances). Scenario harness: seed 11 0 lost / 223 won, `--scripts 30000 --seed 7`
  0 lost / 1960 won. Refused on the way: keeping the record on a count fallback (real decodes 5
  better / 2 worse); a veto reading the previous decode alone (6 real-repeat losses at seed 7, the
  first copy re-spelled for good); requiring the record to end with the run (removes every real
  win). Residuals: a re-telling across a trim, a garbage decode and a restoration
  (`か?って書いてあ`, the one left), and a real repeat said after the publishing decode ended and
  re-spelled now reads as a re-telling. NPU, the 10-08 WAV through the shipped path: 290 of 290
  utterances and 408 lines, exactly those 5 changed; retention 0.0532 and long-form §01+§03 0.2153
  with every hypothesis byte-identical. Locks: `tests/test_retell_guard.py` (tester-2, 63 cases:
  the four replay shapes through `process()` and `finish()`, the next decode after a guarded one
  publishing no part of the run, real-repeat and located-only controls, state lifecycle across
  trims, a generated suffix oracle; 26 red on the prior processor, 21 with the guard neutralized).
- **A post-trim decode that re-tells the trimmed text loses that head (`_retold`).** Whisper puts a
  segment's `end_s` up to ~2 s early, so `_trim` keeps audio whose text it just moved out, and the
  next decode re-tells it ahead of the retained text — 10-02 live: `SRC 1173` → `1174` repeated
  `そういうことがあるらしいんですよね。` (post-trim `emitted` = `そ` sat at the head's first character,
  so the anchor found it and agreement committed the rest), `307`/`308`, `877`/`878`, `402`/`403`.
  **Evidence = the retained text alone** (`retained`, the trimming decode past the cut, kept until
  the next trim; one character counts): a later decode stopping inside the re-telling would vouch
  for it, and an empty retained text would gain evidence one decode later (reviewers 7, 8). On every
  decode until the next trim, a head `text[:h]` drops — with its segment characters, before
  agreement — where it re-spells the moved-out text's end (cost ≤ h/4), the retained opener's first
  2 characters follow it, and removing it aligns the retained opening (`RETOLD_OPENING`=8) better
  than keeping it by more than its own cost. So a repeat the trimming decode heard heads the
  retained text and stays, re-spelled included (`明です。` for `説明です。`); a repeat said later
  follows the retained text and stays; a later recurrence of the opener keeps the speech before it;
  an empty retained text is no evidence and the decode stands. One `head_costs` table prices every
  head: per-head alignment cost 4.3-6.9 s per decode at 440 cut characters, the table ~0.03 s.
  The trimming decode already holding the second copy is `_echo`'s (next bullet). **Residuals:** a
  real repeat the trimming decode dropped and the next one restores reads as a re-telling
  (ambiguous by construction); a re-telling stands where the decode re-spells the retained opener's
  first 2 characters, or where dropping it gains no more alignment than it costs. Locks:
  `tests/test_trim_republish.py` (43 cases: tester-3's 35 — the four live shapes, a head lasting to
  the next trim, an empty decode in between, `finish()`, the worker's `SRC` lines, repeat + scope
  controls — and 8 from the reviewer reproducers; 16 red on the prior processor, 8 on the first fix
  shape). Real audio through the shipped NPU path (long, stress_med, stress_long, retention_probe,
  gongitsune §01-§06): 32 trims, no re-telling decode, retained text never empty; retention CER
  0.0532 and long-form §01+§03 0.2153 unchanged, every hypothesis byte-identical to the prior
  processor's.
- **A trimming decode spelling one phrase twice across its last segment boundary keeps the copy the
  audio holds (`_echo`, user ruling: audio-verified).** Whisper ends the penultimate segment with
  the run the last segment opens with, `final_s` commits the first copy, the trim cuts between the
  copies and both publish (10-08 replay of `transcripts/2026-10-08T14-02-20.wav`: SRC 117, 162, 166,
  322, the 162 shape also live as SRC 151/152; 10-02 `9`/`10`, `290`/`291` read the same, no audio
  kept). Text cannot separate it from a phrase said twice, so the audio decides: past
  `buffer_trim_s`, where the penultimate segment ends with k ≥ 2 characters the last opens with and
  the decode ends `run + last`, `process()` decodes `audio[:penultimate end_s]` once more; the
  first copy stays where that decode holds the run within `len(run)//4` edits inside its last
  `len(run) + ANCHOR_DRIFT` characters, else it leaves `text` and the penultimate span before
  agreement, and the drop licenses a cut as a commit does (the dropped copy was SRC 117's whole
  commit; uncut, every later update re-decoded the doubled audio and paid the probe again) — a
  held or stopped-short anchor, or a `_trim` finding no cut, still leaves the buffer uncut.
  Measured: the pre-cut decode heard the run in 0 of the 4 doubled trims and in 7 of 7 constructed
  real repeats (each buffer plus its own post-cut audio, ≤ ¼ edits — the bound is sized on those 7,
  not an operating point); the trigger fired on exactly those 4 of 172 trim-eligible updates in 57
  minutes and on 0 of 35 across every pinned clip, so retention, long-form and the committed traces
  are unchanged by construction (`.scratch/s1008/trigger_census.py`). Cost = one extra decode per
  trigger, drain blocked, estimated ~0.6 s off the latency table below (unmeasured).
  `build_vac_trace.py` asserts one decode per update, so a rebuild whose clip triggers stops there.
  Residuals: a doubling split anywhere but the last boundary stands, as does a duplicate whose probe
  decode happens to end on the run; a real repeat whose probe re-spells it past a quarter of its
  characters loses its first copy. NPU, the 10-08 WAV replayed through the shipped path: 290 of 290
  utterances and 408 lines, exactly the four doubled ones changed, each phrase once; retention
  0.0532 and long-form §01+§03 0.2153 with every hypothesis byte-identical. Locks: `tests/test_trim_echo.py`
  (tester-1, 67 cases: the four replay transitions, the worker's `SRC` lines with and without a
  translator, the 7 measured repeats, probe placement and budget boundaries, the final decode; 38
  red with `_echo` neutralized); `test_trim_republish.py`'s scripted decoder answers the probe with
  the span's own text, every repeat there being a real one.
- The `on_update` seam (`worker` / `_vac_segments` / `replay.py`) is what makes commit timing
  observable at all; `commit_audio_s` is otherwise discarded at `commit, _ = await …`.

## Whisper on the NPU (D-016)

- **The M10 candidate screen is CLOSED and stays closed.** Old multilingual streaming zipformer +
  SenseVoice lack current JA evidence, Moonshine-JA's license is unclear, and ReazonSpeech-k2-v2 adds
  PyTorch/Transformers + remote custom model code. Re-open only if the shipped path FAILS and the
  added runtime surface buys a materially different hypothesis — a re-open condition, never queued
  work. Tournament record → `.agent/archive/m10-asr-tournament.md`.
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
  with it, 0.0583 penalty-free (both turbo, before the boundary fix and the 760M swap).
- **`WhisperPipeline` latches OMITTED language and nothing else; an EXPLICIT token always wins.**
  That split is the whole architecture of two-way, so read both halves. `generate(language=…)`
  persists into every later call and an auto-detect call latches what it detected, so a call that
  OMITS `language` inherits the latch instead of re-detecting — 3 of 3 reps on JA audio through an
  en-latched instance. No reset exists: `language=None` raises
  `Check '!value.is_none()' failed`, `language=""` raises `Check 'lang_to_id.count(*language)'
  failed`, a fresh `WhisperGenerationConfig` raises `ValueError: vector::reserve`,
  `set_generation_config()` and a positional config do not clear it either. But passing
  `"<|ja|>"` to that same en-latched instance returns correct Japanese at **0.983 s against a
  0.971 s cold control** ⇒ switching an existing pipeline explicitly is free, and the design that
  follows is an explicit token per decode, chosen per utterance by a standalone LID — on one
  resident pipeline per MODEL (whisper-ja-760M for `ja`, plus turbo for `en` under `--two-way`).
  Evidence `.scratch/latch-probe.json`, `.scratch/latch-reset.log`.
  What is refused is **whisper's OWN detector as the gate**: it needs a fresh pipeline to re-detect
  (0.46 s p50 / 0.60 s max construct + 0.54 s detect, RSS flat at 201 MB over 40), it is reliable
  from 1 s of real audio, and silence and −30 dB noise both detect as `en`. **Measured and ruled out
  by the user — do not re-propose it, and do not read that ruling as covering a standalone detector
  that hands whisper an explicit token.**
- **Every decode on the VAC path names its language, so the latch is unreachable by construction.**
  `WhisperEngine.generate(language=…)` takes the token for THAT decode and falls back to
  `ASR_LANGUAGE` when the caller names none; `decode_segments()` forwards it, being the callable
  `StreamingProcessor` holds. Three things this got wrong once each, all now load-bearing:
  - **The session fallback is `ASR_LANGUAGE`, never the literal `ja`.** One-way `--source-lang en`
    constructs no detector, so the held token is what every buffer of the whole run decodes under;
    a literal would have decoded English under `"<|ja|>"` and broken transcribe-only mode silently.
  - **The processor's decode callable SNAPSHOTS the token at construction**, one closure per
    processor, so a discarded processor cannot switch language. A single closure reading the mutable
    token at call time is correct only while nothing ever decodes through a replaced processor —
    a guarantee that holds by accident rather than by construction.
  - **`token` is assigned BEFORE each `StreamingProcessor` build, at both sites.** The snapshot makes
    binding order load-bearing: building the replacement first freezes it on the token the rebuild
    exists to abandon, which reverts the whole switch to `ja, ja, ja, ja` with no error anywhere.
  **Every in-repo stand-in for `decode_segments` accepts that keyword, and a wrapper over the real
  method forwards it.** Two of them are invisible to the gate and fail only when someone reaches for
  them — `tests/eval_backpressure.py`'s trace recognizer, whose VAC arm is corpus-gated, and
  `tests/build_vac_trace.py`'s recording wrapper, which regenerates `vac_decode_trace.json` on the
  NPU. Widen those with the rest.
  Retention CER re-derived on the NPU across the change: **0.060891938250428816 before and after**,
  N=1166, S=36, D=35, I=0 over 8 segments, and the hypothesis text compares byte-identical ⇒ routing
  the token per decode moves the shipped one-way decode by nothing at all. Decode COST is not pinned
  and moved within the known machine-state band (109.42 → 95.26 s total, RTF 0.600 → 0.522).
- **A wrong language token is CATASTROPHIC in both directions and graceful in neither.** 150 FLEURS
  clips per language on the shipped NPU whisper, both tokens on the same audio, scored with `cer.py`
  (`.scratch/en_quality_probe.py`, result `.scratch/en-quality.json`):

  | audio | `<\|ja\|>` CER | `<\|en\|>` CER | ref chars | decode p50 |
  | --- | --- | --- | --- | --- |
  | JA (`fleurs-ja-`) | **0.0502** | **2.0552** | 7,593 | 0.92 / 0.78 s |
  | EN (`fleurs-en-`) | **0.6710** | **0.0263** | 15,390 | 0.85 / 0.82 s |

  Two things follow. **English is the recogniser's better language** — 0.0263 against 0.0502, so
  nothing about the EN direction is blocked by decode quality. And a misroute costs **41×** on JA
  audio and **26×** on EN audio, the JA row past 1.0 because a wrong token hallucinates insertions
  rather than degrading ⇒ an LID that guesses under uncertainty is worse than one that abstains and
  holds the previous token. The JA row reproduced to the digit across two independent runs, which is
  the determinism this table rests on; decode p50 moved ~20 % between them, the known machine-state
  band.
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
  these.
- **The two constructor properties are MEASURED and REFUSED; the fixed term stands.** Method, because
  the naive one cannot work: decode varies ~20 % run to run, so a per-arm p50 cannot see the tens of
  ms at stake. Caption text and VAD boundaries reproduce byte-identically across NPU runs ⇒ update k
  of one arm decodes the same buffer as update k of every other, making the per-update delta a PAIRED
  sample; 3 interleaved reps × 180 updates on `retention_probe`, one arm per process. **Decoded text
  was byte-identical across every arm and rep**, so CER is unmoved by construction and needed no
  re-derivation.
  - `NPU_TURBO=true` — median p50 0.5453 → 0.5329 s, paired median **−2.3 ms** per update (mean
    −14.9 ms, p10 −35.2 / p90 +24.1 over n=540). That is 0.4 % of an update, inside the paired
    spread, and it costs a full cold recompile (~126 s) on first use. **Not adopted.**
  - `NPUW_LLM_GENERATE_HINT="BEST_PERF"` — cold compile succeeds (95.9 s) and decodes, then loading
    the resulting CACHED blob **SIGSEGVs, 3 of 3 reps**, with no Python traceback. A property that
    cannot survive its own cache is unusable in a tool that constructs at every startup. **Not
    adopted.** Set beside `NPU_TURBO` it loads fine and buys nothing (paired median −0.8 ms).
  - Baseline note: base measures 0.5453 s here against the committed trace's 0.645 s. That is machine
    state (the ~20 % band), not a regression — never re-baseline a committed trace from a scratch run.
- **Two processes cold-compiling the SAME cache key concurrently write a blob that SIGSEGVs on load.**
  Cost 3 wasted arms and two crashes before it was identified. `OPENVINO_CACHE_DIR` is fully
  regenerable, so the repair is `rm -rf models/openvino/cache`; the guard is to never run a second
  whisper process against the cache while one is compiling. Concurrency also inflates decode itself —
  two pipelines on this one NPU measured 0.564 → 1.218 s per update before segfaulting.
- A resident whisper large-v3-turbo int8 pipeline costs **~2.0 GB RSS** (2253 MB held, 225 MB after
  release), which is what a design holding two of them at once has to budget.
- **Two resident pipelines cost +2015 MB and +0.050 s per update, and buy nothing the explicit token
  does not.** Measured on this machine: VmRSS 2237 → 4252 MB and VmHWM 3870 → 5885 MB when the
  second pipeline is constructed and compiled, and alternating decodes between them costs a paired
  median **+0.050 s per update** against a single-pipeline control, ABBA drift-cancelled
  (`.scratch/coresidency_probe2.py`, `.scratch/coresidency2.json`). Run 1's +0.224 s figure was not
  drift-cancelled and is **RETRACTED — never quote it.** Since one pipeline switches language
  explicitly for free, a second copy of the SAME model buys nothing. A second, DIFFERENT model is
  what the JA-tuned checkpoint funds: whisper-ja-760M decodes Japanese and roughly doubles English
  CER (FLEURS-en 0.0255 → 0.0482), so `--two-way` holds turbo for `<|en|>` decodes
  (`WhisperEngine(..., english_dir)`, user ruling) at about these costs — measured for two turbo
  copies; the mixed 760M + turbo pair is unmeasured. One-way builds one pipeline, and
  `--source-lang en` builds turbo alone.
- **A standalone spoken LID settles that token, and its operating point is measured: ECAPA
  VoxLingua107 on ONNX Runtime CPU, first decision at 2.0 s.** 21M params / 86.657 MB, Apache-2.0,
  an end-to-end raw-PCM ONNX graph with FBANK + per-utterance CMVN folded in, so it shares nothing
  with the NPU recogniser:

  ```sh
  mkdir -p models/lid/d2-ecapa
  base=https://huggingface.co/crash-sv/scribe-ecapa-voxlingua107/resolve/13a135951a7352386984030d35531563f3473aaa
  for f in voxlingua107.onnx lang_map.json manifest.json; do
    curl --fail --location --retry 3 "$base/$f?download=true" -o "models/lid/d2-ecapa/$f"
  done  # voxlingua107.onnx sha256 e2c3c3da39b99e3f9196d15fceef6a65f702320038bbc08813a4f21280255ce8
  ```

  **The decision rule has three parts and no fourth.** The global 107-way argmax must itself be `ja`
  or `en`; its score must be ≥0.35; the absolute `ja`-vs-`en` margin must be ≥0.35. Never
  renormalize over `{ja, en}` — the global argmax IS the non-target rejection, and it rejected 25 of
  25 synthetic silence, noise and hum probes. An absolute cap on the best non-target score changes
  no cell after that argmax and is deliberately not in the gate.
  **Score nothing before 2.0 s of voiced buffer.** At that gate, over the 1,030 JA + 896 EN
  production-VAD buffers cut from the two committed FLEURS corpora: **0 false routes in 1,516
  two-second views**, 1,338 correct, 178 abstentions (11.74 %); held out on the hash-parity split
  the thresholds never saw, 690/776 correct, 0 false, 86 abstain. **1 s is refused and no threshold
  rescues it** — one EN buffer routes to `ja` carrying score 0.9822 and margin 0.9822, and 1 s
  accepts only 933 of 1,925 at all. Retry each later prefix that exists, stop at the first
  acceptance: first-correct lands at 2 s for 1,338 utterances, 3 s for 120, 5 s for 20, 8 s for 2,
  VAD-final for 37, and **409 of 1,926 never accept**. Label flips between accepted prefixes of one
  utterance are **0** once 1 s is suppressed (0/4,528 adjacent pairs, 0/4,535 across a held
  abstention) ⇒ freezing the first accepted label costs nothing measurable here.
  **Abstention is the common case on short speech and its fallback is UNPROVEN.** 410 of 1,926
  VAD-final buffers are shorter than 2 s and the gate accepts only 28 of them, abstaining on 93.17 %,
  so "hold the last accepted label, JA at startup" carries roughly a fifth of utterances. This
  corpus cannot score that rule: it holds no bilingual sequence and therefore no switch rate.
  Holding beats a coin guess only where language persistence exceeds 50 %, and beats forcing the
  pairwise winner only where the switch rate among abstentions stays under **3.93 %** — forcing
  those 178 costs 7 false routes.
  **Cost 27.415 ms p50 at 1 s, 137.559 ms p50 at VAD-final** (p90 29.257 / 239.959) on 4 intra-op /
  1 inter-op threads, 184 MB RSS at 1 s and 434 MB on the longest 22.384 s buffer.
  **At the shipped 2.0 s gate: 35.6 ms**, the minimum over 6 runs × 60 reps under onnxruntime 1.30.0
  `CPUExecutionProvider` on a seeded buffer, 245 MB RSS. Read the minimum as the cost and the spread
  as contention: this box always carries other work (the session's own proxy holds ~1 core, loadavg
  5.05-8.29 of 8 across those runs), which put p50 at 35.6-57.3 ms and stretched p90 to 132 ms. The
  same harness reproduces the recorded 1 s p50 inside 21.6-29.6 ms, which is what credits the
  comparison. The shipped `LanguageDetector.score()` re-measures **33.6 ms min / 34.8 ms p50** over
  40 reps on a real VAD-final 2 s prefix once the box is quiet ⇒ read the gate's cost as ~34-36 ms
  and the rest of any spread as load. Co-residency with NPU whisper is a different question and
  stays unmeasured, below.
  **The runtime is ONNX Runtime `CPUExecutionProvider`, not OpenVINO**, against this repo's usual
  preference: the same graph under OpenVINO CPU retains shape-specialized state to a 2,418 MB peak,
  and exact `GPU.0` is 7.5 ms at a fixed 1 s but recompiles variable VAD-final shapes into a 1.68 s
  p50. Both rejected alternates, so neither is re-priced: NVIDIA AmberNet (29M) matches the accuracy
  but ships under NGC terms rather than an OSI licence and exports only a feature-input core, so
  using it means porting NeMo's PCM frontend first; Silero-95 (4.7M, MIT, deprecated) reaches zero
  false routes only by abstaining on 1,047 of 1,516.
  **Not measured — the implementation must assume none of it:** no live-mic or known-user speech, no
  accents, room noise or overlap, no code-switching inside one utterance, no real third-language
  speech, and no cost of running this CPU detector concurrently with NPU whisper. The held-out zero
  is one sample result, never a zero-error guarantee. Ledger `.scratch/spike-1-lid.md`, operating
  points `.scratch/spike-3-operating-points.json`; the recorded scores survive those gitignored
  files as `tests/lid_census.json` (`evidence-artifacts.md`).
  **Shipped surface, behind `--two-way` and default OFF.** `lid_accept(argmax, score, ja, en)` is the
  three-part rule alone, pure and duration-free — the schedule owns "not before 2.0 s", which is why
  the rule accepts that 1 s English view as `ja` when handed it. `LanguageDetector(model_dir)` holds
  one `CPUExecutionProvider` session at `LID_INTRA_OP_THREADS`=4 / `LID_INTER_OP_THREADS`=1 and
  `ORT_SEQUENTIAL`, reading the label order from `lang_map.json` rather than a table; `.score(pcm)`
  feeds raw float32 `[1, samples]` to input `audio`, takes output `logits`, and returns the global
  argmax with its score plus the raw `ja` and `en` probabilities from a max-shifted float64 softmax
  cast to float32; `.decide(pcm)` is `score()` through `lid_accept()`. PCM is 16 kHz mono float32 in
  [-1, 1] and is never normalized — FBANK and CMVN are folded into the graph. The flag rejects its
  contradictions at parse time and `check_models` preflights the two runtime-consumed LID files under
  it, so absent weights fail at startup rather than at the first 2 s buffer. Flag off constructs no
  detector and opens no ONNX session; flag on constructs exactly one, for the process.
  **The graph's names are law the gate can only assert, never exercise**: `models/` is gitignored, so
  `tests/test_language_detector.py` stubs the runtime and pins `audio` / `logits` there, and a real
  rename surfaces on a weights-bearing box alone.
- **Repetition-loop cause + free repro.** The recogniser is pinned to Japanese, so audio it cannot
  account for is emitted as Japanese tokens until `max_length`=448 — the 444-char live captions. The
  trigger is neither laughter nor room tone: synthetic non-speech (digital silence, −60 dB noise,
  60 Hz hum) reproduces the hallucination PHRASES (`ご視聴ありがとうございました`) but not the loop,
  while **English speech looped it every time it was tried — on ONE clip**, which is the qualifier
  that claim needs. Live, 26 minutes of spoken English into the JA-pinned recogniser published 143
  captions of at most **139 characters** (p50 10, p90 44, p99 92) with **zero at or above 200**, and
  screened 20 more, against the 444-char captions the loop rule was sized on. The screened split is
  unrecoverable — that session kept no stderr log, only a sampled `backlog peak:` digest — so a screen
  firing does not evidence a loop, the latin rule catching plain English being the other arm. Two
  structural statements, and no rate: the loop reproduces deterministically on that clip, and a live
  session of English produced no published caption anywhere near the 444-char scale. Repro without an
  artifact:
  `curl -sSL -o
  .scratch/jfk.flac https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac` (11 s, public
  domain), resample 44.1k→16k, pad 1 s of silence each end, `replay.py --engine whisper` ⇒ segment 3 is
  a 528-char loop at **RTF 1.106** — a runaway costs more than real time. `soundfile` is absent from
  `.venv`; add `--with soundfile` to read the FLAC.
- **One unreplicated ambient capture produced nothing at all.** A single 10 min 20 s live capture of
  handling noise and room ambience published zero captions, logged zero lines and dropped zero
  blocks. Nothing downstream is established by that — it kept no artifact, so even "the VAD never
  opened" is inference, not record. It has not been repeated ⇒ read it as ONE observation consistent
  with the synthetic result above,
  not as a guarantee that a quiet room costs the pipeline nothing. What it does not license: a claim
  about loudness, about rooms in general, or about any ambience richer than the one sampled.

## Publication screen — `caption_defect()`

- The screen runs at PUBLICATION, upstream of every consumer, so `observe_ja`, the translator queue,
  `_turn` and `_failures` cannot see a defective caption by construction and a runaway streak of any
  length costs no strike. `CodexTranslator.submit`'s repetition screen (the loop rule alone) is
  the BACKSTOP ⇒ on the
  shipped path `tskip=` stays 0. **`tskip=N` is a CONTENT decision, never backpressure** — and
  `tstale=N` is a third thing again, a TIMELINESS one (`translation-leg.md`); the three never merge.
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
- **The latin rule keys on the DECODE language** (`caption_defect(text, token)`; `ASR_LANGUAGE`
  when unnamed): it guards the JA pin, so a two-way piece decoded under `<|en|>` skips it — keyed
  on `ASR_LANGUAGE`, which `--two-way` leaves at `ja`, it dropped every English caption
  (`test_two_way_publishes_english_decoded_under_en`). The loop rule spans every language.
- Latin ratio: the 23 latin-dominant live captions split cleanly — 17 true English at ≤0.15
  Japanese-per-character, 6 Japanese-carrying-loanwords at ≥0.27, nothing between. A 1:1 rule
  (`latin > japanese`) drops **6 genuine Japanese captions**, because a Latin letter is one phoneme
  where a Japanese character is a whole syllable, so one loanword outnumbers the kana around it.
  `CAPTION_LATIN_RATIO`=4 cuts the gap at 0.20.
- **Long utterances publish at settled segments (user ruling; supersedes UNCAPPED).** Utterance
  length is unbounded — `VAD_MAX_SPEECH_S`=20 is a soft cap, clean-utterance p99 136-312 characters ≈
  18-40 s, live max 664 ≈ 88 s, and 63.85 % of the 10-01 session's characters sat in utterances of
  100+ — so a one-line rule made every such `TGT` wait for the whole utterance. `settled_boundary`
  advances the open utterance's published prefix whenever a trim moves committed text out of the
  buffer: a cut COUNT into the raw commit stream (len(utterance) == trimmed + len(emitted) through
  `process`/`_trim`/`_force_trim`, the last emptying emitted so only committed text drains), never
  the latest hypothesis, which may re-spell shown text. A piece is whatever whisper segments the trim
  released (p50 36/43 characters on the two traces); a piece without a word character waits, and a
  punctuation-only remainder at speech end publishes only when the utterance published nothing yet.
  Per-piece semantics, locked through the real worker: the screen judges each piece (a dropped piece
  burns no number, a published one is never recalled, and a loop split below 40 repeated characters
  per piece passes — the translator stalls measured began at 120, on four units, fresh threads and
  the previous model, so a split loop's turn cost is unmeasured); the learner
  observes ONCE per utterance over the pieces that passed, so support and lease still count
  utterances; `on_segment` stays one RAW observation per utterance (goldens unmoved, raw pieces
  concatenate to it). Two-way publishes nothing while the label is unaccepted, releases everything
  settled at acceptance, discards it on a token rebuild, and publishes a never-accepted utterance
  whole under `<!>`.

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
  TGT line dies, the last (SRC flushes as it lands; TGT needs the drain). The drain then executes against
  a pty whose master is gone, where a write raises `OSError` errno 5 ⇒ `emit_line` persists to the
  transcript BEFORE stdout, and `write_stdout` latches the stream off on `(OSError, ValueError)`.
  Third and least guessable: **CPython flushes `sys.stdout` during finalization and that flush fails
  the same way, exiting 120** on an otherwise clean shutdown ⇒ the latch also swaps in `os.devnull`.
  stderr is not implicated: its handler flushes per record, so finalization finds nothing buffered.
- **One event is ONE line, and `emit_line` is where that is enforced.** The grammar every reader
  parses is `[<time>] <TAG> <n>: <text>`, so text carrying its own line breaks writes untimestamped
  continuation lines that `session_report.py` and `build_lag_trace.py` drop whole — measured at 4
  multi-paragraph translations and 1808 lost characters in one 2 h live session. `emit_line`
  collapses breaks with `splitlines()`, never `str.split()`, which folds the full-width U+3000 space
  real Japanese captions carry; a break-free caption passes through byte-identical, which is the
  positive control the lock pairs with. The screen shares the collapsed string, so the transcript
  and the terminal cannot diverge.
- **L-006 — TTY-gate the cursor-clear (`\r\x1b[2K`) protocol per stream, evaluated once.** Both halves
  are module-level constants: `_STDOUT_TTY` gates the meter status line and `emit_line`, so redirected
  stdout stays ANSI-clean and the meter DRAWS nothing off-TTY, and `_STDERR_TTY` gates the
  `logging.Formatter` prefix, the rare ~10-line subclass that beats inlining. Any new stdout status
  writer gates the same way. Gating the writer is not gating the information — the meter's counters
  exist only on that status line, so wherever it is not the reader's they move to stderr as
  high-water marks (`backlog peak: …`, logged only when a peak moves, sampled every
  `METER_INTERVAL`=0.1 s while a status line is drawn and `METER_LOG_INTERVAL`=1 s otherwise).
- **The peak log gates on the stream PAIR — `not (_STDOUT_TTY and _STDERR_TTY)`, not on stdout alone.**
  A log record does not corrupt the status line: the formatter's `_LINE_CLEAR` erases it in place and
  the meter redraws below. It DUPLICATES it, and a moving peak would push ten lines a second through
  the caption scrollback the status line exists to protect. Off a shared terminal neither cost
  applies ⇒ **`2> stt.log` keeps the live captions AND the drop timeline**, where the old
  stdout-only gate made those exclusive and charged every capture run its whole status line.
  `live-smoke.md` item 2 reads the one-redirect form.
- **L-010 — keep device-backend imports at device entry points.** PortAudio probes host audio controls
  during `sounddevice` import, so an eager app-module import can hang model-only evaluators on an
  unhealthy device; `run_session` + `--list-devices` own those imports and offline replay/test stays
  hardware-independent.
  Linux CLI entry points run those operations in a supervised session: 15 s per audio operation,
  45 s for normal stop/drain, bounded TERM/KILL cleanup. The child inherits the locked status FD;
  close it without explicit unlock/unlink so a D-state child continues to exclude duplicate starts.
  Model compilation stays outside the audio deadline. `tests/test_audio_startup.py` covers the
  process boundary and terminal-signal forwarding; the existing callback, queues and drain order stay
  in the child. This contains a broken driver; it does not make SIGKILL interrupt kernel sleep.

## Latency budget — per stage, `tests/eval_latency.py`

Every figure re-derives from `vac_decode_trace.json` (whisper-ja-760M, the lower-cost of two NPU
recordings) + `en_pairing_trace.json` in under a second, no hardware. `retention_probe` (182 s
pause-free) is the demanding clip; `stress_long` (44.7 s) reads within ~0.1 s on most rows.

| stage | p50 | p90 | max | what it is |
| --- | --- | --- | --- | --- |
| update decode | 0.667 | 0.942 | 1.107 | one `process()` |
| commit lag | 2.353 | 3.428 | 6.402 | voice → committed character on the meter |
| provisional lag | 1.237 | 1.690 | 2.476 | voice → the same character shown UNCONFIRMED |
| redraw bound | 1.646 | 4.208 | 7.844 | upper bound: every redraw of a slot recharged as a fresh wait |
| publication | 1.222 | 1.414 | 1.414 | speech end → the LAST `SRC n:` = `VAD_MIN_SILENCE_S` + final decode |
| voice → SRC | 5.670 | 8.508 | 10.294 | voice → the character's numbered line (one-line rule: 13.151 / 24.559 / 33.464) |
| translate turn | 2.170 | 4.310 | 6.340 | `SRC n:` → `TGT n:` (steady 2.140, rotating 4.695) |

- **Decode cost is `0.501 s fixed + 5.66 ms/char`** (`stress_long`: 0.552 + 3.78; turbo read 0.417 +
  7.15). The fixed term is Whisper's encoder over a 30 s window — whisper-ja-760M keeps turbo's
  32-layer encoder, which measured FLAT at 0.31 s (turbo) for buffers of 1.0 s through 28.0 s
  via `perf_metrics.get_encode_inference_duration`; feature extraction adds ~1.8 ms per buffer second
  and `return_timestamps=True` is free (`get_word_level_timestamps_processing_duration` = 0). So
  shortening the buffer touches the MARGINAL half only — 11.25 s → 5 s buys ~0.15 s — and 0.35 s is
  the floor under any update cadence.
- **The commit lag is a DISPLAY POLICY cost, not a compute cost.** `lag = audio-time holdback +
  decode_s`, and the holdback is LocalAgreement-2 withholding text until a second decode confirms
  it. The same decode already held that text: showing its unconfirmed tail costs nothing and takes
  p50 to 1.237 s, max to 2.476 s. `provisional_lag_s` is that arm, run on the same virtual clock
  and the same per-character placement as the committed arm, so their difference is the policy alone.
- **Both arms measure FIRST APPEARANCE ⇒ the gap is NOT a settled-text speedup.** Settled text still
  lands at 2.353 s p50 — `emitted` is append-only and the published line is unchanged — and the gap
  buys an earlier UNSETTLED rendering of the same character. `redraws` carries that cost: **90 of
  180 updates** on `retention_probe` (25 of 44 on `stress_long`) diverge from their predecessor
  before it ended, i.e. rewrite already-visible characters, which land in the dimmed tail by
  construction. `redraw_bound_s` is that cost charged as a fresh wait per redraw — a loose UPPER
  bound, double-counting by construction — and reads p50 1.646 / p90 4.208 / max 7.844. Quote 1.237 s
  for time-to-first-glimpse and 2.353 s for time-to-settled; neither alone describes the screen.
  Two-way's withheld first commit moves NEITHER number (subsection below).
- **`VAC_CHUNK_S` is floored by decode cost, not by taste.** Work rate = `decode_s / VAC_CHUNK_S`:
  0.667 today, 0.89 at 0.75 s, **1.33 at 0.5 s** — past real time, where the catch-up rule stretches
  the cadence to the decode (~0.67 s) and every update runs back to back (unmeasured).
  Every candidate below 0.75 s needs the fixed 0.35 s term cut first.
- Translation is **2.170 s p50, not the ~1 s the tournament's 1.38 s implied** — that figure was a
  bare turn, this one is production order over 215 captions.
- **The rotation tax is CUT.** `translator_brief()` renders the glossary in content order
  (longest-first, then lexical) instead of `terms()` recency order, so a brief changes only when its
  CONTENT does. Reorder-only rotations are gone **by construction, not by sampling** — the lock is
  `test_the_brief_is_a_pure_function_of_glossary_content`. Two figures, and they answer different
  questions: **replayed over one fixed trace, old code → new code reads 40 → 13** brief changes, all
  27 removed being reorder-only and the 8 adds / 2 drops / 3 renderings kept — that is the
  like-for-like number. **End to end, two live runs read 39 → 14**, the new run's 14 being 8 adds /
  4 renderings / 2 drops with zero reorder. Never quote a taxonomy across those two runs: they are
  independent samples of a sampled translator and they learn different renderings (4 against 3).
- **`rotations` counts GLOSSARY rotations only** — `eval_en_pairing.py` sets `rotated` from
  `translator._brief != brief`. Production ALSO rotates on the turn cadence, so a 215-caption session
  rotates **16** times (14 glossary + 2 cadence) and opens 17 threads counting `start()`'s own. The
  cadence arm is deliberately untouched and locked by
  `test_the_turn_cadence_rotation_is_untouched`; read the row's "below 15" against the glossary
  metric its own baseline of 39 was measured on.
- **The row's turn-p90-under-3.5 s half is REFUSED.** Excluding every rotation, steady p90 measures
  3.530 s (n=201) against a 3.5 s bar, so the bar sits at or below the rotation-free floor and what
  remains is codex turn latency rather than rotation. **This is ONE sample of sampled model output**
  (`evidence-artifacts.md`) ⇒ read it as evidence that this lever cannot reach the bar, never as a
  rate: a second run could place the floor either side of 3.5 s, and only a lever acting on STEADY
  turns (brief size, effort, serviceTier) can move it. **The refusal's cost, named:** suppressing
  content rotations any further would delay an add, a rendering or a drop from reaching the model
  until the next cadence rotation. Today's design pays a rotation for every content change and so
  never briefs a stale glossary; what was removed is re-sending an identical one.
- **The row's rendering-consistency half is now MEASURED, on the raw stream.** Its acceptance cited
  "`eval_en_pairing.py` distinct spellings 1/1/1" and no such metric existed; `raw_spellings` is it.
  `SessionContext` stores ONE rendering per term by construction, so any check reading `renderings`
  is tautological: mutating the raw EN after pairing (`Gon` → `Gawn`/`Ghone`) leaves both the learned
  map and M12.5's structural verdict green while the raw census goes `distinct` 1 → 3 and `multi_spelled`
  `[]` → `["ゴン"]`. On the committed trace NO paired term carries more than one supported spelling
  (`multi_spelled == []`) — ゴン `Gon`×8, 標柱 `Heijū`×9 against one stray `Gon`, カスケ one `Kasuke`
  and one stray `Gon` with neither supported, 神様 `God`×2. **Never re-cite the learned map as evidence of
  consistency**, and read `readings` against `turns` as coverage and a sub-support spelling as noise
  (`evidence-artifacts.md`).
- Counter-cost, DESCRIPTIVE not causal: the new run's surviving rotations are dearer (p50 3.900 →
  **4.695 s**, max 5.600) and its totals read rotation time 160.4 → **64.7 s**, wall 568.8 →
  542.3 s. Those are sample totals of two independent runs, never attributable savings — the runs
  share their JA but only **46 of 215** EN outputs, 26 old rotations disappear while 1 new one
  appears rather than the same turns moving, and replayed over ONE fixed trace the semantic changes
  fall on the **same 13 turns** under both orderings. So no batching effect exists to explain the
  per-rotation rise; it is run-to-run variation in a sampled model.

### Two-way withholding — `tests/eval_two_way_settled.py`

Two-way commits nothing until the LID accepts a label, which cannot happen before `LID_MIN_SECONDS`
of voiced buffer exists. **The queue row's premise is REFUTED for the shipped case: that hold moves
time-to-SETTLED by nothing.** LocalAgreement-2 already commits almost nothing on update 1, so at the
2.0 s gate both clips read their published numbers unchanged and 12 characters in total are re-dated.
Time-to-FIRST-GLIMPSE is untouched in every arm — the dim tail renders upstream of the commit rule.
The evaluator replays `vac_decode_trace.json` on the same virtual clock, the same per-character
placement and the same `_quantiles` as the commit-lag row above, so its no-withholding arm reproduces
`commit_lag_s` exactly and the difference between arms is the commit rule alone.

| arm | `stress_long` p50/p90/max | `retention_probe` p50/p90/max | re-dated |
| --- | --- | --- | --- |
| one-way, no withholding | 2.435 / 3.740 / 6.405 | 2.353 / 3.428 / 6.402 | 0 / 0 |
| accepted at the 2.0 s gate | 2.435 / 3.740 / 6.405 | 2.353 / 3.428 / 6.402 | 2 / 10 |
| abstained once, accepted at 3.0 s | 2.439 / 3.740 / 6.405 | 2.398 / 3.506 / 6.402 | 36 / 177 |
| never accepted (held label) | 11.641 / 21.591 / 24.744 | 13.151 / 24.559 / 33.464 | 270 / 1080 |

`re-dated` counts characters committed before acceptance and is an UPPER bound on how many moved: one
whose own clock already runs past the accepting update keeps it. **The last row is an upper bound
too, never a forecast** — it forces both long-form clips into the held path, which is not what
abstention does. Abstention concentrates in utterances shorter than 2 s (93.17 % of the 410 finals
under 2 s never accept), where the final update arrives carrying the whole caption at once and no
earlier commit is left to withhold. Read that row as the cost of a detector that never accepts on
long speech, and never as the cost of the 409 held utterances.

## Real-time cost — the instrument is CARRY (D-016(d))

VAC awaits each decode — update and final alike — inside the coroutine draining `audio_q`, so unlike
the sherpa two-stage worker it does NOT feed VAD during decode: capture stacks at 1 s/s into
`AUDIO_HEADROOM_S`=8 s for the whole blocking span. Per-update NPU decode (whisper-ja-760M) measured
p50 0.668/0.667 s and max 1.003/1.107 s on the two pause-free clips against a 1 s update cadence, the
trim rule capping the buffer at 9.252 s (turbo: 0.552/0.645, 0.764/1.006, 11.248 s). **Two rules hold the line, each owning one live failure shape:**

- **The catch-up rule owns sustained overload.** Each update consumes exactly `VAC_CHUNK_S`, so
  decodes slower than that stack `decode_s − 1` of backlog per update. An update due while more than
  `VAC_BACKLOG_S`=0.5 s is queued waits for the drain and fires on the first block leaving ≤ 0.5 s,
  folding all pending audio into one decode ⇒ cadence = max(`VAC_CHUNK_S`, `decode_s`) and the queue
  peak ≤ 0.5 s + one blocking span. At recorded cost a due update sees at most 0.28 s queued (a final
  decode's carry into the next utterance), so the traced trajectory and replay's output stay
  byte-identical; a queue without `queued_samples` (replay's) never waits, and neither does the final
  decode.
- **The 8 s headroom owns single blocking spans.** No drain runs inside one, so the queue must
  outlast the longest: an update decode chained straight into the final one (the VAD closes on the
  block after the update fires) and a runaway decode up to the 448-token cap. 8 s holds two
  back-to-back measured runaways (3.53 s each) on top of the 0.5 s catch-up allowance, 7.56 s; a
  span past what the queue has left drops like any other.
- **Measured live on the 09-24 capture.** A 29-minute session on battery under
  `low-power` (NPU at 950 MHz in 366 of 617 active samples) logged `drop=1949` at 2 s headroom. Its
  10 Hz `q=` meter reads 863 blocking spans, p50 0.60 / p90 0.92 / max 3.53 s, holding 1759 of the
  1949 dropped blocks (188.9 blocks/s saturated ⇒ ~5.3 ms each, ~10 s of speech). Three shapes:
  **S** sustained 1.0-1.4 s updates stacking inside long utterances; **C** update + final chains of
  2.4-2.8 s; **R** a 444-character runaway update at 3.53 s. That is why drops sat in long
  publication gaps without duration alone causing them: only a long utterance on a slow machine
  state stacks S, and C and R need a chain or a runaway. The older 26-minute `drop=9033` session
  kept only a peak digest: its bursts in ≥17 s gaps, the 4 largest near screened captions, FIT S and R
  — an inference that digest cannot decide.
- **Limit: `HARD_TRIM_S` is enforced AFTER a decode**, so once trims have already failed (the
  forced-trim regime, 28 s with no segment cut) a catch-up fold can hand whisper up to 28 s + the
  backlog, of which it reads 30 s. Outside everything measured — the real paced max buffer is
  11.12 s — and inside a regime that already discards un-emitted text.
- **The `live` arm of `tests/eval_backpressure.py` is the lock**: both traced clips at ×1.75 (the
  first rung at or above the live/trace ratios 1.42 and 1.64 of ≥ 1 s updates) with the middle update
  charged the measured 3.53 s. On the turbo trace the old code dropped 97 / 1323 blocks there and the
  fix none (queue peak 3.52 / 3.66 s); on the whisper-ja-760M trace the fix drops none, queue peak
  4.04 / 4.14 s, 0 forced trims. Neither rule alone passes: 8 s without catch-up still dropped 947
  blocks on `retention_probe`, and catch-up at 2 s drops on the stall. The recorded-cost ladder first
  drops at ×6 / ×4 (turbo: ×6 / ×6; ×2.0 / ×1.5 before catch-up). **A `forced_trims=1` on
  `retention_probe` there (turbo trace) is a harness artifact** — off the recorded trajectory the replayed hypotheses no longer match their buffers, so
  `_trim` finds no cut; the same arm on the REAL NPU recogniser trims normally (max buffer 11.12 s,
  0 forced trims, 0 drops).
- **What catch-up costs in accuracy — real NPU recogniser (turbo, before the 760M swap), `retention_probe` at ×1.75 on the virtual
  clock**, trace-interpolated costs with real text and segments:
  `tests/eval_retention.py --pace 1.75 [--stall] [--no-catchup]`.

  | arm | CER | S / D / I | queue peak |
  | --- | --- | --- | --- |
  | catch-up | **0.0600** | 35 / 34 / 1 | 3.12 s |
  | no catch-up, unbounded queue | 0.0609 | 36 / 35 / 0 | 24.26 s |
  | catch-up + 3.53 s stall | **0.0720** | 28 / 52 / 4 | 4.00 s |
  | no catch-up, unbounded + stall | 0.0609 | 36 / 35 / 0 | 26.42 s |

  Sustained overload costs nothing: the stretched cadence reads 0.0600 against 0.0609. Folding a
  runaway-length stall into ONE decode costs **+0.011 on this one sample**; the lossless alternative
  lags the caption by 24-26 s, and the shipped 8 s without catch-up drops 947 blocks. Accepted trade:
  3 of 863 live spans exceeded 2 s, one of them runaway-length. At ×1.0 the real-recogniser run
  reads 0.060891938250428816, turbo's shipped figure of the time to the digit.
- **Aggregate RTF is the wrong instrument** — it shows mean compute below real time and says nothing
  about maximum blockage, which is what the headroom is spent against.
- **Carry is the cross-utterance instrument:** a caption costing more wall time than its own audio
  hands the difference to its successor (silence between captions drains further, so ignoring gaps
  is conservative, and catch-up only lowers a caption's decode sum). Over 215 captions of narration
  (whisper-ja-760M, `caption_trace.json`) the worst carry is **0.137 s**; carry reaches 2 s at ×1.228
  and the 8 s headroom at **×1.338** — against ×1.541 / ×1.762 on turbo's trace. Read the gap as
  partly machine state: this trace is the lower-cost of two recordings at load average ~6 (decode
  sums 473.9 s and 550.8 s, turbo's quieter run 427.9 s), while a same-conditions VAC replay of
  section 01 put whisper-ja-760M ~10 % FASTER (179.0 s vs 199.5 s). Unresolved; the
  `test_backpressure` ×1.25 reserve lock holds.
- **A caption's `decode_s` is a SUM of that utterance's update decodes, never one blocking call** —
  reading its 6.213 s max (turbo: 7.420 s) against `AUDIO_HEADROOM_S` reports a stall that did not
  happen. Never
  re-derive real-time risk from per-caption sums.
- Rerun cost varies ~20 % run to run, and the burst is machine state rather than a path property: the
  same clip/section/device at RTF 1.098 (`git show f25cfb5:tests/caption_trace.json`) carries
  **77.231 s** where turbo's later trace carried 0.000 s (the whisper-ja-760M trace: 0.137 s).
- `drop=` counts backend-sized callback BLOCKS, never samples or speech ⇒ quote blocks, converting
  only with a measured block size. `session_report.py`'s `attribute_drops` places every `backlog
  peak:` drop increase against its bracketing captions, publication gap and screened captions, and
  the peak log's pair gate (L-006 above) lets `2> stt.log` record that timeline with the status line
  kept; the run's own `session: <path>` marker places an increase that precedes the first caption.
  The blocking spans themselves are read off the status line's `q=` sawtooth in a `script -T`
  typescript: `q` rising at 1 s/s is the drain stalled on a decode.

## Known caveats

- Models are a runtime prerequisite — `check_models()` preflights and points at `models/README.md`.
- **The translator is NOT the quality bottleneck — the recogniser is.** Over 30 consecutive live
  SRC/TGT pairs the English is fluent and faithful to whatever Japanese it is handed, and every defect
  is ASR (`パーツ`→`パンツ`, `左肩甲骨`→`左肩骨`, `コロナル`→`セコロナルタ`, a stray `おやすみなさい。`
  hallucination mid-meeting). Benign default-engine quirks: ジェミニ→ゼミニ, 文→分 homophone — the EN
  leg translates through them.
