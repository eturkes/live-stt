# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
`Deferred` there = the pointer + one title per row below, and that list is the spine. Rank = funding
order. Acceptance is written at deferral time while the evidence is fresh, and the funded row is that
unit's whole contract (`assurance-posture.md`). The `/goal` body the user pastes names the row it
funds; closing a row deletes its title from this file and from `.agent/spec.md`'s spine in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and BOTH M14 recovery arms have never met a mic; standing debt = latency
   feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence. **Accept:** the
   user runs `live-smoke.md` and reports; each item lands verified or defective.
2. **Explain the live audio drops.** A 26-minute live session ended at
   `backlog peak: q=2.00s drop=9033 skip=20` — 9033 discarded callback blocks of captured audio, in
   12 counter increases, every one inside a publication gap of ≥17 s, each of the 4 largest with a
   `skip=` increase within 11 s (separations 0, 1, 1, 11 s; quote separations, a ratio here is an
   artifact of the window chosen). **The obvious hypothesis is already refuted: duration alone does
   not cause it.** 16 of the 20 gaps >20 s dropped nothing, including the longest (81 s), and
   `tests/eval_backpressure.py`'s VAC arm paces 182 s of pause-free speech drop-free at a queue peak
   of 1.060 s of 2.000 s ⇒ an arm that merely paces a long utterance would pass vacuously. The
   correlate is a SCREENED caption and no further: `caption_defect` has two arms and only the
   repetition runaway costs more than real time (RTF 1.106), plain English caught by the latin rule
   costing nothing extra. Separating them needs the `caption dropped (…)` lines, and that session kept
   no stderr log — only a sampled digest of its `backlog peak:` lines survived, which cannot attribute
   a drop to a caption. A real log can: off a TTY `live_stt.py` re-logs
   `backlog peak:` whenever the rendered string CHANGES, so every drop increase gets a timestamped
   line at meter cadence, and `logger.warning("caption dropped (%s): %.24s…")` timestamps each screened
   caption with its defect and first 24 characters on any stream. `session_report.py` parses the first
   (`_PEAK`, and it keeps only the last line) and not the second at all ⇒ no production logging change
   is needed, only a reader. **Accept:** in order, (a) `session_report.py` grows drop attribution —
   per `backlog peak:` drop increase, the captions bracketing it, any `caption dropped (…)` line in the
   same window with the defect it named, and the publication gap it sits in — locked by a test over a
   synthetic transcript+log pair proven red; (b) the user runs one live session as
   `live-stt > stt.log 2>&1` that reproduces a nonzero `drop=` — **stdout redirected too, costing the
   live status line for that whole run**, because `meter()` draws the status line on a TTY and logs
   `backlog peak:` only OFF one, so `2> stt.log` records no drop timeline at all and step (a) would
   have nothing to read (`live-smoke.md` item 2 names that second run). This row may spend a
   production change instead: the peak log gates on `_STDOUT_TTY` where what it protects is the
   status line, which a log line corrupts only when the LOG shares that TTY ⇒ gating it on stderr
   would let `2> stt.log` keep both. L-004 puts the choice behind the user's terminal, so it is this
   row's to make and not a docs fix; (c) then
   either the mechanism is named and reproduced as a
   `tests/eval_backpressure.py` arm proven RED against today's code and fixed green with retention CER
   ≤ 0.0609 re-derived, or the row records a refusal naming the measured headroom shortfall and what
   the user loses. Do not skip (a) and (b): the evidence that would decide this does not exist yet.
3. **Two-way translation (JA↔EN), research first.** Direction chosen per utterance: Japanese in →
   English out, English in → Japanese out. **Live-stt must stay usable throughout** (user ruling) ⇒
   every step lands behind a default-off flag and the JA→EN path is green at every commit; with the
   flag absent, construct only today's JA pipeline and keep passing `"<|ja|>"`. Architectures priced
   so far: text-side routing is REJECTED as a production design — a JA-pinned decode can return
   English as Japanese-script hallucination, so a script ratio has unobservable false negatives;
   fresh-pipeline-per-utterance is REJECTED at ~1.00 s of setup+detect before normal decode (0.46 s
   construct p50, 0.54 s detect); the two live candidates are ONE resident pipeline with per-call
   autodetection on the first voiced VAC update, and TWO resident pipelines behind a cheap standalone
   LID (VoxLingua107 ECAPA on CPU/OpenVINO the leading candidate). Budget the second pipeline at the
   MEASURED ~2.0 GB resident RSS (2253 MB held, 225 MB after release — `asr-pipeline.md`), never at a
   compiled-blob size on disk, which is not evidence of allocated memory. The fork is a
   single measurement: `openvino-genai==2026.3.1.0` source copies per-call Whisper config and
   re-detects when `language` is absent, which contradicts this repo's measured NPU language latch,
   and measurement governs. **Accept:** a `res`/`spike` wave lands four numbers, each on a named
   input, then WRITES the implementation row's acceptance from them; no production edit lands until
   that row exists. (i) LATCH — one `WhisperPipeline` instance on NPU decodes the same buffer three
   times, `language="<|ja|>"` then `"<|en|>"` then the argument omitted, in the token form
   `live_stt.py` already passes, 3 reps. Read the pipeline's OWN reported language where the API
   exposes it: decoded-text equality alone does not say which language was applied, only that the
   output did not move. (ii) ENGLISH QUALITY — **this repo has no committed English audio**; every
   corpus here is Japanese (`ja_asr.common_voice_8_0`, 「ごん狐」, `retention_probe`), so the row
   acquires an English set under L-017 (`evidence-artifacts.md`) — FLEURS `en_us` pairs with the
   `ja_jp` side L-028 already characterized — and scores EN-pinned against JA-pinned decode with
   `cer.py`. `.scratch/jfk.flac` (11 s, public domain, `curl` line in `asr-pipeline.md`) is the
   zero-cost SMOKE check only; one clip cannot carry this number.
   (iii) CO-RESIDENCY — peak RSS with two pipelines constructed and both compiled, against the
   measured 2253 MB single-pipeline figure, plus per-update decode p50 while alternating between them,
   paired per update against the single-pipeline control over `retention_probe` using Unit C's
   interleaved method (`asr-pipeline.md`). (iv) LID, only if (i) says the latch holds — JA/EN
   confusion matrix and abstention rate of the selected detector over that acquired English set and
   the Japanese corpus, at the utterance durations the VAD actually produces. A number without its
   input does not close this row. `TRANSLATOR_INSTRUCTIONS` pins the leg
   Japanese→English and declares every turn "one block of transcribed Japanese speech", so the EN→JA
   direction needs its own instructions and its own degrade story.
4. **Prove EN rendering consistency on the raw stream.** The rotation-tax row's acceptance cited
   `eval_en_pairing.py` "distinct spellings 1/1/1" and no such metric exists there. `SessionContext`
   keeps ONE rendering per term by construction, so every check reading `renderings` is tautological:
   mutating post-pairing EN (`Gon` → `Gawn`/`Ghone`) leaves both the learned map and the M12.5
   verdict green while a one-spelling assertion reddens. Consistency is a property of the RAW EN
   stream, which is where `translation-leg.md`'s 9-of-9 figure was counted. **Accept:**
   `eval_en_pairing.py` grows a raw-stream metric counting distinct spellings of each learned term's
   rendering across `turns[*].en`, reported per run and locked by a test proven red under that exact
   mutation; the committed trace then re-derives one spelling per paired term, or the divergence is
   recorded as the real number.
5. **M10 candidate-screen remainder** — zipformer + SenseVoice lack current JA evidence,
   Moonshine-JA's license is unclear, ReazonSpeech-k2-v2 adds PyTorch/Transformers + remote custom
   model code. **Accept:** re-open only if the shipped path fails AND the added runtime surface buys
   a materially different hypothesis. Tournament record → `.agent/archive/m10-asr-tournament.md`.
