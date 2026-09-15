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
   costing nothing extra.
   **(a) LANDED** — `session_report.py`'s `attribute_drops` places every `backlog peak:` drop increase
   against the captions bracketing it, the publication gap between them, the `skip=` step across the
   same peak line, and every `caption dropped (…)` line inside that gap with the defect it named;
   locked over a synthetic transcript+log pair in `tests/test_session_report.py`, proven red by
   neutralization. **The production choice this row held open is MADE, not refused** — the peak log
   gates on the stream PAIR (`not (_STDOUT_TTY and _STDERR_TTY)`) instead of on `_STDOUT_TTY` alone, so
   `live-stt 2> stt.log` keeps the live status line AND records the drop timeline; `live-smoke.md`
   item 2 and the README read that one-redirect form.
   **Still owed, both user-only** (L-004): **(b)** one live session run as `live-stt 2> stt.log` that
   reproduces a nonzero `drop=`, with the log kept — no agent-side work creates that evidence;
   **(c)** then either the mechanism is named and reproduced as a `tests/eval_backpressure.py` arm
   proven RED against today's code and fixed green with retention CER ≤ 0.0609 re-derived, or the row
   records a refusal naming the measured headroom shortfall and what the user loses.
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
4. **Price the EN lag behind every JA line.** The one live session (26 min, 143 captions, 143 of 143
   translated) delivered `EN n:` at **p50 2 s, p90 4 s, max 11 s** behind its `JA n:`, against an
   `Intent` line that asks for about a second. `Intent` is the user's ⇒ record the gap, never
   re-label the ask. **Queueing is REFUTED for the MEDIAN and only for it**: 130 of the 143 captions
   were emitted with no earlier EN outstanding and carry that same p50 2 s, while the warm-thread
   bench median is 1.38 s general / 1.71 s clinical (`translation-leg.md`) ⇒ the ~0.6 s median gap
   belongs to the TURN, not the backlog. The tail is unexplained: the max-lag caption did have one
   ahead of it, and the slowest turn with nothing ahead still took 7 s. **Accept, agent-side half:**
   a bench through the real `codex app-server` under L-026's rules — fresh thread per measured turn,
   a real-input canary behind every risky one, configs sequential, repetitions interleaved — pricing
   the SHIPPED turn against a bare thread over one input set, one cumulative arm per candidate, so
   each of the `developerInstructions` + `translator_brief` payload, a `serviceTier` echo that did
   not land, thread age across a `TRANSLATE_ROTATE_TURNS` boundary, and the emit path between
   `turn/completed` and `emit_line` is ATTRIBUTED or REFUTED by a paired median rather than assumed.
   **User half (L-004):** a second live session's lag distribution, since one sample cannot separate
   a turn-latency level from that session. The row closes on the bench plus a written statement of
   what the live half still owes.
5. **M10 candidate-screen remainder** — zipformer + SenseVoice lack current JA evidence,
   Moonshine-JA's license is unclear, ReazonSpeech-k2-v2 adds PyTorch/Transformers + remote custom
   model code. **Accept:** re-open only if the shipped path fails AND the added runtime surface buys
   a materially different hypothesis. Tournament record → `.agent/archive/m10-asr-tournament.md`.
