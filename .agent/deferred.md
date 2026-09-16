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
3. **Two-way translation (JA↔EN), implementation.** Research is CLOSED and its four numbers are law.
   An EXPLICIT language token always beats the NPU pipeline's latch, so ONE resident whisper pipeline
   switches direction per utterance for free. A wrong token costs 41× CER on JA audio and 26× on EN
   and never degrades gracefully ⇒ a detector that guesses under uncertainty is worse than one that
   abstains. A second resident pipeline costs +2015 MB and +0.050 s per update and buys nothing ⇒
   unfunded fallback. The detector is ECAPA VoxLingua107 on ONNX Runtime CPU, first decision at 2.0 s
   of voiced buffer, 0 false routes in 1,516 two-second views at 11.74 % abstention;
   `asr-pipeline.md` carries every threshold, rate, cost and non-measurement, and the settled label
   also picks one of two immutable direction-specific `codex app-server` threads
   (`translation-leg.md`). **Live-stt stays usable throughout** (user ruling) ⇒ every unit lands
   behind `--two-way`, default OFF: with the flag absent the process constructs today's JA pipeline,
   opens ONE thread and keeps passing `"<|ja|>"`, and JA→EN is green at every commit.
   **Accept, five units, one commit each.**
   **(0) the grammar** (user ruling, added to this row): the published line becomes `SRC n:` /
   `TGT n:` in EVERY mode, one-way included, because two-way makes a language tag name a different
   role each utterance; `session_report.py` keeps parsing legacy `JA`/`EN` transcripts so the first
   live session stays comparable. Held-label marking is `SRC n <!>: text`, landed by unit (2).
   **(1) the detector.** `models/lid/d2-ecapa/` acquired by the pinned command in `asr-pipeline.md`
   with its SHA-256 verified, wrapped as a raw-PCM scorer returning label + score + margin, and the
   three-part gate at its named thresholds — global 107-way argmax ∈ {`ja`, `en`}, score ≥0.35,
   margin ≥0.35, never renormalized. The unit MEASURES the 2 s inference cost on this box and records
   it, the spike having timed only 1 s and VAD-final. Locks: flag-off constructs no detector and
   opens no ONNX session; the gate's decision table over recorded scores, including the 1 s buffer
   that routes EN→`ja` at 0.9822 and must never be scored.
   **(2) the schedule.** No LID call before 2.0 s of voiced buffer, a retry on each later VAC update
   until one accepts, the accepted label FROZEN for the rest of the utterance and reset at VAD close.
   Text decoded before acceptance renders wholly DIM and commits nothing, and an accepted label
   differing from the token that produced that text resets the `StreamingProcessor`. An utterance
   that never accepts — 409 of 1,926, and 93.17 % of the 410 finals shorter than 2 s — decodes and
   translates under the HELD label (JA at startup) and is MARKED the way a degrade is marked; no
   caption is ever withheld, because withholding a fifth of utterances breaks the standing usability
   ruling. The row records that this hold is UNPROVEN on the corpus, which carries no bilingual
   sequence: it beats forcing the pairwise winner only while the switch rate among abstentions stays
   under 3.93 %. Withholding the first commit until update 2 shifts time-to-SETTLED ⇒ re-derive it
   offline by replaying `vac_decode_trace.json` under the new commit rule, which needs no hardware.
   **(3) the token.** The frozen label selects `"<|ja|>"`/`"<|en|>"` on every `generate()`. Retention
   CER ≤ 0.0609 re-derived, this being the unit that touches decode. **SUPERSEDED by user ruling:**
   `--source-lang` is no longer this row's manual override — `--two-way` with `--source-lang` is a
   parse error, as is `--two-way` with a sherpa `--engine`, and `--source-lang` alone keeps today's
   one-way meaning (`spec.md`).
   **(4) the reverse leg.** A second immutable thread with its own EN→JA `developerInstructions` and
   the same glossary rendered in its direction, the settled label routing each turn; app-server EOF
   disables BOTH directions, a poisoned turn replaces only its own thread, and the transcript stays
   source-only on failure.
   **NOT owned by this row:** a measured `gpt-5.6-luna` one-thread-vs-two quality comparison, which
   exists nowhere; and everything the LID spike could not measure — live-mic or known-user speech,
   accents, room noise, overlap, code-switching inside one utterance, real third-language speech, and
   the cost of running the CPU detector concurrently with NPU whisper. Those close with the user
   under L-004, never here.
4. **Price the EN lag behind every JA line.** The one live session (26 min, 143 captions, 143 of 143
   translated) delivered `EN n:` at **p50 2 s, p90 4 s, max 11 s** behind its `JA n:`, against an
   `Intent` line that asks for about a second. `Intent` is the user's ⇒ record the gap, never
   re-label the ask. **Queueing is REFUTED for the MEDIAN and only for it**: 130 of the 143 captions
   were emitted with no earlier EN outstanding and carry that same p50 2 s, while the warm-thread
   bench median is 1.38 s general / 1.71 s clinical (`translation-leg.md`) ⇒ the ~0.6 s median gap
   belongs to the TURN, not the backlog. The tail is unexplained: the max-lag caption did have one
   ahead of it, and the slowest turn with nothing ahead still took 7 s.
   **Agent-side half LANDED** — 300 real turns through the app-server under L-026 (12 committed
   captions × 5 reps × 5 cumulative arms, arms
   sequential, reps interleaved, a fresh thread per measured turn for A0-A3, a real-input canary
   behind every one, paired bootstrap CIs), full table in `translation-leg.md`. All four candidates
   resolved and every one came back NEGATIVE: the `developerInstructions` + `translator_brief`
   payload **−1.169 s**, the `serviceTier` request **−1.246 s** — and its echo DOES land, `default`
   on A0-A2 and `priority` on A3-A4, which retires this row's "echo that did not land" suspicion —
   thread age across the rotation boundary **−1.455 s** ⇒ a fresh boundary turn costs **+1.455 s**,
   and the emit path REFUTED at a 5.609 µs p50, six orders below the lag. The shipped configuration
   is FASTER than a bare thread; what remains is **~5.42 s of bare model + transport**, attributed
   to no sub-factor. A4 + emit accounts for 77 % of the live p50, 55 % of the p90, 27 % of the max.
   **Still owed, user-only (L-004):** a second live session's lag distribution. One sample cannot
   separate a turn-latency level from that session, and the live TAIL is precisely what the bench
   fails to explain — 1.782 s unexplained at p90, 8.009 s at max. Run `live-smoke.md` keeping
   `2> stt.log`, then `session_report.py --log stt.log`, and compare its EN-lag p50/p90/max against
   the first session's. The row closes on that distribution; no agent-side work is left in it.
5. **M10 candidate-screen remainder** — zipformer + SenseVoice lack current JA evidence,
   Moonshine-JA's license is unclear, ReazonSpeech-k2-v2 adds PyTorch/Transformers + remote custom
   model code. **Accept:** re-open only if the shipped path fails AND the added runtime surface buys
   a materially different hypothesis. Tournament record → `.agent/archive/m10-asr-tournament.md`.
6. **Re-derive the LID census from the committed corpora.** `tests/lid_census.json` is a reduction of
   spike outputs that are gitignored and will be lost (`models/lid/results/*.json.gz`,
   `models/lid/analysis.json`), so its generator sits in `.scratch/` and cannot rerun from committed
   state — while the decision table it feeds is the gate's only evidence that the three-part rule
   holds over 1,926 utterances. The reduction path itself is PROVEN, so the port is the whole cost:
   clip → `make_vad()` buffers → 2 s prefix → shipped `LanguageDetector.score()` re-derived 6 of 6
   sampled `2s` rows exactly at the fixture's 6-decimal rounding. What has no committed source is the
   ORDERING — `utterance` is an integer index whose id map lives only in the gitignored spike output
   ⇒ the regenerator must define the order from the corpora themselves and the fixture must then be
   compared as a set of rows, not by index. **Accept:** `tests/build_lid_census.py` regenerates the
   file byte-identically from the two committed FLEURS corpora through the shipped
   `LanguageDetector`, weight-gated with an `absent:` reason, and the `.scratch/` generator dies in
   the same commit.
