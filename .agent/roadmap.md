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

- Milestone: **M13 the shipped pipeline degenerates on live audio** — **IN-PROGRESS**, and the next
  milestone by user direction. **M13.1 is DONE and no unit is OPEN**, so the next session's mode is
  PLANNING for M13's recogniser scope — which is **BLOCKED on a standing precondition: a retained
  WAV of laughter / throat-clearing / room tone** (`## Decisions pending from user`). Evidence
  unchanged ⇒ read-only close, and run a standing option instead — **M12.5 is now DONE, so the
  M12 tail is gone and P-019 is the one funded-looking alternative awaiting a ruling.** Two
  real-world sessions, both flagged:
  session 1 = 9 captions that are 76-100 % one short unit repeated 33-148 times (341-517 chars);
  session 2 = 2 more, new max **890 chars** against a caption p50 of 17. No committed artifact
  reproduces it. **Session 2 also converted the EN-leg failure from inference to measurement, and
  it is a SECOND defect** — a repeated short unit makes the TRANSLATOR generate without terminating
  (measured again at 25 turns in M13.1), while 480 chars of real speech costs 7.0 s.
  **M13.1 shipped that mitigation on 2026-09-03 and is DONE**: a degeneracy screen in
  `CodexTranslator.submit` declines the caption before the queue, so the EN leg survives any streak
  of runaways. **The recogniser side is the whole remaining milestone and is still UNPLANNED** — it
  needs audio before it can be planned, and the next ask is a short retained WAV of laughter,
  throat-clearing and room tone (session 2's n=43 is laughter after the longest speech gap).
- Previous milestone: **M12 does the EN rendering learner hold up on real ASR output?** —
  **IMPLEMENTED** (M12.1-M12.5 all DONE). **The answer is measured and it is mixed.** Both modes
  P-012 named are refuted: the recogniser does not fragment a name (M12.1) and the dead pairings the
  offline screen found do not exist against a real translator (M12.5, 0 live against 2 simulated).
  What the milestone found instead are two modes P-012 never named, and both are real: the learner
  **stabilises a WRONG key** (M12.1 — the two terms it trusted all session were mis-recognitions,
  while the protagonist sat below a floor M12.4 then lowered), and it **learns a WRONG rendering**
  (M12.5 — `標柱 = I`, because English capitalizes its first-person pronoun). The first is fixed;
  the second is `polish.md` **P-019** and awaits a user ruling. Opened by
  user decision on the P-012 register row, which outgrew polish.
  `observe_en` (D-015, shipped by P-002 in `16a842b`) keys a learned English spelling on the JA
  string the RECOGNISER produced, and every P-002 arm ran on clean Aozora text — the learner's best
  case. Live the key is a hypothesis, so two modes were unmeasured: **(a) dead pairing** — a
  mis-recognised name pins a correct spelling to a key that never recurs; **(b) split key** — the
  recogniser alternates JA forms of one name and each form carries its own rendering. **M12.1
  answered both on real NPU output and neither fires:** (b) is refuted (7 of 8 occurrences take one
  form) and (a) is unreachable on a 67-caption clip (the key recurs to caption 65; a 60-segment lease
  cannot expire). What the census found instead is a **third mode P-012 never named — the learner
  stabilises a WRONG key**: both terms it trusts all session are mis-recognitions, while the
  correctly-recognised protagonist ゴン sits below `_TERM_RUN`'s 3-character katakana floor and is
  invisible to it — **M12.4 ruled on that floor and lowered it to 2**, admitting ゴン as the
  session's earliest and healthiest trust episode without disturbing any other. **M12.3 then made
  (a) reachable and it fires: 2 dead pairings over the whole
  story** (鼻腔, イワシ), both by lapsed lease, one of them paired on its very last sighting —
  **but only in simulation. M12.5 put all 215 captions through the real translator and neither
  pairing exists**, because a rendering needs a proper noun and both keys are ordinary English
  nouns; the control ゴン → `Gon` pairs in the same run, so that is a refutation and not a null.
  The **fourth mode**, and the one still open, is what that run found instead: the learner pairs
  `標柱 = I` off the English first-person pronoun (`polish.md` **P-019**).
- Previous milestone: **M11 production-qualify the shipped whisper+VAC+NPU path** — **IMPLEMENTED**
  (all units DONE). Both questions M11 opened are answered and both answers are green: the VAC
  branch does not drop audio in real time (M11.4, with a 1.25-1.75× reserve), and D-016's retention
  CER is real (M11.5, 0.0583 reproduced exactly). Judgment-review sessions were retired on
  2026-09-02 (L-032, D-012), so IMPLEMENTED is this project's terminal milestone state.
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
  narration **on the sherpa fallback only**, plus the shipped path's one accuracy gate (M11.5
  `eval_retention.py`, 182 s pause-free, CER 0.0583 on NPU); deterministic paced backpressure on
  BOTH branches —
  44.722 s at RTF 0.20 on the sherpa fallback, 44.722 s + 182.482 s at real per-update NPU cost on
  the shipped VAC path (M11.4); shutdown/stage-failure/translation-degradation mechanics.
- User-only debt: latency feel, `-o`, soak, sustained live cadence, Ctrl+C-mid-decode, and the whole
  VAC partial-caption cadence (`.agent/memory.md` § Smoke, L-004).

## Open (do these; lowest ID first)

- **M13 — The shipped pipeline degenerates on live audio: the recogniser emits repetition loops and
  the translator never terminates on them. [IN-PROGRESS]**
  Next by user direction. **Two independent defects. The translator one was funded first and
  M13.1 shipped it (see `## Done`), so the EN leg no longer dies on a runaway.** The RECOGNISER
  side stays **UNPLANNED** and still needs its own planning session before any code: the two
  hypotheses below take different fixes and the corpus does not exist yet. **Planning is blocked on
  audio, not on a decision** — the ask is a short retained WAV, not a transcript.
  **The measurement** — `transcripts/2026-09-03T14-03-43.txt` (gitignored, 241 captions / 193
  translations / ~41 min). Screen each caption for its longest adjacent repeated substring (unit
  ≥2 chars, ≥3 repeats, span ≥12 chars): **12 captions flagged, 9 of them 76-100 % repeat** — n=196
  `'次は、'`×148 filling 444 of 444 chars, n=224 `'、私は'`×148 of 517, n=138 `'副部の'`×109, n=144
  `'クラブの'`×109, n=195 `'私は、'`×98, n=130 `'アーメンの'`×88, n=240 `'中学院の'`×72, n=227×33.
  Caption length p50 **15**, p90 92, **max 517**. Three captions are `ご視聴ありがとうございました` /
  `ありがとうございました` — whisper's canonical Japanese hallucination on non-speech — so some of
  these fire on silence or room tone, not on speech.
  **The negative control is committed and clean.** The same screen over `tests/caption_trace.json`
  (215 NPU captions, whole 「ごん狐」 story, shipped VAC path) flags **0**; max caption length **69**
  against the live 517, p90 33 against 92. **Nothing in tree reproduces this** — every pinned clip
  is clean continuous narration. Live audio is not retained, so that session is evidence, not a
  repro, and acquiring a triggering corpus is the milestone's first real cost.
  **It is what took the EN leg down.** EN flowed through n=194 on short captions; n=195/196/197 are
  **three consecutive runaways** (341/444/86 chars) = exactly `TRANSLATE_MAX_FAILURES`=3 ⇒
  permanent JA-only for the last 47 turns. The one earlier isolated EN gap, n=144, is also a
  runaway. Single runaways at n=130 and n=138 did translate, so one is survivable and three in a
  row are not. The translator behaved as D-009/D-011 design; the recogniser is the fault.
  **M13.1 has since shipped, so this specific kill path is closed:** a caption repeating one short
  unit for ≥40 characters is declined before the translation queue, which costs an EN line and no
  strike, so no streak of runaways can disable the leg again. Every runaway named above is declined
  by the shipped screen. What remains is the recogniser emitting them at all.

  **SESSION 2 reproduces the mode and measures the translator side (`transcripts/2026-09-03T16-07-24.txt`,
  gitignored, 195 captions / 194 translations / 37 min, plus the user-captured `stt.log`).** Same
  screen: **2 flagged at 100 % repeat** — n=22 `'中央の'`×111 (333 chars) and n=43 `'あ'+'は'`×889
  (**890 chars**, a new max against session 1's 517). Caption length p50 **17**, p90 102. Two short
  runs earlier the same hour (8 and 7 captions) screen **0**, so the mode is not per-run.
  **The whole session's stderr is one line**, and it is the runaway's: `[16:18:20] WARNING
  translation failed (); JA-only for this block`, 16 s after n=43's caption at 16:18:04. The empty
  `()` is `TimeoutError`, whose `str()` is `''` — `asyncio.wait_for(self._turn(ja),
  TRANSLATE_TIMEOUT_S)` at `live_stt.py:1025`. n=43 is the session's ONLY missing EN; n=44/45/46
  translated normally, so `_failures` reset and the leg survived at 194/195.
  **The EN leg is killed by a translator defect, not merely by caption size.** Four probes through
  the real `CodexTranslator` (fresh thread, `_turn` under a 120 s bound instead of the shipped 15 s,
  control/runaway interleaved per L-026) separate length from degeneracy, and length is exonerated:

  | input | chars | turn |
  |---|---|---|
  | real speech n=47 / n=98 | 237 / 264 | 4.0 s / 4.6 s |
  | real speech, 6 captions concatenated | 890 | **8.1 s**, 2364 EN chars |
  | n=22 runaway `'中央の'`×111 | 333 | **4.7 s**, 971 EN chars |
  | n=43 laughter `'あ'+'は'*(N-1)` | 20 / 60 | 2.5 s / 2.2 s |
  | same, longer | 160 / 333 / 445 / 890 | **>120 s, all four** |

  So: **890 chars of real speech is fine, a 3-character meaningful unit repeated 111× is fine, and a
  single-character run somewhere between 60 and 160 characters stops terminating.** The inputs are
  literals (`"あ" + "は" * (N-1)`), so this reruns with no artifact. `TRANSLATE_TIMEOUT_S` is the only
  thing containing it; `_abort_turn` then interrupts the turn server-side. Consequence for the plan:
  a recogniser fix removes the trigger but leaves the translator unbounded on any degenerate input,
  and a caption-side degeneracy screen — NOT a length cap, which n=22 and the 890-char control both
  refute — is a mitigation independent of hypotheses (a) and (b) below.
  **Ordinary-turn numbers from the same session, for sizing:** EN latency p50 **3.0 s**, p90 5 s,
  max 11 s over 194 turns; EN/JA character amplification p50 **2.57×**, max 5.43× (n=22); silence
  gap before a caption p50 7 s, p90 24 s, max 86 s. n=43 follows the neighbourhood's longest gap
  (50 s), consistent with the trigger being non-speech.
  **Do not confuse this with P-009** (`memory.md`, CLOSED by user ruling, do not re-derive or
  re-fix): that is a 4-character re-spelling duplication from `streaming.py`'s `emitted`
  bookkeeping, measured at 0 trims. This is decoder degeneration two to three orders of magnitude
  larger, and a fix for one is not a fix for the other.
  **Two hypotheses, different fixes, separate them first.** (a) **Nothing bounds a degenerate
  decode** — `WhisperEngine.generate` (`live_stt.py:331`) passes only `language`, `task`,
  `return_timestamps` and optional `hotwords`, so on this build `repetition_penalty`=1.0,
  `no_repeat_ngram_size`=SIZE_MAX and `max_new_tokens`=SIZE_MAX are all effectively off and a 1-2 s
  buffer may emit 517 characters. (b) **VAC ratifies the loop** — LocalAgreement-2 commits the
  common prefix of two consecutive decodes, and two degenerate decodes of one buffer agree on the
  repeated prefix, so `streaming.py` cannot separate "stable because correct" from "stable because
  degenerate".
  **Named unknowns for the plan:** whether the NPU `StaticWhisperPipeline` HONORS those three knobs
  (it already refuses `initial_prompt` and `hotwords`, so assume nothing and measure); how to
  acquire audio that triggers it; and whether a `streaming.py`-side degeneracy reject, running the
  screen above on a hypothesis before it is committed, is cheaper and more portable than a
  decode-side knob. Prefer whichever the fresh session can gate without hardware.
  **The corpus problem got cheaper.** Session 2 names a candidate trigger the user can produce on
  demand — n=43 is **laughter** (`あははは…`) after the neighbourhood's longest speech gap, and n=22's
  `中央の` loop follows a 13 s gap. A short mic recording of laughter, throat-clearing and room tone
  is a far smaller ask than "capture a 40-minute session and hope". Ask for a retained WAV, not a
  transcript: audio is what closes the (a)/(b) split.
  **Capture limitation to state when asking.** `stt.log` is stderr only, and that is all it CAN be:
  the meter status line is gated on `_STDOUT_TTY` (`live_stt.py:1365`, L-006), so redirecting stdout
  silences `q=`/`seg=`/`drop=` entirely and no redirection captures them. A run's backlog evidence
  is therefore unobtainable today without a code change — size that into any evidence request.

## Done (ID · outcome · decisions/lessons produced)

- **M12.5 — Confirm M12.3's two dead pairings against the real translator. [DONE] — REFUTED: neither
  pairing exists live, the control does, and the run found a defect the simulation could not.**
  `tests/eval_en_pairing.py` + the committed `tests/en_pairing_trace.json`: all **215 captions
  translated through the real `CodexTranslator`** over a live `codex app-server`, in production's own
  order (`observe_ja` → `_translate` → `observe_en`), 0 failures, 0 restarts, 0 declined by M13.1's
  screen, 39 thread rotations, 569 s wall, EN latency p50 2.33 s / p90 4.19 s / max 6.2 s.
  Episode bookkeeping is `eval_term_census.learner`, now taking its English side from a supplier, so
  M12.3's table and this one are comparable by construction rather than by restatement — the census
  re-derives byte-identically through the refactor.

  | term | trusted@ | openings best/live | paired@ best/live | live rendering |
  |---|---|---|---|---|
  | ゴン **(control)** | 20 | 2/4 | 23/**31** | **Gon** |
  | 標柱 | 38 | 2/13 | 44/**127** | **I** ← defect |
  | 鼻腔 **(candidate)** | 48 | 2/2 | 50/**never** | — |
  | イワシ **(candidate)** | 123 | 2/4 | 125/**never** | — |
  | 神様 | 185 | 2/3 | 189/**194** | **God** |

  **The verdict M12 closes on: 0 dead pairings live against the best case's 2.** 鼻腔's two openings
  offered `Gon` then nothing; イワシ's four offered `Anke`, nothing, `Hyōjū`, nothing — no two turns
  agree, and neither term is a proper noun in English, exactly as predicted. **The control separates
  that from a broken harness:** ゴン pairs to `Gon`, 神様 to `God`, so renderings are acquirable on
  this corpus and the candidates' nulls are structural, not underpowered. **P-012's arm matrix is
  therefore not funded** — there is no real pairing for it to protect.
  **The live arm is not the simulation, and the openings column is why.** A term the English never
  pairs never leaves `observe_en`'s gate, so it keeps closing its NEIGHBOURS' openings while opening
  on every one of its own sightings: 標柱 opens 13 times live against the best case's 2. The JA side
  is arm-independent (`observe_ja` never reads `renderings`), so `trusted_at` / `expired_at` /
  `mechanism` / sighting counts are identical in both arms and the report checks that.
  **Out of contract, registered not fixed — `polish.md` P-019, and it is the sharper finding.**
  `observe_en` learned **`標柱 = I`**: "I" is capitalized everywhere in English, so it passes the
  not-sentence-initial rule, and it is the **most common sole proper noun in the whole stream — 20
  of the 63 turns carrying exactly one, ahead of `Gon` at 17**. The session ended briefing the
  translator to render a mis-recognised key as a pronoun, for 13 of its 26 sightings. Dropping the
  single token `"I"` removes it and leaves both correct renderings identical; the user owns whether
  that shape is right, since the same rule would pin the hallucinated `Okkawa`/`Anke` too.
  Suite 256 → **265 passed / 1 skipped / 15.5 s** (9 locks: common-noun English never pairs while a
  mid-sentence name does, an unpaired key keeps closing its neighbours' openings, the verdict moves
  only pairing columns, three refusals binding a turn trace to its captions, the prompted-trace
  guard, the committed run still yielding this verdict, and the floor arm comparing behaviour rather
  than the simulated placeholder). All 7 new predicates proved non-vacuous
  by neutralization (L-022): dropping the sentence-initial rule reds 3, replaying the best case
  instead of the recorded English reds 4, and the caption-text binding / duplicate-index refusal /
  hotwords guard / `openings_before_quiet` omission / floor-arm placeholder each red 1. An eighth
  mutant is a deliberate EQUIVALENCE check and reds 0: calling `observe_en` on every turn (what
  production does) is equivalent to calling it only where the JA gate is open. Gate 6/6.
  Cost: 215 live turns ≈ 0.2 pp of the weekly window. `main=86% 206K/240K` against `est 150K` ⇒
  **1.37**, at the low end of the band — the deliverable was one measurement whose whole surface
  M12.3/M12.4 had already enumerated, and the overrun was the out-of-contract finding (L-031).
  Calibration series: M12.2 1.34 · M12.3 1.53 · M12.4 2.14 · M12.5 1.37 ⇒ ratio **1.60**, spread
  1.34-2.14. No teammates funded — the probe is script-derivable and the ruling is MAIN's.

- **M13.1 — Decline a degenerate caption before it reaches the translator. [DONE] — SHIPPED: the
  EN leg now survives what killed session 1, and the threshold is corpus-picked, not guessed.**
  `repeat_span()` + a screen in `CodexTranslator.submit` (`live_stt.py:1005`), 26 production lines.
  A caption is declined when its longest adjacent repetition of a unit ≤`TRANSLATE_REPEAT_UNIT_CHARS`
  =8 reaches `TRANSLATE_REPEAT_MAX_CHARS`=40. The seam is BEFORE the queue, so `_turn`, `_failures`
  and `observe_en` are untouched **by construction**; the numbered `JA n:` line still prints and is
  still saved, one WARNING names the caption and how much of it repeats, and the meter gained a
  dedicated `tskip=` (kept out of `tdrop=`, which means translation fell behind).
  **The calibration matrix, 25 turns through the real app-server (`tests/eval_translate_repeat.py`, inputs
  are literals + the committed trace, so it reruns with no artifact):** fresh thread per turn, 30 s
  bound, real-speech canary after every degenerate turn — **every canary healthy at 2.8-7.0 s**, so
  every row is credible.

  | chars | real speech | `は` | `中央の` | `クラブの` | `アーメンの` |
  |---|---|---|---|---|---|
  | 20 | 2.9 s | 2.9 s | 3.4 s | 2.7 s | 3.2 s |
  | 60 | 3.9 s | 3.4 s | 4.8 s | 2.9 s | 3.5 s |
  | 120 | 5.2 s | **HANG** | 5.4 s | 6.4 s | 3.3 s |
  | 240 | 6.0 s | **HANG** | 10.8 s (2465 EN chars) | 4.6 s | 5.0 s |
  | 480 | 7.0 s (1059 EN chars) | **HANG** | **HANG** | **HANG** | **HANG** |

  So the defect is **not single characters only** — every unit up to 5 characters stalls at 480 —
  and it is **not length**: 480 characters of real speech cost 7.0 s. That killed the "longest run
  of one character" rule the plan offered as the cheaper option, and the general screen ships.
  **v1 of the probe was thrown away and it is why the harness carries canaries:** on ONE shared
  thread a stalled turn made a later CONTROL hang, and `は`×60 read as fatal where a fresh thread
  measures 3.4 s. `_abort_turn`'s interrupt + drain is not enough for a stall (L-026 extended).
  **The false-positive side is committed, hardware-free and wide:** over every real caption in tree
  — 215 NPU captions + all 25 replay goldens' texts + 6.9 K characters of Aozora reference — the
  longest adjacent repetition is **8** (`ポンポンポンポン`, the story's own onomatopoeia), five times
  under the threshold, against a smallest measured stall of 120. The **unit bound is what keeps a
  person out**: widened to 20 characters the screen starts seeing a speaker repeat a PHRASE
  (`、うなぎが食べたい`×2 = 18 chars, caption 108).
  Suite 241 → **256 passed / 1 skipped / 16.4 s** (14 translator locks + 1 meter lock). Gate 6/6.
  All 7 new predicates proved non-vacuous by neutralization (L-022): removing the screen reds 3,
  merging the counter into `tdrop` reds 2, a threshold past the measured stall reds 13, one below
  the real corpus reds 1, widening the unit bound reds 2, counting a single occurrence as a
  repetition reds 1, dropping the meter counter reds 2 — baseline and post-restore both 77 passed.
  **The first neutralization matrix was WRONG and the fix is now L-022:** three mutants reported a
  copy of one earlier red set, because CPython reuses a `.pyc` when two edits a second apart change
  the module by the same byte count. `main=88% 212K/240K` against `est 90K → cal 150K` ⇒ 2.36 on the
  raw estimate, 1.41 on the calibrated one; no teammates funded.
  Out of contract, registered not fixed: **P-018** — `observe_ja` still sees a declined caption, so
  three runaways repeating one unit could promote a decode artifact into the term list.

- **M12.4 — Rule on `_TERM_RUN`'s katakana floor. [DONE] — SHIPPED at 2: the floor was hiding the
  one name in the story and guarding a slot ordinary vocabulary does not occupy.** One production
  character (`[ァ-ヺー]{3,}` → `{2,}`), matching the kanji floor that was already 2.
  **The evidence, rerunnable with no hardware in <1 s:** `eval_term_census.py --floor 3` over the
  committed 215-caption trace. Floor 2 admits **10 forms that floor 3 rejects; exactly 1 reaches
  support and it is the protagonist** — ゴン, 40 captions. The other 9 are the ordinary vocabulary
  the floor was built to stop (キス 2 captions; カゴ カン キレ ゴミ ドン ヒバ モズ ラー 1 each) and
  **every one dies below `CONTEXT_TERM_SUPPORT`**, so none ever enters a brief. Floor 3 admits
  nothing of its own. `--floor N` rewrites the shipped pattern rather than restating it, so an arm
  differs from production in the floor and in nothing else, and it refuses a `_TERM_RUN` it cannot
  locate exactly one floor in.
  **The control, over the whole story: ごん = 50 reference occurrences, 0 dropped.** At floor 3 only
  3 land as a candidate at all (ゴギツネ, `Gong`, 言語), none reaching support ⇒ the protagonist was
  invisible in **47 of 50**. At floor 2, **43 of 50** land, 40 of them as the single stable form ゴン.
  **Non-interference is measured, not assumed.** Admitting a term also spends a capacity slot and
  closes its neighbours' `observe_en` openings while it is unpaired, so the arm compares shared
  episodes whole: candidates 123 → 133, episodes 7 → 8, dead pairings 2 → 2, evictions 0 → 0, and
  **all 7 pre-existing episodes are identical** — same `trusted_at`, `paired_at`, `expired_at`,
  mechanism. `CONTEXT_MAX_TERMS`=12 still never binds (6 live). ゴン is the session's earliest and
  healthiest episode: trusted@20 (18 captions before the first mis-recognition), paired@23, 36 of
  its 38 sightings carrying the rendering, live at the end — not a new dead-pairing candidate.
  **What actually decided it, and it is not the raw count.** The kanji floor was ALREADY 2, and 5 of
  the 7 terms it admits are ordinary vocabulary (鼻腔, 物置, 二人, 神様, イワシ). 2-character ordinary
  vocabulary was therefore never a new admission class — the asymmetry simply made the stricter side
  the one that carries Japanese names. The filter that works is `CONTEXT_TERM_SUPPORT`, not the
  floor: it rejected 9 of the 10 admissions here unaided.
  **Why the 2-character katakana slot is nearly empty of vocabulary:** across every JA REFERENCE in
  tree — Aozora 「ごん狐」 4,888 chars, the 182 s retention probe 1,354, the replay goldens 637 —
  there is **1 maximal 2-character katakana token in 6,879 characters** (ドン) against 70 of 3+
  characters, because borrowed words are mostly ≥3 katakana. That slot is populated by the
  RECOGNISER instead (50 tokens in 4,508 caption characters, 40 of them ゴン), which is exactly where
  a short native name lands when whisper writes it in katakana.
  **Honest limit:** one corpus, katakana-sparse by genre. In a loanword-dense domain (ドア, バス,
  ケア, メモ) short vocabulary would reach support, and the protection against it now rests on the
  support threshold alone. The cost of a wrong term is bounded and known — brief tokens, a thread
  rotation, one of 12 slots, and a blocked pairing opening while unpaired — and the learner already
  pays it 5 times over on the kanji side.
  **Blast radius confirmed empty on the shipped path.** `hotwords_reachable: false` on NPU, so
  `set_hotwords` drops the list and decode cannot move; measured, not inferred — all **25 replay
  goldens** reproduced with the accel farm sourced, including the whisper NPU row that normally
  skips (`tests/test_replay.py` 32/32, 0 skipped), so no golden text or boundary shifted and
  D-016's CER numbers stand untouched.
  Suite 235 → **241 passed / 1 skipped / 19.1 s** (6 new locks: a 2-character katakana term is
  learned, a lone katakana character is still not, the arm moves only the katakana floor and
  restores it on an exception, it refuses an unlocatable floor, admitted-vs-trusted are separate
  counts, and admitting a term can strand a neighbour's rendering). Gate 6/6. All 5 new predicates
  proved non-vacuous by neutralization (L-022): reverting the floor reds 5 tests, dropping the arm's
  restore reds 3, and the refusal / `shared_episodes_identical` / replay-derived `trusted` red 1
  each. `main=71% 171K/240K` against `est 80K` ⇒ 2.14; no teammates funded — the arm is
  script-derivable and the ruling is MAIN's.

- **M12.3 — Record the full story and screen it for dead pairings. [DONE] — the mode is real: 2 dead
  pairings over 215 captions, and neither is a name.** `build_caption_trace.py` now decodes every
  pinned section in one pass and numbers the captions continuously: **215 captions / 848.350 s /
  6 sections**, decode 427.9 s, RTF 0.504, `hotwords_reachable: false`. One decode per section is
  faithful because nothing on the shipped path carries state between utterances (D-016(c) deleted
  prev-text, the NPU refuses prompts), so the only cross-section state is the learner's and it lives
  downstream — which is exactly what makes `CONTEXT_TERM_LEASE`=60 reachable at 215 captions.
  **Determinism reproduced a third time:** section 01's 67 captions came back byte-identical to
  M12.1's committed trace in BOTH text and VAD boundaries.
  **The screen (`eval_term_census.py`) models trust as EPISODES** — one term's whole lifetime,
  promoted → sighted → paired → expired — because `renderings` is discarded with its term, so an
  episode is exactly one chance to acquire a spelling and lose it. It reports per episode: trusted
  at, sightings, last sighting, published captions surviving after it, pairing openings (and how
  many landed before it went quiet), paired at, sightings that used the rendering, expiry + whether
  the lease lapsed or capacity evicted it. The English side is simulated at the learner's BEST case
  through the REAL `observe_en`, fed a unique letters-only proper noun per term; that also keeps the
  gate honest, since a paired term stops blocking its neighbours' openings.
  **Result — 7 trust episodes, 2 DEAD PAIRINGS, 0 evictions** (`CONTEXT_MAX_TERMS`=12 never binds at
  ≤5 live). 鼻腔 paired@50 on its very last sighting, 0 sightings used it, expired@110, 165 captions
  of session left; イワシ paired@125, 2 sightings used it, expired@194, 81 left. **The roadmap's own
  predicted candidate missed by 27 captions:** 加助 → カスケ goes quiet at caption 182 and its lease
  would have lapsed at 242, past the story's 215.
  **Two report defects the longer stream exposed and this fixed.** `reaches_support` counts sightings
  anywhere in the story, but the learner also has to hold a candidate for `CONTEXT_TERM_MEMORY`=40
  segments — 標準 is sighted in 5 captions [48, 100, 112, 182, 195] and is **never trusted**, because
  no three fall inside one 40-caption window. The form view now carries the replay's verdict
  (`trusted`) instead of implying support. Second: openings saturate at `CONTEXT_EN_SUPPORT`, since
  a paired term leaves the gate — so the count answers "could a rendering be acquired", never "how
  much evidence was available", and `sightings_after_paired` carries the latter.
  Alignment is per section, not over the concatenated story: it keeps every occurrence inside the
  audio it was read from and holds the DP at six small matrices instead of one ~4.9k × 4.5k. 兵十:
  38 reference occurrences, 38 located, 0 dropped, 6 distinct forms, only 標柱 trusted.
  Suite 229 → **235 passed / 1 skipped / 14.2 s** (6 new locks: the dead-pairing shape at its
  sharpest, a key still being said, a key that goes quiet unpaired, eviction-vs-lease, per-section
  merging, placeholder distinctness). Both new predicates proved non-vacuous by neutralization
  (L-022): dropping the paired requirement fails 2 tests, calling every expiry a lease fails 1.
  Gate 6/6. Cost: one 6-section NPU pass ≈ 8 min. `main=77% 184K/240K` against `est 120K` ⇒ 1.53;
  no teammates funded — the screen is script-derivable.

- **M12.2 — Acquire 「ごん狐」 二〜六 into the pinned corpus. [DONE] — the whole story is pinned, and
  section 一 reproduced bit-for-bit through the parameterization.** `tests/long_form.json` went
  1 section → **6**, 288.521 s → **848.351 s (14:08, 2.94×)**, 66 → **213 VAD segments**, 30 KB →
  62 KB, schema `{source, sections{"01".."06"}}` keyed by the zero-padded number that is also the
  `id`, the WAV name and the MP3 suffix.
  **No row number survives in the script**, which is what the acceptance asked for: a section's range
  comes from grouping the alignment on its own audio-file column, its title/author announcement row
  from matching that text, and its spoken heading / `章おわり` from reading the aligned text — so
  sections 三/五 correctly get no heading and no chapter-end marker (the reader does not speak them
  there), and a re-released alignment fails a check instead of mis-cropping. Three structural
  cross-checks replace the old hand-written constants: sections must be 1..N, row ids consecutive
  within a section, and the six ranges must tile rows 2..178 with no gap or overlap (`test_cer.py`).
  **The finding that forced a design change: Kokoro leaves 4-10 s of narration UNALIGNED inside four
  of the six sections** (三 4.574 s, 四 4.191 s, 五 9.532 s, 六 6.954 s), so M9.6's "rows must be
  sample-contiguous" rule would have rejected them. A hole is unaligned audio inside one continuous
  crop, never a splice, and the Aozora reference covers it — what the hole shortens is the
  *automatic* text, so `alignment_check` now spends the unaligned FRACTION of the span as budget over
  the flat 0.10 surface allowance. Every section passes with margin (surface disagreement S+I alone
  is 0.015-0.077 everywhere); align_cer 0.0513 / 0.0773 / 0.0766 / 0.0852 / 0.1893 / 0.1425.
  **Reproduction, not carry-forward.** Section 一's `build` (11/11 keys incl. `wav_sha256`
  `a70d7443…`), `reference` (4/4 incl. `normalized_sha256` and the N=1383 S=36 D=1 I=34 check) and
  `vad` (7/7) blocks came out identical to `HEAD`, and re-running `--score` re-derived both sherpa
  rows byte-identically (k2v2 CER 0.25380, parakeet 0.23572) rather than copying them. M12.1's term
  census reruns unchanged end to end — 8/8 occurrences, same two trusted errors, same openings.
  **L-017 exercised on all three paths**, each producing the same WAV: valid cache (no network),
  fresh download, and a deliberately truncated cache entry that re-fetches. Downloads are now cached
  and staged (`.part` → rename), so a rebuild costs no network and a torn write cannot install.
  Scoring is `--score`-only and a section keeps its rows exactly while its WAV + reference hashes
  hold, which is what lets acquisition rerun without paying for a decode; only `01` carries scores,
  because sherpa CER on five more sections of a non-default fallback answers nothing M12 asks.
  Suite 228 → **229 passed / 1 skipped / 17.3 s**; gate 6/6. `main=78% 187K/240K` against
  `est 140K` ⇒ calibration ratio **1.34**; no teammates funded.

- **M12.1 — What does the shipped path actually give the learner as a key? [DONE] — a
  mis-recognition, and it hides the one name that matters.** `tests/build_caption_trace.py` →
  **`tests/caption_trace.json`**: 67 captions over the pinned 288.521 s 「ごん狐」 narration, hash-gated
  against `long_form.json` `build.wav_sha256` before decoding, real whisper/VAC on NPU.
  `tests/eval_term_census.py` derives the census from that file alone — no model, no accelerator, no
  audio, <1 s, so a fresh clone reruns it (the `eval_vac_lag.py` shape). It locates the term by
  ALIGNING reference to captions with the shipped scorer, never by searching for guessed spellings,
  which is why `cer.py` now exposes `alignment()` (the pairs `align()` already counted; one DP, one
  tie order, `align`'s vectors unchanged).
  **Census of 兵十 — 8 reference occurrences, 8 landed as candidates, 0 dropped.** Two forms:
  **標柱** in 7 captions [33, 34, 38, 44, 46, 58, 65] → REACHES `CONTEXT_TERM_SUPPORT`=3, and 標準
  once [48] → below support, so it is never keyed. Over the whole session the learner trusts
  **exactly 2 terms, and both are errors**: 標柱 (兵十, "marker post") first trusted at caption 38,
  鼻腔 (びく, the fish creel → "nasal cavity") at 48. Pairing openings — captions where exactly one
  trusted term is unpaired, which is `observe_en`'s JA-side gate — 標柱 n=5, 鼻腔 n=2, both ≥
  `CONTEXT_EN_SUPPORT`=2, so renderings WILL be acquired.
  **The control is the sharper finding.** ごん, the protagonist, occurs 16× and whisper recognises it
  correctly as ゴン in 15 of them — and the learner sees it in NONE, because `_TERM_RUN`'s katakana
  floor is 3 characters and ゴン is 2. Only the compound ゴギツネ matched, once. So on real ASR output
  the learner is briefing the translator about a phantom while the name a reader would actually
  notice never enters the picture. `--term <any>` censuses any other name.
  **Determinism, measured not assumed:** two full NPU replays produced byte-identical caption text
  and byte-identical VAD boundaries (67/67), which is what licenses M12.2 to replay this file offline
  instead of paying for its own accelerator run. Decode cost did NOT reproduce — 284.7 s vs 275.7 s,
  each run carrying one burst of 3-4 stalled captions → `polish.md` **P-014**, out of contract here.
  Suite 221 → **228 passed / 1 skipped / 22.7 s** (`tests/test_term_census.py`, 7 model-free locks on
  the two rules that decide the census: alignment-located occurrences, and a rendering below
  `_TERM_RUN`'s floor being invisible however often it recurs). Gate 6/6. Cost: 2 NPU replays ≈ 5 min
  each. `main=77% 184K/240K`; no teammates funded — the census is script-derivable.

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
- **M11.4 — Does the VAC path drop audio in real time? [DONE] — no, with a 1.25-1.75× reserve.**
  Three pieces, one commit. (1) `on_update` seam through `_vac_segments`/`worker`/`replay.py`: one
  call per `StreamingProcessor.process`, in order, carrying `(buffer_s, buffer_end_s,
  commit_audio_s, text, final, decode_s)` — five deterministic fields plus the one measured one, and
  it is what recovers the commit audio endpoint `live_stt.py` discarded at `commit, _ = await …`.
  (2) `tests/build_vac_trace.py` → `tests/vac_decode_trace.json`: real per-update NPU decode cost for
  both pause-free clips, plus the hypothesis behind each. Storing the hypotheses is the unlock —
  `StreamingProcessor` is a pure function of decode outputs and buffer lengths, so replaying them
  reproduces the measured commit/trim trajectory with no model. (3) `eval_backpressure.py` VAC arm +
  4 tests: a `decode_segments` trace recogniser, real silero, production cadence/trim, executor
  interception charging each decode its recorded cost, and `max_segment_depth == 0` in place of the
  legacy `0 < depth <= 8`. Five tests: the seam's exactly-once/in-order contract over the stubbed
  VAD, both clips drop-free, and the two guards below.
  **Result — 44.722 s stressor / 182.482 s retention, paced at 20 ms:** `dropped == 0`, `divergences
  == 0` (44/44 and 180/180 updates on-trajectory), `forced_trims == 0`, segment queue 0, audio-queue
  high-water **0.760 s / 1.060 s** of 2.000 s, longest contiguous decode **0.764 s / 1.006 s**, duty
  0.544 / 0.642, buffer capped at 11.248 s by the trim rule. Contiguous == max single decode in both,
  which proves one update fires per queue drain and updates never bunch.
  **Two guards make that non-vacuous**, both committed tests: the same replay at ×4 decode cost
  drops, and a series shifted by one update reports 44/44 divergences while still dropping nothing —
  so the divergence counter, not the drop counter, is what certifies each cost was charged to the
  buffer it was measured on. `SCALE_LADDER` reports the margin: dropping starts at ×1.5 (retention)
  and ×2.0 (stressor).
  **Honest limits.** Decode cost varies ~20 % run to run (max 0.949 → 0.764 s across two builds), so
  the committed trace is one sample and the ladder is the margin around it. The retention queue
  high-water drifts 0.84 → 1.06 s across the clip as mean decode rises 0.606 → 0.674 s; duty < 1 and
  the 11.248 s buffer cap bound it, but the trend is real. Lag was not measured — the derivation is
  parked as `polish.md` P-011. `README.md`'s "Capture and VAD feeding continue during decode" was
  false for the shipped branch and is corrected to what was measured. Suite 207 → **212 tests /
  14.8 s**; `main=94% 225K/240K`.
- **M11.5 — Is D-016's retention CER 0.0583 real? [DONE] — yes, exactly.** The number reproduced to
  four decimals from committed inputs, so nothing in D-016 is superseded. **CER 0.0583** (S=33, D=35,
  **I=0**, N=1166, hyp 1131 chars) over the 182.482 s pause-free probe on the shipped whisper/VAC
  path. Both claims that rested on it hold: the headline retention improvement (0.1587 → 0.0583) and
  the append-only-`emitted` fix's `I` 12 → 0.
  The fix for "unsourced" is a producing artifact, not another loose figure: **`tests/eval_retention.py`**
  (on-demand, never a gate step, ~4 min warm) hash-gates the probe WAV against `retention_probe.json`
  before decoding — so a published number is always bound to the pinned input, not to whatever sits
  in the ignored cache — then replays through the real VAC path and scores with `cer.py`.
  **Placement is an exact-target inference, not a readback**, as the acceptance required:
  `openvino_genai.WhisperPipeline` wraps its compiled model without exposing it, so there is no
  `EXECUTION_DEVICES` to query; the run establishes `requested_device="NPU"` plus a successful decode,
  and `check_device` admits exact names only (an `AUTO:` spelling would relocate silently).
  **One honest divergence, recorded not smoothed:** decode 111.9 s / 182.482 s = **RTF 0.613**, just
  above D-016's recorded 0.48-0.60. It sits inside M11.4's measured ~20 % run-to-run decode variance,
  so the band is widened to **0.48-0.61** (headroom 1.6-2.1×) in D-016(e) and in `README.md` rather
  than left to read as a miss. 8 VAD utterances over the pause-free clip, matching D-016(a)'s 8.
  Docs realigned onto the shipped path: `models/README.md` rewritten (it still named k2v2 the default
  and never mentioned whisper or OpenVINO at all) with the verified `hf download` acquisition,
  per-engine sizes, and the regenerable compile cache; `README.md` lost its retired sherpa-cadence
  claims — the headline "~0.1 s after endpointing" is the VAD-segment rule, replaced by the VAC
  cadence + partial captions — and gained the status-line/partial-caption description, the
  `VAC_CHUNK_S`/`VAC_TRIM_S`/`ASR_DEVICE` constants, `gate.py` + the four evaluators, and correct
  sherpa-only scoping on the 8-segment queue and the >10 s chunker. `grep -nP '[\x{2013}\x{2014}]'
  README.md` clean (rc=1, positive control fires). Suite unchanged at **212 passed / 1 skipped /
  14.9 s** — the evaluator adds no tests by design. Gate 6/6. `main=65% 155K/240K`; no teammates
  funded (the measurement is script-derivable, the docs judgment-bearing), so `mate=` n/a.
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

Carried out of M11's standing scope boundary, still closed: engine/model selection (fixed by D-016) ·
NPU applicability for the sherpa fallbacks · energy per watt (RAPL `energy_uj` is mode 400
`nobody:nogroup`, unreadable in-container even under sudo, so NPU-vs-GPU perf/W is a host
measurement). **EN-rendering learning left this list by shipping** — it was out of M11's scope, not
rejected, and `16a842b` landed `observe_en` (D-015); M12 now owns the follow-up question. D-016's declared limits also stand and are not
re-litigated: the hotwords gain used an ORACLE term list drawn from the reference; long_form absolute
CER is inflated by period-vs-modern orthography in the 「ごん狐」 reference; one clip per corpus.

## Decisions pending from user

**Two open. The first is an EVIDENCE request, not a design call: M13's recogniser side needs audio
before it can be planned.** M13.1 shipped the translator mitigation, so nothing else in M13 is
executable — the two hypotheses take different fixes and no committed clip reproduces the defect.
**The ask: a short retained WAV (any length, a minute is plenty) of laughter, throat-clearing,
room tone and a few seconds of silence, recorded on the live-stt mic.** Session 2 named the trigger
— n=43 is laughter (`あははは…`) after the neighbourhood's longest speech gap, and n=22's loop
follows a 13 s gap — so this is a far smaller ask than "capture a 40-minute session and hope". A
transcript cannot substitute: audio is the only thing that separates hypothesis (a) an unbounded
decode from (b) VAC ratifying the loop. Without it M13 cannot open its next unit, and the standing
options below (a live-mic validation pass, a maintenance pass, a new capability milestone) or M12.5
are what a session would run instead.
**A second decision is now waiting and it is a design call, not evidence: `polish.md` P-019.**
M12.5 measured the learner pairing **`標柱 = I`** on the shipped path — English capitalizes its
first-person pronoun, so it passes `observe_en`'s not-sentence-initial test, and over 215 real turns
"I" is the most common sole proper noun in the stream (20 of 63, ahead of `Gon` at 17). The session
ended telling the translator to render a mis-recognised key as a pronoun. Three shapes, and the
user chose the JA-side plausibility shape on 2026-09-04 **conditional on it being more robust
long-term, and measurement says it is not** — a katakana-key gate drops the CORRECT `神様 = God` and
removes `標柱 = I` only because 標柱 is kanji, so it blocks every kanji name (兵十, 加助 are kanji
names in this story). The defect's key is itself name-shaped, so no key-shape test can separate the
cases; the separation is purely English-side. **The shape that survives measurement is English-side
and mechanical: a lexical stop set (`I` + contractions) plus a positional fix so a quote-opening
word counts as sentence-initial.** `polish.md` P-019 carries the arms and the revised acceptance;
it is still `size S`. Hallucinated-but-genuine proper nouns stay unfixable by any such rule, and
`CONTEXT_EN_SUPPORT`=2 is the lever there. M12 is closed either way; this is a defect in shipped
behaviour, not a milestone.
The polish register also holds **P-014** (NPU decode-stall bursts, with M12.3's 6-section
counter-sample), pickable by `/session-polish` whenever. Standing options, none of them blocking:
(0) **Rule on P-019 above** — the only open defect in shipped behaviour, and (i) is a small unit.
(a) **A live-mic validation pass** — the user-only debt is the largest untested surface and only the
    user can run it: latency feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, and the whole
    VAC partial-caption cadence (`memory.md` § Smoke). Agent coverage cannot substitute here.
(b) **A maintenance pass** (L-018) — dependency/CVE sweep, lock bumps, full gate, re-verify the codex
    leg.
(c) **A new capability milestone** — needs a direction from the user, since `## Out of scope` closes
    most of the obvious ones.

(Last resolved: **M12.5 needed no ruling** — its acceptance named the verdict in advance and the
measurement returned the predicted branch, so M12 closed on the refutation without a user call.
Before that: **which M13 defect goes first** — the user ruled the translator mitigation ahead of
the recogniser work on 2026-09-03, and M13.1 shipped it the same day. Before that: **M12's remaining scope** — M12.1's census answered M12's two named modes without a
translator turn, leaving the question of what to fund next. Offered (a) run the arm matrix as
scoped on the 67-caption clip, (b) close M12 on the census and rule on P-012's fix on its merits, or
(c) acquire a longer corpus first, which is the only route to a dead-pairing test that can return
something other than zero. The user chose **(c) on 2026-09-03, all five remaining sections**, and
ruled the candidate-floor finding (ゴン invisible below `_TERM_RUN`'s 3-character katakana floor)
**in scope as its own unit** rather than a recorded limit. Before that: **the P-012 disposition** — the row asked whether `observe_en` survives real ASR
output, and re-sizing it against tree showed it was a milestone wearing a `size=M` label. Offered
(a) close it the way P-009 closed, accepting the learner with D-015's recorded clean-text limit, or
(b) fund it as a roadmap unit; the user chose **(b)** on 2026-09-02, to be carried out in the next
session, and authorized the two stale-line corrections in this file at the same time. Before that:
**assurance-apparatus scope** — measurement showed ~11,100 lines of apparatus
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
