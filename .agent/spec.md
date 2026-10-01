# live-stt — spec

## Intent

Personal real-time Japanese→English live captioner, one user, one machine. Japanese appears while I
speak (partials on the meter status line); one utterance = one numbered `JA n:` line = one
translation turn, `EN n:` ~1 s later; every run saves a transcript.

- STT stays fully local, no API keys, no cloud fallback ever: silero VAD → VAC controller → whisper
  large-v3-turbo int8 on the Intel NPU via OpenVINO.
- Translation rides my ChatGPT/Codex subscription through a persistent `codex app-server` (zero
  marginal cost). Absent or failing codex degrades to JA-only — a hard requirement.
- The CLI's UX is settled and approved. Remaining work = make the shipped path as robust and as
  performant as possible.
- Bar = performance + correctness for one known user on one known machine. Skip robustness and
  flexibility aimed at unpredictable users or use-cases.
- **Expanding to two-way**: translate Japanese→English AND English→Japanese, direction chosen per
  utterance. It arrives incrementally and **the shipped one-way tool stays usable the whole time** —
  I need live-stt working while that feature is built. `--source-lang en` is today's transcribe-only
  half of it.

## Artifacts

Container work carries `UV_PROJECT_ENVIRONMENT=.venv` (`toolchain.md`).

- `live_stt.py` — the app; the constants at its top are the whole config surface.
  `uv run live-stt` · `uv run live-stt --list-devices` · `--engine k2v2|parakeet` · `--asr-device` ·
  `--save-audio`.
- `streaming.py` — the VAC LocalAgreement-2 hypothesis buffer, pure text-in/text-out.
- `gate.py` — THE gate, 7 blocking steps, hermetic. `uv run --no-sync python gate.py` (`--only NAME`,
  `-v`). Green = 7 pass + exactly 1 skip (the whisper NPU replay golden, which needs the accel
  prelude).
- `replay.py` — WAV → real `worker` replay, the "did the output change" harness (D-014).
  `uv run python replay.py WAV [--engine E] [--json]`; goldens in `tests/replay_goldens.json`.
- `cer.py` — shared `normalize`/`alignment`/`align` scoring primitive, pure stdlib.
- `session_report.py` — what a live session did, re-derived from `transcripts/*.txt` + a redirected
  stderr log (`live-stt 2> stt.log`, which keeps the status line); no hardware, weights or network,
  and it imports the shipped screen rather than restating it. Answers include drop attribution, one
  row per `backlog peak:` drop increase. `uv run python session_report.py [--log F] [--json]
  [--source-lang L]`.
- `tests/` — fast locks (`uv run pytest -q`) + ten on-demand evaluators, inventory and per-file
  contract in `evidence-artifacts.md`: `eval_cer.py`, `eval_long_form.py`, `eval_backpressure.py`,
  `eval_retention.py`, `eval_translate_repeat.py` need weights, a corpus or the real translator;
  `eval_latency.py`, `eval_two_way_settled.py`, `eval_term_census.py`, `eval_en_pairing.py` and
  `eval_lag.py` replay committed traces and run in under a second in a fresh clone. Each runs as
  `uv run python tests/<name>.py`, `eval_long_form.py` adding `--with soundfile`; flags per script in
  `evidence-artifacts.md`.
- `tests/lag_sessions.json` + `tests/build_lag_trace.py` — the text-free reduction of the two live
  sessions the translation-lag claim rests on, and its regenerator. `transcripts/` is gitignored, so
  this is what keeps those numbers re-derivable once the recordings are gone.
  `uv run python tests/build_lag_trace.py [--check]`.
- `transcripts/<local-start-time>.txt` — gitignored, one file per run, saving ON by default.
- `README.md` — the only human-facing doc, with the CLI strings in `live_stt.py`.
- `.agent/archive/` — the closed record, read on demand: `milestones-m1-m14.md` (M1-M14),
  `polish-register.md`, plus four evidence records.
- [Audio-hang post-mortem](postmortems/2026-09-11-audio-driver-hang.md) — SoundWire kernel Oops,
  Linux session containment, regression evidence, and the successful user-run live check.

## Decisions

Detail → `.claude/rules/`, which each `D-###` names.

- **D-009** STT fully local, no API keys. Codex absent or failing ⇒ source-only degrade, never a
  cloud STT fallback. **D-011** translation = persistent `codex app-server`, `gpt-6-luna`.
- **D-016** Whisper int8 on OpenVINO **NPU** is the shipped recogniser — whisper-ja-760M for
  Japanese, large-v3-turbo for English (user ruling after the checkpoint re-open; the turbo
  wording in Intent is the user's to edit); `hotwords` forfeited with the NPU. Engine selection
  stays closed, and
  so are its two constructor properties: `NPU_TURBO` buys a paired −2.3 ms per update for a ~126 s cold recompile and
  `NPUW_LLM_GENERATE_HINT="BEST_PERF"` SIGSEGVs loading its own cached blob — both REFUSED, never
  re-derive (`asr-pipeline.md`). **D-010** sherpa k2v2/parakeet stay as the `--engine` CPU fallback
  (VAD-segment decode, no partials).
- **JA-tuned checkpoint** (user rulings): decided AFTER the
  boundary-artifact fix, re-measured under the fixed policy; `whisper-ja-760M` won every JA leg and
  is ADOPTED: it decodes Japanese and turbo stays resident for English (`--two-way`, `--source-lang
  en`); its undeclared licence is acceptable for this personal, local-only tool. NPU, one harness,
  fixed processor, turbo → 760M: FLEURS-ja 0.0499 → 0.0469, retention 0.0592 → 0.0532, long-form
  §01+§03 0.2486 → 0.2153, FLEURS-en 0.0255 → 0.0482, quiet per-update p90 0.761 s. kotoba-v2.0 eliminated
  on every arm (EN 0.9945, retention 0.2238); 1.5B by the first WIN leg (FLEURS-ja 0.0560 > 0.0499),
  its VAC/long-form/jfk arms deliberately not run (~9x slower decode = real-time leg hopeless).
- **D-002** one file: `live_stt.py` + `streaming.py`. A further split needs a cohesive one-way
  subsystem boundary named out loud. **D-006** never re-densify `live_stt.py` for "LLM readability".
- **D-014** deterministic WAV replay is the regression harness. **D-015** `observe_en` learns an
  English rendering keyed on the JA string the recogniser produced.
- **D-007** pre-commit via `.githooks/` + `core.hooksPath`, not the `pre-commit` framework.
- **D-017** security scanning splits by hermeticity: ruff `S` (in `ruff-check`) + a `detect-secrets`
  `secrets` step run offline in the gate; `pip-audit` stays in the L-018 recipe. Update automation =
  `.github/dependabot.yml` (`uv`, weekly, grouped), the only `.github/` file — CI stays REJECTED
  (`upstream-sync.md`). Scan scoping + its coverage limit → `toolchain.md`.
- **D-012** judgment-review sessions are retired (L-032): a unit's check set closes inside its own
  session, no review ledger, no contract fingerprints, no claim registry, no mutation matrix.
- **The published line grammar is `SRC n:` / `TGT n:`, in every mode** (user ruling) — `SRC` names
  the line that was spoken, `TGT` the line that was translated, screen and transcript alike. It
  supersedes the `JA n:` / `EN n:` wording still in `Intent`, which is the user's to edit. Two-way
  makes language tags unstable — an English utterance would publish `EN` first — so the tags name
  roles instead. `session_report.py` keeps reading legacy `JA`/`EN` transcripts, mapping `JA` to
  source and `EN` to translation: the first live session's saved transcript is in that grammar and a
  queued unit closes by comparing against it. An utterance whose language was HELD rather than
  detected marks its source line `SRC n <!>: text`; the `TGT` line stays unmarked.
- Two-way rejects a contradictory invocation at parse time rather than silently winning (user
  ruling): `--two-way` with `--source-lang` errors, because one asks for a per-utterance decision and
  the other pins every utterance, and `--two-way` with `--engine k2v2|parakeet` errors, both sherpa
  models being Japanese-only. This SUPERSEDES the queue row's "`--source-lang` stays the manual
  override and pins the label outright when given". `--source-lang` without `--two-way` is unchanged.
- The status line shows the whole latest decode: committed text normal, the tail LocalAgreement-2
  still withholds **dimmed**. Time-to-FIRST-GLIMPSE 2.353 → 1.237 s p50, 6.402 → 2.476 s max, at zero
  compute and zero CER cost (whisper-ja-760M trace; turbo's read 2.535 → 1.187, 8.157 → 2.385).
  Time-to-SETTLED is unchanged at 2.353 s and the dim tail is rewritten on 90 of 180 updates
  (`redraws`, `eval_latency.py`) — quote the two numbers separately. The published line and the transcript stay committed-only + append-only.
- **The published boundary is aligned, never counted** (user ruling reopened the closed
  duplication): a decode that re-spells published text no longer re-commits or skips a character.
  NPU retention CER turbo 0.0609 → 0.0592, whisper-ja-760M 0.0626 → 0.0532; the `whisper/long`
  golden lost its duplication; mechanism, trim rules + ambiguous cases → `asr-pipeline.md`.
- **Long utterances publish at settled segments** (user ruling; supersedes UNCAPPED): each trim
  that moves committed text out of the VAC buffer publishes it at once as its own `SRC n` line + turn,
  the remainder at speech end; no length cap exists. Voice → `SRC` per character on
  `retention_probe` p50 13.15 → 5.67 s, max 33.46 → 10.29 s (`eval_latency.py`, whisper-ja-760M;
  turbo's trace read 13.08 → 5.55, 33.20 → 11.98). The screen judges
  each piece, the learner observes once per utterance; mechanics → `asr-pipeline.md`. The
  one-utterance-one-line wording in Intent is the user's to edit.
- A runaway caption (each published piece is one) is **DROPPED whole**, never collapsed or truncated; the screen sits at
  PUBLICATION, upstream of every consumer. `repetition_penalty`=1.2 ships despite retention CER
  0.0583 → 0.0609 (measured on turbo). `CAPTION_REPEAT_UNIT_CHARS`=13; `CAPTION_REPEAT_MAX_CHARS`=40 is CLOSED —
  adjudicated over 1409 live captions and REFUSED, the tripling invariant flooring it at 40 and
  excluding the whole 31..36 admissible band. Language gate is **text-side only**
  (`latin > 4 × japanese`, on text DECODED under `ja` — a two-way piece decoded under `en` skips it);
  a whisper LID gate is measured, feasible and REFUSED.
- EN-leg recovery = **respawn (app-server EOF) + cooldown re-probe (3-strike disable)**, one
  mechanism, arm picked by `_alive()`, 5-attempt budget, doubling cooldown, marked in the transcript.
- A queued caption whose translation would land long after it was spoken is **published
  source-only**: `TRANSLATE_MAX_STALENESS_S`=15.0, read at DEQUEUE and only AFTER the recovery block,
  since `run()` alone drives `_recover()` and an earlier check would strand a down leg behind a stale
  queue. One `logger.warning` plus one `-- translation skipped (stale): n` transcript note per
  caption, counted in `stale_translations` and shown as the meter's own `tstale=` — a third
  category, never folded into `tdrop=` (backlog eviction) or `tskip=` (content). Over the two
  committed live sessions the bound drops 8 of 696 and 0 of 143, every drop inside one
  wedge-and-recover cascade (`translation-leg.md`).
- Capture holds `AUDIO_HEADROOM_S`=**8 s** and a slow decoder CATCHES UP rather than stacking (user
  ruling): an update due while more than `VAC_BACKLOG_S`=0.5 s of capture is queued waits for the
  drain, so the cadence stretches to the decode. That pair is the mechanism + fix measured on the
  09-24 capture's drops; shapes, numbers and the `live` backpressure arm → `asr-pipeline.md`.
- Capture resamples through ONE band-limited soxr stream per session (`Resampler`), drained at stop;
  the per-block linear recipe survives as `linear_resample` for the hash-pinned corpora alone. No
  CER change on 300 clean clips (0.1131 → 0.1113, CI spans 0); shipped as the correctness fix
  (`asr-pipeline.md`).
- Transcripts save by default; `-o PATH` overrides, `--no-save` opts out. `--save-audio` (opt-in)
  adds the capture WAV `transcripts/<start>.wav` (~115 MB/h), the real-audio input `replay.py` reads.
- Linux live/device entry points isolate the audio session with deadlines and an inherited lock
  (L-010). This contains kernel audio hangs; it does not patch the SoundWire driver. Capture and the
  SRC/TGT drain remain in the session, validated by regression tests and the user-run live check.
- Source language is a flag, Japanese by default (`--source-lang ja|en`). `en` transcribes
  English directly and retires the text-side latin screen, which exists only because the
  recogniser is pinned to Japanese; the repetition screen still runs. `en` is transcribe-only:
  `TRANSLATOR_INSTRUCTIONS` pins the leg Japanese→English and declares every turn "one block
  of transcribed Japanese speech", so English input has no defined behaviour there.
  **Two-way translation SHIPS behind `--two-way`, default OFF**: with the flag absent the process
  constructs today's JA pipeline, builds ONE translation leg and decodes under `ASR_LANGUAGE`. The
  architecture the research picked is law — an EXPLICIT language token per utterance, settled by a
  standalone ECAPA LID on ONNX Runtime CPU (`asr-pipeline.md`), routes each decode to whisper-ja-760M
  (`ja`) or a resident large-v3-turbo (`en`, loaded under `--two-way` alone; user ruling), the settled
  label also selecting one of two immutable direction-specific threads (`translation-leg.md`), the
  reverse one opening LAZILY on its own first turn. The flag decides HOW MANY legs exist and nothing
  else, so no `--two-way` conditional sits inside `CodexTranslator`. One app-server is behind both
  directions ⇒ its EOF disables both, while a poisoned turn replaces only its own direction's
  thread, and `TRANSLATOR_INSTRUCTIONS_EN` is UNMEASURED (`translation-leg.md`). Under
  the flag a turn renders wholly DIM and commits nothing until its label is accepted, and **no caption
  is ever withheld** — an utterance that never accepts publishes under the held label. Every decode
  names its language explicitly and each `StreamingProcessor` decodes under the token it was built
  with; the session fallback is `ASR_LANGUAGE` in every mode, so one-way `--source-lang en` decodes
  under `"<|en|>"` rather than a literal Japanese token.
- Personal-tool posture: `.claude/rules/assurance-posture.md` binds over the
  `CLAUDE.md` template, and `upstream-sync.md` names every override a refresh must not reinstate.
- MAINTAIN has no phase close ⇒ each request session is its own (user ruling): the commit closing a
  funded row deletes it from `.agent/deferred.md` and from `Tasks` together, and a `- [x] <sha>` row
  lives only between the commits of one session. Every open `Tasks` row is one queue row, rank and
  title verbatim, in queue order — a find on the unit's path lands as a queue row with its acceptance
  check plus a `Tasks` row, in one commit. `tests/test_law_consistency.py` locks the pairing.
- **Out of scope, do not redebate:** config files / YAML / TOML for tunables · multi-mic mixing ·
  speaker diarization · web UI · auth / multi-user · metrics beyond the backlog/drop counters ·
  package split beyond `streaming.py` · CI mirroring the local hook · NPU for the sherpa fallbacks ·
  perf/W (RAPL unreadable in-container).

## Tasks

- [ ] **1** Live-mic validation pass
  - User-only (L-004); accept → `.agent/deferred.md` → *Live-mic validation pass*: the user runs
    `live-smoke.md` and reports, each item landing verified or defective.
  - Blocks the rest: M13.2, the four polish fixes and M14's `_respawn` arm have never met a mic, so
    every agent-side claim about the live path stays provisional until the user runs
    `live-smoke.md`. M14's `_probe` arm is now the one exception: it fired on a real mic on
    2026-09-18 and recovered the leg on attempt 1 (`translation-leg.md`).
- [ ] **2** Reconcile the human-facing doc set
  - Docs tier; accept → `.agent/deferred.md` → *Reconcile the human-facing doc set*: every
    `human-facing` statement in `.agent/spec.md` + `.claude/rules/` names `human-docs.md`'s set.
- [ ] **3** Screen each segment inside a released piece
  - Kernel tier, unfunded; accept → `.agent/deferred.md` → *Screen each segment inside a released piece*.
- [ ] **4** Session report reads two-way transcripts
  - Data tier, unfunded; accept → `.agent/deferred.md` → *Session report reads two-way transcripts*.

Queue → `.agent/deferred.md`: rank = funding order, acceptance written at deferral time, the funded
row = that unit's whole contract; the session body names the rows it funds, and a `maintain.md`
Queue body naming none funds every row.

## Phase

**MAINTAIN — scope: the whole product** (user ruling), the one-way JA→EN captioner plus two-way
behind its default-off `--two-way` flag. M1-M14 shipped and closed
(`.agent/archive/milestones-m1-m14.md`); 0.1.0 runs, the latency budget is committed and
re-derivable (`tests/eval_latency.py`, table in `asr-pipeline.md`), and IMPLEMENT closed with `Tasks`
above carrying what it did not fund.

One session per `maintain.md` body — Request, Queue, Security review, Dependency upgrade — the body
naming the `.agent/deferred.md` rows it funds or the maintenance task it wants, each closing
gate-green under `uv run --no-sync python gate.py` with this file current
— plus, where the request touches decode quality, a CER number the commit body records. The standing
MAINTAIN work is the security review and dependency upgrade, whose recipe is L-018
(`packaging-deps.md`); Dependabot files the advisories between requests.

**Live-stt in its current form must stay usable through every unit** (user ruling): a feature that
takes more than one unit lands behind a default-off flag and the shipped JA→EN path is green at every
commit.
