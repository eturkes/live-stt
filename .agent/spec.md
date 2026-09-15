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
  `uv run live-stt` · `uv run live-stt --list-devices` · `--engine k2v2|parakeet` · `--asr-device`.
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
- `tests/` — fast locks (`uv run pytest -q`) + eight on-demand evaluators, inventory and per-file
  contract in `evidence-artifacts.md`: `eval_cer.py`, `eval_long_form.py`, `eval_backpressure.py`,
  `eval_retention.py`, `eval_translate_repeat.py` need weights, a corpus or the real translator;
  `eval_latency.py`, `eval_term_census.py` and `eval_en_pairing.py` replay committed traces and run
  in under a second in a fresh clone.
- `transcripts/<local-start-time>.txt` — gitignored, one file per run, saving ON by default.
- `README.md` — the only human-facing doc, with the CLI strings in `live_stt.py`.
- `.agent/archive/` — the closed record, read on demand: `milestones-m1-m14.md` (M1-M14),
  `polish-register.md`, plus four evidence records.
- [Audio-hang post-mortem](postmortems/2026-09-11-audio-driver-hang.md) — SoundWire kernel Oops,
  Linux session containment, regression evidence, and the successful user-run live check.

## Decisions

Detail → `.claude/rules/`, which each `D-###` names.

- **D-009** STT fully local, no API keys. Codex absent or failing ⇒ JA-only degrade, never a cloud
  STT fallback. **D-011** translation = persistent `codex app-server`, `gpt-5.6-luna`.
- **D-016** whisper large-v3-turbo int8 on OpenVINO **NPU** is the shipped recogniser; `hotwords`
  forfeited with that choice. Engine/model selection is closed, and so are its two constructor
  properties: `NPU_TURBO` buys a paired −2.3 ms per update for a ~126 s cold recompile and
  `NPUW_LLM_GENERATE_HINT="BEST_PERF"` SIGSEGVs loading its own cached blob — both REFUSED, never
  re-derive (`asr-pipeline.md`). **D-010** sherpa k2v2/parakeet stay as the `--engine` CPU fallback
  (VAD-segment decode, no partials).
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
- The status line shows the whole latest decode: committed text normal, the tail LocalAgreement-2
  still withholds **dimmed**. Time-to-FIRST-GLIMPSE 2.535 → 1.187 s p50, 8.157 → 2.385 s max, at zero
  compute and zero CER cost (retention CER 0.0609, unmoved). Time-to-SETTLED is unchanged at 2.535 s
  and the dim tail is rewritten on 105 of 180 updates (`redraws`, `eval_latency.py`) — quote the two
  numbers separately. The published line and the transcript stay committed-only + append-only.
- Utterances stay **UNCAPPED** — one utterance is one line and one turn, at any length.
- A runaway caption is **DROPPED whole**, never collapsed or truncated; the screen sits at
  PUBLICATION, upstream of every consumer. `repetition_penalty`=1.2 ships despite retention CER
  0.0583 → 0.0609. `CAPTION_REPEAT_UNIT_CHARS`=13; `CAPTION_REPEAT_MAX_CHARS`=40 is CLOSED —
  adjudicated over 1409 live captions and REFUSED, the tripling invariant flooring it at 40 and
  excluding the whole 31..36 admissible band. Language gate is **text-side only**
  (`latin > 4 × japanese`); a whisper LID gate is measured, feasible and REFUSED.
- EN-leg recovery = **respawn (app-server EOF) + cooldown re-probe (3-strike disable)**, one
  mechanism, arm picked by `_alive()`, 5-attempt budget, doubling cooldown, marked in the transcript.
- Transcripts save by default; `-o PATH` overrides, `--no-save` opts out.
- Linux live/device entry points isolate the audio session with deadlines and an inherited lock
  (L-010). This contains kernel audio hangs; it does not patch the SoundWire driver. Capture and the
  JA/EN drain remain in the session, validated by regression tests and the user-run live check.
- Source language is a flag, Japanese by default (`--source-lang ja|en`). `en` transcribes
  English directly and retires the text-side latin screen, which exists only because the
  recogniser is pinned to Japanese; the repetition screen still runs. `en` is transcribe-only:
  `TRANSLATOR_INSTRUCTIONS` pins the leg Japanese→English and declares every turn "one block
  of transcribed Japanese speech", so English input has no defined behaviour there.
  **Two-way translation is queued, research first, and live-stt must stay usable throughout it**
  (user ruling): every step lands behind a default-off flag, JA→EN green at every commit.
- Personal-tool posture: `.claude/rules/assurance-posture.md` binds over the
  `CLAUDE.md` template, and `upstream-sync.md` names every override a refresh must not reinstate.
- **Out of scope, do not redebate:** config files / YAML / TOML for tunables · multi-mic mixing ·
  speaker diarization · web UI · auth / multi-user · metrics beyond the backlog/drop counters ·
  package split beyond `streaming.py` · CI mirroring the local hook · NPU for the sherpa fallbacks ·
  perf/W (RAPL unreadable in-container).

## Deferred

Queue → `.agent/deferred.md`, rank = funding order, acceptance written at deferral time, the funded
row = that unit's whole contract; the `/goal` body names the row it funds.

The spine, in funding order: **1** Live-mic validation pass · **2** Explain the live audio drops ·
**3** Two-way translation (JA↔EN), research first · **4** Prove EN rendering consistency on the raw
stream · **5** Attribute drops that precede the first caption · **6** M10 candidate-screen remainder.
Row 6 sits there because its acceptance is a re-open condition, not work.

Blocking the spine: **Live-mic validation pass** is user-only (L-004) — M13.2, the four polish fixes
and both M14 recovery arms have never met a mic, so every agent-side claim about the live path stays
provisional until the user runs `live-smoke.md`.

## Phase

**MAINTAIN.** M1-M14 shipped and closed (`.agent/archive/milestones-m1-m14.md`); 0.1.0 runs, the
latency budget is committed and re-derivable (`tests/eval_latency.py`, table in `asr-pipeline.md`),
and IMPLEMENT closed with the spine above carrying what it did not fund.

One `/goal` per request, the body naming the `.agent/deferred.md` row it funds or the maintenance
task it wants, each closing gate-green under `uv run --no-sync python gate.py` with this file current
— plus, where the request touches decode quality, a CER number the commit body records. The standing
MAINTAIN work is the security review and dependency upgrade, whose recipe is L-018
(`packaging-deps.md`); Dependabot files the advisories between requests.

**Live-stt in its current form must stay usable through every unit** (user ruling): a feature that
takes more than one unit lands behind a default-off flag and the shipped JA→EN path is green at every
commit.
