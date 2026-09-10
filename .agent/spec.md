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
  stderr log; no hardware, weights or network, and it imports the shipped screen rather than
  restating it. `uv run python session_report.py [--log F] [--json]`.
- `tests/` — fast locks (`uv run pytest -q`) + on-demand evaluators `eval_cer.py`,
  `eval_long_form.py`, `eval_backpressure.py`, `eval_retention.py`, and `eval_latency.py`, the
  per-stage end-to-end latency budget (committed traces only, no hardware, <1 s).
- `transcripts/<local-start-time>.txt` — gitignored, one file per run, saving ON by default.
- `README.md` — the only human-facing doc, with the CLI strings in `live_stt.py`.
- `.agent/archive/` — the closed record, read on demand: `milestones-m1-m14.md` (M1-M14),
  `polish-register.md`, plus four evidence records.

## Decisions

Detail → `.claude/rules/`, which each `D-###` names.

- **D-009** STT fully local, no API keys. Codex absent or failing ⇒ JA-only degrade, never a cloud
  STT fallback. **D-011** translation = persistent `codex app-server`, `gpt-5.6-luna`.
- **D-016** whisper large-v3-turbo int8 on OpenVINO **NPU** is the shipped recogniser; `hotwords`
  forfeited with that choice. Engine/model selection is closed. **D-010** sherpa k2v2/parakeet stay
  as the `--engine` CPU fallback (VAD-segment decode, no partials).
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
- Japanese-only source. Personal-tool posture: `.claude/rules/assurance-posture.md` binds over the
  `CLAUDE.md` template, and `upstream-sync.md` names every override a refresh must not reinstate.
- **Out of scope, do not redebate:** config files / YAML / TOML for tunables · multi-mic mixing ·
  speaker diarization · web UI · auth / multi-user · metrics beyond the backlog/drop counters ·
  package split beyond `streaming.py` · CI mirroring the local hook · NPU for the sherpa fallbacks ·
  perf/W (RAPL unreadable in-container).

## Deferred

Queue → `.agent/deferred.md`, rank = funding order, acceptance written at deferral time, the funded
row = that unit's whole contract; the `/goal` body names the row it funds.

The spine, in funding order: **1** Cut the EN-leg thread-rotation tax · **2** Maintenance + security
pass · **3** Live-mic validation pass · **4** Probe the two open NPU constructor properties · **5**
Parameterize source language · **6** M10 candidate-screen remainder. The last two sit there because
their acceptance is a re-open condition, not work.

Blocking the spine: **Live-mic validation pass** is user-only (L-004) — M13.2, the four polish fixes
and both M14 recovery arms have never met a mic, so every agent-side claim about the live path stays
provisional until the user runs `live-smoke.md`.

## Phase

**IMPLEMENT.** M1-M14 shipped and closed (`.agent/archive/milestones-m1-m14.md`); 0.1.0 runs. The
latency budget is committed and re-derivable (`tests/eval_latency.py`, table in `asr-pipeline.md`).
Each `/goal` body I paste = one unit from `.agent/deferred.md`, closed under `python gate.py` plus, where
it touches decode quality, a CER number the commit body records.
