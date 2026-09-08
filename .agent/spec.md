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
- `gate.py` — THE gate, 6 blocking steps, hermetic. `uv run --no-sync python gate.py` (`--only NAME`,
  `-v`). Green = 6 pass + exactly 1 skip (the whisper NPU replay golden, which needs the accel
  prelude).
- `replay.py` — WAV → real `worker` replay, the "did the output change" harness (D-014).
  `uv run python replay.py WAV [--engine E] [--json]`; goldens in `tests/replay_goldens.json`.
- `cer.py` — shared `normalize`/`alignment`/`align` scoring primitive, pure stdlib.
- `session_report.py` — what a live session did, re-derived from `transcripts/*.txt` + a redirected
  stderr log; no hardware, weights or network, and it imports the shipped screen rather than
  restating it. `uv run python session_report.py [--log F] [--json]`.
- `tests/` — fast locks (`uv run pytest -q`) + on-demand evaluators `eval_cer.py`,
  `eval_long_form.py`, `eval_backpressure.py`, `eval_retention.py`.
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
- **D-012** judgment-review sessions are retired (L-032): a unit's check set closes inside its own
  session, no review ledger, no contract fingerprints, no claim registry, no mutation matrix.
- Utterances stay **UNCAPPED** — one utterance is one line and one turn, at any length.
- A runaway caption is **DROPPED whole**, never collapsed or truncated; the screen sits at
  PUBLICATION, upstream of every consumer. `repetition_penalty`=1.2 ships despite retention CER
  0.0583 → 0.0609. `CAPTION_REPEAT_UNIT_CHARS`=13. Language gate is **text-side only**
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

Rank = funding order; acceptance written at deferral time.

1. **Cut end-to-end latency on the shipped NPU path.** Today ≈2.5 s voice→JA, EN ≈1 s later.
   **Accept:** a committed measurement of the current per-stage budget (VAC update decode, commit
   lag, publication, translate turn) on a replayed WAV, then a shipped change that moves the total
   with retention CER no worse than 0.0609 and `tests/eval_backpressure.py` carry unchanged.
2. **Rule on `CAPTION_REPEAT_MAX_CHARS`=40's known false negatives.** Three live captions repeat a
   phrase exactly 4× and survive at 36 / 30 / 28 chars; the largest repetition a SPEAKER produced is
   20, so the usable range is 21..36. **Accept:** re-derive the live population at the candidate
   threshold, adjudicate every newly dropped caption as loop or speech, keep a ≥1.5× margin over the
   largest genuine repetition — or record the refusal with that margin as the reason.
   `tests/test_translator.py`'s corpus + boundary locks and `test_shipped_path.py`'s pass-list move
   with it.
3. **Maintenance + security pass** (L-018 recipe, `packaging-deps.md`). `sherpa-onnx` 1.13.4 →
   1.13.7, `sounddevice` 0.5.5 → 0.5.6, `ruff` 0.15.21 → 0.16.6; `numpy`, `openvino`,
   `openvino-genai`, `pytest` current. **Accept:** pip-audit clean, full gate green, codex leg
   re-verified against a real app-server, `uv.lock` committed.
4. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and BOTH M14 recovery arms have never met a mic; standing debt = latency
   feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence. **Accept:** the
   user runs `live-smoke.md` and reports; each item lands verified or defective.
5. **Parameterize source language** (T2.2) — Japanese-only by design. **Accept:** re-open only if the
   use-case expands.
6. **M10 candidate-screen remainder** — zipformer + SenseVoice lack current JA evidence,
   Moonshine-JA's license is unclear, ReazonSpeech-k2-v2 adds PyTorch/Transformers + remote custom
   model code. **Accept:** re-open only if the shipped path fails AND the added runtime surface buys
   a materially different hypothesis.

## Phase

**IMPLEMENT.** M1-M14 shipped and closed (`.agent/archive/milestones-m1-m14.md`); 0.1.0 runs. Each
`/goal` body I paste = one unit from `Deferred`, closed under `python gate.py` plus, where it touches
decode quality, a CER number the commit body records.
