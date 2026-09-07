# Orientation

Real-time Japanese STT + English translation. STT is fully local, no API keys (D-009). Personal
single-user tool at 0.1.0: performance is the bar — skip robustness and flexibility aimed at
unpredictable users or use-cases. One developer (agent), one user.

Pipeline: mic → `resample` → 2 s `AudioQueue` → silero VAD → **VAC controller** (speech start opens a
LocalAgreement-2 buffer in `streaming.py`; every `VAC_CHUNK_S`=1 s re-decodes the open buffer and
commits what two decodes agree on; speech end flushes the tail) → **whisper large-v3-turbo int8 on
OpenVINO NPU** (D-016) → `CodexTranslator` over a persistent `codex app-server` (D-011) → `emit_line`
→ stdout + `transcripts/<start-time>.txt`. Partial text renders on the meter status line; one
utterance = one numbered `JA` line = one translation turn. `--engine k2v2|parakeet` selects the sherpa
fallback path (VAD-segment decode, D-010); `--asr-device` picks the OpenVINO device. Absent or failing
codex degrades to JA-only — a hard requirement, never a cloud fallback.

## Files — read one only when the task implicates it

- `live_stt.py` (~1.9 K lines) — the whole app; the constants at its top are the config surface.
- `streaming.py` — the VAC hypothesis buffer, pure text-in/text-out.
- `gate.py` — the executable quality gate. `replay.py` — WAV → real `worker` replay (D-014).
  `cer.py` — the shared `normalize`/`alignment`/`align` scoring primitive, pure stdlib.
- `tests/` — fast locks, on-demand `eval_*.py` evaluators, committed corpora + traces.
- `models/` — gitignored weights; `models/README.md` carries the download commands.
- `transcripts/` — gitignored saved sessions, one file per run, saving ON by default.
- `README.md` — the only human-facing doc, together with the CLI strings in `live_stt.py`.
- `.agent/roadmap.md` = plan + status · `.agent/polish.md` = deferred-perfection register. Both ride
  every session whole ⇒ keep them minimal. `.agent/archive/<record>.md` = closed-milestone detail,
  committed, outside that attached set, read on demand.
- `.claude/rules/` = this law, the sole carrier of what a teammate must hold. `CLAUDE.md` +
  `.claude/commands/` are refreshed byte-for-byte from upstream ⇒ never write a project delta into
  them (`assurance-posture.md`). `.scratch/` = gitignored session workspace, nothing durable.
- `.serena/` = committed project configuration; its own `.gitignore` owns `cache/` +
  `project.local.yml` (D-013). Serena memories carry no project memory; these rules do.

`D-###` / `L-###` ids anchor a code or test comment to the rule that explains it. They live in these
rules files, numbering monotonic with gaps where a rule died; keep an id while a citer still names it.

## House style

- No frameworks, DI, config systems or back-compat shims. Module-level imports. Inline by default —
  three similar lines beat a premature abstraction (L-005).
- Comments carry the WHY: the constraint, measurement or upstream quirk behind a peculiar decision.
  `live_stt.py`'s comments encode bench-derived rationale — preserve them, and do not re-densify the
  file for "LLM readability"; denser naming saves ~30 tokens at every call site's expense (D-006,
  L-001).
- Edit code only when you can name the failure mode the edit prevents (L-001). A refactor pass over
  mature guarded code correctly yields almost nothing — record the rejects in `roadmap.md` instead of
  manufacturing edits against the pink-elephant pull (L-019).
- D-002 — one file. A further split needs a cohesive one-way subsystem boundary named out loud;
  `streaming.py` is the one taken, on being pure text-in/text-out with no app config, state or
  shutdown coupling. The feeder/decoder stages share config, state, replay hooks and shutdown order
  ⇒ they stay together. Revisit on an actual navigation failure.
- A single user preference is state, not an `always` law: record placement and config choices with
  their source, and ask before reversing one (L-008).

`rg` skips dotdirs → pass `--hidden --glob '!.git/**'` to reach `.agent/`, `.claude/`, `.serena/`.
