---
paths:
  - "README.md"
  - "models/README.md"
---

# Human-facing documentation

**L-021 — `CLAUDE.md`'s human register applies to human-facing surfaces ALONE.** Those surfaces are
`README.md`, `models/README.md`, and the user-facing CLI strings in `live_stt.py`. Everything else —
`.agent/*`, `.claude/rules/`, `.claude/commands/`, code comments — stays agent-dense, em-dashes
deliberate. Confirm the dash rule with `command grep -nP '[\x{2013}\x{2014}]' README.md`, and keep any
literal duplicated between code and README in sync.

`README.md` is the only shipped prose document: it covers install, CLI, behaviour and the coverage
boundary. `models/README.md` owns the weight-download commands and the engine layouts; engine-selection
rationale lives in `.claude/rules/asr-pipeline.md` (D-016 for the whisper default and its device, D-010
for the sherpa pair), so the READMEs point there rather than restating it.

Do not re-add superseded spike reports (D-005). The Gemini-era and metered-backend spike reports plus
the rejected `codex_ws/AGENTS.md` prompt mechanism were removed at user direction; git history preserves
them, and the live cloud→local trail is D-009 / D-010 / D-011 plus the roadmap ledger. Record a decision
in these rules instead.
