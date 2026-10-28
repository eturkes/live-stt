---
paths:
  - "README.md"
  - "models/README.md"
  - "tests/test_human_docs.py"
---

# Human-facing documentation

**L-021 — `CLAUDE.md`'s human register (`Authoring`, Human-facing) applies to human-facing surfaces
ALONE.** Those surfaces are `README.md`, `models/README.md`, and the user-facing CLI strings in
`live_stt.py`. Everything else —
`.agent/*`, `.claude/rules/`, code comments — stays agent-dense, em-dashes
deliberate. Keep any literal duplicated between code and README in sync.

**The gate decides the dash rule and two literal sets; the rest of the register is judgment.**
`tests/test_human_docs.py` fails on an en or em dash in either README, in the rendered `--help`,
and in every string part of `live_stt.py`'s `print(...)` / `.error(...)` calls plus the preflight
messages `check_models` / `check_device` return (an AST scan, whose reach the playback error
proves); `test_the_readme_shows_the_playback_banner_the_session_prints` (`tests/test_playback.py`)
holds README's `Playback:` example to what `run_session` prints. Firing, each planted alone from a
`cp` snapshot and restored: an em dash in a README heading, an en dash in `models/README.md`, an en
dash in a `--help` string, an en dash in the supervisor's lock error, an em dash in `check_models`'
message, and the banner's decode rate drifted to 16001.
`test_the_readme_constant_table_names_live_constants_at_their_values` holds every `| `NAME` |
value |` row of README's constants table to `live_stt.NAME`: red on the README that still named
`TRANSLATE_MAX_RESPAWNS` / `TRANSLATE_RESPAWN_WAIT_S` after their rename, and on a planted
`VAC_CHUNK_S` of 2 s. **Coverage limit:** a string reaching the user any other way, and every
other duplicated literal, stay a reader's call.

`README.md` is the user doc: it covers install, CLI, behaviour and the coverage boundary.
`models/README.md` owns the weight-download commands and the engine layouts; engine-selection
rationale lives in `.claude/rules/asr-pipeline.md` (D-016 for the whisper default and its device, D-010
for the sherpa pair), so the READMEs point there rather than restating it.

Do not re-add superseded spike reports (D-005). The Gemini-era and metered-backend spike reports plus
the rejected `codex_ws/AGENTS.md` prompt mechanism were removed at user direction; git history preserves
them, and the live cloud→local trail is D-009 / D-010 / D-011 plus the roadmap ledger. Record a decision
in these rules instead.
