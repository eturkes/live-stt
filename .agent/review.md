# Review ledger

## `--playback` — code review of `3fb2dbf..9b96e34`

Scope = the request's diff (steered Code review body; request: caption audio playback, e.g. a Zoom
call). Check set fixed before any reviewer read; rows = lens × module, conformance = lens × clause,
compound + risk-ranked (`assurance-posture.md`). Modules: M1 `live_stt.py` · M2
`tests/test_playback.py` · M3 `README.md` + `--help` · M4 `.claude/rules/{asr-pipeline,live-smoke}.md`
· M5 `.agent/{spec,deferred}.md`. Security stays with the Security review body (MAIN works any met).

| id | lens × target | subchecks | owner | verdict |
| --- | --- | --- | --- | --- |
| R01 | baseline × closing tree | full gate green; drive each runnable `Artifacts` entry the diff reaches: `live-stt --playback` (private stack), `--list-devices`, `--help`, `--playback --device`, `session_report.py` over a playback transcript, `replay.py` smoke | MAIN | open |
| R02 | Correctness × M1 `run_session` + `playback_device` | env set before import + every PCM open; exact-name lookup; `device` reaches `query_devices` + `InputStream`; error path opens nothing; mic output byte-identical; `_supervise_session` JSON round-trip; non-Linux path; `--two-way` / sherpa / `--source-lang` interplay; env inherited by the codex child; callers + callees of each changed function | reviewer-1 | open |
| R03 | Correctness × M1 CLI | parse contradiction; default off; `--list-devices` + `--playback`; help text vs behavior | reviewer-1 | open |
| R04 | Cleanup × M1 + M2 | helpers reused (grep first); duplication (test harness vs `test_audio._run_session_with_fake_mic`); dead code; constants placement; comment density per house style | reviewer-1 | open |
| R05 | Altitude × M1 | env-var mechanism vs a special case on shared infrastructure; the leveling limit queued vs a mechanism fix due now | reviewer-1 | open |
| R06 | Verification integrity × M2 | red-first + L-022 records in the commit body; fake `sounddevice` fidelity (`query_devices()` shapes, `kind="input"`); a test passing for the wrong reason; gate detection; mic pin credit | reviewer-2 | open |
| R07 | Claim soundness × M3 + code comments | each README / `--help` / comment claim vs behavior + evidence (follows the default "also when it changes", needs PipeWire, Zoom `Same as System`, limitation numbers, one session at a time, every sound reaches the recognizer) | reviewer-2 | open |
| R08 | Claim soundness × M4 + M5 | law + state claims vs evidence (versions, 42 gaps, CER arms, "proof" vs scratch evidence, agent-covered statements); guarantee-vs-claim gaps | reviewer-2 | open |
| R09 | Spec fit × behavior vs `Intent` + `Decisions` | STT local; NPU default; one-way usable (default OFF); out-of-scope list (multi-mic mixing); D-002 / D-006; `LeveledVad` ruling untouched; the new Decisions entry readable alone | reviewer-3 | open |
| R10 | `CLAUDE.md` conformance × `human-docs.md` L-021 | ASD-STE100 register on README + CLI strings; no en/em dash in README; code↔README literals in sync; tool-decidable part → a gate check that fires | reviewer-3 + MAIN | open |
| R11 | `CLAUDE.md` conformance × `toolchain.md` + `gate.md` | ruff / pyright (tests incl.) / secrets / `absent:` skip discipline → steps fire on their seeds | MAIN | open |
| R12 | `CLAUDE.md` conformance × `orientation.md` house style | comments carry the why; inline, no frameworks (L-005); edits name a failure mode (L-001); D-002 / D-006; constants at top = config surface | reviewer-3 | open |
| R13 | `CLAUDE.md` conformance × law locks | `test_law_consistency.py` invariants (Tasks↔queue pairing, rank refs, queue links, ruling tags, `Intent` quotes) run + fire on a planted bad input | MAIN | open |
| R14 | `CLAUDE.md` conformance × `assurance-posture.md` + `Engineering` runnable-change | L-004 "Did not verify" list; tier declared in the queue row; the real app driven to the changed behavior | reviewer-3 | open |
| R15 | `CLAUDE.md` conformance × `Authoring` + `Execution` evidence | new law/state text agent-dense, no origin story; empty-output findings carry rc + a positive control | reviewer-3 | open |
| R16 | `CLAUDE.md` conformance × `Engineering` durable-claim gate | scratch-local harnesses behind M4's claims → regeneration path in a `paths:` rule + port scheduled as a queue row | reviewer-3 | open |
