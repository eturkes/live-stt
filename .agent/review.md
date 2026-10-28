# Review ledger

## `--playback` — code review of `3fb2dbf..9b96e34`

Scope = the request's diff (steered Code review body; request: caption audio playback, e.g. a Zoom
call). Check set fixed before any reviewer read; rows = lens × module, conformance = lens × clause,
compound + risk-ranked (`assurance-posture.md`). Modules: M1 `live_stt.py` · M2
`tests/test_playback.py` · M3 `README.md` + `--help` · M4 `.claude/rules/{asr-pipeline,live-smoke}.md`
· M5 `.agent/{spec,deferred}.md`. Security stays with the Security review body (MAIN works any met).

| id | lens × target | subchecks | owner | verdict |
| --- | --- | --- | --- | --- |
| R01 | baseline × closing tree | full gate green; drive each runnable `Artifacts` entry the diff reaches: `live-stt --playback` (private stack), `--list-devices`, `--help`, `--playback --device`, `session_report.py` over a playback transcript, `replay.py` smoke | MAIN | pass — gate 8/8 green at every commit (2 declared `absent:` skips); `--playback` driven on the NPU twice per clip set (scratch + `eval_playback_routing.py --e2e`), `--list-devices` in the hw-hidden namespace, `--help`, `--playback --device 2` rc=2, `replay.py --engine whisper` over a playback capture (30 segments), 5 trace evaluators + `build_lag_trace.py --check` rc=0; register: `session_report.py` on a never-translated run → queue row 2 (340be4b) |
| R02 | Correctness × M1 `run_session` + `playback_device` | env set before import + every PCM open; exact-name lookup; `device` reaches `query_devices` + `InputStream`; error path opens nothing; mic output byte-identical; `_supervise_session` JSON round-trip; non-Linux path; `--two-way` / sherpa / `--source-lang` interplay; env inherited by the codex child; callers + callees of each changed function | reviewer-1 | pass (reviewer-1: 34 probes + an AST differential of the mic path, byte-identical) |
| R03 | Correctness × M1 CLI | parse contradiction; default off; `--list-devices` + `--playback`; help text vs behavior | reviewer-1 | pass (reviewer-1) |
| R04 | Cleanup × M1 + M2 | helpers reused (grep first); duplication (test harness vs `test_audio._run_session_with_fake_mic`); dead code; constants placement; comment density per house style | reviewer-1 | pass (reviewer-1) |
| R05 | Altitude × M1 | env-var mechanism vs a special case on shared infrastructure; the leveling limit queued vs a mechanism fix due now | reviewer-1 | pass (reviewer-1) |
| R06 | Verification integrity × M2 | red-first + L-022 records in the commit body; fake `sounddevice` fidelity (`query_devices()` shapes, `kind="input"`); a test passing for the wrong reason; gate detection; mic pin credit | reviewer-2 | findings P2-01 major → fixed 9b10138, P2-02 minor → credited (mutmut 3.8.0, rulings by reviewer-2; commit after 7ecb70b); the rulings also exposed the translator hermeticity seam → fixed 7ecb70b |
| R07 | Claim soundness × M3 + code comments | each README / `--help` / comment claim vs behavior + evidence (follows the default "also when it changes", needs PipeWire, Zoom `Same as System`, limitation numbers, one session at a time, every sound reaches the recognizer) | reviewer-2 | finding P2-03 minor → fixed 3e0ccb5 |
| R08 | Claim soundness × M4 + M5 | law + state claims vs evidence (versions, 42 gaps, CER arms, "proof" vs scratch evidence, agent-covered statements); guarantee-vs-claim gaps | reviewer-2 | finding P2-03 minor + stale host WirePlumber version (MAIN) → fixed 3e0ccb5 |
| R09 | Spec fit × behavior vs `Intent` + `Decisions` | STT local; NPU default; one-way usable (default OFF); out-of-scope list (multi-mic mixing); D-002 / D-006; `LeveledVad` ruling untouched; the new Decisions entry readable alone | reviewer-3 | pass (reviewer-3) |
| R10 | `CLAUDE.md` conformance × `human-docs.md` L-021 | ASD-STE100 register on README + CLI strings; no en/em dash in README; code↔README literals in sync; tool-decidable part → a gate check that fires | reviewer-3 + MAIN | findings F31 minor → fixed 9d6f0b3, F32 minor → fixed 38414de; class sweep found G31 → fixed 5faa961 |
| R11 | `CLAUDE.md` conformance × `toolchain.md` + `gate.md` | ruff / pyright (tests incl.) / secrets / `absent:` skip discipline → steps fire on their seeds | MAIN | pass — planted in the new test file, each step red alone: ruff-check (F401), ruff-format, pyright-tests (`x: int = "s"`), secrets (AWS-shaped key), pytest (bare skip → undeclared demotion) |
| R12 | `CLAUDE.md` conformance × `orientation.md` house style | comments carry the why; inline, no frameworks (L-005); edits name a failure mode (L-001); D-002 / D-006; constants at top = config surface | reviewer-3 | pass (reviewer-3) |
| R13 | `CLAUDE.md` conformance × law locks | `test_law_consistency.py` invariants (Tasks↔queue pairing, rank refs, queue links, ruling tags, `Intent` quotes) run + fire on a planted bad input | MAIN | pass — planted: Tasks title rename → `test_open_tasks_pair_every_queue_row_with_its_rank` red; dangling queue link → `test_every_queue_link_resolves_to_a_live_row` red |
| R14 | `CLAUDE.md` conformance × `assurance-posture.md` + `Engineering` runnable-change | L-004 "Did not verify" list; tier declared in the queue row; the real app driven to the changed behavior | reviewer-3 | pass (reviewer-3); tier: the unit ran at `kernel` depth (red-first locks, L-022, live drive, 3-reviewer pass); the L-004 list rides MAIN's closing message |
| R15 | `CLAUDE.md` conformance × `Authoring` + `Execution` evidence | new law/state text agent-dense, no origin story; empty-output findings carry rc + a positive control | reviewer-3 | finding F33 minor → fixed 265b6a5 + 9f513c8; the 11 pre-existing dated ruling tags → the closing question |
| R16 | `CLAUDE.md` conformance × `Engineering` durable-claim gate | scratch-local harnesses behind M4's claims → regeneration path in a `paths:` rule + port scheduled as a queue row | reviewer-3 | finding F34 major → fixed 265b6a5 (both harnesses committed as on-demand evaluators) |

### Findings

Reviewers: reviewer-1 (R02-R05, all pass), reviewer-2 (R06-R08), reviewer-3 (R09-R16); MAIN ran
R01, R11, R13 and adjudicated every row against the bytes it cites.

| id | sev | where | proof | fix |
| --- | --- | --- | --- | --- |
| P2-01 | major | `tests/test_playback.py:74` re-patched the spy `:141` set | red witness: one early load, shipped lock green | 9b10138; planted early load → 3 cases red |
| P2-02 | minor | `tests/test_playback.py:109` mic pin, green at 3fb2dbf, credited by hand-picked L-022 | `assurance-posture.md` "Hand-picked L-022 mutants never credit such a pin" | mutmut 3.8.0: 672 mutants, 420 survivors ruled by reviewer-2 (155 specified → all killed, 265 open recorded); commit after 7ecb70b |
| S-T8 | major | 4 fake-session harnesses never stubbed `CodexTranslator` | mutant 154 survived by reaching the real app-server path | 7ecb70b; flip → 11+ cases red |
| P2-03 | minor | `README.md:90`, `asr-pipeline.md:117`, `deferred.md:22` "loud … correct/clean" | e2e `SRC 7: 元はもう寝ました。`, `SRC 10: 7時です。` | 3e0ccb5 |
| F31 | minor | L-021 dash + README↔code literal had no gate check | planted em dash + rate drift accepted by all 6 README checks | 9d6f0b3; 6 plants red |
| F32 | minor | `live_stt.py:2768` `(pipewire-alsa), then check with --list-devices.` | two instructions, one sentence | 38414de |
| F33 | minor | `spec.md:183` `user request 10-28`, `asr-pipeline.md:91`, `deferred.md:23` | Authoring: omit dates | 265b6a5 + 9f513c8 |
| F34 | major | `asr-pipeline.md:99,119` cited `.scratch/playback/` | `git ls-tree 9b96e34`: no harness tracked | 265b6a5 |
| G31 | minor | `README.md:420-421` dead constant names, `:433` "Japanese-only" | scripted table sweep: 2 of 25 absent | 5faa961 (+ lock, red first) |
| R01-reg | register | `session_report.py:385-408` never-translated run → 22 "leg recovered" | baseline drive over the e2e transcript | queue row 2 (340be4b) |
