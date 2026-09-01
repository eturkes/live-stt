# M11.1 adversarial review

## Checks

| ID | Verdict | Predicate | Evidence |
|---|---|---|---|
| R01 | unknown | Acceptance-contract coverage and touched-surface inventory are complete. | pending |
| R02 | unknown | Hard OpenVINO dependencies make the default Whisper engine installable; metadata states supported Python/platform bounds honestly. | pending |
| R03 | unknown | pyproject dependency declarations and uv.lock are mutually consistent without a stale whisper extra. | pending |
| R04 | unknown | Wheel and sdist include streaming.py and only the intended source/package metadata; include patterns are correctly anchored. | pending |
| R05 | unknown | Deleting speaking is behavior-identical at all nine former sites, including speech-open before processor assignment and finalize/error paths. | pending |
| R06 | unknown | Decoder callable typing matches every producer and consumer without weakening runtime behavior. | pending |
| R07 | unknown | WhisperPipeline.generate sample typing fix is semantically valid for every accepted sample representation. | pending |
| R08 | unknown | Evaluator recognizer TypeError handling catches only the intended compatibility failure and preserves other failures. | pending |
| R09 | unknown | Production and test Pyright gates cover the claimed files and pass without unjustified suppression. | pending |
| R10 | unknown | gate.py exposes exactly the contracted seven steps, correct order, blocking policy, aggregation, and process exit semantics. | pending |
| R11 | unknown | Each seeded-failure test exercises the real blocking-step wiring rather than merely causing an unrelated command failure. | pending |
| R12 | unknown | --only and --files cannot make a selected blocking step pass vacuously or silently omit required checks. | pending |
| R13 | unknown | touched_py() handles staged, unstaged, untracked, renamed, copied, and deleted Python paths correctly. | pending |
| R14 | unknown | Gate inventory locks and tests detect step drift, command drift, and blocking-policy drift. | pending |
| R15 | unknown | aggregate-only remains intentionally non-blocking while faithfully reporting the carried whole-file fingerprint defect. | pending |
| R16 | unknown | live_stt.py formatting changes preserve behavior and conform to D-006/L-001; remaining format debt is honestly scoped/deferred. | pending |
| R17 | unknown | replay.py claims match current behavior while the k2v2 default remains within the M11.3 boundary. | pending |
| R18 | unknown | The full gate result, blocking-step count, and non-blocking aggregate failure reproduce from the reviewable state. | pending |
| R19 | unknown | The claimed pytest count and skip count reproduce; new gate tests account for the delta. | pending |
| R20 | unknown | The claimed wheel member set, isolated import, and CLI help without Whisper dependencies are supported without prohibited rebuild/install work. | pending |
| R21 | unknown | The one-clip decode evidence establishes the intended backend/device behavior, including EXECUTION_DEVICES when P7.2 requires it. | pending |
| R22 | unknown | Documented installation/runtime instructions disclose the required Intel acceleration environment and do not overclaim plain-install sufficiency. | pending |
| R23 | unknown | README, roadmap, memory, and polish claims are mutually consistent with the code, measurements, acceptance contract, and milestone boundaries. | pending |
| R24 | unknown | Touched durable prose conforms to project Authoring rules: dense agent-facing text and natural ASD-STE100 human-facing text. | pending |
| R25 | unknown | Touched code conforms to project Engineering rules: tight scope, cause-level fixes, deterministic checks, and no unrequested M11.2–M11.6 work. | pending |

## Register

None recorded.
