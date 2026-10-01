# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). The session body the user pastes names the
rows it funds, and a `maintain.md` Queue body naming none funds every row in rank order; closing a
row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and M14's `_respawn` arm have never met a mic (`_probe` has); standing
   debt = latency feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.

2. **Reconcile the human-facing doc set** — `.agent/spec.md` `Artifacts` and `orientation.md` call
   `README.md` the only human-facing doc, while `human-docs.md`, L-021's owner, names `README.md`,
   `models/README.md` and the CLI strings ⇒ an agent reading the first two can write
   `models/README.md` in agent register. Docs tier.
   **Accept:** `rg -n --hidden 'human-facing' .agent/spec.md .claude/rules/` → every hit names the
   same surface set as `human-docs.md` or points at it; the gate stays green.

3. **JA-tuned Whisper checkpoint** — most live errors are misrecognitions, not streaming artifacts
   (≤218 of 27,184 characters of the 10-01 run are adjacent repeats). User ruling: REOPEN D-016's
   checkpoint selection inside the same OpenVINO NPU pipeline. Candidates fixed before measuring
   (researcher-1): `efwkjn/whisper-ja-760M@00daeaf`, `kotoba-tech/kotoba-whisper-v2.0@7eb5752`,
   `efwkjn/whisper-ja-1.5B@72c13fb`, int8_asym like the shipped turbo. Kernel.
   **Accept:** (a) each candidate compiles on exact `NPU` and decodes with timestamps from a warm
   cache, or its blocker is recorded; (b) one harness for all (`.scratch/perf/model_eval.py`):
   FLEURS-ja 150 + FLEURS-en 150 whole-clip CER (S/D/I), `retention_probe` + long-form §01/§03 CER
   through the shipped VAC path, per-update decode p50/p90/max, jfk runaway; (c) WIN = lower JA CER
   on FLEURS-ja AND retention, long-form not worse by >0.005, per-update p90 ≤ 1.0 s on a quiet NPU;
   an EN CER rise >0.01 or an undeclared licence ⇒ user ruling before any swap; (d) a swap moves
   `models/README.md`, the model path, the whisper replay golden + retention CER figure (graders
   named here), `vac_decode_trace.json` + `caption_trace.json` and the rule numbers derived from
   them; no win ⇒ the table lands in `asr-pipeline.md` as the D-016 re-open record; the gate stays
   green.

4. **Streaming boundary artifacts** — live captions carry duplicated fragments
   (`大学院生が大学院生が`) and lost heads (`ュアル` for ビジュアル); user ruling: REOPEN the duplication
   closed after two attempts. Offline the committed traces show I=0 (no reproducer); one mechanism
   is witnessed — spans whose join outruns the stripped text let `process()` commit past `emitted`
   (`会議会議`, reviewer-1). Kernel.
   **Accept:** (a) instrumented replay classifies every insertion and deletion against references
   (`cer.alignment`) as same-length re-spelling, segment-sum overflow, normal trim cut, forced
   trim, final flush or elsewhere, on `retention_probe`, ≥2 long-form sections, `long.wav` and any
   user-recorded `--save-audio` session; (b) a fix that removes the `whisper/long` golden
   duplication (that golden moves — named here) and cuts boundary-attributed I+D, with retention
   CER ≤ 0.0609, long-form CER not worse, max buffer ≤ 12 s, forced trims 0, and a red lock
   separating a real re-spelling from span jitter; or (c) the failed attempt recorded with what it
   taught; the gate stays green.

5. **Screen each segment inside a released piece** — one trim can release several whisper segments
   as one piece, and the screen drops that piece whole, so a loop segment takes its clean
   neighbours with it (tester-1 trace: `繰り返し`×12 + one clean sentence, both dropped). Kernel.
   **Accept:** a piece is screened per released segment (segment counts from the trimming decode,
   never hypothesis text), the surviving segments publish as one line, a lock shows a loop segment
   dropped alone with its clean neighbour published, red on the unfixed worker; the gate stays
   green.

6. **Session report reads two-way transcripts** — a transcript records no per-line decode language,
   so `session_report.py` re-screens every source line under one `--source-lang`: an English line a
   two-way session published reads as a latin `declined` caption (reviewer-6, authored mixed
   transcript: `latin_drops=1`). Data tier.
   **Accept:** a two-way session's English source lines are never reported as screen declines — the
   transcript carries what the report needs, or the report takes a flag that skips the latin rule
   for two-way sessions; a lock over an authored mixed transcript, red before; the gate stays green.
