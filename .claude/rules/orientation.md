# Orientation

Real-time Japanese STT + English translation. STT is fully local, no API keys (D-009). Personal
single-user tool at 0.1.0: performance is the bar — skip robustness and flexibility aimed at
unpredictable users or use-cases. One developer (agent), one user.

Pipeline: mic → `Resampler` → 8 s `AudioQueue` → silero VAD (on a leveled copy, `LeveledVad`) →
**VAC controller** (speech start opens a LocalAgreement-2 buffer in `streaming.py`; every
`VAC_CHUNK_S`=1 s re-decodes the open buffer and commits what two decodes agree on; speech end flushes the tail) → **whisper int8 on OpenVINO NPU**
(D-016: whisper-ja-760M for Japanese, large-v3-turbo for English) → `CodexTranslator` over a persistent `codex app-server` (D-011) → `emit_line`
→ stdout + `transcripts/<start-time>.txt`. Partial text renders on the meter status line; one
settled piece = one numbered `SRC` line = one translation turn — a trim publishes the committed text
it moved out of the buffer, speech end publishes the rest. `--engine k2v2|parakeet` selects the sherpa
fallback path (VAD-segment decode, D-010); `--asr-device` picks the OpenVINO device. Absent or failing
codex degrades to source-only — a hard requirement, never a cloud fallback.

## Files — read one only when the task implicates it

- `live_stt.py` (~1.9 K lines) — the whole app; the constants at its top are the config surface.
- `streaming.py` — the VAC hypothesis buffer, pure text-in/text-out.
- `gate.py` — the executable quality gate. `replay.py` — WAV → real `worker` replay (D-014).
  `cer.py` — the shared `normalize`/`alignment`/`align` scoring primitive, pure stdlib.
- `tests/` — fast locks, on-demand `eval_*.py` evaluators, committed corpora + traces.
- `models/` — gitignored weights; `models/README.md` carries the download commands.
- `transcripts/` — gitignored saved sessions, one file per run, saving ON by default.
- `README.md` — the user doc; it, `models/README.md` and the CLI strings in `live_stt.py` are the
  human-facing set (`human-docs.md`).
- `.agent/spec.md` = Intent · Artifacts · Decisions · Tasks · Phase; the sole attached state,
  imported by `CLAUDE.md` ⇒ it rides every session whole. Every line binds current or future work, so
  each unit prunes what its own close made dead; size is emergent. `.agent/deferred.md` = the
  deferral queue, `.agent/review.md` = the review ledger (`CLAUDE.md`), written once a review runs,
  and `.agent/archive/<record>.md` = closed-milestone detail — all committed, outside that attached
  set, read on demand.
- `.claude/rules/` = this law, the sole carrier of what a teammate must hold. `CLAUDE.md`,
  `.claude/agents/` and `.agent/spec-lint` = template copies, overwritten whole at each refresh ⇒
  never write a project delta into them. A repo ruling on a template clause lives in the rules file
  owning its subject as a ruling tag: the clause's quoted text + a bold verdict (`_RULING_TAG` in
  `tests/test_law_consistency.py` owns the form and holds each quote to live template text). A
  clause without one binds as written. `.scratch/` = gitignored session workspace, nothing durable.

Rulings on the template's spec layout (detail → `.agent/spec.md` `Decisions`):

- Template ruling: "the phase checklist: `- [ ]` open units in order" + "ticked rows cleared at each
  phase close + each MAINTAIN request's close" → **ADAPTED** (user rulings) — MAINTAIN has no phase
  close ⇒ `Tasks` = the open queue rows 1:1, rank + title verbatim, in queue order, then the queue
  pointer; a `- [x] <sha>` row lives only between one session's commits; a paused session adds its
  one `- [ ] RESUME: …` note, a routing session its one `- [ ] ROUTE: …` row. A session's own step
  checklist lives in `.scratch/tasks.md`; durable progress = commits + the paired rows.
- Template ruling: "Routed state: `Phase` line 1 = `<PHASE> — <scope>`" → **ADAPTED** at the `Tasks`
  lock alone — a routed phase's own checklist rows, beside a suspended MAINTAIN's queue rows, fail
  its 1:1 pairing ⇒ the routing session asks the user how the lock follows before it writes them.
- Template ruling: "detail → `docs/` or `.agent/archive/`" → **ADAPTED** — no `docs/`: decision
  detail → the `.claude/rules/` file each `D-###` names; closed records → `.agent/archive/`;
  incident records → `.agent/postmortems/`.

Template ruling: "omit provenance — dates, verification/discovery events, origin stories" →
**ADAPTED** (user ruling) — a user ruling keeps its date (`user ruling 10-09`) as its identifier:
spec and law cite one ruling from several files, and the date is what tells two rulings on one
subject apart. Every other provenance date stays out, as do discovery events and origin stories.

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
  mature guarded code correctly yields almost nothing — record the rejects in `spec.md` instead of
  manufacturing edits against the pink-elephant pull (L-019).
- D-002 — one file. A further split needs a cohesive one-way subsystem boundary named out loud;
  `streaming.py` is the one taken, on being pure text-in/text-out with no app config, state or
  shutdown coupling. The feeder/decoder stages share config, state, replay hooks and shutdown order
  ⇒ they stay together. Revisit on an actual navigation failure.
- A single user preference is state, not an `always` law: record placement and config choices with
  their source, and ask before reversing one (L-008).

`rg` skips dotdirs → pass `--hidden --glob '!.git/**'` to reach `.agent/` + `.claude/`.
