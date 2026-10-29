# Review ledger

Closed reviews → `.agent/archive/review-*.md`.

## Definitive code review — the whole implementation at `56c9eae`

Scope = the implementation (steered `Code review` body, no narrower request): code, tests, gates,
config + docs. Check set fixed + committed before any reviewer read; rows = lens × module, conformance
= lens × clause (inventory below, one row per law paragraph / bullet / code block of `CLAUDE.md` +
`.claude/rules/*.md` at `56c9eae`). Security stays with the `Security review` body: out of every
brief; a security defect met on another lens = MAIN's finding alone. Template copies (`CLAUDE.md`,
`.claude/agents/`, `.agent/spec-lint`) are law, not targets: a flaw in one → named to the user.
The check set precedes every reviewer read; MAIN read M5a's `gate.py` + hook and M5b's config files
in full beforehand, to discover the gate command — reviewer-14/15/31 own those rows.

Severity = `blocker` | `major` | `minor`; register = outside every row's contract → a
`.agent/deferred.md` row. Proof = a red test (command + failing output at `56c9eae`) for behavior,
anchor bytes `rg -Fn` matches + `file:line` for text. Verdict = `pass` | `finding <ids>` |
`n/a (<why>)`; a row closes once MAIN has adjudicated every finding it carries against the bytes.

### Modules + tiers

Tier = where a defect is re-checked for free (`CLAUDE.md` `Engineering`); `kernel` → two independent
reviewers on the same rows.

| id | module | tier |
| --- | --- | --- |
| M1a | `live_stt.py` capture front: the constants block (config surface), `Resampler`, `resample`, `linear_resample`, `RingBuffer`, `LeveledVad`, `make_vad`, `State`, `AudioQueue`, `enqueue_audio`, `submit_audio_sentinel`, `_feed_segments`, `playback_device` | standard (replay goldens + backpressure locks re-check) |
| M1b | `live_stt.py` recognisers + language: `WhisperEngine`, `load_recognizer`, `_decode`, `_split_decode_segment`, `_merge_chunk_text`, `lid_accept`, `LanguageDetector`, `check_models`, `name_opencl_vendors`, `check_device`, `_decode_segments` | kernel (a wrong language token is catastrophic, unseen until read live) |
| M1c | `live_stt.py` publication: `settled_boundary`, `repeat_span`, `caption_defect`, `drop_caption`, `write_stdout`, `emit_line`, `emit_note`, `caption_body`, `_vac_segments`, `worker` | kernel (the published line is the product; nothing downstream re-checks it) |
| M1d | `live_stt.py` translation leg: `_en_names`, `SessionContext`, `failure_cause`, `_Leg`, `CodexTranslator` | kernel (the degrade contract is a hard requirement) |
| M1e | `live_stt.py` session + CLI: `_StderrFormatter`, `_configure_logging`, `log_session_marker`, `session_stamp`, `transcript_path`, `audio_path`, `log_path`, `redirect_stderr`, `TranscriptFile`, `AudioRecording`, `_backlog`, `meter`, `_write_session_status` … `_session_child`, `run_session`, `_install_signal_handlers`, `main`, `_run_cli` | standard |
| M2 | `streaming.py` | kernel (the published boundary) |
| M3 | `session_report.py` | data (its reader re-derives from the transcript) |
| M4 | `replay.py` + `cer.py` | kernel (D-014's harness + the CER primitive grade every decode claim) |
| M5a | `gate.py`, `.githooks/pre-commit`, `tests/test_gate.py`, `tests/test_law_consistency.py` | kernel (nothing re-checks the gate) |
| M5b | `pyproject.toml`, `uv.lock` (bounded queries), `.gitignore`, `.envrc`, `.github/dependabot.yml`, `.python-version` | data (uv + git re-validate at use) |
| M6a | publication locks: `tests/test_{streaming,boundary_anchor,thin_rewrite,retell_guard,trim_echo,trim_republish,split_publication,dropped_mark,line_grammar,two_way_screen}.py` | standard |
| M6b | translator locks: `tests/test_{translator,context,en_pairing,term_census,rotation_tax,turn_correlation,reverse_leg,staleness}.py` | standard |
| M6c | audio locks: `tests/test_{audio,audio_startup,backpressure,vac_catchup,vad_leveling,resampler,playback}.py` | standard |
| M6d | routing + session locks: `tests/test_{shipped_path,gpu_mode,model_routing,language_token,language_detector,language_schedule}.py` | standard |
| M6e | evidence + report + doc locks: `tests/test_{asr_screen,cer,corpus,en_corpus,fleurs_parallel,human_docs,lag_trace,lid_census,replay,retention_probe,session_report}.py` | standard |
| M7a | trace evaluators: `tests/eval_{latency,two_way_settled,term_census,en_pairing,lag,anchor_scenarios}.py` | data |
| M7b | hardware evaluators: `tests/eval_{cer,long_form,backpressure,retention,asr_screen,playback_silence,playback_routing,translate_repeat}.py` | data |
| M7c | builders + corpora: `tests/build_*.py`, `tests/fetch_*.py`, `tests/gen_replay_goldens.py`, `tests/*.json` (bounded queries), `tests/__init__.py` | data |
| M8 | human docs: `README.md`, `models/README.md`, `--help` + every user-facing CLI string | docs |
| M9 | state: `.agent/spec.md` `Artifacts` + `.agent/deferred.md` | docs |

### Lenses (subchecks every row of that lens carries)

- **BL baseline** — the full gate + each runnable `Artifacts` entry driven by its run command; a red
  step or failed run = a finding. MAIN, the sole NPU/GPU/codex holder.
- **CO Correctness** — each function read with its callers + callees; each function's contract vs
  every caller's use; invariants hold on every path (error, early return, cancellation, shutdown);
  the language's classic pitfalls (mutable defaults, late-binding closures, async cancellation +
  task leaks, swallowed exceptions, off-by-one, float equality, encoding, integer/float division);
  wrappers route to the wrapped instance; config defaults set by intent; resource open/close
  symmetric.
- **CU Cleanup** — existing helpers reused (grep before judging); derivable state computed, never
  duplicated; near-duplicate blocks; dead code; independent steps parallel; startup + hot paths
  lean; closures capture only what they use; house style (`orientation.md`: inline by default
  L-005, no frameworks, module-level imports bar L-010's device entry points, D-006 no
  re-densification) — every finding names the failure mode it prevents (L-001, L-019).
- **AL Altitude** — a special case on shared infrastructure that signals a fix due in the mechanism.
- **CS Claim soundness** — each claim (comment, docstring, README, `--help`, `Artifacts`, law) vs
  behavior + the evidence it cites (numbers, names, paths, flags resolve); guarantee-vs-claim gaps.
- **VI Verification integrity** (`CLAUDE.md` `Engineering` + `assurance-posture.md`) — each test can
  fail through observable behavior (the production mutation that reddens it named, and run wherever
  reading leaves doubt); no gate detection, no fixture's expected value returned, no expected-output
  table the contract does not own; skip discipline (`absent:`); hermetic: no network, hardware or
  real codex outside an `absent:`-gated case; fakes faithful to the real API's shapes; setup +
  teardown symmetric. Per test file two columns: *can fail* (verified) and *credit on record*
  (red-first / L-022 / mutation tool in the body of the commit that added it: present | absent). A
  test that cannot fail = a finding fixed here; an absent record = a tally, its fix (the stack's
  mutation tool per file, survivors ruled) sized at harvest — past this session → the route
  question.
- **SF Spec fit** — behavior vs each `Intent` clause + each `Decisions` entry; the out-of-scope list
  unviolated; one-way usable with every flag off.
- **CF Conformance** — per clause (inventory below): binds a tracked file? → tool-decidable → its gate
  check named, run, and fired on a planted bad input in the reviewer's worktree (rc + failing line);
  decidable but unchecked → a finding carrying the check's name + its plant recipe, MAIN wiring it
  per class (a class past this session → the route question); a lock holding law prose equal to
  code constants is no such check — it turns law into a grading table; decidable only on hardware → the
  on-demand evaluator named (hermetic gate, `toolchain.md`); judgment → the bound files conform or a
  finding; binds no tracked file → `n/a`.

### Rows

| id | lens × module | owner | verdict |
| --- | --- | --- | --- |
| B01 | BL × full gate: `gate.py` 8/8, 2 declared skips; under the accelerator prelude 0 skips | MAIN | open |
| B02 | BL × `live-stt` CLI: `--help`, `--list-devices`, each parse-time error, `--playback` end to end (`tests/eval_playback_routing.py --e2e pinned` + `probe`), mic paths → L-004 list | MAIN | open |
| B03 | BL × `replay.py`: k2v2, parakeet, whisper NPU + GPU, `--json`; the replay goldens | MAIN | open |
| B04 | BL × `session_report.py`: bare, `--log`, `--json`, `--source-lang en`, `--two-way` over real transcripts | MAIN | open |
| B05 | BL × trace evaluators + `build_lag_trace.py --check` (M7a's run commands) | MAIN | open |
| B06 | BL × hardware evaluators (M7b's run commands, flags per `evidence-artifacts.md`) | MAIN | open |
| B07 | BL × library + hook entries: `streaming` / `cer` import-and-call smoke; `.githooks/pre-commit`; `uv lock --check` | MAIN | open |
| C01 | CO × M1a | reviewer-1 | open |
| C02 | CU × M1a | reviewer-1 | open |
| C03 | AL × M1a | reviewer-1 | open |
| C04 | CS × M1a comments + docstrings | reviewer-1 | open |
| C05 | CO × M1b | reviewer-2, reviewer-3 | open |
| C06 | CU × M1b | reviewer-2, reviewer-3 | open |
| C07 | AL × M1b | reviewer-2, reviewer-3 | open |
| C08 | CS × M1b comments + docstrings | reviewer-2, reviewer-3 | open |
| C09 | CO × M1c | reviewer-4, reviewer-5 | open |
| C10 | CU × M1c | reviewer-4, reviewer-5 | open |
| C11 | AL × M1c | reviewer-4, reviewer-5 | open |
| C12 | CS × M1c comments + docstrings | reviewer-4, reviewer-5 | open |
| C13 | CO × M1d | reviewer-6, reviewer-7 | open |
| C14 | CU × M1d | reviewer-6, reviewer-7 | open |
| C15 | AL × M1d | reviewer-6, reviewer-7 | open |
| C16 | CS × M1d comments + docstrings | reviewer-6, reviewer-7 | open |
| C17 | CO × M1e | reviewer-8 | open |
| C18 | CU × M1e | reviewer-8 | open |
| C19 | AL × M1e | reviewer-8 | open |
| C20 | CS × M1e comments + docstrings | reviewer-8 | open |
| C21 | CO × M2 | reviewer-9, reviewer-10 | open |
| C22 | CU × M2 | reviewer-9, reviewer-10 | open |
| C23 | AL × M2 | reviewer-9, reviewer-10 | open |
| C24 | CS × M2 comments + docstrings | reviewer-9, reviewer-10 | open |
| C25 | CO × M3 | reviewer-11 | open |
| C26 | CU × M3 | reviewer-11 | open |
| C27 | AL × M3 | reviewer-11 | open |
| C28 | CS × M3 comments, docstrings + `--help` | reviewer-11 | open |
| C29 | CO × M4 | reviewer-12, reviewer-13 | open |
| C30 | CU × M4 | reviewer-12, reviewer-13 | open |
| C31 | AL × M4 | reviewer-12, reviewer-13 | open |
| C32 | CS × M4 comments, docstrings + `--help` | reviewer-12, reviewer-13 | open |
| C33 | CO × M5a | reviewer-14, reviewer-15 | open |
| C34 | CU × M5a | reviewer-14, reviewer-15 | open |
| C35 | AL × M5a | reviewer-14, reviewer-15 | open |
| C36 | CS × M5a comments + docstrings | reviewer-14, reviewer-15 | open |
| C37 | VI × M5a: each step fires on its seed through the real gate; detection limits as `gate.md` records | reviewer-14, reviewer-15 | open |
| C38 | CO + CU × M5b: wheel/sdist include sets, ruff/pyright/pytest config vs the gate, ignore coverage, `.envrc` vs `_session_env`, dependabot vs `D-017` | reviewer-31 | open |
| C39 | VI × M6a | reviewer-16 | open |
| C40 | CU × M6a | reviewer-16 | open |
| C41 | VI × M6b | reviewer-17 | open |
| C42 | CU × M6b | reviewer-17 | open |
| C43 | VI × M6c | reviewer-18 | open |
| C44 | CU × M6c | reviewer-18 | open |
| C45 | VI × M6d | reviewer-19 | open |
| C46 | CU × M6d | reviewer-19 | open |
| C47 | VI × M6e | reviewer-20 | open |
| C48 | CU × M6e | reviewer-20 | open |
| C49 | CO + CU × M7a | reviewer-21 | open |
| C50 | VI × M7a: thresholds that exit vs reports; traces as contract-owned output | reviewer-21 | open |
| C51 | CO + CU × M7b | reviewer-22 | open |
| C52 | VI × M7b: thresholds that exit vs reports; production failure handling mirrored (L-026); completeness accounting (L-025) | reviewer-22 | open |
| C53 | CO + CU × M7c | reviewer-23 | open |
| C54 | VI × M7c: pinned payloads → canonical PCM → content-addressed index (L-017); goldens regenerate byte-identically from the recorded command | reviewer-23 | open |
| C55 | CS × M8 `README.md` vs behavior | reviewer-24 | open |
| C56 | CS × M8 `models/README.md` vs the loaders + `check_models` | reviewer-24 | open |
| C57 | CS × M8 `--help` + CLI strings vs behavior; L-021 register | reviewer-24 | open |
| C58 | CS × M9 `Artifacts` vs the tree + each run command | reviewer-24 | open |
| C59 | SF × `Intent` vs behavior | reviewer-25 | open |
| C60 | SF × `Decisions` entries `D-009` … `Long utterances publish at settled segments`, one subcheck per entry | reviewer-25 | open |
| C61 | CS × `asr-pipeline.md` claims (AP-001..017, AP-057..074) | reviewer-26 | open |
| C62 | CS × `asr-pipeline.md` claims (AP-018..056) | reviewer-27 | open |
| C63 | CS × `asr-pipeline.md` (AP-075..105) + `live-smoke.md` claims | reviewer-28 | open |
| C64 | CS × `translation-leg.md` claims | reviewer-29 | open |
| C65 | CS × `evidence-artifacts.md` + `openvino-accel.md` claims | reviewer-30 | open |
| C66 | CS × `toolchain.md`, `gate.md`, `packaging-deps.md`, `human-docs.md` claims | reviewer-31 | open |
| C67 | CS × `orientation.md`, `assurance-posture.md` + M9 `.agent/deferred.md` claims | reviewer-32 | open |
| C68 | CF × the clause inventory below, every row | reviewer-26 … reviewer-32 | open |
| C69 | SF × `Decisions` entries `A runaway caption` … `Out of scope`, one subcheck per entry; the out-of-scope list unviolated | reviewer-33 | open |

### Clause inventory (CF)

Anchor = the clause's first line at `56c9eae`, truncated.

| id | clause | owner | verdict |
| --- | --- | --- | --- |
| CM-001 | `CLAUDE.md:1` [] @.agent/spec.md | reviewer-32 | open |
| CM-002 | `CLAUDE.md:7` [Collaboration] - Uncertain / needs a decision / benefits from my input → ask, as exhaustively as useful:  | reviewer-32 | open |
| CM-003 | `CLAUDE.md:8` [Collaboration] - When discussion may improve the work, open one proactively: surface settled context, pro | reviewer-32 | open |
| CM-004 | `CLAUDE.md:9` [Collaboration] - Stay objective; push back on my ideas when warranted — these are collaborations. Root ca | reviewer-32 | open |
| CM-005 | `CLAUDE.md:10` [Collaboration] - A failed attempt is a valid result: report it with what it taught; a restart from scratc | reviewer-32 | open |
| CM-006 | `CLAUDE.md:14` [Execution] - Install/configure project-local; work within the launch dir + children; every file there | reviewer-32 | open |
| CM-007 | `CLAUDE.md:15` [Execution] - Long horizon → orient, then execute: one session carries a phase end to end across compa | reviewer-32 | open |
| CM-008 | `CLAUDE.md:16` [Execution] - Finish line = a pasted body's `Met when`, else one derived from my request + stated in y | reviewer-32 | open |
| CM-009 | `CLAUDE.md:17` [Execution] - Research, whenever a tooling, method or design choice is open = your knowledge + web sea | reviewer-32 | open |
| CM-010 | `CLAUDE.md:18` [Execution] - Git: creds in the global gitconfig; standing permission for all local-repo commands + `c | reviewer-32 | open |
| CM-011 | `CLAUDE.md:19` [Execution] - Gate command = the full check chain `.claude/rules/` records; it prints each step's wall | reviewer-32 | open |
| CM-012 | `CLAUDE.md:20` [Execution] - CI gate: a public GitHub repo whose `main` CI is green runs its full gate as a parallel  | reviewer-32 | open |
| CM-013 | `CLAUDE.md:21` [Execution] - Evidence: capture rc on the spot (`cmd; rc=$?`); an empty-output finding reports rc + a  | reviewer-32 | open |
| CM-014 | `CLAUDE.md:22` [Execution] - Command output rides the context → the quietest useful form: quiet reporters, `--stat`/` | reviewer-32 | open |
| CM-015 | `CLAUDE.md:23` [Execution] - Task close → remove the temporary + stale artifacts + dangling symlinks it left in touch | reviewer-32 | open |
| CM-016 | `CLAUDE.md:24` [Execution] - UI design (artifact, variant or unit): the subject's domain, materials + vernacular supp | reviewer-32 | open |
| CM-017 | `CLAUDE.md:25` [Execution] - Charts + figures: the data's job picks the form, ahead of color — magnitude → bar (grid  | reviewer-32 | open |
| CM-018 | `CLAUDE.md:29` [Authoring] - AI agents = the sole developers ⇒ agent-optimized = the default for EVERY text artifact  | reviewer-32 | open |
| CM-019 | `CLAUDE.md:30` [Authoring] - State rules, facts + warnings plainly; omit provenance — dates, verification/discovery e | reviewer-32 | open |
| CM-020 | `CLAUDE.md:31` [Authoring] - Future-facing text, esp. prompts → state the desired action positively (`always`/`must`) | reviewer-32 | open |
| CM-021 | `CLAUDE.md:32` [Authoring] - Prompts + instruction files state task, finish line + constraints; thinking depth = the  | reviewer-32 | open |
| CM-022 | `CLAUDE.md:33` [Authoring] - Guidance + checks on text = goals + reasons, applied with judgment; length + size follow | reviewer-32 | open |
| CM-023 | `CLAUDE.md:34` [Authoring] - Instruction files = yours to maintain → update any the moment it's improvable; a flaw in | reviewer-32 | open |
| CM-024 | `CLAUDE.md:35` [Authoring] - `.claude/rules/` reaches MAIN + every teammate of an interactive session with no brief l | reviewer-32 | open |
| CM-025 | `CLAUDE.md:36` [Authoring] - `@path` imports (instruction files) + attaches (my messages) a file CC leaves unloaded ( | reviewer-32 | open |
| CM-026 | `CLAUDE.md:37` [Authoring] - Human-facing = strictly user-facing surfaces a person reads at consumption time: shipped | reviewer-32 | open |
| CM-027 | `CLAUDE.md:41` [Engineering] - Elegant, tightly-scoped modular components; deduplicate; KISS + UNIX where apt; refactor | reviewer-32 | open |
| CM-028 | `CLAUDE.md:42` [Engineering] - Code = agent-read artifact → concise within three bounds: performant, bug-free, maximall | reviewer-32 | open |
| CM-029 | `CLAUDE.md:43` [Engineering] - Comments cost tokens → spend them on the `why` fresh agents would otherwise re-derive ev | reviewer-32 | open |
| CM-030 | `CLAUDE.md:44` [Engineering] - Target sufficient scope, evidence-backed claims + real success criteria. | reviewer-32 | open |
| CM-031 | `CLAUDE.md:45` [Engineering] - Established methods (TDD red-green-refactor, differential oracles, adversarial review) + | reviewer-32 | open |
| CM-032 | `CLAUDE.md:46` [Engineering] - Open tooling decisions (language/library/package…) → research (`Execution`) + select for | reviewer-32 | open |
| CM-033 | `CLAUDE.md:47` [Engineering] - Deterministic checks own every rule a tool can decide: linters, type checkers, static an | reviewer-32 | open |
| CM-034 | `CLAUDE.md:48` [Engineering] - Tests/verification: derive scope from requested outcome + regression risk + repo posture | reviewer-32 | open |
| CM-035 | `CLAUDE.md:49` [Engineering] - Runnable change → run the real app by its recorded command + drive it to the changed beh | reviewer-32 | open |
| CM-036 | `CLAUDE.md:50` [Engineering] - Verification integrity binds every check on the implementation; a prototype runs under P | reviewer-32 | open |
| CM-037 | `CLAUDE.md:51` [Engineering] - Assurance tier, declared per unit with its contract: `kernel` (judgment-bearing code/spe | reviewer-32 | open |
| CM-038 | `CLAUDE.md:52` [Engineering] - Units = the shortest path to the consumable artifact, and that alone. Off-path improveme | reviewer-32 | open |
| CM-039 | `CLAUDE.md:53` [Engineering] - A gate backing a durable claim must rerun from committed state; scratch-local validator  | reviewer-32 | open |
| CM-040 | `CLAUDE.md:54` [Engineering] - Repairs to a generated artifact land as one idempotent script replayable from a clean ba | reviewer-32 | open |
| CM-041 | `CLAUDE.md:55` [Engineering] - Adversarial review (code or session) → scrutinize correctness + logic, claim soundness,  | reviewer-32 | open |
| CM-042 | `CLAUDE.md:56` [Engineering] - Code diff lenses = the author's self-review before each commit + the default rows of any | reviewer-32 | open |
| CM-043 | `CLAUDE.md:57` [Engineering] - Split assurance by what it costs the decider: mechanical checks (suites, differential or | reviewer-32 | open |
| CM-044 | `CLAUDE.md:58` [Engineering] - Review terminates on a check set fixed before the diff is read: adjudicate every row, sh | reviewer-32 | open |
| CM-045 | `CLAUDE.md:59` [Engineering] - Delegated reports are two-tier: a verdict table the decider always reads, over detail se | reviewer-32 | open |
| CM-046 | `CLAUDE.md:60` [Engineering] - Remotely-exploitable code → highest security standard: periodically audit, update softwa | reviewer-32 | open |
| CM-047 | `CLAUDE.md:64` [Teammates] - MAIN does the work: a planned unit's code, docs + config, debugging included, are MAIN's | reviewer-32 | open |
| CM-048 | `CLAUDE.md:65` [Teammates] - Offload economics: fresh-context + blindness fits pay at any size; context fits pay once | reviewer-32 | open |
| CM-049 | `CLAUDE.md:66` [Teammates] - Briefs + prompts (teammates, `SendMessage`, `claude -p`) = broader intent + one narrow t | reviewer-32 | open |
| CM-050 | `CLAUDE.md:67` [Teammates] - Names = role + gapless ordinal (`reviewer-2`), so `SendMessage` can revive one; nested s | reviewer-32 | open |
| CM-051 | `CLAUDE.md:68` [Teammates] - Harvest: marker + turn end → read the deliverable + stop the teammate; a missing deliver | reviewer-32 | open |
| CM-052 | `CLAUDE.md:69` [Teammates] - A code-writing teammate works in `git worktree add -b wt/<name> .scratch/worktrees/<name | reviewer-32 | open |
| CM-053 | `CLAUDE.md:70` [Teammates] - Shared mutable resources (build stores, DBs, ports, work tree, desktop, browser account  | reviewer-32 | open |
| CM-054 | `CLAUDE.md:71` [Teammates] - Advisor, where present = MAIN's call per case at the `Session flow` bar, my request bind | reviewer-32 | open |
| CM-055 | `CLAUDE.md:75` [Session flow] - Phases, `Phase:` in `.agent/spec.md` = the phase + its scope (the whole product, or one  | reviewer-32 | open |
| CM-056 | `CLAUDE.md:76` [Session flow] - `.agent/spec.md` = the sole attached state, imported by the `@.agent/spec.md` line at th | reviewer-32 | open |
| CM-057 | `CLAUDE.md:77` [Session flow] - PROTOTYPE = the fastest path to an artifact I can inspect, chosen for inspectability ove | reviewer-32 | open |
| CM-058 | `CLAUDE.md:78` [Session flow] - IMPLEMENT = the best version of the intended software, built greenfield from the locked  | reviewer-32 | open |
| CM-059 | `CLAUDE.md:79` [Session flow] - Routing: phase per kind of work — an open direction or nothing inspectable yet (new prod | reviewer-32 | open |
| CM-060 | `CLAUDE.md:80` [Session flow] - Routed state: `Phase` line 1 = `<PHASE> — <scope>`; below it my request verbatim, `>` on | reviewer-32 | open |
| CM-061 | `CLAUDE.md:81` [Session flow] - This file + `.claude/agents/` = copies of a shared template, overwritten whole at each r | reviewer-32 | open |
| CM-062 | `CLAUDE.md:82` [Session flow] - Rules upkeep rides the `Tasks` clearing: each phase close + each MAINTAIN request close  | reviewer-32 | open |
| CM-063 | `CLAUDE.md:83` [Session flow] - Checkpointing: a phase session compacts many times ⇒ state lives on disk — one commit pe | reviewer-32 | open |
| CM-064 | `CLAUDE.md:84` [Session flow] - Teammates: fits, reasons + mechanics = `Teammates`; the phase sets teammate cost; the di | reviewer-32 | open |
| CM-065 | `CLAUDE.md:85` [Session flow] - PROTOTYPE + ITERATE: teammates = cheap + shallow, artifacts disposable, pace = the payof | reviewer-32 | open |
| CM-066 | `CLAUDE.md:86` [Session flow] - Builds = teammates: every artifact, component, variant, fixture set + round change, inde | reviewer-32 | open |
| CM-067 | `CLAUDE.md:87` [Session flow] - Repo hygiene = breadth over depth ⇒ `janitor` from each session's first commit (PROTOTYP | reviewer-32 | open |
| CM-068 | `CLAUDE.md:88` [Session flow] - Direction = unsettled + drift invisible from inside the build ⇒ `navigator` before each  | reviewer-32 | open |
| CM-069 | `CLAUDE.md:89` [Session flow] - Variant fan-out where a choice is open (separate contexts keep premises distinct). Check | reviewer-32 | open |
| CM-070 | `CLAUDE.md:90` [Session flow] - IMPLEMENT + MAINTAIN: teammates = deep; each unit's depth assessed with its tier. Deep = | reviewer-32 | open |
| CM-071 | `CLAUDE.md:91` [Session flow] - IMPLEMENT runs once + locks its decisions ⇒ assurance outranks teammate cost, heaviest a | reviewer-32 | open |
| CM-072 | `CLAUDE.md:92` [Session flow] - Interpretation, before any decision: ≥3 independent `navigator`s read `Intent` (human-wr | reviewer-32 | open |
| CM-073 | `CLAUDE.md:93` [Session flow] - Each decision costly to reverse (stack, architecture, data model, persistence format, pr | reviewer-32 | open |
| CM-074 | `CLAUDE.md:94` [Session flow] - Unit plan → a `consultant` verdict + a pre-mortem: one `consultant` per failure lens (sc | reviewer-32 | open |
| CM-075 | `CLAUDE.md:95` [Session flow] - Before the first unit, once the public-interface rulings sit in `Decisions`: a `tester`  | reviewer-32 | open |
| CM-076 | `CLAUDE.md:96` [Session flow] - IMPLEMENT + MAINTAIN unit fits, any depth: `researcher` = SOTA survey on an open tooling | reviewer-32 | open |
| CM-077 | `CLAUDE.md:97` [Session flow] - Advisor (fits + rulings = `Teammates`): IMPLEMENT + MAINTAIN = low bar: every fit, a clo | reviewer-32 | open |
| CM-078 | `CLAUDE.md:98` [Session flow] - Read cost control = `.gitignore` kept current — content search (`grep`, `rg`) honors it. | reviewer-32 | open |
| AP-001 | `asr-pipeline.md:12` [Capture, VAD, ring] - **silero opens a segment 0.2-0.7 s LATE and exposes no pad field. This binds EVERY engin | reviewer-26 | open |
| AP-002 | `asr-pipeline.md:16` [Capture, VAD, ring] - **silero hears a LEVELED copy; everything else keeps raw audio (`LeveledVad`, user rulin | reviewer-26 | open |
| AP-003 | `asr-pipeline.md:43` [Capture, VAD, ring] - **`VoiceActivityDetector` owns two buffers with different drain rules; every feeder must | reviewer-26 | open |
| AP-004 | `asr-pipeline.md:52` [Capture, VAD, ring] - **Capture resamples through ONE stateful band-limited stream (`Resampler`, soxr HQ).** T | reviewer-26 | open |
| AP-005 | `asr-pipeline.md:80` [Capture, VAD, ring] - **`--save-audio` records at CAPTURE** (`AudioRecording`, opt-in): every resampled block  | reviewer-26 | open |
| AP-006 | `asr-pipeline.md:90` [Capture, VAD, ring] - **`--playback` captures the default OUTPUT through the mic's own PortAudio path (user re | reviewer-26 | open |
| AP-007 | `asr-pipeline.md:113` [Capture, VAD, ring] - **Digital silence in playback freezes `LeveledVad`'s history.** A monitor carries EXACT  | reviewer-26 | open |
| AP-008 | `asr-pipeline.md:133` [Capture, VAD, ring] - `RING_SECONDS`=60 bounds the tested envelope: a VAD segment outliving the ring loses its | reviewer-26 | open |
| AP-009 | `asr-pipeline.md:135` [Capture, VAD, ring] - silero's `max_speech_duration` (20 s) is a SOFT cap — it raises thresholds (0.5→0.9, | reviewer-26 | open |
| AP-010 | `asr-pipeline.md:143` [VAC hypothesis buffer (`streaming.py`)] - `StreamingProcessor` = LocalAgreement-2 over repeated decodes of ONE open utterance buff | reviewer-26 | open |
| AP-011 | `asr-pipeline.md:147` [VAC hypothesis buffer (`streaming.py`)] - Append-only is load-bearing: a shrinking `emitted` re-commits on-screen characters. Hold | reviewer-26 | open |
| AP-012 | `asr-pipeline.md:150` [VAC hypothesis buffer (`streaming.py`)] - **The published boundary is ALIGNED, never counted (`StreamingProcessor._anchor`).** `em | reviewer-26 | open |
| AP-013 | `asr-pipeline.md:206` [VAC hypothesis buffer (`streaming.py`)] - **Below half agreement the boundary moves only once two decodes agree (`_thin`, user rul | reviewer-26 | open |
| AP-014 | `asr-pipeline.md:239` [VAC hypothesis buffer (`streaming.py`)] - **A decode re-telling the published end past the located boundary skips that run | reviewer-26 | open |
| AP-015 | `asr-pipeline.md:268` [VAC hypothesis buffer (`streaming.py`)] - **A post-trim decode that re-tells the trimmed text loses that head (`_retold`).** Whisp | reviewer-26 | open |
| AP-016 | `asr-pipeline.md:295` [VAC hypothesis buffer (`streaming.py`)] - **A trimming decode spelling one phrase twice across its last segment boundary keeps the | reviewer-26 | open |
| AP-017 | `asr-pipeline.md:324` [VAC hypothesis buffer (`streaming.py`)] - The `on_update` seam (`worker` / `_vac_segments` / `replay.py`) is what makes commit tim | reviewer-26 | open |
| AP-018 | `asr-pipeline.md:329` [Whisper on the NPU (D-016)] - **The M10 candidate screen is CLOSED and stays closed.** Old multilingual streaming zipf | reviewer-27 | open |
| AP-019 | `asr-pipeline.md:336` [Whisper on the NPU (D-016)] - **Policy.** Waiting for silero to close a segment bounds the first character's latency b | reviewer-27 | open |
| AP-020 | `asr-pipeline.md:343` [Whisper on the NPU (D-016)] - **Device.** The NPU costs nothing in accuracy (VAC CER 0.2321/0.0686 vs GPU 0.2292/0.069 | reviewer-27 | open |
| AP-021 | `asr-pipeline.md:351` [Whisper on the NPU (D-016)] - **Never feed prose back to the recogniser.** Recent transcript as prev-text measured CER | reviewer-27 | open |
| AP-022 | `asr-pipeline.md:363` [Whisper on the NPU (D-016)] - **`repetition_penalty` is the ONLY repetition knob that reaches this build.** | reviewer-27 | open |
| AP-023 | `asr-pipeline.md:370` [Whisper on the NPU (D-016)] - **`WhisperPipeline` latches OMITTED language and nothing else; an EXPLICIT token always  | reviewer-27 | open |
| AP-024 | `asr-pipeline.md:388` [Whisper on the NPU (D-016)] - **Every decode on the VAC path names its language, so the latch is unreachable by constr | reviewer-27 | open |
| AP-025 | `asr-pipeline.md:392` [Whisper on the NPU (D-016)] - **The session fallback is `ASR_LANGUAGE`, never the literal `ja`.** One-way `--source-la | reviewer-27 | open |
| AP-026 | `asr-pipeline.md:395` [Whisper on the NPU (D-016)] - **The processor's decode callable SNAPSHOTS the token at construction**, one closure per | reviewer-27 | open |
| AP-027 | `asr-pipeline.md:399` [Whisper on the NPU (D-016)] - **`token` is assigned BEFORE each `StreamingProcessor` build, at both sites.** The snaps | reviewer-27 | open |
| AP-028 | `asr-pipeline.md:411` [Whisper on the NPU (D-016)] - **A wrong language token is CATASTROPHIC in both directions and graceful in neither.** 1 | reviewer-27 | open |
| AP-029 | `asr-pipeline.md:415` [Whisper on the NPU (D-016)] \| audio \| `<\\|ja\\|>` CER \| `<\\|en\\|>` CER \| ref chars \| decode p50 \| | reviewer-27 | open |
| AP-030 | `asr-pipeline.md:420` [Whisper on the NPU (D-016)] Two things follow. **English is the recogniser's better language** — 0.0263 against 0.0502 | reviewer-27 | open |
| AP-031 | `asr-pipeline.md:427` [Whisper on the NPU (D-016)] - `ASRDecodedResults` fields: `chunks, language, perf_metrics, scores, texts, words`. Ther | reviewer-27 | open |
| AP-032 | `asr-pipeline.md:429` [Whisper on the NPU (D-016)] - **The 0.35-0.42 s fixed decode term is structural in genai 2026.3.1 — three levers are c | reviewer-27 | open |
| AP-033 | `asr-pipeline.md:438` [Whisper on the NPU (D-016)] - **The two constructor properties are MEASURED and REFUSED; the fixed term stands.** Meth | reviewer-27 | open |
| AP-034 | `asr-pipeline.md:445` [Whisper on the NPU (D-016)] - `NPU_TURBO=true` — median p50 0.5453 → 0.5329 s, paired median **−2.3 ms** per update (m | reviewer-27 | open |
| AP-035 | `asr-pipeline.md:448` [Whisper on the NPU (D-016)] - `NPUW_LLM_GENERATE_HINT="BEST_PERF"` — cold compile succeeds (95.9 s) and decodes, then  | reviewer-27 | open |
| AP-036 | `asr-pipeline.md:452` [Whisper on the NPU (D-016)] - Baseline note: base measures 0.5453 s here against the committed trace's 0.645 s. That i | reviewer-27 | open |
| AP-037 | `asr-pipeline.md:454` [Whisper on the NPU (D-016)] - **Two processes cold-compiling the SAME cache key concurrently write a blob that SIGSEGV | reviewer-27 | open |
| AP-038 | `asr-pipeline.md:459` [Whisper on the NPU (D-016)] - A resident whisper large-v3-turbo int8 pipeline costs **~2.0 GB RSS** (2253 MB held, 225 | reviewer-27 | open |
| AP-039 | `asr-pipeline.md:461` [Whisper on the NPU (D-016)] - **Two resident pipelines cost +2015 MB and +0.050 s per update, and buy nothing the expl | reviewer-27 | open |
| AP-040 | `asr-pipeline.md:473` [Whisper on the NPU (D-016)] - **A standalone spoken LID settles that token, and its operating point is measured: ECAPA | reviewer-27 | open |
| AP-041 | `asr-pipeline.md:478` [Whisper on the NPU (D-016)] ```sh | reviewer-27 | open |
| AP-042 | `asr-pipeline.md:486` [Whisper on the NPU (D-016)] **The decision rule has three parts and no fourth.** The global 107-way argmax must itself | reviewer-27 | open |
| AP-043 | `asr-pipeline.md:550` [Whisper on the NPU (D-016)] - **Repetition-loop cause + free repro.** The recogniser is pinned to Japanese, so audio i | reviewer-27 | open |
| AP-044 | `asr-pipeline.md:568` [Whisper on the NPU (D-016)] - **One unreplicated ambient capture produced nothing at all — before `LeveledVad`, so it  | reviewer-27 | open |
| AP-045 | `asr-pipeline.md:578` [GPU mode (`--asr-device GPU`, user reque] **Shape (user rulings):** the existing `--asr-device`, choices `NPU\|GPU` (CPU dropped: 4.4 | reviewer-27 | open |
| AP-046 | `asr-pipeline.md:585` [GPU mode (`--asr-device GPU`, user reque] **Accuracy, one harness** = `tests/eval_asr_screen.py` (`evidence-artifacts.md`): the ship | reviewer-27 | open |
| AP-047 | `asr-pipeline.md:590` [GPU mode (`--asr-device GPU`, user reque] \| setup \| FLEURS-ja 150 (whole clip) \| retention (VAC) \| long-form §01+§03 (VAC) \| CV8 300 | reviewer-27 | open |
| AP-048 | `asr-pipeline.md:599` [GPU mode (`--asr-device GPU`, user reque] - **760M is device-neutral**: GPU ≡ NPU at reading level on every shared leg; 2.3× faster  | reviewer-27 | open |
| AP-049 | `asr-pipeline.md:600` [GPU mode (`--asr-device GPU`, user reque] - **whisper-ja-1.5B buys nothing measurable at 2.8× decode**: its long-form char win is | reviewer-27 | open |
| AP-050 | `asr-pipeline.md:606` [GPU mode (`--asr-device GPU`, user reque] - **Cohere wins short read clips and loses the live domain.** Whole-utterance decodes of t | reviewer-27 | open |
| AP-051 | `asr-pipeline.md:613` [GPU mode (`--asr-device GPU`, user reque] - **Refused levers, measured:** beam search on the GPU raises `iremote_tensor.hpp:32 Not | reviewer-27 | open |
| AP-052 | `asr-pipeline.md:618` [GPU mode (`--asr-device GPU`, user reque] - **English: large-v3's whole-clip gain does not survive the live path.** FLEURS-en 150 wh | reviewer-27 | open |
| AP-053 | `asr-pipeline.md:623` [GPU mode (`--asr-device GPU`, user reque] - **Runaway budget** = a decode forced to its cap (`min_new_tokens`, 11 s retention buffer | reviewer-27 | open |
| AP-054 | `asr-pipeline.md:628` [GPU mode (`--asr-device GPU`, user reque] - **Latency** = `tests/vac_decode_trace_gpu.json` (`build_vac_trace.py --device GPU --out  | reviewer-27 | open |
| AP-055 | `asr-pipeline.md:636` [GPU mode (`--asr-device GPU`, user reque] - Regression lock: the `whisper@GPU` replay golden (`long`, device-tagged `GPU`), beside t | reviewer-27 | open |
| AP-056 | `asr-pipeline.md:637` [GPU mode (`--asr-device GPU`, user reque] - **Not verified (L-004):** a live GPU session on the host (`live-smoke.md` item 10), powe | reviewer-27 | open |
| AP-057 | `asr-pipeline.md:642` [Publication screen — `caption_defect()`] - The screen runs at PUBLICATION, upstream of every consumer, so `observe_ja`, the transla | reviewer-26 | open |
| AP-058 | `asr-pipeline.md:648` [Publication screen — `caption_defect()`] - Thresholds, corpus-picked against 1073 live JA captions over 6 sessions (`transcripts/*. | reviewer-26 | open |
| AP-059 | `asr-pipeline.md:654` [Publication screen — `caption_defect()`] - **The unit bound is bounded by the REPEAT COUNT, not by the phrase length.** A drop take | reviewer-26 | open |
| AP-060 | `asr-pipeline.md:662` [Publication screen — `caption_defect()`] - **`CAPTION_REPEAT_MAX_CHARS`=40 is CLOSED — measured, then REFUSED.** Re-derived over 14 | reviewer-26 | open |
| AP-061 | `asr-pipeline.md:677` [Publication screen — `caption_defect()`] - **The latin rule keys on the DECODE language** (`caption_defect(text, token)`; `ASR_LANG | reviewer-26 | open |
| AP-062 | `asr-pipeline.md:681` [Publication screen — `caption_defect()`] - Latin ratio: the 23 latin-dominant live captions split cleanly — 17 true English at ≤0.1 | reviewer-26 | open |
| AP-063 | `asr-pipeline.md:686` [Publication screen — `caption_defect()`] - **Long utterances publish at settled segments (user ruling; supersedes UNCAPPED).** Utte | reviewer-26 | open |
| AP-064 | `asr-pipeline.md:705` [Publication screen — `caption_defect()`] - **A piece is screened per released segment, then as one line.** Screened whole, one trim | reviewer-26 | open |
| AP-065 | `asr-pipeline.md:739` [Sherpa fallback path (D-010)] - Both engines TTFT ≤0.10 s post-audio, $0/hr. **k2v2 is the default WITHIN the pair** (RT | reviewer-26 | open |
| AP-066 | `asr-pipeline.md:746` [Sherpa fallback path (D-010)] - **Raw long segments collapse offline decode — this path only.** Segment LENGTH (>~15-20  | reviewer-26 | open |
| AP-067 | `asr-pipeline.md:753` [Sherpa fallback path (D-010)] - sherpa online gotchas: `from_transducer(enable_endpoint_detection=…)` surfaces as config | reviewer-26 | open |
| AP-068 | `asr-pipeline.md:759` [Shutdown, streams, meter] - **Closing the terminal is a shipped shutdown path and cost three separate fixes.** SIGHU | reviewer-26 | open |
| AP-069 | `asr-pipeline.md:767` [Shutdown, streams, meter] - **One event is ONE line, and `emit_line` is where that is enforced.** The grammar every  | reviewer-26 | open |
| AP-070 | `asr-pipeline.md:775` [Shutdown, streams, meter] - **No C0, DEL or C1 control from recogniser or translator text reaches a sink** (`_CONTRO | reviewer-26 | open |
| AP-071 | `asr-pipeline.md:785` [Shutdown, streams, meter] - **L-006 — TTY-gate the cursor-clear (`\r\x1b[2K`) protocol per stream, evaluated once.** | reviewer-26 | open |
| AP-072 | `asr-pipeline.md:794` [Shutdown, streams, meter] - **The peak log gates on the stream PAIR — `not (_STDOUT_TTY and _STDERR_TTY)`, not on st | reviewer-26 | open |
| AP-073 | `asr-pipeline.md:801` [Shutdown, streams, meter] - **`--save-log` = `2>` from inside the process (`redirect_stderr`).** At the session stam | reviewer-26 | open |
| AP-074 | `asr-pipeline.md:811` [Shutdown, streams, meter] - **L-010 — keep device-backend imports at device entry points.** PortAudio probes host au | reviewer-26 | open |
| AP-075 | `asr-pipeline.md:824` [Latency budget — per stage, `tests/eval_] Every figure re-derives from `vac_decode_trace.json` (whisper-ja-760M, the lower-cost of t | reviewer-28 | open |
| AP-076 | `asr-pipeline.md:828` [Latency budget — per stage, `tests/eval_] \| stage \| p50 \| p90 \| max \| what it is \| | reviewer-28 | open |
| AP-077 | `asr-pipeline.md:838` [Latency budget — per stage, `tests/eval_] - **Decode cost is `0.501 s fixed + 5.66 ms/char`** (`stress_long`: 0.552 + 3.78; turbo re | reviewer-28 | open |
| AP-078 | `asr-pipeline.md:845` [Latency budget — per stage, `tests/eval_] - **The commit lag is a DISPLAY POLICY cost, not a compute cost.** `lag = audio-time holdb | reviewer-28 | open |
| AP-079 | `asr-pipeline.md:850` [Latency budget — per stage, `tests/eval_] - **Both arms measure FIRST APPEARANCE ⇒ the gap is NOT a settled-text speedup.** Settled  | reviewer-28 | open |
| AP-080 | `asr-pipeline.md:859` [Latency budget — per stage, `tests/eval_] - **`VAC_CHUNK_S` is floored by decode cost, not by taste.** Work rate = `decode_s / VAC_C | reviewer-28 | open |
| AP-081 | `asr-pipeline.md:865` [Latency budget — per stage, `tests/eval_] - Translation is **2.170 s p50, not the ~1 s the tournament's 1.38 s implied** — that figu | reviewer-28 | open |
| AP-082 | `asr-pipeline.md:867` [Latency budget — per stage, `tests/eval_] - **The rotation tax is CUT.** `translator_brief()` renders the glossary in content order | reviewer-28 | open |
| AP-083 | `asr-pipeline.md:876` [Latency budget — per stage, `tests/eval_] - **`rotations` counts GLOSSARY rotations only** — `eval_en_pairing.py` sets `rotated` fro | reviewer-28 | open |
| AP-084 | `asr-pipeline.md:882` [Latency budget — per stage, `tests/eval_] - **The row's turn-p90-under-3.5 s half is REFUSED.** Excluding every rotation, steady p90 | reviewer-28 | open |
| AP-085 | `asr-pipeline.md:891` [Latency budget — per stage, `tests/eval_] - **The row's rendering-consistency half is now MEASURED, on the raw stream.** Its accepta | reviewer-28 | open |
| AP-086 | `asr-pipeline.md:901` [Latency budget — per stage, `tests/eval_] - Counter-cost, DESCRIPTIVE not causal: the new run's surviving rotations are dearer (p50  | reviewer-28 | open |
| AP-087 | `asr-pipeline.md:911` [Two-way withholding — `tests/eval_two_wa] Two-way commits nothing until the LID accepts a label, which cannot happen before `LID_MIN | reviewer-28 | open |
| AP-088 | `asr-pipeline.md:920` [Two-way withholding — `tests/eval_two_wa] \| arm \| `stress_long` p50/p90/max \| `retention_probe` p50/p90/max \| re-dated \| | reviewer-28 | open |
| AP-089 | `asr-pipeline.md:927` [Two-way withholding — `tests/eval_two_wa] `re-dated` counts characters committed before acceptance and is an UPPER bound on how many | reviewer-28 | open |
| AP-090 | `asr-pipeline.md:937` [Real-time cost — the instrument is CARRY] VAC awaits each decode — update and final alike — inside the coroutine draining `audio_q`, | reviewer-28 | open |
| AP-091 | `asr-pipeline.md:943` [Real-time cost — the instrument is CARRY] - **The catch-up rule owns sustained overload.** Each update consumes exactly `VAC_CHUNK_S | reviewer-28 | open |
| AP-092 | `asr-pipeline.md:951` [Real-time cost — the instrument is CARRY] - **The 8 s headroom owns single blocking spans.** No drain runs inside one, so the queue  | reviewer-28 | open |
| AP-093 | `asr-pipeline.md:956` [Real-time cost — the instrument is CARRY] - **Measured live on the 09-24 capture.** A 29-minute session on battery under | reviewer-28 | open |
| AP-094 | `asr-pipeline.md:966` [Real-time cost — the instrument is CARRY] - **Limit: `HARD_TRIM_S` is enforced AFTER a decode**, so once trims have already failed ( | reviewer-28 | open |
| AP-095 | `asr-pipeline.md:970` [Real-time cost — the instrument is CARRY] - **The `live` arm of `tests/eval_backpressure.py` is the lock**: both traced clips at ×1. | reviewer-28 | open |
| AP-096 | `asr-pipeline.md:980` [Real-time cost — the instrument is CARRY] - **What catch-up costs in accuracy — real NPU recogniser (turbo, before the 760M swap), ` | reviewer-28 | open |
| AP-097 | `asr-pipeline.md:984` [Real-time cost — the instrument is CARRY] \| arm \| CER \| S / D / I \| queue peak \| | reviewer-28 | open |
| AP-098 | `asr-pipeline.md:991` [Real-time cost — the instrument is CARRY] Sustained overload costs nothing: the stretched cadence reads 0.0600 against 0.0609. Foldi | reviewer-28 | open |
| AP-099 | `asr-pipeline.md:996` [Real-time cost — the instrument is CARRY] - **Aggregate RTF is the wrong instrument** — it shows mean compute below real time and sa | reviewer-28 | open |
| AP-100 | `asr-pipeline.md:998` [Real-time cost — the instrument is CARRY] - **Carry is the cross-utterance instrument:** a caption costing more wall time than its o | reviewer-28 | open |
| AP-101 | `asr-pipeline.md:1007` [Real-time cost — the instrument is CARRY] - **A caption's `decode_s` is a SUM of that utterance's update decodes, never one blocking | reviewer-28 | open |
| AP-102 | `asr-pipeline.md:1011` [Real-time cost — the instrument is CARRY] - Rerun cost varies ~20 % run to run, and the burst is machine state rather than a path pr | reviewer-28 | open |
| AP-103 | `asr-pipeline.md:1014` [Real-time cost — the instrument is CARRY] - `drop=` counts backend-sized callback BLOCKS, never samples or speech ⇒ quote blocks, co | reviewer-28 | open |
| AP-104 | `asr-pipeline.md:1025` [Known caveats] - Models are a runtime prerequisite — `check_models()` preflights and points at `models/RE | reviewer-28 | open |
| AP-105 | `asr-pipeline.md:1026` [Known caveats] - **The translator is NOT the quality bottleneck — the recogniser is.** Over 30 consecutiv | reviewer-28 | open |
| AS-001 | `assurance-posture.md:3` [Assurance posture] This file owns what assurance the repo actually runs. Each ruling on a template clause bel | reviewer-32 | open |
| AS-002 | `assurance-posture.md:7` [Assurance posture] **User ruling: personal tool, not an industrial product** — the apparatus reached ~11,100  | reviewer-32 | open |
| AS-003 | `assurance-posture.md:13` [Assurance posture] **Verification integrity binds every check here.** Template ruling: "it lives where `Artif | reviewer-32 | open |
| AS-004 | `assurance-posture.md:20` [Assurance posture] - Template ruling: "A test counts once seen red on the unfixed revision" → **ADAPTED** — * | reviewer-32 | open |
| AS-005 | `assurance-posture.md:34` [Assurance posture] - Template ruling: "a threshold, case or gate changes only in a unit I approve" → **ADAPTE | reviewer-32 | open |
| AS-006 | `assurance-posture.md:45` [Assurance posture] - Template ruling: "A skipped, xfailed, deleted or tier-demoted case" → **ADAPTED** — **a  | reviewer-32 | open |
| AS-007 | `assurance-posture.md:54` [Assurance posture] **L-004 — mic and real-terminal paths are agent-unverifiable.** No mic, no interactive TTY | reviewer-32 | open |
| AS-008 | `assurance-posture.md:62` [Assurance posture] **L-031 — size work by the decisions it forces, not by its line count.** Measured over six | reviewer-32 | open |
| AS-009 | `assurance-posture.md:70` [Assurance posture] **L-032 — keep assurance proportional to the artifact; measure the ratio periodically.** A | reviewer-32 | open |
| AS-010 | `assurance-posture.md:80` [Assurance posture] L-032 binds the items retired below alone: the review ledger `.agent/review.md` and the re | reviewer-32 | open |
| AS-011 | `assurance-posture.md:83` [Assurance posture] Retired — never reintroduce: | reviewer-32 | open |
| AS-012 | `assurance-posture.md:85` [Assurance posture] - Template ruling: "contracts + tiers" → **ADAPTED** — `.agent/contracts/` retired ⇒ a uni | reviewer-32 | open |
| AS-013 | `assurance-posture.md:89` [Assurance posture] - Contract fingerprints · claim registries · mutation matrices. A mutation-tool run the te | reviewer-32 | open |
| AS-014 | `assurance-posture.md:91` [Assurance posture] - A separate project-memory file under `.agent/` ⇒ its law lives in these rules files, whi | reviewer-32 | open |
| AS-015 | `assurance-posture.md:99` [Assurance posture] `CLAUDE.md`'s `Teammates` + the `.claude/agents/` roles own the fits; this repo adds no te | reviewer-32 | open |
| AS-016 | `assurance-posture.md:109` [Assurance posture] Template ruling: "one row per check × target" → **ADAPTED** — each report (`CLAUDE.md`'s | reviewer-32 | open |
| AS-017 | `assurance-posture.md:116` [Assurance posture] `<window>` in a recorded gauge reading = the loop's compaction trigger, its occupancy = th | reviewer-32 | open |
| EA-001 | `evidence-artifacts.md:9` [Evidence artifacts — corpora, traces, ev] Fast locks run in the gate. **The fourteen `eval_*.py` scripts are ON-DEMAND and never gat | reviewer-30 | open |
| EA-002 | `evidence-artifacts.md:18` [Committed artifacts and what each certif] - `replay_goldens.json` + `gen_replay_goldens.py` — characterization snapshots, engine-fir | reviewer-30 | open |
| EA-003 | `evidence-artifacts.md:25` [Committed artifacts and what each certif] - `short_corpus.json` + `fetch_real_clips.py` + `real_clips.json` + `test_corpus.py` — pin | reviewer-30 | open |
| EA-004 | `evidence-artifacts.md:28` [Committed artifacts and what each certif] - `en_clips.json` + `fetch_real_clips.py`'s `en_us` arm + `test_en_corpus.py` — the pinned | reviewer-30 | open |
| EA-005 | `evidence-artifacts.md:37` [Committed artifacts and what each certif] - `lid_census.json` + `build_lid_census.py` + `test_lid_census.py` — the LID decision tabl | reviewer-30 | open |
| EA-006 | `evidence-artifacts.md:64` [Committed artifacts and what each certif] - `stressor_clips.json` + `build_stressor.py` — the 44.7 s genuinely-continuous stressor:  | reviewer-30 | open |
| EA-007 | `evidence-artifacts.md:68` [Committed artifacts and what each certif] - `retention_probe.json` + `build_retention_probe.py` + `test_retention_probe.py` — the 18 | reviewer-30 | open |
| EA-008 | `evidence-artifacts.md:70` [Committed artifacts and what each certif] - `long_form.json` + `eval_long_form.py` — the pinned LibriVox/Kokoro/Aozora narration: al | reviewer-30 | open |
| EA-009 | `evidence-artifacts.md:80` [Committed artifacts and what each certif] - `eval_anchor_scenarios.py` — scripted decode runs through `StreamingProcessor` over the | reviewer-30 | open |
| EA-010 | `evidence-artifacts.md:92` [Committed artifacts and what each certif] - `vac_decode_trace.json` (NPU) + `vac_decode_trace_gpu.json` (GPU mode, `--device GPU --o | reviewer-30 | open |
| EA-011 | `evidence-artifacts.md:102` [Committed artifacts and what each certif] - `eval_two_way_settled.py` — the same trace under two-way's commit rule, which withholds  | reviewer-30 | open |
| EA-012 | `evidence-artifacts.md:109` [Committed artifacts and what each certif] - `caption_trace.json` + `build_caption_trace.py` — the shipped path's caption stream over | reviewer-30 | open |
| EA-013 | `evidence-artifacts.md:117` [Committed artifacts and what each certif] - `en_pairing_trace.json` + `eval_en_pairing.py` + `test_en_pairing.py` — what a REAL tran | reviewer-30 | open |
| EA-014 | `evidence-artifacts.md:144` [Committed artifacts and what each certif] - `eval_term_census.py` + `test_term_census.py` — what the recogniser gives `SessionContex | reviewer-30 | open |
| EA-015 | `evidence-artifacts.md:150` [Committed artifacts and what each certif] - `cer_baseline.json` + `eval_cer.py` — model-gated two-engine CER / stressor / RTF evalua | reviewer-30 | open |
| EA-016 | `evidence-artifacts.md:155` [Committed artifacts and what each certif] - `eval_backpressure.py` + `test_backpressure.py` — the production two-stage worker paced  | reviewer-30 | open |
| EA-017 | `evidence-artifacts.md:166` [Committed artifacts and what each certif] - `eval_asr_screen.py` — the recogniser screen behind GPU mode (`asr-pipeline.md` `## GPU  | reviewer-30 | open |
| EA-018 | `evidence-artifacts.md:196` [Committed artifacts and what each certif] ```sh | reviewer-30 | open |
| EA-019 | `evidence-artifacts.md:208` [Committed artifacts and what each certif] - `session_report.py` + `test_session_report.py` — what a LIVE session did, re-derived fro | reviewer-30 | open |
| EA-020 | `evidence-artifacts.md:246` [Committed artifacts and what each certif] - `eval_playback_routing.py` — `--playback`'s routing on a private PipeWire + WirePlumber  | reviewer-30 | open |
| EA-021 | `evidence-artifacts.md:258` [Committed artifacts and what each certif] - `eval_playback_silence.py` — the same 14 clips (7 CV8 rows: the quiet five the old zero- | reviewer-30 | open |
| EA-022 | `evidence-artifacts.md:265` [Committed artifacts and what each certif] - `test_gate.py` locks `gate.py`'s step inventory with one seeded defect per blocking step | reviewer-30 | open |
| EA-023 | `evidence-artifacts.md:269` [Committed artifacts and what each certif] - `lag_sessions.json` + `build_lag_trace.py` + `eval_lag.py` + `test_lag_trace.py` — how f | reviewer-30 | open |
| EA-024 | `evidence-artifacts.md:304` [Rules that keep the evidence honest] - **A resource gate declares itself: `absent: <what>`.** Every skip in this suite gates on | reviewer-30 | open |
| EA-025 | `evidence-artifacts.md:310` [Rules that keep the evidence honest] - **The goldens and traces ARE contract-owned expected output, not a table smuggled past a | reviewer-30 | open |
| EA-026 | `evidence-artifacts.md:317` [Rules that keep the evidence honest] - **D-014 — deterministic WAV replay is the regression harness.** `replay.py` drives the r | reviewer-30 | open |
| EA-027 | `evidence-artifacts.md:325` [Rules that keep the evidence honest] - **L-024 — backpressure capacity is TIME; prove it with paced production replay.** Callba | reviewer-30 | open |
| EA-028 | `evidence-artifacts.md:332` [Rules that keep the evidence honest] - **L-025 — evaluator completeness is input/event accounting, not nonempty text.** On the  | reviewer-30 | open |
| EA-029 | `evidence-artifacts.md:340` [Rules that keep the evidence honest] - **L-017 — corpus evidence = pinned payloads → canonical PCM → content-addressed index.** | reviewer-30 | open |
| EA-030 | `evidence-artifacts.md:347` [Rules that keep the evidence honest] - **L-026 — a model bench is evidence only if the harness mirrors production failure handl | reviewer-30 | open |
| EA-031 | `evidence-artifacts.md:359` [Rules that keep the evidence honest] - **L-027 — a deterministic content probe measures your regex until you read the misses.** | reviewer-30 | open |
| EA-032 | `evidence-artifacts.md:368` [Rules that keep the evidence honest] - **L-028 — a parallel-corpus reference is a judge-validity probe, never a translation cei | reviewer-30 | open |
| EA-033 | `evidence-artifacts.md:377` [Rules that keep the evidence honest] - **L-022 — a claimed fix ships with a NON-VACUOUS lock, proven by neutralization.** Neutr | reviewer-30 | open |
| EA-034 | `evidence-artifacts.md:395` [Rules that keep the evidence honest] - **Per-character caption lag, never per-commit.** A commit carries several characters ove | reviewer-30 | open |
| EA-035 | `evidence-artifacts.md:406` [Rules that keep the evidence honest] - **L-016 — gitignored bulk corpora stay reachable by a script's runtime `open()`** even w | reviewer-30 | open |
| EA-036 | `evidence-artifacts.md:412` [On-demand commands] code block | reviewer-30 | open |
| EA-037 | `evidence-artifacts.md:436` [On-demand commands] Whisper, NPU and GPU runs need the prelude first (`openvino-accel.md`). | reviewer-30 | open |
| GM-001 | `gate.md:7` [Gate checks — mechanics, coverage limits] The law each check enforces lives in `toolchain.md`; this file holds how each check fires  | reviewer-31 | open |
| GM-002 | `gate.md:12` [Steps] **Every step ships the input that fires it, and the gate runs those inputs on itself.** | reviewer-31 | open |
| GM-003 | `gate.md:35` [Steps] The pytest step runs on `PYTEST_WORKERS`=6 pytest-xdist workers (serial ≈39 s → ≈24 s on t | reviewer-31 | open |
| GM-004 | `gate.md:51` [Steps] Two locks own the `ruff-format` Markdown scope: `test_the_format_step_reaches_a_markdown_c | reviewer-31 | open |
| GM-005 | `gate.md:57` [Steps] Each step prints its wall-clock and the run its total; the pytest step runs `--durations=5 | reviewer-31 | open |
| GM-006 | `gate.md:65` [Steps] `spec-lint` runs `.agent/spec-lint` through `sh`, the template's shape check, a copy each  | reviewer-31 | open |
| GM-007 | `gate.md:75` [Law locks (`tests/test_law_consistency.p] Pairing is checked ORDERED because independent rank/title membership passes a swap of two  | reviewer-31 | open |
| GM-008 | `gate.md:86` [Law locks (`tests/test_law_consistency.p] The third, the queue-link lock, is mutation-proven red by planting a pointer at *Maintenan | reviewer-31 | open |
| GM-009 | `gate.md:89` [Law locks (`tests/test_law_consistency.p] None of the five locks needs a seeded fixture — the tree itself was the firing input, red  | reviewer-31 | open |
| GM-010 | `gate.md:97` [Law locks (`tests/test_law_consistency.p] The fourth holds the ruling tags. Mutation-proven red, each mutant planted from a `cp` sna | reviewer-31 | open |
| HD-001 | `human-docs.md:10` [Human-facing documentation] **L-021 — `CLAUDE.md`'s human register (`Authoring`, Human-facing) applies to human-facing | reviewer-31 | open |
| HD-002 | `human-docs.md:16` [Human-facing documentation] **The gate decides the dash rule and two literal sets; the rest of the register is judgmen | reviewer-31 | open |
| HD-003 | `human-docs.md:31` [Human-facing documentation] `README.md` is the user doc: it covers install, CLI, behaviour and the coverage boundary. | reviewer-31 | open |
| HD-004 | `human-docs.md:36` [Human-facing documentation] Do not re-add superseded spike reports (D-005). The Gemini-era and metered-backend spike r | reviewer-31 | open |
| LS-001 | `live-smoke.md:9` [Live smoke — agent-covered vs user-only ] This is the fixed procedure a "**Did not verify (L-004)**" disclaimer points at. Each item | reviewer-28 | open |
| LS-002 | `live-smoke.md:12` [Live smoke — agent-covered vs user-only ] **Agent-covered, so do not send these to the user.** The two-engine replay/CER gates own s | reviewer-28 | open |
| LS-003 | `live-smoke.md:25` [Live smoke — agent-covered vs user-only ] **User-only boundary:** mic capture, device select, latency feel, real Ctrl+C flush, `-o`  | reviewer-28 | open |
| LS-004 | `live-smoke.md:28` [Live smoke — agent-covered vs user-only ] **Setup:** models downloaded; `codex` installed on the PATH of the machine running live-st | reviewer-28 | open |
| LS-005 | `live-smoke.md:34` [Live-mic pass (~5 min)] 1. **Devices** — `uv run live-stt --list-devices` prints the `sd.query_devices()` table an | reviewer-28 | open |
| LS-006 | `live-smoke.md:130` [Live-mic pass (~5 min)] **Every `SRC n:` criterion above is the ONE-WAY grammar**, and `--two-way` is default OFF  | reviewer-28 | open |
| LS-007 | `live-smoke.md:139` [Soak (1-3 h) — watch at start and end] - **Backlog / drops** — `q=` / `seg=` / `drop=` / `tdrop=` / `tstale=` stay absent (`q=` / | reviewer-28 | open |
| LS-008 | `live-smoke.md:152` [Soak (1-3 h) — watch at start and end] - **Stale skips (never met a mic, L-004)** — after any degrade, read the transcript for | reviewer-28 | open |
| LS-009 | `live-smoke.md:161` [Soak (1-3 h) — watch at start and end] - **Thread rotation** — about every 100 translation turns one `TGT` lands a few seconds sl | reviewer-28 | open |
| LS-010 | `live-smoke.md:163` [Soak (1-3 h) — watch at start and end] - **Quota** — out of band via `account/rateLimits/read`; expect ≈0 % primary-window moveme | reviewer-28 | open |
| LS-011 | `live-smoke.md:164` [Soak (1-3 h) — watch at start and end] - **Memory** (external `ps`/`top`): RSS flat — ring, audio headroom and segment queue are  | reviewer-28 | open |
| LS-012 | `live-smoke.md:169` [Soak (1-3 h) — watch at start and end] **Run `uv run python session_report.py` after any soak** — it answers every question in th | reviewer-28 | open |
| LS-013 | `live-smoke.md:176` [Soak (1-3 h) — watch at start and end] TGT stopping while SRC continues is the sanctioned source-only degrade (D-009); the transc | reviewer-28 | open |
| OA-001 | `openvino-accel.md:14` [OpenVINO accelerator path] Device-selection law (exact names, static shapes, `EXECUTION_DEVICES` readback) is machine | reviewer-30 | open |
| OA-002 | `openvino-accel.md:19` [The whisper prelude — both halves, every] code block | reviewer-30 | open |
| OA-003 | `openvino-accel.md:24` [The whisper prelude — both halves, every] - **Without `env.sh` the NPU aborts**: `Cannot load library ".../libopenvino_intel_npu_com | reviewer-30 | open |
| OA-004 | `openvino-accel.md:26` [The whisper prelude — both halves, every] - **The `PYTHONPATH` half has two sources, either alone fatal**: the host login profile so | reviewer-30 | open |
| OA-005 | `openvino-accel.md:30` [The whisper prelude — both halves, every] - **`import openvino` still succeeds and is the trap.** OpenVINO extends `__path__`, so it | reviewer-30 | open |
| OA-006 | `openvino-accel.md:34` [The whisper prelude — both halves, every] - **The `LD_LIBRARY_PATH` half: a tarball at the wheel's own version swaps in its genai bu | reviewer-30 | open |
| OA-007 | `openvino-accel.md:45` [The whisper prelude — both halves, every] - direnv applies `.envrc` in hooked interactive shells ALONE, and the user's shells hook n | reviewer-30 | open |
| OA-008 | `openvino-accel.md:59` [Trees, layers, hardware] - Core Ultra 7 268V: Arc 140V iGPU (`8086:64a0`, `xe`, `/dev/dri/renderD128`) + AI Boost N | reviewer-30 | open |
| OA-009 | `openvino-accel.md:61` [Trees, layers, hardware] - Two GenAI trees exist and only one loads in the container (paths → the user-level OpenVI | reviewer-30 | open |
| OA-010 | `openvino-accel.md:65` [Trees, layers, hardware] - **The deployment path is the PyPI wheels.** `openvino` / `openvino-genai` publish cp314 | reviewer-30 | open |
| OA-011 | `openvino-accel.md:73` [Trees, layers, hardware] - `intel-accel/` is a reversible container-only symlink farm over the host Intel drivers,  | reviewer-30 | open |
| OA-012 | `openvino-accel.md:79` [Trees, layers, hardware] - **The NPU compiler pair must be the UMD release's own.** UMD 1.38 dlopens | reviewer-30 | open |
| OA-013 | `openvino-accel.md:94` [Placement + benchmarking] - `openvino_genai.WhisperPipeline` wraps its compiled model without exposing it, so there  | reviewer-30 | open |
| OA-014 | `openvino-accel.md:99` [Placement + benchmarking] - **Never infer acceleration from a requested provider.** sherpa-onnx 1.13.2 and 1.13.4 do | reviewer-30 | open |
| OA-015 | `openvino-accel.md:102` [Placement + benchmarking] - Decode cost varies ~20 % run to run on identical inputs; treat any single trace as one s | reviewer-30 | open |
| OA-016 | `openvino-accel.md:104` [Placement + benchmarking] - Energy is unmeasurable here: RAPL `energy_uj` is mode 400 `nobody:nogroup` in this conta | reviewer-30 | open |
| OA-017 | `openvino-accel.md:109` [Model cache] `models/openvino/cache` is `OPENVINO_CACHE_DIR` and fully regenerable — `live_stt.py` mkdi | reviewer-30 | open |
| OA-018 | `openvino-accel.md:116` [Model cache] - **A cache hit hides the compiler.** A warm run imports blobs and never compiles, so a co | reviewer-30 | open |
| OA-019 | `openvino-accel.md:119` [Model cache] - **A wheel upgrade can reject a cached decoder blob and DELETE it on load**: 2026.4.1 dro | reviewer-30 | open |
| OA-020 | `openvino-accel.md:128` [Host OpenCL loader] - **Exit 139 usually means SIGSEGV (128 + 11), a native crash: Python prints no traceback, | reviewer-30 | open |
| OA-021 | `openvino-accel.md:132` [Host OpenCL loader] - The host ocl-icd loader crashes in `clGetPlatformIDs` (`readdir64`) on this host's layou | reviewer-30 | open |
| OA-022 | `openvino-accel.md:143` [Host OpenCL loader] - `--asr-device GPU` still starts that loader on the host by design ⇒ it needs | reviewer-30 | open |
| OR-001 | `orientation.md:3` [Orientation] Real-time Japanese STT + English translation. STT is fully local, no API keys (D-009). Per | reviewer-32 | open |
| OR-002 | `orientation.md:7` [Orientation] Pipeline: mic → `Resampler` → 8 s `AudioQueue` → silero VAD (on a leveled copy, `LeveledVa | reviewer-32 | open |
| OR-003 | `orientation.md:19` [Files — read one only when the task impl] - `live_stt.py` (~1.9 K lines) — the whole app; the constants at its top are the config su | reviewer-32 | open |
| OR-004 | `orientation.md:20` [Files — read one only when the task impl] - `streaming.py` — the VAC hypothesis buffer, pure text-in/text-out. | reviewer-32 | open |
| OR-005 | `orientation.md:21` [Files — read one only when the task impl] - `gate.py` — the executable quality gate. `replay.py` — WAV → real `worker` replay (D-014 | reviewer-32 | open |
| OR-006 | `orientation.md:23` [Files — read one only when the task impl] - `tests/` — fast locks, on-demand `eval_*.py` evaluators, committed corpora + traces. | reviewer-32 | open |
| OR-007 | `orientation.md:24` [Files — read one only when the task impl] - `models/` — gitignored weights; `models/README.md` carries the download commands. | reviewer-32 | open |
| OR-008 | `orientation.md:25` [Files — read one only when the task impl] - `transcripts/` — gitignored saved sessions, one file per run, saving ON by default. | reviewer-32 | open |
| OR-009 | `orientation.md:26` [Files — read one only when the task impl] - `README.md` — the user doc; it, `models/README.md` and the CLI strings in `live_stt.py`  | reviewer-32 | open |
| OR-010 | `orientation.md:28` [Files — read one only when the task impl] - `.agent/spec.md` = Intent · Artifacts · Decisions · Tasks · Phase; the sole attached sta | reviewer-32 | open |
| OR-011 | `orientation.md:34` [Files — read one only when the task impl] - `.claude/rules/` = this law, the sole carrier of what a teammate must hold. `CLAUDE.md`, | reviewer-32 | open |
| OR-012 | `orientation.md:41` [Files — read one only when the task impl] Rulings on the template's spec layout (detail → `.agent/spec.md` `Decisions`): | reviewer-32 | open |
| OR-013 | `orientation.md:43` [Files — read one only when the task impl] - Template ruling: "the phase checklist: `- [ ]` open units in order" + "ticked rows clear | reviewer-32 | open |
| OR-014 | `orientation.md:49` [Files — read one only when the task impl] - Template ruling: "Routed state: `Phase` line 1 = `<PHASE> — <scope>`" → **ADAPTED** at t | reviewer-32 | open |
| OR-015 | `orientation.md:52` [Files — read one only when the task impl] - Template ruling: "detail → `docs/` or `.agent/archive/`" → **ADAPTED** — no `docs/`: dec | reviewer-32 | open |
| OR-016 | `orientation.md:56` [Files — read one only when the task impl] Template ruling: "omit provenance — dates, verification/discovery events, origin stories"  | reviewer-32 | open |
| OR-017 | `orientation.md:61` [Files — read one only when the task impl] `D-###` / `L-###` ids anchor a code or test comment to the rule that explains it. They liv | reviewer-32 | open |
| OR-018 | `orientation.md:66` [House style] - No frameworks, DI, config systems or back-compat shims. Module-level imports. Inline by  | reviewer-32 | open |
| OR-019 | `orientation.md:68` [House style] - Comments carry the WHY: the constraint, measurement or upstream quirk behind a peculiar  | reviewer-32 | open |
| OR-020 | `orientation.md:72` [House style] - Edit code only when you can name the failure mode the edit prevents (L-001). A refactor  | reviewer-32 | open |
| OR-021 | `orientation.md:75` [House style] - D-002 — one file. A further split needs a cohesive one-way subsystem boundary named out  | reviewer-32 | open |
| OR-022 | `orientation.md:79` [House style] - A single user preference is state, not an `always` law: record placement and config choi | reviewer-32 | open |
| OR-023 | `orientation.md:82` [House style] `rg` skips dotdirs → pass `--hidden --glob '!.git/**'` to reach `.agent/` + `.claude/`. | reviewer-32 | open |
| PD-001 | `packaging-deps.md:11` [Packaging + dependency maintenance] **Two packaging traps, fixed, never reintroduce.** The wheel `only-include` must list **ev | reviewer-31 | open |
| PD-002 | `packaging-deps.md:17` [Packaging + dependency maintenance] `openvino` is a hard dependency, not an extra — plain `uv sync` installs it. `sherpa-onnx` | reviewer-31 | open |
| PD-003 | `packaging-deps.md:21` [Packaging + dependency maintenance] **D-007 — pre-commit via `.githooks/` + `core.hooksPath`, not the `pre-commit` framework.* | reviewer-31 | open |
| PD-004 | `packaging-deps.md:32` [Packaging + dependency maintenance] **L-018 — maintenance-pass recipe.** Inventory = every `uv.lock` package against its PyPI  | reviewer-31 | open |
| PD-005 | `packaging-deps.md:48` [Packaging + dependency maintenance] **Pins outside `uv.lock` re-qualify by rebuild, never by edit** — each fires on an upgrade | reviewer-31 | open |
| PD-006 | `packaging-deps.md:50` [Packaging + dependency maintenance] - `tests/fetch_real_clips.py` `BUILD_DEPENDENCIES` holds numpy + pyarrow (+ soundfile, lib | reviewer-31 | open |
| PD-007 | `packaging-deps.md:57` [Packaging + dependency maintenance] - `tests/fetch_fleurs_parallel.py` stamps the interpreter's patch version and | reviewer-31 | open |
| PD-008 | `packaging-deps.md:60` [Packaging + dependency maintenance] - `models/README.md`'s export recipe re-qualifies by DECODE, never bytes — its decoder `.b | reviewer-31 | open |
| PD-009 | `packaging-deps.md:76` [Packaging + dependency maintenance] - `tests/eval_cer.py` rewrites `tests/cer_baseline.json` on a pass ⇒ attribute a moved cel | reviewer-31 | open |
| PD-010 | `packaging-deps.md:79` [Packaging + dependency maintenance] `pip-audit` lives HERE and not in `gate.py` because it needs an advisory feed and the gate | reviewer-31 | open |
| TC-001 | `toolchain.md:3` [Toolchain] **Gate = `uv run --no-sync python gate.py`** — 8 blocking steps, `--only NAME` runs one, ` | reviewer-31 | open |
| TC-002 | `toolchain.md:15` [Toolchain] **Fast gate (iteration) = the steps + tests the change reaches, launched as parallel calls | reviewer-31 | open |
| TC-003 | `toolchain.md:22` [Toolchain] **Every skip declares itself, because pytest exits 0 on a skipped case.** A resource gate  | reviewer-31 | open |
| TC-004 | `toolchain.md:29` [Toolchain] **Every step ships the input that fires it, and the gate runs those inputs on itself** (`g | reviewer-31 | open |
| TC-005 | `toolchain.md:32` [Toolchain] **`ruff-format` governs Markdown too, and `CLAUDE.md` is carved out of it.** ruff 0.16 for | reviewer-31 | open |
| TC-006 | `toolchain.md:39` [Toolchain] `tests/test_law_consistency.py` rides the pytest step and locks the five law invariants a  | reviewer-31 | open |
| TC-007 | `toolchain.md:46` [Toolchain] The third is the same hole in the other direction: **law may name a queue row, and a row d | reviewer-31 | open |
| TC-008 | `toolchain.md:54` [Toolchain] The fourth locks the ruling tags (`orientation.md`), which a refresh re-reads against the  | reviewer-31 | open |
| TC-009 | `toolchain.md:68` [Toolchain] The fifth locks what law quotes out of `.agent/spec.md`'s `Intent`: **a unit renames the t | reviewer-31 | open |
| TC-010 | `toolchain.md:78` [Toolchain] **Pin the venv layer in every agent command.** `.venv` = container (agent dev + test), `.v | reviewer-31 | open |
| TC-011 | `toolchain.md:84` [Toolchain] code block | reviewer-31 | open |
| TC-012 | `toolchain.md:97` [Toolchain] Template ruling: "gates + CI from its first commit" + "security scanning + update automati | reviewer-31 | open |
| TC-013 | `toolchain.md:105` [Toolchain] Template ruling: "runs its full gate as a parallel CI fan-out" + "`ci/*` pushes" → **INAPP | reviewer-31 | open |
| TC-014 | `toolchain.md:111` [Toolchain] **Security scanning is split by hermeticity.** Static analysis rides the existing `ruff-ch | reviewer-31 | open |
| TC-015 | `toolchain.md:118` [Toolchain] **The secret scan runs with `HexHighEntropyString` off and skips `tests/*.json`.** This re | reviewer-31 | open |
| TC-016 | `toolchain.md:126` [Toolchain] `ruff` is not on `PATH` — the module form (`python -m ruff`) works from any environment th | reviewer-31 | open |
| TC-017 | `toolchain.md:132` [Toolchain] **Whisper, NPU or GPU work needs BOTH prelude halves, every time, in order:** source the | reviewer-31 | open |
| TC-018 | `toolchain.md:138` [Toolchain] Template ruling: "Project files stay portable: repo-relative paths; machine-bound facts → | reviewer-31 | open |
| TC-019 | `toolchain.md:146` [Toolchain] **Never run two whisper processes against one cache while either is compiling.** Two proce | reviewer-31 | open |
| TC-020 | `toolchain.md:160` [Toolchain] `sounddevice` dlopens system PortAudio on the live and device entry paths; without it they | reviewer-31 | open |
| TC-021 | `toolchain.md:166` [Toolchain] **Teammate worktree gate recipe** (`.scratch/worktrees/<name>`): use the primary tree's `. | reviewer-31 | open |
| TC-022 | `toolchain.md:177` [Toolchain] **A teammate commits inside its worktree with `git -c core.hooksPath=/dev/null commit`.**  | reviewer-31 | open |
| TL-001 | `translation-leg.md:17` [Transport + configuration (D-011)] Persistent `codex app-server` subprocess, newline-delimited JSON-RPC over stdio: `initiali | reviewer-29 | open |
| TL-002 | `translation-leg.md:22` [Transport + configuration (D-011)] - **Model `gpt-6-luna` + `effort:low`** — user ruling on release, swapped in for `gpt-5.6- | reviewer-29 | open |
| TL-003 | `translation-leg.md:37` [Transport + configuration (D-011)] - **Tool-feature disables in `_CODEX_CONFIG` are THE latency lever.** Re-enabling | reviewer-29 | open |
| TL-004 | `translation-leg.md:40` [Transport + configuration (D-011)] - **A translation thread holds no tool that reads the machine — its input is room audio.** | reviewer-29 | open |
| TL-005 | `translation-leg.md:62` [Transport + configuration (D-011)] - **`serviceTier:"priority"`** = Codex's "Fast" tier (1.5× speed, increased usage), set PE | reviewer-29 | open |
| TL-006 | `translation-leg.md:67` [Transport + configuration (D-011)] - **Role pinned via `developerInstructions` on thread/start** — injection-resistant, imper | reviewer-29 | open |
| TL-007 | `translation-leg.md:75` [Transport + configuration (D-011)] - Marginal cost ≈180 in + 7-60 out tokens/turn. Luna bills the shared `codex` bucket, so t | reviewer-29 | open |
| TL-008 | `translation-leg.md:78` [Transport + configuration (D-011)] - Codex auth is external state (`~/.codex/auth.json` — check existence only, never read).  | reviewer-29 | open |
| TL-009 | `translation-leg.md:81` [Transport + configuration (D-011)] - Revisit the whole choice if Luna latency or entitlement changes, if sustained quota burn | reviewer-29 | open |
| TL-010 | `translation-leg.md:83` [Transport + configuration (D-011)] - **Two-way (JA↔EN) rides TWO immutable direction-specific threads in the ONE app-server,  | reviewer-29 | open |
| TL-011 | `translation-leg.md:108` [Degradation contract (locked by `tests/t] - A >64 KiB line, a broken transport, or EOF routes into cleanup that enqueues one `{metho | reviewer-29 | open |
| TL-012 | `translation-leg.md:112` [Degradation contract (locked by `tests/t] - **A turn collects only its own notes, and an abort interrupts the turn it names.** Every | reviewer-29 | open |
| TL-013 | `translation-leg.md:136` [Degradation contract (locked by `tests/t] - **L-033 — validate liveness at the COMMIT point, not only at spawn.** `start()` re-check | reviewer-29 | open |
| TL-014 | `translation-leg.md:139` [Degradation contract (locked by `tests/t] - Both disable paths run through `_disable` ⇒ one stderr line **plus one | reviewer-29 | open |
| TL-015 | `translation-leg.md:146` [Degradation contract (locked by `tests/t] - **Recovery is `_recover()`, driven inline from `run()` on the caption that finds the leg | reviewer-29 | open |
| TL-016 | `translation-leg.md:149` [Degradation contract (locked by `tests/t] - **EOF ⇒ `_respawn`.** Nothing is left to talk to, so `start()` re-runs whole. **4.8 s**  | reviewer-29 | open |
| TL-017 | `translation-leg.md:151` [Degradation contract (locked by `tests/t] - **3 strikes ⇒ `_probe`**: a fresh thread plus one turn on the SURVIVING process. Measure | reviewer-29 | open |
| TL-018 | `translation-leg.md:157` [Degradation contract (locked by `tests/t] - **The probe opens a fresh thread**: a turn there cannot be steered into one the strikes  | reviewer-29 | open |
| TL-019 | `translation-leg.md:162` [Degradation contract (locked by `tests/t] - The design fork was priced and inline won: a background recovery task buys idle-gap repa | reviewer-29 | open |
| TL-020 | `translation-leg.md:165` [Degradation contract (locked by `tests/t] - **Drain `_notes` before the handshake**, on both arms. The EOF cleanup's own wake sentin | reviewer-29 | open |
| TL-021 | `translation-leg.md:169` [Degradation contract (locked by `tests/t] - `submit` queues while the leg is down but still recoverable, because `run()` is parked o | reviewer-29 | open |
| TL-022 | `translation-leg.md:172` [Degradation contract (locked by `tests/t] - Bounds, shared by both arms so a leg that dies two ways cannot spend twice: | reviewer-29 | open |
| TL-023 | `translation-leg.md:178` [Degradation contract (locked by `tests/t] - **Shutdown latches recovery off at both ends**, `submit_sentinel` and `close()`, each wi | reviewer-29 | open |
| TL-024 | `translation-leg.md:182` [Degradation contract (locked by `tests/t] - **The app-server spawns `start_new_session=True`, OUT of the terminal's process group.** | reviewer-29 | open |
| TL-025 | `translation-leg.md:190` [Degradation contract (locked by `tests/t] - `_restore` mirrors `_disable` into both channels ⇒ `-- translation restored: codex app-s | reviewer-29 | open |
| TL-026 | `translation-leg.md:195` [Degradation contract (locked by `tests/t] - **A caption too stale to be worth translating is published source-only** — | reviewer-29 | open |
| TL-027 | `translation-leg.md:215` [Degradation contract (locked by `tests/t] - **A repeated short unit makes the translator generate without terminating.** Measured th | reviewer-29 | open |
| TL-028 | `translation-leg.md:227` [Live validation — two sessions on a real] The leg's only mic-side evidence, and what it does and does not settle. Both sessions are  | reviewer-29 | open |
| TL-029 | `translation-leg.md:231` [Live validation — two sessions on a real] \| session \| span \| captions \| translated \| degrade \| restore \| p50 \| p90 \| max \| | reviewer-29 | open |
| TL-030 | `translation-leg.md:236` [Live validation — two sessions on a real] - **TGT behind SRC replicates at p50 2 s.** `spec.md`'s `Intent` asks for `TGT n:` about a | reviewer-29 | open |
| TL-031 | `translation-leg.md:244` [Live validation — two sessions on a real] - **The tail is NOT turn latency. It is one wedge-and-recover cascade, and it is now bound | reviewer-29 | open |
| TL-032 | `translation-leg.md:256` [Live validation — two sessions on a real] - **The rotation tax did not surface in either session.** Session 1: 4 turns ≥6 s, **none  | reviewer-29 | open |
| TL-033 | `translation-leg.md:263` [Live validation — two sessions on a real] - **`_probe` IS live-validated; `_respawn` is not.** The 3-strike arm fired on a real mic  | reviewer-29 | open |
| TL-034 | `translation-leg.md:270` [What the target lag is made of — 300 rea] The bench half of the lag question, run under L-026: 12 committed captions from | reviewer-29 | open |
| TL-035 | `translation-leg.md:278` [What the target lag is made of — 300 rea] \| arm \| what it adds \| median \| p90 \| max \| | reviewer-29 | open |
| TL-036 | `translation-leg.md:286` [What the target lag is made of — 300 rea] - **Every candidate the row suspected came back NEGATIVE: the shipped configuration is FAS | reviewer-29 | open |
| TL-037 | `translation-leg.md:295` [What the target lag is made of — 300 rea] - **`serviceTier` DOES land — that suspicion is retired.** `thread/start` echoed the field | reviewer-29 | open |
| TL-038 | `translation-leg.md:297` [What the target lag is made of — 300 rea] - **The emit path is REFUTED as a lag term by six orders of magnitude.** `turn/completed`  | reviewer-29 | open |
| TL-039 | `translation-leg.md:301` [What the target lag is made of — 300 rea] - **The residual IS the bare turn, and it dwarfs every isolated term.** Subtracting the th | reviewer-29 | open |
| TL-040 | `translation-leg.md:306` [What the target lag is made of — 300 rea] - **What this explains about the live sessions, and what it does not.** A4 + emit = 1.549  | reviewer-29 | open |
| TL-041 | `translation-leg.md:319` [Session context learner (D-015)] Learned from the session's own captions, held in memory, discarded at exit. Locked by | reviewer-29 | open |
| TL-042 | `translation-leg.md:322` [Session context learner (D-015)] - **Candidates are restricted script runs, not n-grams** (`_TERM_RUN`: katakana ≥2, kanji  | reviewer-29 | open |
| TL-043 | `translation-leg.md:327` [Session context learner (D-015)] - **Promotion + lease.** A candidate is promoted after `CONTEXT_TERM_SUPPORT`=3 sightings  | reviewer-29 | open |
| TL-044 | `translation-leg.md:333` [Session context learner (D-015)] - Bounds: `CONTEXT_MAX_TERMS`=12 · `CONTEXT_TERM_MEMORY`=40 segments of candidate patience | reviewer-29 | open |
| TL-045 | `translation-leg.md:335` [Session context learner (D-015)] - `--context TEXT` is the user's correction channel: NFKC-normalized, trusted at once, nev | reviewer-29 | open |
| TL-046 | `translation-leg.md:337` [Session context learner (D-015)] - One consumer: `translator_brief()` rides the thread's `developerInstructions`, never tur | reviewer-29 | open |
| TL-047 | `translation-leg.md:344` [Session context learner (D-015)] - **`observe_en` pairs each trusted term with its English spelling, and the pairing is wha | reviewer-29 | open |
| TL-048 | `translation-leg.md:364` [Session context learner (D-015)] - **`_EN_STOP` (the pronoun "I" + its four contractions) and `_EN_SENTENCE` are the shippe | reviewer-29 | open |
| TL-049 | `translation-leg.md:373` [Session context learner (D-015)] - **Rejected, never re-propose: a JA-side plausibility test on the KEY.** It drops the cor | reviewer-29 | open |
| TL-050 | `translation-leg.md:376` [Session context learner (D-015)] - **Rejected: LLM-written running summaries and background extractor turns** — codex must  | reviewer-29 | open |
| TL-051 | `translation-leg.md:378` [Session context learner (D-015)] - `_TERM_RUN`'s katakana floor is **2**, matching the kanji floor. Over the 215-caption st | reviewer-29 | open |
| TL-052 | `translation-leg.md:387` [Session context learner (D-015)] - **Honest limits, all measured.** A persistent ASR error recurring 3× IS learned, and `-- | reviewer-29 | open |
| TL-053 | `translation-leg.md:391` [Session context learner (D-015)] - **Simulation is the learner's best case; check a claim against the live arm.** An offlin | reviewer-29 | open |
