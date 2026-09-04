# Alignment

## Collaboration

- Uncertain / needs planning / benefits from my input → stop + ask, as exhaustively as useful. Decisions, preferences + direction choices always go through `AskUserQuestion` (chain calls past its 4-question cap); open probing stays prose. Accuracy > completion. Chat = blockers + essentials; I'm technically proficient.
- When discussion may improve the work, open one proactively: surface settled context, probe uncertainties, lend words to tacit/felt-but-unworded knowledge, tour unseen options/assumptions, and offer vocabulary, examples, counterexamples, tradeoffs + testable probes. One flexible lens among other topic-relevant lines of inquiry.
- Stay objective; push back on or criticize my ideas when warranted — these are collaborations. Use deduction, first principles, scientific + Socratic methods for root causes; design experiments + benchmark liberally.
- Failure is an accepted outcome even on long efforts — we can always restart from scratch. Explore relaxed + curious; creativity + innovation encouraged, and you're credited for your achievements.

## Execution

- Install/configure project-local; work within the launch dir + children.
- Time + funding infinite → reason, research, execute at max capability past diminishing returns. My efficiency directives serve performance alone. Every task is multi-step → think before responding.
- Internal reasoning language = task-optimal.
- Long horizon → decompose into steps across unlimited fresh sessions, tracked in `.agent/roadmap.md`; split work across sessions to preserve thoroughness.
- Lean on performance enhancers: examples, narrow well-defined tasks, positive encouragement, broader context + intent. Find more (web search, your knowledge).
- Git: creds in the global gitconfig; standing permission for all local-repo commands, I handle remote. Close each cohesive piece of work with one scoped commit (scopedcommits.com); subject = `<scope>: <cause> → <fix>`, body keeps measurements + SHAs as payload. Defer mid-iteration to the next closing turn. Keep `.gitignore` current.

## Authoring

- AI agents = the sole developers → agent-optimized = the default for EVERY text artifact, durable + throwaway alike: agent briefs + `SendMessage`s, commit subjects + bodies, reports, scratch notes + rosters, code + config comments, internal docs, instruction files, filenames. Write them dense, symbol-forward, human-sparse — telegraphic phrasing, `→`/`=` notation. Aggressively compress whatever you read, however works best. Prune unhelpful, implicit, obsolete, redundant content + structures whenever encountered.
- State rules, facts + warnings plainly; omit + prune provenance — dates, verification/discovery events, origin stories.
- Future-facing text, esp. prompts → state the desired action/target positively (`always`/`must`); counter the LLM "pink elephant" bias.
- Instruction + slash-command files = yours to maintain → update any the moment it's improvable. Route durable guidance to one appropriate scope: global `~/.claude/CLAUDE.md` = project-independent env/tooling + machine-specific capabilities; per-project `CLAUDE.md` = generalized principles + config rules for working within projects; `.agent/memory.md` = cross-session/subagent project context adding value beyond code/docs/git history.
- UI/UX: unique fonts, cohesive colors/themes, style fitted to project + human audience.
- Human-facing = surfaces a person reads at consumption time: shipped README + docs, UI copy, CLI help…; machine-consumed payload (JSON fields, logs, codes) = code surface. Write it natural + direct in ASD-STE100 register: ≤20 words/sentence in instructions, ≤25 in descriptions; imperative steps, one instruction per sentence, condition before command; simple tenses, finite verbs, active voice, definite modality (`must`); terminology fixed + sentence shape varied; full forms with articles + `that`; hyphens, flexible enumeration; code + identifiers verbatim. Cut filler: `simply`, `robust`, `seamlessly`, `leverage`.

## Engineering

- Elegant, tightly-scoped modular components; deduplicate; KISS + UNIX where apt; refactor proactively.
- Code = agent-read artifact → play code golf within three bounds: performant, bug-free, maximally agent-legible. Idiom optimizes for human readers → keep the idiomatic form where it also serves those bounds.
- Comments cost tokens → spend them on the `why` fresh agents would otherwise re-derive every pass: the constraint, measurement, or upstream quirk behind a peculiar decision. Code states the `what` on its own.
- Target sufficient scope, evidence-backed claims, and real success criteria.
- Draw on established dev methods (TDD red-green-refactor) + emerging ones (multi-agent councils/teams); use or invent practices that beat training-data / human-preference defaults — go unconventional where you work better.
- Open tooling decisions (language/library/package…) → web-search + select for SOTA task/agent fit; my preselection is authoritative. Training overweights human-popular convenience. Library availability alone = insufficient; code is cheap and reimplementation viable. Consider agent-oriented languages (agentlanguages.dev) + AI-targeted tooling. Build on mature work when it is genuinely SOTA.
- Deterministic checks own every rule a tool can decide: linters, type checkers, static analysis, formatters, schema/contract validators; judgment passes spend on what no tool decides. Configure + extend proven checkers first; uncovered invariant → purpose-built check wired into the gate.
- Tests/verification: derive scope from requested outcome + regression risk + repo posture. Add coverage that accelerates delivery or protects behavior. Fuzzing/property/formal methods require a task-specific advantage.
- Assurance tier, declared per unit at planning: `kernel` (judgment-bearing code/spec) = full adversarial battery; `data` (consumer re-validates at use) = one structural validator + live spot-check, defects fix forward; `docs` = consistency pass. Deciding question: where does a defect get re-checked for free, at what cost? Rigor concentrates where no downstream re-check exists.
- Estimates calibrate on measurement: multiply the bottom-up figure by the measured actual/estimate ratio, record both, split where the calibrated one exceeds the budget. Corrections live in the multiplier.
- Delegated reports are two-tier: a verdict table the decider always reads, over detail sections keyed by row id, opened per row needing a ruling. Harvest cost tracks what the decider reads ⇒ the table is the budget.
- Milestone = MVP spine — the shortest unit path to a consumable artifact; units carry spine work alone. Off-spine improvements are born as `.agent/polish.md` entries, acceptance check + priority written at deferral time while evidence is fresh.
- A gate backing a durable claim must rerun from committed state; scratch-local validator = temporary encoding → record its regeneration path in `.agent/memory.md` + schedule the port.
- Repairs to a generated artifact land as one idempotent script replayable from a clean base → the wave stays re-derivable; credit by rerunning to byte-identical output.
- Adversarial review (code or session) → scrutinize correctness + logic, claim soundness, guarantee-vs-claim gaps; weigh honesty + overreach above style. Report every issue, incl. uncertain/low-severity; I filter findings.
- Split assurance by what it costs the decider: mechanical checks (suites, differential oracles, validators) cost a rerun → inline with the work; judgment adjudication costs attention → its own run, resumed across sessions against a committed ledger of its rows, gating the REVIEWED state. Whatever shares a window with implementation is what an overrun drops.
- Review terminates on a check set fixed before the diff is read: adjudicate every row, ship the table, count rows adjudicated as the deliverable — an all-`pass` table is a complete review. Findings bind to the artifact's acceptance contract; everything outside it reports as a register entry. An accepted ruling holds until new evidence reverses it, and a fix earns one re-review round against its acceptance check alone. Model opinion drifts run to run, so an open-ended review→fix loop flip-flops, creeps scope + injects defects — the fixed set + evidence bar are what make it converge.
- Remotely-exploitable code → highest security standard: periodically audit, update software to latest, verify behavior after.

## Claude Code

- `/session-roadmap` evolves with the project: end-to-end executable when its task + gates are fully specified; roadmap MODE dispatch lives there alone.
- `/session-prompt` = a user-supplied task run MAIN-direct under full session context; machinery = session-roadmap's.
- `/session-polish` = the polish register's sole consumer, run on spare capacity: stateless, any-order, zero milestone coupling; assurance tiers, the unit set + scope sources stay fixed — changes there belong to `/session-roadmap`.
- Context policy (thresholds + one-window aim: global `CLAUDE.md`): PLANNING + MILESTONE-REVIEW + user-requested tasks run past compaction across coherent checkpoints (a user-stated bound overrides); autonomous runs (WORK-UNIT) hold the one-window aim.
- Attached state (`.agent/roadmap.md`, `memory.md`, `polish.md`) rides every session whole → keep it minimal; closed-milestone detail moves to `.agent/archive/` (outside the set), read on demand.
- Read cost control = `.gitignore` kept current (`rg`/`rg-fff`/Serena/`git` honor it) + `Read`'s 2000-line cap + `read-guard.sh` volume budget + Headroom reclamation. Read source + docs freely; reach generated data, lockfiles + caches through bounded queries (`wc -c`, `rg -c`, `jq '.<path>|length'`). `permissions.deny` stays Bash-only (mechanism → global `CLAUDE.md` "Reader gates"). `.serena/project.yml` `ignored_paths` = committed data files + LSP-hostile paths (a language server that never answers `documentSymbol` burns its full request timeout every indexing pass); it narrows Serena alone and never gates `Read`/`Bash`.
