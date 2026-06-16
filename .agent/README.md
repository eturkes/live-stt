# `.agent/` — Memory/Notetaking System

Persistent context for agent sessions. Read on every fresh session before touching project code.

**Fresh session:** run `/session-prompt [TASK]` — the `.claude/commands/session-prompt.md` slash command (successor to the old `.agent/SESSION_PROMPT.md`). It walks the files below in order. Blank `TASK` follows the `PLAN.md` roadmap; a value (task ID or free text) overrides the roadmap for that session.

## Files

| File | Role | Read when |
|---|---|---|
| `orientation.md` | Project-specific facts: file map, smoke-test constraints, style guide | First read after `CLAUDE.md` |
| `journal.md` | Chronological session log: dated entries of what each session did | Catching up on recent history |
| `lessons.md` | Mistakes-and-fixes; generalizable rules harvested from past errors | Before attempting a task similar to a past one |
| `decisions.md` | Architectural/design decisions with rationale (ADR-style, compact) | Before re-litigating a settled design call |
| `smoke.md` | Live-path smoke + soak checklist: the user-only L-004 paths, each item tied to a `live_stt.py` observable | About to smoke-test the live path, or closing an L-004 disclaimer |
| `scratch/` | Ephemeral per-task notes (committed). One file per task: `YYYY-MM-DD_<topic>.md` | While actively working a task |

## Conventions

- All files are LLM-optimized: dense, structured, machine-parseable. Tables and bullets over prose.
- Append to `journal.md` at end of each session — one section per session, prefixed with ISO date. Then prune to the **≤4 most-recent entries** (see Pruning).
- When you discover a generalizable error pattern, promote it from `scratch/` into `lessons.md`.
- When you make a non-obvious design choice, log it in `decisions.md` with a short rationale.
- Do **not** duplicate `CLAUDE.md`. Reference its rules by topic (e.g. "per CLAUDE.md memory-system rule") instead.
- Every entry must earn its place: capture only what a future agent cannot already get from the codebase, Git history, project docs, or `CLAUDE.md`. Favor durable facts — invariants, rationale, gotchas — over drift-prone detail like exact package versions. Per CLAUDE.md memory-system rule.
- Scratch notes are committed (per user choice) — name them descriptively so they remain useful in retrospect.

## Pruning

`journal.md` keeps the **≤4 most-recent entries**. Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are **not** moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. (No `archive/` dir — it duplicated what git already provides.)

`lessons.md` / `decisions.md` are append-only and grow slowly; prune only genuinely obsolete content, and supersede via a new ID rather than deleting (per CLAUDE.md 200K-context rule). A superseded entry may be trimmed to a one-line pointer at its successor — full text stays in git.
