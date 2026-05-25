# `.agent/` — Memory/Notetaking System

Persistent context for agent sessions. Read on every fresh session before touching project code.

## Files

| File | Role | Read when |
|---|---|---|
| `SESSION_PROMPT.md` | Reusable bootstrap prompt for a new agent session | Starting fresh — copy/paste into the new session |
| `orientation.md` | Project-specific facts: file map, smoke-test constraints, style guide | First read after `CLAUDE.md` |
| `journal.md` | Chronological session log: dated entries of what each session did | Catching up on recent history |
| `lessons.md` | Mistakes-and-fixes; generalizable rules harvested from past errors | Before attempting a task similar to a past one |
| `decisions.md` | Architectural/design decisions with rationale (ADR-style, compact) | Before re-litigating a settled design call |
| `scratch/` | Ephemeral per-task notes (committed). One file per task: `YYYY-MM-DD_<topic>.md` | While actively working a task |
| `archive/` | Superseded artifacts preserved for history (e.g. old prompts) | Rarely; for tracing decisions back |

## Conventions

- All files are LLM-optimized: dense, structured, machine-parseable. Tables and bullets over prose.
- Append to `journal.md` at end of each session — one section per session, prefixed with ISO date.
- When you discover a generalizable error pattern, promote it from `scratch/` into `lessons.md`.
- When you make a non-obvious design choice, log it in `decisions.md` with a short rationale.
- Do **not** duplicate `CLAUDE.md`. Reference its rules by topic (e.g. "per CLAUDE.md memory-system rule") instead.
- Scratch notes are committed (per user choice) — name them descriptively so they remain useful in retrospect.

## Pruning

Per CLAUDE.md (200K-context rule): Prune redundant/obsolete content here from time to time. When pruning, move content to `archive/` rather than deleting outright unless the content is purely noise.
