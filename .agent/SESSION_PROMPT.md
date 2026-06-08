# Reusable Session Bootstrap Prompt

Paste the block below into a fresh Claude Code session (or any coding agent with file access) when resuming work on `live-stt`. Append session-specific steering text below the delimiter if you want to focus the session.

---

You are continuing development on **live-stt**, a real-time Japanese speech-to-text + English translation tool. STT is fully local (silero VAD + sherpa-onnx, no API keys); translation rides a Codex subscription via a persistent `codex app-server` subprocess, degrading to JA-only when unavailable. Single-file Python script (`live_stt.py`, ~800 lines), managed with `uv`; model weights in gitignored `models/` (download cmds in `models/README.md`).

## Bootstrap order (do this before anything else)

Read in this order, then summarize back to the user what the next priority task appears to be:

1. **`CLAUDE.md`** — meta-instructions and constraints. Treat as authoritative.
2. **`.agent/orientation.md`** — file map, smoke-test constraints, style guide, how-to-work loop.
3. **`.agent/journal.md`** — all entries (kept ≤4) for recent history.
4. **`.agent/lessons.md`** — generalizable rules from past mistakes. Skim all; reread any that pattern-match the task you're about to do.
5. **`.agent/decisions.md`** — settled architectural choices. Avoid re-litigating these without new evidence.
6. **`PLAN.md`** — open tasks in priority order (lowest task ID first). Pick the lowest-numbered open task and restate its acceptance criteria before starting.

## Working agreement

- Follow `.agent/orientation.md` § "How to work" for the per-task loop.
- Append to `.agent/journal.md` at session end, then prune it to the **≤4 most-recent entries** (git holds older — see `.agent/README.md` § Pruning). Promote any new generalizable lesson into `.agent/lessons.md`. Log non-obvious choices in `.agent/decisions.md`.
- Use `.agent/scratch/YYYY-MM-DD_<task-id>.md` for non-trivial in-task planning. Commit these (per project policy).
- Edit `live_stt.py` minimally. Comments in that file encode optimization rationale — preserve them.
- You may rewrite `CLAUDE.md` at any time if content becomes obsolete or better phrasing is found.
- Commit at end-of-turn when closing cohesive work; defer if mid-iteration awaiting user input. Single focused commit, message optimized for LLM parsing, co-author line per `git log` style.

## Smoke-test constraints

You cannot verify these from inside the agent — flag them for the user every time they're touched: the mic-capture, device-enumeration, live-latency, Ctrl+C/signal, and multi-hour-session paths. The authoritative list lives in `.agent/orientation.md` § "Smoke-test constraints (agent cannot verify)" — read it there rather than maintaining a second copy here.

## What to do right now

State the task you've selected from `PLAN.md`, the acceptance criteria, and ask the user to confirm or redirect before making changes. If the user provided a specific task in their initial message, do that instead and skip `PLAN.md` selection.

---

<!-- USER STEERING (optional) — append below this line to focus the session, e.g. "work on T2.2 only" or "skip the scaffolding step, the project is already set up". -->
