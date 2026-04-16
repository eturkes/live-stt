# Reusable Prompt for Continued Development

Paste the block below into a fresh Claude Code session (or any coding agent) when resuming work on `live-stt`. Replace `<TASK_ID>` with an item from `PLAN.md` (e.g. `T1.1`), or describe an ad-hoc task in the "Task" line.

---

You are continuing development on **live-stt**, a real-time Japanese speech-to-text tool that streams microphone audio to the Gemini API. The project is a single-file Python script (`live_stt.py`, ~270 lines) managed with `uv`.

## Ground rules

- **Keep it simple.** This is a personal tool, not a service. The most recent commit (`3930a8e`) removed VAD and calibration logic — the author actively prefers less code over more. Do not introduce abstractions, config systems, DI, or frameworks. Do not add features beyond what the task asks for.
- **Respect the existing style.** Single file unless the task explicitly requires new ones. Constants at the top. Minimal comments — only where the *why* is non-obvious. No docstrings longer than one line.
- **No backwards-compat shims.** This is a 0.1.0 tool with one user. If you change an interface, just change it.
- **Don't write docs unless asked.** Update `README.md` only when user-visible behavior changes (new flag, changed default). Do not create new `.md` files.
- **Test before declaring done.** Pure-function changes: run `uv run pytest` if tests exist. Runtime/audio changes: state explicitly that you could not verify the mic path and ask the user to smoke-test.

## Project orientation

- `live_stt.py` — main app. Audio capture loop is `main()`; transcription loop is `transcription_worker()`.
- `list_live_models.py` — utility to enumerate Gemini models that support the Live API.
- `pyproject.toml` — deps managed by `uv`; entry point is `live-stt`.
- `PLAN.md` — prioritized roadmap with task IDs (T1.1, T1.2, …). Read this first.
- `.env` — holds `GEMINI_API_KEY`, loaded via `python-dotenv`.

## How to work

1. **Read `PLAN.md`** to understand priorities and what's out of scope.
2. **Confirm the task** with the user if ambiguous. For plan items, restate acceptance criteria before starting.
3. **Make the smallest change that satisfies the task.** Reference specific `live_stt.py:<line>` anchors in your edits.
4. **Run what you can.** `uv run python -c "import live_stt"` at minimum to catch syntax/import errors. `uv run pytest` if tests exist.
5. **Update `README.md`** only if user-visible CLI or behavior changed. Update `PLAN.md` if the task is done (mark completed) or scope shifted.
6. **Report back concisely.** What changed, what you verified, what the user needs to smoke-test (especially anything that touches the live audio path — you cannot run a mic).
7. **Do not commit unless asked.** When asked, use a single focused commit; follow the existing co-author style from `git log`.

## Known constraints you cannot verify from the agent

- Microphone capture, device enumeration, and real-time latency. Always flag these for user smoke-testing.
- Gemini API behavior under rate limits — mock or skip in tests.

## Task

`<describe the task here, or reference a PLAN.md ID like "Execute T1.1 and T1.4 as a single PR per the suggestion in PLAN.md">`
