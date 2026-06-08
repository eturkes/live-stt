# Project Orientation — `live-stt`

Single-file Python tool. Streams microphone audio to the Gemini Live API and prints JA/EN transcripts of Japanese speech in real time.

## File map

| Path | Role |
|---|---|
| `live_stt.py` | Main app (~545 lines). Audio capture in `audio_callback`; transcription via `run_session` → `sender` / `receiver`. |
| `list_live_models.py` | Utility: enumerate Gemini models that support `bidiGenerateContent`. |
| `tests/test_audio.py` | Pure-function tests for `resample()`, `pcm16_bytes()`, `emit_block()`. Run with `uv run pytest`. |
| `pyproject.toml` | `uv`-managed deps. Entry point: `live-stt`. Python ≥ 3.11 (for `TaskGroup`). |
| `.githooks/pre-commit` | Project-local git hook: runs `uv run pytest -q` and aborts commit on failure. Enabled via `git config --local core.hooksPath .githooks` (per-clone, one-time). |
| `.env` | Holds `GEMINI_API_KEY` (removed at T4.5). Loaded via `python-dotenv` at runtime. Gitignored. Fully off-limits to agent tools incl. Bash — ask the user about its contents. |
| `.claude/settings.json` | `permissions.deny` `Read()` rules keeping low-value paths out of context: `.git`, `.venv`, `.env*`, `uv.lock`, `LICENSE`, spike cache, tool caches. Deny-listed paths are refused via **every** tool, Bash included (verified 2026-06-08, D-008 amendment) — ask the user instead of probing. Rationale: D-008. |
| `README.md` | User-facing docs (GitHub-visible). Update only on user-visible behavior changes. |
| `PLAN.md` | Roadmap with task IDs (T1.x, T2.x, T3.x). Source of truth for what to do next. |
| `SPIKE_REPORT.md` | T3.1 spike: REST → Gemini Live migration. Latency/cost data, decision record. Historical. |
| `SPIKE_REPORT_BACKENDS.md` | Backends spike: comparison of 5 streaming STT providers. Awaiting API keys to finalize. |
| `spike/backends/` | Prototypes, research notes, bench harness from the backends spike. |
| `CLAUDE.md` | Meta-instructions for the agent. Agent may rewrite at any time. |
| `compaction.sh` | Context-usage gauge; vendored snapshot of the shared `$HOME/.claude/` tool (re-sync if that changes — L-008). `sh compaction.sh` prints `PCT USED/WINDOW` (e.g. `12% 24K/200K`) from the session JSONL; needs `jq`. A statusline branch (stdin JSON, ANSI color at ≥80/60%) also exists. Backs the CLAUDE.md 80% compaction rule. |
| `.agent/` | This memory system. |

## Smoke-test constraints (agent cannot verify)

The agent must flag these for the user every time they're touched:

- **Microphone capture** — `sd.InputStream` boundary crossing, `loop.call_soon_threadsafe`, `audio_callback` timing.
- **Device enumeration / selection** — `--device N`, `--list-devices`.
- **Real-time latency under live mic** — TTFT, sustained sessions > 2 min.
- **Gemini rate-limit behavior** — mock or skip in tests.
- **Ctrl+C / signal handling** in a real terminal.

## How to work (per-task loop)

1. **Read** `CLAUDE.md` → `.agent/orientation.md` (this file) → `.agent/journal.md` (kept ≤4 entries) → `.agent/lessons.md` → `PLAN.md`.
2. **Pick** the next open task in priority order (T1 → T2 → T3, numerical within). Restate its acceptance criteria.
3. **Plan** in a scratch file under `.agent/scratch/YYYY-MM-DD_<task-id>.md` if the task is non-trivial.
4. **Edit** the smallest change that satisfies the acceptance criteria. Reference `live_stt.py:<line>` anchors in edits.
5. **Verify** what you can: `uv run python -c "import live_stt"` (syntax/imports), `uv run pytest` (pure fns). For audio/network paths, state explicitly that you couldn't smoke-test and list what the user needs to verify.
6. **Update** `PLAN.md` (mark shipped, or revise open). Update `README.md` only if user-visible CLI/behavior changed.
7. **Log** to `.agent/journal.md` at end of session, then prune it to the **≤4 most-recent entries** (git holds the rest — see `.agent/README.md` § Pruning). Promote any generalizable lesson to `.agent/lessons.md`.
8. **Commit** at end-of-turn when closing cohesive work; defer if mid-iteration awaiting user input. Single focused commit; message optimized for LLM parsing; co-author line per `git log` style.

## Style conventions for `live_stt.py`

- Single file. Constants at the top. No frameworks, no DI, no config systems.
- Comments explain *why*, not *what*. Most existing comments document optimizations (cache reuse, ufunc choice, allocation avoidance). Preserve them.
- Avoid adding abstractions speculatively. The author actively prefers less code over more.
- No backwards-compat shims; this is 0.1.0 with one user.
- Function-local imports avoided; module-level imports preferred for readability.

## Build/test commands

```sh
uv sync                                          # install deps
git config --local core.hooksPath .githooks      # one-time: enable pre-commit hook
uv run live-stt                                  # run with defaults
uv run live-stt --list-devices                   # enumerate audio devices
uv run pytest                                    # run pure-function tests
uv run python -c "import live_stt"               # cheap import smoke-check
uv run python list_live_models.py                # list Gemini Live-capable models
sh compaction.sh                                 # context-usage gauge: PCT USED/WINDOW (needs jq)
```

The pre-commit hook runs `uv run pytest -q` on every `git commit`. A fresh clone needs the `core.hooksPath` step once; `uv sync` does not configure it.

`sounddevice` dlopens system PortAudio at import, so a host/container lacking it fails the import smoke-check (and therefore pytest and the hook) with `OSError: PortAudio library not found`. Install once on Debian: `sudo apt-get install libportaudio2`. `uv sync` does not provide it.

## Known caveats

- Live API audio-only sessions cap at 15 min wall-clock per connection; underlying WS times out at ~10 min. Mitigated by reconnect loop + `SessionResumptionConfig` + `ContextWindowCompressionConfig` in `build_config()`.
- Session resumption handles valid for ~2 h. After expiry, reconnect starts fresh (conversation history lost).
- Native-audio Live models bill audio-output tokens even when discarded (~$0.018/min at list price).
- `python-genai#1224`: `session.receive()` exits its async iterator on `turn_complete`. Worked around via the outer `while not state.stopping` loop in `receiver()` — see `live_stt.py:248-300` (outer while at line 258).
- `python-genai#1859`: scrambled transcripts on >20 s continuous speech. Not reproduced in spike but watched.
