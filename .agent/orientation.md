# Project Orientation — `live-stt`

Single-file Python tool. Local JA speech-to-text (silero VAD + sherpa-onnx, CPU) with EN translation over a Codex-subscription `codex app-server` subprocess. No API keys (D-009). Prints numbered `JA n:` / `EN n:` lines in real time; degrades to JA-only without codex.

## File map

| Path | Role |
|---|---|
| `live_stt.py` | Main app (~800 lines). `audio_callback` → queue → `worker` (VAD + `RingBuffer` pre-pad re-slice + executor decode) → `emit_line`; `CodexTranslator` (JSON-RPC/stdio per D-011) consumes a sequential queue. Constants at top are the config surface. |
| `models/` | STT weights, gitignored except `models/README.md` (download cmds, expected layout, ~800 MB). |
| `tests/test_audio.py` | Pure-function tests: `resample`, `RingBuffer`, `emit_line`. Run with `uv run pytest`. |
| `pyproject.toml` | `uv`-managed deps (numpy, sounddevice, sherpa-onnx + sherpa-onnx-core). Entry point: `live-stt`. Python ≥ 3.11. |
| `.githooks/pre-commit` | Runs `uv run pytest -q`; aborts commit on failure. Enabled via `git config --local core.hooksPath .githooks` (per-clone, one-time). |
| `.claude/settings.json` | `permissions.deny` `Read()` rules keeping low-value paths out of context: `.git`, `.venv`, `.env*`, `uv.lock`, `LICENSE`, spike cache, `.serena/` (cache + memories + project.local.yml), tool caches. Deny-listed paths are refused via **every** tool, Bash included (D-008 amendment) — ask the user instead of probing. Runtime reads by the app itself are unaffected. Also `enabledPlugins`: pyright-lsp, project-scoped (server = user-level `~/.local/bin/pyright-langserver`; D-008 amendment b). Pyright venv resolution = `[tool.pyright]` in pyproject.toml; the LSP server reads config at session start, so after config edits the in-session diagnostics are stale — run the pyright CLI (build/test commands) for an immediate check. Env (subagent model=opus, effort=max) comes from the **global** `~/.claude/settings.json` — set there, not here. |
| `README.md` | User-facing docs. Update only on user-visible behavior changes. |
| `PLAN.md` | Roadmap with task IDs. Source of truth for what to do next. |
| `SPIKE_REPORT.md` | Historical: REST → Gemini Live migration (architecture removed at T4.5). |
| `SPIKE_REPORT_BACKENDS.md` | Historical: 5-provider streaming-STT comparison; premise (API keys) voided by D-009. |
| `spike/backends/` | Bench harness (`harness.py`, `scenarios.py`, `bench.py`), prototypes (`prototype_local.py` = T4.1 winner pattern, `prototype_gemini.py` = old baseline), `codex_client.py` (T4.2 bench tool, donor of the `CodexTranslator` pattern), `codex_ws/AGENTS.md` (rejected agents-mode comparator). Cached bench WAVs live in the deny-listed `cache/`. |
| `CLAUDE.md` | Meta-instructions for the agent. Agent may rewrite at any time. |
| `compaction.sh` | Context-usage gauge; vendored snapshot of the shared `$HOME/.claude/` tool (re-sync if that changes — L-008). `sh compaction.sh` prints `PCT USED/WINDOW`; needs `jq`. Backs the CLAUDE.md 80% compaction rule. |
| `.serena/` | Headroom/Serena LSP state (Headroom compresses everything the agent reads — CLAUDE.md). `project.yml` = tracked LSP config; `cache/`, `memories/`, `project.local.yml` are git-ignored (nested `.serena/.gitignore`) **and** deny-listed → the agent ignores them (D-013). The project memory system is `.agent/`, **not** `.serena/memories/`. |
| `.agent/` | This memory system. |

## Smoke-test constraints (agent cannot verify)

The agent must flag these for the user every time they're touched:

- **Microphone capture** — `sd.InputStream` boundary crossing, `loop.call_soon_threadsafe`, `audio_callback` timing.
- **Device enumeration / selection** — `--device N`, `--list-devices`.
- **Real-time latency under live mic** — VAD endpointing feel, decode lag, EN cadence on real speech.
- **Ctrl+C / signal handling** in a real terminal (mid-utterance flush, translator drain).
- **Multi-hour sessions** — Codex quota burn over time, thread rotation, memory growth.

The Codex leg itself IS agent-verifiable (subprocess + synthetic turns); the local decode path is verifiable against cached bench WAVs via `spike/backends/harness.py` loaders.

## How to work (per-task loop)

1. **Read** `CLAUDE.md` → `.agent/orientation.md` (this file) → `.agent/journal.md` (kept ≤4 entries) → `.agent/lessons.md` → `PLAN.md`.
2. **Pick** the next open task in priority order. Restate its acceptance criteria.
3. **Plan** in a scratch file under `.agent/scratch/YYYY-MM-DD_<task-id>.md` if the task is non-trivial.
4. **Edit** the smallest change that satisfies the acceptance criteria.
5. **Verify** what you can: `uv run python -c "import live_stt"` (syntax/imports), `uv run pytest` (pure fns), synthetic E2E via cached WAVs if decode/translation paths changed. For mic/signal paths, state explicitly what the user needs to smoke-test.
6. **Update** `PLAN.md` (mark shipped, or revise open). Update `README.md` only if user-visible CLI/behavior changed.
7. **Log** to `.agent/journal.md` at end of session, then prune it to the **≤4 most-recent entries** (git holds the rest — see `.agent/README.md` § Pruning). Promote any generalizable lesson to `.agent/lessons.md`.
8. **Commit** at end-of-turn when closing cohesive work; defer if mid-iteration awaiting user input. Single focused commit in scoped-commit form (`Scope: summary`, scopedcommits.com); message optimized for LLM parsing; co-author line per `git log` style.

## Style conventions for `live_stt.py`

- Single file. Constants at the top. No frameworks, no DI, no config systems.
- Comments explain *why*, not *what*. Most existing comments document optimizations or bench-derived tunings (cache reuse, pre-pad rationale, latency lever). Preserve them.
- Avoid adding abstractions speculatively. The author actively prefers less code over more.
- No backwards-compat shims; this is 0.1.0 with one user.
- Function-local imports avoided; module-level imports preferred for readability.

## Build/test commands

```sh
uv sync                                          # install deps
git config --local core.hooksPath .githooks      # one-time: enable pre-commit hook
uv run live-stt                                  # run with defaults (k2v2 + Codex translation)
uv run live-stt --engine parakeet --no-translate # A/B engine, JA-only
uv run live-stt --list-devices                   # enumerate audio devices
uv run pytest                                    # run pure-function tests
uv run python -c "import live_stt"               # cheap import smoke-check
~/.local/share/lsp-node/node_modules/.bin/pyright --project . live_stt.py  # typecheck (CLI)
sh compaction.sh                                 # context-usage gauge (needs jq)
codex login --device-auth                        # user-interactive: enable EN leg
```

`sounddevice` dlopens system PortAudio at import; a container lacking it fails the import smoke-check (and pytest and the hook) with `OSError: PortAudio library not found`. Install once on Debian: `sudo apt-get install libportaudio2`. Check native libs by importing the binding, not `ldconfig -p` (L-010).

## Known caveats

- **Models are a runtime prerequisite** — `check_models()` preflights and points at `models/README.md`; weights are gitignored, fresh clones must download (~800 MB).
- **Silero onset clipping** (D-010): segments open 0.2–0.7 s late → every segment is re-sliced from the ring with `VAD_PRE_PAD_S=0.4` lead-in. Touching VAD/worker code must preserve this.
- **Synthetic continuous TTS is not a valid robustness test** (D-010): multi-sentence single-segment TTS clips collapse continuous decode; bench clips are per-sentence with real silences. Real mic + VAD is unaffected.
- **Codex auth is external state** — `~/.codex/auth.json` from user-interactive `codex login`. Plan reports `prolite`; Spark entitled regardless (D-011). Quota check: `account/rateLimits/read` on the app-server.
- **Tool-feature disables are THE latency lever** (D-011): re-enabling web_search/browser/apps features re-injects ~15 K tokens/turn and 400s at low effort. Spawn config in `_CODEX_CONFIG` must keep them off.
- **uv quirk**: `uv add sherpa-onnx` once skipped its declared dep `sherpa-onnx-core` (carries libonnxruntime); both are pinned in pyproject — keep both.
- Known STT quirks at default engine: ジェミニ→ゼミニ, paused 文→分 homophone (bench-observed, harmless; EN leg translates through them).
