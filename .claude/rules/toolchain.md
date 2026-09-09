# Toolchain

**Gate = `uv run --no-sync python gate.py`** — 7 blocking steps, `--only NAME` runs one, `-v` prints
step output. The script owns the exact step set and file list and `tests/test_gate.py` locks the
inventory, so read the script instead of restating it: a gate that lives in prose gets silently
shortened, which is how four commits reported a passing gate while its pyright step stayed red. Every
step is fast and hermetic — no weights, no hardware, no network — and clears `PYTHONPATH` itself. A green run still reports **one
skip**, the whisper NPU replay golden, which needs the prelude below; anything more skipping means
absent weights or corpus.

**Every step ships the input that fires it, and the gate runs those inputs on itself.**
`tests/test_gate.py` seeds one defect of each step's own class into a throwaway tree and asserts
`gate FAILED: <step>` — pytest an `assert False`, ruff-check an `F401`, ruff-format bad spacing,
pyright and pyright-tests `x: int = "s"`, secrets an AWS-shaped key, import a module that raises. The
seeds ride the pytest step, so a green gate is simultaneously its own positive control: a step that
stopped being able to fail turns the gate red instead of quietly green. One scope hole that seeding
cannot see gets its own lock — the seeded tree holds a single root-level file, so an over-grown
`SCAN_SKIP_DIRS` or `tests/` exclusion would still catch it while the real tree went unscanned ⇒
`test_secret_scan_reaches_the_real_tree` pins the walk to production, a nested file and a dotdir
file. Feature locks prove themselves by neutralization instead (L-022), never by seeding.

**Pin the venv layer in every agent command.** `.venv` = container (agent dev + test), `.venv-host` =
host (live-mic runtime, lowest latency). A venv path-bakes its layer, so a bare `uv run` from the
wrong one rebuilds and clobbers the other. `.envrc` selects by path prefix but direnv acts in hooked
interactive shells alone ⇒ agent Bash carries whatever the launch env held. Always spell it:
`UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync …` for container work.

```sh
uv sync                                            # deps, openvino included (hard dep, no extra)
git config --local core.hooksPath .githooks        # one-time: enable the pre-commit hook (D-007)
codex login --device-auth                          # user-interactive: enable the EN leg
uv run --no-sync python gate.py                    # THE gate
uv run live-stt                                    # whisper on NPU + VAC + codex + saved transcript
uv run live-stt --list-devices                     # enumerate audio devices, needs no models
uv run pytest -q                                   # the fast suite alone
uv run python -c "import live_stt"                 # cheap import smoke-check
uv run python replay.py WAV [--engine E] [--json]  # replay a WAV through the live pipeline
uvx pyright@1.1.410 --project . live_stt.py replay.py cer.py streaming.py   # typecheck
```

**Security scanning is split by hermeticity.** Static analysis rides the existing `ruff-check` step
(ruff's `S` / flake8-bandit ruleset). The `secrets` step runs `detect-secrets` over a tree walk
`gate.py` owns — both offline, so the gate keeps its no-network contract. `pip-audit` needs an
advisory feed and stays in the L-018 recipe (`packaging-deps.md`); Dependabot
(`.github/dependabot.yml`, `package-ecosystem: uv`) watches between commits.

**The secret scan runs with `HexHighEntropyString` off and skips `tests/*.json`.** This repo's
evidence layer is MADE of SHA-256 — corpus fingerprints, content-addressed cache names, pinned
download digests (L-017) — so that plugin fires only false positives and the idiom grows with every
corpus; pragma-ing each site would be a guard escaped every time it fires (L-032). Every provider
pattern, the private-key detector and the keyword detector stay on, which is what a leaked codex or
GitHub credential trips. **Coverage limit: a bare hex credential is invisible, as is anything inside
`tests/*.json`.** A deliberate fake credential in a fixture carries `# pragma: allowlist secret`.

`ruff` is not on `PATH` — the module form (`python -m ruff`) works from any environment that has it.
`uvx` is self-contained; the `~/.local` pyright is dangling. Keep test files pyright-clean too (the
LSP and a `--project . tests/` run flag them); the house idiom for a fake→typed-attr assignment is
`# type: ignore[assignment]`, and unused-param hints (`time_info`, `_ja`) are tolerated.

**Whisper, NPU or GPU work needs BOTH prelude halves, every time:**
`source /var/home/eturkes/.local/app/intel-accel/env.sh` **and** `unset PYTHONPATH` (or
`env -u PYTHONPATH …`). Failure modes + the accelerator's shape: `openvino-accel.md`.

`sounddevice` dlopens system PortAudio on the live and device entry paths; without it they fail
`OSError: PortAudio library not found` → `sudo apt-get install libportaudio2` (Debian). Offline
import, pytest and evaluator paths avoid the binding on purpose. Confirm a native lib by importing
its binding, never through `ldconfig -p` — this container's `ld.so.cache` is empty while the libs
resolve fine (L-010).

**Teammate worktree gate recipe** (`.scratch/worktrees/<name>`): use the primary tree's `.venv`
read-only — `UV_PROJECT_ENVIRONMENT=<primary>/.venv uv run --no-sync …` from inside the worktree
rebuilds nothing and runs concurrently. pyright additionally needs `--venvpath <primary>`, since
`[tool.pyright] venvPath = "."` otherwise resolves to the worktree's absent `.venv` and reports
phantom import errors across production and tests. A bare worktree skips the model- and corpus-gated
tests, because gitignored `models/` + `spike/backends/cache/` are absent; `cp -al <primary>/spike
spike` + `cp -al <primary>/models/*/ <primary>/models/*.onnx models/` reproduces the primary gate
exactly (~5.7 GB linked in under a second). Use `cp -al`, never symlinks — git refuses pathspecs
beyond a symbolic link, so the acquisition-provenance tests fail `git check-ignore` with rc=128.
Hardlinks share inodes ⇒ treat a linked tree as read-only and re-acquire weights in the primary tree.
