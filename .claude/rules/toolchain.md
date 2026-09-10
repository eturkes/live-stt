# Toolchain

**Gate = `uv run --no-sync python gate.py`** — 7 blocking steps, `--only NAME` runs one, `-v` prints
step output. The script owns the exact step set and file list and `tests/test_gate.py` locks the
inventory, so read the script instead of restating it: a gate that lives in prose gets silently
shortened, which is how four commits reported a passing gate while its pyright step stayed red. Every
step is fast and hermetic — no weights, no hardware, no network — and clears `PYTHONPATH` itself. A
green run still reports **one skip**, the whisper NPU replay golden, which needs the prelude below;
anything more skipping means absent weights or corpus.

**Every skip declares itself, because pytest exits 0 on a skipped case.** The step runs `-rsxX` and
`gate.py`'s `undeclared_demotions` reads the summary: a resource gate opens its reason with
`absent: ` and keeps its case live, while a bare `pytest.mark.skip`, an `xfail` or an `xpass` fails
the step. That is what a demotion is, and a demotion earns a `.agent/deferred.md` row and the user's
approval first (`assurance-posture.md`). The gate also names its skips in `-v` output, so `green`
reports them rather than leaving the count to prose — which is where the invariant sat while any new
skip rode green. **The step owns its effective options, because `--no-summary` and `--runxfail` each
empty that input in silence:** `-o addopts=` drops project-configured ones, `PYTEST_ADDOPTS` is
stripped from the step env, and the closing tally is cross-checked against what the summary named, so
whatever else suppresses the detail turns the step red instead of quiet — the tally is read off the
LAST tallying line, so a plugin trailer printed after it cannot displace it. **Coverage limits, all
three loud or out of reach rather than silently green:** the checker reads emitted
`SKIPPED`/`XFAIL`/`XPASS` lines, so a DELETED case and a demoted tier leave no trace and stay under
the approval law alone; in-tree pytest configuration outranks it, since a `conftest.py` setting
`config.option.runxfail` reconfigures pytest itself and this repo ships no conftest at all; and a
directory named with a `:<digits>: ` inside it false-REDS a declared skip, pytest's `location: reason`
line being genuinely ambiguous there.

**Every step ships the input that fires it, and the gate runs those inputs on itself.**
`tests/test_gate.py` seeds one defect of each step's own class into a throwaway tree and asserts
`gate FAILED: <step>` — pytest an `assert False`, ruff-check an `F401`, ruff-format bad spacing,
pyright and pyright-tests `x: int = "s"`, secrets an AWS-shaped key, import a module that raises. The
seeds ride the pytest step, so a green gate is simultaneously its own positive control: a step that
stopped being able to fail turns the gate red instead of quietly green. The skip rule seeds four more
pytest trees: `@pytest.mark.skip("flaky")`, the same tree under `addopts = "--no-summary --runxfail"`,
a `PYTEST_ADDOPTS`-suppressed summary the tally still catches, and a declared-`absent:` twin as the
positive control — without that twin, a check that flagged EVERY skip would pass the seed. A path
holding `": "` is locked too, since the location ends at its line number. One scope hole that seeding
cannot see gets its own lock — the seeded tree holds a single root-level file, so an over-grown
`SCAN_SKIP_DIRS` or `tests/` exclusion would still catch it while the real tree went unscanned ⇒
`test_secret_scan_reaches_the_real_tree` pins the walk to production, a nested file and a dotdir
file. Feature locks prove themselves by neutralization instead (L-022), never by seeding.

`tests/test_law_consistency.py` rides the pytest step and locks the three law invariants a tool can
decide. Two are the deferral queue's: `.agent/spec.md`'s spine pairs every `.agent/deferred.md` row
with its rank, title verbatim, and no scanned law file names a row by `rank N` at all. Pairing is
checked ORDERED because independent rank/title membership passes a swap of two titles; that swap and
the `at rank N` evasion form are both proven red by mutation, restoring from a `cp` snapshot rather
than `git checkout` (L-022).

The third locks `upstream-sync.md`'s override table, which a refresh re-reads against the template.
Three things hold: the table opens on its exact header and every row sits in that one contiguous
block, so a row below the prose or in a second table fails instead of going unchecked; every key
quotes a phrase of at least 12 characters; and every such anchor occurs verbatim in `CLAUDE.md` or in
the global `CLAUDE.md`. Any pipe outside the block counts as that escape, because GFM builds a table
out of `a|b` with no spaces too ⇒ a pipe wanted in that file's prose fails loudly rather than quietly
reopening the hole. An unquoted prose key cannot be rechecked at all, and a reworded or deleted
clause leaves its row overriding nothing while still reading as live law — both silent in prose.
**It does not decide that an anchor is still its own row's clause**: membership is literal, so a
phrase surviving elsewhere in the template reads as live. The 12-character floor is what keeps a key
off a word like `rev`, which occurs everywhere and identifies nothing; the rest is a reader's call.

None of the three locks needs a seeded fixture — the tree itself was the firing input, red on two
renamed spine titles, three rank references (one already retargeted onto the wrong unit) and one
prose-keyed row whose clause upstream had long dropped. Mutation-proven red for the third: a dead
anchor, a generic short anchor, curly quotes, a changed header cell, a parse that yields no rows, and
a row moved below the prose, into a pipe-less table or into a second table — each escape in both the
spaced and the no-space pipe form. Every mutant is restored from a `cp` snapshot (L-022). **Coverage limit: the global half of the haystack needs `~/.claude/CLAUDE.md`** — present in
every session here, so the check reports absence as a failure naming the anchor rather than skipping.

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
