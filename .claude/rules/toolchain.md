# Toolchain

**Gate = `uv run --no-sync python gate.py`** — 8 blocking steps, `--only NAME` runs one, `-v` prints
step output. The script owns the exact step set and file list and `tests/test_gate.py` locks the
inventory, so read the script instead of restating it: a gate that lives in prose gets silently
shortened, which is how four commits reported a passing gate while its pyright step stayed red. Every
step is fast and hermetic — no weights, no hardware, no network — and clears `PYTHONPATH` itself. A
green run still reports **one skip**, the whisper NPU replay golden, which needs the prelude below;
anything more skipping means absent weights or corpus. Each check's mechanics, coverage limits and
firing record live in `gate.md`, which loads with the check's files. Every run prints each step's
wall-clock and the suite's five slowest tests, since each run blocks the critical path.

**Fast gate (iteration) = the steps + tests the change reaches, launched as parallel calls:**
`uv run --no-sync python -m pytest -q --ff <test files the change reaches>`
(`test_law_consistency.py` for law files, `test_gate.py` for `gate.py`) beside
`uv run --no-sync python gate.py --only <step>` for each other step it reaches — `ruff-check` +
`ruff-format` for Python or Markdown, `pyright` / `pyright-tests` for production / test code,
`secrets` for new files, `spec-lint` for `.agent/spec.md`. The full gate runs before every commit.

**Every skip declares itself, because pytest exits 0 on a skipped case.** A resource gate opens its
reason with `absent: ` and keeps its case live, while a bare `pytest.mark.skip`, an `xfail` or an
`xpass` fails the step. That is what a demotion is, and a demotion earns a `.agent/deferred.md` row
and the user's approval first (`assurance-posture.md`). The gate also names its skips in `-v` output,
so `green` reports them rather than leaving the count to prose — which is where the invariant sat
while any new skip rode green.

**Every step ships the input that fires it, and the gate runs those inputs on itself** (`gate.md`).
Feature locks prove themselves by neutralization instead (L-022), never by seeding.

**`ruff-format` governs Markdown too, and `CLAUDE.md` is carved out of it.** ruff 0.16 formats
python blocks inside `.md`, so a bare version bump moved the step from 39 files to 62 — every repo
`.md`, `CLAUDE.md` included, a template copy the repo never edits (`orientation.md`).
`[tool.ruff] extend-exclude` in `pyproject.toml` carves out `CLAUDE.md` + `CLAUDE.local.md`;
repo-owned prose keeps the coverage. **A `python`-tagged block in repo prose is now gate-governed**
— write it formatted, or tag it `sh`.

`tests/test_law_consistency.py` rides the pytest step and locks the five law invariants a tool can
decide. Three are the deferral queue's: `.agent/spec.md`'s `Tasks` pairs every `.agent/deferred.md`
row with one open `- [ ] **N** Title` row, rank and title verbatim, any other checklist item being a
rankless `- [x] <sha> Title`, the one `- [ ] RESUME: …` note that `pause.md` | `pause-now.md` write
and `resume.md` deletes, or the one `- [ ] ROUTE: …` row that `route.md` writes and deletes before its
last commit; and no scanned law file names a row by `rank N` at all.

The third is the same hole in the other direction: **law may name a queue row, and a row dies at its
close while the pointer at it does not.** Every reference in the scanned files that names
`.agent/deferred.md` and then italicises a row title must resolve to a live row, matched on
whitespace-squashed text since prose wrapping splits one across a line. `_QUEUE_LINK` in the test
owns the exact form, and prose must not restate it — this paragraph would match its own example. A
rank retargets onto the wrong unit; a title survives its unit and reads as live law describing
queued work that no longer exists.

The fourth locks the ruling tags (`orientation.md`), which a refresh re-reads against the template
before acting on its words. `_RULING_TAG` owns the form, and prose must not restate the tag's opener
— a scanned paragraph would parse as a broken tag. Three things hold in every `.claude/rules/`
file, on whitespace-squashed text since wrapping splits a quote across lines: each opener parses in
full, so a near-miss (curly quotes, a missing arrow, an unknown verdict) fails by name instead of
escaping; each bold verdict sits inside a parsed tag, so a ruling written without its key fails; and
every quoted anchor runs at least 12 characters and occurs in `CLAUDE.md` or a `.claude/agents/`
role — the repo's own template copies, so the lock runs in a fresh clone. An unquoted prose key
cannot be rechecked at all, and a reworded or deleted clause leaves its tag overriding nothing while
still reading as live law — both silent in prose. **It does not decide that an anchor is still its
own tag's clause**: membership is literal, so a phrase surviving elsewhere in the template reads as
live. The 12-character floor is what keeps a key off a word like `review`, which occurs everywhere
and identifies nothing; the rest is a reader's call.

The fifth locks what law quotes out of `.agent/spec.md`'s `Intent`: **a unit renames the thing
quoted, rewrites its own prose with the new name, and leaves `Intent` behind** —
leaving a citation that attributes wording `Intent` never carried while still reading as the ask.
Every backticked fragment after a backticked `Intent` in one whitespace-squashed sentence must occur
in the `## Intent` section; text before the citation names the file rather than quoting it, a bare
`Intent` listing the five section names cites nothing, and the heading rides the haystack so a
citation may name the section. The queue is scanned here and nowhere else in this file, its rows
being acceptance contracts that cite `Intent` the same way, while the rank and title locks read it as
their source of truth.

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
uvx pyright@1.1.414 --project . live_stt.py replay.py cer.py streaming.py   # typecheck
```

Template ruling: "gates + CI from its first commit" + "security scanning + update automation in
gate + CI" → **ADAPTED** (user rulings) — CI is RETIRED: the `.githooks/` pre-commit hook (D-007)
runs the suite on every commit of a single-user repo, so CI adds ≈ zero catch, and the one novel
hazard here, moved-dir venv shebangs, cannot occur in CI's fresh clone (L-019); `.agent/spec.md`
rules CI mirroring the hook out of scope. Update automation = `.github/dependabot.yml`
(`package-ecosystem: uv`, weekly, grouped), the only file under `.github/`, since an advisory feed
between commits is the one thing a local hook cannot watch. Security scanning splits as below.

**Security scanning is split by hermeticity.** Static analysis rides the existing `ruff-check` step
(ruff's `S` / flake8-bandit ruleset). The `secrets` step runs `detect-secrets` over every file a
commit can carry, git's tracked + untracked-unignored list (`scan_files`) — both offline, so the
gate keeps its no-network contract. `pip-audit` needs an advisory feed and stays in the L-018 recipe
(`packaging-deps.md`); Dependabot (`.github/dependabot.yml`, `package-ecosystem: uv`) watches
between commits.

**The secret scan runs with `HexHighEntropyString` off and skips `tests/*.json`.** This repo's
evidence layer is MADE of SHA-256 — corpus fingerprints, content-addressed cache names, pinned
download digests (L-017) — so that plugin fires only false positives and the idiom grows with every
corpus; pragma-ing each site would be a guard escaped every time it fires (L-032). Every provider
pattern, the private-key detector and the keyword detector stay on, which is what a leaked codex or
GitHub credential trips. **Coverage limit: a bare hex credential is invisible, as is anything inside
`tests/*.json`.** A deliberate fake credential in a fixture carries `# pragma: allowlist secret`.

`ruff` is not on `PATH` — the module form (`python -m ruff`) works from any environment that has it.
`uvx` is self-contained and version-pinned, so pyright needs no install of its own. Keep test files
pyright-clean too (the LSP and a `--project . tests/` run flag them); the house idiom for a
fake→typed-attr assignment is `# type: ignore[assignment]`, and unused-param hints (`time_info`,
`_ja`) are tolerated.

**Whisper, NPU or GPU work needs BOTH prelude halves, every time, in order:** source the
accelerator's `env.sh` (its path = machine-specific, in `CLAUDE.local.md`) **then** `source .envrc`
from the repo root (unsets `PYTHONPATH`, strips tarball libs off `LD_LIBRARY_PATH`).
`uv run live-stt` alone cleans its own session child (`_session_env`), so there only `env.sh`
matters, container-side. Failure
modes + the accelerator's shape: `openvino-accel.md`.

**Never run two whisper processes against one cache while either is compiling.** Two processes
cold-compiling the SAME `OPENVINO_CACHE_DIR` key concurrently write a blob that SIGSEGVs on every
later load, with no Python traceback, so the damage outlives the run that caused it. Concurrency also
inflates decode itself — two pipelines on this one NPU measured 0.564 → 1.218 s per update before
segfaulting. The cache is fully regenerable ⇒ repair is `rm -rf models/openvino/cache` and the next
run recompiles (~95-126 s). This matters here because **an agent Bash call can invoke one command
TWICE CONCURRENTLY**: shell `flock -n` does not catch it and in-process `fcntl.flock` does, so a
harness that must hold the NPU takes its lock inside the worker process and re-checks for completed
work after acquiring it. One measured source of that second run: Python 3.14's `forkserver` default
re-imports an unguarded script as `__mp_main__` in every `multiprocessing` child, and
`openvino-telemetry` spawned one per compile, so a probe with no `__main__` guard ran top to bottom
under two PIDs. Telemetry is opted out on both homes (`CLAUDE.local.md`); guard every script that
compiles regardless.

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

**A teammate commits inside its worktree with `git -c core.hooksPath=/dev/null commit`.** The
pre-commit hook (D-007) runs a bare `uv run`, which re-points the shared `.venv`'s editable
`live-stt` install at the worktree and breaks `uv run live-stt` machine-wide until a primary-tree
`UV_PROJECT_ENVIRONMENT=.venv uv sync` restores it. The teammate's own gate run covers what the hook
would have checked, and `PYTHONPATH`-free `import live_stt` from a directory outside the repo prints
the live target, so read `__file__` there rather than trusting the absence of a warning. Harvesting a
worktree that GENERATED gitignored payloads takes two more steps: `cp -al` each payload directory
into the primary cache, then rerun the producing script in the primary tree and require
byte-identical manifests (`sha256sum -c`), which is what credits the artifact.
