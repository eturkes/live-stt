---
paths:
  - "pyproject.toml"
  - ".githooks/pre-commit"
  - "uv.lock"
  - "models/README.md"
  - "tests/test_packaging.py"
  - "tests/test_gate.py"
  - "tests/test_project_config.py"
  - "tests/test_corpus.py"
  - "tests/short_corpus.json"
  - "tests/en_clips.json"
  - ".github/dependabot.yml"
---

# Packaging + dependency maintenance

**Two packaging traps, fixed, never reintroduce.** The wheel `only-include` must list **every**
production module (`live_stt.py` + `streaming.py`) — a missing one installs cleanly and fails at
import, which an editable install hides. Every sdist `include` pattern must be **root-anchored with a
leading `/`**: unanchored `README.md` matched at every depth and pulled third-party copies out of a
stale `.venv-npu` tree, making the member list depend on local state.

`openvino` is a hard dependency, not an extra — plain `uv sync` installs it. `sherpa-onnx` and
`sherpa-onnx-core` (which carries libonnxruntime) are both pinned deliberately; keep both, since uv once
skipped the declared sub-dependency. Read `uv.lock` for dependency work alone, through bounded queries.

`tests/test_packaging.py` follows the entry points' AST import closure (function-local backend imports
included), checks wheel + sdist inclusion and hard runtime declarations, and ships rejected twins:
missing `streaming.py`, unanchored sdist include, missing `openvino`. Each plant reddens the tracked
configuration lock; restore → green. Coverage = build configuration, not emitted archives: an
isolated `uv build --offline` cannot resolve uncached hatchling. No backend import, build download
or editable-install result stands in for package membership.

**D-007 — pre-commit via `.githooks/` + `core.hooksPath`, not the `pre-commit` framework.** The hook
is one line (`uv run pytest -q -n 6`, the gate's `PYTEST_WORKERS`, which `test_gate.py` holds
equal); the framework adds schema, per-hook venv cache and network bootstrap for nothing at this
scope. Each clone opts in once (`git config --local core.hooksPath .githooks`); `--no-verify`
bypasses in an emergency. Revisit if the hook grows past one or two commands. **Coverage limit: the
hook runs the suite, not `gate.py`**, so the skip-declaration check (`toolchain.md`) reaches a
demotion at the unit's closing gate rather than at the commit that introduced it. Routing the hook
through `gate.py --only pytest` would close that, at the cost of replacing pytest's live progress
with one line printed after the run — the human at the terminal is who the hook reports to, so the
suite stays.

`test_gate.py` executes a copied hook with a fake `uv` first on PATH: exactly one pytest argv,
`PYTEST_WORKERS`, success and a planted exit 29 must all survive dispatch. Shipped rejection twins
cover an earlier `exit 0`, swallowed failure, wrong worker count and a duplicate invocation. An
earlier exit planted in the tracked hook leaves the old last-line lock green but reddens execution;
restore → green. No real uv, commit, dependency sync or nested suite runs inside this check.

**L-018 — maintenance-pass recipe.** Inventory = every `uv.lock` package against its PyPI latest
(`uv tree --outdated --depth 1` hides transitives and any release `requires-python` excludes — it
showed 1 of 12 bumps) plus the pins outside the lock: `gate.py`'s pyright, `.python-version`, the
evaluators' `--with` pins, `models/README.md`'s export recipe, the codex CLI → CVE-scan with
`uv export --format requirements-txt --all-groups --no-emit-project >/tmp/reqs.txt && uvx pip-audit -r
/tmp/reqs.txt` (pip-audit scans its own environment, so feed it the export), the `--with` pins
fed the same way → history secret scan `gitleaks git --redact --no-banner .` (the repo is public
and `detect-secrets` reads the tree alone; gitleaks = its GitHub release binary,
checksum-verified), credited only beside a throwaway repo holding one planted fake key that must
report 1 → apply lock-only bumps
(`uv lock --upgrade-package NAME`) unless the older declared floor is unsupported or security-blocked,
in which case raise it deliberately → run the full gate → re-verify the codex leg, which pytest does not
cover, with a synthetic `CodexTranslator` turn (`start()` warm-up + `_translate`). Keep package versions
out of durable prose; they drift. A warm-up `init failed (TimeoutError)` can be backend latency
rather than the new codex: rerun it against the previous codex release before blaming the upgrade.

**Pins outside `uv.lock` re-qualify by rebuild, never by edit** — each fires on an upgrade:

- `tests/fetch_real_clips.py` `BUILD_DEPENDENCIES` holds numpy + pyarrow (+ soundfile, libsndfile)
  exact and refuses any other set ⇒ a numpy or pyarrow bump rebuilds the corpora fresh on a cache
  holding the five source payloads alone — a `git worktree` with the upgraded `uv.lock` copied in,
  since `uv run --with` resolves its overlay against the worktree's own lock — and credits only
  pinned index hashes plus every built file byte-identical. Then take the new manifests, the docs'
  `--with pyarrow==` pins, and re-pin `tests/test_en_corpus.py`'s `short_corpus.json` hash (user
  ruling: the stamp moved, the content did not). `test_corpus.py` compares both committed manifests'
  builder dependency dictionaries to `BUILD_DEPENDENCIES`, without loading decoder packages. A planted
  numpy `0.0.0` producer pin reddens both; clean twins pass. The shipped pin-only rejection twins
  fail if that consistency check is neutralized. This is producer/provenance equality, never a
  rebuild or a content-requalification substitute.
- `tests/fetch_fleurs_parallel.py` stamps the interpreter's patch version and
  `test_fleurs_parallel.py` holds it equal to the running one ⇒ every Python patch upgrade reddens
  the suite until that script reruns on a fresh cache and rebuilds byte-identically.
- `models/README.md`'s export recipe re-qualifies by DECODE, never bytes — its decoder `.bin`
  differs between runs of unchanged pins: export into scratch, then `tests/eval_asr_screen.py
  --arms fleurs,retention --device NPU --cache-dir <empty dir> --json F` must decode every
  hypothesis byte-identically to the shipped weights' `--json` (the private cache keeps the
  shared one's blobs from hiding the compiler).
  **One index per package, every pin exact.** torch + torchvision are direct wheel URLs off the
  PyTorch CPU index carrying its `#sha256=` (uv refuses a mismatch) and the venv's cp tag ⇒ a
  version or `--python` move re-picks both from `https://download.pytorch.org/whl/cpu/<name>/`.
  torchvision rides that index too because PyPI's is a CUDA build (`_C_stable.so` needs
  `libtorch_cuda.so`) and optimum-intel requires it. Every other name resolves from PyPI alone
  under `--exclude-newer`, which moves with any bump. Never add `--index`/`--extra-index-url`:
  any second index serves its own copies of PyPI names (`uv pip compile --emit-index-annotation`
  shows the source of each). `optimum-cli` runs under `env -u PYTHONPATH -u LD_LIBRARY_PATH`:
  both layers' login env puts a GenAI tarball's `openvino` there (`openvino-accel.md`), which
  shadows the pinned wheel — container: `GLIBC_2.43` ImportError; host: its 3.13 bindings load,
  a silent pin bypass (inferred, unrun).
- `tests/eval_cer.py` rewrites `tests/cer_baseline.json` on a pass ⇒ attribute a moved cell by
  running it on the old lock too (worktree at the base commit, its own venv).

`pip-audit` lives HERE and not in `gate.py` because it needs an advisory feed and the gate is
hermetic (`toolchain.md`). The offline half of the scan already runs on every commit — ruff `S`
inside `ruff-check`, plus the `secrets` step — so this recipe owns the dependency-CVE arm and the
history scan alone.
Dependabot (`.github/dependabot.yml`) raises the same bumps between passes; a grouped PR it opens is
this recipe's input, never a substitute for running it.

`tests/test_project_config.py` parses that YAML and git's tracked + untracked-unignored `.github/`
files: uv, project root, weekly schedule, one unrestricted wildcard group, no other GitHub file.
Shipped rejection twins alter each field, exclude a dependency or add a workflow. Plant monthly in
the tracked config → the structural lock reds while gate inventory stays green; restore → green.
The check grades local configuration, never Dependabot's remote operation or equality with law text.
