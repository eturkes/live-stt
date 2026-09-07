---
paths:
  - "pyproject.toml"
  - ".githooks/pre-commit"
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

**D-007 — pre-commit via `.githooks/` + `core.hooksPath`, not the `pre-commit` framework.** The hook is
one line (`uv run pytest -q`); the framework adds schema, per-hook venv cache and network bootstrap for
nothing at this scope. Each clone opts in once
(`git config --local core.hooksPath .githooks`); `--no-verify` bypasses in an emergency. Revisit if the
hook grows past one or two commands.

**L-018 — maintenance-pass recipe.** Inventory with `uv tree --outdated --depth 1` → CVE-scan with
`uv export --format requirements-txt --all-groups --no-emit-project >/tmp/reqs.txt && uvx pip-audit -r
/tmp/reqs.txt` (pip-audit scans its own environment, so feed it the export) → apply lock-only bumps
(`uv lock --upgrade-package NAME`) unless the older declared floor is unsupported or security-blocked,
in which case raise it deliberately → run the full gate → re-verify the codex leg, which pytest does not
cover, with a synthetic `CodexTranslator` turn (`start()` warm-up + `_translate`). Keep package versions
out of durable prose; they drift.
