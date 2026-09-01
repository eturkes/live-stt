#!/usr/bin/env python3
"""The project quality gate: one script owning the exact step set and file list.

Run it from the repository root:

    uv run python gate.py

Exit code 0 means every blocking step passed. This script exists because prose
cannot be executed: four commits reported "gate passed" while the pyright step
had silently been dropped from what was actually run, so the step set now lives
here and `tests/test_gate.py` locks it.

Two steps are shaped by measured traps, not preference:

- `ruff-format` checks touched `*.py` files only. Repo-wide is still red on ten
  files carrying pre-existing wrapping drift, so a unit leaves its own files
  clean and the repo converges file by file. The list is filtered to `*.py`
  because an explicitly-passed path with another extension is parsed as Python,
  which makes a `.json` argument exit 1 proposing Python layout.
- `aggregate-only` rebuilds `tests/model_baseline.json` from cached details and
  is blocking as of M11.3, which retired the whole-file pipeline fingerprint that
  had kept it permanently red. It needs the gitignored pinned corpus and detail
  cache, so it fails on a machine that has not acquired them.

Every step runs with `PYTHONPATH` cleared: an inherited entry can shadow the
installed OpenVINO wheel with a host build that cannot execute here, which
surfaces as a confusing `AxisSet` ImportError rather than a missing module.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass

PROD_FILES = ["live_stt.py", "replay.py", "cer.py", "streaming.py"]
# uvx is self-contained and version-pinned; the ~/.local pyright is dangling.
PYRIGHT = ["uvx", "pyright@1.1.410", "--project", "."]
# `ruff` is not on PATH; the module form works from any environment that has it.
RUFF = [sys.executable, "-m", "ruff"]
ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


@dataclass(frozen=True)
class Step:
    name: str
    blocking: bool
    argv: list[str]  # empty = nothing to check, which passes


def steps(files: list[str]) -> list[Step]:
    py = [f for f in files if f.endswith(".py")]
    return [
        Step("pytest", True, [sys.executable, "-m", "pytest", "-q"]),
        Step("ruff-check", True, [*RUFF, "check", "."]),
        Step("ruff-format", True, [*RUFF, "format", "--check", *py] if py else []),
        Step("pyright", True, [*PYRIGHT, *PROD_FILES]),
        Step("pyright-tests", True, [*PYRIGHT, "tests/"]),
        Step("import", True, [sys.executable, "-c", "import live_stt"]),
        Step("aggregate-only", True, [sys.executable, "tests/eval_models.py", "--aggregate-only"]),
    ]


def touched_py() -> list[str]:
    """Python files changed against HEAD, plus untracked ones.

    Renames report their new path, which is the one that must be formatted.
    """

    def git(*args: str) -> list[str]:
        out = subprocess.run(["git", *args], capture_output=True, text=True, env=ENV, check=False)
        return out.stdout.split() if out.returncode == 0 else []

    seen = dict.fromkeys(
        git("diff", "--name-only", "HEAD") + git("ls-files", "--others", "--exclude-standard")
    )
    return [f for f in seen if f.endswith(".py") and os.path.exists(f)]


def run(step: Step, verbose: bool) -> bool:
    if not step.argv:
        print(f"skip {step.name}: nothing to check", flush=True)
        return True
    done = subprocess.run(step.argv, capture_output=True, text=True, env=ENV, check=False)
    ok = done.returncode == 0
    label = "pass" if ok else "FAIL"
    if not step.blocking:
        label += " (non-blocking)"
    print(f"{label} {step.name}", flush=True)
    if verbose or not ok:
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
    return ok


def main() -> int:
    names = [s.name for s in steps([])]
    ap = argparse.ArgumentParser(description="Run the project quality gate.")
    ap.add_argument("--only", choices=names, help="Run one step instead of all of them.")
    ap.add_argument(
        "--files",
        nargs="*",
        help="Override the touched-file list that the format step checks.",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="Print output of every step.")
    args = ap.parse_args()

    files = args.files if args.files is not None else touched_py()
    selected = [s for s in steps(files) if args.only is None or s.name == args.only]
    failed = [s.name for s in selected if not run(s, args.verbose) and s.blocking]
    if failed:
        print(f"gate FAILED: {', '.join(failed)}", flush=True)
        return 1
    print(f"gate passed: {len(selected)} steps", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
