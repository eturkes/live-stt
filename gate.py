#!/usr/bin/env python3
"""The project quality gate: one script owning the exact step set and file list.

Run it from the repository root:

    uv run python gate.py

Exit code 0 means every blocking step passed. This script exists because prose
cannot be executed: four commits reported "gate passed" while the pyright step
had silently been dropped from what was actually run, so the step set now lives
here and `tests/test_gate.py` locks it.

`ruff-format` checks the repository, not a touched-file list: passing paths
explicitly is what made the step fragile, because a path with a non-Python
extension is parsed as Python and a `.json` argument exits 1 proposing Python
layout. Traversing a directory skips those extensions instead.

Every step here is fast and hermetic. Model scoring is deliberately NOT a gate
step: `tests/eval_cer.py` and `tests/eval_long_form.py` need gitignored weights
and minutes of compute, so they run on demand when a decode change raises an
accuracy question, not on every commit.

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
    argv: list[str]


def steps() -> list[Step]:
    return [
        Step("pytest", True, [sys.executable, "-m", "pytest", "-q"]),
        Step("ruff-check", True, [*RUFF, "check", "."]),
        Step("ruff-format", True, [*RUFF, "format", "--check", "."]),
        Step("pyright", True, [*PYRIGHT, *PROD_FILES]),
        Step("pyright-tests", True, [*PYRIGHT, "tests/"]),
        Step("import", True, [sys.executable, "-c", "import live_stt"]),
    ]


def run(step: Step, verbose: bool) -> bool:
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
    names = [s.name for s in steps()]
    ap = argparse.ArgumentParser(description="Run the project quality gate.")
    ap.add_argument("--only", choices=names, help="Run one step instead of all of them.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print output of every step.")
    args = ap.parse_args()

    selected = [s for s in steps() if args.only is None or s.name == args.only]
    failed = [s.name for s in selected if not run(s, args.verbose) and s.blocking]
    if failed:
        print(f"gate FAILED: {', '.join(failed)}", flush=True)
        return 1
    print(f"gate passed: {len(selected)} steps", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
