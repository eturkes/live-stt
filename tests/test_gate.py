"""Locks for the quality gate itself.

The gate is the one artifact whose failure is silent: a dropped step still
reports "gate passed". So the step inventory is asserted as data, and every
blocking step is proved able to fail the runner by seeding a defect of its own
class into a throwaway tree. Each seed tree is minimal on purpose -- the pytest
seed carries its own `testpaths`, so the step under test cannot re-collect this
suite and recurse.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from gate import PROD_FILES, Step, run, steps

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "gate.py"

# name, blocking -- order included, because order is part of the contract.
INVENTORY = [
    ("pytest", True),
    ("ruff-check", True),
    ("ruff-format", True),
    ("pyright", True),
    ("pyright-tests", True),
    ("import", True),
]
BLOCKING = [name for name, blocking in INVENTORY if blocking]


def gate(tmp: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the real gate against a throwaway tree; it clears PYTHONPATH itself."""
    return subprocess.run(
        [sys.executable, str(GATE), *args], cwd=tmp, capture_output=True, text=True, check=False
    )


def test_inventory_is_the_contract():
    assert [(s.name, s.blocking) for s in steps([])] == INVENTORY


def test_production_pyright_file_list():
    """streaming.py is a production module; leaving it off the list is how it went unchecked."""
    assert PROD_FILES == ["live_stt.py", "replay.py", "cer.py", "streaming.py"]
    argv = {s.name: s.argv for s in steps([])}["pyright"]
    for name in PROD_FILES:
        assert name in argv


def test_format_step_takes_touched_python_only():
    """An explicitly-passed .json arg is parsed as Python and exits 1 proposing dict layout."""
    argv = {s.name: s.argv for s in steps(["a.py", "b.json", "c.md"])}["ruff-format"]
    assert argv[-1:] == ["a.py"]
    assert "b.json" not in argv and "c.md" not in argv


def test_format_step_is_empty_when_nothing_is_touched():
    assert {s.name: s.argv for s in steps([])}["ruff-format"] == []


def test_empty_argv_passes_as_skip(tmp_path):
    done = gate(tmp_path, "--only", "ruff-format", "--files")
    assert done.returncode == 0
    assert "skip ruff-format" in done.stdout


def seed(tmp: Path, step: str) -> list[str]:
    """Write a tree that fails exactly `step`; return extra gate arguments."""
    if step == "pytest":
        (tmp / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
        (tmp / "tests").mkdir()
        (tmp / "tests" / "test_seed.py").write_text("def test_seed():\n    assert False\n")
    elif step == "ruff-check":
        (tmp / "seed.py").write_text("import os\n")  # F401
    elif step == "ruff-format":
        (tmp / "seed.py").write_text("x = {  'a' :1}\n")
        return ["--files", "seed.py"]
    elif step in ("pyright", "pyright-tests"):
        (tmp / "pyproject.toml").write_text("[tool.pyright]\n")
        for name in PROD_FILES:
            (tmp / name).write_text("")
        (tmp / "tests").mkdir()
        # Seed both targets; --only selects which one is under test.
        (tmp / "live_stt.py").write_text('x: int = "s"\n')
        (tmp / "tests" / "seed.py").write_text('y: int = "s"\n')
    elif step == "import":
        (tmp / "live_stt.py").write_text('raise RuntimeError("seeded")\n')
    else:
        raise AssertionError(f"no seed for {step}")
    return []


@pytest.mark.parametrize("step", BLOCKING)
def test_blocking_step_failure_fails_the_gate(step, tmp_path):
    extra = seed(tmp_path, step)
    done = gate(tmp_path, "--only", step, *extra)
    assert done.returncode != 0, done.stdout + done.stderr
    assert f"FAIL {step}" in done.stdout
    assert f"gate FAILED: {step}" in done.stdout


def test_non_blocking_runner_still_labels_and_tolerates_a_failure(capsys):
    # M11.3 made every step blocking. The runner keeps the non-blocking branch for
    # the next step that needs it, so the branch stays proved rather than dead.
    assert not [name for name, blocking in INVENTORY if not blocking]
    ok = run(Step("probe", False, [sys.executable, "-c", "raise SystemExit(1)"]), verbose=False)
    assert ok is False
    assert "FAIL (non-blocking) probe" in capsys.readouterr().out
