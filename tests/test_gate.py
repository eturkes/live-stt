"""Locks for the quality gate itself.

The gate is the one artifact whose failure is silent: a dropped step still
reports "gate passed". So the step inventory is asserted as data, and every
blocking step is proved able to fail the runner by seeding a defect of its own
class into a throwaway tree. Each seed tree is minimal on purpose -- the pytest
seed carries its own `testpaths`, so the step under test cannot re-collect this
suite and recurse.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gate import (
    PROD_FILES,
    SKIP_DECLARATION,
    Step,
    run,
    scan_files,
    steps,
    undeclared_demotions,
)

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "gate.py"

# name, blocking -- order included, because order is part of the contract.
INVENTORY = [
    ("pytest", True),
    ("ruff-check", True),
    ("ruff-format", True),
    ("pyright", True),
    ("pyright-tests", True),
    ("secrets", True),
    ("import", True),
]
BLOCKING = [name for name, blocking in INVENTORY if blocking]


def gate(tmp: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the real gate against a throwaway tree; it clears PYTHONPATH itself."""
    return subprocess.run(
        [sys.executable, str(GATE), *args], cwd=tmp, capture_output=True, text=True, check=False
    )


def test_inventory_is_the_contract():
    assert [(s.name, s.blocking) for s in steps()] == INVENTORY


def test_production_pyright_file_list():
    """streaming.py is a production module; leaving it off the list is how it went unchecked."""
    assert PROD_FILES == [
        "live_stt.py",
        "replay.py",
        "cer.py",
        "streaming.py",
        "session_report.py",
    ]
    argv = {s.name: s.argv for s in steps()}["pyright"]
    for name in PROD_FILES:
        assert name in argv


def test_format_step_traverses_the_repository():
    """Traversal is what skips .json/.md; an explicit path of either exits 1 as Python."""
    assert {s.name: s.argv for s in steps()}["ruff-format"][-3:] == ["format", "--check", "."]


def test_secret_scan_reaches_the_real_tree(monkeypatch):
    """The seeded control below cannot see the scan's scope shrink.

    Its throwaway tree holds one root-level `leak.py`, so a skip list that grew
    to prune subdirectories, dotdirs or the whole repo would still catch it while
    the real tree went unscanned -- and an unscanned clean tree prints the same
    green. Pin the scope against the tree the gate actually runs on.
    """
    monkeypatch.chdir(ROOT)
    found = set(scan_files())
    # A production file, a nested one, and a dotdir one: dotdirs carry the rules.
    assert {"./live_stt.py", "./tests/test_gate.py", "./.claude/rules/toolchain.md"} <= found
    assert "./tests/replay_goldens.json" not in found  # the declared evidence-JSON exclusion
    assert not [p for p in found if p.startswith("./models")]  # the declared bulk prune


def seed(tmp: Path, step: str) -> None:
    """Write a tree that fails exactly `step`."""
    if step == "pytest":
        (tmp / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
        (tmp / "tests").mkdir()
        (tmp / "tests" / "test_seed.py").write_text("def test_seed():\n    assert False\n")
    elif step == "ruff-check":
        (tmp / "seed.py").write_text("import os\n")  # F401
    elif step == "ruff-format":
        (tmp / "seed.py").write_text("x = {  'a' :1}\n")
    elif step in ("pyright", "pyright-tests"):
        (tmp / "pyproject.toml").write_text("[tool.pyright]\n")
        for name in PROD_FILES:
            (tmp / name).write_text("")
        (tmp / "tests").mkdir()
        # Seed both targets; --only selects which one is under test.
        (tmp / "live_stt.py").write_text('x: int = "s"\n')
        (tmp / "tests" / "seed.py").write_text('y: int = "s"\n')
    elif step == "secrets":
        # A credential shape the scan must catch with the hex plugin off.
        key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"  # pragma: allowlist secret
        (tmp / "leak.py").write_text(f'aws_secret_access_key = "{key}"\n')
    elif step == "import":
        (tmp / "live_stt.py").write_text('raise RuntimeError("seeded")\n')
    else:
        raise AssertionError(f"no seed for {step}")


@pytest.mark.parametrize("step", BLOCKING)
def test_blocking_step_failure_fails_the_gate(step, tmp_path):
    seed(tmp_path, step)
    done = gate(tmp_path, "--only", step)
    assert done.returncode != 0, done.stdout + done.stderr
    assert f"FAIL {step}" in done.stdout
    assert f"gate FAILED: {step}" in done.stdout


def _skipping_tree(tmp: Path, marker: str, addopts: str = "", subdir: str = "tests") -> None:
    """A throwaway suite whose one case skips for `marker`, carrying its own testpaths."""
    ini = '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    (tmp / "pyproject.toml").write_text(ini + (f'addopts = "{addopts}"\n' if addopts else ""))
    (tmp / subdir).mkdir(parents=True)
    (tmp / subdir / "test_seed.py").write_text(
        f'import pytest\n\n\n@pytest.mark.skip("{marker}")\ndef test_seed():\n    pass\n'
    )


def test_an_undeclared_skip_fails_the_pytest_step(tmp_path):
    """pytest exits 0 on a skipped case, so this defect is invisible to the step itself.

    A demotion earns a `.agent/deferred.md` row and the user's approval first;
    the gate is what refuses one that arrived without them.
    """
    _skipping_tree(tmp_path, "flaky")
    done = gate(tmp_path, "--only", "pytest")
    assert done.returncode != 0, done.stdout + done.stderr
    assert "undeclared demotion" in done.stdout
    assert "gate FAILED: pytest" in done.stdout


def test_a_declared_resource_gate_still_passes(tmp_path):
    """The control the seed above needs: the step rejects demotions, not every skip."""
    _skipping_tree(tmp_path, f"{SKIP_DECLARATION}seeded weights")
    done = gate(tmp_path, "--only", "pytest")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "pass pytest" in done.stdout


def test_configured_options_cannot_switch_the_skip_report_off(tmp_path):
    """`--runxfail` turns a passing xfail into an ordinary pass, emitting no XPASS line.

    The tally cannot see that one -- the case is counted as passed -- so `-o
    addopts=` is what holds here, and this is the seed that isolates it.
    """
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = "--runxfail"\n'
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_seed.py").write_text(
        'import pytest\n\n\n@pytest.mark.xfail(reason="flaky")\ndef test_seed():\n    pass\n'
    )
    done = gate(tmp_path, "--only", "pytest")
    assert done.returncode != 0, done.stdout + done.stderr
    assert "gate FAILED: pytest" in done.stdout


def test_a_suppressed_summary_is_caught_by_the_tally(tmp_path):
    """The backstop for whatever else can suppress the detail: a conftest, a plugin.

    `PYTEST_ADDOPTS` reaches pytest past `-o addopts=`, so it stands in here for
    that class -- the tally still reports the skip the summary no longer names.
    """
    _skipping_tree(tmp_path, f"{SKIP_DECLARATION}seeded weights")
    step = next(s for s in steps() if s.name == "pytest")
    done = subprocess.run(
        step.argv,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTEST_ADDOPTS": "--no-summary"},
        check=False,
    )
    assert done.returncode == 0, done.stdout  # a declared skip; only the report is gone
    for trailer in ("", "plugin trailer\n"):  # a trailer must not displace the tally
        complaint = undeclared_demotions(done.stdout + trailer)
        assert complaint is not None and "summary suppressed" in complaint, done.stdout


def test_an_inherited_pytest_addopts_cannot_blind_the_step(tmp_path):
    """`PYTEST_ADDOPTS` reaches pytest past `-o addopts=`, so the step env drops it.

    Green here means the summary survived: leave the variable in and the tally
    outruns what the summary named, which reds this declared skip instead.
    """
    _skipping_tree(tmp_path, f"{SKIP_DECLARATION}seeded weights")
    done = subprocess.run(
        [sys.executable, str(GATE), "--only", "pytest"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTEST_ADDOPTS": "--no-summary"},
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_a_path_holding_a_colon_still_reads_as_declared(tmp_path):
    """The location ends at its line number; splitting on the first ": " cut inside it."""
    _skipping_tree(tmp_path, f"{SKIP_DECLARATION}seeded weights", subdir="tests/odd: dir")
    done = gate(tmp_path, "--only", "pytest")
    assert done.returncode == 0, done.stdout + done.stderr


def test_non_blocking_runner_still_labels_and_tolerates_a_failure(capsys):
    # M11.3 made every step blocking. The runner keeps the non-blocking branch for
    # the next step that needs it, so the branch stays proved rather than dead.
    assert not [name for name, blocking in INVENTORY if not blocking]
    ok = run(Step("probe", False, [sys.executable, "-c", "raise SystemExit(1)"]), verbose=False)
    assert ok is False
    assert "FAIL (non-blocking) probe" in capsys.readouterr().out
