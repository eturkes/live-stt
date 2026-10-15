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
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import takewhile

PROD_FILES = ["live_stt.py", "replay.py", "cer.py", "streaming.py", "session_report.py"]
# uvx is self-contained and version-pinned, so pyright needs no install of its own.
PYRIGHT = ["uvx", "pyright@1.1.414", "--project", "."]
# `ruff` is not on PATH; the module form works from any environment that has it.
RUFF = [sys.executable, "-m", "ruff"]
# detect-secrets reads a SHA-256 as a secret, and this repo's evidence layer is
# MADE of them -- corpus fingerprints, content-addressed cache names, pinned
# download digests (L-017) -- so the hex plugin is off and the committed evidence
# JSON is out of scope. Pragma-ing each site instead would be a guard escaped
# every time it fires (L-032), and the idiom grows with every corpus. Every
# provider pattern, the private-key detector and the keyword detector stay on,
# which is what a leaked codex or GitHub credential trips. Coverage limit: a bare
# hex credential is invisible, and so is anything inside `tests/*.json`.
SECRETS = [
    sys.executable,
    "-m",
    "detect_secrets.pre_commit_hook",
    "--disable-plugin",
    "HexHighEntropyString",
]
# Bulk the fallback walk must skip (`scan_files`): weights, bench corpora, venvs and
# caches, none of them tracked. Pruning by name keeps that walk off several GB.
SCAN_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".venv-host",
        ".venv-npu",
        ".direnv",
        ".ruff_cache",
        ".pytest_cache",
        ".scratch",
        "__pycache__",
        "models",
        "spike",
        "transcripts",
    }
)
# `PYTEST_ADDOPTS` survives an `-o addopts=` override, and `--no-summary` or
# `--runxfail` from either source blinds the skip check below.
ENV = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTEST_ADDOPTS")}
# A case that skips or xfails is a demotion unless a resource is absent, and a
# demotion needs a queue row and the user's approval first
# (`assurance-posture.md`). So the pytest step DECLARES its skips: a resource
# gate opens its reason with this prefix and keeps its case live, while a bare
# `pytest.mark.skip` or any xfail fails the step. Prose carried the invariant
# before -- "a green run reports exactly one skip" -- and prose cannot fail, so
# a new skip rode a green gate.
SKIP_DECLARATION = "absent: "
# `SKIPPED [n] path:lineno: reason`, anchored on the line number: a path can hold
# ": " itself, and splitting on the first one then cuts inside the location.
_SKIPPED = re.compile(r"^SKIPPED \[(\d+)\] .+?:\d+: (.*)$")
# The closing tally, which survives everything that suppresses the detail above.
_TALLY = re.compile(r"\b(\d+) (?:skipped|xfailed|xpassed)\b")
# Every gate run blocks the critical path, so the suite names where its time goes.
SLOWEST = 5
# pytest-xdist workers. Serial, the suite took ~39 s on this 8-core host; 3/4/6 workers read
# 23-30 / 23-27 / 22-23 s interleaved under load 8, and 8 workers stretched the LID census lock
# (4 ONNX threads of its own, the suite's floor) from 5 s to 15 s.
PYTEST_WORKERS = 6


def scan_files() -> list[str]:
    """Files the secret scan reads: every file a commit can carry, minus evidence JSON.

    At a repository's top level that is git's own list, tracked + untracked-unignored.
    Walking the disk instead read gitignored live-session logs (16.5 of 18.5 MB scanned,
    most of the step's 36 s) and pruned all of `models/`, tracked `models/README.md`
    included. Anywhere else -- the gate's seed trees, which may sit under an ignored
    directory -- nothing is ignored, so the walk minus bulk stands.

    The hook takes explicit filenames and its own `--exclude-files` does not
    filter them, so the exclusion has to happen here.
    """
    prefix = subprocess.run(
        ["git", "rev-parse", "--show-prefix"], capture_output=True, text=True, check=False
    )
    if prefix.returncode == 0 and not prefix.stdout.strip():
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True,
        )
        # A tracked file deleted from the work tree is still listed; it carries nothing.
        paths = sorted(
            os.path.join(".", p) for p in listed.stdout.split("\0") if p and os.path.isfile(p)
        )
    else:
        paths = []
        for root, dirs, names in os.walk("."):
            dirs[:] = sorted(d for d in dirs if d not in SCAN_SKIP_DIRS)
            paths += (os.path.join(root, name) for name in sorted(names))
    return [
        p
        for p in paths
        if not (p.endswith(".json") and os.path.basename(os.path.dirname(p)) == "tests")
    ]


def undeclared_demotions(out: str) -> str | None:
    """Skips and xfails in `-rsxX` output that no resource gate declared.

    The step's own exit code cannot see these: pytest exits 0 on a skipped case.
    Cross-checking the tally against what the summary named is what makes the
    reading trustworthy -- suppressing the summary empties this input, and an
    unread input and a clean tree otherwise emit the same green.
    """
    lines = out.splitlines()
    bad = []
    named = 0
    for line in lines:
        if line.startswith(("XFAIL ", "XPASS ")):
            bad.append(line)
            named += 1
        elif line.startswith("SKIPPED "):
            hit = _SKIPPED.match(line)
            named += int(hit.group(1)) if hit else 1
            if hit is None or not hit.group(2).startswith(SKIP_DECLARATION):
                bad.append(line)
    # The LAST tallying line, not the last line: a plugin trailer printed after it
    # would otherwise displace the tally and take this backstop with it.
    tail = next((s for s in reversed(lines) if _TALLY.search(s)), "")
    tallied = sum(int(n) for n in _TALLY.findall(tail))
    if tallied > named:
        bad.append(f"{tallied} demoted case(s) tallied, {named} named: summary suppressed")
    return "undeclared demotion (assurance-posture.md):\n" + "\n".join(bad) if bad else None


def slowest_tests(out: str) -> str | None:
    """pytest's `--durations` block, which a green run would otherwise swallow."""
    lines = out.splitlines()
    head = next((i for i, s in enumerate(lines) if "slowest" in s and "durations" in s), None)
    if head is None:
        return None
    block = list(takewhile(lambda s: s.strip() and not s.startswith(("=", "(")), lines[head + 1 :]))
    return "\n".join(f"  {s}" for s in block) or None


@dataclass(frozen=True)
class Step:
    name: str
    blocking: bool
    argv: list[str]
    # Reads the step's stdout for a defect its exit code passes over.
    check: Callable[[str], str | None] | None = None
    # Reads the step's stdout for what a green run prints beside its wall-clock.
    report: Callable[[str], str | None] | None = None


def steps() -> list[Step]:
    return [
        # `-o addopts=` drops any project-configured options, which is what keeps
        # the step's own `-rsxX` reporting from being turned off underneath it.
        Step(
            "pytest",
            True,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-rsxX",
                "-o",
                "addopts=",
                f"--durations={SLOWEST}",
                "-n",
                str(PYTEST_WORKERS),
            ],
            undeclared_demotions,
            slowest_tests,
        ),
        Step("ruff-check", True, [*RUFF, "check", "."]),
        Step("ruff-format", True, [*RUFF, "format", "--check", "."]),
        Step("pyright", True, [*PYRIGHT, *PROD_FILES]),
        Step("pyright-tests", True, [*PYRIGHT, "tests/"]),
        Step("secrets", True, [*SECRETS, *scan_files()]),
        Step("import", True, [sys.executable, "-c", "import live_stt"]),
        # A byte copy of the template's shape check, replaced whole at each refresh;
        # `sh` keeps the step independent of the file's exec bit.
        Step("spec-lint", True, ["sh", ".agent/spec-lint"]),
    ]


def run(step: Step, verbose: bool) -> bool:
    start = time.perf_counter()
    done = subprocess.run(step.argv, capture_output=True, text=True, env=ENV, check=False)
    elapsed = time.perf_counter() - start
    complaint = step.check(done.stdout) if step.check and done.returncode == 0 else None
    ok = done.returncode == 0 and complaint is None
    label = "pass" if ok else "FAIL"
    if not step.blocking:
        label += " (non-blocking)"
    print(f"{label} {step.name} {elapsed:.1f}s", flush=True)
    if complaint:
        print(complaint, flush=True)
    if verbose or not ok:
        sys.stdout.write(done.stdout)
        sys.stderr.write(done.stderr)
    elif step.report and (report := step.report(done.stdout)):
        print(report, flush=True)
    return ok


def main() -> int:
    names = [s.name for s in steps()]
    ap = argparse.ArgumentParser(description="Run the project quality gate.")
    ap.add_argument("--only", choices=names, help="Run one step instead of all of them.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print output of every step.")
    args = ap.parse_args()

    selected = [s for s in steps() if args.only is None or s.name == args.only]
    start = time.perf_counter()
    failed = [s.name for s in selected if not run(s, args.verbose) and s.blocking]
    total = f"{time.perf_counter() - start:.1f}s"
    if failed:
        print(f"gate FAILED: {', '.join(failed)} ({total})", flush=True)
        return 1
    print(f"gate passed: {len(selected)} steps ({total})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
