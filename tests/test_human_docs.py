"""L-021's decidable half: the human-facing surfaces carry no en or em dash.

`human-docs.md` scopes the register to README.md, models/README.md and the CLI strings;
the dash rule is the one part of it a tool can decide, so it lives here rather than in a
manual grep. The CLI half reads the rendered `--help`, which is what a user sees.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import live_stt

ROOT = Path(live_stt.__file__).resolve().parent
DASH = re.compile("[–—]")


def _dashed(name: str, text: str) -> list[str]:
    return [
        f"{name}:{n}: {line}" for n, line in enumerate(text.splitlines(), 1) if DASH.search(line)
    ]


@pytest.mark.parametrize("doc", ["README.md", "models/README.md"])
def test_the_readmes_carry_no_en_or_em_dash(doc: str) -> None:
    assert _dashed(doc, (ROOT / doc).read_text(encoding="utf-8")) == []


def test_the_cli_help_carries_no_en_or_em_dash(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["live-stt", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        live_stt.main()
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--playback" in help_text  # the rendered help, not an empty capture
    assert _dashed("--help", help_text) == []


def _cli_literals() -> list[tuple[int, str]]:
    """String parts of what live_stt.py prints to the user: `print(...)`, `parser.error(...)` and
    the preflight messages `check_models` / `check_device` return for `_run_cli` to print."""
    import ast  # noqa: PLC0415 -- this lock alone parses source

    tree = ast.parse((ROOT / "live_stt.py").read_text(encoding="utf-8"))
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (
            (isinstance(node.func, ast.Name) and node.func.id == "print")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "error")
        ):
            roots += node.args
        elif isinstance(node, ast.FunctionDef) and node.name in ("check_models", "check_device"):
            roots += [r.value for r in ast.walk(node) if isinstance(r, ast.Return) and r.value]
    return [
        (n.lineno, n.value)
        for root in roots
        for n in ast.walk(root)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def test_the_cli_strings_carry_no_en_or_em_dash() -> None:
    literals = _cli_literals()
    assert any("--playback needs PipeWire" in text for _, text in literals)  # the scan reaches them
    assert [f"live_stt.py:{n}: {text}" for n, text in literals if DASH.search(text)] == []


def test_the_readme_constant_table_names_live_constants_at_their_values() -> None:
    """Each `| `NAME` | value |` row of README's table: NAME exists in live_stt.py at that value."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([A-Z][A-Z0-9_]+)` \| ([^|]+?) \|", readme, re.MULTILINE)
    assert len(rows) >= 20  # the table, not an empty match
    drift = []
    for name, shown in rows:
        if not hasattr(live_stt, name):
            drift.append(f"{name}: absent from live_stt.py")
            continue
        value = getattr(live_stt, name)
        text = shown.strip().strip("`").removesuffix(" s")
        same = text == value if isinstance(value, str) else float(text) == float(value)
        if not same:
            drift.append(f"{name}: README {shown.strip()!r}, code {value!r}")
    assert drift == []
