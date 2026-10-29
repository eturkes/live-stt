"""Packaging configuration locks; the editable install cannot expose missing wheel modules."""

from __future__ import annotations

import ast
import copy
import fnmatch
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _config():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _import_closure(root, entrypoints):
    pending = [target.split(":", 1)[0] for target in entrypoints.values()]
    modules, external = set(), set()
    while pending:
        module = pending.pop()
        path = Path(module.replace(".", "/") + ".py")
        assert (root / path).is_file(), f"entry-point import missing: {module}"
        if path.as_posix() in modules:
            continue
        modules.add(path.as_posix())
        for node in ast.walk(ast.parse((root / path).read_text())):
            imports = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module]
                if isinstance(node, ast.ImportFrom) and node.module
                else []
            )
            for name in imports:
                top = name.split(".", 1)[0]
                if (root / f"{top}.py").is_file():
                    pending.append(top)
                elif top not in sys.stdlib_module_names:
                    external.add(top.replace("_", "-").lower())
    return modules, external


def _assert_packaging(config, root=ROOT):
    project = config["project"]
    modules, external = _import_closure(root, project["scripts"])
    targets = config["tool"]["hatch"]["build"]["targets"]
    wheel = targets["wheel"]["only-include"]
    missing = {path for path in modules if not any(fnmatch.fnmatchcase(path, p) for p in wheel)}
    assert not missing, f"wheel omits entry-point import closure: {sorted(missing)}"
    includes = targets["sdist"]["include"]
    assert includes and all(p.startswith("/") for p in includes), (
        "sdist include is not root-anchored"
    )
    assert all(".." not in Path(p).parts for p in includes), "sdist include escapes its root"
    missing = {
        path for path in modules if not any(fnmatch.fnmatchcase(path, p[1:]) for p in includes)
    }
    assert not missing, f"sdist omits entry-point import closure: {sorted(missing)}"
    dependencies = {
        re.split(r"[<>=!~;\[ ]", requirement, maxsplit=1)[0].replace("_", "-").lower()
        for requirement in project["dependencies"]
    }
    # The native sherpa runtime has no Python import to include in the closure.
    missing = (external | {"sherpa-onnx-core"}) - dependencies
    assert not missing, f"hard runtime dependencies missing: {sorted(missing)}"


def test_packaging_carries_the_entrypoint_import_closure_and_runtime_dependencies():
    _assert_packaging(_config())


@pytest.mark.parametrize("plant", ["wheel-module", "sdist-root", "sdist-module", "hard-dependency"])
def test_packaging_rejects_incomplete_or_unanchored_targets(plant):
    config = copy.deepcopy(_config())
    targets = config["tool"]["hatch"]["build"]["targets"]
    if plant == "wheel-module":
        targets["wheel"]["only-include"].remove("streaming.py")
        reason = "wheel omits.*streaming"
    elif plant == "sdist-root":
        targets["sdist"]["include"].append("README.md")
        reason = "not root-anchored"
    elif plant == "sdist-module":
        targets["sdist"]["include"].remove("/streaming.py")
        reason = "sdist omits.*streaming"
    else:
        config["project"]["dependencies"] = [
            dep for dep in config["project"]["dependencies"] if not dep.startswith("openvino>=")
        ]
        reason = "hard runtime dependencies missing.*openvino"
    with pytest.raises(AssertionError, match=reason):
        _assert_packaging(config)


def test_the_import_closure_follows_transitive_modules_and_local_backend_imports(tmp_path):
    (tmp_path / "entry.py").write_text("import middle\n")
    (tmp_path / "middle.py").write_text("def load():\n    import backend_package\n")
    modules, external = _import_closure(tmp_path, {"tool": "entry:main"})
    assert modules == {"entry.py", "middle.py"}
    assert external == {"backend-package"}
