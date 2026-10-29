"""Shell/child filtering shares fixture paths; no accelerator libraries load."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import live_stt

ROOT = Path(__file__).resolve().parent.parent


def _shell_env(source, environ, layer, tmp_path):
    script = tmp_path / "envrc"
    script.write_text(source)
    done = subprocess.run(
        [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            'PWD=$1; source "$2"; /usr/bin/env -0',
            "bash",
            layer,
            str(script),
        ],
        env={"PATH": "/usr/bin:/bin", **environ},
        capture_output=True,
        timeout=5,
        check=True,
    )
    return dict(entry.split("=", 1) for entry in done.stdout.decode().split("\0") if entry)


def _assert_shell_policy(source, tmp_path, layer, venv):
    libraries = {}
    for name, library in (
        ("core", "libopenvino.so"),
        ("versioned core", "libopenvino.so.2641"),
        ("tbb", "libtbb.so"),
        ("versioned tbb", "libtbb.so.12"),
        ("farm", "libopenvino_intel_npu_compiler_loader.so"),
        ("unrelated", "libother.so"),
    ):
        directory = tmp_path / name
        directory.mkdir()
        (directory / library).touch()
        libraries[name] = str(directory)
    shadow = tmp_path / "python-shadow"
    (shadow / "openvino").mkdir(parents=True)
    seam = tmp_path / "unrelated-python"
    seam.mkdir()
    supplied = {
        "LD_LIBRARY_PATH": ":" + ":".join(libraries.values()) + "::",
        "PYTHONPATH": f"{shadow}:{seam}",
        "OCL_ICD_VENDORS": "/fixture/vendors",
        "ZE_ENABLE_ALT_DRIVERS": "/fixture/driver.so",
    }
    before = dict(supplied)
    shell = _shell_env(source, supplied, layer, tmp_path)
    child = live_stt._session_env(supplied)
    kept = os.pathsep.join(libraries[name] for name in ("farm", "unrelated"))
    assert shell.get("LD_LIBRARY_PATH") == child.get("LD_LIBRARY_PATH") == kept, (
        "shell/child library filtering must drop core and TBB, retaining farm and unrelated dirs"
    )
    assert shell["UV_PROJECT_ENVIRONMENT"] == venv, "shell selected the wrong venv layer"
    assert "PYTHONPATH" not in shell, "shell left a shadowing PYTHONPATH"
    assert child["PYTHONPATH"] == str(seam), "child lost its non-OpenVINO import seam"
    assert shell["UV_LINK_MODE"] == "copy"
    for key in ("OCL_ICD_VENDORS", "ZE_ENABLE_ALT_DRIVERS"):
        assert shell[key] == child[key] == supplied[key]
    assert supplied == before


@pytest.mark.parametrize(
    ("layer", "venv"),
    [("/run/host/home/user/project", ".venv"), ("/home/user/project", ".venv-host")],
)
def test_the_shell_prelude_matches_child_library_filtering_and_selects_its_layer(
    tmp_path, layer, venv
):
    _assert_shell_policy((ROOT / ".envrc").read_text(), tmp_path, layer, venv)


@pytest.mark.parametrize("value", [None, "", ":", "::"])
def test_empty_search_paths_are_unset_in_the_shell_and_child(tmp_path, value):
    supplied = {} if value is None else {"PYTHONPATH": value, "LD_LIBRARY_PATH": value}
    shell = _shell_env((ROOT / ".envrc").read_text(), supplied, "/home/user/project", tmp_path)
    child = live_stt._session_env(supplied)
    for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        assert key not in shell and key not in child


@pytest.mark.parametrize("plant", ["core", "tbb", "layer", "python"])
def test_the_shell_policy_lock_rejects_independent_prelude_regressions(tmp_path, plant):
    source = (ROOT / ".envrc").read_text()
    old, new, reason = {
        "core": ("libopenvino.so*", "libnotopenvino.so*", "library filtering"),
        "tbb": ("libtbb.so*", "libnottbb.so*", "library filtering"),
        "layer": (
            "UV_PROJECT_ENVIRONMENT=.venv ;;",
            "UV_PROJECT_ENVIRONMENT=.venv-host ;;",
            "venv layer",
        ),
        "python": ("unset PYTHONPATH", ": # retained PYTHONPATH", "shadowing PYTHONPATH"),
    }[plant]
    assert old in source
    with pytest.raises(AssertionError, match=reason):
        _assert_shell_policy(
            source.replace(old, new), tmp_path, "/run/host/home/user/project", ".venv"
        )
