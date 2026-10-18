"""GPU mode (`--asr-device GPU`): what the device selects beyond the device itself.

Contract (spec Decisions, asr-pipeline.md): the same models as the NPU mode -- the routing
locks in `test_model_routing.py` cover GPU like every device -- so what is GPU-specific is the
CLI surface and the host's OpenCL loader, which gets its vendor dir before any GPU probe.
"""

import os
import subprocess
import sys
import types

import pytest

import live_stt
from tests.test_audio_startup import ROOT

# --- host OpenCL loader -------------------------------------------------------


@pytest.fixture
def vendor_dirs(monkeypatch, tmp_path):
    default, shipped = tmp_path / "etc-vendors", tmp_path / "share-vendors"
    shipped.mkdir()
    monkeypatch.setattr(live_stt, "OPENCL_VENDOR_DIRS", (default, shipped))
    # setenv first records the original state, so teardown also removes a value the
    # preflight writes into os.environ; delenv alone records nothing when it is unset.
    monkeypatch.setenv("OCL_ICD_VENDORS", "unset-by-the-fixture")
    monkeypatch.delenv("OCL_ICD_VENDORS")
    return default, shipped


def test_the_host_layout_gets_its_shipped_vendor_dir(vendor_dirs):
    _default, shipped = vendor_dirs
    env = {"HOME": "/h"}

    live_stt.name_opencl_vendors(env)

    assert env == {"HOME": "/h", "OCL_ICD_VENDORS": str(shipped)}


def test_an_explicit_vendor_dir_wins(vendor_dirs):
    env = {"OCL_ICD_VENDORS": "/farm/ocl_vendors"}

    live_stt.name_opencl_vendors(env)

    assert env == {"OCL_ICD_VENDORS": "/farm/ocl_vendors"}


@pytest.mark.parametrize("layout", ["default-present", "nothing-shipped"])
def test_a_working_layout_is_left_alone(vendor_dirs, layout):
    default, shipped = vendor_dirs
    if layout == "default-present":
        default.mkdir()
    else:
        shipped.rmdir()
    env: dict[str, str] = {}

    live_stt.name_opencl_vendors(env)

    assert env == {}


@pytest.mark.parametrize("device", ["NPU", "GPU"])
def test_the_gpu_preflight_names_the_vendor_dir_before_it_probes(monkeypatch, vendor_dirs, device):
    """The probe starts the GPU plugin, so the name must already be in the process env."""
    _default, shipped = vendor_dirs
    seen = []

    class Core:
        def get_property(self, probed, name):
            seen.append(os.environ.get("OCL_ICD_VENDORS"))
            return f"fake {probed}"

    monkeypatch.setitem(sys.modules, "openvino", types.SimpleNamespace(Core=Core))

    assert live_stt.check_device("whisper", device) is None
    assert seen == [str(shipped) if device == "GPU" else None]


def test_the_cli_lists_both_modes_in_its_help():
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import live_stt, sys; sys.argv = ['live-stt', '-h']; live_stt.main()",
        ],
        cwd=ROOT,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=False,
    ).stdout

    assert "--asr-device {NPU,GPU}" in out
    assert "only task on this computer" in " ".join(out.split())
