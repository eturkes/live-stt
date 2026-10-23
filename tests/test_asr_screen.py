"""eval_asr_screen's `--cache-dir`: the private compile cache an export re-qualifies on.

A cache hit hides the compiler (`openvino-accel.md`), so a re-exported model must
compile into an empty cache, never into the shared one the shipped weights warm.
"""

from __future__ import annotations

import sys

import pytest

import live_stt
from tests import eval_asr_screen as screen


@pytest.fixture
def dry_screen(monkeypatch, tmp_path):
    """main() with no model, corpus or device: records the cache each Screen meets."""
    seen = []

    class Screen:
        def __init__(self, *_args):
            seen.append(live_stt.OPENVINO_CACHE_DIR)

    monkeypatch.setattr(screen, "Screen", Screen)
    monkeypatch.setattr(screen, "NPU_LOCK", tmp_path / "npu.lock")
    monkeypatch.setattr(screen, "_kana", lambda: None)
    monkeypatch.setattr(screen, "retention_probe", lambda: None)
    monkeypatch.setattr(screen, "run_arm", lambda *_args: {})
    monkeypatch.setattr(screen, "headline", lambda *_args: "")
    monkeypatch.setattr(live_stt, "check_device", lambda *_args: None)
    # main() moves the module constant; monkeypatch restores it for the rest of the suite.
    monkeypatch.setattr(live_stt, "OPENVINO_CACHE_DIR", live_stt.OPENVINO_CACHE_DIR)

    def run(*flags):
        argv = ["eval_asr_screen.py", "--model", "m", "--device", "NPU", "--arms", "retention"]
        monkeypatch.setattr(sys, "argv", [*argv, *flags])
        screen.main()
        return seen

    return run


def test_the_model_compiles_into_the_cache_dir_named(dry_screen, tmp_path):
    cache = tmp_path / "cache"
    assert dry_screen("--cache-dir", str(cache)) == [cache]


def test_without_the_flag_the_shared_cache_stands(dry_screen):
    assert dry_screen() == [live_stt.MODELS_DIR / "openvino/cache"]
