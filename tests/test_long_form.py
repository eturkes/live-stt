"""Long-form --score retains completed ASR misses and propagates real replay failures."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from tests import eval_cer
from tests.evaluator_fakes import hermetic_evaluator  # noqa: F401

pytestmark = pytest.mark.usefixtures("hermetic_evaluator")


def test_long_form_scores_completed_empty_decodes(monkeypatch, tmp_path):
    path = Path(__file__).with_name("eval_long_form.py")
    spec = importlib.util.spec_from_file_location("long_form_scoring", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setitem(sys.modules, "soundfile", types.ModuleType("soundfile"))
    spec.loader.exec_module(module)
    manifest = tmp_path / "long_form.json"
    vad = tmp_path / "vad.onnx"
    vad.touch()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    monkeypatch.setattr(module, "VAD_MODEL", vad)
    monkeypatch.setattr(module, "AUDIO_SHA256", {1: "unused"})
    monkeypatch.setattr(module, "check_models", lambda _: None)
    monkeypatch.setattr(module, "fetch", lambda *_: b"")
    monkeypatch.setattr(module, "zip_member", lambda *_: b"")
    monkeypatch.setattr(module, "alignment_sections", lambda _: {1: []})
    monkeypatch.setattr(module, "report", lambda _: None)
    monkeypatch.setattr(
        module,
        "build_section",
        lambda *_: {
            "reference": {"text": "日本語"},
            "build": {"wav": "unused.wav", "audio_s": 1.0},
        },
    )
    monkeypatch.setattr(eval_cer, "decode_hyp", lambda *_: ("", []))
    monkeypatch.setattr(module, "score_wav", eval_cer.score_wav)
    monkeypatch.setattr(sys, "argv", ["eval_long_form.py", "--score"])
    module.main()
    rows = json.loads(manifest.read_text())["sections"]["01"]["scores"]
    assert set(rows) == set(module.ENGINES)
    assert all(row["D"] == row["N"] == 3 and row["cer"] == 1 for row in rows.values())

    def failed_decode(*_):
        raise RuntimeError("worker stopped during replay")

    previous = manifest.read_bytes()
    monkeypatch.setattr(eval_cer, "decode_hyp", failed_decode)
    with pytest.raises(RuntimeError, match="worker stopped during replay"):
        module.main()
    assert manifest.read_bytes() == previous
