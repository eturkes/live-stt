"""Long-form acquisition preserves valid scores; scoring retains completed ASR misses."""

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


@pytest.fixture
def long_form(monkeypatch):
    path = Path(__file__).with_name("eval_long_form.py")
    spec = importlib.util.spec_from_file_location("long_form_scoring", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setitem(sys.modules, "soundfile", types.ModuleType("soundfile"))
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("changed", [None, "wav", "reference", "both", "unscored"])
def test_recorded_scores_survive_only_when_both_input_hashes_hold(long_form, changed):
    scores = {"k2v2": {"D": 1}, "parakeet": {"D": 2}}
    old = {
        "build": {"wav_sha256": "wav before"},
        "reference": {"normalized_sha256": "reference before"},
        **({} if changed == "unscored" else {"scores": scores}),
    }
    rebuilt = {
        "build": {"wav_sha256": "wav after" if changed in ("wav", "both") else "wav before"},
        "reference": {
            "normalized_sha256": "reference after"
            if changed in ("reference", "both")
            else "reference before"
        },
    }
    assert long_form.keep_scores(old, rebuilt) == (scores if changed is None else None)


def test_acquisition_retains_scores_for_identical_inputs_and_unselected_sections(
    long_form, monkeypatch, tmp_path
):
    manifest = tmp_path / "long_form.json"
    prior = {
        f"{n:02d}": {
            "build": {"wav_sha256": f"wav {n}", "audio_s": 1},
            "reference": {"normalized_sha256": f"ref {n}"},
            "scores": {"k2v2": {"D": n}, "parakeet": {"D": n + 1}},
        }
        for n in (1, 2, 3)
    }
    manifest.write_text(json.dumps({"sections": prior}), encoding="utf-8")
    vad = tmp_path / "vad.onnx"
    vad.touch()
    monkeypatch.setattr(long_form, "ROOT", tmp_path)
    monkeypatch.setattr(long_form, "MANIFEST", manifest)
    monkeypatch.setattr(long_form, "VAD_MODEL", vad)
    monkeypatch.setattr(long_form, "AUDIO_SHA256", {1: "unused", 2: "unused", 3: "unused"})
    monkeypatch.setattr(long_form, "fetch", lambda *_: b"")
    monkeypatch.setattr(long_form, "zip_member", lambda *_: b"")
    monkeypatch.setattr(long_form, "alignment_sections", lambda _: {1: [], 2: [], 3: []})
    monkeypatch.setattr(long_form, "report", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["eval_long_form.py", "--sections", "1,2"])

    def rebuild(n, *_):
        return {
            "build": {"wav_sha256": f"wav {n}", "audio_s": 1},
            "reference": {"normalized_sha256": "changed" if n == 2 else "ref 1"},
        }

    def no_decode(*_):
        raise AssertionError("acquisition re-decoded prior scores")

    monkeypatch.setattr(long_form, "build_section", rebuild)
    monkeypatch.setattr(long_form, "score_wav", no_decode)
    long_form.main()
    rows = json.loads(manifest.read_text(encoding="utf-8"))["sections"]
    assert rows["01"] == prior["01"]
    assert "scores" not in rows["02"]
    assert rows["03"] == prior["03"]
    first = manifest.read_bytes()
    long_form.main()
    assert manifest.read_bytes() == first


def test_long_form_scores_completed_empty_decodes(long_form, monkeypatch, tmp_path):
    module = long_form
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
