"""Hermetic locks on the VAC trace builder's evidence boundaries."""

import hashlib
import json
import wave
from pathlib import Path

import pytest

import live_stt
from tests import build_vac_trace as builder


@pytest.mark.parametrize("corrupt_clip", builder.CLIPS)
def test_vac_builder_refuses_unpinned_wavs_before_any_model_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corrupt_clip: str
):
    (tmp_path / "tests").mkdir()
    hashes = {}
    for name in builder.CLIPS:
        path = tmp_path / f"{name}.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(live_stt.SAMPLE_RATE)
            output.writeframes(b"\0\0" * live_stt.SAMPLE_RATE)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    pins = dict(hashes)
    pins[corrupt_clip] = "f" * 64

    def write_pins():
        (tmp_path / "tests" / "stressor_clips.json").write_text(
            json.dumps({"stressors": {"stress_long": {"audio_sha256": pins["stress_long"]}}}),
            encoding="utf-8",
        )
        (tmp_path / "tests" / "retention_probe.json").write_text(
            json.dumps({"probe": {"audio_sha256": pins["retention_probe"]}}),
            encoding="utf-8",
        )

    write_pins()
    probes = []
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "CACHE", tmp_path)
    monkeypatch.setattr(builder, "check_models", lambda engine: probes.append("models"))
    monkeypatch.setattr(builder, "check_device", lambda engine, device: probes.append("device"))
    monkeypatch.setattr(builder, "load_recognizer", lambda engine, device: probes.append("load"))
    monkeypatch.setattr(
        builder,
        "trace_clip",
        lambda path, rec: {"wav_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    with pytest.raises(RuntimeError, match="(?i)(hash|fingerprint|pinned|match)"):
        builder.build("NPU")
    assert probes == []

    pins[corrupt_clip] = hashes[corrupt_clip]
    write_pins()
    report = builder.build("NPU")
    assert probes == ["models", "device", "load"]
    assert {name: row["wav_sha256"] for name, row in report["clips"].items()} == hashes
