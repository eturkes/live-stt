"""Stressor geometry/continuity records + synthetic cap-off controls; no audio inference."""

import json

import numpy as np
import pytest

from tests import build_stressor as stressor


def test_stressor_manifest_continuity_and_cap_control():
    manifest = json.loads(stressor.MANIFEST.read_text(encoding="utf-8"))
    for row in manifest["stressors"].values():
        order, joins, validation = row["order"], row["join_samples"], row["validation"]
        assert row["component_count"] == len(order) > 1
        assert len(joins) == len(order) - 1
        assert joins == sorted(set(joins))
        assert 0 < joins[0] < joins[-1] < row["audio_s"] * stressor.SAMPLE_RATE
        assert row["ja_ref"] == " ".join(manifest["components"][cid]["ja_ref"] for cid in order)
        assert validation["joins_inside_segments"] is True
        assert validation["soft_cap_fired"] is True
        assert validation["vad_segs_capon"] > validation["vad_segs_capoff"] > 0
        assert set(validation["per_engine"]) == {"k2v2", "parakeet"}
        for engine in validation["per_engine"].values():
            assert engine["n_seg"] == validation["vad_segs_capon"]
            assert engine["preflush_s"] >= 20


@pytest.mark.parametrize(
    ("join", "on_count", "short_engine", "inside", "expected"),
    [
        (2, 2, None, True, True),
        (7, 2, None, True, True),
        (1, 2, None, False, False),
        (8, 2, None, False, False),
        (5, 1, None, True, False),
        (5, 2, "k2v2", True, False),
        (5, 2, "parakeet", True, False),
    ],
)
def test_continuity_uses_cap_off_geometry_and_each_engines_preflush(
    monkeypatch, join, on_count, short_engine, inside, expected
):
    monkeypatch.setattr(stressor.replay, "load_wav_f32_16k", lambda _: np.zeros(1, np.float32))

    def vad(_audio, max_speech_s=None):
        return [(0, 1)] * on_count if max_speech_s is None else [(2, 6)]

    def decode(_path, engine):
        # Empty completed decodes still score D=N; only continuity/cap controls vary.
        return "", [
            dict(text="", dur_s=19 if engine == short_engine else 20),
            dict(text="", dur_s=1),
        ]

    monkeypatch.setattr(stressor, "vad_segments", vad)
    monkeypatch.setattr(stressor, "decode_unsplit_hyp", decode)
    passed, rows = stressor.validate(
        {"probe": dict(join_samples=[join], order=["a", "b"], ja_ref="AB")},
        {cid: {engine: {"D": 0} for engine in stressor.ENGINES} for cid in ("a", "b")},
    )

    assert rows["probe"]["joins_inside_segments"] is inside
    assert rows["probe"]["soft_cap_fired"] is (on_count > 1)
    assert passed is expected
