"""Model-free locks for the committed M10.5b state-retention probe."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
RETENTION_PROBE = json.loads((TESTS / "retention_probe.json").read_text(encoding="utf-8"))
REAL_CLIPS = json.loads((TESTS / "real_clips.json").read_text(encoding="utf-8"))
CACHE_WAV = ROOT / "spike" / "backends" / "cache" / "retention_probe.wav"

SAMPLE_RATE = 16_000
SOURCE_CLIPS = ["cv_short", "cv_med", "cv_long", "cv_kana", "cv_xlong"]
TRIMMED_DUR_S = {
    "cv_short": 1.478,
    "cv_med": 1.862,
    "cv_long": 3.014,
    "cv_kana": 2.566,
    "cv_xlong": 5.734,
}


def _join_offsets(part_lens: list[int], xfade: int, lead: int) -> list[int]:
    centers: list[int] = []
    cumulative = part_lens[0]
    for part_len in part_lens[1:]:
        centers.append(lead + cumulative - xfade // 2)
        cumulative += part_len - xfade
    return centers


def test_recipe_and_component_provenance():
    recipe = RETENTION_PROBE["recipe"]
    components = RETENTION_PROBE["components"]

    assert recipe["source_clips"] == SOURCE_CLIPS
    assert recipe["crossfade_ms"] == 10.0
    assert recipe["lead_s"] == 0.3
    assert recipe["tail_s"] == 0.6
    assert recipe["target_s"] == 180.0
    assert set(components) == set(SOURCE_CLIPS)
    assert {cid: components[cid]["trimmed_dur_s"] for cid in SOURCE_CLIPS} == TRIMMED_DUR_S

    for cid in SOURCE_CLIPS:
        assert components[cid]["ja_ref"] == REAL_CLIPS[cid]["ja_ref"]
        assert components[cid]["source"] == REAL_CLIPS[cid]["source"]


def test_probe_geometry_reference_and_continuity():
    recipe = RETENTION_PROBE["recipe"]
    components = RETENTION_PROBE["components"]
    probe = RETENTION_PROBE["probe"]
    order = probe["order"]

    assert probe["component_count"] == len(order) > 1
    assert all(cid in components for cid in order)
    assert probe["ja_ref"] == " ".join(components[cid]["ja_ref"] for cid in order)

    xfade = round(recipe["crossfade_ms"] * SAMPLE_RATE / 1000)
    lead = round(recipe["lead_s"] * SAMPLE_RATE)
    tail = round(recipe["tail_s"] * SAMPLE_RATE)
    part_lens = [round(components[cid]["trimmed_dur_s"] * SAMPLE_RATE) for cid in order]
    expected_joins = _join_offsets(part_lens, xfade, lead)
    committed_joins = probe["join_samples"]
    total_samples = sum(part_lens) - (len(part_lens) - 1) * xfade + lead + tail

    assert len(committed_joins) == len(order) - 1
    assert all(left < right for left, right in pairwise(committed_joins))
    assert all(0 < offset < total_samples for offset in committed_joins)
    # Each 3-decimal component duration can hide at most half a millisecond.
    rounding_per_part = SAMPLE_RATE * 0.0005
    for preceding_parts, (committed, expected) in enumerate(
        zip(committed_joins, expected_joins, strict=True), start=1
    ):
        assert abs(committed - expected) <= preceding_parts * rounding_per_part

    expected_audio_s = (
        sum(components[cid]["trimmed_dur_s"] for cid in order)
        - (len(order) - 1) * recipe["crossfade_ms"] / 1000
        + recipe["lead_s"]
        + recipe["tail_s"]
    )
    assert probe["audio_s"] == pytest.approx(expected_audio_s, abs=len(order) * 0.0005)
    assert 178.0 <= probe["audio_s"] <= 187.0

    validation = probe["validation"]
    assert validation["vad_segs"] == 1
    assert validation["joins_inside_segments"] is True
    segment = validation["segment"]
    segment_end = segment["start"] + segment["length"]
    assert segment["length"] > 0
    assert all(segment["start"] <= offset < segment_end for offset in committed_joins)


def test_cached_probe_hash_when_present():
    if CACHE_WAV.exists():
        assert (
            hashlib.sha256(CACHE_WAV.read_bytes()).hexdigest()
            == RETENTION_PROBE["probe"]["audio_sha256"]
        )
