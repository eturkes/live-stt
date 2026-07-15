"""Model-free locks for CER primitives and the committed evaluation baseline."""

from __future__ import annotations

import json
from pathlib import Path

from cer import align, cer, normalize

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
BASELINE = json.loads((TESTS / "cer_baseline.json").read_text(encoding="utf-8"))
REAL_CLIPS = json.loads((TESTS / "real_clips.json").read_text(encoding="utf-8"))
REPLAY_GOLDENS = json.loads((TESTS / "replay_goldens.json").read_text(encoding="utf-8"))
STRESSOR_CLIPS = json.loads((TESTS / "stressor_clips.json").read_text(encoding="utf-8"))

ENGINES = {"k2v2", "parakeet"}
STRESSOR_IDS = {"stress_long", "stress_med"}
RTF_LENGTHS = {"5", "10", "20", "40"}


def test_normalize_fixed_vectors():
    assert normalize("Ａb, C！￥") == "abc"
    assert normalize("空が青いです。.") == "空が青いです"
    assert align(normalize("七"), normalize("7")) == (1, 0, 0)


def test_align_and_cer_fixed_vectors():
    assert align("abc", "abc") == (0, 0, 0)
    assert align("abc", "") == (0, 3, 0)
    assert align("", "abc") == (0, 0, 3)
    assert align("a", "b") == (1, 0, 0)
    assert align("ab", "ba") == (2, 0, 0)
    assert cer("abc", "adc") == 1 / 3
    assert cer("", "x") == 0.0


def test_scored_rows_recompute_from_committed_text():
    for section in ("corpus", "stressors"):
        for rows in BASELINE[section].values():
            for row in rows.values():
                ref = normalize(row["ref"])
                counts = align(ref, normalize(row["hyp"]))
                assert counts == (row["S"], row["D"], row["I"])
                assert cer(row["ref"], row["hyp"]) == row["cer"]
                assert len(ref) == row["N"]
                assert row["D"] / row["N"] == row["del_rate"]
                assert row["I"] / row["N"] == row["ins_rate"]


def test_baseline_structure_and_provenance():
    assert set(BASELINE) == {"corpus", "stressors", "rtf_by_length"}
    for section in BASELINE.values():
        assert set(section) == ENGINES

    real_ids = set(REAL_CLIPS)
    for engine in ENGINES:
        assert set(BASELINE["corpus"][engine]) == real_ids
        assert set(BASELINE["stressors"][engine]) == STRESSOR_IDS
        assert set(BASELINE["rtf_by_length"][engine]) == RTF_LENGTHS

        for cid, meta in REAL_CLIPS.items():
            row = BASELINE["corpus"][engine][cid]
            assert row["ref"] == meta["ja_ref"]
            assert row["hyp"] == "".join(
                segment["text"] for segment in REPLAY_GOLDENS[engine][cid]["segments"]
            )

        for sid in STRESSOR_IDS:
            assert (
                BASELINE["stressors"][engine][sid]["ref"]
                == STRESSOR_CLIPS["stressors"][sid]["ja_ref"]
            )


def test_stressor_excess_uses_component_baseline_and_meets_shipped_gate():
    for engine in ENGINES:
        for sid in STRESSOR_IDS:
            row = BASELINE["stressors"][engine][sid]
            order = STRESSOR_CLIPS["stressors"][sid]["order"]
            baseline_d = sum(
                STRESSOR_CLIPS["components"][cid]["baseline"][engine]["D"] for cid in order
            )
            assert row["excess_D"] == row["D"] - baseline_d
            assert row["excess_del_rate"] == round(row["excess_D"] / row["N"], 4)
            if engine == "k2v2":  # shipped default: M9.4 acceptance <=4% absolute
                assert row["excess_del_rate"] <= 0.04


def test_stressor_manifest_retains_defect_era_reproduction():
    # The M9.1 manifest is the immutable before-fix construction/QC record;
    # cer_baseline.json tracks the current production pipeline after M9.4.
    assert "M9.4 chunking disabled" in STRESSOR_CLIPS["recipe"]["decode_control"]
    for sid in STRESSOR_IDS:
        validation = STRESSOR_CLIPS["stressors"][sid]["validation"]["per_engine"]["k2v2"]
        assert validation["excess_rate"] >= 0.10


def test_rtf_rows_have_valid_shape_and_long_form_output():
    expected_keys = {"audio_s", "decode_s", "rtf", "n_seg", "n_nonempty", "viable"}
    for engine in ENGINES:
        for row in BASELINE["rtf_by_length"][engine].values():
            assert set(row) == expected_keys
            assert row["n_seg"] >= row["n_nonempty"] >= 0
            assert row["viable"] == (row["n_nonempty"] > 0)
        assert BASELINE["rtf_by_length"][engine]["40"]["viable"]
