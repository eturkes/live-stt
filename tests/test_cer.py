"""Model-free locks for CER primitives and the committed evaluation baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cer import align, cer, normalize
from tests.eval_cer import MAX_CER, validation_failures

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
BASELINE = json.loads((TESTS / "cer_baseline.json").read_text(encoding="utf-8"))
LONG_FORM = json.loads((TESTS / "long_form.json").read_text(encoding="utf-8"))
REAL_CLIPS = json.loads((TESTS / "real_clips.json").read_text(encoding="utf-8"))
REPLAY_GOLDENS = json.loads((TESTS / "replay_goldens.json").read_text(encoding="utf-8"))
STRESSOR_CLIPS = json.loads((TESTS / "stressor_clips.json").read_text(encoding="utf-8"))

ENGINES = {"k2v2", "parakeet"}
STRESSOR_IDS = {"stress_long", "stress_med"}
RTF_LENGTHS = {"5", "10", "20", "40"}


def test_normalize_fixed_vectors():
    assert normalize("Ａb, C！￥") == "abc"
    assert normalize("a\r\n\tb") == "ab"
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
            assert row["cer"] <= MAX_CER
            if engine == "k2v2":  # shipped default: M9.4 acceptance <=4% absolute
                assert row["excess_del_rate"] <= 0.04


def test_stressor_gate_rejects_insertion_regression_despite_zero_excess_deletion():
    out = {
        "stressors": {
            "k2v2": {"stress_long": {"excess_del_rate": 0.0, "cer": 0.05}},
            "parakeet": {"stress_long": {"excess_del_rate": 0.0, "cer": MAX_CER + 0.01}},
        }
    }
    failures = validation_failures(out)
    assert len(failures) == 1
    assert failures[0].startswith("parakeet/stress_long CER")


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


def test_long_form_rows_recompute_from_committed_text():
    scored = [s for s in LONG_FORM["sections"].values() if "scores" in s]
    assert scored, "the long-form manifest records no scored section"
    for section in scored:
        ref = section["reference"]["text"]
        assert set(section["scores"]) == ENGINES
        for row in section["scores"].values():
            assert row["ref"] == ref
            counts = align(normalize(ref), normalize(row["hyp"]))
            assert counts == (row["S"], row["D"], row["I"])
            assert cer(ref, row["hyp"]) == row["cer"]
            assert len(normalize(ref)) == row["N"]
            assert row["D"] / row["N"] == row["del_rate"]
            assert row["I"] / row["N"] == row["ins_rate"]


def test_long_form_provenance_alignment_and_natural_endpointing():
    source = LONG_FORM["source"]
    sections = LONG_FORM["sections"]

    assert source["license"].startswith("CC0-1.0")
    for artifact in (source["alignment"], source["text"]):
        assert artifact["url"].startswith("https://")
        assert len(artifact["sha256"]) == 64
    assert sorted(sections) == [f"{n:02d}" for n in range(1, 7)]
    assert sum(s["build"]["audio_s"] for s in sections.values()) > 14 * 60

    for key, section in sections.items():
        build, reference, vad = section["build"], section["reference"], section["vad"]
        assert section["id"] == f"gongitsune_{key}"
        assert section["audio"]["url"].endswith(f"gongitsune_{key}_niimi_64kb.mp3")
        assert len(section["audio"]["sha256"]) == 64
        assert build["aligned_end"] > build["aligned_start"] > 0
        assert build["row_count"] == build["last_row"] - build["first_row"] + 1
        assert len(build["wav_sha256"]) == 64

        ref_norm = normalize(reference["text"])
        assert hashlib.sha256(ref_norm.encode()).hexdigest() == reference["normalized_sha256"]
        counts = align(ref_norm, normalize(reference["kokoro_alignment_text"]))
        check = reference["alignment_check"]
        assert counts == (check["S"], check["D"], check["I"])
        assert check["N"] == len(ref_norm)
        assert check["cer"] == sum(counts) / len(ref_norm)
        # Narration Kokoro left unaligned is the only licence for disagreement
        # above the flat surface budget; anything else is a bad extraction.
        span = build["aligned_end"] - build["aligned_start"]
        assert check["cer"] <= 0.10 + build["unaligned_samples"] / span

        durations = vad["segment_durations_s"]
        assert vad["n_segments"] == len(durations) > 1
        assert vad["max_segment_s"] == max(durations)
        assert vad["max_resliced_upper_bound_s"] <= vad["decode_split_trigger_s"]
        assert vad["decode_split_candidates_upper_bound"] == 0


def test_long_form_sections_partition_the_alignment():
    # Row 1 is the separately spoken title/author, so narration starts at 2 and
    # the six sections must tile the rest -- a gap or overlap means a section was
    # cropped from the wrong rows.
    ranges = sorted(
        (s["build"]["first_row"], s["build"]["last_row"]) for s in LONG_FORM["sections"].values()
    )
    assert ranges[0][0] == 2
    for (_, previous_last), (next_first, _) in zip(ranges, ranges[1:], strict=False):
        assert next_first == previous_last + 1
