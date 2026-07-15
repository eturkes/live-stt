"""Model-free locks for M10's model-neutral evaluator contract and evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests import eval_models as evaluator

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "model_baseline.json"


def _row(
    case_id: str,
    source: str,
    *,
    n: int,
    substitutions: int,
    deletions: int,
    insertions: int,
    duration_s: int,
    gender: str | None = None,
) -> dict:
    return {
        "D": deletions,
        "I": insertions,
        "N": n,
        "S": substitutions,
        "accepted_samples": duration_s * evaluator.SAMPLE_RATE,
        "case_id": case_id,
        "cer": (substitutions + deletions + insertions) / n,
        "complete": True,
        "duration_bucket": evaluator._duration_bucket(duration_s * evaluator.SAMPLE_RATE),
        "duration_samples": duration_s * evaluator.SAMPLE_RATE,
        "eof_count": 1,
        "gender": gender,
        "hyp": "仮説",
        "ref": "参照",
        "segments": [],
        "source": source,
    }


def test_aggregate_rows_reports_micro_macro_fleurs_strata_and_tail():
    rows = [
        _row(
            "cv",
            "common_voice_8",
            n=10,
            substitutions=1,
            deletions=1,
            insertions=0,
            duration_s=4,
        ),
        _row(
            "f-m",
            "fleurs",
            n=5,
            substitutions=0,
            deletions=1,
            insertions=0,
            duration_s=7,
            gender="male",
        ),
        _row(
            "f-f",
            "fleurs",
            n=5,
            substitutions=1,
            deletions=0,
            insertions=1,
            duration_s=21,
            gender="female",
        ),
    ]
    rows[0]["hyp"] = ""

    aggregate = evaluator.aggregate_rows(rows)

    fleurs = aggregate["by_source"]["fleurs"]
    assert fleurs["micro"] == {"D": 1, "I": 1, "N": 10, "S": 1, "cer": 0.3}
    assert fleurs["macro"] == {
        "d_rate": 0.1,
        "i_rate": 0.1,
        "s_rate": 0.1,
        "cer": 0.3,
    }
    assert aggregate["fleurs_gender"]["female"]["rows"] == 1
    assert aggregate["fleurs_gender"]["male"]["rows"] == 1
    assert aggregate["fleurs_duration"]["0-5"]["rows"] == 0
    assert aggregate["fleurs_duration"]["5-10"]["rows"] == 1
    assert aggregate["fleurs_duration"]["20+"]["rows"] == 1
    assert aggregate["worst_tail"]["fleurs"][0]["case_id"] == "f-f"
    assert aggregate["completion"] == {
        "accepted_all_audio_rows": 3,
        "complete": True,
        "complete_rows": 3,
        "empty_case_ids": ["cv"],
        "empty_hypotheses": 1,
        "eof_once_rows": 3,
        "rows": 3,
    }


def _control(cv: float, fleurs: float, long_form: float, *, empty: int = 0) -> dict:
    def scored(value: float, n: int = 1000) -> dict:
        substitutions = round(value * n)
        return {"D": 0, "I": 0, "N": n, "S": substitutions, "cer": value}

    return {
        "aggregates": {
            "by_source": {
                "common_voice_8": {"micro": scored(cv)},
                "fleurs": {"micro": scored(fleurs)},
            },
            "completion": {"complete": True, "empty_hypotheses": empty, "rows": 5133},
        },
        "compatibility": {
            "long_form": {evaluator.LONG_FORM_ID: scored(long_form)},
            "stressors": {
                case_id: {
                    **scored(0.10, 100),
                    "excess_D": 2,
                    "excess_del_rate": 0.02,
                }
                for case_id in evaluator.STRESSOR_IDS
            },
        },
    }


def test_content_and_resource_verdicts_remain_independent():
    comparator = _control(0.20, 0.10, 0.20)
    candidate = _control(0.17, 0.105, 0.205)

    content = evaluator.content_verdict(candidate, comparator)
    resource = evaluator.resource_verdict(
        {
            "cpu": {
                "actual_execution_device": "CPU",
                "post_warm_rtf": 0.25,
                "production_eligible": True,
            }
        }
    )

    assert content["qualified"] is True
    assert resource["qualified"] is False
    assert "post_warm_rtf" not in json.dumps(content)
    assert "cer" not in json.dumps(resource)

    some_empty = copy.deepcopy(candidate)
    some_empty["aggregates"]["completion"]["empty_hypotheses"] = 10
    assert evaluator.content_verdict(some_empty, comparator)["qualified"] is True

    empty_run = copy.deepcopy(candidate)
    empty_run["aggregates"]["completion"]["empty_hypotheses"] = 5133
    assert evaluator.content_verdict(empty_run, comparator)["qualified"] is False


def test_content_gate_uses_exact_counts_not_rounded_cer():
    comparator = _control(0.20, 0.10, 0.20)
    candidate = _control(0.17, 0.105, 0.205)
    comparator["aggregates"]["by_source"]["common_voice_8"]["micro"] = {
        "D": 0,
        "I": 0,
        "N": 1_000_000_000,
        "S": 1_000_000_000,
        "cer": 1.0,
    }
    candidate["aggregates"]["by_source"]["common_voice_8"]["micro"] = {
        "D": 0,
        "I": 0,
        "N": 1_000_000_000,
        "S": 900_000_001,
        "cer": 0.9,
    }

    verdict = evaluator.content_verdict(candidate, comparator)

    assert verdict["checks"]["common_voice_relative_cer"] == {
        "observed_improvement": 0.1,
        "pass": False,
    }
    assert verdict["qualified"] is False


def test_comparator_uses_exact_micro_counts_and_fixed_tie_break():
    controls = {
        "parakeet": {
            "aggregates": {
                "by_source": {"common_voice_8": {"micro": {"N": 3, "S": 1, "D": 0, "I": 0}}}
            }
        },
        "k2v2": {
            "aggregates": {
                "by_source": {"common_voice_8": {"micro": {"N": 6, "S": 2, "D": 0, "I": 0}}}
            }
        },
    }
    assert evaluator.select_comparator(controls)["engine"] == "k2v2"


def test_detail_reader_rejects_incomplete_or_reordered_rows(tmp_path):
    cases = [
        evaluator.EvalCase("a", "x", tmp_path / "a.wav", "参照", 1),
        evaluator.EvalCase("b", "x", tmp_path / "b.wav", "参照", 1),
    ]
    detail = tmp_path / "detail.jsonl"
    detail.write_bytes(evaluator._json_bytes({"case_id": "b"}, compact=True))
    digest = hashlib.sha256(detail.read_bytes()).hexdigest()

    with pytest.raises(RuntimeError, match="incomplete, duplicated, or reordered"):
        evaluator._detail_rows(detail, cases, digest)

    case = cases[0]
    corrupt = {
        "D": 0,
        "I": 0,
        "N": 2,
        "S": 1,
        "case_id": case.case_id,
        "cer": 0.5,
        "duration_bucket": None,
        "duration_samples": 1,
        "gender": None,
        "hyp": "参照",
        "ref": "参照",
        "segments": [{"text": "参照"}],
        "source": "x",
    }
    detail.write_bytes(evaluator._json_bytes(corrupt, compact=True))
    digest = hashlib.sha256(detail.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="detail score drift"):
        evaluator._detail_rows(detail, [case], digest)


def test_install_hash_failure_preserves_existing_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_bytes(b"old\n")
    staged = {}
    summaries = {}
    for engine in evaluator.CONTROL_ENGINES:
        path = tmp_path / f"{engine}.jsonl"
        path.write_bytes(engine.encode())
        staged[engine] = path
        summaries[engine] = {"details": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
    summaries["parakeet"]["details"]["sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="changed before install"):
        evaluator.install_evidence(
            {}, summaries, staged, baseline=baseline, details_dir=tmp_path / "details"
        )

    assert baseline.read_bytes() == b"old\n"
    assert not (tmp_path / "details").exists()


def _committed_baseline() -> dict:
    if not BASELINE.exists():
        pytest.skip("M10.3 control baseline not generated yet")
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_committed_baseline_locks_gates_compatibility_and_measurement_separation():
    baseline = _committed_baseline()
    deterministic = baseline["deterministic"]
    assert baseline["schema_version"] == evaluator.SCHEMA_VERSION
    assert deterministic["displacement_gates"] == evaluator.DISPLACEMENT_GATES
    assert deterministic["metric_contract"] == evaluator.METRIC_CONTRACT
    assert set(deterministic["controls"]) == set(evaluator.CONTROL_ENGINES)
    assert baseline["measurements"]["excluded_from_deterministic_equality"] is True

    goldens = json.loads(evaluator.REPLAY_GOLDENS.read_text(encoding="utf-8"))
    legacy_cer = json.loads(evaluator.CER_BASELINE.read_text(encoding="utf-8"))
    legacy_long = json.loads(evaluator.LONG_FORM.read_text(encoding="utf-8"))
    for engine, control in deterministic["controls"].items():
        for case_id, golden in goldens[engine].items():
            row = control["compatibility"]["short"][case_id]
            assert row["hyp"] == "".join(segment["text"] for segment in golden["segments"])
            assert (row["N"], row["S"], row["D"], row["I"]) == evaluator._expected_score(
                golden["ja_ref"], row["hyp"]
            )
        for case_id, expected in legacy_cer["stressors"][engine].items():
            row = control["compatibility"]["stressors"][case_id]
            assert (row["hyp"], row["N"], row["S"], row["D"], row["I"]) == (
                expected["hyp"],
                expected["N"],
                expected["S"],
                expected["D"],
                expected["I"],
            )
        long_row = control["compatibility"]["long_form"][evaluator.LONG_FORM_ID]
        expected = legacy_long["scores"][engine]
        assert (long_row["hyp"], long_row["S"], long_row["D"], long_row["I"]) == (
            expected["hyp"],
            expected["S"],
            expected["D"],
            expected["I"],
        )
        measurement = baseline["measurements"]["controls"][engine]
        assert len(measurement["post_warm"]["runs"]) == evaluator.TIMING_RUNS
        assert measurement["device_memory_mib"] is None
        assert measurement["rss_mib"]["ru_maxrss_process"] > 0
    assert "compatibility_inputs" in deterministic["pipeline"]
    assert len(deterministic["pipeline"]["evaluator_contract_sha256"]) == 64


def test_ignored_details_hash_and_aggregates_rebuild_when_cache_is_present():
    baseline = _committed_baseline()
    try:
        _, cases = evaluator.load_corpus_cases(verify_pcm=False)
    except RuntimeError:
        pytest.skip("ignored M10 corpus cache absent")

    for control in baseline["deterministic"]["controls"].values():
        detail = ROOT / control["details"]["path"]
        if not detail.exists():
            pytest.skip("ignored M10 evaluator details absent")
        rows = evaluator._detail_rows(detail, cases, control["details"]["sha256"])
        assert evaluator.aggregate_rows(rows) == control["aggregates"]
