"""Model-free locks for M10's model-neutral evaluator contract and evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from tests import eval_models as evaluator

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "model_baseline.json"
DETAIL_SHA256 = {
    "cohere_transcribe": "48cf513841deec2265b8ced9a5b9dd5bec185b8b4919e7a7a7db6a59d6ffaec3",
    "k2v2": "29f1de5a60fc76ba57c0c002d26748df616c29200ef3fec690e681307251950d",
    "parakeet": "810f0f25e629b73bed69e0f6b5273220ae91148ccca1818e5a5a24318a8c9c76",
    "qwen3_asr": "1fec91b3c38faa918075a3ed344efbbcb4ce69ec4d668966a54a4474260654fd",
}
CANDIDATE_OUTPUT_SHA256 = {
    "cohere_transcribe": {
        "aggregates": "9f148582aa903358f85664f385de9d5b0f6eb164c6d6fcbd37ed1b1e3249bbc1",
        "compatibility": "2115db34ec2c2b604be4c7ae5496f0aa8e27da707573b4069cdaf4abb21192c6",
    },
    "qwen3_asr": {
        "aggregates": "08e334469693a70d3926fb4d9f4a88c8aa64abd7744ac439310ee0e79690c07f",
        "compatibility": "ea3c40b5051c564e44f9ca972e473238868b32ade92551224dc69c31eeb8897b",
    },
}
QWEN_DIAGNOSTIC_CASE_IDS = [
    "cv8-ja-test-001632",
    "cv8-ja-test-003098",
    "cv8-ja-test-004405",
    "fleurs-ja-test-10710420115318518641",
    "fleurs-ja-test-4343504618128851925",
    "fleurs-ja-test-7887185863950876644",
]


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
        "diagnostics": evaluator.summarize_candidate_diagnostics([]),
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

    truncated = copy.deepcopy(candidate)
    truncated["diagnostics"] = evaluator.summarize_candidate_diagnostics(
        [
            {
                "case_id": "cv-case",
                "code": "generation_max_new_tokens",
                "message": "Result is truncated. max_new_tokens 128 is too small",
                "phase": "corpus",
            }
        ]
    )
    verdict = evaluator.content_verdict(truncated, comparator)
    assert verdict["checks"]["decoder_diagnostics"] == {
        "case_ids": ["cv-case"],
        "observed_events": 1,
        "pass": False,
    }
    assert verdict["qualified"] is False


def test_native_stderr_diagnostics_are_captured_classified_and_restored(capfd):
    expected = evaluator.DecodeObservation(
        content=evaluator.Transcript("", (), 0, 1, True),
        decode_seconds=0.0,
        wall_seconds=0.0,
    )

    def decode() -> evaluator.DecodeObservation:
        os.write(2, b"Result is truncated. max_new_tokens 128 is too small\n")
        return expected

    observed, messages = evaluator._decode_with_native_diagnostics(decode)
    os.write(2, b"after decode\n")

    assert observed is expected
    assert messages == ("Result is truncated. max_new_tokens 128 is too small",)
    assert evaluator._native_diagnostic_code(messages[0]) == "generation_max_new_tokens"
    assert capfd.readouterr().err.endswith("after decode\n")


def test_candidate_diagnostic_summary_fails_closed_on_tampering():
    events = [
        {
            "case_id": "case",
            "code": "audio_context_truncated",
            "message": "Truncating audio placeholders",
            "phase": "compatibility/short",
        },
        {
            "case_id": "stress_long",
            "code": "native_stderr",
            "message": "timing warning",
            "phase": "timing/post_warm/1",
        },
    ]
    summary = evaluator.summarize_candidate_diagnostics(events)
    phase_cases = {
        "compatibility/short": {"case"},
        "timing/post_warm/1": {"stress_long"},
    }

    evaluator.validate_candidate_diagnostics("candidate", summary, phase_cases)
    assert summary["content_event_case_ids"] == ["case"]
    assert summary["content_events"] == events[:1]

    corrupt = copy.deepcopy(summary)
    corrupt["events"][0]["code"] = "native_stderr"
    with pytest.raises(RuntimeError, match="diagnostic code drifted"):
        evaluator.validate_candidate_diagnostics("candidate", corrupt, phase_cases)

    corrupt = copy.deepcopy(summary)
    corrupt["events"][0]["case_id"] = "not-a-real-case"
    corrupt["content_events"][0]["case_id"] = "not-a-real-case"
    corrupt["content_event_case_ids"] = ["not-a-real-case"]
    with pytest.raises(RuntimeError, match="diagnostic phase/case drifted"):
        evaluator.validate_candidate_diagnostics("candidate", corrupt, phase_cases)


def test_candidate_stressor_baseline_uses_same_candidate_component_rows():
    row = _row(
        "stress",
        "continuous_stressor",
        n=100,
        substitutions=1,
        deletions=8,
        insertions=0,
        duration_s=10,
    )
    compatibility = {"short": {"a": {"D": 2}, "b": {"D": 1}}}
    expected = {"stressor_manifest": {"stressors": {"stress": {"order": ["a", "b", "a"]}}}}

    output = evaluator._candidate_compatibility_row(
        "stressors",
        evaluator.EvalCase("stress", "continuous_stressor", Path("x.wav"), "参照", 1),
        row,
        compatibility,
        expected,
    )

    assert output["baseline_D"] == 5
    assert output["excess_D"] == 3
    assert output["excess_del_rate"] == 0.03


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
    for engine in evaluator.OFFLINE_MODELS:
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


def test_aggregate_migration_allows_only_provenance_code_with_exact_model_artifacts():
    implementation = [
        {"path": "live_stt.py", "sha256": "live"},
        {"path": "tests/fetch_eval_models.py", "sha256": "old"},
    ]
    previous = {"implementation": implementation, "values": {"threads": 4}}
    current = copy.deepcopy(previous)
    current["implementation"][1]["sha256"] = "new"

    assert evaluator._pipeline_reaggregate_compatible(previous, current)

    changed_decode = copy.deepcopy(current)
    changed_decode["implementation"][0]["sha256"] = "changed"
    assert not evaluator._pipeline_reaggregate_compatible(previous, changed_decode)

    old_model = {
        "artifacts": [{"path": "model.onnx", "sha256": "exact", "bytes": 1}],
        "bytes": 1,
        "directory": "models/candidate",
        "provenance": {"lineage": "old"},
    }
    corrected_provenance = {**old_model, "provenance": {"lineage": "corrected"}}
    assert evaluator._model_decode_identity(old_model) == evaluator._model_decode_identity(
        corrected_provenance
    )

    changed_model = copy.deepcopy(corrected_provenance)
    changed_model["artifacts"][0]["sha256"] = "changed"
    assert evaluator._model_decode_identity(old_model) != evaluator._model_decode_identity(
        changed_model
    )


def _committed_baseline() -> dict:
    if not BASELINE.exists():
        pytest.skip("M10 model baseline not generated yet")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if baseline.get("schema_version") != evaluator.SCHEMA_VERSION:
        pytest.skip("M10.4 tournament baseline not generated yet")
    return baseline


def test_committed_baseline_locks_gates_compatibility_and_measurement_separation():
    baseline = _committed_baseline()
    deterministic = baseline["deterministic"]
    assert baseline["schema_version"] == evaluator.SCHEMA_VERSION
    assert deterministic["displacement_gates"] == evaluator.DISPLACEMENT_GATES
    assert deterministic["metric_contract"] == evaluator.METRIC_CONTRACT
    assert set(deterministic["controls"]) == set(evaluator.CONTROL_ENGINES)
    assert set(deterministic["candidates"]) == set(evaluator.OFFLINE_CANDIDATES)
    assert baseline["measurements"]["excluded_from_deterministic_equality"] is True
    assert not set(evaluator.OFFLINE_CANDIDATES) & evaluator.ENGINE_DIRS.keys()

    goldens = json.loads(evaluator.REPLAY_GOLDENS.read_text(encoding="utf-8"))
    legacy_cer = json.loads(evaluator.CER_BASELINE.read_text(encoding="utf-8"))
    legacy_long = json.loads(evaluator.LONG_FORM.read_text(encoding="utf-8"))
    for engine, control in deterministic["controls"].items():
        assert control["details"]["sha256"] == DETAIL_SHA256[engine]
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

    comparator = deterministic["controls"][deterministic["comparator"]["engine"]]
    for engine, candidate in deterministic["candidates"].items():
        assert candidate["details"]["sha256"] == DETAIL_SHA256[engine]
        for key, expected_sha256 in CANDIDATE_OUTPUT_SHA256[engine].items():
            assert (
                hashlib.sha256(evaluator._json_bytes(candidate[key], compact=True)).hexdigest()
                == expected_sha256
            )
        assert candidate["content_verdict"] == evaluator.content_verdict(candidate, comparator)
        assert candidate["content_verdict"]["qualified"] is False
        evaluator.validate_candidate_diagnostics(engine, candidate["diagnostics"])
        provenance = candidate["adapter"]["model"]["provenance"]
        assert provenance["license"]["spdx"] == "Apache-2.0"
        assert provenance["lineage"]["archive_build_revision"] is None
        config = candidate["adapter"]["recognizer_config"]
        if engine == "qwen3_asr":
            qwen = config["qwen3_asr"]
            assert {
                key: qwen[key] for key in ("hotwords", "max_new_tokens", "max_total_len", "seed")
            } == {
                "hotwords": "",
                "max_new_tokens": 128,
                "max_total_len": 512,
                "seed": 42,
            }
            assert qwen["temperature"] == pytest.approx(1e-6)
            assert qwen["top_p"] == pytest.approx(0.8)
        else:
            assert config["cohere_transcribe"] == {
                "language": "ja",
                "use_itn": True,
                "use_punct": True,
            }
        measurement = baseline["measurements"]["candidates"][engine]
        assert measurement["resource_verdict"] == evaluator.resource_verdict(
            {
                "sherpa_cpu": {
                    "actual_execution_device": "CPU",
                    "post_warm_rtf": measurement["post_warm"]["median_decode_rtf"],
                    "production_eligible": True,
                }
            }
        )
        assert measurement["resource_verdict"]["qualified"] is False
    assert (
        deterministic["candidates"]["qwen3_asr"]["diagnostics"]["content_event_case_ids"]
        == QWEN_DIAGNOSTIC_CASE_IDS
    )
    assert (
        deterministic["candidates"]["cohere_transcribe"]["diagnostics"]["content_event_case_ids"]
        == []
    )
    assert "compatibility_inputs" in deterministic["pipeline"]
    assert len(deterministic["pipeline"]["evaluator_contract_sha256"]) == 64


def test_ignored_details_hash_and_aggregates_rebuild_when_cache_is_present():
    baseline = _committed_baseline()
    try:
        _, cases = evaluator.load_corpus_cases(verify_pcm=False)
    except RuntimeError:
        pytest.skip("ignored M10 corpus cache absent")

    for group in ("controls", "candidates"):
        for result in baseline["deterministic"][group].values():
            detail = ROOT / result["details"]["path"]
            if not detail.exists():
                pytest.skip("ignored M10 evaluator details absent")
            rows = evaluator._detail_rows(detail, cases, result["details"]["sha256"])
            assert evaluator.aggregate_rows(rows) == result["aggregates"]


# --- M11.3: the ASR contract fingerprint replaces whole-file live_stt.py hashing ---


def _contract_tree(tmp_path: Path, edit=None) -> Path:
    """A minimal ROOT holding the three contract sources, one optionally edited."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in ("live_stt.py", "cer.py", "replay.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        if edit is not None:
            text = edit(name, text)
        (tmp_path / name).write_text(text, encoding="utf-8")
    return tmp_path


def _contract_sha(monkeypatch, root: Path) -> str:
    monkeypatch.setattr(evaluator, "ROOT", root)
    return hashlib.sha256(evaluator.asr_contract_source()).hexdigest()


def _replacing(target: str, old: str, new: str):
    def edit(name: str, text: str) -> str:
        if name != target:
            return text
        assert text.count(old) == 1, f"{target}: {text.count(old)} matches for {old!r}"
        return text.replace(old, new)

    return edit


def test_asr_contract_closure_is_derived_and_every_exclusion_is_declared():
    # Inclusion is derived from the import seeds; exclusion is declared with a reason.
    # A new call into live_stt must land in one set or the other, never silently.
    reachable = evaluator.asr_contract_closure(cut=False)
    contract = evaluator.asr_contract_closure(cut=True)
    cuts = set(evaluator.ASR_CONTRACT_CUTS)

    assert reachable == contract | cuts
    assert not cuts - reachable, "declared cut is no longer reachable"
    assert not contract & cuts
    assert all(evaluator.ASR_CONTRACT_CUTS[name].strip() for name in cuts)


def test_asr_contract_covers_the_decode_reaching_surface():
    contract = evaluator.asr_contract_closure()
    # The audio -> decoder-input path, plus the two constants that whole-file hashing
    # was the only thing covering: neither appears in pipeline_fingerprint()["values"].
    assert {
        "RingBuffer",
        "SEGMENT_QUEUE_MAX",
        "_DECODE_MERGE_MAX_CHARS",
        "_decode",
        "_decode_segments",
        "_feed_segments",
        "_merge_chunk_text",
        "_split_decode_segment",
        "load_recognizer",
        "make_vad",
        "resample",
        "worker",
    } <= contract
    values = evaluator.pipeline_fingerprint()["values"]
    assert "segment_queue_max" not in values
    assert "decode_merge_max_chars" not in values


@pytest.mark.parametrize(
    ("target", "old", "new"),
    [
        # The VAC branch and the whisper constructor are cut, so a sherpa row cannot
        # see them; emit_line is presentation; the translator is outside the closure.
        ("live_stt.py", "async def _vac_segments(", "async def _vac_segments(  # noqa: D103\n"),
        ("live_stt.py", "class WhisperEngine:", "class WhisperEngine:\n    UNUSED = 1\n"),
        ("live_stt.py", "def emit_line(", "def emit_line(  # noqa: D103\n"),
        ("live_stt.py", "TRANSLATE_TIMEOUT_S = 15", "TRANSLATE_TIMEOUT_S = 16"),
    ],
)
def test_irrelevant_live_stt_edits_leave_the_contract_unchanged(
    monkeypatch, tmp_path, target, old, new
):
    base = _contract_sha(monkeypatch, _contract_tree(tmp_path / "base"))
    edited = _contract_sha(
        monkeypatch, _contract_tree(tmp_path / "edited", _replacing(target, old, new))
    )
    assert base == edited


@pytest.mark.parametrize(
    ("target", "old", "new"),
    [
        ("live_stt.py", "VAD_PRE_PAD_S = 0.4", "VAD_PRE_PAD_S = 0.5"),
        ("live_stt.py", "_DECODE_MERGE_MAX_CHARS = ", "_DECODE_MERGE_MAX_CHARS = 1 + "),
        ("live_stt.py", "def resample(", "def resample_renamed("),
        ("cer.py", "def normalize(", "def normalize_renamed("),
        ("replay.py", 'default="k2v2"', 'default="parakeet"'),
    ],
)
def test_decode_input_edits_change_the_contract(monkeypatch, tmp_path, target, old, new):
    base = _contract_sha(monkeypatch, _contract_tree(tmp_path / "base"))
    edited = _contract_sha(
        monkeypatch, _contract_tree(tmp_path / "edited", _replacing(target, old, new))
    )
    assert base != edited


def test_contract_ignores_comments_docstrings_and_reformatting(monkeypatch, tmp_path):
    # polish.md P-003 will run `ruff format` over cer.py; that must not requalify the
    # evidence. Whole-file byte hashing could not tell a rewrap from a decode change.
    def edit(name: str, text: str) -> str:
        if name != "cer.py":
            return text
        return text.replace('"""', '"""Reworded first line.\n\n', 1) + "\n# trailing comment\n"

    base = _contract_sha(monkeypatch, _contract_tree(tmp_path / "base"))
    edited = _contract_sha(monkeypatch, _contract_tree(tmp_path / "edited", edit))
    assert base == edited


def test_committed_pipeline_block_carries_the_contract_and_no_whole_file_live_stt():
    pipeline = _committed_baseline()["deterministic"]["pipeline"]
    assert len(pipeline["asr_contract_sha256"]) == 64
    assert (
        pipeline["asr_contract_sha256"]
        == hashlib.sha256(evaluator.asr_contract_source()).hexdigest()
    )
    assert [row["path"] for row in pipeline["implementation"]] == ["tests/fetch_eval_models.py"]


def test_runtime_provenance_is_recorded_but_never_gates_a_rebuild():
    runtime = evaluator.runtime_fingerprint()
    assert set(runtime["provenance"]) == {"kernel", "uv_lock_sha256"}
    # A host kernel bump and an unrelated uv.lock entry must not refuse a rebuild,
    # while every decode-relevant package version still must.
    moved = copy.deepcopy(runtime)
    moved["provenance"]["kernel"] = "0.0.0-does-not-exist"
    assert evaluator._comparable_runtime(moved) == evaluator._comparable_runtime(runtime)
    # The pre-M11.3 spelling reduces to the same comparable set, so an older baseline
    # needs no migration clause of its own.
    legacy = {**evaluator._comparable_runtime(runtime), "kernel": "7.1.3-1-default"}
    assert evaluator._comparable_runtime(legacy) == evaluator._comparable_runtime(runtime)
    packaged = copy.deepcopy(runtime)
    packaged["packages"]["sherpa-onnx"] = "0.0.0"
    assert evaluator._comparable_runtime(packaged) != evaluator._comparable_runtime(runtime)


def test_the_one_time_migration_is_pinned_to_exactly_one_prior_fingerprint():
    current = evaluator.pipeline_fingerprint()
    retired = [
        {"bytes": 1, "path": path, "sha256": sha}
        for path, sha in sorted(evaluator._M11_3_RETIRED_IMPLEMENTATION.items())
    ]
    previous = {
        **{key: value for key, value in current.items() if key != "asr_contract_sha256"},
        "evaluator_contract_sha256": evaluator._M11_3_RETIRED_EVALUATOR_CONTRACT,
        "implementation": [*retired, *current["implementation"]],
    }
    assert evaluator._pipeline_reaggregate_compatible(previous, current)

    # One wrong retired hash refuses: the clause cannot waive a second transition, and
    # it cannot fire once the committed baseline already carries the contract.
    wrong = copy.deepcopy(previous)
    wrong["implementation"][0]["sha256"] = "0" * 64
    assert not evaluator._pipeline_reaggregate_compatible(wrong, current)
    assert not evaluator._pipeline_reaggregate_compatible(current, {**current, "x": 1})
