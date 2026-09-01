"""M11.3b: the VAC evaluator kernel, proved with no model, no OpenVINO and no NPU.

Every probe drives the real module. The decode seam is a fake callable, and the one
probe that needs a recogniser builds the REAL `WhisperEngine` over a fake
`openvino_genai` injected into `sys.modules` (M11.2's technique).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_models as shared  # noqa: E402
import eval_vac  # noqa: E402

import live_stt  # noqa: E402

# Frozen by M11.3b P1.5: this unit parameterized the closure walker the sherpa contract
# uses, so the sherpa hash is asserted against the value recorded before that edit.
ASR_CONTRACT_SHA256 = "aff253181730b72354ca34042b18d1cc1e54f0b1f7931c1721d599f91c2192ed"
TRANSCRIPT = "甲乙丙丁戊己庚辛壬癸"
ARM_IDS = ("arm-a", "arm-b")
CORPUS_IDS = ("corpus-1", "corpus-2", "corpus-3")


def _corpora(tmp_path: Path) -> tuple[eval_vac.VacCorpus, ...]:
    corpora = []
    for index, corpus_id in enumerate(CORPUS_IDS):
        manifest = tmp_path / f"{corpus_id}.json"
        wav = tmp_path / f"{corpus_id}.wav"
        wav.write_bytes(corpus_id.encode())
        manifest.write_text(
            json.dumps(
                {
                    "probe": {
                        "ja_ref": TRANSCRIPT[: 6 + index],
                        "audio_sha256": shared.file_sha256(wav),
                    }
                }
            ),
            encoding="utf-8",
        )
        corpora.append(
            eval_vac.VacCorpus(corpus_id, manifest, wav, "probe.ja_ref", "probe.audio_sha256")
        )
    return tuple(corpora)


def _cases(tmp_path: Path) -> list[eval_vac.VacCase]:
    """Six rows, arm-major: the kernel's canonical ordering."""
    corpora = _corpora(tmp_path)
    return [
        eval_vac.VacCase(
            arm=eval_vac.VacArm(arm_id, "whisper", "NPU"),
            corpus=corpus,
            wav=corpus.wav,
            reference=shared._load_json(corpus.manifest)["probe"]["ja_ref"],
            duration_samples=16000,
        )
        for arm_id in ARM_IDS
        for corpus in corpora
    ]


def _content(case: eval_vac.VacCase, *, chars: int = 5) -> eval_vac.VacTranscript:
    text = TRANSCRIPT[:chars]
    return eval_vac.VacTranscript(
        hypothesis=text,
        segments=tuple(
            {"n": n, "seg_len": 1600, "start": 1600 * (n - 1), "text": char}
            for n, char in enumerate(text, 1)
        ),
        accepted_samples=case.duration_samples,
    )


def _decoder(calls: list[str], *, fail_at: int | None = None):
    def decode(case: eval_vac.VacCase) -> eval_vac.VacObservation:
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError("simulated interruption")
        calls.append(case.row_id)
        return eval_vac.VacObservation(_content(case), decode_seconds=0.5, wall_seconds=0.75)

    return decode


def _row(tmp_path: Path, **overrides) -> tuple[eval_vac.VacCase, dict]:
    case = _cases(tmp_path)[0]
    row = eval_vac.score_row(case, _content(case))
    row.update(overrides)
    return case, row


# --- P1 contract fingerprint --------------------------------------------------------


def test_vac_roots_are_the_declared_seven():
    assert set(eval_vac.VAC_CONTRACT_ROOTS) == {
        "ASR_DEVICE",
        "OPENVINO_CACHE_DIR",
        "VAC_CHUNK_S",
        "VAC_TRIM_S",
        "WHISPER_ENGINES",
        "WhisperEngine",
        "_vac_segments",
    }
    assert eval_vac.VAC_CONTRACT_MODULES == ("live_stt", "streaming")


def test_vac_closure_reaches_streaming_through_the_import_edge():
    surfaces = {
        name: shared.module_surface(eval_vac.ROOT / f"{name}.py")
        for name in eval_vac.VAC_CONTRACT_MODULES
    }
    closure = shared.contract_closure(
        surfaces,
        [("live_stt", name) for name in eval_vac.VAC_CONTRACT_ROOTS],
        require=True,
    )
    reached = {name for module, name in closure if module == "streaming"}
    assert {"Segment", "StreamingProcessor"} <= reached
    assert {("live_stt", name) for name in eval_vac.VAC_CONTRACT_ROOTS} <= closure


def _module_copy(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    root.mkdir()
    for name in eval_vac.VAC_CONTRACT_MODULES:
        shutil.copy(eval_vac.ROOT / f"{name}.py", root / f"{name}.py")
    return root


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    assert source.count(old) == 1, (old, source.count(old))
    path.write_text(source.replace(old, new), encoding="utf-8")


@pytest.mark.parametrize(
    ("module", "old", "new"),
    [
        ("streaming", "HARD_TRIM_S = 28.0", "HARD_TRIM_S = 27.0"),
        ("live_stt", "VAC_CHUNK_S = 1.0", "VAC_CHUNK_S = 1.5"),
        ("live_stt", "VAC_TRIM_S = 8.0", "VAC_TRIM_S = 9.0"),
    ],
)
def test_a_vac_surface_edit_moves_the_vac_hash(tmp_path, module, old, new):
    root = _module_copy(tmp_path)
    before = eval_vac.vac_contract_sha256(root)
    assert before == eval_vac.vac_contract_sha256()
    _replace_once(root / f"{module}.py", old, new)
    assert eval_vac.vac_contract_sha256(root) != before


def test_a_sherpa_only_edit_holds_the_vac_hash(tmp_path):
    root = _module_copy(tmp_path)
    before = eval_vac.vac_contract_sha256(root)
    _replace_once(root / "live_stt.py", "DECODE_CHUNK_S = 2.0", "DECODE_CHUNK_S = 2.1")
    assert eval_vac.vac_contract_sha256(root) == before


def test_a_comment_edit_is_inert(tmp_path):
    root = _module_copy(tmp_path)
    before = eval_vac.vac_contract_sha256(root)
    path = root / "streaming.py"
    path.write_text(
        f"# structural hashing drops this\n{path.read_text(encoding='utf-8')}", encoding="utf-8"
    )
    assert eval_vac.vac_contract_sha256(root) == before


def test_the_sherpa_contract_is_frozen_across_this_unit():
    assert shared._sha256_bytes(shared.asr_contract_source()) == ASR_CONTRACT_SHA256
    assert len(shared.ASR_CONTRACT_SEEDS) == 20
    assert len(shared.ASR_CONTRACT_CUTS) == 17
    assert len(shared.asr_contract_closure()) == 25
    assert eval_vac.vac_contract_sha256() != ASR_CONTRACT_SHA256


def test_a_missing_root_raises_instead_of_being_dropped():
    surfaces = {"live_stt": shared.module_surface(eval_vac.ROOT / "live_stt.py")}
    with pytest.raises(RuntimeError, match="not a top-level definition"):
        shared.contract_closure(surfaces, [("live_stt", "no_such_symbol")], require=True)
    assert shared.contract_closure(surfaces, [("live_stt", "no_such_symbol")]) == set()


@pytest.mark.parametrize(
    "line",
    ["from streaming import *", "from . import streaming"],
)
def test_an_import_form_that_hides_the_surface_fails_closed(tmp_path, line):
    """R10: a silently-dropped edge is the escape a derived contract exists to stop."""
    root = _module_copy(tmp_path)
    path = root / "live_stt.py"
    path.write_text(f"{line}\n{path.read_text(encoding='utf-8')}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="star import|relative import"):
        eval_vac.vac_contract_sha256(root)


def test_an_aliased_import_resolves_to_its_source_symbol(tmp_path):
    """R09: the closure keys on the exported name, not the local alias."""
    root = _module_copy(tmp_path)
    path = root / "live_stt.py"
    _replace_once(
        path,
        "from streaming import Segment, StreamingProcessor",
        "from streaming import Segment, StreamingProcessor as _SP",
    )
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("StreamingProcessor", "_SP")
        .replace("from streaming import Segment, _SP as _SP", "from streaming import Segment, _SP"),
        encoding="utf-8",
    )
    _replace_once(
        path,
        "from streaming import Segment, _SP",
        "from streaming import Segment, StreamingProcessor as _SP",
    )
    surfaces = {
        name: shared.module_surface(root / f"{name}.py") for name in eval_vac.VAC_CONTRACT_MODULES
    }
    closure = shared.contract_closure(
        surfaces,
        [("live_stt", name) for name in eval_vac.VAC_CONTRACT_ROOTS],
        require=True,
    )
    assert ("streaming", "StreamingProcessor") in closure


def test_a_vac_root_is_included_even_though_the_sherpa_contract_cuts_it():
    """R11: the two contracts are independent; a cut belongs to the sherpa hash alone."""
    cut_roots = set(eval_vac.VAC_CONTRACT_ROOTS) & set(shared.ASR_CONTRACT_CUTS)
    assert cut_roots == set(eval_vac.VAC_CONTRACT_ROOTS)  # every VAC root is a sherpa cut
    surfaces = {
        name: shared.module_surface(eval_vac.ROOT / f"{name}.py")
        for name in eval_vac.VAC_CONTRACT_MODULES
    }
    closure = shared.contract_closure(
        surfaces,
        [("live_stt", name) for name in eval_vac.VAC_CONTRACT_ROOTS],
        require=True,
    )
    assert {("live_stt", name) for name in cut_roots} <= closure
    assert cut_roots.isdisjoint(shared.asr_contract_closure())


def test_cross_module_edges_stop_at_the_surface_set():
    """The sherpa closure stays single-module because streaming is not in its surfaces."""
    surfaces = {"live_stt": shared.module_surface(eval_vac.ROOT / "live_stt.py")}
    closure = shared.contract_closure(
        surfaces,
        [("live_stt", "_vac_segments")],
        require=True,
    )
    assert {module for module, _ in closure} == {"live_stt"}


# --- P2 schema + row identity -------------------------------------------------------


def test_schema_version_is_the_vac_evaluators_own():
    assert eval_vac.VAC_SCHEMA_VERSION == 1
    assert eval_vac.VAC_BASELINE.name == "vac_baseline.json"
    assert eval_vac.VAC_DETAILS_DIR != shared.DETAILS_DIR


def test_row_identity_is_arm_major(tmp_path):
    cases = _cases(tmp_path)
    assert [case.row_id for case in cases] == [
        f"{arm_id}/{corpus_id}" for arm_id in ARM_IDS for corpus_id in CORPUS_IDS
    ]
    eval_vac.validate_case_order(cases)


def test_a_duplicate_row_id_is_rejected(tmp_path):
    cases = _cases(tmp_path)
    with pytest.raises(RuntimeError, match="not unique"):
        eval_vac.validate_case_order([*cases, cases[0]])


def test_interleaved_arms_are_rejected(tmp_path):
    cases = _cases(tmp_path)
    with pytest.raises(RuntimeError, match="arm-major"):
        eval_vac.validate_case_order([cases[0], cases[3], cases[1]])


def test_an_empty_corpus_is_rejected():
    with pytest.raises(RuntimeError, match="cannot be empty"):
        eval_vac.validate_case_order([])


def test_an_empty_corpus_fails_before_touching_the_filesystem(tmp_path):
    detail = tmp_path / "nested" / "arm.jsonl"
    with pytest.raises(RuntimeError, match="cannot be empty"):
        eval_vac.write_detail_resumable(detail, [], _decoder([]))
    assert not detail.parent.exists()


def test_a_scored_row_carries_every_declared_key(tmp_path):
    case, row = _row(tmp_path)
    assert set(row) == eval_vac.DETERMINISTIC_ROW_KEYS
    assert row["row_id"] == case.row_id == "arm-a/corpus-1"
    assert row["N"] == len(shared.normalize(case.reference))
    assert row["cer"] == (row["S"] + row["D"] + row["I"]) / row["N"]
    eval_vac.validate_vac_row(case, row)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"row_id": "arm-b/corpus-1"}, "identity drifted"),
        ({"arm_id": "arm-b"}, "identity drifted"),
        ({"corpus_id": "corpus-2"}, "identity drifted"),
        ({"schema_version": 2}, "schema version drifted"),
        ({"schema_version": True}, "schema version drifted"),
        ({"ref": "別"}, "reference drifted"),
        ({"hyp": 5}, "content types drifted"),
        ({"segments": {}}, "content types drifted"),
        ({"accepted_samples": True}, "sample accounting drifted"),
        ({"accepted_samples": 1.5}, "sample accounting drifted"),
        ({"cer": 0.0}, "does not match its own content"),
        ({"S": 1}, "does not match its own content"),
        ({"hyp": "甲"}, "does not match its own content"),
    ],
)
def test_row_rejection_is_structural(tmp_path, overrides, match):
    case, row = _row(tmp_path, **overrides)
    with pytest.raises(RuntimeError, match=match):
        eval_vac.validate_vac_row(case, row)


def test_a_row_with_a_missing_or_extra_key_is_rejected(tmp_path):
    case, row = _row(tmp_path)
    for mutated in ({**row, "extra": 1}, {k: v for k, v in row.items() if k != "cer"}):
        with pytest.raises(RuntimeError, match="keys drifted"):
            eval_vac.validate_vac_row(case, mutated)
    with pytest.raises(RuntimeError, match="keys drifted"):
        eval_vac.validate_vac_row(case, ["not", "a", "row"])  # type: ignore[arg-type]


def test_segment_shape_is_enforced(tmp_path):
    case, row = _row(tmp_path)
    row["segments"] = [{**row["segments"][0], "n": 7}, *row["segments"][1:]]
    with pytest.raises(RuntimeError, match="segment numbering drifted"):
        eval_vac.validate_vac_row(case, row)
    case, row = _row(tmp_path)
    row["segments"] = [{"n": 1, "text": "甲"}, *row["segments"][1:]]
    with pytest.raises(RuntimeError, match="segment keys drifted"):
        eval_vac.validate_vac_row(case, row)


@pytest.mark.parametrize(
    ("measurement", "match"),
    [
        ({"decode_seconds": True}, "value drifted"),
        ({"decode_seconds": float("nan")}, "value drifted"),
        ({"decode_seconds": float("inf")}, "value drifted"),
        ({"decode_seconds": -0.5}, "value drifted"),
        ({"decode_seconds": "0.5"}, "value drifted"),
        ({"wall_seconds": 0.25}, "timing drifted"),
        ({"row_id": "arm-b/corpus-1"}, "identity drifted"),
    ],
)
def test_measurement_rejection_is_structural(tmp_path, measurement, match):
    case = _cases(tmp_path)[0]
    row = {"decode_seconds": 0.5, "row_id": case.row_id, "wall_seconds": 0.75, **measurement}
    with pytest.raises(RuntimeError, match=match):
        eval_vac.validate_measurement(case, row)


def test_zero_cost_and_equal_wall_are_legal_measurements(tmp_path):
    case = _cases(tmp_path)[0]
    eval_vac.validate_measurement(
        case, {"decode_seconds": -0.0, "row_id": case.row_id, "wall_seconds": 0.0}
    )
    eval_vac.validate_measurement(
        case, {"decode_seconds": 1.0, "row_id": case.row_id, "wall_seconds": 1.0}
    )


def test_measurement_keys_are_exact(tmp_path):
    case = _cases(tmp_path)[0]
    with pytest.raises(RuntimeError, match="keys drifted"):
        eval_vac.validate_measurement(case, {"decode_seconds": 0.5, "row_id": case.row_id})


# --- P3 automatic resume ------------------------------------------------------------


def _run(tmp_path: Path, cases) -> tuple[list[str], dict]:
    calls: list[str] = []
    detail = tmp_path / "staging" / "arm-a.jsonl"
    return calls, eval_vac.write_detail_resumable(detail, cases, _decoder(calls))


def _interrupt(tmp_path: Path, cases, fail_at: int) -> list[str]:
    """Stop mid-corpus, leaving the journals exactly as a crash would."""
    calls: list[str] = []
    detail = tmp_path / "staging" / "arm-a.jsonl"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        eval_vac.write_detail_resumable(detail, cases, _decoder(calls, fail_at=fail_at))
    return calls


def test_a_complete_run_publishes_only_the_full_detail(tmp_path):
    cases = _cases(tmp_path)
    calls, summary = _run(tmp_path, cases)
    detail = tmp_path / "staging" / "arm-a.jsonl"
    part, measurements, state = eval_vac.resume_paths(detail)
    assert calls == [case.row_id for case in cases]
    assert summary["details"]["rows"] == 6
    assert summary["measurement"]["rows_reused_on_resume"] == 0
    assert detail.is_file() and not part.exists()
    assert measurements.is_file() and state.is_file()
    assert shared.file_sha256(detail) == summary["details"]["sha256"]
    assert len(detail.read_text(encoding="utf-8").splitlines()) == 6


def test_a_complete_prior_run_redecodes_nothing(tmp_path):
    cases = _cases(tmp_path)
    _run(tmp_path, cases)
    calls, summary = _run(tmp_path, cases)
    assert calls == []
    assert summary["measurement"]["rows_reused_on_resume"] == 6


def test_an_interrupted_run_resumes_from_its_prefix(tmp_path):
    cases = _cases(tmp_path)
    first = _interrupt(tmp_path, cases, 4)
    assert len(first) == 4
    second, summary = _run(tmp_path, cases)
    assert second == [case.row_id for case in cases[4:]]
    assert summary["measurement"]["rows_reused_on_resume"] == 4


def test_a_crash_between_the_paired_fsyncs_redecodes_exactly_one_row(tmp_path):
    cases = _cases(tmp_path)
    _interrupt(tmp_path, cases, 4)
    detail = tmp_path / "staging" / "arm-a.jsonl"
    _, measurements, _ = eval_vac.resume_paths(detail)
    lines = measurements.read_bytes().splitlines(keepends=True)
    measurements.write_bytes(b"".join(lines[:-1]))  # detail 4 rows, measurements 3
    calls, summary = _run(tmp_path, cases)
    assert calls == [case.row_id for case in cases[3:]]
    assert summary["measurement"]["rows_reused_on_resume"] == 3


def test_a_truncated_final_line_reconciles_to_the_last_complete_row(tmp_path):
    cases = _cases(tmp_path)
    _interrupt(tmp_path, cases, 4)
    detail = tmp_path / "staging" / "arm-a.jsonl"
    part, measurements, _ = eval_vac.resume_paths(detail)
    for path in (part, measurements):
        path.write_bytes(path.read_bytes()[:-1])  # kill both trailing newlines
    calls, _ = _run(tmp_path, cases)
    assert calls == [case.row_id for case in cases[3:]]


def test_an_invalid_row_mid_prefix_truncates_from_there(tmp_path):
    cases = _cases(tmp_path)
    _interrupt(tmp_path, cases, 4)
    detail = tmp_path / "staging" / "arm-a.jsonl"
    part, _, _ = eval_vac.resume_paths(detail)
    lines = part.read_bytes().splitlines(keepends=True)
    corrupted = json.loads(lines[1])
    corrupted["cer"] = 0.0
    lines[1] = shared._json_bytes(corrupted, compact=True)
    part.write_bytes(b"".join(lines))
    calls, _ = _run(tmp_path, cases)
    assert calls == [case.row_id for case in cases[1:]]


@pytest.mark.parametrize("key", ["row_ids", "schema_version", "vac_contract_sha256"])
def test_a_resume_identity_mismatch_resets_every_journal(tmp_path, key):
    cases = _cases(tmp_path)
    _interrupt(tmp_path, cases, 4)
    detail = tmp_path / "staging" / "arm-a.jsonl"
    _, _, state = eval_vac.resume_paths(detail)
    identity = shared._load_json(state)
    identity[key] = "drifted" if key != "row_ids" else ["arm-a/other"]
    state.write_text(json.dumps(identity), encoding="utf-8")
    calls, summary = _run(tmp_path, cases)
    assert calls == [case.row_id for case in cases]
    assert summary["measurement"]["rows_reused_on_resume"] == 0


def test_journals_without_a_state_file_are_discarded(tmp_path):
    cases = _cases(tmp_path)
    _interrupt(tmp_path, cases, 4)
    detail = tmp_path / "staging" / "arm-a.jsonl"
    eval_vac.resume_paths(detail)[2].unlink()
    calls, _ = _run(tmp_path, cases)
    assert calls == [case.row_id for case in cases]


def test_an_immutable_detail_with_a_short_measurement_journal_is_rebuilt(tmp_path):
    cases = _cases(tmp_path)
    _run(tmp_path, cases)
    detail = tmp_path / "staging" / "arm-a.jsonl"
    _, measurements, _ = eval_vac.resume_paths(detail)
    measurements.write_bytes(b"")
    calls, summary = _run(tmp_path, cases)
    assert calls == [case.row_id for case in cases]
    assert summary["measurement"]["rows_reused_on_resume"] == 0


def test_detail_reaches_stable_storage_before_its_measurement(tmp_path, monkeypatch):
    cases = _cases(tmp_path)[:2]
    order: list[int] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        order.append(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(eval_vac.os, "fsync", spy)
    _run(tmp_path, cases)
    detail = tmp_path / "staging" / "arm-a.jsonl"
    _, measurements, _ = eval_vac.resume_paths(detail)
    pairs = [
        (order[index], order[index + 1])
        for index in range(0, len(order) - 1)
        if order[index] == detail.stat().st_ino
    ]
    assert order.count(detail.stat().st_ino) == 2
    assert pairs and all(nxt == measurements.stat().st_ino for _, nxt in pairs)


# --- P4 install + atomic aggregate --------------------------------------------------


def _published(tmp_path: Path, cases) -> tuple[dict, dict, dict, Path, Path]:
    summaries: dict[str, dict] = {}
    staged: dict[str, Path] = {}
    for arm_id in ARM_IDS:
        arm_rows = eval_vac.arm_cases(cases, arm_id)
        detail = tmp_path / "staging" / f"{arm_id}.jsonl"
        summaries[arm_id] = eval_vac.write_detail_resumable(detail, arm_rows, _decoder([]))
        staged[arm_id] = detail
    manifest = eval_vac.build_vac_manifest(cases, summaries, staged)
    baseline = tmp_path / "vac_baseline.json"
    details_dir = tmp_path / "installed"
    eval_vac.install_vac_evidence(
        manifest, summaries, staged, baseline=baseline, details_dir=details_dir
    )
    return manifest, summaries, staged, baseline, details_dir


def test_details_install_content_addressed(tmp_path):
    cases = _cases(tmp_path)
    manifest, summaries, _, baseline, details_dir = _published(tmp_path, cases)
    for arm_id in ARM_IDS:
        sha = summaries[arm_id]["details"]["sha256"]
        installed = details_dir / f"{arm_id}-{sha[:16]}.jsonl"
        assert installed.is_file()
        assert shared.file_sha256(installed) == sha
        assert manifest["deterministic"]["arms"][arm_id]["details"]["path"] == str(installed)
    assert baseline.is_file()


def test_a_content_addressed_collision_raises(tmp_path):
    cases = _cases(tmp_path)
    _, summaries, staged, baseline, details_dir = _published(tmp_path, cases)
    sha = summaries["arm-a"]["details"]["sha256"]
    (details_dir / f"arm-a-{sha[:16]}.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="collision"):
        eval_vac.install_vac_evidence(
            eval_vac.build_vac_manifest(cases, summaries, staged),
            summaries,
            staged,
            baseline=baseline,
            details_dir=details_dir,
        )


def test_reinstalling_identical_bytes_is_a_no_op(tmp_path):
    cases = _cases(tmp_path)
    _, summaries, staged, baseline, details_dir = _published(tmp_path, cases)
    before = {path: path.read_bytes() for path in sorted(details_dir.iterdir())}
    eval_vac.install_vac_evidence(
        eval_vac.build_vac_manifest(cases, summaries, staged),
        summaries,
        staged,
        baseline=baseline,
        details_dir=details_dir,
    )
    assert {path: path.read_bytes() for path in sorted(details_dir.iterdir())} == before


def test_a_staged_detail_that_changed_before_install_raises(tmp_path):
    cases = _cases(tmp_path)
    summaries: dict[str, dict] = {}
    staged: dict[str, Path] = {}
    for arm_id in ARM_IDS:
        detail = tmp_path / "staging" / f"{arm_id}.jsonl"
        summaries[arm_id] = eval_vac.write_detail_resumable(
            detail, eval_vac.arm_cases(cases, arm_id), _decoder([])
        )
        staged[arm_id] = detail
    manifest = eval_vac.build_vac_manifest(cases, summaries, staged)
    staged["arm-b"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="staged detail hash changed"):
        eval_vac.install_vac_evidence(
            manifest,
            summaries,
            staged,
            baseline=tmp_path / "vac_baseline.json",
            details_dir=tmp_path / "installed",
        )


def test_the_aggregate_is_written_last(tmp_path, monkeypatch):
    cases = _cases(tmp_path)
    summaries: dict[str, dict] = {}
    staged: dict[str, Path] = {}
    for arm_id in ARM_IDS:
        detail = tmp_path / "staging" / f"{arm_id}.jsonl"
        summaries[arm_id] = eval_vac.write_detail_resumable(
            detail, eval_vac.arm_cases(cases, arm_id), _decoder([])
        )
        staged[arm_id] = detail
    manifest = eval_vac.build_vac_manifest(cases, summaries, staged)
    baseline = tmp_path / "vac_baseline.json"
    details_dir = tmp_path / "installed"
    real = shared.write_atomic

    def fail_on_baseline(path: Path, chunks):
        if path == baseline:
            raise OSError("no space left on device")
        real(path, chunks)

    monkeypatch.setattr(eval_vac.shared, "write_atomic", fail_on_baseline)
    with pytest.raises(OSError):
        eval_vac.install_vac_evidence(
            manifest, summaries, staged, baseline=baseline, details_dir=details_dir
        )
    assert not baseline.exists()
    assert len(list(details_dir.iterdir())) == len(ARM_IDS)


def test_measurements_are_a_separate_excluded_block(tmp_path):
    cases = _cases(tmp_path)
    manifest, _, _, _, _ = _published(tmp_path, cases)
    assert manifest["measurements"]["excluded_from_deterministic_equality"] is True
    deterministic = json.dumps(manifest["deterministic"])
    for banned in ("wall_seconds", "decode_seconds", "total_wall_s", "overall_rtf"):
        assert banned not in deterministic
    assert set(manifest["measurements"]["arms"]) == set(ARM_IDS)


def test_the_manifest_carries_the_vac_contract_hash(tmp_path):
    cases = _cases(tmp_path)
    manifest, _, _, _, _ = _published(tmp_path, cases)
    pipeline = manifest["deterministic"]["pipeline"]
    assert pipeline["vac_contract_sha256"] == eval_vac.vac_contract_sha256()
    assert pipeline["vac_chunk_s"] == live_stt.VAC_CHUNK_S
    assert pipeline["vac_trim_s"] == live_stt.VAC_TRIM_S
    assert manifest["deterministic"]["row_ids"] == [case.row_id for case in cases]


def test_an_incomplete_arm_set_is_rejected(tmp_path):
    cases = _cases(tmp_path)
    _, summaries, staged, _, _ = _published(tmp_path, cases)
    with pytest.raises(RuntimeError, match="required model set"):
        eval_vac.build_vac_manifest(
            cases, {"arm-a": summaries["arm-a"]}, {"arm-a": staged["arm-a"]}
        )


def test_aggregate_rows_is_micro_over_the_arm(tmp_path):
    cases = _cases(tmp_path)
    rows = [eval_vac.score_row(case, _content(case)) for case in cases[:3]]
    aggregate = eval_vac.aggregate_rows(rows)
    totals = aggregate["totals"]
    assert totals == {key: sum(row[key] for row in rows) for key in ("S", "D", "I", "N")}
    assert aggregate["micro_cer"] == (totals["S"] + totals["D"] + totals["I"]) / totals["N"]
    assert set(aggregate["corpora"]) == set(CORPUS_IDS)
    assert aggregate["rows"] == 3
    with pytest.raises(RuntimeError, match="at least one row"):
        eval_vac.aggregate_rows([])


# --- P5 aggregate-only --------------------------------------------------------------


def _reaggregate(tmp_path, monkeypatch, cases, baseline) -> bytes:
    monkeypatch.setattr(eval_vac, "load_vac_cases", lambda **_: cases)
    eval_vac.reaggregate_parent(baseline=baseline, details_dir=tmp_path / "installed")
    return baseline.read_bytes()


def test_aggregate_only_rebuilds_byte_identically_twice(tmp_path, monkeypatch, capsys):
    cases = _cases(tmp_path)
    _, _, _, baseline, _ = _published(tmp_path, cases)
    published = baseline.read_bytes()
    first = _reaggregate(tmp_path, monkeypatch, cases, baseline)
    # R12: rebuild identity is content, never filesystem metadata.
    for installed in (tmp_path / "installed").iterdir():
        os.utime(installed, (0, 0))
    second = _reaggregate(tmp_path, monkeypatch, cases, baseline)
    capsys.readouterr()
    assert first == second == published


def test_aggregate_only_refuses_a_changed_vac_contract(tmp_path, monkeypatch):
    cases = _cases(tmp_path)
    _, _, _, baseline, _ = _published(tmp_path, cases)
    before = baseline.read_bytes()
    manifest = shared._load_json(baseline)
    manifest["deterministic"]["pipeline"]["vac_contract_sha256"] = "0" * 64
    baseline.write_bytes(shared._json_bytes(manifest))
    monkeypatch.setattr(eval_vac, "load_vac_cases", lambda **_: cases)
    with pytest.raises(RuntimeError, match="VAC contract changed"):
        eval_vac.reaggregate_parent(baseline=baseline, details_dir=tmp_path / "installed")
    assert baseline.read_bytes() != before  # only the deliberate edit above


def test_aggregate_only_refuses_a_changed_row_set(tmp_path, monkeypatch):
    cases = _cases(tmp_path)
    _, _, _, baseline, _ = _published(tmp_path, cases)
    monkeypatch.setattr(eval_vac, "load_vac_cases", lambda **_: cases[:5])
    with pytest.raises(RuntimeError, match="row set changed"):
        eval_vac.reaggregate_parent(baseline=baseline, details_dir=tmp_path / "installed")


def test_aggregate_only_refuses_a_missing_detail(tmp_path, monkeypatch):
    cases = _cases(tmp_path)
    _, _, _, baseline, details_dir = _published(tmp_path, cases)
    before = baseline.read_bytes()
    next(iter(sorted(details_dir.iterdir()))).unlink()
    monkeypatch.setattr(eval_vac, "load_vac_cases", lambda **_: cases)
    with pytest.raises(RuntimeError, match="installed detail is missing"):
        eval_vac.reaggregate_parent(baseline=baseline, details_dir=details_dir)
    assert baseline.read_bytes() == before


def test_aggregate_only_refuses_a_tampered_detail(tmp_path, monkeypatch):
    cases = _cases(tmp_path)
    _, _, _, baseline, details_dir = _published(tmp_path, cases)
    before = baseline.read_bytes()
    installed = sorted(details_dir.iterdir())[0]
    installed.write_text(
        installed.read_text(encoding="utf-8").replace("甲", "乙"), encoding="utf-8"
    )
    monkeypatch.setattr(eval_vac, "load_vac_cases", lambda **_: cases)
    with pytest.raises(RuntimeError, match="installed detail hash changed"):
        eval_vac.reaggregate_parent(baseline=baseline, details_dir=details_dir)
    assert baseline.read_bytes() == before


# --- P6 CLI surface -----------------------------------------------------------------


def _cli(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_vac.py", *argv])
    eval_vac.main()


def test_only_aggregate_only_is_public(capsys, monkeypatch):
    with pytest.raises(SystemExit) as exit_info:
        _cli(monkeypatch, ["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--aggregate-only" in help_text
    for hidden in ("--worker", "--details", "--summary"):
        assert hidden not in help_text
    for absent in ("--out", "--resume", "--cache", "--corpus", "--wav"):
        assert absent not in help_text


@pytest.mark.parametrize(
    "argv",
    [
        ["--worker", "arm-a", "--aggregate-only"],
        ["--worker", "arm-a"],
        ["--worker", "arm-a", "--details", "d.jsonl"],
        ["--details", "d.jsonl"],
        ["--summary", "s.json"],
    ],
)
def test_worker_flags_are_internal_and_validated(monkeypatch, argv):
    with pytest.raises(SystemExit) as exit_info:
        _cli(monkeypatch, argv)
    assert exit_info.value.code == 2


def test_the_default_run_is_the_parent(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(eval_vac, "run_parent", lambda: seen.append("parent"))
    monkeypatch.setattr(eval_vac, "reaggregate_parent", lambda: seen.append("aggregate"))
    _cli(monkeypatch, [])
    _cli(monkeypatch, ["--aggregate-only"])
    assert seen == ["parent", "aggregate"]


def test_the_worker_runs_one_arm(tmp_path, monkeypatch, capsys):
    cases = _cases(tmp_path)
    monkeypatch.setattr(eval_vac, "load_vac_cases", lambda **_: cases)
    monkeypatch.setattr(eval_vac, "arm_decoder", lambda arm: _decoder([]))
    detail = tmp_path / "worker" / "arm-b.jsonl"
    summary = tmp_path / "worker" / "arm-b.summary.json"
    _cli(monkeypatch, ["--worker", "arm-b", "--details", str(detail), "--summary", str(summary)])
    capsys.readouterr()
    rows = [json.loads(line) for line in detail.read_text(encoding="utf-8").splitlines()]
    assert [row["row_id"] for row in rows] == [f"arm-b/{cid}" for cid in CORPUS_IDS]
    assert shared._load_json(summary)["details"]["rows"] == 3


def test_an_unknown_arm_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_vac, "load_vac_cases", lambda **_: _cases(tmp_path))
    with pytest.raises(RuntimeError, match="unknown VAC arm"):
        eval_vac.run_worker("arm-z", tmp_path / "d.jsonl", tmp_path / "s.json")


# --- P7 hardware independence -------------------------------------------------------


class _FakePipeline:
    """The binding's call/return contract, and its one production-relevant refusal."""

    def __init__(self, model_dir: str, device: str, **kwargs):
        self.device = device

    def generate(self, samples, **kwargs):
        raise AssertionError("this probe never decodes")


def test_the_arm_decoder_builds_the_real_engine_through_production_routing(tmp_path, monkeypatch):
    binding = ModuleType("openvino_genai")
    binding.WhisperPipeline = _FakePipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openvino_genai", binding)
    monkeypatch.setattr(live_stt, "OPENVINO_CACHE_DIR", tmp_path / "cache")
    built: list[tuple[object, str]] = []
    monkeypatch.setattr(
        eval_vac.replay,
        "replay_recognizer",
        lambda path, rec, engine: (
            built.append((rec, engine))
            or {
                "audio_s": 2.0,
                "total_decode_s": 0.8,
                "segments": [
                    {
                        "n": 1,
                        "seg_len": 1600,
                        "start": 0,
                        "text": "甲",
                        "decode_s": 0.4,
                        "rtf": 0.2,
                    },
                    {
                        "n": 2,
                        "seg_len": 1600,
                        "start": 1600,
                        "text": "乙",
                        "decode_s": 0.4,
                        "rtf": 0.2,
                    },
                ],
            }
        ),
    )
    case = _cases(tmp_path)[0]
    observation = eval_vac.arm_decoder(case.arm)(case)
    recognizer, engine = built[0]
    assert isinstance(recognizer, live_stt.WhisperEngine)
    assert engine == "whisper"
    assert observation.content.hypothesis == "甲乙"
    assert observation.content.accepted_samples == 2 * live_stt.SAMPLE_RATE
    assert observation.decode_seconds == 0.8
    assert observation.wall_seconds >= 0.0
    # Timing never enters the deterministic row.
    assert all(set(segment) == eval_vac.SEGMENT_KEYS for segment in observation.content.segments)


def test_the_committed_corpora_are_the_two_pinned_manifests():
    assert [corpus.corpus_id for corpus in eval_vac.VAC_CORPORA] == ["long_form", "retention"]
    assert [arm.arm_id for arm in eval_vac.VAC_ARMS] == ["vac_npu"]
    assert eval_vac.VAC_ARMS[0].device == live_stt.ASR_DEVICE == "NPU"
    for corpus in eval_vac.VAC_CORPORA:
        assert corpus.manifest.is_file()


def test_the_pinned_corpora_load_without_audio():
    cases = eval_vac.load_vac_cases(verify_wav=False)
    assert [case.row_id for case in cases] == ["vac_npu/long_form", "vac_npu/retention"]
    assert all(case.reference.strip() for case in cases)


def test_a_manifest_key_path_that_does_not_resolve_raises(tmp_path):
    corpus = _corpora(tmp_path)[0]
    manifest = shared._load_json(corpus.manifest)
    with pytest.raises(RuntimeError, match="key path does not resolve"):
        eval_vac._key_path(manifest, "probe.absent", corpus.corpus_id)
