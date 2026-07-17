"""Model-free locks for M10 offline-candidate acquisition and provenance."""

from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from tests import fetch_eval_models as acquisition

ROOT = Path(__file__).resolve().parent.parent


def _spec(data: bytes = b"exact-model") -> acquisition.CandidateSpec:
    return acquisition.CandidateSpec(
        model_id="candidate",
        directory="candidate-root",
        archive="candidate.tar.bz2",
        archive_size=0,
        archive_sha256="",
        archive_published_at="2026-01-01T00:00:00Z",
        archive_url="https://example.invalid/candidate.tar.bz2",
        docs_url="https://example.invalid/docs",
        release_pr="https://example.invalid/pr",
        source_model="owner/model",
        source_revision="a" * 40,
        source_url="https://example.invalid/model",
        license="Apache-2.0",
        license_evidence_url="https://example.invalid/license",
        conversion_url="https://example.invalid/conversion",
        conversion_role="exporter_reference",
        lineage_note="Exact archive build revision is unpublished.",
        artifacts=(
            acquisition.ArtifactSpec("model.bin", len(data), hashlib.sha256(data).hexdigest()),
        ),
    )


def _write_archive(path: Path, members: list[tuple[tarfile.TarInfo, bytes]]) -> None:
    path.parent.mkdir(parents=True)
    with tarfile.open(path, "w:bz2") as archive:
        for member, data in members:
            archive.addfile(member, io.BytesIO(data) if member.isfile() else None)


def _file(name: str, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    return member, data


def _pinned_archive(
    tmp_path: Path,
    members: list[tuple[tarfile.TarInfo, bytes]],
    spec: acquisition.CandidateSpec,
) -> tuple[acquisition.CandidateSpec, Path, Path]:
    models = tmp_path / "models"
    downloads = tmp_path / "downloads"
    archive = downloads / spec.archive
    _write_archive(archive, members)
    return (
        replace(
            spec,
            archive_size=archive.stat().st_size,
            archive_sha256=acquisition.file_sha256(archive),
        ),
        models,
        downloads,
    )


def test_candidate_install_streams_full_archive_and_keeps_only_pinned_artifacts(tmp_path):
    data = b"exact-model"
    spec = _spec(data)
    members = [
        _file(f"{spec.directory}/model.bin", data),
        _file(f"{spec.directory}/test_wavs/ignored.wav", b"ignored"),
    ]
    spec, models, downloads = _pinned_archive(tmp_path, members, spec)

    target = acquisition.install_candidate(spec, models=models, downloads=downloads)

    assert (target / "model.bin").read_bytes() == data
    assert sorted(
        path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
    ) == ["model.bin"]
    assert acquisition.validate_installed(spec, models) == target


def test_candidate_install_rejects_internal_artifact_mismatch_without_publishing(tmp_path):
    spec = _spec(b"good")
    members = [_file(f"{spec.directory}/model.bin", b"evil")]
    spec, models, downloads = _pinned_archive(tmp_path, members, spec)

    with pytest.raises(RuntimeError, match="extracted artifact mismatch"):
        acquisition.install_candidate(spec, models=models, downloads=downloads)

    assert not (models / spec.directory).exists()
    assert not (models / f".{spec.directory}.part").exists()


@pytest.mark.parametrize("unsafe", ["candidate-root/../escape", "/absolute"])
def test_candidate_install_rejects_unsafe_trailing_member(tmp_path, unsafe):
    data = b"exact-model"
    spec = _spec(data)
    members = [
        _file(f"{spec.directory}/model.bin", data),
        _file(unsafe, b"escape"),
    ]
    spec, models, downloads = _pinned_archive(tmp_path, members, spec)

    with pytest.raises(RuntimeError, match="unsafe archive path"):
        acquisition.install_candidate(spec, models=models, downloads=downloads)

    assert not (models / spec.directory).exists()


def test_candidate_install_rejects_link_and_preserves_existing_invalid_tree(tmp_path):
    data = b"exact-model"
    spec = _spec(data)
    link = tarfile.TarInfo(f"{spec.directory}/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    members = [_file(f"{spec.directory}/model.bin", data), (link, b"")]
    spec, models, downloads = _pinned_archive(tmp_path, members, spec)

    with pytest.raises(RuntimeError, match="non-regular archive member"):
        acquisition.install_candidate(spec, models=models, downloads=downloads)

    target = models / spec.directory
    target.mkdir(parents=True)
    invalid = target / "model.bin"
    invalid.write_bytes(b"user-corrupt")
    with pytest.raises(RuntimeError, match="installed artifact mismatch"):
        acquisition.install_candidate(spec, models=models, downloads=downloads)
    assert invalid.read_bytes() == b"user-corrupt"


def test_declared_candidate_artifacts_are_licensed_pinned_and_gitignored():
    for spec in acquisition.CANDIDATE_SPECS.values():
        record = acquisition.provenance(spec)
        assert record["license"]["spdx"] == "Apache-2.0"
        assert len(record["upstream_model"]["license_evidence_revision"]) == 40
        assert record["lineage"]["archive_build_revision"] is None
        assert "unpublished" in record["lineage"]["note"]
        assert record["license"]["scope"].endswith("contain no license file")
        assert len(record["archive"]["sha256"]) == 64
        paths = [
            f"models/.archives/{spec.archive}",
            *(f"models/{spec.directory}/{artifact.path}" for artifact in spec.artifacts),
        ]
        for path in paths:
            completed = subprocess.run(
                ["git", "check-ignore", "--quiet", path], cwd=ROOT, check=False
            )
            assert completed.returncode == 0
