#!/usr/bin/env python3
"""Acquire M10's pinned offline-candidate archives into ignored ``models/``.

Each installed file is locked by archive + file size/SHA-256. Extraction streams
into a staging directory, accepts only regular members below the declared archive
root, installs only runtime artifacts, and publishes the directory after the full
archive has been checked. Existing invalid directories are preserved for inspection
and fail closed.

Run from the repository root:

    UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python tests/fetch_eval_models.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DOWNLOADS = MODELS / ".archives"
DOWNLOAD_TIMEOUT_S = 90


@dataclass(frozen=True)
class ArtifactSpec:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class CandidateSpec:
    model_id: str
    directory: str
    archive: str
    archive_size: int
    archive_sha256: str
    archive_published_at: str
    archive_url: str
    docs_url: str
    release_pr: str
    source_model: str
    source_revision: str
    source_url: str
    license: str
    license_evidence_url: str
    conversion_url: str
    conversion_role: str
    lineage_note: str
    artifacts: tuple[ArtifactSpec, ...]


CANDIDATE_SPECS = {
    "qwen3_asr": CandidateSpec(
        model_id="qwen3_asr",
        directory="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25",
        archive="sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2",
        archive_size=878_702_423,
        archive_sha256="393f8a14e2f5fb96746aaab342997a40641001fbd5bf9592a080a8329178ee96",
        archive_published_at="2026-04-07T09:52:28Z",
        archive_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2"
        ),
        docs_url="https://k2-fsa.github.io/sherpa/onnx/qwen3-asr/pretrained.html",
        release_pr="https://github.com/k2-fsa/sherpa-onnx/pull/3409",
        source_model="Qwen/Qwen3-ASR-0.6B",
        source_revision="5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
        source_url="https://huggingface.co/Qwen/Qwen3-ASR-0.6B",
        license="Apache-2.0",
        license_evidence_url=(
            "https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/"
            "5eb144179a02acc5e5ba31e748d22b0cf3e303b0/README.md"
        ),
        conversion_url=(
            "https://github.com/Wasser1462/Qwen3-ASR-onnx/tree/"
            "62ba70332b47efa55b79db1d5db18090e44dc7fd"
        ),
        conversion_role="exporter_reference",
        lineage_note=(
            "Archive README names a ModelScope copy and this exporter; "
            "exact upstream-model and export revisions are unpublished."
        ),
        artifacts=(
            ArtifactSpec(
                "README.md",
                328,
                "bbc6dbeb9dce5b4ed0e839057137e9cba4bf05c5797277d36a80e22594414e14",
            ),
            ArtifactSpec(
                "conv_frontend.onnx",
                44_148_281,
                "d22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e",
            ),
            ArtifactSpec(
                "encoder.int8.onnx",
                182_491_662,
                "60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9",
            ),
            ArtifactSpec(
                "decoder.int8.onnx",
                755_914_231,
                "4f6885be5959ae26af3089d38ee7972c5fafbeeb1cf8d5e76eab6d8b61ca5771",
            ),
            ArtifactSpec(
                "tokenizer/merges.txt",
                1_671_853,
                "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
            ),
            ArtifactSpec(
                "tokenizer/tokenizer_config.json",
                12_487,
                "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c",
            ),
            ArtifactSpec(
                "tokenizer/vocab.json",
                2_776_833,
                "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
            ),
        ),
    ),
    "cohere_transcribe": CandidateSpec(
        model_id="cohere_transcribe",
        directory="sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01",
        archive="sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2",
        archive_size=1_699_791_751,
        archive_sha256="bd582588d50685a795dcd2807ab77e11361b8312d96c53884682def45ab4206d",
        archive_published_at="2026-04-02T07:38:13Z",
        archive_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2"
        ),
        docs_url="https://k2-fsa.github.io/sherpa/onnx/cohere_transcribe/pretrained.html",
        release_pr="https://github.com/k2-fsa/sherpa-onnx/pull/3453",
        source_model="CohereLabs/cohere-transcribe-03-2026",
        source_revision="d96e814882d88c982f39018cbf1d7d930c7722d0",
        source_url="https://huggingface.co/CohereLabs/cohere-transcribe-03-2026",
        license="Apache-2.0",
        license_evidence_url=(
            "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/blob/"
            "d96e814882d88c982f39018cbf1d7d930c7722d0/README.md"
        ),
        conversion_url=(
            "https://github.com/k2-fsa/sherpa-onnx/commit/aeeb8910dcc02516398fb71423501c180e645681"
        ),
        conversion_role="release_packaging",
        lineage_note=(
            "Archive README names the upstream model; release packaging downloads converted "
            "artifacts from ModelScope master, so exact source and conversion revisions are "
            "unpublished."
        ),
        artifacts=(
            ArtifactSpec(
                "README.md",
                294,
                "c25c8573c9e60a28a4a220fefafe3dfc664677d9083338fdeeeb8723433d27ea",
            ),
            ArtifactSpec(
                "encoder.int8.onnx",
                3_090_822,
                "cf704f8cfa90e3f0a76f9ffc05998bdf00ba9ae983192c14a85a3a5eb008b367",
            ),
            ArtifactSpec(
                "encoder.int8.onnx.data",
                2_731_503_072,
                "bcf1b7148c8518ae52df1ad2d2fc2b4e89261ea23e6c874eef1d9f55bcbaa4a3",
            ),
            ArtifactSpec(
                "decoder.int8.onnx",
                153_250_705,
                "8372ca6c8ff4db8b916ca3592f5c757a715e691b9edec751ba19b29fc854baf9",
            ),
            ArtifactSpec(
                "tokens.txt",
                207_437,
                "013ede043ae2480e3a9205cc34550d9686100cc682bacc90f702facdfbb93035",
            ),
        ),
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_valid(path: Path, spec: CandidateSpec) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == spec.archive_size
        and file_sha256(path) == spec.archive_sha256
    )


def fetch_archive(spec: CandidateSpec, downloads: Path = DOWNLOADS) -> Path:
    """Return exact release bytes; failed refreshes cannot replace cached bytes."""
    downloads.mkdir(parents=True, exist_ok=True)
    path = downloads / spec.archive
    if _source_valid(path, spec):
        print(f"{spec.model_id}: cached archive verified")
        return path

    part = path.with_name(f"{path.name}.part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(spec.archive_url, headers={"User-Agent": "live-stt-m10/1"})
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as response,
            part.open("xb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        actual = digest.hexdigest()
        if size != spec.archive_size or actual != spec.archive_sha256:
            raise RuntimeError(
                f"{spec.model_id}: archive mismatch; expected "
                f"{spec.archive_size}/{spec.archive_sha256}, got {size}/{actual}"
            )
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    print(f"{spec.model_id}: downloaded archive verified")
    return path


def _safe_artifact_path(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise RuntimeError(f"unsafe artifact path in candidate specification: {relative!r}")
    return path


def _artifact_map(spec: CandidateSpec) -> dict[str, ArtifactSpec]:
    artifacts = {artifact.path: artifact for artifact in spec.artifacts}
    if len(artifacts) != len(spec.artifacts):
        raise RuntimeError(f"{spec.model_id}: duplicate artifact specification")
    for relative in artifacts:
        _safe_artifact_path(relative)
    return artifacts


def validate_installed(spec: CandidateSpec, models: Path = MODELS) -> Path:
    """Return an exact installed candidate directory or raise without mutation."""
    target = models / spec.directory
    if not target.is_dir() or target.is_symlink():
        raise RuntimeError(f"{spec.model_id}: candidate model directory is absent or unsafe")
    expected = _artifact_map(spec)
    observed: set[str] = set()
    for path in target.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"{spec.model_id}: installed model contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"{spec.model_id}: installed model contains a non-regular path")
        relative = path.relative_to(target).as_posix()
        artifact = expected.get(relative)
        if artifact is None:
            raise RuntimeError(f"{spec.model_id}: unexpected installed artifact {relative!r}")
        if path.stat().st_size != artifact.size or file_sha256(path) != artifact.sha256:
            raise RuntimeError(f"{spec.model_id}: installed artifact mismatch: {relative}")
        observed.add(relative)
    missing = expected.keys() - observed
    if missing:
        raise RuntimeError(f"{spec.model_id}: missing installed artifacts: {sorted(missing)}")
    return target


def _copy_member(source, output: Path, artifact: ArtifactSpec) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with output.open("xb") as destination:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    if size != artifact.size or digest.hexdigest() != artifact.sha256:
        raise RuntimeError(f"extracted artifact mismatch: {artifact.path}")


def _extract_candidate(archive_path: Path, spec: CandidateSpec, staging: Path) -> None:
    """Validate the entire stream while extracting only declared runtime files."""
    artifacts = _artifact_map(spec)
    found: set[str] = set()
    members: set[str] = set()
    try:
        archive = tarfile.open(archive_path, "r|bz2")
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(f"{spec.model_id}: cannot open candidate archive: {exc}") from exc
    with archive:
        for member in archive:
            raw = member.name.rstrip("/")
            path = PurePosixPath(raw)
            if (
                not raw
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != raw
                or not path.parts
                or path.parts[0] != spec.directory
            ):
                raise RuntimeError(f"{spec.model_id}: unsafe archive path {member.name!r}")
            if raw in members:
                raise RuntimeError(f"{spec.model_id}: duplicate archive member {raw!r}")
            members.add(raw)
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"{spec.model_id}: non-regular archive member {raw!r}")
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            artifact = artifacts.get(relative)
            if artifact is None:
                continue
            if member.size != artifact.size:
                raise RuntimeError(f"{spec.model_id}: archive artifact size mismatch: {relative}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"{spec.model_id}: cannot read archive artifact: {relative}")
            with source:
                _copy_member(source, staging.joinpath(*PurePosixPath(relative).parts), artifact)
            found.add(relative)
    missing = artifacts.keys() - found
    if missing:
        raise RuntimeError(f"{spec.model_id}: archive lacks artifacts: {sorted(missing)}")


def install_candidate(
    spec: CandidateSpec,
    *,
    models: Path = MODELS,
    downloads: Path = DOWNLOADS,
) -> Path:
    target = models / spec.directory
    if target.exists() or target.is_symlink():
        path = validate_installed(spec, models)
        print(f"{spec.model_id}: installed artifacts verified")
        return path

    archive = fetch_archive(spec, downloads)
    workspace = models / f".{spec.directory}.part"
    staging = workspace / spec.directory
    shutil.rmtree(workspace, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        _extract_candidate(archive, spec, staging)
        validate_installed(spec, workspace)
        staging.replace(target)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
    print(f"{spec.model_id}: installed exact runtime artifacts")
    return target


def provenance(spec: CandidateSpec) -> dict:
    return {
        "archive": {
            "published_at": spec.archive_published_at,
            "release_tag": "asr-models",
            "sha256": spec.archive_sha256,
            "size_bytes": spec.archive_size,
            "url": spec.archive_url,
        },
        "build_reference": {
            "role": spec.conversion_role,
            "url": spec.conversion_url,
        },
        "documentation": spec.docs_url,
        "lineage": {
            "archive_build_revision": None,
            "note": spec.lineage_note,
        },
        "license": {
            "evidence_url": spec.license_evidence_url,
            "scope": (
                "upstream model card at the pinned evidence revision; "
                "installed runtime artifacts contain no license file"
            ),
            "spdx": spec.license,
        },
        "release_pr": spec.release_pr,
        "upstream_model": {
            "id": spec.source_model,
            "license_evidence_revision": spec.source_revision,
            "url": spec.source_url,
        },
    }


def _selected(values: Iterable[str]) -> list[CandidateSpec]:
    names = list(values)
    if not names or names == ["all"]:
        return [CANDIDATE_SPECS[name] for name in CANDIDATE_SPECS]
    return [CANDIDATE_SPECS[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "models",
        nargs="*",
        choices=[*CANDIDATE_SPECS, "all"],
        default=["all"],
        help="Candidate IDs to acquire (default: all).",
    )
    args = parser.parse_args()
    for spec in _selected(args.models):
        install_candidate(spec)


if __name__ == "__main__":
    main()
