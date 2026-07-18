#!/usr/bin/env python3
"""Acquire M10.5's pinned Nemotron streaming archives into ignored ``models/``.

Each installed file is locked by archive + file size/SHA-256. Extraction streams
into a staging directory, accepts only regular members below the declared archive
root, installs only declared runtime and smoke artifacts, and publishes the directory
after the full archive has been checked. Existing invalid directories are preserved
for inspection and fail closed.

The optional architecture smoke loads one forced-Japanese online transducer and feeds
the archive's pinned real Japanese clip in 20 ms blocks.

Run from the repository root:

    UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python tests/fetch_eval_streaming.py
    UV_PROJECT_ENVIRONMENT=.venv uv run --no-sync python tests/fetch_eval_streaming.py --smoke 560ms
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
    chunk_ms: int
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


@dataclass(frozen=True)
class SmokeResult:
    variant: str
    audio: str
    text: str
    eof_count: int
    finalization_count: int


SMOKE_AUDIO = "test_wavs/ja.wav"

CANDIDATE_SPECS = {
    "80ms": CandidateSpec(
        model_id="nemotron_streaming_80ms",
        chunk_ms=80,
        directory="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11",
        archive="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11.tar.bz2",
        archive_size=475_274_007,
        archive_sha256="fb170128c496db33a1fb9f5f9f823257f42f911224ee218bb429f3c2eaf90a8d",
        archive_published_at="2026-07-09T03:39:17Z",
        archive_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-80ms-int8-2026-06-11.tar.bz2"
        ),
        docs_url="https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html",
        release_pr="https://github.com/k2-fsa/sherpa-onnx/pull/3671",
        source_model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        source_revision="f3d333391852ba876df169dcc9ba902d25b6ab0b",
        source_url="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
        license="OpenMDW-1.1",
        license_evidence_url=(
            "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/"
            "f3d333391852ba876df169dcc9ba902d25b6ab0b/README.md"
        ),
        conversion_url=(
            "https://github.com/k2-fsa/sherpa-onnx/commit/b74c4dfee9b446c00eb88cf1364d61e26ee9b531"
        ),
        conversion_role="release_packaging",
        lineage_note=(
            "Archive README names the upstream model and chunk size; PR #3671 adds "
            "the exporter and release workflow, but exact upstream-model and "
            "archive-build/export revisions are unpublished. Installed artifacts "
            "contain no license file."
        ),
        artifacts=(
            ArtifactSpec(
                "README.md",
                213,
                "b509192985e57e967a9e2d5d1214e6067f74fbcf900be46d792ce959e0009c81",
            ),
            ArtifactSpec(
                "encoder.int8.onnx",
                657_601_516,
                "411e1222810f4a4cf0a3704c7609597a12def5b4ad2c7347a24ccd40d895484d",
            ),
            ArtifactSpec(
                "decoder.int8.onnx",
                14_978_075,
                "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0",
            ),
            ArtifactSpec(
                "joiner.int8.onnx",
                9_504_438,
                "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce",
            ),
            ArtifactSpec(
                "tokens.txt",
                131_440,
                "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a",
            ),
            ArtifactSpec(
                SMOKE_AUDIO,
                719_916,
                "780f95a86ba6cc33a4431fcafeacd213417dfa0a6613f93e4400c18f4dd467b0",
            ),
        ),
    ),
    "160ms": CandidateSpec(
        model_id="nemotron_streaming_160ms",
        chunk_ms=160,
        directory="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11",
        archive="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11.tar.bz2",
        archive_size=475_273_363,
        archive_sha256="a81909a1780d84cff16d73c15e13e67d9d81d8839faf14870d507d8499f7a61a",
        archive_published_at="2026-07-09T03:37:49Z",
        archive_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11.tar.bz2"
        ),
        docs_url="https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html",
        release_pr="https://github.com/k2-fsa/sherpa-onnx/pull/3671",
        source_model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        source_revision="f3d333391852ba876df169dcc9ba902d25b6ab0b",
        source_url="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
        license="OpenMDW-1.1",
        license_evidence_url=(
            "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/"
            "f3d333391852ba876df169dcc9ba902d25b6ab0b/README.md"
        ),
        conversion_url=(
            "https://github.com/k2-fsa/sherpa-onnx/commit/b74c4dfee9b446c00eb88cf1364d61e26ee9b531"
        ),
        conversion_role="release_packaging",
        lineage_note=(
            "Archive README names the upstream model and chunk size; PR #3671 adds "
            "the exporter and release workflow, but exact upstream-model and "
            "archive-build/export revisions are unpublished. Installed artifacts "
            "contain no license file."
        ),
        artifacts=(
            ArtifactSpec(
                "README.md",
                214,
                "8fb4862bd09efe1745ff8ef93c13ff3d0bd7081347a3578c738d70615267cdb8",
            ),
            ArtifactSpec(
                "encoder.int8.onnx",
                657_601_518,
                "e1b39e5e16bef578a54ed2fba5f031438e000cc36c3ea2ca49d55699d5baebd4",
            ),
            ArtifactSpec(
                "decoder.int8.onnx",
                14_978_075,
                "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0",
            ),
            ArtifactSpec(
                "joiner.int8.onnx",
                9_504_438,
                "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce",
            ),
            ArtifactSpec(
                "tokens.txt",
                131_440,
                "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a",
            ),
            ArtifactSpec(
                SMOKE_AUDIO,
                719_916,
                "780f95a86ba6cc33a4431fcafeacd213417dfa0a6613f93e4400c18f4dd467b0",
            ),
        ),
    ),
    "560ms": CandidateSpec(
        model_id="nemotron_streaming_560ms",
        chunk_ms=560,
        directory="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        archive="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2",
        archive_size=475_271_763,
        archive_sha256="c6bf5e0df765f9d5b43bc9e0536d4b4b3e7d40bdf5ecf13e45f134c51c05ae3a",
        archive_published_at="2026-07-09T03:39:16Z",
        archive_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11.tar.bz2"
        ),
        docs_url="https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html",
        release_pr="https://github.com/k2-fsa/sherpa-onnx/pull/3671",
        source_model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        source_revision="f3d333391852ba876df169dcc9ba902d25b6ab0b",
        source_url="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
        license="OpenMDW-1.1",
        license_evidence_url=(
            "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/"
            "f3d333391852ba876df169dcc9ba902d25b6ab0b/README.md"
        ),
        conversion_url=(
            "https://github.com/k2-fsa/sherpa-onnx/commit/b74c4dfee9b446c00eb88cf1364d61e26ee9b531"
        ),
        conversion_role="release_packaging",
        lineage_note=(
            "Archive README names the upstream model and chunk size; PR #3671 adds "
            "the exporter and release workflow, but exact upstream-model and "
            "archive-build/export revisions are unpublished. Installed artifacts "
            "contain no license file."
        ),
        artifacts=(
            ArtifactSpec(
                "README.md",
                214,
                "4cec75ccd38f289f3bd39055bd7033bfcbaa145d38b85b31e3943b8f03ae86f1",
            ),
            ArtifactSpec(
                "encoder.int8.onnx",
                657_601_403,
                "012e9321373af99021415e0b0eb3ec827b4be3153be6f30d9b448fe65e896e68",
            ),
            ArtifactSpec(
                "decoder.int8.onnx",
                14_978_075,
                "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0",
            ),
            ArtifactSpec(
                "joiner.int8.onnx",
                9_504_438,
                "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce",
            ),
            ArtifactSpec(
                "tokens.txt",
                131_440,
                "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a",
            ),
            ArtifactSpec(
                SMOKE_AUDIO,
                719_916,
                "780f95a86ba6cc33a4431fcafeacd213417dfa0a6613f93e4400c18f4dd467b0",
            ),
        ),
    ),
    "1120ms": CandidateSpec(
        model_id="nemotron_streaming_1120ms",
        chunk_ms=1120,
        directory="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11",
        archive="sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2",
        archive_size=475_276_334,
        archive_sha256="adbdd5e9fef87300c37cebfcfc4f1ebe56845c860c8a760af0a1dd65ce9beed3",
        archive_published_at="2026-07-09T03:36:55Z",
        archive_url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
            "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2"
        ),
        docs_url="https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html",
        release_pr="https://github.com/k2-fsa/sherpa-onnx/pull/3671",
        source_model="nvidia/nemotron-3.5-asr-streaming-0.6b",
        source_revision="f3d333391852ba876df169dcc9ba902d25b6ab0b",
        source_url="https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b",
        license="OpenMDW-1.1",
        license_evidence_url=(
            "https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/"
            "f3d333391852ba876df169dcc9ba902d25b6ab0b/README.md"
        ),
        conversion_url=(
            "https://github.com/k2-fsa/sherpa-onnx/commit/b74c4dfee9b446c00eb88cf1364d61e26ee9b531"
        ),
        conversion_role="release_packaging",
        lineage_note=(
            "Archive README names the upstream model and chunk size; PR #3671 adds "
            "the exporter and release workflow, but exact upstream-model and "
            "archive-build/export revisions are unpublished. Installed artifacts "
            "contain no license file."
        ),
        artifacts=(
            ArtifactSpec(
                "README.md",
                215,
                "31a0ca29d86abbe7728a26544afd2f0b3980b76ee02975fd140680f1555e88a9",
            ),
            ArtifactSpec(
                "encoder.int8.onnx",
                657_601_521,
                "2fff2166acaa535bd969fb223c1f0783d71029f143cb298bc54c2afe85abf772",
            ),
            ArtifactSpec(
                "decoder.int8.onnx",
                14_978_075,
                "19f9c98fc6d0a2c33a65a43b36fdb2e914c26c0aa9764be3aebc502a1e982fb0",
            ),
            ArtifactSpec(
                "joiner.int8.onnx",
                9_504_438,
                "4101c7c679a0bc30483794b27a059e34e79232aa2068d78d51231a22c8b0d7ce",
            ),
            ArtifactSpec(
                "tokens.txt",
                131_440,
                "729cc103155bafa785f9cd45746cd41cabe97eab7182fc04d594129587958f8a",
            ),
            ArtifactSpec(
                SMOKE_AUDIO,
                719_916,
                "780f95a86ba6cc33a4431fcafeacd213417dfa0a6613f93e4400c18f4dd467b0",
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
                "installed runtime and smoke artifacts contain no license file"
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


def _has_cjk(text: str) -> bool:
    ranges = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF))
    return any(start <= ord(character) <= end for character in text for start, end in ranges)


def smoke_candidate(spec: CandidateSpec, *, models: Path = MODELS) -> SmokeResult:
    """Run one forced-Japanese online-transducer smoke on the pinned archive clip."""
    import wave

    import numpy as np
    import sherpa_onnx

    from live_stt import SAMPLE_RATE, resample

    target = validate_installed(spec, models)
    audio_path = target / SMOKE_AUDIO
    with wave.open(str(audio_path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        compression = source.getcomptype()
        raw = source.readframes(frame_count)
    if compression != "NONE" or sample_width != 2 or channels < 1:
        raise RuntimeError(
            f"{spec.model_id}: smoke WAV must be uncompressed 16-bit PCM with channels"
        )

    pcm = np.frombuffer(raw, dtype="<i2")
    if pcm.size != frame_count * channels:
        raise RuntimeError(f"{spec.model_id}: smoke WAV payload is truncated")
    samples = pcm.astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    samples /= 32768.0
    if sample_rate != SAMPLE_RATE:
        samples = resample(samples, sample_rate, SAMPLE_RATE).copy()
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError(f"{spec.model_id}: smoke WAV contains no samples")

    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(target / "tokens.txt"),
            encoder=str(target / "encoder.int8.onnx"),
            decoder=str(target / "decoder.int8.onnx"),
            joiner=str(target / "joiner.int8.onnx"),
            provider="cpu",
            decoding_method="greedy_search",
            modeling_unit="cjkchar",
            num_threads=4,
            enable_endpoint_detection=False,
        )
        stream = recognizer.create_stream()
        stream.set_option("language", "ja")
    except Exception as exc:
        version = getattr(sherpa_onnx, "__version__", "unknown")
        raise RuntimeError(
            f"{spec.model_id}: sherpa_onnx {version} forced-ja load failed: {exc}"
        ) from exc

    block_samples = SAMPLE_RATE // 50
    for start in range(0, samples.size, block_samples):
        stream.accept_waveform(SAMPLE_RATE, samples[start : start + block_samples])
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

    eof_count = 0
    stream.input_finished()
    eof_count += 1
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)

    finalization_count = 0
    text = recognizer.get_result(stream)
    finalization_count += 1
    if not text or not _has_cjk(text):
        raise RuntimeError(f"{spec.model_id}: forced-ja smoke produced no Japanese text: {text!r}")
    if eof_count != 1 or finalization_count != 1:
        raise RuntimeError(
            f"{spec.model_id}: EOF/finalization accounting mismatch: "
            f"EOF={eof_count}, finalizations={finalization_count}"
        )
    return SmokeResult(
        variant=spec.model_id,
        audio=audio_path.relative_to(ROOT).as_posix(),
        text=text,
        eof_count=eof_count,
        finalization_count=finalization_count,
    )


def _selected(values: Iterable[str]) -> list[CandidateSpec]:
    names = list(values)
    if not names or names == ["all"]:
        return [CANDIDATE_SPECS[name] for name in CANDIDATE_SPECS]
    return [CANDIDATE_SPECS[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "variants",
        nargs="*",
        choices=[*CANDIDATE_SPECS, "all"],
        help="Streaming variants to acquire (default: all, or the smoke variant).",
    )
    parser.add_argument(
        "--smoke",
        choices=[*CANDIDATE_SPECS],
        metavar="VARIANT",
        help="Run the forced-ja architecture smoke for one installed variant.",
    )
    args = parser.parse_args()
    requested = args.variants or ([args.smoke] if args.smoke else ["all"])
    for spec in _selected(requested):
        install_candidate(spec)
    if args.smoke:
        result = smoke_candidate(CANDIDATE_SPECS[args.smoke])
        print(f"{result.variant}: finalized text: {result.text}")
        print(
            f"{result.variant}: EOF={result.eof_count}, "
            f"finalizations={result.finalization_count}, audio={result.audio}"
        )


if __name__ == "__main__":
    main()
