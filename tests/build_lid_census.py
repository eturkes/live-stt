#!/usr/bin/env python3
"""Regenerate `tests/lid_census.json` from the two committed FLEURS corpora.

The census is the gate's only evidence that the three-part LID rule holds over 1,926
utterances, and it used to be a reduction of gitignored spike output that could not be
rerun from committed state. This script is that reduction re-expressed over inputs the
repo pins, so the fixture survives the loss of `models/lid/results/*.json.gz`.

Contract -- every clause is what byte-identity depends on:

- **Inputs.** The FLEURS half of the two pinned corpora, located and identity-checked
  through `fetch_real_clips`'s `EXPECTED_INDEX_SHA256` / `EN_EXPECTED_INDEX_SHA256`,
  plus the shipped silero VAD and the ECAPA LID weights.
- **Order, which no committed file recorded and which the corpora themselves define.**
  `utterance` is a global index assigned in first-appearance order: languages in
  `LANGUAGES` order, clips `sorted()` by filename inside each corpus, VAD buffers in the
  order the cut produces them. A clip whose audio opens no buffer contributes nothing
  (5 EN clips do this), and the index is dense over what remains.
- **Cut.** Each clip is bracketed by 1 s of digital silence and fed to the shipped
  `make_vad()` in its native windows. A buffer opens at the `is_speech_detected` rising
  edge, re-sliced from a `RingBuffer` by `VAD_PRE_PAD_S`, and closes at the falling edge;
  an open buffer at end of audio is flushed.
- **Views.** Fixed prefixes of `PREFIX_SECONDS`, kept only where the buffer reaches that
  duration, then `VADfin` for the whole buffer.
- **Scores.** The shipped `LanguageDetector.score()` -- the RAW 107-way softmax, never
  renormalized over `{ja, en}`, because renormalizing is the thing the rule rejects.
  Rounded to `ROUND` decimals, which moves no decision.
- **Serialization.** `json.dumps(separators=(",", ":"), sort_keys=False)` plus one
  trailing newline.

The script is self-validating: it rebuilds and compares against the committed fixture, so
a green run IS the byte-identity credit. `--write` is for a deliberate regeneration.

    uv run python tests/build_lid_census.py              # full rebuild + byte compare
    uv run python tests/build_lid_census.py --clips 20   # bounded re-derivation
    uv run python tests/build_lid_census.py --write      # rewrite the fixture
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

import live_stt
from tests import fetch_real_clips as corpus

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "lid_census.json"
SAMPLE_RATE = 16_000
PREFIX_SECONDS = (1, 2, 3, 5, 8)
ROUND = 6
BOUNDARY_SECONDS = 1
# Census order: `utterance` numbers every `ja` buffer before the first `en` one.
LANGUAGES = ("ja", "en")
VIEW_FIELDS = ["spoken", "utterance", "bucket", "argmax", "argmax_score", "ja", "en"]
SYNTHETIC_FIELDS = ["id", "seconds", "argmax", "argmax_score", "ja", "en"]
RUNTIME = "onnxruntime CPUExecutionProvider"
MODEL_NAME = "voxlingua107.onnx"


def corpus_dir(language: str) -> Path:
    """The pinned cache directory holding `language`'s FLEURS WAVs."""
    if language == "ja":
        name = f"short_corpus-v1-{corpus.EXPECTED_INDEX_SHA256[:16]}"
    elif language == "en":
        name = f"{corpus.EN_CACHE_PREFIX}-{corpus.EN_EXPECTED_INDEX_SHA256[:16]}"
    else:
        raise ValueError(f"unsupported language: {language}")
    return corpus.CACHE / name / "fleurs"


def missing_resources() -> list[str]:
    """Absent inputs, named for a skip reason; empty when the rebuild can run."""
    paths = [
        live_stt.VAD_MODEL,
        live_stt.LID_MODEL_DIR / MODEL_NAME,
        live_stt.LID_MODEL_DIR / "lang_map.json",
        *(corpus_dir(language).parent / "index.jsonl" for language in LANGUAGES),
        *(corpus_dir(language) for language in LANGUAGES),
    ]
    return [str(path.relative_to(ROOT)) for path in paths if not path.exists()]


def corpus_clips(language: str, limit: int | None = None) -> list[Path]:
    """FLEURS clips of `language` in census order, after an index-fingerprint check."""
    directory = corpus_dir(language).parent
    expected = corpus.EXPECTED_INDEX_SHA256 if language == "ja" else corpus.EN_EXPECTED_INDEX_SHA256
    entries = corpus.validate_cached_index(directory, expected)
    clips = sorted(directory / entry["wav"] for entry in entries if entry["source"] == "fleurs")
    return clips[:limit]


def read_clip(path: Path) -> np.ndarray:
    """16 kHz mono PCM16 WAV as float32 in [-1, 1]."""
    with wave.open(str(path)) as source:
        if (
            source.getframerate() != SAMPLE_RATE
            or source.getnchannels() != 1
            or source.getsampwidth() != 2
        ):
            raise ValueError(f"{path}: expected 16 kHz mono PCM16 WAV")
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(np.float32)
    return pcm / 32768.0


def vad_buffers(pcm: np.ndarray) -> list[np.ndarray]:
    """The production VAD's speech buffers for one clip, in order."""
    vad, window = live_stt.make_vad()
    ring = live_stt.RingBuffer(live_stt.RING_SECONDS * SAMPLE_RATE)
    pad = round(live_stt.VAD_PRE_PAD_S * SAMPLE_RATE)
    boundary = np.zeros(BOUNDARY_SECONDS * SAMPLE_RATE, dtype=np.float32)
    audio = np.concatenate((boundary, pcm, boundary))
    start: int | None = None
    consumed = 0
    buffers = []

    for offset in range(0, len(audio), window):
        block = audio[offset : offset + window]
        vad.accept_waveform(block)
        while not vad.empty():
            vad.pop()
        ring.append(block)
        consumed += len(block)
        detected = vad.is_speech_detected()
        if detected and start is None:
            start = max(0, consumed - len(block) - pad)
        elif start is not None and not detected:
            buffers.append(ring.slice(start, consumed))
            start = None

    vad.flush()
    while not vad.empty():
        vad.pop()
    if start is not None:
        buffers.append(ring.slice(start, consumed))
    return buffers


def bucket_views(buffer: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """`(bucket, pcm)` for each prefix the buffer reaches, then `VADfin`."""
    views = []
    for seconds in PREFIX_SECONDS:
        samples = seconds * SAMPLE_RATE
        if len(buffer) >= samples:
            views.append((f"{seconds}s", buffer[:samples]))
    views.append(("VADfin", buffer))
    return views


def synthetic_probes(seconds: int) -> list[tuple[str, np.ndarray]]:
    """The five deterministic non-speech probes of one duration."""
    samples = seconds * SAMPLE_RATE
    time = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    probes = [(f"silence-{seconds}s", np.zeros(samples, dtype=np.float32))]
    for seed, db in ((1000 + seconds, -60), (2000 + seconds, -30), (3000 + seconds, -10)):
        rng = np.random.default_rng(seed)
        rms = 10 ** (db / 20)
        noise = (rng.standard_normal(samples) * rms).astype(np.float32)
        probes.append((f"noise-{db}dB-{seconds}s", noise))
    hum = (10 ** (-30 / 20) * np.sin(2 * np.pi * 60 * time)).astype(np.float32)
    probes.append((f"hum-30dB-{seconds}s", hum))
    return probes


def score_row(detector: live_stt.LanguageDetector, pcm: np.ndarray) -> list:
    """`[argmax, argmax_score, ja, en]`, rounded to `ROUND`."""
    argmax, argmax_score, ja, en = detector.score(pcm)
    return [argmax, round(argmax_score, ROUND), round(ja, ROUND), round(en, ROUND)]


def language_views(
    detector: live_stt.LanguageDetector, language: str, limit: int | None = None
) -> list[list]:
    """`views` rows for one language, `utterance` local to that language (0-based)."""
    rows = []
    utterance = 0
    for path in corpus_clips(language, limit):
        for buffer in vad_buffers(read_clip(path)):
            for bucket, pcm in bucket_views(buffer):
                rows.append([language, utterance, bucket, *score_row(detector, pcm)])
            utterance += 1
    return rows


def synthetic_views(detector: live_stt.LanguageDetector) -> list[list]:
    """`synthetic` rows, one per probe, in `PREFIX_SECONDS` x probe order."""
    return [
        [probe_id, seconds, *score_row(detector, pcm)]
        for seconds in PREFIX_SECONDS
        for probe_id, pcm in synthetic_probes(seconds)
    ]


def census(detector: live_stt.LanguageDetector) -> dict:
    """The whole payload, `utterance` global and dense across `LANGUAGES`."""
    views = []
    utterances = 0
    for language in LANGUAGES:
        rows = language_views(detector, language)
        language_utterances = 0 if not rows else rows[-1][1] + 1
        for row in rows:
            row[1] += utterances
        views.extend(rows)
        utterances += language_utterances

    model = live_stt.LID_MODEL_DIR / MODEL_NAME
    with model.open("rb") as source:
        model_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    labels = json.loads((live_stt.LID_MODEL_DIR / "lang_map.json").read_text(encoding="utf-8"))
    return {
        "model": MODEL_NAME,
        "model_sha256": model_sha256,
        "runtime": RUNTIME,
        "labels_total": len(labels),
        "view_fields": VIEW_FIELDS,
        "synthetic_fields": SYNTHETIC_FIELDS,
        "utterances": utterances,
        "views": views,
        "synthetic": synthetic_views(detector),
    }


def render(payload: dict) -> bytes:
    """The committed serialization of a payload."""
    return (json.dumps(payload, separators=(",", ":"), sort_keys=False) + "\n").encode()


def committed_views(language: str) -> list[list]:
    """The fixture's rows for one language, `utterance` re-localized to 0-based.

    Re-localizing is what lets a bounded run compare: a `--clips N` rebuild numbers `en`
    from the utterances it actually cut, never from the full corpus's 1,030.
    """
    payload = json.loads(OUT.read_text(encoding="utf-8"))
    rows = [row for row in payload["views"] if row[0] == language]
    if not rows:
        raise ValueError(f"committed census has no {language} views")
    first = rows[0][1]
    return [[row[0], row[1] - first, *row[2:]] for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--clips",
        type=int,
        metavar="N",
        help="check only the first N clips of each language",
    )
    parser.add_argument("--write", action="store_true", help="rewrite the committed fixture")
    args = parser.parse_args(argv)
    if args.clips is not None and args.clips < 1:
        parser.error("--clips must be at least 1")
    if args.clips is not None and args.write:
        parser.error("--clips and --write are mutually exclusive")

    missing = missing_resources()
    if missing:
        print(f"absent: {', '.join(missing)}", file=sys.stderr)
        return 1

    detector = live_stt.LanguageDetector()
    if args.clips is not None:
        views = 0
        utterances = 0
        for language in LANGUAGES:
            actual = language_views(detector, language, args.clips)
            expected = committed_views(language)[: len(actual)]
            if actual != expected:
                print(
                    f"error: {language} rows from the first {args.clips} clips differ",
                    file=sys.stderr,
                )
                return 1
            views += len(actual)
            utterances += 0 if not actual else actual[-1][1] + 1

        actual_synthetic = synthetic_views(detector)
        expected_synthetic = json.loads(OUT.read_text(encoding="utf-8"))["synthetic"]
        if actual_synthetic != expected_synthetic:
            print("error: synthetic rows differ", file=sys.stderr)
            return 1
        print(
            f"bounded rows match: clips={args.clips}/language "
            f"utterances={utterances} views={views} synthetic={len(actual_synthetic)}"
        )
        return 0

    payload = census(detector)
    raw = render(payload)
    digest = hashlib.sha256(raw).hexdigest()
    relative = OUT.relative_to(ROOT)
    print(f"{relative}: {len(raw)} B sha256={digest}")
    print(
        f"utterances={payload['utterances']} views={len(payload['views'])} "
        f"synthetic={len(payload['synthetic'])}"
    )
    if args.write:
        OUT.write_bytes(raw)
        print("wrote regenerated fixture")
        return 0
    if not OUT.is_file() or OUT.read_bytes() != raw:
        print("error: regenerated fixture is not byte-identical", file=sys.stderr)
        return 1
    print("byte-identical to committed fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
