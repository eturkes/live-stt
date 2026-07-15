#!/usr/bin/env python3
"""Build + score the genuine long-form Japanese corpus (M9.6).

Source = chapter 1 of LibriVox's CC0 recording of 新美南吉「ごん狐」. The
recording is the exact 64 kb/s MP3 used by Kokoro-Speech v1.3; its published
Kokoro-Align sample coordinates remove the title/LibriVox announcement without
splicing the narration. The reference is section 一 from Aozora Bunko's pinned
Shift-JIS text, with ruby + editor notes removed and the spoken「章おわり」added.

Every remote input is SHA-256 pinned. The 62 selected alignment rows are
sample-contiguous, so the resulting 4:48 WAV is one untouched real narration
span plus 0.4/0.6 s of source-native lead/tail. It lands in the established
gitignored replay cache; only the self-substantiating provenance/results table
is committed.

Requires both model sets plus transient libsndfile bindings for MP3 decode:

    uv run --with soundfile python tests/eval_long_form.py
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf  # pyright: ignore[reportMissingImports]  (uv run --with soundfile)

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

from build_stressor import (  # noqa: E402
    CACHE,
    ENGINES,
    sha256_file,
    vad_segments,
    write_wav,
)
from eval_cer import score_wav  # noqa: E402

from cer import align, normalize  # noqa: E402
from live_stt import (  # noqa: E402
    DECODE_SPLIT_TRIGGER_S,
    SAMPLE_RATE,
    VAD_PRE_PAD_S,
    check_models,
    resample,
)

MANIFEST = TESTS / "long_form.json"
WAV = CACHE / "gongitsune_01.wav"

ALIGNMENT_URL = (
    "https://github.com/kaiidams/Kokoro-Speech-Dataset/releases/download/1.3/"
    "kokoro-speech-v1_3.zip"
)
ALIGNMENT_SHA256 = "5a4a290672016ebe70372ed3d47063f846d86e2b96aa9e9b9d35161670d9f666"
ALIGNMENT_MEMBER = "gongitsune-by-nankichi-niimi.metadata.txt"

AUDIO_URL = (
    "https://archive.org/download/gongitsune_um_librivox/"
    "gongitsune_01_niimi_64kb.mp3"
)
AUDIO_SHA256 = "f2dd16a2e9400d54819f0967ccd77a7948b1437dcf739d702f792a3e933fc141"
AUDIO_FILE = "gongitsune_01_niimi_64kb.mp3"
AUDIO_SAMPLE_RATE = 22_050

TEXT_URL = "https://www.aozora.gr.jp/cards/000121/files/628_ruby_649.zip"
TEXT_SHA256 = "2f214158ddc83a89c88400c8ca63cb0d6add8b625b3cea0599a26fdc19274189"
TEXT_MEMBER = "gongitsune.txt"

FIRST_ROW = 2
LAST_ROW = 63
LEAD_S = 0.4
TAIL_S = 0.6
SECTION_1 = "［＃７字下げ］一［＃「一」は中見出し］"
SECTION_2 = "［＃７字下げ］二［＃「二」は中見出し］"
USER_AGENT = "live-stt-long-form-evaluator/1"


def download(url: str, expected_sha256: str) -> bytes:
    """Fetch one pinned source; refuse silent upstream replacement."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"source hash mismatch: {url}\nexpected {expected_sha256}\nactual   {actual}"
        )
    print(f"fetched {len(data):,} bytes: {url}")
    return data


def zip_member(archive: bytes, member: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        return zf.read(member)


def alignment_span(raw: bytes) -> tuple[int, int, str]:
    """Return aligned [start,end) samples + its automatic transcript.

    Row 1 is the separately spoken title/author. Rows 2..63 are chapter 一,
    including its heading and spoken chapter-end marker. Requiring exact sample
    adjacency proves that cropping the outer bounds preserves the source's
    continuous narration rather than concatenating dataset clips.
    """
    selected: list[tuple[int, str, int, int, str]] = []
    for line in raw.decode("utf-8").splitlines():
        clip_id, audio_file, start, end, text, _reading = line.split("|")
        row = int(clip_id.rsplit("-", 1)[1])
        if FIRST_ROW <= row <= LAST_ROW:
            selected.append((row, audio_file, int(start), int(end), text))

    expected = list(range(FIRST_ROW, LAST_ROW + 1))
    if [row[0] for row in selected] != expected:
        raise ValueError("Kokoro alignment row range is incomplete or reordered")
    if any(row[1] != AUDIO_FILE for row in selected):
        raise ValueError("Kokoro alignment row moved to an unexpected audio file")
    if any(
        left[3] != right[2]
        for left, right in zip(selected, selected[1:], strict=False)
    ):
        raise ValueError("Kokoro alignment rows are no longer sample-contiguous")
    if not selected[0][4].startswith("一 ") or not selected[-1][4].endswith("章 おわり"):
        raise ValueError("Kokoro alignment no longer brackets chapter 一")
    return selected[0][2], selected[-1][3], " ".join(row[4] for row in selected)


def aozora_reference(raw: bytes) -> str:
    """Extract clean section 一 surface text from Aozora's ruby source."""
    text = raw.decode("shift_jis")
    # SECTION_1 appears once in Aozora's notation legend and once as the real
    # heading. Taking the last occurrence avoids admitting the legend/divider.
    starts = [match.start() for match in re.finditer(re.escape(SECTION_1), text)]
    if len(starts) != 2:
        raise ValueError(f"expected legend + section 一 markers, got {len(starts)}")
    start = starts[-1] + len(SECTION_1)
    end = text.find(SECTION_2, start)
    if end < 0:
        raise ValueError("section 二 marker missing from Aozora source")
    body = re.sub(r"《[^》]*》|［＃[^］]*］", "", text[start:end])
    body = re.sub(r"\s+", " ", body).strip()
    return f"一 {body} 章おわり"


def alignment_check(ref: str, automatic_text: str) -> dict:
    """Quantify clean-Aozora vs automatic-alignment surface disagreement."""
    ref_norm = normalize(ref)
    s, d, ins = align(ref_norm, normalize(automatic_text))
    row = {
        "N": len(ref_norm),
        "S": s,
        "D": d,
        "I": ins,
        "cer": (s + d + ins) / len(ref_norm),
    }
    # Kokoro's automatic text mixes surface kanji and phonetic substitutions;
    # this is a same-material sanity check, not the recognizer's reference.
    if row["cer"] > 0.10:
        raise ValueError(f"Aozora/alignment text mismatch too large: {row['cer']:.1%}")
    return row


def decode_crop(mp3: bytes, aligned_start: int, aligned_end: int) -> np.ndarray:
    audio, sample_rate = sf.read(io.BytesIO(mp3), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1).astype(np.float32)
    if sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError(f"unexpected source sample rate: {sample_rate}")
    start = aligned_start - round(LEAD_S * sample_rate)
    end = aligned_end + round(TAIL_S * sample_rate)
    if start < 0 or end > len(audio) or aligned_start >= aligned_end:
        raise ValueError("alignment crop falls outside decoded source audio")
    crop = np.ascontiguousarray(audio[start:end], dtype=np.float32)
    return resample(crop, sample_rate, SAMPLE_RATE)


def main() -> None:
    for engine in ENGINES:
        err = check_models(engine)
        if err:
            print(f"error: engine {engine}: {err.splitlines()[0]}", file=sys.stderr)
            sys.exit(1)

    alignment_zip = download(ALIGNMENT_URL, ALIGNMENT_SHA256)
    text_zip = download(TEXT_URL, TEXT_SHA256)
    mp3 = download(AUDIO_URL, AUDIO_SHA256)

    aligned_start, aligned_end, automatic_text = alignment_span(
        zip_member(alignment_zip, ALIGNMENT_MEMBER)
    )
    ref = aozora_reference(zip_member(text_zip, TEXT_MEMBER))
    ref_check = alignment_check(ref, automatic_text)
    audio = decode_crop(mp3, aligned_start, aligned_end)

    CACHE.mkdir(parents=True, exist_ok=True)
    write_wav(WAV, audio)
    segment_lengths = [length for _, length in vad_segments(audio)]
    durations = [round(length / SAMPLE_RATE, 3) for length in segment_lengths]
    prepad = round(VAD_PRE_PAD_S * SAMPLE_RATE)
    split_trigger = round(DECODE_SPLIT_TRIGGER_S * SAMPLE_RATE)
    max_resliced_s = round((max(segment_lengths, default=0) + prepad) / SAMPLE_RATE, 3)

    scores: dict[str, dict] = {}
    for engine in ENGINES:
        scores[engine] = score_wav(ref, WAV, engine)
        row = scores[engine]
        if not row["hyp"]:
            raise RuntimeError(f"{engine} produced an empty long-form transcript")
        print(
            f"{engine}: D={row['D']}/{row['N']} CER={row['cer']:.1%} "
            f"hyp_chars={len(normalize(row['hyp']))}"
        )

    manifest = {
        "source": {
            "id": "gongitsune_01",
            "title": "ごん狐",
            "author": "新美南吉",
            "chapter": "一",
            "reader": "ekzemplaro",
            "license": "CC0-1.0 (LibriVox/Internet Archive recording)",
            "librivox": "https://librivox.org/gongitsune-by-nankichi-niimi/",
            "archive_item": "https://archive.org/details/gongitsune_um_librivox",
            "audio": {"url": AUDIO_URL, "sha256": AUDIO_SHA256},
            "alignment": {
                "url": ALIGNMENT_URL,
                "sha256": ALIGNMENT_SHA256,
                "release": "Kokoro-Speech v1.3 / Kokoro-Align",
                "member": ALIGNMENT_MEMBER,
            },
            "text": {
                "url": TEXT_URL,
                "sha256": TEXT_SHA256,
                "card": "https://www.aozora.gr.jp/cards/000121/card628.html",
                "member": TEXT_MEMBER,
            },
        },
        "build": {
            "source_rate": AUDIO_SAMPLE_RATE,
            "aligned_start": aligned_start,
            "aligned_end": aligned_end,
            "first_row": FIRST_ROW,
            "last_row": LAST_ROW,
            "row_count": LAST_ROW - FIRST_ROW + 1,
            "lead_s": LEAD_S,
            "tail_s": TAIL_S,
            "wav": str(WAV.relative_to(ROOT)),
            "wav_sha256": sha256_file(WAV),
            "audio_s": round(len(audio) / SAMPLE_RATE, 3),
        },
        "reference": {
            "text": ref,
            "normalized_sha256": hashlib.sha256(normalize(ref).encode()).hexdigest(),
            "kokoro_alignment_text": automatic_text,
            "alignment_check": ref_check,
        },
        "vad": {
            "segment_durations_s": durations,
            "n_segments": len(durations),
            "max_segment_s": max(durations, default=0.0),
            "prepad_s": VAD_PRE_PAD_S,
            "max_resliced_upper_bound_s": max_resliced_s,
            "decode_split_trigger_s": DECODE_SPLIT_TRIGGER_S,
            "decode_split_candidates_upper_bound": sum(
                length + prepad > split_trigger for length in segment_lengths
            ),
        },
        "scores": scores,
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"VAD: {len(durations)} natural segments; max raw={max(durations):.3f}s; "
        f"resliced upper bound={max_resliced_s:.3f}s"
    )
    print(f"wrote {WAV.relative_to(ROOT)} and {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
