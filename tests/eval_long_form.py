#!/usr/bin/env python3
"""Build the pinned long-form Japanese corpus: all six sections of 「ごん狐」 (M9.6, M12.2).

Source = LibriVox's CC0 reading of 新美南吉「ごん狐」, one 64 kb/s MP3 per section --
the exact encode Kokoro-Speech v1.3 aligned, so its published Kokoro-Align sample
coordinates crop each LibriVox announcement off without splicing the narration.
Each reference is the matching section of Aozora Bunko's pinned Shift-JIS text,
with ruby + editor notes removed.

Every remote input is SHA-256 pinned and cached under the ignored replay cache,
verified on the cached path as well as the fresh one (L-017). No row number is
written down: a section's range is discovered by grouping the alignment on its own
audio-file column, and its spoken heading / chapter-end marker are read off the
aligned text. A re-released alignment therefore fails a check instead of silently
mis-cropping.

Kokoro leaves 4-10 s of narration unaligned inside four of the six sections. The
crop stays one continuous span either way -- a hole is unaligned audio, never a
splice -- but the automatic text is then genuinely short of the Aozora reference,
so `alignment_check` spends the unaligned FRACTION of the span as extra CER budget
rather than failing an honest build.

Scoring the sherpa fallbacks is opt-in: a section keeps its recorded scores while
its WAV and reference hashes both hold, so an acquisition run rebuilds the corpus
without re-decoding. Needs transient libsndfile bindings for MP3 decode:

    uv run --with soundfile python tests/eval_long_form.py [--sections 1,3] [--score]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import NamedTuple

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
    VAD_MODEL,
    VAD_PRE_PAD_S,
    check_models,
    resample,
)

MANIFEST = TESTS / "long_form.json"

TITLE = "ごん狐"
AUTHOR = "新美南吉"
READER = "ekzemplaro"
LICENSE = "CC0-1.0 (LibriVox/Internet Archive recording)"
LIBRIVOX = "https://librivox.org/gongitsune-by-nankichi-niimi/"
ARCHIVE_ITEM = "https://archive.org/details/gongitsune_um_librivox"

ALIGNMENT_URL = (
    "https://github.com/kaiidams/Kokoro-Speech-Dataset/releases/download/1.3/kokoro-speech-v1_3.zip"
)
ALIGNMENT_SHA256 = "5a4a290672016ebe70372ed3d47063f846d86e2b96aa9e9b9d35161670d9f666"
ALIGNMENT_MEMBER = "gongitsune-by-nankichi-niimi.metadata.txt"

AUDIO_BASE = "https://archive.org/download/gongitsune_um_librivox"
AUDIO_NAME = "gongitsune_{n:02d}_niimi_64kb.mp3"
# One pin per section; `_07` is a 404 and the Aozora text carries no 七 heading,
# so six is the whole story and a seventh sibling would fail both cross-checks.
AUDIO_SHA256 = {
    1: "f2dd16a2e9400d54819f0967ccd77a7948b1437dcf739d702f792a3e933fc141",
    2: "b7cbbea06363ca695b8b683cbf727f4d7cd539f93c10cc28fb34ed3bda7ed8e3",
    3: "90d5d802e2d2373e9f8400b6967facc30e971fa89bbce6ac422c24a784c5587d",
    4: "c0d26884e92ffcd58dd5f0c7b2b962b92b80236723eb0441300c87a585240847",
    5: "dd73097db911af6064e1ae4cb9f746809c457a0967843ced6a226b125cd95443",
    6: "c631ad41975b96c8e189e7a0b20f97f525715830cb21f208b86765fdebf98188",
}
AUDIO_SAMPLE_RATE = 22_050

TEXT_URL = "https://www.aozora.gr.jp/cards/000121/files/628_ruby_649.zip"
TEXT_SHA256 = "2f214158ddc83a89c88400c8ca63cb0d6add8b625b3cea0599a26fdc19274189"
TEXT_MEMBER = "gongitsune.txt"

NUMERALS = "一二三四五六七八九十"
HEADING = "［＃７字下げ］{k}［＃「{k}」は中見出し］"
COLOPHON = "底本："
CHAPTER_END = "章 おわり"
LEAD_S = 0.4
TAIL_S = 0.6
SURFACE_BUDGET = 0.10
USER_AGENT = "live-stt-long-form-evaluator/1"


class Row(NamedTuple):
    """One Kokoro-Align row: sample span + its automatic transcript."""

    row: int
    audio: str
    start: int
    end: int
    text: str


def fetch(url: str, expected_sha256: str, cache_name: str) -> bytes:
    """Return one pinned source, cached; refuse silent upstream replacement.

    The hash is checked on the cached path too (L-017), so a truncated or edited
    cache entry re-fetches instead of becoming evidence, and the download is
    staged before it is installed under its real name.
    """
    cached = CACHE / cache_name
    if cached.exists():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == expected_sha256:
            return data
        print(f"cached {cache_name} fails its pin; re-fetching", file=sys.stderr)

    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"source hash mismatch: {url}\nexpected {expected_sha256}\nactual   {actual}"
        )
    CACHE.mkdir(parents=True, exist_ok=True)
    stage = cached.with_name(cached.name + ".part")
    stage.write_bytes(data)
    stage.replace(cached)
    print(f"fetched {len(data):,} bytes: {url}")
    return data


def zip_member(archive: bytes, member: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        return zf.read(member)


def alignment_sections(raw: bytes) -> dict[int, list[Row]]:
    """Group the alignment into {section number: rows}, announcement dropped.

    The audio-file column is what says which section a row belongs to, so the row
    ranges are read out of the alignment rather than written down here. Row 1 is
    the separately spoken title/author, identified by its own text; every other
    row is narration.
    """
    announcement = normalize(TITLE + AUTHOR)
    grouped: dict[int, list[Row]] = {}
    for line in raw.decode("utf-8").splitlines():
        clip_id, audio, start, end, text, _reading = line.split("|")
        match = re.fullmatch(r"gongitsune_(\d+)_niimi_64kb\.mp3", audio)
        if not match:
            raise ValueError(f"Kokoro alignment names an unexpected audio file: {audio}")
        row = Row(int(clip_id.rsplit("-", 1)[1]), audio, int(start), int(end), text)
        grouped.setdefault(int(match.group(1)), []).append(row)

    if sorted(grouped) != list(range(1, len(grouped) + 1)):
        raise ValueError(f"Kokoro alignment section numbers are not 1..N: {sorted(grouped)}")
    for n, rows in grouped.items():
        kept = [row for row in rows if normalize(row.text) != announcement]
        if kept != rows[len(rows) - len(kept) :]:
            raise ValueError(f"section {n}: title/author announcement is not a leading row")
        if not kept:
            raise ValueError(f"section {n}: no narration rows survive the announcement drop")
        if [row.row for row in kept] != list(range(kept[0].row, kept[0].row + len(kept))):
            raise ValueError(f"section {n}: Kokoro alignment row ids are not consecutive")
        if any(right.start < left.end for left, right in zip(kept, kept[1:], strict=False)):
            raise ValueError(f"section {n}: Kokoro alignment rows overlap or run backwards")
        grouped[n] = kept
    return grouped


def aozora_body(text: str, n: int, last: int) -> str:
    """Extract clean section-`n` surface text from Aozora's ruby source."""
    heading = HEADING.format(k=NUMERALS[n - 1])
    starts = [match.start() for match in re.finditer(re.escape(heading), text)]
    # 一 also names itself in Aozora's notation legend; the real heading is last.
    if not 1 <= len(starts) <= 2:
        raise ValueError(f"section {n}: expected 1-2 heading markers, got {len(starts)}")
    start = starts[-1] + len(heading)
    terminator = COLOPHON if n == last else HEADING.format(k=NUMERALS[n])
    end = text.find(terminator, start)
    if end < 0:
        raise ValueError(f"section {n}: terminator {terminator!r} missing from Aozora source")
    body = re.sub(r"《[^》]*》|［＃[^］]*］", "", text[start:end])
    return re.sub(r"\s+", " ", body).strip()


def reference_text(body: str, automatic: str, numeral: str) -> str:
    """Aozora body, bracketed by whatever the reader actually spoke.

    The reader announces a section number on some sections and a chapter-end
    marker on others; both are taken from the aligned text, because a marker the
    crop excludes would otherwise score as a deletion against every engine.
    """
    head = f"{numeral} " if automatic.startswith(f"{numeral} ") else ""
    tail = f" {CHAPTER_END.replace(' ', '')}" if automatic.endswith(CHAPTER_END) else ""
    return f"{head}{body}{tail}"


def alignment_check(ref: str, automatic: str, unaligned: int, span: int) -> dict:
    """Quantify clean-Aozora vs automatic-alignment surface disagreement.

    Kokoro's automatic text mixes surface kanji and phonetic substitutions, so a
    few percent of disagreement is expected; this is a same-material sanity check
    on the extraction, not the recognizer's reference. Narration Kokoro left
    unaligned cannot appear in that text at all, so the unaligned fraction of the
    span is added to the budget -- otherwise an honest build fails on a hole.
    """
    ref_norm = normalize(ref)
    s, d, ins = align(ref_norm, normalize(automatic))
    budget = SURFACE_BUDGET + unaligned / span
    row = {
        "N": len(ref_norm),
        "S": s,
        "D": d,
        "I": ins,
        "cer": (s + d + ins) / len(ref_norm),
    }
    if row["cer"] > budget:
        raise ValueError(
            f"Aozora/alignment text mismatch too large: {row['cer']:.1%} > {budget:.1%}"
        )
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


def vad_profile(audio: np.ndarray) -> dict:
    """Natural endpointing over the built section: how the shipped VAD hears it."""
    lengths = [length for _, length in vad_segments(audio)]
    durations = [round(length / SAMPLE_RATE, 3) for length in lengths]
    prepad = round(VAD_PRE_PAD_S * SAMPLE_RATE)
    split_trigger = round(DECODE_SPLIT_TRIGGER_S * SAMPLE_RATE)
    return {
        "segment_durations_s": durations,
        "n_segments": len(durations),
        "max_segment_s": max(durations, default=0.0),
        "prepad_s": VAD_PRE_PAD_S,
        "max_resliced_upper_bound_s": round((max(lengths, default=0) + prepad) / SAMPLE_RATE, 3),
        "decode_split_trigger_s": DECODE_SPLIT_TRIGGER_S,
        "decode_split_candidates_upper_bound": sum(
            length + prepad > split_trigger for length in lengths
        ),
    }


def build_section(n: int, rows: list[Row], aozora: str, last: int) -> dict:
    """Fetch, crop, install and describe one section of the narration."""
    numeral = NUMERALS[n - 1]
    name = AUDIO_NAME.format(n=n)
    mp3 = fetch(f"{AUDIO_BASE}/{name}", AUDIO_SHA256[n], name)

    aligned_start, aligned_end = rows[0].start, rows[-1].end
    unaligned = sum(right.start - left.end for left, right in zip(rows, rows[1:], strict=False))
    automatic = " ".join(row.text for row in rows)
    ref = reference_text(aozora_body(aozora, n, last), automatic, numeral)
    check = alignment_check(ref, automatic, unaligned, aligned_end - aligned_start)

    audio = decode_crop(mp3, aligned_start, aligned_end)
    wav = CACHE / f"gongitsune_{n:02d}.wav"
    CACHE.mkdir(parents=True, exist_ok=True)
    stage = wav.with_name(wav.name + ".part")
    write_wav(stage, audio)
    stage.replace(wav)

    return {
        "id": f"gongitsune_{n:02d}",
        "chapter": numeral,
        "audio": {
            "url": f"{AUDIO_BASE}/{name}",
            "sha256": AUDIO_SHA256[n],
            "bytes": len(mp3),
        },
        "build": {
            "source_rate": AUDIO_SAMPLE_RATE,
            "aligned_start": aligned_start,
            "aligned_end": aligned_end,
            "first_row": rows[0].row,
            "last_row": rows[-1].row,
            "row_count": len(rows),
            "unaligned_samples": unaligned,
            "lead_s": LEAD_S,
            "tail_s": TAIL_S,
            "wav": str(wav.relative_to(ROOT)),
            "wav_sha256": sha256_file(wav),
            "audio_s": round(len(audio) / SAMPLE_RATE, 3),
        },
        "reference": {
            "text": ref,
            "normalized_sha256": hashlib.sha256(normalize(ref).encode()).hexdigest(),
            "kokoro_alignment_text": automatic,
            "alignment_check": check,
        },
        "vad": vad_profile(audio),
    }


def keep_scores(old: dict, section: dict) -> dict | None:
    """Prior scores for this section, iff the inputs they were measured on hold.

    score_wav() is a function of (reference, WAV bytes, engine), so both hashes
    matching is exactly the condition under which recorded rows still describe this
    build -- which is what lets acquisition rerun without re-decoding, and what
    drops a row the moment a rebuild moves the artifact under it.
    """
    scores = old.get("scores")
    if scores is None:
        return None
    if old["build"]["wav_sha256"] != section["build"]["wav_sha256"]:
        return None
    if old["reference"]["normalized_sha256"] != section["reference"]["normalized_sha256"]:
        return None
    return scores


def report(section: dict) -> None:
    build, vad, check = section["build"], section["vad"], section["reference"]["alignment_check"]
    print(
        f"{section['id']} ({section['chapter']}): rows {build['first_row']}-{build['last_row']} "
        f"{build['audio_s']:.3f}s unaligned={build['unaligned_samples'] / AUDIO_SAMPLE_RATE:.3f}s "
        f"align_cer={check['cer']:.4f} vad={vad['n_segments']}seg "
        f"max={vad['max_segment_s']:.3f}s splits={vad['decode_split_candidates_upper_bound']}"
    )
    for engine, row in section.get("scores", {}).items():
        print(f"  {engine}: D={row['D']}/{row['N']} CER={row['cer']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--sections", default="", help="comma-separated numbers (default: every one)")
    ap.add_argument("--score", action="store_true", help="also decode on the sherpa fallbacks")
    args = ap.parse_args()

    if not VAD_MODEL.exists():
        print(f"error: missing {VAD_MODEL}; see models/README.md", file=sys.stderr)
        sys.exit(1)
    if args.score:
        for engine in ENGINES:
            err = check_models(engine)
            if err:
                print(f"error: engine {engine}: {err.splitlines()[0]}", file=sys.stderr)
                sys.exit(1)

    alignment = alignment_sections(
        zip_member(
            fetch(ALIGNMENT_URL, ALIGNMENT_SHA256, "kokoro-speech-v1_3.zip"), ALIGNMENT_MEMBER
        )
    )
    aozora = zip_member(fetch(TEXT_URL, TEXT_SHA256, "628_ruby_649.zip"), TEXT_MEMBER).decode(
        "shift_jis"
    )
    last = len(alignment)
    if set(AUDIO_SHA256) != set(alignment):
        raise ValueError(f"pinned sections {sorted(AUDIO_SHA256)} != aligned {sorted(alignment)}")
    if HEADING.format(k=NUMERALS[last]) in aozora:
        raise ValueError(f"Aozora source carries a section {NUMERALS[last]} the alignment lacks")

    selected = [int(x) for x in args.sections.split(",")] if args.sections else sorted(alignment)
    previous = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    sections = dict(previous.get("sections", {}))

    for n in selected:
        section = build_section(n, alignment[n], aozora, last)
        if args.score:
            ref, wav = section["reference"]["text"], ROOT / section["build"]["wav"]
            scores = {engine: score_wav(ref, wav, engine) for engine in ENGINES}
            for engine, row in scores.items():
                if not row["hyp"]:
                    raise RuntimeError(f"{engine} produced an empty transcript for section {n}")
            section["scores"] = scores
        else:
            scores = keep_scores(previous.get("sections", {}).get(f"{n:02d}", {}), section)
            if scores is not None:
                section["scores"] = scores
        sections[f"{n:02d}"] = section
        report(section)

    manifest = {
        "source": {
            "title": TITLE,
            "author": AUTHOR,
            "reader": READER,
            "license": LICENSE,
            "librivox": LIBRIVOX,
            "archive_item": ARCHIVE_ITEM,
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
        "sections": {key: sections[key] for key in sorted(sections)},
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    total = sum(s["build"]["audio_s"] for s in manifest["sections"].values())
    print(f"wrote {MANIFEST.relative_to(ROOT)}: {len(manifest['sections'])} sections, {total:.3f}s")


if __name__ == "__main__":
    main()
