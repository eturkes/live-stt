#!/usr/bin/env python3
"""Record raw utterance observations over the pinned narration.

Each pinned 「ごん狐」 section runs through the real VAC worker on whisper-ja-760M
int8. `replay.replay_recognizer` observes `on_segment`: ONE RAW utterance, before
publication screening/splitting, not the numbered SRC stream. Legacy schema names
`captions` / `n_captions` / `idx` mean raw observations / their count / observation ids;
ids are continuous across sections, never published line numbers. A screened utterance
remains here and a trimmed utterance can publish several lines for one raw row.

Each section starts a fresh VAD. The recogniser has no cross-utterance text state,
but `LeveledVad` keeps level history; independent section replay equals a continuous
cut only where gain remains 1.0, as measured on this pinned loud narration, not on
arbitrary quiet audio. `eval_term_census.py` keeps one learner across all raw rows.
Its result describes these raw inputs, not today's screened publication stream.
`eval_en_pairing.py` similarly samples one translation turn per frozen turbo raw row;
it does not reproduce today's settled-piece translation turns.

No context reaches the recogniser, so one raw trace serves every downstream arm.
Traces recorded while that channel existed carry `run.hotwords_reachable`, false
in every committed one; nothing reads it. Raw text + VAD boundaries reproduce across
NPU runs on these inputs; decode cost does not.

On-demand, never a gate step: gitignored weights + narration WAVs, ~8 min on NPU.
`<intel-accel>` = the driver farm the user-level instruction file names.

    source <intel-accel>/env.sh && source .envrc && \
        uv run python tests/build_caption_trace.py [--device NPU]

Every WAV is hash-checked against `long_form.json` BEFORE any decoder loads.
Placement is exact-target inference: `WhisperPipeline` exposes no compiled model
for device readback, and `check_device` admits exact names only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

from build_stressor import sha256_file  # noqa: E402

import replay  # noqa: E402  (after sys.path injection)
from live_stt import (  # noqa: E402
    ASR_DEVICE,
    check_device,
    check_models,
    load_recognizer,
)

MANIFEST = TESTS / "long_form.json"
OUT = TESTS / "caption_trace.json"
ENGINE = "whisper"


def pinned_wavs() -> dict[str, tuple[dict, Path, str]]:
    """Every corpus section with its on-disk WAV, hash-checked against the manifest."""
    sections = json.loads(MANIFEST.read_text(encoding="utf-8"))["sections"]
    checked = {}
    for key, section in sorted(sections.items()):
        wav = ROOT / section["build"]["wav"]
        if not wav.exists():
            raise SystemExit(
                f"missing narration WAV: {wav}\nbuild it: uv run --with soundfile python "
                "tests/eval_long_form.py"
            )
        digest = sha256_file(wav)
        if digest != section["build"]["wav_sha256"]:
            raise SystemExit(
                f"section {key} WAV does not match the manifest\n  on disk: {digest}\n  pinned:  "
                f"{section['build']['wav_sha256']}\nrebuild it: uv run --with soundfile python "
                "tests/eval_long_form.py"
            )
        checked[key] = (section, wav, digest)
    return checked


def build(device: str) -> dict:
    """Decode every pinned section on `device` and return the raw-utterance trace."""
    checked = pinned_wavs()
    for err in (check_models(ENGINE), check_device(ENGINE, device)):
        if err:
            raise SystemExit(f"error: {err.splitlines()[0]}")

    rec = load_recognizer(ENGINE, device)
    sections: dict[str, dict] = {}
    captions: list[dict] = []
    offset_s = 0.0
    for key, (section, wav, digest) in checked.items():
        report = replay.replay_recognizer(wav, rec, ENGINE)
        first = len(captions) + 1
        for seg in report["segments"]:
            captions.append(
                {
                    "idx": len(captions) + 1,
                    "section": key,
                    "section_idx": seg["idx"],
                    **{k: seg[k] for k in ("start_s", "end_s", "dur_s", "decode_s", "text")},
                }
            )
        sections[key] = {
            "chapter": section["chapter"],
            "wav": section["build"]["wav"],
            "wav_sha256": digest,
            "audio_s": report["audio_s"],
            # Raw utterance times stay section-local, exactly as decoded; this is the
            # offset that places them on the continuous timeline.
            "offset_s": offset_s,
            "first_idx": first,
            "last_idx": len(captions),
            "n_captions": report["n_segments"],
            "n_nonempty": report["n_nonempty"],
            "decode_s": report["total_decode_s"],
            "rtf": report["overall_rtf"],
        }
        offset_s += report["audio_s"]
        print(
            f"  {key} {section['chapter']}: {report['n_segments']} raw utterances, "
            f"audio={report['audio_s']:.3f}s decode={report['total_decode_s']:.1f}s "
            f"rtf={report['overall_rtf']:.3f}",
            flush=True,
        )

    decode_s = sum(s["decode_s"] for s in sections.values())
    return {
        "source": {
            "manifest": "tests/long_form.json",
            "sections": sorted(sections),
            "audio_s": offset_s,
        },
        "run": {
            "engine": ENGINE,
            "requested_device": device,
            "placement": (
                f"exact-target inference: check_device admits exact names only and "
                f"{device!r} decoded successfully; WhisperPipeline exposes no compiled "
                f"model, so EXECUTION_DEVICES is unreadable"
            ),
            "n_captions": len(captions),
            "n_nonempty": sum(1 for c in captions if c["text"]),
            "total_decode_s": decode_s,
            "overall_rtf": (decode_s / offset_s) if offset_s > 0 else 0.0,
        },
        "sections": sections,
        "captions": captions,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--device", default=ASR_DEVICE, help=f"OpenVINO device (default: {ASR_DEVICE})")
    args = ap.parse_args()

    trace = build(args.device)
    OUT.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run = trace["run"]
    print(
        f"wrote {OUT.relative_to(ROOT)}: {run['n_captions']} raw utterances "
        f"({run['n_nonempty']} non-empty) over {len(trace['sections'])} sections on "
        f"{run['requested_device']}, audio={trace['source']['audio_s']:.3f}s "
        f"decode={run['total_decode_s']:.1f}s rtf={run['overall_rtf']:.3f}"
    )


if __name__ == "__main__":
    main()
