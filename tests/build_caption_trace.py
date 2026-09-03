#!/usr/bin/env python3
"""Record the shipped path's caption stream over the pinned narration (M12.1, M12.3).

Replays every pinned 「ごん狐」 section through the real VAC path -- silero opening
a LocalAgreement-2 buffer, whisper large-v3-turbo int8 on the requested OpenVINO
device -- and writes one row per PUBLISHED caption to `caption_trace.json`,
numbered continuously across the six sections.

**One decode per section, one session downstream.** Nothing on the shipped path
carries state from one utterance to the next: D-016(c) deleted prev-text
conditioning and the NPU refuses prompts outright, so section k cannot decode
differently for what preceded it and six replays yield the captions one
continuous run would. The state that DOES span the story is the learner's, and it
lives downstream in `eval_term_census.py`, which reads `captions` as the one
stream its `idx` numbers. That is what makes `CONTEXT_TERM_LEASE`=60 segments
reachable at all -- one section yields ~67 captions, the whole story ~200.

The trace is what makes M12 affordable. `ASR_DEVICE` is "NPU" and
`ASR_HOTWORDS_DEVICES` excludes it, so `WhisperEngine.set_hotwords` drops the
term list unconditionally and `generate` passes no `hotwords` keyword: session
context cannot reach the recogniser on the shipped device. The caption stream is
therefore arm-independent, one NPU pass serves every downstream arm, and each of
those arms is a CPU-only offline replay of this file (`eval_term_census.py` is
the first).

On-demand, never a gate step: it needs the gitignored weights, the gitignored
narration WAVs, and ~14 min of accelerator time for the whole story.

    source ~/.local/app/intel-accel/env.sh && env -u PYTHONPATH \
        uv run python tests/build_caption_trace.py [--device NPU]

Every WAV is content-checked against `long_form.json` before ANY of them is
decoded, so a stale cache fails in seconds rather than six sections in, and every
caption stays bound to the pinned input. Placement is an exact-target inference,
not a device readback, for the reason `eval_retention.py` records:
`openvino_genai.WhisperPipeline` wraps its compiled model without exposing it,
and `check_device` admits exact names only.
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
    WhisperEngine,
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
    """Decode every pinned section on `device` and return the continuous trace."""
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
            # Caption times stay section-local, exactly as decoded; this is the
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
            f"  {key} {section['chapter']}: {report['n_segments']} captions, "
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
            # Recorded because it is the premise the whole milestone rests on: with
            # no term list reaching the model, every downstream arm may replay these
            # captions instead of paying for its own accelerator run.
            "hotwords_reachable": isinstance(rec, WhisperEngine) and rec.supports_hotwords,
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
        f"wrote {OUT.relative_to(ROOT)}: {run['n_captions']} captions "
        f"({run['n_nonempty']} non-empty) over {len(trace['sections'])} sections on "
        f"{run['requested_device']}, audio={trace['source']['audio_s']:.3f}s "
        f"decode={run['total_decode_s']:.1f}s rtf={run['overall_rtf']:.3f} "
        f"hotwords_reachable={run['hotwords_reachable']}"
    )


if __name__ == "__main__":
    main()
