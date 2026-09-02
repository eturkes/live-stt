#!/usr/bin/env python3
"""Record the shipped path's caption stream over the pinned narration (M12.1).

Replays `gongitsune_01.wav` through the real VAC path -- silero opening a
LocalAgreement-2 buffer, whisper large-v3-turbo int8 on the requested OpenVINO
device -- and writes one row per PUBLISHED caption to `caption_trace.json`.

The trace is what makes M12 affordable. `ASR_DEVICE` is "NPU" and
`ASR_HOTWORDS_DEVICES` excludes it, so `WhisperEngine.set_hotwords` drops the
term list unconditionally and `generate` passes no `hotwords` keyword: session
context cannot reach the recogniser on the shipped device. The caption stream is
therefore arm-independent, one NPU replay serves every downstream arm, and each
of those arms is a CPU-only offline replay of this file (`eval_term_census.py`
is the first).

On-demand, never a gate step: it needs the gitignored weights, the gitignored
narration WAV, and minutes of accelerator time.

    source ~/.local/app/intel-accel/env.sh && env -u PYTHONPATH \
        uv run python tests/build_caption_trace.py [--device NPU]

The WAV is content-checked against `long_form.json` before decoding, so every
caption is bound to the pinned input rather than to whatever sits in the ignored
cache. Placement is an exact-target inference, not a device readback, for the
reason `eval_retention.py` records: `openvino_genai.WhisperPipeline` wraps its
compiled model without exposing it, and `check_device` admits exact names only.
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


def build(device: str) -> dict:
    """Decode the pinned narration on `device` and return the caption trace."""
    build_meta = json.loads(MANIFEST.read_text(encoding="utf-8"))["build"]
    wav = ROOT / build_meta["wav"]
    if not wav.exists():
        raise SystemExit(
            f"missing narration WAV: {wav}\nbuild it: uv run --with soundfile python "
            "tests/eval_long_form.py"
        )
    digest = sha256_file(wav)
    if digest != build_meta["wav_sha256"]:
        raise SystemExit(
            f"narration WAV does not match the manifest\n  on disk: {digest}\n  pinned:  "
            f"{build_meta['wav_sha256']}\nrebuild it: uv run --with soundfile python "
            "tests/eval_long_form.py"
        )
    for err in (check_models(ENGINE), check_device(ENGINE, device)):
        if err:
            raise SystemExit(f"error: {err.splitlines()[0]}")

    rec = load_recognizer(ENGINE, device)
    report = replay.replay_recognizer(wav, rec, ENGINE)
    return {
        "source": {
            "wav": build_meta["wav"],
            "wav_sha256": digest,
            "manifest": "tests/long_form.json",
            "audio_s": report["audio_s"],
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
            "n_captions": report["n_segments"],
            "n_nonempty": report["n_nonempty"],
            "total_decode_s": report["total_decode_s"],
            "overall_rtf": report["overall_rtf"],
        },
        "captions": [
            {k: s[k] for k in ("idx", "start_s", "end_s", "dur_s", "decode_s", "text")}
            for s in report["segments"]
        ],
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
        f"({run['n_nonempty']} non-empty) on {run['requested_device']}, "
        f"audio={trace['source']['audio_s']:.3f}s decode={run['total_decode_s']:.1f}s "
        f"rtf={run['overall_rtf']:.3f} hotwords_reachable={run['hotwords_reachable']}"
    )


if __name__ == "__main__":
    main()
