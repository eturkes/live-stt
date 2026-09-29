#!/usr/bin/env python3
"""Re-derive D-016's pause-free state-retention CER on the shipped path (M11.5).

Replays the 182 s pause-free probe through the real VAC path -- silero opening a
LocalAgreement-2 buffer, whisper large-v3-turbo int8 on the requested OpenVINO
device -- and scores the committed transcript against the probe's own reference.

On-demand, never a gate step: it needs the gitignored weights, the gitignored
probe WAV, and minutes of accelerator time. Run it when a decode or streaming
change puts the retention number in question.

    source /var/home/eturkes/.local/app/intel-accel/env.sh && source .envrc && \
        uv run python tests/eval_retention.py [--device NPU] [--json] \
        [--pace X [--stall] [--no-catchup]]

`--pace` decodes on eval_backpressure's virtual clock instead of replay's unpaced
queue, so the catch-up rule and the capture headroom engage: the accuracy side of
a slow machine state, which unpaced replay cannot reach (asr-pipeline.md).

The WAV is content-checked against `retention_probe.json` before decoding, so a
published number is always bound to the pinned input rather than to whatever
happens to sit in the ignored cache; rebuild it with `build_retention_probe.py`.

Placement is recorded as an exact-target inference, not a device readback:
`openvino_genai.WhisperPipeline` wraps its compiled model without exposing it, so
there is no `EXECUTION_DEVICES` property to query. What the run does establish is
that the exact device name was requested and that decoding succeeded on it --
`check_device` admits exact names only, and an unavailable or unsupported target
raises rather than silently relocating (`AUTO:` would relocate, which is why the
shipped constant is a bare name).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

import eval_backpressure  # noqa: E402
from build_stressor import sha256_file  # noqa: E402

import live_stt  # noqa: E402
import replay  # noqa: E402  (after sys.path injection)
from cer import align, normalize  # noqa: E402
from live_stt import ASR_DEVICE, check_device, check_models, load_recognizer  # noqa: E402

MANIFEST = TESTS / "retention_probe.json"
WAV = ROOT / "spike" / "backends" / "cache" / "retention_probe.wav"
ENGINE = "whisper"


def _paced(rec, pace: float, stall: bool, catchup: bool) -> dict:
    """The VAC path on eval_backpressure's virtual clock, decoding with the REAL recogniser.

    Decode points and queue come from paced production replay at `pace` x the recorded
    trace cost (`stall` charges the middle update LIVE_STALL_S); text and segments are
    real, so trims are too -- off the recorded trajectory the trace recogniser's are not.
    `catchup=False` puts VAC_BACKLOG_S out of reach and unbounds the queue: the lossless
    counterfactual, whose price is caption lag rather than drops.
    """
    series = eval_backpressure.load_vac_trace()["clips"]["retention_probe"]["series"]
    lines: list[str] = []

    def emit(_tag, _seq, text, _output_file, *, held=False):
        lines.append(text)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(eval_backpressure, "_TraceRecognizer", lambda _: rec))
        stack.enter_context(mock.patch.object(live_stt, "emit_line", emit))
        if not catchup:
            # Far past any queue; int() of an infinite threshold raises.
            stack.enter_context(mock.patch.object(live_stt, "VAC_BACKLOG_S", 1e6))
        run = eval_backpressure.vac_paced_replay(
            replay.load_wav_f32_16k(WAV),
            series,
            headroom_s=live_stt.AUDIO_HEADROOM_S if catchup else 1e6,
            cost_scale=pace,
            stall=(len(series) // 2, eval_backpressure.LIVE_STALL_S) if stall else None,
        )
    return {
        "segments": [{"text": text} for text in lines],
        "audio_s": run["audio_s"],
        "n_segments": len(lines),
        "total_decode_s": run["total_decode_s"],
        "overall_rtf": run["duty"],
        "paced": {
            key: run[key]
            for key in (
                "cost_scale",
                "updates",
                "max_audio_s",
                "drops",
                "trims",
                "forced_trims",
                "max_buffer_s",
            )
        }
        | {"stall_s": eval_backpressure.LIVE_STALL_S if stall else 0.0, "catchup": catchup},
    }


def score(
    device: str, pace: float | None = None, stall: bool = False, catchup: bool = True
) -> dict:
    """Decode the pinned probe on `device` and return a self-substantiating row."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["probe"]
    if not WAV.exists():
        raise SystemExit(
            f"missing probe WAV: {WAV}\nbuild it: uv run python tests/build_retention_probe.py"
        )
    digest = sha256_file(WAV)
    if digest != manifest["audio_sha256"]:
        raise SystemExit(
            f"probe WAV does not match the manifest\n  on disk: {digest}\n  pinned:  "
            f"{manifest['audio_sha256']}\nrebuild it: uv run python tests/build_retention_probe.py"
        )
    for err in (check_models(ENGINE), check_device(ENGINE, device)):
        if err:
            raise SystemExit(f"error: {err.splitlines()[0]}")

    rec = load_recognizer(ENGINE, device)
    if pace is None:
        report = replay.replay_recognizer(WAV, rec, ENGINE)
    else:
        report = _paced(rec, pace, stall, catchup)
    # One utterance per committed VAC line; the probe is pause-free, so the
    # transcript is their concatenation in emission order.
    hyp = "".join(s["text"] for s in report["segments"])
    ref_norm = normalize(manifest["ja_ref"])
    s, d, ins = align(ref_norm, normalize(hyp))
    n = len(ref_norm)
    return {
        "engine": ENGINE,
        "requested_device": device,
        "placement": (
            f"exact-target inference: check_device admits exact names only and "
            f"{device!r} decoded successfully; WhisperPipeline exposes no compiled "
            f"model, so EXECUTION_DEVICES is unreadable"
        ),
        "audio_sha256": digest,
        "audio_s": report["audio_s"],
        "n_segments": report["n_segments"],
        "total_decode_s": report["total_decode_s"],
        "overall_rtf": report["overall_rtf"],
        "hyp": hyp,
        "hyp_chars": len(normalize(hyp)),
        "N": n,
        "S": s,
        "D": d,
        "I": ins,
        "cer": (s + d + ins) / n,
        "del_rate": d / n,
        "ins_rate": ins / n,
        **({"paced": report["paced"]} if "paced" in report else {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--device", default=ASR_DEVICE, help=f"OpenVINO device (default: {ASR_DEVICE})")
    ap.add_argument("--json", action="store_true", help="Emit the row as JSON.")
    ap.add_argument(
        "--pace",
        type=float,
        help="Decode on the backpressure virtual clock at this x recorded cost (real text).",
    )
    ap.add_argument("--stall", action="store_true", help="With --pace: one LIVE_STALL_S update.")
    ap.add_argument(
        "--no-catchup", action="store_true", help="With --pace: no catch-up, unbounded queue."
    )
    args = ap.parse_args()

    row = score(args.device, args.pace, args.stall, not args.no_catchup)
    if args.json:
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return
    print(
        f"retention probe on {row['requested_device']}: "
        f"CER={row['cer']:.4f}  N={row['N']}  S={row['S']}  D={row['D']}  I={row['I']}"
    )
    print(
        f"  audio={row['audio_s']:.3f}s  segments={row['n_segments']}  "
        f"decode={row['total_decode_s']:.1f}s  rtf={row['overall_rtf']:.3f}  "
        f"hyp_chars={row['hyp_chars']}"
    )
    if "paced" in row:
        print(f"  paced: {row['paced']}")


if __name__ == "__main__":
    main()
