#!/usr/bin/env python3
"""Record the shipped VAC path's real per-update decode cost (M11.4).

Replays the two pinned pause-free corpora through `live_stt.worker` on the real
OpenVINO engine with the `on_update` seam attached, and writes every streaming
update in order: the buffer that decode saw, what it committed, the raw
hypothesis with its segment spans, and the measured decode duration.

The raw hypothesis is what makes the trace replayable. `StreamingProcessor` is a
pure function of the decode outputs and the buffer lengths, so feeding the
recorded hypotheses back in order reproduces the commit/trim trajectory exactly
without a model -- which is what lets `tests/eval_backpressure.py` pace the VAC
branch deterministically against measured costs instead of a flat RTF.

Aggregate RTF only shows mean compute below real time. VAC awaits each decode
inside the coroutine draining `audio_q`, so what the 2 s capture headroom is
actually spent against is the MAXIMUM contiguous decode, and that needs the
per-update series.

Needs the gitignored whisper weights and the accelerator env:

    source ~/.local/app/intel-accel/env.sh
    env -u PYTHONPATH uv run python tests/build_vac_trace.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import live_stt  # noqa: E402
import replay  # noqa: E402
from live_stt import (  # noqa: E402
    ASR_DEVICE,
    ENGINE_DIRS,
    SAMPLE_RATE,
    VAC_CHUNK_S,
    VAC_TRIM_S,
    check_device,
    check_models,
    load_recognizer,
)
from streaming import HARD_TRIM_S, StreamingProcessor  # noqa: E402

CACHE = ROOT / "spike" / "backends" / "cache"
TRACE = ROOT / "tests" / "vac_decode_trace.json"
# Both pinned pause-free clips: the 44.7 s stressor is the committed backpressure
# case, the 182 s retention probe is the longest continuous audio in the repo and
# therefore the one that can accumulate an unrecoverable backlog.
CLIPS = ("stress_long", "retention_probe")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_clip(path: Path, rec: object) -> dict[str, Any]:
    """Replay one WAV through the real worker, returning its ordered update trace."""
    hypotheses: list[tuple[str, list[live_stt.Segment]]] = []
    decode_segments = rec.decode_segments  # type: ignore[attr-defined]

    def recording_decode(samples):
        result = decode_segments(samples)
        hypotheses.append(result)
        return result

    processors: list[StreamingProcessor] = []

    def tracked_processor(**kwargs) -> StreamingProcessor:
        processor = StreamingProcessor(**kwargs)
        processors.append(processor)
        return processor

    updates: list[dict[str, Any]] = []

    def on_update(buffer_s, buffer_end_s, commit_audio_s, commit, final, decode_s):
        # process() calls decode exactly once, so the k-th hypothesis is this
        # update's; asserting it keeps a future extra decode from silently
        # shifting the whole series.
        assert len(hypotheses) == len(updates) + 1, "one decode per update"
        text, segments = hypotheses[-1]
        row = {
            "buffer_s": round(buffer_s, 6),
            "buffer_end_s": round(buffer_end_s, 6),
            "commit_audio_s": None if commit_audio_s is None else round(commit_audio_s, 6),
            "commit": commit,
            "final": final,
            "segments": [[round(s.start_s, 3), round(s.end_s, 3), s.text] for s in segments],
            "decode_s": round(decode_s, 6),
        }
        # decode_segments drops the spans unless they rejoin to the hypothesis, so
        # segments ARE the hypothesis wherever they exist. Storing it twice would
        # give the trace two editable sources of truth for the same string; store
        # `text` only where there are no spans to derive it from.
        if not segments:
            row["text"] = text
        updates.append(row)

    with (
        mock.patch.object(rec, "decode_segments", recording_decode),
        mock.patch.object(live_stt, "StreamingProcessor", tracked_processor),
    ):
        started = time.perf_counter()
        report = replay.replay_recognizer(path, rec, "whisper", on_update)
        wall_s = time.perf_counter() - started

    return {
        "wav_sha256": _sha256(path),
        "audio_s": round(report["audio_s"], 3),
        "utterances": report["n_segments"],
        "updates": len(updates),
        "trims": sum(p.trims for p in processors),
        # A nonzero count means the trim rule failed and dropped audio whose text
        # was never emitted -- a content loss the queue counters cannot see.
        "forced_trims": sum(p.forced_trims for p in processors),
        "total_decode_s": round(sum(u["decode_s"] for u in updates), 3),
        "max_decode_s": round(max((u["decode_s"] for u in updates), default=0.0), 6),
        "max_buffer_s": round(max((u["buffer_s"] for u in updates), default=0.0), 6),
        "wall_s": round(wall_s, 3),
        "series": updates,
    }


def build(device: str) -> dict[str, Any]:
    paths = {clip: CACHE / f"{clip}.wav" for clip in CLIPS}
    missing = [str(p.relative_to(ROOT)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing VAC trace inputs: " + ", ".join(missing))
    for check in (check_models("whisper"), check_device("whisper", device)):
        if check:
            raise RuntimeError(check)

    rec = load_recognizer("whisper", device)
    return {
        "model": ENGINE_DIRS["whisper"].name,
        # The pipeline exposes no compiled model, so placement is recorded as the
        # exact target that compiled and decoded, not as EXECUTION_DEVICES.
        "requested_device": device,
        "sample_rate": SAMPLE_RATE,
        "vac_chunk_s": VAC_CHUNK_S,
        "vac_trim_s": VAC_TRIM_S,
        "hard_trim_s": HARD_TRIM_S,
        "clips": {clip: trace_clip(path, rec) for clip, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--device", default=ASR_DEVICE, help="Exact OpenVINO device name.")
    parser.add_argument("--out", type=Path, default=TRACE)
    args = parser.parse_args()

    try:
        report = build(args.device)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for clip, row in report["clips"].items():
        print(
            f"{clip}: audio={row['audio_s']:.3f}s updates={row['updates']} "
            f"decode={row['total_decode_s']:.3f}s max_decode={row['max_decode_s']:.3f}s "
            f"max_buffer={row['max_buffer_s']:.3f}s trims={row['trims']} "
            f"forced_trims={row['forced_trims']} wall={row['wall_s']:.1f}s"
        )
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
