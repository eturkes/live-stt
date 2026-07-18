#!/usr/bin/env python3
"""Build the pause-free streaming-ASR state-retention probe (M10.5b).

The probe cycles the five pinned Common Voice single utterances to about three
minutes of continuous speech. It reuses the M9.1 stressor trim/crossfade helpers
so there is one deterministic construction recipe and performs no ASR decode.
A cap-raised silero pass must see one speech segment containing every join before
the manifest is written; the cache WAV remains ignored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_stressor import (  # noqa: E402
    COMPONENTS,
    CROSSFADE_S,
    LEAD_S,
    SAMPLE_RATE,
    TAIL_S,
    crossfade_join,
    cycle_order,
    frame,
    join_offsets,
    load_component,
    sha256_file,
    speech_extent,
    vad_segments,
    write_wav,
)

CACHE = ROOT / "spike" / "backends" / "cache"
REAL_CLIPS = ROOT / "tests" / "real_clips.json"
MANIFEST = ROOT / "tests" / "retention_probe.json"
WAV = CACHE / "retention_probe.wav"

PROBE_TARGET_S = 180.0
CAP_OFF_S = 300.0  # above the probe, far below max_speech_duration int32 overflow


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    real_clips = json.loads(REAL_CLIPS.read_text(encoding="utf-8"))

    trimmed: dict[str, np.ndarray] = {}
    trimmed_dur: dict[str, float] = {}
    for cid in COMPONENTS:
        samples = load_component(cid)
        lo, hi = speech_extent(samples)
        speech = np.ascontiguousarray(samples[lo:hi], dtype=np.float32)
        trimmed[cid] = speech
        trimmed_dur[cid] = len(speech) / SAMPLE_RATE
        print(f"trim {cid}: {trimmed_dur[cid]:.3f} s speech")

    order = cycle_order(trimmed_dur, PROBE_TARGET_S)
    xfade = int(CROSSFADE_S * SAMPLE_RATE)
    lead = int(LEAD_S * SAMPLE_RATE)
    part_lens = [len(trimmed[cid]) for cid in order]
    joined = crossfade_join([trimmed[cid] for cid in order], xfade)
    audio = frame(joined, LEAD_S, TAIL_S)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = (audio / peak).astype(np.float32)
    write_wav(WAV, audio)

    audio_s = len(audio) / SAMPLE_RATE
    joins = join_offsets(part_lens, xfade, lead)
    segs = vad_segments(audio, CAP_OFF_S)
    failures: list[str] = []
    if len(segs) != 1:
        failures.append(f"expected one cap-off VAD segment, got {len(segs)}: {segs}")
    joins_inside = False
    if len(segs) == 1:
        start, length = segs[0]
        outside = [offset for offset in joins if not start <= offset < start + length]
        joins_inside = not outside
        if outside:
            failures.append(f"join offsets outside the VAD segment: {outside}")
    if not 178.0 <= audio_s <= 187.0:
        failures.append(f"audio duration {audio_s:.3f} s is outside the 178-187 s band")
    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        print("validation FAILED -- manifest not written", file=sys.stderr)
        sys.exit(1)

    start, length = segs[0]
    manifest = {
        "recipe": {
            "source_clips": COMPONENTS,
            "trim": "silero speech extent via make_vad()",
            "join": "equal-power crossfade",
            "crossfade_ms": round(CROSSFADE_S * 1000, 1),
            "lead_s": LEAD_S,
            "tail_s": TAIL_S,
            "target_s": PROBE_TARGET_S,
            "purpose": (
                "pause-free continuous-speech state-retention probe for streaming ASR, "
                "M10.5b; no VAD-silence gaps so a streaming recognizer stays in one "
                "un-endpointed state"
            ),
            "continuity_proof": (
                "whole probe is one VAD segment with max_speech_duration raised; "
                "every join_samples offset is inside it"
            ),
        },
        "components": {
            cid: {
                "ja_ref": real_clips[cid]["ja_ref"],
                "source": real_clips[cid]["source"],
                "trimmed_dur_s": round(trimmed_dur[cid], 3),
            }
            for cid in COMPONENTS
        },
        "probe": {
            "ja_ref": " ".join(real_clips[cid]["ja_ref"] for cid in order),
            "order": order,
            "component_count": len(order),
            "audio_s": round(audio_s, 3),
            "peak": round(peak, 4),
            "audio_sha256": sha256_file(WAV),
            "join_samples": joins,
            "validation": {
                "vad_segs": len(segs),
                "joins_inside_segments": joins_inside,
                "segment": {"start": start, "length": length},
            },
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"built retention_probe: {audio_s:.3f} s, {len(order)} components, "
        f"peak {peak:.4f}, vad_segs={len(segs)}"
    )
    print(f"wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
