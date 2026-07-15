#!/usr/bin/env python3
"""Generate the committed CER + long-form viability baseline (M9.2a/M9.4).

Scores the seven real Common Voice clips and both continuous-speech stressors
with each supported engine, then reports decode RTF over synthetic continuous
audio targeting 5/10/20/40 seconds. The sweep reuses M9.1's exact trim,
crossfade, framing, and replay pipeline; its temporary WAVs are never committed.

Requires both model sets and the gitignored replay corpus:
    uv run python tests/eval_cer.py

The M9.4 shipped-config gate is default-engine excess deletion <=4% on each
stressor plus total CER <=15% for both engines. The CER bound prevents overlap
duplication from gaming a deletion-only gate. A miss prints every scored row but
leaves the committed table intact.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

from build_stressor import (  # noqa: E402
    CACHE,
    COMPONENTS,
    CROSSFADE_S,
    LEAD_S,
    TAIL_S,
    crossfade_join,
    cycle_order,
    decode_hyp,
    frame,
    load_component,
    speech_extent,
    write_wav,
)

import replay  # noqa: E402  (after sys.path injection)
from cer import align, normalize  # noqa: E402
from live_stt import check_models  # noqa: E402

BASELINE = TESTS / "cer_baseline.json"
REAL_CLIPS = TESTS / "real_clips.json"
STRESSORS = TESTS / "stressor_clips.json"
ENGINES = ("k2v2", "parakeet")
LENGTHS_S = (5, 10, 20, 40)
DEFAULT_ENGINE = "k2v2"
MAX_EXCESS_DEL_RATE = 0.04
# Current worst stressor CER is 12.12%; 15% retains measured headroom while
# rejecting both the pre-fix collapse and a material overlap-insertion regression.
MAX_CER = 0.15


def score_wav(ref: str, wav: Path, engine: str) -> dict:
    """Decode one WAV and return a self-substantiating CER row."""
    hyp, _ = decode_hyp(wav, engine)
    ref_norm = normalize(ref)
    if not ref_norm:
        raise ValueError(f"reference normalizes to empty: {wav}")
    s, d, ins = align(ref_norm, normalize(hyp))
    n = len(ref_norm)
    return {
        "ref": ref,
        "hyp": hyp,
        "N": n,
        "S": s,
        "D": d,
        "I": ins,
        "cer": (s + d + ins) / n,
        "del_rate": d / n,
        "ins_rate": ins / n,
    }


def validation_failures(out: dict[str, dict]) -> list[str]:
    """Return shipped stressor-gate misses without mutating the baseline."""
    failures = [
        f"{DEFAULT_ENGINE}/{sid} excess deletion {row['excess_del_rate']:.1%} > "
        f"{MAX_EXCESS_DEL_RATE:.1%}"
        for sid, row in out["stressors"][DEFAULT_ENGINE].items()
        if row["excess_del_rate"] > MAX_EXCESS_DEL_RATE
    ]
    failures.extend(
        f"{engine}/{sid} CER {row['cer']:.1%} > {MAX_CER:.1%}"
        for engine, rows in out["stressors"].items()
        for sid, row in rows.items()
        if row["cer"] > MAX_CER
    )
    return failures


def main() -> None:
    for engine in ENGINES:
        err = check_models(engine)
        if err:
            print(f"error: engine {engine}: {err.splitlines()[0]}", file=sys.stderr)
            sys.exit(1)

    real_clips = json.loads(REAL_CLIPS.read_text(encoding="utf-8"))
    stressor_manifest = json.loads(STRESSORS.read_text(encoding="utf-8"))
    out: dict[str, dict] = {
        "corpus": {engine: {} for engine in ENGINES},
        "stressors": {engine: {} for engine in ENGINES},
        "rtf_by_length": {engine: {} for engine in ENGINES},
    }

    for engine in ENGINES:
        for cid, meta in real_clips.items():
            row = score_wav(meta["ja_ref"], CACHE / f"{cid}.wav", engine)
            out["corpus"][engine][cid] = row
            print(f"corpus/{engine}/{cid}: D={row['D']}/{row['N']} CER={row['cer']:.3f}")

        for sid, meta in stressor_manifest["stressors"].items():
            row = score_wav(meta["ja_ref"], CACHE / f"{sid}.wav", engine)
            baseline_d = sum(
                stressor_manifest["components"][cid]["baseline"][engine]["D"]
                for cid in meta["order"]
            )
            row["excess_D"] = row["D"] - baseline_d
            row["excess_del_rate"] = round(row["excess_D"] / row["N"], 4)
            out["stressors"][engine][sid] = row
            print(
                f"stressor/{engine}/{sid}: D={row['D']}/{row['N']} "
                f"excess={row['excess_del_rate']:.1%}"
            )

    trimmed: dict[str, np.ndarray] = {}
    trimmed_dur: dict[str, float] = {}
    for cid in COMPONENTS:
        samples = load_component(cid)
        lo, hi = speech_extent(samples)
        trimmed[cid] = np.ascontiguousarray(samples[lo:hi], dtype=np.float32)
        trimmed_dur[cid] = len(trimmed[cid]) / replay.SAMPLE_RATE

    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        for target_s in LENGTHS_S:
            order = cycle_order(trimmed_dur, target_s)
            audio = frame(
                crossfade_join(
                    [trimmed[cid] for cid in order],
                    int(CROSSFADE_S * replay.SAMPLE_RATE),
                ),
                LEAD_S,
                TAIL_S,
            )
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak > 1.0:
                audio = (audio / peak).astype(np.float32)
            wav = scratch / f"{target_s}.wav"
            write_wav(wav, audio)
            for engine in ENGINES:
                report = replay.replay_wav(str(wav), engine)
                row = {
                    "audio_s": round(report["audio_s"], 3),
                    "decode_s": round(report["total_decode_s"], 3),
                    "rtf": round(report["overall_rtf"], 3),
                    "n_seg": report["n_segments"],
                    "n_nonempty": report["n_nonempty"],
                    "viable": report["n_nonempty"] > 0,
                }
                out["rtf_by_length"][engine][str(target_s)] = row
                print(
                    f"rtf/{engine}/{target_s}: audio={row['audio_s']:.3f}s "
                    f"decode={row['decode_s']:.3f}s rtf={row['rtf']:.3f} "
                    f"segments={row['n_nonempty']}/{row['n_seg']}"
                )

    failures = validation_failures(out)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(f"not writing {BASELINE.relative_to(ROOT)}", file=sys.stderr)
        sys.exit(1)

    BASELINE.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS: {DEFAULT_ENGINE} excess deletion <= {MAX_EXCESS_DEL_RATE:.1%} "
        f"and both-engine CER <= {MAX_CER:.1%} on "
        f"{len(out['stressors'][DEFAULT_ENGINE])} stressors"
    )
    print(f"wrote {BASELINE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
