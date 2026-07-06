#!/usr/bin/env python3
"""Build the continuous-speech stressor corpus (T9.1). Deterministic, no network.

Long pause-less Japanese speech is where the offline decoder deletes content
wholesale (L-023, D-010) and silero's max_speech_duration soft cap (20 s default,
unset by make_vad) cuts mid-stream. The short CV corpus (tests/real_clips.json)
cannot exercise this -- every clip is one short sentence closing on its own
silence. This tool synthesizes a genuinely continuous stressor from those real
clips: silero-trim each to its speech extent, equal-power crossfade-join (~10 ms)
the trimmed extents into a >=35 s stressor plus a ~20 s medium variant, cycling
the source clips to reach the target length.

Continuity is the whole point (L-023): gap-concatenation leaves clip-edge quiet
the VAD rightly reads as silence and splits on, so the segments never grow long
enough to collapse. Trimming to the speech extent and crossfading removes every
splittable gap -- the VAD then sees one long utterance and the soft cap fires.

Excess-deletion methodology (the honest headline): the silero trim clips onsets
(silero opens 0.2-0.7 s late), so a stressor's raw deletion count overstates the
length effect. Each trimmed component is therefore decoded in isolation through
the very same pipeline -- its per-component baseline D -- and the reported metric
is EXCESS deletion = stressor D - sum(component baseline D) over the concatenated
reference. Baseline and stressor share the identical trim + framing + decode
path, so the excess isolates the length/segmentation effect alone. cer.py
supplies normalize + S/D/I alignment.

Outputs (deny-listed cache constructed here, never named on a command line; the
manifest sits beside real_clips.json):
    spike/backends/cache/stress_long.wav   (>=35 s continuous)
    spike/backends/cache/stress_med.wav    (~20 s continuous)
    tests/stressor_clips.json              (refs + recipe + per-component baselines)

    uv run python tests/build_stressor.py            # build + QC + validate + write
"""

from __future__ import annotations

import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import replay  # noqa: E402  (after sys.path injection)
from cer import align, normalize  # noqa: E402
from live_stt import SAMPLE_RATE, check_models, make_vad  # noqa: E402

CACHE = ROOT / "spike" / "backends" / "cache"
REAL_CLIPS = ROOT / "tests" / "real_clips.json"
MANIFEST = ROOT / "tests" / "stressor_clips.json"
ENGINES = ["k2v2", "parakeet"]

# Source components: the real single-utterance CV clips (real_clips.json SINGLES).
# The multi-utterance concats (cv_multi/cv_paused) carry internal silence and are
# excluded -- a stressor component must be one continuous utterance.
COMPONENTS = ["cv_short", "cv_med", "cv_long", "cv_kana", "cv_xlong"]

CROSSFADE_S = 0.010  # equal-power overlap per join: long enough to bridge one
#                      onset/offset, short enough to leave no splittable dip.
LEAD_S, TAIL_S = 0.3, 0.6  # frame the whole stressor like real_clips: a clean
#                            first onset for the pre-pad, a tail > VAD_MIN_SILENCE.
STRESS_LONG_S = 40.0  # trimmed-speech target; > 2x the 20 s soft cap so the cut
#                       fires mid-stream and a second long segment still forms.
STRESS_MED_S = 20.0
PREFLUSH_MIN_S = 20.0  # a soft-cap cut must yield a pre-flush segment this long.
EXCESS_TARGET = 0.10  # k2v2 excess-deletion acceptance floor (planning: ~0.13).


def load_component(cid: str) -> np.ndarray:
    """Cached CV clip as float32 mono @ SAMPLE_RATE (the pipeline representation)."""
    return replay.load_wav_f32_16k(CACHE / f"{cid}.wav")


def speech_extent(samples: np.ndarray) -> tuple[int, int]:
    """[first onset, last offset) in samples, per the production silero VAD.

    Feeds make_vad() exactly as worker() does (window frames, remainder to flush)
    so the trimmed extent matches where the live pipeline hears speech begin/end.
    Silero opens 0.2-0.7 s late (D-010), so the extent clips onsets -- captured in
    each component's baseline D and cancelled by the excess metric.
    """
    vad, window = make_vad()
    pos, n = 0, len(samples)
    while n - pos >= window:
        vad.accept_waveform(samples[pos : pos + window])
        pos += window
    if pos < n:
        vad.accept_waveform(samples[pos:n])
    vad.flush()
    lo: int | None = None
    hi = 0
    while not vad.empty():
        start = int(vad.front.start)
        length = len(vad.front.samples)
        vad.pop()
        if lo is None:
            lo = start
        hi = start + length
    if lo is None:  # no speech detected -> keep the clip whole
        return 0, n
    return lo, hi


def crossfade_join(parts: list[np.ndarray], xfade: int) -> np.ndarray:
    """Equal-power crossfade-concatenate `parts`, overlapping `xfade` samples/join.

    Equal-power (sin/cos) rather than linear so two uncorrelated speech signals
    keep roughly constant power across the join and leave no amplitude dip for
    the VAD to split on.
    """
    if not parts:
        return np.zeros(0, dtype=np.float32)
    out = parts[0]
    for nxt in parts[1:]:
        k = min(xfade, len(out), len(nxt))
        if k <= 0:
            out = np.concatenate([out, nxt])
            continue
        t = (np.arange(k, dtype=np.float32) + 0.5) / k
        fade_in = np.sin(0.5 * np.pi * t)
        fade_out = np.cos(0.5 * np.pi * t)
        blend = out[-k:] * fade_out + nxt[:k] * fade_in
        out = np.concatenate([out[:-k], blend, nxt[k:]])
    return np.ascontiguousarray(out, dtype=np.float32)


def frame(samples: np.ndarray, lead_s: float, tail_s: float) -> np.ndarray:
    """Pad with `lead_s` of leading and `tail_s` of trailing silence."""
    lead = np.zeros(int(lead_s * SAMPLE_RATE), dtype=np.float32)
    tail = np.zeros(int(tail_s * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([lead, samples, tail])


def write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def decode_hyp(path: Path, engine: str) -> tuple[str, list[dict]]:
    """Replay a WAV through the live pipeline; return (concat text, segment rows)."""
    rep = replay.replay_wav(path, engine)
    text = "".join(s["text"] for s in rep["segments"] if s["text"])
    return text, rep["segments"]


def cycle_order(trimmed_dur: dict[str, float], target_s: float) -> list[str]:
    """Component ids cycled in COMPONENTS order until trimmed length >= target_s."""
    order: list[str] = []
    total = 0.0
    i = 0
    while total < target_s:
        cid = COMPONENTS[i % len(COMPONENTS)]
        order.append(cid)
        total += trimmed_dur[cid]
        i += 1
    return order


def main() -> None:
    for engine in ENGINES:
        err = check_models(engine)
        if err:
            print(f"error: engine {engine}: {err.splitlines()[0]}", file=sys.stderr)
            sys.exit(1)
    CACHE.mkdir(parents=True, exist_ok=True)
    refs = {cid: m["ja_ref"] for cid, m in json.loads(REAL_CLIPS.read_text("utf-8")).items()}

    # 1. Trim each component to its silero speech extent.
    trimmed: dict[str, np.ndarray] = {}
    trimmed_dur: dict[str, float] = {}
    for cid in COMPONENTS:
        lo, hi = speech_extent(load_component(cid))
        seg = np.ascontiguousarray(load_component(cid)[lo:hi], dtype=np.float32)
        trimmed[cid] = seg
        trimmed_dur[cid] = len(seg) / SAMPLE_RATE
        print(f"trim {cid}: {trimmed_dur[cid]:.2f} s speech")

    # 2. Component QC: isolated decode (same trim + framing + pipeline) per engine.
    baselines: dict[str, dict] = {cid: {} for cid in COMPONENTS}
    with tempfile.TemporaryDirectory() as td:
        for cid in COMPONENTS:
            wav = Path(td) / f"{cid}.wav"
            write_wav(wav, frame(trimmed[cid], LEAD_S, TAIL_S))
            for engine in ENGINES:
                hyp, _ = decode_hyp(wav, engine)
                r = normalize(refs[cid])
                s, d, ins = align(r, normalize(hyp))
                baselines[cid][engine] = {"hyp": hyp, "N": len(r), "S": s, "D": d, "I": ins}
                print(f"  baseline {cid}/{engine}: D={d} N={len(r)} :: {hyp}")

    # 3. Build the two stressors: crossfade-join cycled components, then frame.
    xfade = int(CROSSFADE_S * SAMPLE_RATE)
    stressors: dict[str, dict] = {}
    for name, target in (("stress_long", STRESS_LONG_S), ("stress_med", STRESS_MED_S)):
        order = cycle_order(trimmed_dur, target)
        audio = frame(crossfade_join([trimmed[c] for c in order], xfade), LEAD_S, TAIL_S)
        write_wav(CACHE / f"{name}.wav", audio)
        stressors[name] = {
            "ja_ref": " ".join(refs[c] for c in order),
            "order": order,
            "component_count": len(order),
            "audio_s": round(len(audio) / SAMPLE_RATE, 3),
        }
        print(f"built {name}: {stressors[name]['audio_s']:.2f} s, {len(order)} components")

    # 4. Manifest beside real_clips.json (refs + recipe + per-component baselines).
    manifest = {
        "recipe": {
            "source_clips": COMPONENTS,
            "trim": "silero speech extent via make_vad()",
            "join": "equal-power crossfade",
            "crossfade_ms": round(CROSSFADE_S * 1000, 1),
            "lead_s": LEAD_S,
            "tail_s": TAIL_S,
            "excess_metric": "stressor D - sum(component baseline D), over concat ref chars",
        },
        "components": {
            cid: {
                "ja_ref": refs[cid],
                "trimmed_dur_s": round(trimmed_dur[cid], 3),
                "baseline": baselines[cid],
            }
            for cid in COMPONENTS
        },
        "stressors": stressors,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")

    validate(stressors, baselines, refs)


def validate(stressors: dict[str, dict], baselines: dict[str, dict], refs: dict[str, str]) -> None:
    """Replay the stressors and report the three T9.1 acceptance signals."""
    print("\n== acceptance ==")
    ok_continuity = True
    ok_preflush = False
    ok_excess = False
    for name, meta in stressors.items():
        order = meta["order"]
        ref_norm = normalize(meta["ja_ref"])
        n = len(ref_norm)
        for engine in ENGINES:
            _, segs = decode_hyp(CACHE / f"{name}.wav", engine)
            hyp = "".join(s["text"] for s in segs if s["text"])
            _, raw_d, _ = align(ref_norm, normalize(hyp))
            baseline_d = sum(baselines[c][engine]["D"] for c in order)
            excess_rate = (raw_d - baseline_d) / n if n else 0.0
            preflush = max((s["dur_s"] for s in segs[:-1]), default=0.0)
            n_seg = len(segs)
            print(
                f"{name}/{engine}: {n_seg} seg (< {len(order)} comp: "
                f"{'ok' if n_seg < len(order) else 'FAIL'}), "
                f"pre-flush max {preflush:.1f}s, "
                f"D {raw_d}-{baseline_d}={raw_d - baseline_d}/{n} "
                f"excess {excess_rate:+.1%}"
            )
            if n_seg >= len(order):
                ok_continuity = False
            if preflush >= PREFLUSH_MIN_S:
                ok_preflush = True
            if name == "stress_long" and engine == "k2v2" and excess_rate >= EXCESS_TARGET:
                ok_excess = True
    verdict = all((ok_continuity, ok_preflush, ok_excess))
    print(
        f"\ncontinuity={ok_continuity} preflush>={PREFLUSH_MIN_S:.0f}s={ok_preflush} "
        f"k2v2_excess>={EXCESS_TARGET:.0%}={ok_excess} -> {'PASS' if verdict else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
