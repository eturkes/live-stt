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
path, so the excess isolates the length + segmentation effect. It is not a purely
acoustic length isolate: any decode difference at a ~10 ms equal-power join rides
along too, but that join is constant-power (no dip, no inserted silence) by
design and the continuity check below confirms it creates no segmentation
artifact -- so the residual join contribution is bounded and acoustically
transparent. cer.py supplies normalize + S/D/I alignment.

Acceptance is proven, not asserted loosely (see validate()):
  continuity -- every crossfade join offset lies INSIDE a VAD speech segment, so
    no join was split on (a real geometric test, not "fewer segments than clips").
  soft cap  -- re-segmenting with the cap raised (control VAD) yields fewer
    segments than with it on: the cap CAUSES the mid-stream cut, and each engine
    decodes a >=20 s pre-flush segment.
  excess    -- k2v2 excess deletion >= EXCESS_TARGET on both stressors.
The full per-(stressor, engine) validation matrix + the generated-WAV sha256 are
persisted into the manifest, so the committed numbers are reproducible and
self-substantiating; a failing run writes nothing and exits nonzero.

Outputs (deny-listed cache constructed here, never named on a command line; the
manifest sits beside real_clips.json):
    spike/backends/cache/stress_long.wav   (>=35 s continuous)
    spike/backends/cache/stress_med.wav    (~20 s continuous)
    tests/stressor_clips.json              (refs + recipe + baselines + validation)

    uv run python tests/build_stressor.py            # build + QC + validate + write
"""

from __future__ import annotations

import hashlib
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
CAP_OFF_S = 120.0  # control VAD cap: above any stressor (so effectively off) yet
#                    far below sherpa's int32 overflow (>=~1e6 s * 16 kHz wraps
#                    the sample count -> spurious splits). At 120 s the whole
#                    continuous stressor is ONE segment; the 20 s cap splits it.


def load_component(cid: str) -> np.ndarray:
    """Cached CV clip as float32 mono @ SAMPLE_RATE (the pipeline representation)."""
    return replay.load_wav_f32_16k(CACHE / f"{cid}.wav")


def vad_segments(samples: np.ndarray, max_speech_s: float | None = None) -> list[tuple[int, int]]:
    """[(start, length), ...] silero speech segments over `samples`, in samples.

    Feeds make_vad() exactly as worker() does (window frames, remainder to flush)
    so the segmentation matches where the live pipeline hears speech. max_speech_s
    raises the soft cap for the control run (see validate()).
    """
    vad, window = make_vad(max_speech_s)
    pos, n = 0, len(samples)
    while n - pos >= window:
        vad.accept_waveform(samples[pos : pos + window])
        pos += window
    if pos < n:
        vad.accept_waveform(samples[pos:n])
    vad.flush()
    segs: list[tuple[int, int]] = []
    while not vad.empty():
        segs.append((int(vad.front.start), len(vad.front.samples)))
        vad.pop()
    return segs


def speech_extent(samples: np.ndarray) -> tuple[int, int]:
    """[first onset, last offset) in samples, per the production silero VAD.

    Silero opens 0.2-0.7 s late (D-010), so the extent clips onsets -- captured in
    each component's baseline D and cancelled by the excess metric.
    """
    segs = vad_segments(samples)
    if not segs:  # no speech detected -> keep the clip whole
        return 0, len(samples)
    return segs[0][0], segs[-1][0] + segs[-1][1]


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


def join_offsets(part_lens: list[int], xfade: int, lead: int) -> list[int]:
    """Join-region center sample offsets in the FRAMED audio, one per join.

    crossfade_join places the join into part i at pre-frame offset [L-xfade, L)
    where L is the current output length; the framed audio shifts everything by
    `lead`. Used to prove each join sits inside a VAD segment (no split there).
    """
    centers: list[int] = []
    cum = part_lens[0]  # output length after part 0
    for i in range(1, len(part_lens)):
        centers.append(lead + cum - xfade // 2)
        cum += part_lens[i] - xfade
    return centers


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_hyp(path: Path, engine: str) -> tuple[str, list[dict]]:
    """Replay a WAV through the live pipeline; return (concat text, segment rows)."""
    rep = replay.replay_wav(path, engine)
    text = "".join(s["text"] for s in rep["segments"] if s["text"])
    return text, rep["segments"]


def cycle_order(trimmed_dur: dict[str, float], target_s: float) -> list[str]:
    """Component ids cycled in COMPONENTS order until trimmed length >= target_s."""
    cycle_total = sum(trimmed_dur[c] for c in COMPONENTS)
    if cycle_total <= 0.0:  # every component empty -> the loop could not advance
        raise ValueError("cannot reach target: all component durations are zero")
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
    lead = int(LEAD_S * SAMPLE_RATE)
    stressors: dict[str, dict] = {}
    for name, target in (("stress_long", STRESS_LONG_S), ("stress_med", STRESS_MED_S)):
        order = cycle_order(trimmed_dur, target)
        joined = crossfade_join([trimmed[c] for c in order], xfade)
        audio = frame(joined, LEAD_S, TAIL_S)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1.0:  # equal-power blend can top full scale -> normalize, not clip
            audio = (audio / peak).astype(np.float32)
        write_wav(CACHE / f"{name}.wav", audio)
        stressors[name] = {
            "ja_ref": " ".join(refs[c] for c in order),
            "order": order,
            "component_count": len(order),
            "audio_s": round(len(audio) / SAMPLE_RATE, 3),
            "peak": round(peak, 4),
            "audio_sha256": sha256_file(CACHE / f"{name}.wav"),
            "join_samples": join_offsets([len(trimmed[c]) for c in order], xfade, lead),
        }
        print(f"built {name}: {stressors[name]['audio_s']:.2f} s, {len(order)} components, peak {peak:.3f}")

    # 4. Validate BEFORE writing the manifest: a failing run must persist nothing.
    ok, validation = validate(stressors, baselines)
    if not ok:
        print("\nvalidation FAILED -- manifest not written", file=sys.stderr)
        sys.exit(1)

    # 5. Manifest beside real_clips.json (refs + recipe + baselines + validation).
    for name in stressors:
        stressors[name]["validation"] = validation[name]
    manifest = {
        "recipe": {
            "source_clips": COMPONENTS,
            "trim": "silero speech extent via make_vad()",
            "join": "equal-power crossfade",
            "crossfade_ms": round(CROSSFADE_S * 1000, 1),
            "lead_s": LEAD_S,
            "tail_s": TAIL_S,
            "excess_metric": "stressor D - sum(component baseline D), over concat ref chars",
            "excess_isolates": "length + segmentation (join is constant-power, no dip)",
            "continuity_proof": "every join_samples offset inside a VAD segment",
            "softcap_proof": "cap-on VAD segments > cap-off (control) segments",
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


def validate(stressors: dict[str, dict], baselines: dict[str, dict]) -> tuple[bool, dict]:
    """Prove the three T9.1 acceptance signals; return (passed, per-stressor report).

    continuity  -- every crossfade join offset lands inside a VAD speech segment.
    soft cap    -- cap-on VAD yields more segments than the cap-off control, and
                   each engine decodes a >=20 s pre-flush (non-final) segment.
    excess      -- k2v2 excess deletion >= EXCESS_TARGET on both stressors.
    """
    print("\n== acceptance ==")
    report: dict[str, dict] = {}
    for name, meta in stressors.items():
        audio = replay.load_wav_f32_16k(CACHE / f"{name}.wav")  # exact decoder input
        segs_on = vad_segments(audio)  # production soft cap (20 s)
        segs_off = vad_segments(audio, max_speech_s=CAP_OFF_S)  # control: cap off
        joins = meta["join_samples"]
        # Continuity = every join lies inside a NATURAL (cap-off) speech segment,
        # so no join is a VAD-splittable gap. Cap-off is the honest view: the soft
        # cap's own cut would otherwise masquerade as a "gap" near a join.
        joins_inside = all(
            any(s <= jc < s + length for (s, length) in segs_off) for jc in joins
        )
        soft_cap_fired = len(segs_on) > len(segs_off)

        order = meta["order"]
        ref_norm = normalize(meta["ja_ref"])
        n = len(ref_norm)
        per_engine: dict[str, dict] = {}
        for engine in ENGINES:
            _, segs = decode_hyp(CACHE / f"{name}.wav", engine)
            hyp = "".join(s["text"] for s in segs if s["text"])
            _, raw_d, _ = align(ref_norm, normalize(hyp))
            baseline_d = sum(baselines[c][engine]["D"] for c in order)
            excess_rate = (raw_d - baseline_d) / n if n else 0.0
            preflush = max((s["dur_s"] for s in segs[:-1]), default=0.0)
            per_engine[engine] = {
                "n_seg": len(segs),
                "preflush_s": round(preflush, 3),
                "raw_D": raw_d,
                "baseline_D": baseline_d,
                "excess_D": raw_d - baseline_d,
                "excess_rate": round(excess_rate, 4),
            }
        report[name] = {
            "vad_segs_capon": len(segs_on),
            "vad_segs_capoff": len(segs_off),
            "joins_inside_segments": joins_inside,
            "soft_cap_fired": soft_cap_fired,
            "per_engine": per_engine,
        }
        for engine in ENGINES:
            pe = per_engine[engine]
            print(
                f"{name}/{engine}: {pe['n_seg']} seg (cap on {len(segs_on)} > off "
                f"{len(segs_off)}: {'ok' if soft_cap_fired else 'FAIL'}), "
                f"joins_inside={joins_inside}, pre-flush {pe['preflush_s']:.1f}s, "
                f"D {pe['raw_D']}-{pe['baseline_D']}={pe['excess_D']}/{n} "
                f"excess {pe['excess_rate']:+.1%}"
            )

    continuity_ok = all(report[nm]["joins_inside_segments"] for nm in stressors)
    softcap_ok = all(report[nm]["soft_cap_fired"] for nm in stressors) and all(
        report[nm]["per_engine"][e]["preflush_s"] >= PREFLUSH_MIN_S
        for nm in stressors
        for e in ENGINES
    )
    excess_ok = all(
        report[nm]["per_engine"]["k2v2"]["excess_rate"] >= EXCESS_TARGET for nm in stressors
    )
    verdict = continuity_ok and softcap_ok and excess_ok
    print(
        f"\ncontinuity(joins_inside)={continuity_ok} "
        f"softcap(cap-caused + preflush>={PREFLUSH_MIN_S:.0f}s)={softcap_ok} "
        f"k2v2_excess>={EXCESS_TARGET:.0%}(both)={excess_ok} -> "
        f"{'PASS' if verdict else 'FAIL'}"
    )
    return verdict, report


if __name__ == "__main__":
    main()
