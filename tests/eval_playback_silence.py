"""--playback's digital-silence regime through the real worker: the `asr-pipeline.md` arms.

A sink monitor carries EXACT zeros whenever nothing plays, and `LeveledVad` admits no window
at or below `VAD_LEVEL_FLOOR_DBFS`, so its 10 s level history never ages across that silence.
This replays the playback-shaped stream offline: 14 clips (the 7 CV8 rows the old zero-pad
probe hit or controlled, then the 7 pinned replay clips), each resampled to 44.1 kHz by soxr
VHQ (PipeWire's stand-in), `GAP_S` of zeros between them after a `LEAD_S` lead-in, cut into
256-frame blocks through the shipped `Resampler` and decoded by the real `worker`. Arms:

  Z  as shipped
  N  a -60 dBFS seeded white floor on the whole 44.1 kHz stream (everyone hears it)
  V  Z to whisper; the floor on silero's copy before leveling (enters the level history)
  W  Z to whisper; the floor on silero's copy after leveling (gain unchanged)

Report-only: per-clip text with S/D/I against the reference, pooled CER, and the segments
starting outside every clip's span. The shipped arm reproduces the live `--playback` verdicts
(cv489 empty, cv894 without 先月, cv904 with an extra はい。). Inputs fail closed on
`INPUT_SHA256`. Needs the accelerator prelude (`openvino-accel.md`) and:

  uv run --with soundfile==0.14.0 --with pyarrow==26.0.0 python tests/eval_playback_silence.py \\
    [--arms ZNVW] [--device NPU|GPU] [--json]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

import fetch_real_clips  # noqa: E402

import live_stt  # noqa: E402
import replay  # noqa: E402
from cer import align, normalize  # noqa: E402

CAPTURE_RATE = 44100  # PortAudio's rate for PipeWire's ALSA plugin (the live `Playback:` line)
BLOCK = 256
LEAD_S = 3.0
GAP_S = 10.0
FLOOR_DBFS = -60.0
STREAM_SEED, VAD_SEED = 7, 8
# Quiet (speech p90 -37..-43 dBFS) rows the old zero-pad probe hit, then its two loud controls.
CV_ROWS = (110, 137, 489, 894, 904, 909, 915)
PINNED = ("short", "med", "long", "kana", "xlong", "multi", "paused")
PINNED_DIR = ROOT / "spike" / "backends" / "cache"
# sha256 over every clip's native-rate float32 PCM + its reference, in stream order.
INPUT_SHA256 = "c89805aba97e28115f1c9c38615f614c5a90018dee3c3fc9c9821e123b2cfcf0"
NPU_LOCK = Path("/tmp/live-stt-npu.lock")  # noqa: S108  (one accelerator holder per box)


def clips() -> list[tuple[str, np.ndarray, int, str]]:
    """(name, mono float32 PCM, rate, reference) in stream order, fingerprint-checked."""
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]  (uv run --with pyarrow)
    import soundfile as sf  # pyright: ignore[reportMissingImports]  (uv run --with soundfile)

    with contextlib.redirect_stdout(sys.stderr):  # its cache notice would corrupt --json
        table = pq.read_table(fetch_real_clips.fetch_source(fetch_real_clips.CV_PARQUET))
    out = []
    for idx in CV_ROWS:
        row = table.slice(idx, 1).to_pylist()[0]
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
        out.append((f"cv{idx}", audio.mean(axis=1).astype(np.float32), rate, row["transcription"]))
    refs = json.loads((TESTS / "real_clips.json").read_text(encoding="utf-8"))
    for name in PINNED:
        path = PINNED_DIR / f"cv_{name}.wav"
        if not path.exists():
            raise SystemExit(f"absent: {path} (tests/fetch_real_clips.py)")
        audio, rate = sf.read(str(path), dtype="float32")
        out.append((f"cv_{name}", np.ascontiguousarray(audio), rate, refs[f"cv_{name}"]["ja_ref"]))
    digest = hashlib.sha256()
    for name, pcm, rate, ref in out:
        digest.update(f"{name}\0{rate}\0{ref}\0".encode())
        digest.update(pcm.tobytes())
    if digest.hexdigest() != INPUT_SHA256:
        raise SystemExit(
            f"inputs drifted\n  derived: {digest.hexdigest()}\n  pinned:  {INPUT_SHA256}"
        )
    return out


def stream(inputs) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """The clips at CAPTURE_RATE behind LEAD_S and between GAP_S of zeros, with their spans."""
    import soxr

    parts, spans, t = [np.zeros(int(LEAD_S * CAPTURE_RATE), np.float32)], [], LEAD_S
    for name, pcm, rate, _ in inputs:
        y = soxr.resample(pcm, rate, CAPTURE_RATE, quality="VHQ").astype(np.float32)
        spans.append((name, t, t + len(y) / CAPTURE_RATE))
        parts += [y, np.zeros(int(GAP_S * CAPTURE_RATE), np.float32)]
        t += len(y) / CAPTURE_RATE + GAP_S
    return np.concatenate(parts), spans


def _floor(rng: np.random.Generator, n: int) -> np.ndarray:
    """float64; N and V add it cast to float32, W adds it at full width (the arms as measured)."""
    return rng.standard_normal(n) * 10 ** (FLOOR_DBFS / 20)


@contextlib.contextmanager
def _vad_arm(arm: str):
    """V and W rewire what silero hears for one arm, then restore the shipped method."""
    shipped = live_stt.LeveledVad.accept_waveform
    rng = np.random.default_rng(VAD_SEED)

    def before_leveling(self, samples):
        return shipped(self, (samples + _floor(rng, len(samples)).astype(np.float32)))

    def after_leveling(self, samples):
        # The shipped body with the floor added past the gain, so it never enters the history.
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms > 10 ** (live_stt.VAD_LEVEL_FLOOR_DBFS / 20):
            self._levels.append(rms)
            if len(self._levels) >= self._min_levels:
                level = float(np.percentile(self._levels, live_stt.VAD_LEVEL_PERCENTILE))
                target = 10 ** (live_stt.VAD_TARGET_DBFS / 20)
                self.gain = min(live_stt.VAD_MAX_GAIN, max(1.0, target / level))
        out = np.clip(samples * self.gain, -1.0, 1.0) if self.gain != 1.0 else samples
        floored = (out + _floor(rng, len(samples))).astype(np.float32)
        live_stt.sherpa_onnx.VoiceActivityDetector.accept_waveform(self, floored)

    live_stt.LeveledVad.accept_waveform = {"V": before_leveling, "W": after_leveling}.get(
        arm, shipped
    )
    try:
        yield
    finally:
        live_stt.LeveledVad.accept_waveform = shipped


def run_arm(arm: str, rec, x441: np.ndarray, spans, refs: dict[str, str]) -> dict:
    if arm == "N":
        x441 = x441 + _floor(np.random.default_rng(STREAM_SEED), len(x441)).astype(np.float32)
    resampler = live_stt.Resampler(CAPTURE_RATE)
    blocks = (resampler(x441[i : i + BLOCK]) for i in range(0, len(x441), BLOCK))
    samples = np.concatenate([*blocks, resampler.flush()])
    with _vad_arm(arm), contextlib.redirect_stdout(io.StringIO()):
        rows = asyncio.run(replay._run_recognizer(samples, rec))
    owned: dict[str, list[str]] = {name: [] for name, _, _ in spans}
    outside = []
    for row in rows:
        start = row["start"] / live_stt.SAMPLE_RATE
        hit = [name for name, a, b in spans if a - 0.5 <= start <= b + 1.0]
        if hit:
            owned[hit[0]].append(row["text"])
        else:
            outside.append([round(start, 2), row["text"]])
    per_clip, errors, total = {}, 0, 0
    for name, _, _ in spans:
        hyp = "".join(owned[name])
        ref = normalize(refs[name])
        s, d, i = align(ref, normalize(hyp))
        errors, total = errors + s + d + i, total + len(ref)
        per_clip[name] = {"S": s, "D": d, "I": i, "segments": len(owned[name]), "text": hyp}
    return {"cer": round(errors / total, 4), "errors": errors, "N": total, "clips": per_clip,
            "outside": outside}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--arms", default="ZNVW", help="any of Z N V W (default: all four)")
    ap.add_argument("--device", default=live_stt.ASR_DEVICE, choices=live_stt.ASR_DEVICES)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if bad := set(args.arms) - set("ZNVW"):
        ap.error(f"unknown arms: {''.join(sorted(bad))}")
    inputs = clips()
    x441, spans = stream(inputs)
    refs = {name: ref for name, _, _, ref in inputs}
    if err := live_stt.check_device("whisper", args.device):
        raise SystemExit(f"absent: {err}")
    with NPU_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rec = live_stt.load_recognizer("whisper", args.device)
        result = {arm: run_arm(arm, rec, x441, spans, refs) for arm in args.arms}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return
    for arm, r in result.items():
        for name, c in r["clips"].items():
            print(f"{arm} {name:9} S{c['S']} D{c['D']} I{c['I']} {c['segments']} seg {c['text']}")
        print(f"{arm} TOTAL CER {r['cer']:.4f} ({r['errors']}/{r['N']}) outside {r['outside']}")


if __name__ == "__main__":
    main()
