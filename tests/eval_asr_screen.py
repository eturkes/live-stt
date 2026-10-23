#!/usr/bin/env python3
"""Screen one whisper model on one exact OpenVINO device: the evidence behind GPU mode.

Every arm decodes through the shipped path -- `WhisperEngine.decode_segments` for a
whole read clip, `replay._run_recognizer` (the real worker, VAC included) for
continuous speech -- and scores char CER with `cer.py`. Japanese arms also score
READING-level CER: both sides to hiragana through pykakasi, which strips the
kanji/kana style that char CER mostly measures on the Aozora narration
(`asr-pipeline.md` `## GPU mode` holds the table these arms reproduce).

Arms (`--arms`, comma-separated; the default runs all but `runaway` and `terms`):

- `fleurs`    150 FLEURS-ja + 150 FLEURS-en clips, whole clip, each under its own token
- `retention` the 182 s pause-free probe through VAC
- `long_form` 「ごん狐」 sections 01 + 03 through VAC
- `cv`        300 Common Voice 8 test clips through VAC
- `cvw`       the same 300 clips, whole clip
- `envac`     the 150 FLEURS-en clips through VAC under `<|en|>`
- `runaway`   the worst blocking span: 11 s of the probe decoded to the token cap
              under each token, best of 2 (`--max-new-tokens` moves the cap)
- `terms`     sections 01-06 as ONE session through VAC, the recogniser biased by
              the session's own terms (`--terms off|learner|seed`, seed = a user
              `--context`). The live path has no such channel; this arm carries the
              one it lost, so a new model can re-ask the question. Not on the NPU,
              which rejects `hotwords`.

Inputs re-derive from committed state and fail closed on a fingerprint: FLEURS =
`Random(20260916)`'s first 150 of each pinned FLEURS index (`short_corpus.json`,
`en_clips.json`); CV = `default_rng(20261001)`'s 300 rows of the pinned CV8 test
parquet (`fetch_real_clips.py`), each cut to 16 kHz with 0.5 s pads and a -60 dBFS
dither, then PCM16 -- the measured input, byte for byte; the retention probe and the
narration sections are hash-checked against their manifests.

On-demand, never a gate step: gitignored weights, corpora and minutes to an hour
of accelerator time per model. Both prelude halves first (`openvino-accel.md`);
`<intel-accel>` = the NPU driver farm dir the user-level instruction file names. Screen-only
models and their acquisition commands: `evidence-artifacts.md`.

    source <intel-accel>/env.sh && source .envrc && \\
        uv run --no-sync --with pykakasi==2.3.0 \\
        [--with soundfile==0.14.0 --with pyarrow==26.0.0] \\
        python tests/eval_asr_screen.py --model DIR --device NPU|GPU \\
        [--arms a,b] [--terms ARM] [--max-new-tokens N] [--vac-chunk S] [--json F]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import io
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

import fetch_real_clips  # noqa: E402
from build_caption_trace import pinned_wavs  # noqa: E402
from build_stressor import sha256_file  # noqa: E402

import live_stt  # noqa: E402
import replay  # noqa: E402  (after sys.path injection)
from cer import align, normalize  # noqa: E402

ARMS = ("fleurs", "retention", "long_form", "cv", "cvw", "envac", "runaway", "terms")
FLEURS_SEED, FLEURS_CLIPS = 20260916, 150
CV_SEED, CV_CLIPS = 20261001, 300
# sha256 over the selected inputs, so a drifted corpus, sampler or decoder changes
# a number this file can refuse rather than one a reader compares in good faith.
FLEURS_SHA256 = "043d9800b0e94a3ec5c030611c19800aa23d4b1ed39af49c9f08953f4831a93c"
CV_SHA256 = "f4d1e25a21cc471de92fad5bf48e0c9af75aeb51d04e6688cc6558afeca2c244"
RETENTION = TESTS / "retention_probe.json"
RETENTION_WAV = ROOT / "spike" / "backends" / "cache" / "retention_probe.wav"
LONG_FORM_SECTIONS = ("01", "03")
# What a user would type for this story: topic + character names (`--context`).
TERMS_SEED = "新美南吉の童話「ごん狐」の朗読。登場人物は、ごん、兵十、加助、茂平。"
# The names the seed carries, plus the forms the biased arms heard instead.
TERMS_NAMES = ("兵十", "加助", "茂平", "ごん", "ゴン", "権", "カスケ", "香助")
# Whisper's max_length, the cap a hallucination loop runs into.
WHISPER_MAX_TOKENS = 448
# The deleted channel's budget: Whisper keeps 223 prev-text tokens.
TERMS_MAX_CHARS = 160
NPU_LOCK = Path("/tmp/live-stt-npu.lock")  # noqa: S108  (one accelerator holder per box)


def _kana():
    import pykakasi  # pyright: ignore[reportMissingImports]  (uv run --with pykakasi)

    convert = pykakasi.kakasi().convert
    return lambda text: "".join(item["hira"] for item in convert(normalize(text)))


def score(pairs: list[tuple[str, str]], kana=None) -> dict:
    """Pooled char CER over (reference, hypothesis) pairs, plus reading-level CER."""
    n = s = d = i = 0
    for ref, hyp in pairs:
        r = normalize(ref)
        es, ed, ei = align(r, normalize(hyp))
        n, s, d, i = n + len(r), s + es, d + ed, i + ei
    row = {"N": n, "S": s, "D": d, "I": i, "cer": round((s + d + i) / n, 4)}
    if kana is not None:
        kn = ke = 0
        for ref, hyp in pairs:
            r = kana(ref)
            ke += sum(align(r, kana(hyp)))
            kn += len(r)
        row["kana"] = {"N": kn, "cer": round(ke / kn, 4)}
    return row


def quantiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {
        "n": len(xs),
        "p50": round(statistics.median(xs), 3),
        "p90": round(xs[int(0.9 * len(xs))], 3),
        "max": round(xs[-1], 3),
    }


def _fingerprint(parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part if isinstance(part, bytes) else part.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _refuse(name: str, actual: str, pinned: str) -> None:
    if actual != pinned:
        raise SystemExit(f"{name} inputs drifted\n  derived: {actual}\n  pinned:  {pinned}")


def fleurs_clips() -> list[dict]:
    """The 150 + 150 clips the screen measured, from the two pinned FLEURS indexes."""
    clips = []
    for lang, load in (
        ("ja", fetch_real_clips._load_cached),
        ("en", fetch_real_clips._load_cached_en),
    ):
        cached = load()
        if cached is None:
            raise SystemExit(f"absent: the pinned FLEURS-{lang} cache (tests/fetch_real_clips.py)")
        entries, corpus_dir = cached
        picked = [e for e in entries if e["corpus_id"].startswith(f"fleurs-{lang}-")]
        # The index holds the dataset's order, and its head is not a sample of it.
        random.Random(FLEURS_SEED).shuffle(picked)  # noqa: S311  (a sample, not a secret)
        clips += [
            {
                "id": e["corpus_id"],
                "lang": lang,
                "wav": corpus_dir / e["wav"],
                "ref": e["reference"],
            }
            for e in picked[:FLEURS_CLIPS]
        ]
    _refuse("FLEURS", _fingerprint(c["id"] for c in clips), FLEURS_SHA256)
    return clips


def cv_clips() -> list[dict]:
    """The 300 Common Voice clips, cut from the pinned parquet exactly as measured."""
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]  (uv run --with pyarrow)
    import soundfile as sf  # pyright: ignore[reportMissingImports]  (uv run --with soundfile)
    import soxr

    table = pq.read_table(fetch_real_clips.fetch_source(fetch_real_clips.CV_PARQUET))
    picks = np.random.default_rng(CV_SEED).choice(table.num_rows, CV_CLIPS, replace=False)
    clips, parts = [], []
    for idx in sorted(picks.tolist()):
        row = table.slice(idx, 1).to_pylist()[0]
        audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
        x = soxr.resample(audio.mean(axis=1), rate, live_stt.SAMPLE_RATE, quality="VHQ")
        pad = np.zeros(live_stt.SAMPLE_RATE // 2, np.float32)
        x = np.concatenate([pad, x.astype(np.float32), pad])
        x = (x + np.random.default_rng(idx).standard_normal(len(x)) * 10 ** (-60 / 20)).astype(
            np.float32
        )
        # The screen decoded these through a PCM16 WAV, so the quantisation is input.
        wav = io.BytesIO()
        sf.write(wav, np.clip(x, -1, 1), live_stt.SAMPLE_RATE, subtype="PCM_16", format="WAV")
        wav.seek(0)
        pcm, _ = sf.read(wav, dtype="int16")
        parts += [pcm.tobytes(), row["transcription"]]
        clips.append(
            {
                "id": f"cv-{idx}",
                "samples": pcm.astype(np.float32) / 32768.0,
                "ref": row["transcription"],
            }
        )
    _refuse("CV", _fingerprint(parts), CV_SHA256)
    return clips


def retention_probe() -> tuple[np.ndarray, str]:
    probe = json.loads(RETENTION.read_text(encoding="utf-8"))["probe"]
    if not RETENTION_WAV.exists():
        raise SystemExit(f"absent: {RETENTION_WAV} (tests/build_retention_probe.py)")
    _refuse("retention", sha256_file(RETENTION_WAV), probe["audio_sha256"])
    return replay.load_wav_f32_16k(RETENTION_WAV), probe["ja_ref"]


class _BiasedContext(live_stt.SessionContext):
    """The learner with the exclusion a biased recogniser needs (D-015).

    A term the model was handed proves nothing when it comes back, so it must not
    count as support or renew a lease. observe_ja already skips seed terms, so
    listing the supplied ones beside the seed for one call is that exclusion
    exactly: nothing else in the call reads `seed_terms` while no rendering exists,
    and this arm never translates.
    """

    supplied: frozenset[str] = frozenset()

    def observe_ja(self, text: str) -> None:
        seed = self.seed_terms
        self.seed_terms = [*seed, *self.supplied]
        try:
            super().observe_ja(text)
        finally:
            self.seed_terms = seed


def term_list(context: live_stt.SessionContext) -> tuple[str, frozenset[str]]:
    """The deleted `asr_hotwords`: trusted terms within budget, and the set it carries.

    A TERM LIST, never running transcript: recent captions on the same slot made
    the recogniser loop (`asr-pipeline.md`).
    """
    budget, used = TERMS_MAX_CHARS, []
    for term in context.terms():
        if budget >= len(term) + 1:
            used.append(term)
            budget -= len(term) + 1
    return "、".join(used), frozenset(used)


class _Pipeline:
    """The engine's real pipeline, with the arm's generate keywords appended.

    Wrapping the pipeline instead of the engine keeps the shipped `generate` --
    token, task, timestamps, `repetition_penalty` -- in every decode.
    """

    def __init__(self, pipeline, extra: dict, context: _BiasedContext | None = None):
        self._pipeline, self._extra, self.context = pipeline, extra, context
        self.hotwords, self._fresh = "", True
        self.cap = extra.get("max_new_tokens", WHISPER_MAX_TOKENS)

    @contextlib.contextmanager
    def forced(self, tokens: int):
        """Every decode inside runs to at least `tokens` new tokens."""
        extra = self._extra
        self._extra = {**extra, "min_new_tokens": tokens}
        try:
            yield
        finally:
            self._extra = extra

    def generate(self, samples, **keywords):
        if self.context is not None and self._fresh:
            # An utterance's first decode follows its speech start with the context
            # unchanged in between, which is where the live channel set the list.
            self.hotwords, offered = term_list(self.context)
            self.context.supplied = offered if self.hotwords else frozenset()
            self._fresh = False
        if self.hotwords:
            keywords["hotwords"] = self.hotwords
        return self._pipeline.generate(samples, **keywords, **self._extra)

    def utterance_closed(self) -> None:
        self._fresh = True


class Screen:
    def __init__(self, model: Path, device: str, extra: dict, terms: str | None):
        self.context = (
            None
            if terms in (None, "off")
            else _BiasedContext(TERMS_SEED if terms == "seed" else "")
        )
        self.rec = live_stt.WhisperEngine(model, device)
        self.pipeline = _Pipeline(self.rec._pipeline, extra, self.context)
        self.rec._pipeline = self.pipeline  # type: ignore[assignment]

    def whole(self, samples: np.ndarray, lang: str) -> tuple[str, float]:
        started = time.perf_counter()
        text, _ = self.rec.decode_segments(samples, language=lang)
        return text, time.perf_counter() - started

    def vac(self, samples: np.ndarray, lang: str) -> tuple[str, list[tuple[float, bool]]]:
        """One run of the real worker; the decodes it charged, final or not."""
        decodes: list[tuple[float, bool]] = []

        def on_update(_buffer_s, _end_s, _commit_s, _commit, final, decode_s):
            decodes.append((decode_s, final))
            if final:
                self.pipeline.utterance_closed()

        held, live_stt.ASR_LANGUAGE = live_stt.ASR_LANGUAGE, lang
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rows = asyncio.run(
                    replay._run_recognizer(samples, self.rec, on_update, self.context)
                )
        finally:
            live_stt.ASR_LANGUAGE = held
        return "".join(row["text"] for row in rows), decodes


def _updates(decodes: list[tuple[float, bool]]) -> dict:
    return quantiles([d for d, final in decodes if not final] or [0.0])


def run_arm(arm: str, screen: Screen, kana, inputs: dict) -> dict:
    if arm == "fleurs":
        rows, times = [], {"ja": [], "en": []}
        for clip in inputs["fleurs"]:
            hyp, took = screen.whole(replay.load_wav_f32_16k(clip["wav"]), clip["lang"])
            times[clip["lang"]].append(took)
            rows.append({"id": clip["id"], "lang": clip["lang"], "hyp": hyp})
        refs = {c["id"]: c["ref"] for c in inputs["fleurs"]}
        out = {}
        for lang in ("ja", "en"):
            pairs = [(refs[r["id"]], r["hyp"]) for r in rows if r["lang"] == lang]
            out[lang] = {
                **score(pairs, kana if lang == "ja" else None),
                "decode": quantiles(times[lang]),
            }
        return {**out, "rows": rows}
    if arm == "retention":
        samples, ref = inputs["retention"]
        hyp, decodes = screen.vac(samples, "ja")
        return {**score([(ref, hyp)], kana), "update": _updates(decodes), "hyp": hyp}
    if arm in ("long_form", "terms"):
        keys = LONG_FORM_SECTIONS if arm == "long_form" else tuple(inputs["sections"])
        sections, decodes = {}, []
        for key in keys:
            section, wav, _ = inputs["sections"][key]
            hyp, ran = screen.vac(replay.load_wav_f32_16k(wav), "ja")
            decodes += ran
            sections[key] = {"hyp": hyp, **score([(section["reference"]["text"], hyp)])}
        refs = [(inputs["sections"][k][0]["reference"]["text"], sections[k]["hyp"]) for k in keys]
        out = {**score(refs, kana), "update": _updates(decodes), "sections": sections}
        if arm == "terms":
            reference = "".join(ref for ref, _ in refs)
            heard = "".join(hyp for _, hyp in refs)
            out["names"] = {t: [reference.count(t), heard.count(t)] for t in TERMS_NAMES}
            out["trusted_end"] = screen.context.terms() if screen.context is not None else []
        return out
    if arm in ("cv", "cvw"):
        rows, times = [], []
        for clip in inputs["cv"]:
            if arm == "cv":
                hyp, ran = screen.vac(clip["samples"], "ja")
                times += [d for d, _ in ran]
            else:
                hyp, took = screen.whole(clip["samples"], "ja")
                times.append(took)
            rows.append({"id": clip["id"], "hyp": hyp})
        pairs = [(c["ref"], r["hyp"]) for c, r in zip(inputs["cv"], rows, strict=True)]
        return {**score(pairs, kana), "decode": quantiles(times), "rows": rows}
    if arm == "runaway":
        samples, _ = inputs["retention"]
        piece = samples[20 * live_stt.SAMPLE_RATE : 31 * live_stt.SAMPLE_RATE]
        out: dict = {"cap": screen.pipeline.cap}
        for lang in ("ja", "en"):
            with screen.pipeline.forced(screen.pipeline.cap - 8):
                out[lang] = round(min(screen.whole(piece, lang)[1] for _ in range(2)), 3)
        return out
    if arm == "envac":
        clips = [c for c in inputs["fleurs"] if c["lang"] == "en"]
        pairs, decodes = [], []
        for clip in clips:
            hyp, ran = screen.vac(replay.load_wav_f32_16k(clip["wav"]), "en")
            decodes += ran
            pairs.append((clip["ref"], hyp))
        return {**score(pairs), "clips": len(clips), "update": _updates(decodes)}
    raise ValueError(arm)


def headline(arm: str, row: dict) -> str:
    def cell(r: dict) -> str:
        return f"{r['cer']:.4f}" + (f" / {r['kana']['cer']:.4f}" if "kana" in r else "")

    if arm == "runaway":
        return f"cap {row['cap']} tokens: ja {row['ja']} s  en {row['en']} s"
    if arm == "fleurs":
        ja, en = row["ja"], row["en"]
        return (
            f"ja {cell(ja)}  en {cell(en)}  "
            f"decode p50 {ja['decode']['p50']} / {en['decode']['p50']} s"
        )
    timing = row["update"] if "update" in row else row["decode"]
    extra = f"  names {row['names']}  trusted {row['trusted_end']}" if arm == "terms" else ""
    return f"{cell(row)}  N={row['N']}  p50 {timing['p50']} max {timing['max']} s{extra}"


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model", type=Path, required=True, help="OpenVINO whisper model directory")
    ap.add_argument("--device", required=True, help="exact OpenVINO device (NPU, GPU, CPU)")
    ap.add_argument("--arms", default="fleurs,retention,long_form,cv,cvw,envac")
    ap.add_argument("--terms", choices=("off", "learner", "seed"), default="learner")
    ap.add_argument("--max-new-tokens", type=int, help="cap every decode (large-v3's EN arm: 224)")
    ap.add_argument("--vac-chunk", type=float, help=f"VAC cadence in s ({live_stt.VAC_CHUNK_S})")
    ap.add_argument("--json", type=Path, help="write every hypothesis + score here")
    args = ap.parse_args()

    arms = args.arms.split(",")
    if unknown := sorted(set(arms) - set(ARMS)):
        ap.error(f"unknown arms: {unknown}")
    if "terms" in arms and arms != ["terms"]:
        ap.error("run the terms arm alone: its session context would bias every other arm")
    if "terms" in arms and args.device == "NPU":
        ap.error("the terms arm needs hotwords, and the NPU rejects them")
    if args.vac_chunk is not None:
        live_stt.VAC_CHUNK_S = args.vac_chunk
    err = live_stt.check_device("whisper", args.device)
    if err:
        raise SystemExit(f"error: {err.splitlines()[0]}")

    kana = _kana()
    inputs: dict = {}
    if {"fleurs", "envac"} & set(arms):
        inputs["fleurs"] = fleurs_clips()
    if {"cv", "cvw"} & set(arms):
        inputs["cv"] = cv_clips()
    if {"retention", "runaway"} & set(arms):
        inputs["retention"] = retention_probe()
    if {"long_form", "terms"} & set(arms):
        inputs["sections"] = pinned_wavs()

    extra = {} if args.max_new_tokens is None else {"max_new_tokens": args.max_new_tokens}
    report = {
        "model": str(args.model),
        "device": args.device,
        "extra": extra,
        "vac_chunk_s": live_stt.VAC_CHUNK_S,
        "terms": args.terms if "terms" in arms else None,
    }
    # Two processes compiling one OpenVINO cache key corrupt it (toolchain.md).
    with NPU_LOCK.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        screen = Screen(args.model, args.device, extra, args.terms if "terms" in arms else None)
        for arm in arms:
            started = time.perf_counter()
            report[arm] = run_arm(arm, screen, kana, inputs)
            print(
                f"{arm:9s} {headline(arm, report[arm])}  ({time.perf_counter() - started:.0f} s)",
                flush=True,
            )
            if args.json:
                args.json.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
