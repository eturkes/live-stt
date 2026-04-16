"""Drive all prototypes against all scenarios, collect results.

Usage
    uv run python spike/backends/bench.py                 # all backends/clips
    uv run python spike/backends/bench.py --only gemini   # single backend
    uv run python spike/backends/bench.py --clip medium   # single clip

Loads `.env` from project root so API keys resolve automatically. Backends
whose key isn't set are recorded as SKIPPED (KEY_MISSING: VAR_NAME).

Outputs
    results.json  — full structured data
    results.md    — human-readable table + per-run transcripts
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

# Make sibling modules importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv  # noqa: E402

from harness import BenchResult, run_bench, summarize  # noqa: E402
from scenarios import CLIPS, ensure_all, load_clip  # noqa: E402


# Load .env from project root (two levels up from this file).
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


# Vendor pricing for hourly cost estimation (April 2026, list price, USD).
# Translation cascade adds ~$0.02/hr on top (gemini-flash-lite, ~20 turns/min).
PRICING_USD_PER_HOUR = {
    "gemini":   1.40,          # audio-in + unavoidable audio-out tokens
    "openai":   0.42,          # gpt-realtime-mini, output_modalities=['text']
    "deepgram": 0.55 + 0.02,   # Nova-3 streaming + cascade
    "elevenlabs": 0.39 + 0.02,  # Scribe v2 RT + cascade
    "azure":    2.50,          # Speech Translation integrated
}


BACKENDS = [
    # (name, module, key_env_var, extra_kwargs_from_env)
    ("gemini",     "prototype_gemini",           "GEMINI_API_KEY",     {}),
    ("openai",     "prototype_openai_realtime",  "OPENAI_API_KEY",     {}),
    ("deepgram",   "prototype_deepgram",         "DEEPGRAM_API_KEY",   {}),
    ("elevenlabs", "prototype_elevenlabs",       "ELEVENLABS_API_KEY", {}),
    ("azure",      "prototype_azure",            "AZURE_SPEECH_KEY",   {"region_env": "AZURE_SPEECH_REGION"}),
]


def _probe_keys() -> dict[str, str | None]:
    return {name: os.environ.get(key_env) for name, _, key_env, _ in BACKENDS}


async def _run_one(
    backend_name: str,
    module_name: str,
    key_env: str,
    extra: dict,
    clip_id: str,
    pcm: bytes,
    duration_s: float,
    translate: bool,
) -> BenchResult:
    api_key = os.environ.get(key_env)
    if not api_key:
        r = BenchResult(
            backend=backend_name,
            clip_id=clip_id,
            clip_duration_s=duration_s,
            translate=translate,
            skipped_reason=f"KEY_MISSING: {key_env}",
        )
        return r

    import importlib
    mod = importlib.import_module(module_name)

    kwargs: dict = {}
    if "region_env" in extra:
        kwargs["region"] = os.environ.get(extra["region_env"])

    r = await run_bench(
        backend=backend_name,
        stream_fn=mod.stream,
        clip_id=clip_id,
        pcm=pcm,
        duration_s=duration_s,
        translate=translate,
        api_key=api_key,
        **kwargs,
    )
    # Fill cost estimate based on the per-hour vendor rate.
    rate = PRICING_USD_PER_HOUR.get(backend_name)
    if rate is not None:
        r.est_cost_per_hr_usd = rate
        r.est_cost_usd = rate * (duration_s / 3600.0)
    return r


async def main_async(args):
    # Ensure all clips are synthesized/cached.
    print("Preparing scenario clips...")
    try:
        await ensure_all()
    except Exception as e:
        print(f"[scenarios] synthesis failed: {e}")
        print("Proceeding with whatever clips are cached.")

    clips_to_run = [c for c in CLIPS if not args.clip or c.id == args.clip]
    backends_to_run = [b for b in BACKENDS if not args.only or b[0] == args.only]

    results: list[BenchResult] = []

    for clip in clips_to_run:
        try:
            pcm, duration = load_clip(clip.id)
        except FileNotFoundError:
            print(f"[skip] clip {clip.id} not cached")
            continue
        for backend_name, module_name, key_env, extra in backends_to_run:
            print(f"\n=== {backend_name} × {clip.id} ===")
            t0 = time.monotonic()
            r = await _run_one(
                backend_name, module_name, key_env, extra,
                clip.id, pcm, duration, args.translate,
            )
            r.infos.append(f"wall_clock={time.monotonic() - t0:.2f}s")
            results.append(r)
            print(summarize(r))

    # Write outputs.
    out_dir = Path(__file__).parent
    (out_dir / "results.json").write_text(
        json.dumps([dataclasses.asdict(r) for r in results], indent=2, ensure_ascii=False)
    )
    _write_markdown(out_dir / "results.md", results)
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'results.md'}")


def _write_markdown(path: Path, results: list[BenchResult]):
    lines: list[str] = []
    lines.append("# Backends Spike — Results\n")
    lines.append("Generated by `bench.py`. See `DESIGN.md` for methodology and "
                 "`REPORT.md` for analysis.\n")

    # Summary table per clip
    clips = sorted({r.clip_id for r in results})
    backends = sorted({r.backend for r in results})
    lines.append("## Summary\n")
    lines.append("| clip | backend | connect | ttft | total | blocks | $/hr | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for clip_id in clips:
        for backend in backends:
            r = next((x for x in results if x.clip_id == clip_id and x.backend == backend), None)
            if r is None:
                continue
            status = r.skipped_reason or ("OK" if not r.errors else "ERR")
            row = "| {clip} | {b} | {c} | {t} | {tot} | {n} | {p} | {s} |".format(
                clip=r.clip_id,
                b=r.backend,
                c=f"{r.connect_s:.2f}" if r.connect_s is not None else "-",
                t=f"{r.ttft_s:.2f}" if r.ttft_s is not None else "-",
                tot=f"{r.total_s:.2f}" if r.total_s is not None else "-",
                n=r.n_blocks,
                p=f"${r.est_cost_per_hr_usd:.2f}" if r.est_cost_per_hr_usd is not None else "-",
                s=status,
            )
            lines.append(row)

    # Per-run detail with transcripts
    lines.append("\n## Per-run detail\n")
    for r in results:
        lines.append(f"### {r.backend} × {r.clip_id}\n")
        lines.append(f"- translate: {r.translate}")
        lines.append(f"- duration: {r.clip_duration_s:.2f}s")
        if r.skipped_reason:
            lines.append(f"- **SKIPPED**: {r.skipped_reason}")
            lines.append("")
            continue
        lines.append(f"- connect: {r.connect_s}")
        lines.append(f"- ttft: {r.ttft_s}")
        lines.append(f"- total: {r.total_s}")
        lines.append(f"- blocks: {r.n_blocks}")
        lines.append(f"- est $/hr: {r.est_cost_per_hr_usd}")
        if r.ja_text:
            lines.append(f"- JA: `{r.ja_text}`")
        if r.en_text:
            lines.append(f"- EN: `{r.en_text}`")
        if r.infos:
            lines.append(f"- infos: {r.infos}")
        if r.errors:
            lines.append(f"- **errors**: {r.errors}")
        lines.append("")

    path.write_text("\n".join(lines))


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", help="Run only this backend (gemini|openai|deepgram|elevenlabs|azure)")
    p.add_argument("--clip", help="Run only this clip id (see scenarios.py)")
    p.add_argument("--no-translate", dest="translate", action="store_false", default=True,
                   help="Skip English translation (transcribe only)")
    return p.parse_args()


def main():
    args = _parse_args()
    print("Probing API keys...")
    for name, key in _probe_keys().items():
        mark = "✓" if key else "—"
        print(f"  {mark} {name}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
