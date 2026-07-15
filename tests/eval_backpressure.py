#!/usr/bin/env python3
"""Deterministic paced-replay backpressure harness (M9.3).

Drives the production ``live_stt.worker`` with the production silero VAD, ring
buffer, a real bounded ``asyncio.Queue``, and ``enqueue_audio`` drop policy.
Only decoding is replaced: the event loop's executor and ``live_stt._decode``
are patched so each segment costs ``segment_duration * decode_rtf`` on a virtual
sample clock. Audio blocks arrive every ``chunk_ms`` on that same clock. No
wall-clock delay or recognizer model participates, so arrival-sampled queue
depth and drops are deterministic.

The default defect-B reproduction models a PipeWire-like 20 ms callback and a
slow host decoding at RTF 0.20. A ~20 s stressor segment then stalls ``worker``
for ~4 s, beyond ``AUDIO_QUEUE_MAX=100``'s ~2 s of block headroom; the long
stressor drops audio while every clip in the ordinary short corpus stays clean.

Requires ``models/silero_vad.onnx`` and the gitignored replay WAV corpus:

    uv run python tests/eval_backpressure.py
    uv run python tests/eval_backpressure.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import heapq
import io
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import live_stt  # noqa: E402
import replay  # noqa: E402
from live_stt import (  # noqa: E402
    AUDIO_QUEUE_MAX,
    SAMPLE_RATE,
    VAD_MODEL,
    State,
    enqueue_audio,
    make_vad,
    worker,
)

CACHE = ROOT / "spike" / "backends" / "cache"
REAL_CLIPS = ROOT / "tests" / "real_clips.json"
SHORT_CLIPS = tuple(json.loads(REAL_CLIPS.read_text(encoding="utf-8")))
STRESSOR = "stress_long"

DEFAULT_CHUNK_MS = 20.0
DEFAULT_DECODE_RTF = 0.20
_SETTLE_TURNS = 4


class _VirtualClock:
    """Minimal discrete-event clock for asyncio tasks; ticks are audio samples."""

    def __init__(self) -> None:
        self.now = 0
        self._order = 0
        self._events: list[tuple[int, int, asyncio.Future, object]] = []

    def after(self, ticks: int, result: object = None) -> asyncio.Future:
        future = asyncio.get_running_loop().create_future()
        self._order += 1
        heapq.heappush(self._events, (self.now + ticks, self._order, future, result))
        return future

    async def sleep(self, ticks: int) -> None:
        await self.after(ticks)

    async def _settle(self) -> None:
        # Queue.put_nowait wakes a getter Future, which then schedules its Task.
        # Four zero-time turns drain that finite callback chain without advancing
        # wall time. The worker then waits on the queue or this clock again.
        for _ in range(_SETTLE_TURNS):
            await asyncio.sleep(0)

    async def run_until_done(self, tasks: tuple[asyncio.Task, ...]) -> None:
        await self._settle()
        while not all(task.done() for task in tasks):
            while self._events and self._events[0][2].done():
                heapq.heappop(self._events)
            if not self._events:
                await self._settle()
                while self._events and self._events[0][2].done():
                    heapq.heappop(self._events)
                if all(task.done() for task in tasks):
                    break
                if not self._events:
                    raise RuntimeError("virtual-clock deadlock: tasks wait with no timed event")

            tick = self._events[0][0]
            if tick < self.now:
                raise RuntimeError("virtual clock moved backwards")
            self.now = tick
            while self._events and self._events[0][0] == tick:
                _, _, future, result = heapq.heappop(self._events)
                if not future.done():
                    future.set_result(result)
            await self._settle()

        await asyncio.gather(*tasks)


async def _run_paced(
    samples: np.ndarray,
    chunk_samples: int,
    decode_rtf: float,
    queue_max: int,
) -> dict:
    clock = _VirtualClock()
    state = State()
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
    vad, window = make_vad()
    timeline: list[dict[str, int | bool]] = []
    decodes: list[dict[str, int]] = []

    async def produce() -> None:
        for start in range(0, len(samples), chunk_samples):
            chunk = samples[start : start + chunk_samples]
            accepted = enqueue_audio(audio_q, state, chunk)
            timeline.append(
                {
                    "at_sample": clock.now,
                    "depth": audio_q.qsize(),
                    "dropped": state.dropped,
                    "accepted": accepted,
                }
            )
            await clock.sleep(len(chunk))
        # Unlike the live Ctrl+C path, replay has no dead worker and should retain
        # every accepted tail block. Waiting for a slot avoids an uncounted eviction.
        await audio_q.put(None)

    def fake_decode(_rec: object, _samples: np.ndarray) -> str:
        return ""

    def fake_run_in_executor(_executor, fn, *args):
        segment = args[-1]
        cost = max(1, round(len(segment) * decode_rtf))
        decodes.append(
            {
                "start_sample": clock.now,
                "segment_samples": len(segment),
                "cost_samples": cost,
            }
        )
        return clock.after(cost, fn(*args))

    loop = asyncio.get_running_loop()
    with (
        mock.patch.object(live_stt, "_decode", fake_decode),
        mock.patch.object(loop, "run_in_executor", fake_run_in_executor),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        producer_task = asyncio.create_task(produce())
        worker_task = asyncio.create_task(worker(object(), vad, window, audio_q, state, None))
        tasks = (producer_task, worker_task)
        try:
            await clock.run_until_done(tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    if state.stopping:
        raise RuntimeError("production worker stopped during paced replay")
    return {
        "audio_samples": len(samples),
        "audio_s": round(len(samples) / SAMPLE_RATE, 3),
        "chunk_samples": chunk_samples,
        "decode_rtf": decode_rtf,
        "queue_max": queue_max,
        "arrivals": len(timeline),
        "accepted": len(timeline) - state.dropped,
        "drops": state.dropped,
        "max_depth": max((row["depth"] for row in timeline), default=0),
        "end_sample": clock.now,
        "decodes": decodes,
        "timeline": timeline,
    }


def paced_replay(
    samples: np.ndarray,
    *,
    chunk_ms: float = DEFAULT_CHUNK_MS,
    decode_rtf: float = DEFAULT_DECODE_RTF,
    queue_max: int = AUDIO_QUEUE_MAX,
) -> dict:
    """Replay float32 16 kHz audio and return deterministic queue telemetry."""
    if not 10.0 <= chunk_ms <= 100.0:
        raise ValueError("chunk_ms must be between 10 and 100")
    if decode_rtf <= 0:
        raise ValueError("decode_rtf must be positive")
    if queue_max <= 0:
        raise ValueError("queue_max must be positive")
    chunk_samples = round(chunk_ms * SAMPLE_RATE / 1000)
    pcm = np.ascontiguousarray(samples, dtype=np.float32)
    return asyncio.run(_run_paced(pcm, chunk_samples, decode_rtf, queue_max))


def paced_wav(
    path: Path,
    *,
    chunk_ms: float = DEFAULT_CHUNK_MS,
    decode_rtf: float = DEFAULT_DECODE_RTF,
    queue_max: int = AUDIO_QUEUE_MAX,
) -> dict:
    return paced_replay(
        replay.load_wav_f32_16k(path),
        chunk_ms=chunk_ms,
        decode_rtf=decode_rtf,
        queue_max=queue_max,
    )


def evaluate(chunk_ms: float, decode_rtf: float, queue_max: int) -> dict:
    paths = [CACHE / f"{STRESSOR}.wav", *(CACHE / f"{cid}.wav" for cid in SHORT_CLIPS)]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if not VAD_MODEL.is_file():
        missing.append(str(VAD_MODEL.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError("missing paced-replay inputs: " + ", ".join(missing))

    kwargs = {"chunk_ms": chunk_ms, "decode_rtf": decode_rtf, "queue_max": queue_max}
    return {
        "scenario": {
            "chunk_ms": chunk_ms,
            "decode_rtf": decode_rtf,
            "queue_max": queue_max,
        },
        "long": paced_wav(CACHE / f"{STRESSOR}.wav", **kwargs),
        "short": {cid: paced_wav(CACHE / f"{cid}.wav", **kwargs) for cid in SHORT_CLIPS},
    }


def validate(report: dict) -> list[str]:
    failures = []
    if report["long"]["drops"] <= 0:
        failures.append(f"{STRESSOR}: expected drops > 0")
    dirty_short = {cid: row["drops"] for cid, row in report["short"].items() if row["drops"]}
    if dirty_short:
        failures.append(f"short corpus dropped blocks: {dirty_short}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--chunk-ms", type=float, default=DEFAULT_CHUNK_MS)
    parser.add_argument("--decode-rtf", type=float, default=DEFAULT_DECODE_RTF)
    parser.add_argument("--queue-max", type=int, default=AUDIO_QUEUE_MAX)
    parser.add_argument("--json", action="store_true", help="Include the full queue timeline.")
    args = parser.parse_args()

    try:
        report = evaluate(args.chunk_ms, args.decode_rtf, args.queue_max)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
    failures = validate(report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        long = report["long"]
        print(
            f"long/{STRESSOR}: audio={long['audio_s']:.3f}s "
            f"decodes={len(long['decodes'])} max_q={long['max_depth']}/{long['queue_max']} "
            f"drops={long['drops']}"
        )
        print(
            f"short/{len(report['short'])} clips: "
            f"max_q={max(row['max_depth'] for row in report['short'].values())} "
            f"drops={sum(row['drops'] for row in report['short'].values())}"
        )
        if not failures:
            print("PASS: long-form saturation reproduced; short corpus drop-free")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
