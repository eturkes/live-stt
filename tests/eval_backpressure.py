#!/usr/bin/env python3
"""Deterministic paced-replay backpressure gate (M9.3/M9.5).

Drives the production ``live_stt.worker`` with the production silero VAD, ring
buffer, seconds-bounded ``AudioQueue``, bounded segment queue, and
``enqueue_audio`` drop policy. Only decoding is replaced: the event loop's
executor and ``live_stt._decode`` are patched so each decode view costs
``duration * decode_rtf`` on a virtual sample clock. Audio blocks arrive every
``chunk_ms`` on that same clock. No wall-clock delay or recognizer model
participates, so queue depths and drops are deterministic.

The default case retains defect B's PipeWire-like 20 ms callback, slow-host RTF
0.20, 2 s audio headroom, and 44.7 s stressor. Before M9.5, sequential decode
stalled feeding and dropped 139 blocks. The two-stage worker must now drain VAD
concurrently, finish drop-free, and keep both queues bounded.

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
    AUDIO_HEADROOM_S,
    SAMPLE_RATE,
    SEGMENT_QUEUE_MAX,
    VAD_MODEL,
    AudioQueue,
    State,
    enqueue_audio,
    make_vad,
    submit_audio_sentinel,
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
    headroom_s: float,
) -> dict:
    clock = _VirtualClock()
    state = State()
    audio_q = AudioQueue(headroom_s=headroom_s)
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
                    "audio_blocks": audio_q.qsize(),
                    "audio_samples": audio_q.queued_samples,
                    "segment_depth": state.segment_queue_depth,
                    "dropped": state.dropped,
                    "accepted": accepted,
                }
            )
            await clock.sleep(len(chunk))
        await submit_audio_sentinel(audio_q)

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
        "headroom_s": headroom_s,
        "arrivals": len(timeline),
        "accepted": len(timeline) - state.dropped,
        "drops": state.dropped,
        "max_audio_blocks": max((row["audio_blocks"] for row in timeline), default=0),
        "max_audio_samples": audio_q.max_queued_samples,
        "max_audio_s": audio_q.max_queued_samples / SAMPLE_RATE,
        "segment_queue_max": SEGMENT_QUEUE_MAX,
        "max_segment_depth": state.max_segment_queue_depth,
        "end_sample": clock.now,
        "decodes": decodes,
        "timeline": timeline,
    }


def paced_replay(
    samples: np.ndarray,
    *,
    chunk_ms: float = DEFAULT_CHUNK_MS,
    decode_rtf: float = DEFAULT_DECODE_RTF,
    headroom_s: float = AUDIO_HEADROOM_S,
) -> dict:
    """Replay float32 16 kHz audio and return deterministic queue telemetry."""
    if not 10.0 <= chunk_ms <= 100.0:
        raise ValueError("chunk_ms must be between 10 and 100")
    if decode_rtf <= 0:
        raise ValueError("decode_rtf must be positive")
    if headroom_s <= 0:
        raise ValueError("headroom_s must be positive")
    chunk_samples = round(chunk_ms * SAMPLE_RATE / 1000)
    pcm = np.ascontiguousarray(samples, dtype=np.float32)
    return asyncio.run(_run_paced(pcm, chunk_samples, decode_rtf, headroom_s))


def paced_wav(
    path: Path,
    *,
    chunk_ms: float = DEFAULT_CHUNK_MS,
    decode_rtf: float = DEFAULT_DECODE_RTF,
    headroom_s: float = AUDIO_HEADROOM_S,
) -> dict:
    return paced_replay(
        replay.load_wav_f32_16k(path),
        chunk_ms=chunk_ms,
        decode_rtf=decode_rtf,
        headroom_s=headroom_s,
    )


def evaluate(chunk_ms: float, decode_rtf: float, headroom_s: float) -> dict:
    paths = [CACHE / f"{STRESSOR}.wav", *(CACHE / f"{cid}.wav" for cid in SHORT_CLIPS)]
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    if not VAD_MODEL.is_file():
        missing.append(str(VAD_MODEL.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError("missing paced-replay inputs: " + ", ".join(missing))

    kwargs = {"chunk_ms": chunk_ms, "decode_rtf": decode_rtf, "headroom_s": headroom_s}
    return {
        "scenario": {
            "chunk_ms": chunk_ms,
            "decode_rtf": decode_rtf,
            "headroom_s": headroom_s,
        },
        "long": paced_wav(CACHE / f"{STRESSOR}.wav", **kwargs),
        "short": {cid: paced_wav(CACHE / f"{cid}.wav", **kwargs) for cid in SHORT_CLIPS},
    }


def validate(report: dict) -> list[str]:
    failures = []
    long = report["long"]
    if long["drops"]:
        failures.append(f"{STRESSOR}: dropped {long['drops']} blocks")
    if long["max_audio_s"] > report["scenario"]["headroom_s"]:
        failures.append(f"{STRESSOR}: audio headroom exceeded")
    if not 0 < long["max_segment_depth"] <= long["segment_queue_max"]:
        failures.append(
            f"{STRESSOR}: segment depth {long['max_segment_depth']}/"
            f"{long['segment_queue_max']} is invalid"
        )
    dirty_short = {cid: row["drops"] for cid, row in report["short"].items() if row["drops"]}
    if dirty_short:
        failures.append(f"short corpus dropped blocks: {dirty_short}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--chunk-ms", type=float, default=DEFAULT_CHUNK_MS)
    parser.add_argument("--decode-rtf", type=float, default=DEFAULT_DECODE_RTF)
    parser.add_argument("--headroom-s", type=float, default=AUDIO_HEADROOM_S)
    parser.add_argument("--json", action="store_true", help="Include the full queue timeline.")
    args = parser.parse_args()

    try:
        report = evaluate(args.chunk_ms, args.decode_rtf, args.headroom_s)
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
            f"decodes={len(long['decodes'])} "
            f"audio_q={long['max_audio_s']:.3f}/{long['headroom_s']:.3f}s "
            f"segment_q={long['max_segment_depth']}/{long['segment_queue_max']} "
            f"drops={long['drops']}"
        )
        print(
            f"short/{len(report['short'])} clips: "
            f"max_audio_q={max(row['max_audio_s'] for row in report['short'].values()):.3f}s "
            f"max_segment_q={max(row['max_segment_depth'] for row in report['short'].values())} "
            f"drops={sum(row['drops'] for row in report['short'].values())}"
        )
        if not failures:
            print("PASS: two-stage worker is bounded and drop-free")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
