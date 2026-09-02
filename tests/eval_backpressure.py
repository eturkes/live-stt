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

The VAC arm (M11.4) covers the SHIPPED branch, which the sherpa arms cannot: it
has no segment queue and awaits each decode inside the coroutine draining
``audio_q``, so capture is buffered for the length of a decode rather than fed
through one. A flat RTF cannot pace it -- cost tracks the growing buffer, not the
audio -- so it replays ``tests/vac_decode_trace.json``: the real per-update NPU
decode costs and the hypotheses that produced them, which reproduce the measured
commit/trim trajectory with no model. Each traced clip runs twice, at measured
cost and at ``OVERLOAD_SCALE``, because a drop-free result means nothing until
the same scenario is shown to drop.

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
from streaming import Segment, StreamingProcessor  # noqa: E402

CACHE = ROOT / "spike" / "backends" / "cache"
REAL_CLIPS = ROOT / "tests" / "real_clips.json"
SHORT_CLIPS = tuple(json.loads(REAL_CLIPS.read_text(encoding="utf-8")))
STRESSOR = "stress_long"
VAC_TRACE = ROOT / "tests" / "vac_decode_trace.json"

DEFAULT_CHUNK_MS = 20.0
DEFAULT_DECODE_RTF = 0.20
# Decode-cost multipliers each traced clip is paced at. The first rung must be
# 1.0 (the measured costs) and the last must overload: a drop-free result proves
# nothing until the same scenario is shown to drop, and the rung where dropping
# starts is the margin -- how much slower every decode could get before the
# capture headroom stops absorbing it. That margin is the honest answer to a
# recorded maximum being one sample of a run-to-run variable quantity.
SCALE_LADDER = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0)
# A divergence shifts the buffer by at least one VAD window (32 ms), so this
# absorbs the trace's 1e-6 rounding and nothing else.
BUFFER_MATCH_TOL_S = 1e-3
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


async def _produce(
    clock: _VirtualClock,
    audio_q: AudioQueue,
    state: State,
    samples: np.ndarray,
    chunk_samples: int,
    timeline: list[dict[str, int | bool]],
) -> None:
    """Arrive one capture block per block-duration of virtual time, then sentinel."""
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
        await _produce(clock, audio_q, state, samples, chunk_samples, timeline)

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


class _WatchedAudioQueue(AudioQueue):
    """AudioQueue that reports every successful drain.

    VAC awaits each decode inside the coroutine draining this queue, so what the
    2 s capture headroom is spent against is the span between two drains, not one
    decode. ``asyncio.Queue.get`` routes every successful get through
    ``get_nowait``, which ``AudioQueue`` already relies on for its own accounting.
    """

    def __init__(self, headroom_s: float, on_drain) -> None:
        super().__init__(headroom_s=headroom_s)
        self._on_drain = on_drain

    def get_nowait(self):
        item = super().get_nowait()
        self._on_drain()
        return item


class _TraceRecognizer:
    """Replays recorded whisper hypotheses in call order, with no model.

    ``worker`` dispatches on ``decode_segments``, so this drives the real VAC
    branch. ``StreamingProcessor`` is a pure function of the hypotheses and the
    buffer lengths, so replaying the recorded ones reproduces the measured
    commit/trim trajectory exactly -- which is what makes a measured decode cost
    apply to the buffer it was measured on. Past the recorded series it returns
    nothing, which is the honest answer once a drop has changed the audio.
    """

    def __init__(self, series: list[dict]) -> None:
        self.series = series
        self.calls = 0

    def decode_segments(self, _samples: np.ndarray) -> tuple[str, list[Segment]]:
        row = self.series[self.calls] if self.calls < len(self.series) else None
        self.calls += 1
        if row is None:
            return "", []
        segments = [Segment(start, end, text) for start, end, text in row["segments"]]
        text = "".join(s.text for s in segments).strip() if segments else row["text"]
        return text, segments


async def _run_vac_paced(
    samples: np.ndarray,
    series: list[dict],
    chunk_samples: int,
    headroom_s: float,
    cost_scale: float,
) -> dict:
    clock = _VirtualClock()
    state = State()
    timeline: list[dict[str, int | bool]] = []
    decodes: list[dict] = []
    contiguous = {"run": 0, "max": 0}
    audio_q = _WatchedAudioQueue(headroom_s, lambda: contiguous.__setitem__("run", 0))
    vad, window = make_vad()
    rec = _TraceRecognizer(series)
    processors: list[StreamingProcessor] = []
    buffers = np.array([row["buffer_s"] for row in series], dtype=np.float64)
    costs = np.array([row["decode_s"] for row in series], dtype=np.float64)
    order = np.argsort(buffers)

    def tracked_processor(**kwargs) -> StreamingProcessor:
        processor = StreamingProcessor(**kwargs)
        processors.append(processor)
        return processor

    def fake_run_in_executor(_executor, fn, *args):
        # VAC's only executor call is StreamingProcessor.process, so the bound
        # instance carries the buffer this decode is about to see.
        buffer_s = len(fn.__self__.audio) / SAMPLE_RATE
        ordinal = len(decodes)
        row = series[ordinal] if ordinal < len(series) else None
        matched = row is not None and abs(row["buffer_s"] - buffer_s) <= BUFFER_MATCH_TOL_S
        # A mismatch means the audio itself changed, which only a drop can do.
        # Interpolating the measured curve keeps such a run finishing with an
        # honest cost; `divergences` is what says the trajectory left the trace.
        decode_s = (
            row["decode_s"]
            if matched and row is not None
            else float(np.interp(buffer_s, buffers[order], costs[order]))
        )
        cost = max(1, round(decode_s * cost_scale * SAMPLE_RATE))
        contiguous["run"] += cost
        contiguous["max"] = max(contiguous["max"], contiguous["run"])
        decodes.append(
            {
                "start_sample": clock.now,
                "buffer_s": round(buffer_s, 6),
                "cost_samples": cost,
                "matched": matched,
            }
        )
        return clock.after(cost, fn(*args))

    loop = asyncio.get_running_loop()
    with (
        mock.patch.object(live_stt, "StreamingProcessor", tracked_processor),
        mock.patch.object(loop, "run_in_executor", fake_run_in_executor),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        producer_task = asyncio.create_task(
            _produce(clock, audio_q, state, samples, chunk_samples, timeline)
        )
        worker_task = asyncio.create_task(worker(rec, vad, window, audio_q, state, None))
        tasks = (producer_task, worker_task)
        try:
            await clock.run_until_done(tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    if state.stopping:
        raise RuntimeError("production worker stopped during paced VAC replay")
    audio_s = len(samples) / SAMPLE_RATE
    decode_s = sum(row["cost_samples"] for row in decodes) / SAMPLE_RATE
    return {
        "audio_samples": len(samples),
        "audio_s": round(audio_s, 3),
        "chunk_samples": chunk_samples,
        "cost_scale": cost_scale,
        "headroom_s": headroom_s,
        "arrivals": len(timeline),
        "accepted": len(timeline) - state.dropped,
        "drops": state.dropped,
        "updates": len(decodes),
        "trace_updates": len(series),
        "divergences": sum(1 for row in decodes if not row["matched"]),
        "max_audio_blocks": max((row["audio_blocks"] for row in timeline), default=0),
        "max_audio_samples": audio_q.max_queued_samples,
        "max_audio_s": audio_q.max_queued_samples / SAMPLE_RATE,
        "max_segment_depth": state.max_segment_queue_depth,
        "max_decode_s": max((row["cost_samples"] for row in decodes), default=0) / SAMPLE_RATE,
        "max_contiguous_s": contiguous["max"] / SAMPLE_RATE,
        "max_buffer_s": max((row["buffer_s"] for row in decodes), default=0.0),
        "total_decode_s": round(decode_s, 3),
        "duty": round(decode_s / audio_s, 4) if audio_s else 0.0,
        "trims": sum(processor.trims for processor in processors),
        # Nonzero means the trim rule failed and dropped audio whose text was
        # never emitted: content loss no queue counter can see.
        "forced_trims": sum(processor.forced_trims for processor in processors),
        "end_sample": clock.now,
        "decodes": decodes,
        "timeline": timeline,
    }


def vac_paced_replay(
    samples: np.ndarray,
    series: list[dict],
    *,
    chunk_ms: float = DEFAULT_CHUNK_MS,
    headroom_s: float = AUDIO_HEADROOM_S,
    cost_scale: float = 1.0,
) -> dict:
    """Pace the VAC branch on one clip's recorded per-update decode costs."""
    if not 10.0 <= chunk_ms <= 100.0:
        raise ValueError("chunk_ms must be between 10 and 100")
    if headroom_s <= 0:
        raise ValueError("headroom_s must be positive")
    if cost_scale <= 0:
        raise ValueError("cost_scale must be positive")
    chunk_samples = round(chunk_ms * SAMPLE_RATE / 1000)
    pcm = np.ascontiguousarray(samples, dtype=np.float32)
    return asyncio.run(_run_vac_paced(pcm, series, chunk_samples, headroom_s, cost_scale))


def vac_paced_wav(path: Path, series: list[dict], **kwargs) -> dict:
    return vac_paced_replay(replay.load_wav_f32_16k(path), series, **kwargs)


def load_vac_trace() -> dict:
    return json.loads(VAC_TRACE.read_text(encoding="utf-8"))


def evaluate_vac(trace: dict, chunk_ms: float, headroom_s: float) -> dict:
    """Pace every traced clip along SCALE_LADDER; rung 0 is the measured run."""
    clips = {}
    for clip, row in trace["clips"].items():
        samples = replay.load_wav_f32_16k(CACHE / f"{clip}.wav")
        runs = [
            vac_paced_replay(
                samples,
                row["series"],
                chunk_ms=chunk_ms,
                headroom_s=headroom_s,
                cost_scale=scale,
            )
            for scale in SCALE_LADDER
        ]
        dropping = [run["cost_scale"] for run in runs if run["drops"]]
        clips[clip] = {
            "recorded": {key: value for key, value in row.items() if key != "series"},
            "measured": runs[0],
            "drop_scale": dropping[0] if dropping else None,
            "ladder": [
                {key: run[key] for key in ("cost_scale", "drops", "max_audio_s", "divergences")}
                for run in runs
            ],
        }
    return {
        "provenance": {key: value for key, value in trace.items() if key != "clips"},
        "scale_ladder": list(SCALE_LADDER),
        "clips": clips,
    }


def validate_vac(report: dict) -> list[str]:
    failures = []
    for clip, row in report["clips"].items():
        measured = row["measured"]
        headroom_s = measured["headroom_s"]
        if measured["cost_scale"] != 1.0:
            failures.append(f"vac/{clip}: ladder rung 0 is not the measured cost")
        if measured["drops"]:
            failures.append(f"vac/{clip}: dropped {measured['drops']} blocks")
        if measured["divergences"] or measured["updates"] != measured["trace_updates"]:
            failures.append(
                f"vac/{clip}: {measured['updates']}/{measured['trace_updates']} updates, "
                f"{measured['divergences']} off the recorded buffer trajectory"
            )
        if measured["max_audio_s"] > headroom_s:
            failures.append(f"vac/{clip}: audio headroom exceeded")
        if measured["max_contiguous_s"] >= headroom_s:
            failures.append(
                f"vac/{clip}: contiguous decode {measured['max_contiguous_s']:.3f}s reaches "
                f"the {headroom_s:.3f}s capture headroom"
            )
        if measured["max_segment_depth"]:
            failures.append(
                f"vac/{clip}: streaming branch queued {measured['max_segment_depth']} segments"
            )
        if row["drop_scale"] is None:
            failures.append(
                f"vac/{clip}: no ladder rung up to x{report['scale_ladder'][-1]} dropped, "
                "so the drop-free result is vacuous"
            )
    return failures


def evaluate(chunk_ms: float, decode_rtf: float, headroom_s: float) -> dict:
    trace = load_vac_trace()
    paths = [
        CACHE / f"{STRESSOR}.wav",
        *(CACHE / f"{cid}.wav" for cid in SHORT_CLIPS),
        *(CACHE / f"{clip}.wav" for clip in trace["clips"]),
    ]
    missing = sorted({str(path.relative_to(ROOT)) for path in paths if not path.is_file()})
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
        "vac": evaluate_vac(trace, chunk_ms, headroom_s),
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
    return failures + validate_vac(report["vac"])


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
        for clip, row in report["vac"]["clips"].items():
            measured = row["measured"]
            drop_scale = row["drop_scale"]
            print(
                f"vac/{clip}: audio={measured['audio_s']:.3f}s "
                f"updates={measured['updates']}/{measured['trace_updates']} "
                f"duty={measured['duty']:.3f} "
                f"decode_max={measured['max_decode_s']:.3f}s "
                f"contiguous_max={measured['max_contiguous_s']:.3f}s "
                f"audio_q={measured['max_audio_s']:.3f}/{measured['headroom_s']:.3f}s "
                f"segment_q={measured['max_segment_depth']} "
                f"buffer_max={measured['max_buffer_s']:.3f}s "
                f"trims={measured['trims']} forced_trims={measured['forced_trims']} "
                f"drops={measured['drops']} "
                f"| drops first at x{drop_scale}"
            )
        forced = {
            clip: row["measured"]["forced_trims"]
            for clip, row in report["vac"]["clips"].items()
            if row["measured"]["forced_trims"]
        }
        if forced:
            print(f"WARN: forced trims discarded un-emitted audio: {forced}")
        if not failures:
            print("PASS: two-stage worker is bounded and drop-free")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
