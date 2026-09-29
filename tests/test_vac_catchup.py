"""VAC catch-up: queue-row contract, hermetic production-worker probes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest import mock

import numpy as np
import pytest

import live_stt
from tests.eval_backpressure import _VirtualClock


class _Vad:
    def __init__(self, close_at: int | None = None) -> None:
        self.close_at = close_at
        self.windows = 0
        self.accepted = 0
        self.queued = 0

    def accept_waveform(self, samples: np.ndarray) -> None:
        self.windows += 1
        self.accepted += len(samples)
        if self.windows == self.close_at:
            self.queued += 1

    def is_speech_detected(self) -> bool:
        return self.windows > 0 and (self.close_at is None or self.windows < self.close_at)

    def empty(self) -> bool:
        return not self.queued

    def pop(self) -> None:
        assert self.queued
        self.queued -= 1

    def flush(self) -> None:
        pass


@dataclass
class _Decode:
    samples: np.ndarray
    queued_samples: int | None
    at_sample: int
    final: bool = False


@dataclass
class _Run:
    calls: list[_Decode]
    state: live_stt.State
    consumed: int
    remaining: int
    timeline: list[tuple[int, int, bool]]


def _run(
    blocks: list[np.ndarray | None],
    window: int,
    *,
    plain: bool = False,
    close_at: int | None = None,
    decode_s: float = 0.0,
    paced: bool = False,
) -> _Run:
    async def scenario() -> _Run:
        clock = _VirtualClock()
        queue = asyncio.Queue() if plain else live_stt.AudioQueue()
        state = live_stt.State()
        vad = _Vad(close_at)
        calls: list[_Decode] = []
        timeline: list[tuple[int, int, bool]] = []

        class Recognizer:
            def decode_segments(self, samples: np.ndarray, language: str | None = None):
                calls.append(
                    _Decode(samples.copy(), getattr(queue, "queued_samples", None), clock.now)
                )
                # No timestamped segments => no trim obscures PCM accounting.
                return "あ", []

        def execute(_executor, fn, *args):
            return clock.after(round(decode_s * live_stt.SAMPLE_RATE), fn(*args))

        def updated(_buffer_s, _end_s, _commit_s, _text, final, _decode_s):
            calls[-1].final = final

        async def produce() -> None:
            assert isinstance(queue, live_stt.AudioQueue)
            for block in blocks:
                if block is None:
                    await live_stt.submit_audio_sentinel(queue)
                else:
                    accepted = live_stt.enqueue_audio(queue, state, block)
                    timeline.append((clock.now, queue.queued_samples, accepted))
                    await clock.sleep(len(block))

        if not paced:
            for block in blocks:
                queue.put_nowait(block)
        loop = asyncio.get_running_loop()
        with mock.patch.object(loop, "run_in_executor", execute):
            worker = asyncio.create_task(
                live_stt.worker(Recognizer(), vad, window, queue, state, None, on_update=updated)
            )
            tasks = (worker, asyncio.create_task(produce())) if paced else (worker,)
            try:
                await clock.run_until_done(tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        assert not state.stopping, "worker raised instead of exercising the scheduling contract"
        return _Run(calls, state, vad.accepted, queue.qsize(), timeline)

    return asyncio.run(scenario())


def _blocks(samples: np.ndarray, window: int) -> list[np.ndarray | None]:
    return [samples[start : start + window] for start in range(0, len(samples), window)] + [None]


def test_audio_headroom_is_eight_seconds() -> None:
    assert live_stt.AUDIO_HEADROOM_S == 8.0


def test_vac_backlog_threshold_is_half_a_second() -> None:
    assert live_stt.VAC_BACKLOG_S == 0.5  # type: ignore[attr-defined]


def test_default_audio_queue_accepts_eight_seconds_and_rejects_one_sample_more() -> None:
    queue = live_stt.AudioQueue()
    state = live_stt.State()
    samples = np.zeros(8 * live_stt.SAMPLE_RATE, dtype=np.float32)

    assert live_stt.enqueue_audio(queue, state, samples)
    assert queue.queued_samples == len(samples)
    assert not live_stt.enqueue_audio(queue, state, np.zeros(1, dtype=np.float32))
    assert state.dropped == 1
    assert queue.get_nowait() is samples
    assert queue.queued_samples == 0


@pytest.mark.parametrize("excess_samples", [0, 1, 1600, 1601, 7999])
def test_due_update_fires_on_first_block_at_or_below_backlog_bound(excess_samples: int) -> None:
    rate = live_stt.SAMPLE_RATE
    window = rate // 10
    threshold = rate // 2
    due = round(live_stt.VAC_CHUNK_S * rate)
    samples = np.arange(due + threshold + excess_samples, dtype=np.float32)
    run = _run(_blocks(samples, window), window)
    drained = ((excess_samples + window - 1) // window) * window
    first = run.calls[0]

    assert not first.final
    assert first.queued_samples == threshold + excess_samples - drained
    # Sample identity catches a delayed decode that only retained its last chunk.
    np.testing.assert_array_equal(first.samples, samples[: due + drained])
    assert run.state.dropped == 0
    assert run.remaining == 0


def test_backlog_rule_is_invariant_to_capture_block_size() -> None:
    rate = live_stt.SAMPLE_RATE
    due = round(live_stt.VAC_CHUNK_S * rate)
    threshold = rate // 2
    # Exhaust both sides of each block boundary across distinct callback sizes.
    for window in (80, 160, 400, 800, 1600):
        for excess in (1, window - 1, window, window + 1, threshold - 1):
            samples = np.arange(due + threshold + excess, dtype=np.float32)
            run = _run(_blocks(samples, window), window)
            drained = ((excess + window - 1) // window) * window
            first = run.calls[0]
            assert first.queued_samples == threshold + excess - drained, (window, excess)
            np.testing.assert_array_equal(first.samples, samples[: due + drained])
            assert not first.final


@pytest.mark.parametrize("decode_s", [1.1, 1.4, 2.25])
def test_sustained_overload_catches_up_without_drops(decode_s: float) -> None:
    rate = live_stt.SAMPLE_RATE
    window = rate // 100
    samples = np.arange(36 * rate, dtype=np.float32)
    run = _run(_blocks(samples, window), window, decode_s=decode_s, paced=True)
    peak = max(queued for _, queued, _ in run.timeline)
    limit = round((0.5 + decode_s) * rate) + window

    assert run.state.dropped == 0, f"dropped={run.state.dropped}, peak_s={peak / rate}"
    assert peak <= limit, f"peak_s={peak / rate}, bound_s={limit / rate}"
    assert all(accepted for _, _, accepted in run.timeline)
    assert run.consumed == len(samples)
    assert run.remaining == 0
    updates = [call for call in run.calls if not call.final]
    assert len(updates) >= 10
    assert all(
        call.queued_samples is not None and call.queued_samples <= rate // 2 for call in updates
    )
    assert len(run.calls) < 36, "overload must stretch the update cadence"
    assert run.calls[-1].final


def test_plain_replay_queue_keeps_the_chunk_grid_despite_prefilled_audio() -> None:
    rate = live_stt.SAMPLE_RATE
    window = rate // 10
    due = round(live_stt.VAC_CHUNK_S * rate)
    samples = np.arange(4 * due + window, dtype=np.float32)
    run = _run(_blocks(samples, window), window, plain=True)
    updates = [call for call in run.calls if not call.final]

    assert [len(call.samples) for call in updates] == list(range(due, 4 * due + 1, due))
    for call in updates:
        assert call.queued_samples is None
        np.testing.assert_array_equal(call.samples, samples[: len(call.samples)])
    assert run.calls[-1].final
    np.testing.assert_array_equal(run.calls[-1].samples, samples)


@pytest.mark.parametrize("finish", ["vad-close", "sentinel"])
def test_final_decode_does_not_wait_for_capture_backlog(finish: str) -> None:
    rate = live_stt.SAMPLE_RATE
    window = rate // 10
    prefix = np.arange(4 * window, dtype=np.float32)
    backlog = np.full(rate, -1, dtype=np.float32)
    blocks = _blocks(prefix, window)[:-1]
    if finish == "vad-close":
        blocks += [np.zeros(window, dtype=np.float32), backlog, None]
        run = _run(blocks, window, close_at=5)
    else:
        # A callback can already be scheduled when shutdown enqueues its sentinel.
        blocks += [None, backlog]
        run = _run(blocks, window)

    (final,) = run.calls
    assert final.final
    assert final.queued_samples == rate
    np.testing.assert_array_equal(final.samples[: len(prefix)], prefix)
    assert final.at_sample == 0
    assert run.remaining == (1 if finish == "sentinel" else 0)
