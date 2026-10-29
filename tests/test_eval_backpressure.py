"""Paced evaluator completion is consumed PCM + EOF, never a success-shaped report."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from tests import eval_backpressure as backpressure
from tests.evaluator_fakes import hermetic_evaluator  # noqa: F401

pytestmark = pytest.mark.usefixtures("hermetic_evaluator")


@pytest.mark.parametrize("vac", [False, True], ids=["sherpa", "vac"])
@pytest.mark.parametrize("stop_after", [1, 4])
def test_paced_replay_requires_consumed_audio_and_eof(monkeypatch, vac, stop_after):
    consumed = []

    def run(samples, headroom=2.0):
        return asyncio.run(
            backpressure._run_vac_paced(samples, [], 320, headroom, 1.0)
            if vac
            else backpressure._run_paced(samples, 320, 0.2, headroom)
        )

    async def partial_worker(_rec, _vad, _window, queue, state, _output):
        state.max_segment_queue_depth = 1
        for _ in range(stop_after):
            consumed.append(await queue.get())

    monkeypatch.setattr(backpressure, "make_vad", lambda: (object(), 512))
    monkeypatch.setattr(backpressure, "worker", partial_worker)
    with pytest.raises(RuntimeError, match="incomplete paced replay"):
        run(np.zeros(4 * 320, dtype=np.float32))
    assert len(consumed) == stop_after

    # Both controls complete without text; rejection alone is not an incomplete run.
    duplicate, discard = False, False

    async def complete_worker(_rec, _vad, _window, queue, _state, _output):
        if discard:
            queue.get_nowait()  # A shutdown eviction is not worker consumption.
        while (chunk := await queue.get()) is not None:
            consumed.append(len(chunk))
        if duplicate:
            queue.put_nowait(None)
            await queue.get()

    monkeypatch.setattr(backpressure, "worker", complete_worker)
    for drop_all in (False, True):
        consumed.clear()
        samples = np.zeros(4 * 320 + (0 if drop_all else 17), dtype=np.float32)
        headroom = 100 / backpressure.SAMPLE_RATE if drop_all else 2.0
        row = run(samples, headroom)
        assert row["audio_samples"] == len(samples)
        assert row["drops"] == (4 if drop_all else 0)
        assert sum(consumed) == (0 if drop_all else len(samples))

    duplicate = True
    with pytest.raises(RuntimeError, match="incomplete paced replay"):
        run(np.zeros(4 * 320, dtype=np.float32))
    duplicate, discard = False, True
    with pytest.raises(RuntimeError, match="incomplete paced replay"):
        run(np.zeros(4 * 320, dtype=np.float32))
