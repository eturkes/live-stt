"""Regression tests for the M9.3 paced-replay backpressure harness."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from live_stt import AUDIO_QUEUE_MAX, VAD_MODEL, State, enqueue_audio
from tests.eval_backpressure import CACHE, SHORT_CLIPS, STRESSOR, paced_wav


def test_enqueue_audio_uses_drop_newest_policy_and_counts_saturation():
    queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    state = State()
    first = np.array([1.0], dtype=np.float32)
    second = np.array([2.0], dtype=np.float32)
    rejected = np.array([3.0], dtype=np.float32)

    assert enqueue_audio(queue, state, first)
    assert enqueue_audio(queue, state, second)
    assert not enqueue_audio(queue, state, rejected)
    assert state.dropped == 1
    assert queue.qsize() == 2
    assert queue.get_nowait() is first
    assert queue.get_nowait() is second


def _resources_ready(paths: tuple[Path, ...]) -> bool:
    return VAD_MODEL.is_file() and all(path.is_file() for path in paths)


LONG_PATH = CACHE / f"{STRESSOR}.wav"
SHORT_PATHS = tuple(CACHE / f"{cid}.wav" for cid in SHORT_CLIPS)


@pytest.mark.skipif(
    not _resources_ready((LONG_PATH,)), reason="silero model or stressor WAV absent"
)
def test_long_paced_replay_reproduces_audio_queue_drops():
    report = paced_wav(LONG_PATH)
    assert report["drops"] > 0
    assert report["max_depth"] == AUDIO_QUEUE_MAX
    assert report["accepted"] + report["drops"] == report["arrivals"]


@pytest.mark.skipif(
    not _resources_ready(SHORT_PATHS), reason="silero model or short replay corpus absent"
)
def test_short_corpus_is_drop_free_and_pacing_is_deterministic():
    reports = {cid: paced_wav(path) for cid, path in zip(SHORT_CLIPS, SHORT_PATHS, strict=True)}
    assert all(report["drops"] == 0 for report in reports.values())
    # A repeat includes every virtual timestamp/depth transition, not wall-clock
    # decode timings, so whole-report equality is the determinism lock.
    first_id = SHORT_CLIPS[0]
    assert reports[first_id] == paced_wav(CACHE / f"{first_id}.wav")
