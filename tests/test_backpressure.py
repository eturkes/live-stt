"""Regression tests for the M9.3/M9.5 paced-replay backpressure gate."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from live_stt import (
    AUDIO_HEADROOM_S,
    SAMPLE_RATE,
    SEGMENT_QUEUE_MAX,
    VAD_MODEL,
    AudioQueue,
    State,
    enqueue_audio,
    submit_audio_sentinel,
    worker,
)
from tests.eval_backpressure import (
    CACHE,
    SCALE_LADDER,
    SHORT_CLIPS,
    STRESSOR,
    load_vac_trace,
    paced_wav,
    vac_paced_wav,
)


class _FakeVad:
    def __init__(self):
        self.segments = []
        self.accepted = 0

    def accept_waveform(self, samples):
        self.segments.append(SimpleNamespace(start=self.accepted, samples=samples.copy()))
        self.accepted += len(samples)

    def flush(self):
        pass

    def empty(self):
        return not self.segments

    @property
    def front(self):
        return self.segments[0]

    def pop(self):
        self.segments.pop(0)


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


def test_audio_queue_bounds_headroom_by_samples_not_callback_blocks():
    state = State()
    queue = AudioQueue(headroom_s=4 / SAMPLE_RATE)
    three_samples = np.ones(3, dtype=np.float32)

    assert enqueue_audio(queue, state, three_samples)
    assert not enqueue_audio(queue, state, np.ones(2, dtype=np.float32))
    assert queue.queued_samples == 3
    assert queue.qsize() == 1
    assert state.dropped == 1

    assert queue.get_nowait() is three_samples
    for _ in range(4):
        assert enqueue_audio(queue, state, np.ones(1, dtype=np.float32))
    assert not enqueue_audio(queue, state, np.ones(1, dtype=np.float32))
    assert queue.queued_samples == 4
    assert queue.qsize() == 4
    assert state.dropped == 2


def test_worker_feeds_vad_and_flushes_while_decode_is_blocked():
    async def scenario():
        window = 16
        vad = _FakeVad()
        state = State()
        audio_q = AudioQueue(headroom_s=2 * window / SAMPLE_RATE)
        decode_started = asyncio.Event()
        release_decode = asyncio.Event()
        rows = []

        def fake_run_in_executor(_executor, _fn, *_args):
            decode_started.set()

            async def finish_decode():
                await release_decode.wait()
                return "decoded"

            return asyncio.create_task(finish_decode())

        def on_segment(start, n, seg_len, decode_s, text):
            rows.append((start, n, seg_len, text))

        loop = asyncio.get_running_loop()
        with mock.patch.object(loop, "run_in_executor", fake_run_in_executor):
            assert enqueue_audio(audio_q, state, np.ones(window, dtype=np.float32))
            worker_task = asyncio.create_task(
                worker(object(), vad, window, audio_q, state, None, on_segment=on_segment)
            )
            await asyncio.wait_for(decode_started.wait(), 1.0)

            # This block and shutdown sentinel arrive during the first decode.
            # The feeder must consume + flush both before the decoder is released.
            assert enqueue_audio(audio_q, state, np.ones(window, dtype=np.float32))
            await submit_audio_sentinel(audio_q)
            for _ in range(5):
                await asyncio.sleep(0)
            assert vad.accepted == 2 * window
            assert audio_q.empty()
            assert not worker_task.done()

            release_decode.set()
            await asyncio.wait_for(worker_task, 1.0)

        assert rows == [
            (0, window, window, "decoded"),
            (window, window, 2 * window, "decoded"),
        ]
        assert 0 < state.max_segment_queue_depth <= SEGMENT_QUEUE_MAX
        assert not state.stopping

    asyncio.run(scenario())


def test_worker_failure_cancels_both_stages_and_requests_stop(caplog):
    async def scenario():
        window = 16
        state = State()
        state.stop_event = asyncio.Event()
        audio_q = AudioQueue(headroom_s=window / SAMPLE_RATE)
        loop = asyncio.get_running_loop()

        def fail_decode(_executor, _fn, *_args):
            failed = loop.create_future()
            failed.set_exception(RuntimeError("decode failed"))
            return failed

        assert enqueue_audio(audio_q, state, np.ones(window, dtype=np.float32))
        with mock.patch.object(loop, "run_in_executor", fail_decode):
            await asyncio.wait_for(worker(object(), _FakeVad(), window, audio_q, state, None), 1.0)

        assert state.stopping
        assert state.stop_event.is_set()
        assert state.segment_queue_depth == 0
        assert audio_q.empty()
        current = asyncio.current_task()
        assert all(task is current or task.done() for task in asyncio.all_tasks())

    asyncio.run(scenario())
    assert "worker died" in caplog.text


def _resources_ready(paths: tuple[Path, ...]) -> bool:
    return VAD_MODEL.is_file() and all(path.is_file() for path in paths)


LONG_PATH = CACHE / f"{STRESSOR}.wav"
SHORT_PATHS = tuple(CACHE / f"{cid}.wav" for cid in SHORT_CLIPS)


@pytest.mark.skipif(
    not _resources_ready((LONG_PATH,)), reason="silero model or stressor WAV absent"
)
def test_long_paced_replay_is_drop_free_with_bounded_two_stage_queues():
    report = paced_wav(LONG_PATH)
    assert report["drops"] == 0
    assert report["max_audio_s"] <= AUDIO_HEADROOM_S
    assert 0 < report["max_segment_depth"] <= SEGMENT_QUEUE_MAX
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


# --- VAC branch (M11.4) -------------------------------------------------------
# The sherpa arms above cannot speak for the shipped path: it has no segment
# queue and awaits each decode inside the coroutine draining audio_q. These
# replay tests/vac_decode_trace.json -- real NPU per-update costs plus the
# hypotheses that produced them -- so the shipped branch is gated without a model.

VAC_TRACE_CLIPS = tuple(load_vac_trace()["clips"])
VAC_PATHS = tuple(CACHE / f"{clip}.wav" for clip in VAC_TRACE_CLIPS)


def _vac_run(clip: str, **kwargs) -> dict:
    series = load_vac_trace()["clips"][clip]["series"]
    return vac_paced_wav(CACHE / f"{clip}.wav", series, **kwargs)


@pytest.mark.skipif(
    not _resources_ready(VAC_PATHS), reason="silero model or VAC trace corpus absent"
)
@pytest.mark.parametrize("clip", VAC_TRACE_CLIPS)
def test_vac_paced_replay_is_drop_free_at_measured_decode_cost(clip):
    trace = load_vac_trace()["clips"][clip]
    report = _vac_run(clip)
    # Zero divergences is what makes the measured costs apply: every decode saw
    # the buffer its cost was recorded against, so the replayed policy is the
    # measured one rather than a lookalike.
    assert report["updates"] == len(trace["series"])
    assert report["divergences"] == 0
    assert report["drops"] == 0
    assert report["forced_trims"] == 0
    assert report["max_segment_depth"] == 0  # VAC owns no second queue
    assert report["max_contiguous_s"] < AUDIO_HEADROOM_S
    assert report["max_audio_s"] <= AUDIO_HEADROOM_S
    assert report["accepted"] + report["drops"] == report["arrivals"]


@pytest.mark.skipif(
    not _resources_ready((CACHE / f"{STRESSOR}.wav",)),
    reason="silero model or stressor WAV absent",
)
def test_vac_trajectory_guard_catches_a_shifted_series():
    """The divergence counter, not the drop counter, is what certifies the costs.

    Shifting the series by one update leaves every later decode charged against a
    buffer it never saw. Drops stay at zero, so only this counter separates a
    measured run from a plausible-looking one.
    """
    series = load_vac_trace()["clips"][STRESSOR]["series"]
    shifted = vac_paced_wav(CACHE / f"{STRESSOR}.wav", series[1:])
    assert shifted["divergences"] == shifted["updates"] > 0
    assert shifted["drops"] == 0


@pytest.mark.skipif(
    not _resources_ready((CACHE / f"{STRESSOR}.wav",)),
    reason="silero model or stressor WAV absent",
)
def test_vac_paced_replay_drops_when_decode_is_scaled_past_the_headroom():
    """Non-vacuity: the same clip and policy must drop once decode outruns capture."""
    overloaded = _vac_run(STRESSOR, cost_scale=SCALE_LADDER[-1])
    assert overloaded["drops"] > 0
    assert _vac_run(STRESSOR) == _vac_run(STRESSOR)  # virtual clock, so bit-identical


# --- Long-form decode duty (P-014) --------------------------------------------
# The VAC arm above needs the gitignored WAV corpus; this one needs nothing but a
# committed trace, so the shipped path's real-time margin stays checkable in a
# fresh clone -- over 848 s of narration rather than the two pause-free clips the
# per-update trace covers.
#
# caption_trace.json's `decode_s` is `utterance_decode_s`: the SUM of one
# utterance's VAC update decodes, never a single blocking call. Reading a 7.42 s
# caption as a 7.42 s stall against AUDIO_HEADROOM_S is therefore wrong twice
# over. What the headroom is really spent against is CARRY: the loop decodes
# instead of draining audio_q, so a caption costing more wall time than its own
# audio hands the difference to its successor, and only that accumulation can
# outgrow a bounded queue. Silence between captions drains further still, so
# ignoring the gaps keeps the model conservative.

CAPTION_TRACE = Path(__file__).resolve().parent / "caption_trace.json"


def _worst_carry(captions) -> float:
    """Peak backlog seconds a caption stream hands forward, floored at empty."""
    carried = worst = 0.0
    for caption in captions:
        carried = max(0.0, carried + caption["decode_s"] - caption["dur_s"])
        worst = max(worst, carried)
    return worst


def _carry_at(captions, scale: float) -> float:
    return _worst_carry([{**c, "decode_s": c["decode_s"] * scale} for c in captions])


def test_long_form_decode_duty_holds_real_time_with_a_scale_ladder_reserve():
    """Margin, not a point reading -- decode cost varies ~20 % run to run (D-016).

    Measured carry is 0.017 s of the 2 s headroom over 215 captions, one of them
    over duty by that 0.017 s, so the queue empties inside every utterance and its
    peak is a single update decode -- which this trace does not record and M11.4's
    per-update trace measures at 0.764/1.006 s. The load-bearing number is the
    knee: carry reaches AUDIO_HEADROOM_S at x1.541 decode cost, so 848 s of
    narration independently reproduces M11.4's x1.5 SCALE_LADDER reserve on 4.4x
    the audio and against a corpus its two pause-free clips never covered.
    """
    captions = json.loads(CAPTION_TRACE.read_text(encoding="utf-8"))["captions"]
    for scale in (rung for rung in SCALE_LADDER if rung <= 1.25):
        assert _carry_at(captions, scale) < AUDIO_HEADROOM_S


def test_carry_outgrows_the_headroom_at_the_cost_that_raised_the_question():
    """Non-vacuity, sized on the real event rather than an invented one.

    M12.1's burst run is `git show f25cfb5:tests/caption_trace.json` -- same clip,
    same section, same device, decode sum 284.673 s against 259.216 s of audio for
    RTF 1.098 against this run's 0.551, and it carries 77.231 s. The nearest rung
    to that ratio is x2.0, so the ladder that certifies the reserve above is the
    same one that fails on the state P-014 was raised about.
    """
    captions = json.loads(CAPTION_TRACE.read_text(encoding="utf-8"))["captions"]
    assert _carry_at(captions, 2.0) > AUDIO_HEADROOM_S
