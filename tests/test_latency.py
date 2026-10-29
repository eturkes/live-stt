"""Synthetic clock/placement oracles; independent of recorded latency values."""

import pytest

from tests import eval_latency as latency


def test_empty_updates_keep_the_endpoint_and_final_text_reaches_the_buffer_end():
    clip = {
        "series": [
            dict(
                buffer_end_s=1, buffer_s=1, decode_s=0.2, commit="", commit_audio_s=0.9, final=False
            ),
            dict(
                buffer_end_s=2,
                buffer_s=2,
                decode_s=0.3,
                commit="AB",
                commit_audio_s=1.5,
                final=False,
            ),
            dict(
                buffer_end_s=3, buffer_s=3, decode_s=0.4, commit="CD", commit_audio_s=2, final=True
            ),
        ]
    }
    lags, counts = latency.clip_lags(clip)

    # Emissions 2.3/3.4; character midpoints .375/1.125/1.875/2.625.
    assert lags == pytest.approx([1.925, 1.175, 1.525, 0.775])
    assert counts == dict(updates=3, commits=2, unplaced=0, non_monotone=0, finals=1, redraws=0)


def test_clock_carries_decode_cost_clamps_backward_ends_and_resets_at_speech_end():
    clip = {
        "series": [
            dict(
                buffer_end_s=2, buffer_s=1, decode_s=2, commit="AB", commit_audio_s=1.8, final=False
            ),
            dict(
                buffer_end_s=3,
                buffer_s=2,
                decode_s=0.5,
                commit="C",
                commit_audio_s=1.5,
                final=False,
            ),
            dict(
                buffer_end_s=3.5,
                buffer_s=2.5,
                decode_s=0.5,
                commit="X",
                commit_audio_s=None,
                final=False,
            ),
            dict(
                buffer_end_s=4, buffer_s=3, decode_s=0.25, commit="D", commit_audio_s=1, final=True
            ),
            dict(
                buffer_end_s=8,
                buffer_s=1,
                decode_s=0.5,
                commit="E",
                commit_audio_s=None,
                final=True,
            ),
        ]
    }
    lags, counts = latency.clip_lags(clip)

    assert lags == pytest.approx([2.8, 2.4, 2.7, 2.35, 1])
    assert counts == dict(updates=5, commits=5, unplaced=1, non_monotone=1, finals=2, redraws=0)


def test_provisional_fresh_tail_and_redraws_use_the_hypothesis_span():
    clip = {
        "series": [
            dict(
                buffer_end_s=2, buffer_s=2, decode_s=0.25, segments=[(0, 1.5, "ABC")], final=False
            ),
            dict(
                buffer_end_s=3, buffer_s=3, decode_s=0.25, segments=[(0, 2.5, "AXCDE")], final=False
            ),
            dict(
                buffer_end_s=4, buffer_s=4, decode_s=0.25, segments=[(0, 3, "AXCDEF")], final=True
            ),
        ]
    }
    lags, counts = latency.clip_lags(clip, provisional=True)

    assert lags == pytest.approx([2, 1.5, 1, 1.5, 1, 1.375, 0.625])
    assert counts == dict(updates=3, commits=3, unplaced=0, non_monotone=0, finals=1, redraws=1)
    assert latency.redraw_bound_lags(clip) == pytest.approx([2, 1.5, 1, 2.5, 2, 1.5, 1, 7 / 12])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], dict(n=0, p50=None, mean=None, p90=None, max=None)),
        ([1.23456], dict(n=1, p50=1.235, mean=1.235, p90=1.235, max=1.235)),
        ([5, 1, 9, 3], dict(n=4, p50=4, mean=4.5, p90=9, max=9)),
        ([30, *range(18, -1, -1)], dict(n=20, p50=9.5, mean=10.05, p90=18, max=30)),
    ],
)
def test_quantiles_sort_use_the_even_median_and_select_the_upper_rank(values, expected):
    assert latency._quantiles(values) == expected
