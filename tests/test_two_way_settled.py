"""Two-way trace replay follows the utterance's first-acceptance latch."""

import json

import pytest

from tests import eval_latency as latency
from tests import eval_two_way_settled as settled


def _early_commit_clip():
    return {
        "series": [
            dict(
                buffer_end_s=1,
                buffer_s=1,
                decode_s=0.2,
                commit="A",
                commit_audio_s=0.75,
                final=False,
            ),
            dict(
                buffer_end_s=2,
                buffer_s=2,
                decode_s=0.3,
                commit="B",
                commit_audio_s=1.5,
                final=False,
            ),
            dict(
                buffer_end_s=3, buffer_s=3, decode_s=0.4, commit="CD", commit_audio_s=2, final=True
            ),
        ]
    }


def test_acceptance_redates_only_the_earlier_commit():
    clip = _early_commit_clip()
    baseline, _ = latency.clip_lags(clip)
    changed, redated = settled.clip_lags(clip, True, gate_s=2)

    assert baseline == pytest.approx([0.825, 1.175, 1.525, 0.775])
    assert changed == pytest.approx([1.925, 1.175, 1.525, 0.775])
    assert redated == 1


def test_no_withholding_reproduces_commit_lags_character_for_character():
    clip = _early_commit_clip()
    expected, _ = latency.clip_lags(clip)
    actual, redated = settled.clip_lags(clip, False)

    assert actual == expected
    assert redated == 0


def test_never_accepted_charges_earlier_commits_to_the_final_decode():
    actual, redated = settled.clip_lags(_early_commit_clip(), True, gate_s=1e9)

    assert actual == pytest.approx([3.025, 2.275, 1.525, 0.775])
    assert redated == 2


@pytest.mark.parametrize("gate", [2.0, 3.0])
@pytest.mark.parametrize("name", ["stress_long", "retention_probe"])
def test_redated_counts_only_commits_before_first_acceptance(name: str, gate: float):
    trace = json.loads(settled.TRACE.read_text(encoding="utf-8"))
    clip = trace["clips"][name]
    expected = 0
    for rows in settled.utterances(clip["series"]):
        beginning = rows[0]["buffer_end_s"] - rows[0]["buffer_s"]
        for row in rows:
            if row["buffer_end_s"] - beginning >= gate or row["final"]:
                break
            expected += len(row["commit"])
    _, redated = settled.clip_lags(clip, True, gate)
    assert redated == expected, (name, gate, redated, expected)
