"""Two-way trace replay follows the utterance's first-acceptance latch."""

import json

import pytest

from tests import eval_two_way_settled as settled


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
