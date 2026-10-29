"""Scenario truth boundaries + tally accounting, never synthetic win-rate locks."""

import json
import random
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

from tests import eval_anchor_scenarios as scenarios


@pytest.mark.parametrize(
    ("output", "expected"), [("AB", "exact"), ("ABB", "dup"), ("A", "loss"), ("AX", "subst")]
)
def test_verdict_distinguishes_exact_duplication_loss_and_substitution(output, expected):
    assert scenarios.verdict(output, "AB") == expected


@pytest.mark.parametrize(
    ("kind", "changed", "head"),
    [
        ("persist", [2, 3, 4, 5], "abXX"),
        ("flip1", [2], "abXX"),
        ("flip2", [2, 3], "abXX"),
        ("headdrop1", [2], "cd"),
        ("headdrop2", [2, 3], "cd"),
        ("totaldrop1", [2], ""),
        ("lastflip", [5], "abXX"),
        ("lasthead", [5], "cd"),
        ("lasttotal", [5], ""),
    ],
)
def test_disturbances_change_only_the_already_published_prefix(monkeypatch, kind, changed, head):
    rng = mock.Mock(spec=random.Random)
    rng.randint.side_effect = [20, 4, 4, 4, 4, 4, 2, 2]
    rng.randrange.side_effect = [0, 2]
    rng.random.return_value = 0.5
    monkeypatch.setattr(scenarios, "respell", lambda *_: "XX")
    truth = "abcdefghijklmnopqrst"
    reaches = [4, 8, 12, 16, 20, 20]
    expected = [truth[:reach] for reach in reaches]
    # The first two decodes agree on abcd; everything after d is unspoken truth.
    for index in changed:
        expected[index] = head + truth[4 : reaches[index]]

    assert scenarios.script(kind, rng, truth * 4, list(truth)) == (truth, expected)


@pytest.mark.parametrize("edit", [False, True])
def test_mark_drop_keeps_the_unpublished_suffix_and_two_agreeing_mark_decodes(monkeypatch, edit):
    rng = mock.Mock(spec=random.Random)
    rng.randint.side_effect = [20, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4]
    rng.randrange.side_effect = [0, 2, 0] if edit else [0, 2]
    rng.random.return_value = 0.5
    rng.choice.side_effect = [4, "Z"] if edit else [4]
    monkeypatch.setattr(scenarios, "respell", lambda *_: "XX")
    truth = "abc、defghijklmnopqrs"
    kept = "Zbc" if edit else "abc"
    expected = [truth[:4]] * 2 + [kept + truth[4:reach] for reach in (8, 12, 16, 20)]

    assert scenarios.script(
        "markdrop_edit" if edit else "markdrop", rng, truth * 4, list(truth)
    ) == (truth, expected + [expected[-1]])


def test_tallies_count_drawn_scripts_once_and_separate_each_sides_exact_cases(
    monkeypatch, tmp_path
):
    captions = tmp_path / "captions.json"
    captions.write_text(json.dumps({"captions": [{"text": "AB"}]}), encoding="utf-8")
    monkeypatch.setattr(scenarios, "CAPTIONS", captions)
    monkeypatch.setattr(scenarios, "SCENARIOS", ("first", "second"))
    current, baseline = ModuleType("current"), ModuleType("baseline")
    monkeypatch.setattr(
        scenarios, "load", lambda path: baseline if path == Path("old") else current
    )
    labels = iter([None, "exact", "dup", "loss", "subst", "current", "baseline", None])

    def draw(*_):
        label = next(labels)
        return None if label is None else ("AB", [label])

    outputs = {
        current: dict(exact="AB", dup="ABB", loss="A", subst="AX", current="AB", baseline="A"),
        baseline: dict(exact="AB", dup="AB", loss="A", subst="AX", current="AX", baseline="AB"),
    }
    monkeypatch.setattr(scenarios, "script", draw)
    monkeypatch.setattr(scenarios, "run", lambda module, texts: outputs[module][texts[0]])

    assert scenarios.evaluate(4, 11, Path("old")) == {
        "first": dict(
            scripts=3,
            exact=1,
            dup=1,
            loss=1,
            baseline_exact=2,
            baseline_loss=1,
            baseline_only_exact=1,
            current_only_exact=0,
        ),
        "second": dict(
            scripts=3,
            subst=1,
            exact=1,
            loss=1,
            baseline_subst=2,
            baseline_exact=1,
            baseline_only_exact=1,
            current_only_exact=1,
        ),
    }
