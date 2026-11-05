"""The translator screen charges each turn to the arm whose model + effort produced it."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from tests import eval_translator_ab as ab
from tests.evaluator_fakes import hermetic_evaluator  # noqa: F401

pytestmark = pytest.mark.usefixtures("hermetic_evaluator")
REPS = 300  # bootstrap reps for contract cases; the screen runs ab.BOOTSTRAP


def _fake_translators(monkeypatch, *, fail_start: str | None = None, fail_text: str = "FAIL"):
    """A CodexTranslator stand-in that, like the real one, reads the constants per request."""
    made: list = []

    class Translator:
        def __init__(self, sources):
            self._notes = asyncio.Queue()
            self._legs = {s: SimpleNamespace(turns=0) for s in sources}
            self.enabled = False
            self.closed = False
            self.sent: list = []
            made.append(self)

        async def _request(self, method, params: dict):
            self.sent.append((method, params))
            if method == "thread/start":
                return {"serviceTier": params["serviceTier"]}
            return {}

        async def _handshake(self, source):
            await self._request(
                "thread/start",
                {"model": ab.live_stt.TRANSLATE_MODEL, "serviceTier": "priority"},
            )
            await self._request("turn/start", {"effort": ab.live_stt.TRANSLATE_EFFORT})

        async def start(self, source):
            if ab.live_stt.TRANSLATE_MODEL == fail_start:
                return False
            await self._handshake(source)
            self.enabled = True
            return True

        async def _translate(self, text, source):
            if not self.enabled:
                return ""
            await self._request("turn/start", {"effort": ab.live_stt.TRANSLATE_EFFORT})
            self._notes.put_nowait(
                {"method": "thread/tokenUsage/updated", "params": {"tokenUsage": {"last": {}}}}
            )
            self._legs[source].turns += 1
            if text == fail_text:
                self.enabled = False  # the third strike: later calls make no attempt
                return ""
            return f"{ab.live_stt.TRANSLATE_MODEL}:{ab.live_stt.TRANSLATE_EFFORT}"

        async def close(self):
            self.closed = True

    monkeypatch.setattr(ab.live_stt, "CodexTranslator", Translator)
    return made


def _arms(*names: str) -> list[ab.Arm]:
    return [ab.Arm(n, *ab.parse_arm(n)) for n in names]


def test_every_request_carries_its_own_arms_pair_and_the_constants_come_back(monkeypatch):
    made = _fake_translators(monkeypatch)
    shipped = ab.live_stt.TRANSLATE_MODEL, ab.live_stt.TRANSLATE_EFFORT
    arms = _arms("m1:low", "m2:none", "m3:low", "m4:none")
    work = [("ja", "一"), ("ja", "二"), ("en", "three"), ("en", "four"), ("ja", "五")]
    rows = asyncio.run(ab.run(arms, work))

    assert all(arm.on_wire() for arm in arms)
    assert all(t.closed for t in made)
    assert all(r["out"] == r["arm"] for r in rows)  # each output came from the arm it is charged to
    assert (ab.live_stt.TRANSLATE_MODEL, ab.live_stt.TRANSLATE_EFFORT) == shipped
    for translator in made:  # both legs warmed before the first measured turn
        assert [m for m, _ in translator.sent[:4]] == ["thread/start", "turn/start"] * 2
    assert [r["arm"] for r in rows] == [a.name for o in ab.schedule(arms, 5) for a in o]
    assert [r["after"] for r in rows[1:]] == [r["arm"] for r in rows[:-1]]


def test_the_wire_check_fails_an_arm_whose_request_carried_another_pair():
    arm = ab.Arm("m:low", "m", "low")
    arm.wire = [("thread", "m", "priority", "priority"), ("turn", "low")]
    assert arm.on_wire()
    for bad in (
        [("thread", "other", "priority", "priority"), ("turn", "low")],
        [("thread", "m", "priority", None), ("turn", "low")],  # tier dropped silently
        [("thread", "m", "priority", "priority"), ("turn", "none")],
        [("thread", "m", "priority", "priority")],
    ):
        arm.wire = bad
        assert not arm.on_wire(), bad


def test_a_failed_start_still_closes_every_translator_it_made(monkeypatch):
    made = _fake_translators(monkeypatch, fail_start="m2")
    with pytest.raises(SystemExit, match="m2:low"):
        asyncio.run(ab.run(_arms("m1:low", "m2:low", "m3:low"), [("ja", "一")]))
    assert made and all(t.closed for t in made)
    assert ab.live_stt.TRANSLATE_MODEL != "m2"


def test_a_failed_turn_and_the_no_attempts_after_it_never_count_as_fast(monkeypatch):
    _fake_translators(monkeypatch)
    arms = _arms("c:low", "x:low")
    work = [("ja", "一"), ("ja", "FAIL"), ("ja", "三")]
    rows = asyncio.run(ab.run(arms, work))
    assert Counter(r["status"] for r in rows) == {"ok": 2, "failed": 2, "disabled": 2}
    assert all(r["seconds"] is None for r in rows if r["status"] != "ok")
    summary = ab.summarize(arms, rows, reps=REPS)
    assert summary["ja"]["x:low"]["failed"] == 2
    assert summary["ja"]["x:low"]["iid"]["n"] == 1
    assert summary["verdict"]["x:low"].startswith("ineligible")


@pytest.mark.parametrize("k", range(1, 8))
def test_each_block_is_position_and_predecessor_balanced(k):
    arms = [f"a{j}" for j in range(k)]
    rows = ab.williams(k)
    width = len(rows)
    for first in (0, width, 3 * width):
        block = ab.schedule(arms, first + width, seed=k)[first:]
        assert all(sorted(order) == arms for order in block)
        for position in range(k):
            assert Counter(order[position] for order in block) == dict.fromkeys(arms, width // k)
        follows = Counter(pair for order in block for pair in itertools.pairwise(order))
        assert set(follows) == {(a, b) for a in arms for b in arms if a != b}
        assert len(set(follows.values())) <= 1


def test_paired_is_antisymmetric_shift_equivariant_and_drops_failed_items():
    rng = random.Random(20261105)  # noqa: S311 — generated contract cases, fixed seed
    for _ in range(12):
        n = rng.randint(1, 30)
        control = [rng.randint(500, 5000) for _ in range(n)]
        challenger = [rng.randint(500, 5000) for _ in range(n)]
        for block in (1, 4, 8):
            base = ab.paired(control, challenger, block=block, reps=REPS)
            swapped = ab.paired(challenger, control, block=block, reps=REPS)
            assert (swapped["median"], swapped["lo"], swapped["hi"], swapped["p90"]) == (
                -base["median"],
                -base["hi"],
                -base["lo"],
                -base["p90"],
            )
            assert (swapped["p90_lo"], swapped["p90_hi"]) == (-base["p90_hi"], -base["p90_lo"])
            c = rng.randint(-400, 400)
            shifted = ab.paired(control, [x + c for x in challenger], block=block, reps=REPS)
            keys = ("median", "lo", "hi", "p90", "p90_lo", "p90_hi")
            assert [shifted[k] - base[k] for k in keys] == [c] * len(keys)
            assert base["lo"] <= base["hi"] and base["p90_lo"] <= base["p90_hi"]
        at = rng.randint(0, n)
        base = ab.paired(control, challenger, reps=REPS)
        for into_control, into_challenger in (([None], [100]), ([100], [None])):
            holed = ab.paired(
                control[:at] + into_control + control[at:],
                challenger[:at] + into_challenger + challenger[at:],
                reps=REPS,
            )
            assert holed == base


def test_the_adjusted_alpha_widens_the_interval():
    rng = random.Random(7)  # noqa: S311 — generated contract cases, fixed seed
    control = [rng.uniform(1, 3) for _ in range(40)]
    challenger = [rng.uniform(1, 3) for _ in range(40)]
    wide = ab.paired(control, challenger, alpha=0.05 / 3, reps=REPS)
    narrow = ab.paired(control, challenger, reps=REPS)
    assert wide["lo"] <= narrow["lo"] <= narrow["hi"] <= wide["hi"]
    assert (wide["lo"], wide["hi"]) != (narrow["lo"], narrow["hi"])  # the level reached the cut


def _contrast(median, lo, hi, *, blocks_hi=None):
    ci = {"median": median, "lo": lo, "hi": hi}
    return {
        "iid": ci,
        "adjusted": ci,
        **{f"block{b}": {**ci, "hi": hi if blocks_hi is None else blocks_hi} for b in ab.BLOCKS},
    }


@pytest.mark.parametrize(
    ("ja", "en", "clean", "want"),
    [
        (_contrast(-0.4, -0.6, -0.2), _contrast(0.0, -0.2, 0.2), True, "faster"),
        (_contrast(-0.4, -0.6, -0.2), None, False, "ineligible"),
        (_contrast(-0.4, -0.6, 0.05), None, True, "no convincing"),  # CI spans 0
        (_contrast(-0.15, -0.3, -0.05), None, True, "no convincing"),  # below MIN_GAIN_S
        (_contrast(-0.4, -0.6, -0.2), _contrast(0.3, 0.1, 0.5), True, "vetoed"),
        (_contrast(-0.4, -0.6, -0.2, blocks_hi=0.01), None, True, "fragile"),
    ],
)
def test_the_frozen_rule(ja, en, clean, want):
    assert ab.verdict(ja, en, clean).startswith(want)


def test_an_arm_names_a_model_and_an_effort():
    assert ab.parse_arm("gpt-6.1-terra:none") == ("gpt-6.1-terra", "none")
    for bad in ("gpt-6-luna", ":low", "gpt-6-luna:"):
        with pytest.raises(argparse.ArgumentTypeError):
            ab.parse_arm(bad)


def test_the_inputs_are_the_committed_story_in_both_directions():
    work = ab.items(60, 40)
    assert [s for s, _ in work] == ["ja"] * 60 + ["en"] * 40
    assert all(text.strip() for _, text in work)
    assert work[0][1] == "木切れなどがごちゃごちゃ入っていましたが、"
