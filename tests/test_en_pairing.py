"""Model-free locks on the live pairing probe (tests/eval_en_pairing.py).

M12.5 rules on M12.3's two dead pairings, and the ruling rests on one rule and
one artifact. The rule is D-015's: a rendering is acquired only where the English
carries a proper noun, so a key an English COMMON noun names can never pair
however open its gate is. The artifact is the committed live run, which is bound
to the caption trace it was translated from and must keep re-deriving the same
verdict.

The second thing these pin is why the live arm is not the simulation: a term that
never pairs never leaves `observe_en`'s gate, so it keeps closing its
NEIGHBOURS' openings -- interference the best case removes by construction.
"""

from __future__ import annotations

import json

import pytest

from live_stt import CONTEXT_EN_SUPPORT, CONTEXT_TERM_LEASE, CONTEXT_TERM_SUPPORT
from tests.eval_en_pairing import (
    CANDIDATES,
    CONTROL,
    TURNS,
    caption_stream,
    replay,
    verdict,
)
from tests.eval_term_census import learner

TERM = "兵十"
COMMON = "The marker post arrived."  # no capital that is not sentence-initial
PROPER = "The visitor was Hyoju."


def _captions(*texts: str) -> list[dict]:
    return [{"idx": i, "text": t} for i, t in enumerate(texts, 1)]


def _turns(captions: list[dict], en: str) -> list[dict]:
    return [{"idx": c["idx"], "ja": c["text"], "en": en} for c in captions]


def _episodes(captions: list[dict], en: str) -> dict[str, dict]:
    return {e["term"]: e for e in replay(captions, _turns(captions, en))["episodes"]}


def test_english_common_nouns_never_pair_a_key():
    """M12.5's whole mechanism, and the reason its prediction was a refutation.

    It also inverts M12.3's opening count. Openings saturate at
    CONTEXT_EN_SUPPORT only because a paired term leaves the gate, so a key real
    English never pairs opens on EVERY later sighting -- more openings than the
    best case, and still no rendering.
    """
    last = CONTEXT_TERM_SUPPORT + CONTEXT_EN_SUPPORT
    caps = _captions(*["兵十がきた。"] * last)
    common = _episodes(caps, COMMON)[TERM]
    assert common["openings"] == list(range(CONTEXT_TERM_SUPPORT, last + 1))
    assert common["paired_at"] is None and common["rendering"] is None
    proper = _episodes(caps, PROPER)[TERM]
    assert proper["openings"] == list(range(CONTEXT_TERM_SUPPORT, last))
    assert proper["paired_at"] == last - 1 and proper["rendering"] == "Hyoju"


def test_an_unpaired_key_keeps_closing_its_neighbours_openings():
    """Why the live arm cannot be read off the simulation (D-015).

    The best case retires a term from the gate as soon as it pairs, which frees
    its neighbour to open; a term real English never pairs stays in the gate and
    keeps both of them shut.
    """
    caps = _captions(*["兵十と鼻腔。"] * CONTEXT_TERM_SUPPORT, "兵十がきた。", "兵十がきた。")
    caps += [{"idx": 6, "text": "兵十と鼻腔。"}, {"idx": 7, "text": "兵十と鼻腔。"}]
    best = {e["term"]: e for e in learner(caps)["episodes"]}
    assert best[TERM]["paired_at"] == 5  # paired alone, then out of the gate
    assert best["鼻腔"]["openings"] == [6, 7] and best["鼻腔"]["paired_at"] == 7
    live = _episodes(caps, COMMON)
    assert live[TERM]["openings"] == [4, 5] and live[TERM]["paired_at"] is None
    assert live["鼻腔"]["openings"] == []  # 兵十 never left the gate


def test_verdict_confines_the_comparison_to_the_pairing_columns():
    """`observe_ja` never reads `renderings`, so trust itself cannot move."""
    caps = _captions(*["兵十がきた。"] * 4, *["ごんは走った。"] * CONTEXT_TERM_LEASE)
    result = verdict(caps, _turns(caps, COMMON))
    row = next(r for r in result["episodes"] if r["term"] == TERM)
    assert row["dead_pairing"] == {"best": True, "live": False}
    assert row["paired_at"] == {"best": 4, "live": None}
    assert result["dead_pairings"] == {"best": [TERM], "live": []}
    assert result["same_episodes"] and result["trust_identical"]
    assert [o["names"] for o in row["names_at_openings"]] == [[]] * len(row["names_at_openings"])


@pytest.mark.parametrize(
    "turns",
    [
        pytest.param([{"idx": 1, "ja": "兵十がきた。", "en": "x"}], id="missing-caption"),
        pytest.param(
            [
                {"idx": 1, "ja": "兵十がきた。", "en": "x"},
                {"idx": 2, "ja": "べつの話。", "en": "x"},
            ],
            id="translated-other-text",
        ),
        pytest.param(
            [
                {"idx": 1, "ja": "兵十がきた。", "en": "x"},
                {"idx": 2, "ja": "兵十がくる。", "en": "x"},
                {"idx": 2, "ja": "兵十がくる。", "en": "y"},
            ],
            id="repeated-index",  # every caption covered, so only the duplicate can refuse it
        ),
    ],
)
def test_a_turn_trace_that_is_not_this_caption_stream_is_refused(turns: list[dict]):
    """A verdict is only worth anything bound to the captions it was measured on."""
    with pytest.raises(SystemExit):
        replay(_captions("兵十がきた。", "兵十がくる。"), turns)


def test_a_prompted_trace_is_refused(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """An empty `prompted` set is only faithful where nothing was ever biased."""
    from tests import eval_en_pairing

    trace = json.loads(eval_en_pairing.TRACE.read_text(encoding="utf-8"))
    trace["run"]["hotwords_reachable"] = True
    forged = tmp_path / "caption_trace.json"
    forged.write_text(json.dumps(trace, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(eval_en_pairing, "TRACE", forged)
    with pytest.raises(SystemExit):
        eval_en_pairing.caption_stream()


def test_the_committed_run_still_yields_m125s_verdict():
    """The ruling, re-derived: the control pairs and neither candidate does.

    This is what a change to `observe_en`, `_TERM_RUN` or the caption trace has
    to face -- M12.5 closed M12 on a structural refutation, so the refutation is
    a regression surface, not a one-off report.
    """
    captions = caption_stream()
    trace = json.loads(TURNS.read_text(encoding="utf-8"))
    result = verdict(captions, trace["turns"])
    assert trace["run"]["n_declined"] == 0  # M13.1's screen never fires on real narration
    assert result["trust_identical"] and result["same_episodes"]
    assert result["control_paired"] is True
    assert result["candidates_paired"] == []
    assert set(result["dead_pairings"]["best"]) == set(CANDIDATES)
    assert CONTROL not in result["dead_pairings"]["live"]
