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
from copy import deepcopy

import pytest

from live_stt import (
    _EN_STOP,
    CONTEXT_EN_SUPPORT,
    CONTEXT_TERM_LEASE,
    CONTEXT_TERM_SUPPORT,
)
from tests import eval_en_pairing
from tests.eval_en_pairing import (
    CANDIDATES,
    CONTROL,
    TURNS,
    caption_stream,
    replay,
    report,
    verdict,
)
from tests.eval_term_census import learner

TERM = "兵十"
COMMON = "The marker post arrived."  # no capital that is not sentence-initial
PROPER = "The visitor was Hyoju."
HEADLINE = "no paired term has MORE THAN ONE supported spelling on the raw stream"


def _captions(*texts: str) -> list[dict]:
    return [{"idx": i, "text": t} for i, t in enumerate(texts, 1)]


def _turns(captions: list[dict], en: str) -> list[dict]:
    return [{"idx": c["idx"], "ja": c["text"], "en": en} for c in captions]


def _episodes(captions: list[dict], en: str) -> dict[str, dict]:
    return {e["term"]: e for e in replay(captions, _turns(captions, en))["episodes"]}


def _raw_spellings(turns: list[dict]) -> dict[str, dict]:
    return eval_en_pairing.raw_spellings(turns)  # type: ignore[attr-defined]


def _re_trusted_unpaired() -> tuple[list[dict], list[dict]]:
    """Pairs, spells itself two SUPPORTED ways, expires with its lease, re-trusts unpaired."""
    names = ["Hyoju"] * CONTEXT_EN_SUPPORT + ["Gon"] * CONTEXT_EN_SUPPORT
    caps = _captions(
        *["兵十がきた。"] * (CONTEXT_TERM_SUPPORT + len(names)),
        *["そうですね。"] * (CONTEXT_TERM_LEASE + 1),  # hiragana only: no candidate at all
        *["兵十がきた。"] * CONTEXT_TERM_SUPPORT,
    )
    turns = _turns(caps, "")
    paired_window = turns[CONTEXT_TERM_SUPPORT : CONTEXT_TERM_SUPPORT + len(names)]
    for turn, name in zip(paired_window, names, strict=True):
        turn["en"] = f"The visitor was {name}."
    return caps, turns


def test_two_trusted_terms_count_a_turn_without_a_spelling_reading():
    """Two JA candidates make attribution unsafe, but both still expose coverage."""
    caps = _captions(
        *["兵十がきた。"] * CONTEXT_TERM_SUPPORT,
        *["加助がきた。"] * CONTEXT_TERM_SUPPORT,
        "兵十と加助。",
    )
    turns = _turns(caps, "")
    turns[-1]["en"] = PROPER
    empty = {"turns": 1, "readings": 0, "spellings": {}, "distinct": 0, "supported": 0}
    assert _raw_spellings(turns) == {TERM: empty, "加助": empty}


def test_two_english_names_count_a_turn_without_a_spelling_reading():
    """Two EN proper nouns make attribution unsafe while the turn stays visible."""
    caps = _captions(*["兵十がきた。"] * (CONTEXT_TERM_SUPPORT + 1))
    turns = _turns(caps, "")
    turns[-1]["en"] = "The visitors saw Hyoju and Gon."
    assert _raw_spellings(turns)[TERM] == {
        "turns": 1,
        "readings": 0,
        "spellings": {},
        "distinct": 0,
        "supported": 0,
    }


def test_a_term_keeps_contributing_readings_after_it_pairs():
    """The raw census must not inherit `observe_en`'s tautological paired-term filter."""
    caps = _captions(*["兵十がきた。"] * (CONTEXT_TERM_SUPPORT + CONTEXT_EN_SUPPORT + 1))
    turns = _turns(caps, "")
    for turn in turns[CONTEXT_TERM_SUPPORT:]:
        turn["en"] = PROPER
    assert _raw_spellings(turns)[TERM] == {
        "turns": CONTEXT_EN_SUPPORT + 1,
        "readings": CONTEXT_EN_SUPPORT + 1,
        "spellings": {"Hyoju": CONTEXT_EN_SUPPORT + 1},
        "distinct": 1,
        "supported": 1,
    }


def test_supported_counts_only_spellings_that_reach_the_pairing_bar():
    """One stray spelling stays visible without becoming a supported inconsistency."""
    caps = _captions(*["兵十がきた。"] * (CONTEXT_TERM_SUPPORT + CONTEXT_EN_SUPPORT * 2 - 1))
    turns = _turns(caps, "")
    names = ["Hyoju"] * CONTEXT_EN_SUPPORT + ["Gon"] * (CONTEXT_EN_SUPPORT - 1)
    for turn, name in zip(turns[CONTEXT_TERM_SUPPORT:], names, strict=True):
        turn["en"] = f"The visitor was {name}."
    assert _raw_spellings(turns)[TERM] == {
        "turns": len(names),
        "readings": len(names),
        "spellings": {"Hyoju": CONTEXT_EN_SUPPORT, "Gon": CONTEXT_EN_SUPPORT - 1},
        "distinct": 2,
        "supported": 1,
    }


def test_spellings_are_ordered_by_count_then_name():
    """Stable reports rank frequent spellings first and break ties lexically."""
    names = ["Gon", "Hyoju", "Anke", "Hyoju", "Gon", "Anke", "Hyoju"]
    caps = _captions(*["兵十がきた。"] * (CONTEXT_TERM_SUPPORT + len(names)))
    turns = _turns(caps, "")
    for turn, name in zip(turns[CONTEXT_TERM_SUPPORT:], names, strict=True):
        turn["en"] = f"The visitor was {name}."
    spelling_counts = _raw_spellings(turns)[TERM]["spellings"]
    assert list(spelling_counts.items()) == [("Hyoju", 3), ("Anke", 2), ("Gon", 2)]


def test_two_supported_spellings_are_already_multi_spelled():
    """The bar is `supported > 1`, not `> CONTEXT_EN_SUPPORT`.

    Two spellings that EACH reached the pairing bar already contradict the one
    rendering the learned map holds, so the second one must not have to clear the
    bar twice. Raising the threshold to `CONTEXT_EN_SUPPORT` leaves the rest of
    this file green, which is why this case exists.
    """
    names = ["Hyoju"] * CONTEXT_EN_SUPPORT + ["Gon"] * CONTEXT_EN_SUPPORT
    caps = _captions(*["兵十がきた。"] * (CONTEXT_TERM_SUPPORT + len(names)))
    turns = _turns(caps, "")
    for turn, name in zip(turns[CONTEXT_TERM_SUPPORT:], names, strict=True):
        turn["en"] = f"The visitor was {name}."
    assert _raw_spellings(turns)[TERM]["supported"] == 2
    assert verdict(caps, turns)["multi_spelled"] == [TERM]


def test_a_term_that_re_trusts_unpaired_keeps_its_raw_inconsistency():
    """`verdict()` keeps ONE episode per term; the census spans the session.

    A term that paired, spelled itself two supported ways, expired with its lease
    and then re-trusted on captions the translator never answered survives in
    `verdict()` carrying the LAST episode, whose `rendering` is None. Gating
    `multi_spelled` on that row hides exactly the inconsistency the metric exists
    to find.
    """
    caps, turns = _re_trusted_unpaired()
    result = verdict(caps, turns)
    row = next(r for r in result["episodes"] if r["term"] == TERM)
    assert row["rendering"] is None  # the surviving episode never paired
    assert row["spellings"]["supported"] == 2  # while the raw stream spelled it twice
    assert result["multi_spelled"] == [TERM]


def test_the_report_headline_states_the_verdict_its_own_rows_carry(
    capsys: pytest.CaptureFixture[str],
):
    """`report()` is the human surface and its headline branch was unlocked.

    Swapping the branch condition for anything the committed trace satisfies
    (`not result["multi_spelled"]` -> `result["episodes"]`) leaves today's output
    byte-identical while every future FAILING run prints the clean headline. The
    histogram rides the same case: `multi_spelled` reads `ever_paired`, so a term
    whose surviving episode carries no rendering is still named by the headline
    and must appear in the table under it.
    """
    caps, turns = _re_trusted_unpaired()
    report(verdict(caps, turns), None)
    failing = capsys.readouterr().out
    histogram = [line for line in failing.splitlines() if TERM in line and "readings" in line]
    clean = _captions(*["兵十がきた。"] * (CONTEXT_TERM_SUPPORT + CONTEXT_EN_SUPPORT))
    clean_turns = _turns(clean, "")
    for turn in clean_turns[CONTEXT_TERM_SUPPORT:]:
        turn["en"] = PROPER
    report(verdict(clean, clean_turns), None)
    passing = capsys.readouterr().out
    assert [
        f"MULTI-SPELLED on the raw stream: {TERM}" in failing,
        HEADLINE in failing,
        [" ".join(line.split()) for line in histogram],
        HEADLINE in passing,
        "MULTI-SPELLED" in passing,
    ] == [
        True,
        False,
        [f"{TERM} -> - 4/4 readings 2 distinct, 2 supported Gon x2, Hyoju x2"],
        True,
        False,
    ]


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


def test_the_committed_run_learns_every_real_name_and_no_pronoun():
    """P-019 + P-020's acceptance, re-derived from the run that measured both.

    The shipped rule ended this session briefing `標柱 = I` — a mis-recognised
    key pinned to a pronoun (P-019). It also read n=182's `…their doing.” Hyōjun
    was startled…` as one sentence, so two names shut the gate and 加助, the
    story's second real character, stayed unspelled (P-020). Every correct
    rendering has to land at the caption that supplies it, and no other entry may
    come back.

    **Pin the pronoun CLASS, not 標柱's null.** The trace moved when the brief
    stopped rotating on recency, and 標柱 now pairs to `Heijū` at n=84 — the
    correct English for the name the recogniser mis-hears as 標柱, which is D-015
    working as designed (a rendering is keyed on the string the recogniser
    produced, not on the string that was spoken). Asserting the null would have
    locked the WEAKER outcome; what P-019 actually forbids is a pronoun reaching
    the brief, and that survives any re-run.
    """
    trace = json.loads(TURNS.read_text(encoding="utf-8"))
    episodes = {e["term"]: e for e in replay(caption_stream(), trace["turns"])["episodes"]}
    learned = {t: (e["rendering"], e["paired_at"]) for t, e in episodes.items() if e["rendering"]}
    assert learned == {
        "ゴン": ("Gon", 57),
        "標柱": ("Heijū", 84),
        "カスケ": ("Kasuke", 181),
        "神様": ("God", 194),
    }
    assert not {r.lower().replace("’", "'") for r, _ in learned.values()} & _EN_STOP


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


def test_the_committed_run_rederives_the_raw_spelling_table():
    """The trace fixes each learned term's raw coverage, noise, and support counts."""
    captions = caption_stream()
    trace = json.loads(TURNS.read_text(encoding="utf-8"))
    census = _raw_spellings(trace["turns"])
    result = verdict(captions, trace["turns"])
    expected = {
        "ゴン": {
            "turns": 38,
            "readings": 8,
            "spellings": {"Gon": 8},
            "distinct": 1,
            "supported": 1,
        },
        "標柱": {
            "turns": 26,
            "readings": 10,
            "spellings": {"Heijū": 9, "Gon": 1},
            "distinct": 2,
            "supported": 1,
        },
        "カスケ": {
            "turns": 4,
            "readings": 2,
            "spellings": {"Gon": 1, "Kasuke": 1},
            "distinct": 2,
            "supported": 0,
        },
        "神様": {
            "turns": 3,
            "readings": 2,
            "spellings": {"God": 2},
            "distinct": 1,
            "supported": 1,
        },
    }
    renderings = {"ゴン": "Gon", "標柱": "Heijū", "カスケ": "Kasuke", "神様": "God"}
    learned_rows = {row["term"]: row for row in result["episodes"] if row["rendering"]}
    assert {
        "raw": {term: census[term] for term in expected},
        "reported": {
            term: {"rendering": row["rendering"], **row["spellings"]}
            for term, row in learned_rows.items()
        },
        "multi_spelled": result["multi_spelled"],
    } == {
        "raw": expected,
        "reported": {
            term: {"rendering": renderings[term], **spellings}
            for term, spellings in expected.items()
        },
        "multi_spelled": [],
    }


def test_raw_spellings_refute_the_learned_map_tautology():
    """The named mutation must move raw consistency while every old proof stays green."""
    captions = caption_stream()
    trace = json.loads(TURNS.read_text(encoding="utf-8"))
    baseline = verdict(captions, trace["turns"])
    paired_at = next(
        row["paired_at"]["live"] for row in baseline["episodes"] if row["term"] == CONTROL
    )
    mutated_turns = deepcopy(trace["turns"])
    replacement_i = 0
    for turn in mutated_turns:
        if turn["idx"] > paired_at and "Gon" in turn["en"]:
            turn["en"] = turn["en"].replace("Gon", ("Gawn", "Ghone")[replacement_i % 2])
            replacement_i += 1
    changed = verdict(captions, mutated_turns)
    baseline_census = _raw_spellings(trace["turns"])
    changed_census = _raw_spellings(mutated_turns)
    learned_before = {
        row["term"]: row["rendering"] for row in baseline["episodes"] if row["rendering"]
    }
    learned_after = {
        row["term"]: row["rendering"] for row in changed["episodes"] if row["rendering"]
    }
    learned = {"ゴン": "Gon", "標柱": "Heijū", "カスケ": "Kasuke", "神様": "God"}
    assert {
        "learned": (learned_before, learned_after),
        "control_paired": (baseline["control_paired"], changed["control_paired"]),
        "candidates_paired": (baseline["candidates_paired"], changed["candidates_paired"]),
        "raw_gon_before": (
            set(baseline_census[CONTROL]["spellings"]),
            baseline_census[CONTROL]["distinct"],
            baseline_census[CONTROL]["supported"],
        ),
        "raw_gon_after": (
            set(changed_census[CONTROL]["spellings"]),
            changed_census[CONTROL]["distinct"],
            changed_census[CONTROL]["supported"],
        ),
        "multi_spelled": (baseline["multi_spelled"], changed["multi_spelled"]),
    } == {
        "learned": (learned, learned),
        "control_paired": (True, True),
        "candidates_paired": ([], []),
        "raw_gon_before": ({"Gon"}, 1, 1),
        "raw_gon_after": ({"Gon", "Gawn", "Ghone"}, 3, 3),
        "multi_spelled": ([], [CONTROL]),
    }
