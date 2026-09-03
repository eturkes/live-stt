"""Model-free locks on the term census (tests/eval_term_census.py).

The census decides M12's branch, and these pin the rules that decide it. M12.1's:
an occurrence is located by ALIGNMENT rather than by searching for guessed
spellings, and it counts only when the recogniser's rendering is a `_TERM_RUN`
candidate -- a lone kanji is invisible to the learner however often it recurs.
M12.3's: a trust episode ends as a DEAD PAIRING only when a rendering was really
acquired and then lost, so a key that goes quiet unpaired, or is dropped for
capacity rather than by lease, is not one.
"""

from __future__ import annotations

from live_stt import CONTEXT_MAX_TERMS, CONTEXT_TERM_LEASE, _en_names
from tests.eval_term_census import _placeholder, census, learner, norm_map, story_census

TERM = "兵十"


def _captions(*texts: str) -> list[dict]:
    return [{"idx": i, "text": t} for i, t in enumerate(texts, 1)]


def _episodes(captions: list[dict]) -> dict[str, dict]:
    """Trust episodes keyed by term; a term opening two of them is a fixture error."""
    rows = learner(captions)["episodes"]
    assert len({e["term"] for e in rows}) == len(rows)
    return {e["term"]: e for e in rows}


def test_norm_map_offsets_point_at_raw_characters():
    text = "「兵十だな」と、ごんは思いました。"
    norm, src = norm_map(text)
    assert "、" not in norm and "「" not in norm  # punctuation dropped
    assert "".join(text[k] for k in src) == norm
    assert text[src[norm.index("兵")]] == "兵"


def test_one_stable_form_reaches_support():
    ref = "きょうは兵十がきた。あしたも兵十がくる。あさっても兵十がいる。"
    caps = _captions("きょうは兵十がきた。", "あしたも兵十がくる。", "あさっても兵十がいる。")
    result = census(TERM, ref, caps)
    assert result["n_occurrences"] == 3
    assert result["n_recognized_as_candidate"] == 3
    assert [(f["form"], f["n_captions"], f["reaches_support"]) for f in result["forms"]] == [
        ("兵十", 3, True)
    ]
    assert _episodes(caps)[TERM]["trusted_at"] == 3


def test_split_key_leaves_every_form_below_support():
    ref = "きょうは兵十がきた。あしたも兵十がくる。あさっても兵十がいる。"
    caps = _captions("きょうは標準がきた。", "あしたも兵十がくる。", "あさっても兵重がいる。")
    result = census(TERM, ref, caps)
    assert result["n_recognized_as_candidate"] == 3
    assert {f["form"] for f in result["forms"]} == {"標準", "兵十", "兵重"}
    assert not any(f["reaches_support"] for f in result["forms"])
    assert TERM not in _episodes(caps)


def test_single_kanji_rendering_is_not_a_candidate():
    """A form below _TERM_RUN's floor can never be learned, however often it recurs."""
    ref = "きょうは兵十がきた。"
    result = census(TERM, ref, _captions("きょうは氷がきた。"))
    assert result["occurrences"][0]["hyp"] == "氷"
    assert result["occurrences"][0]["form"] is None
    assert result["forms"] == [] and result["n_recognized_as_candidate"] == 0


def test_deleted_occurrence_reports_no_caption():
    ref = "きょうは兵十がきた。"
    result = census(TERM, ref, _captions("きょうはがきた。"))
    assert result["n_dropped"] == 1
    assert result["occurrences"][0]["caption"] is None


def test_story_census_aligns_each_section_against_its_own_reference():
    """One alignment per section, so an occurrence lands in the audio it was read from."""
    ref = "きょうは兵十がきた。"
    first, second = _captions("きょうは標柱がきた。"), [{"idx": 2, "text": "きょうは標準がきた。"}]
    result = story_census(TERM, [("01", ref, first), ("02", ref, second)])
    assert [(r["section"], r["caption"], r["form"]) for r in result["occurrences"]] == [
        ("01", 1, "標柱"),
        ("02", 2, "標準"),
    ]
    assert {f["form"]: f["n_captions"] for f in result["forms"]} == {"標柱": 1, "標準": 1}


def test_learner_skips_empty_captions_and_reports_live_terms():
    caps = _captions("兵十がきた。", "", "兵十がくる。", "", "兵十がいる。")
    result = learner(caps)
    assert result["n_published"] == 3
    assert TERM in result["trusted_at_end"]
    episode = _episodes(caps)[TERM]
    assert episode["trusted_at"] == 5  # third PUBLISHED caption, idx 5
    assert episode["openings"] == [5]  # trusted only from its third sighting


def test_a_second_trusted_term_in_one_caption_closes_the_pairing_opening():
    """observe_en takes evidence only where the alignment is forced (D-015)."""
    caps = _captions(*["兵十と鼻腔。"] * 3, "兵十がきた。", "兵十と鼻腔。")
    result = learner(caps)
    assert {e["term"] for e in result["episodes"]} == {TERM, "鼻腔"}
    episodes = _episodes(caps)
    assert episodes[TERM]["openings"] == [4]  # captions 3 and 5 carry both
    assert episodes["鼻腔"]["openings"] == []


def test_a_paired_key_that_goes_quiet_is_a_dead_pairing():
    """The mode M12.3 screens for, at its sharpest: paired on the last sighting."""
    caps = _captions(*["兵十がきた。"] * 4, *["ごんは走った。"] * CONTEXT_TERM_LEASE)
    episode = _episodes(caps)[TERM]
    assert (episode["trusted_at"], episode["paired_at"], episode["last_sighting"]) == (3, 4, 4)
    assert episode["sightings_after_paired"] == 0  # the rendering never served a caption
    assert episode["published_after"] == CONTEXT_TERM_LEASE
    assert episode["expired_at"] == 4 + CONTEXT_TERM_LEASE
    assert episode["mechanism"] == "lease" and episode["dead_pairing"] is True
    assert learner(caps)["dead_pairings"] == [TERM]


def test_a_key_still_being_said_holds_its_pairing():
    caps = _captions(*["兵十がきた。"] * (CONTEXT_TERM_LEASE + 4))
    episode = _episodes(caps)[TERM]
    assert episode["expired_at"] is None and episode["dead_pairing"] is False
    assert episode["sightings_after_paired"] == CONTEXT_TERM_LEASE
    assert learner(caps)["dead_pairings"] == []


def test_a_key_that_goes_quiet_unpaired_is_not_a_dead_pairing():
    """Nothing was acquired, so nothing was stranded -- the guard against over-reporting."""
    caps = _captions(*["兵十と鼻腔。"] * 3, *["ごんは走った。"] * CONTEXT_TERM_LEASE)
    episode = _episodes(caps)[TERM]
    assert episode["openings"] == [] and episode["paired_at"] is None
    assert episode["expired_at"] == 3 + CONTEXT_TERM_LEASE
    assert episode["dead_pairing"] is False
    assert learner(caps)["dead_pairings"] == []


def test_capacity_eviction_is_not_reported_as_a_lapsed_lease():
    """CONTEXT_MAX_TERMS drops the stalest term long before its lease runs out."""
    crowd = [f"{c}山" for c in "甲乙丙丁戊己庚辛壬癸子丑"]
    assert len(crowd) == CONTEXT_MAX_TERMS
    caps = _captions(*["、".join(crowd) + "。"] * 3, *["戌山がきた。"] * 3)
    episode = _episodes(caps)[crowd[0]]  # the stalest of twelve tied at caption 3
    assert episode["expired_at"] == 6 and episode["mechanism"] == "eviction"
    assert episode["published_after"] < CONTEXT_TERM_LEASE
    assert episode["dead_pairing"] is False
    assert _episodes(caps)["戌山"]["trusted_at"] == 6


def test_pairing_placeholders_are_distinct_proper_nouns():
    """A shared spelling would let two terms confirm each other's rendering."""
    names = [_placeholder(i) for i in range(30)]
    assert len(set(names)) == 30
    assert all(_en_names(f"the story mentions {name}.") == [name] for name in names)
