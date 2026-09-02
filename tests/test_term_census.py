"""Model-free locks on M12.1's term census (tests/eval_term_census.py).

The census decides M12's branch, and its two non-obvious rules are what these
pin: an occurrence is located by ALIGNMENT rather than by searching for guessed
spellings, and it counts only when the recogniser's rendering is a `_TERM_RUN`
candidate -- a lone kanji is invisible to the learner however often it recurs.
"""

from __future__ import annotations

from tests.eval_term_census import census, learner, norm_map

TERM = "兵十"


def _captions(*texts: str) -> list[dict]:
    return [{"idx": i, "text": t} for i, t in enumerate(texts, 1)]


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
    assert learner(caps)["first_trusted"]["兵十"] == 3


def test_split_key_leaves_every_form_below_support():
    ref = "きょうは兵十がきた。あしたも兵十がくる。あさっても兵十がいる。"
    caps = _captions("きょうは標準がきた。", "あしたも兵十がくる。", "あさっても兵重がいる。")
    result = census(TERM, ref, caps)
    assert result["n_recognized_as_candidate"] == 3
    assert {f["form"] for f in result["forms"]} == {"標準", "兵十", "兵重"}
    assert not any(f["reaches_support"] for f in result["forms"])
    assert "兵十" not in learner(caps)["first_trusted"]


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


def test_learner_skips_empty_captions_and_reports_live_terms():
    caps = _captions("兵十がきた。", "", "兵十がくる。", "", "兵十がいる。")
    result = learner(caps)
    assert result["n_published"] == 3
    assert result["first_trusted"]["兵十"] == 5  # third PUBLISHED caption, idx 5
    assert "兵十" in result["trusted_at_end"]
    assert result["openings"] == {"兵十": [5]}  # trusted only from its third sighting


def test_a_second_trusted_term_in_one_caption_closes_the_pairing_opening():
    """observe_en takes evidence only where the alignment is forced (D-015)."""
    caps = _captions(*["兵十と鼻腔。"] * 3, "兵十がきた。", "兵十と鼻腔。")
    result = learner(caps)
    assert set(result["first_trusted"]) == {"兵十", "鼻腔"}
    assert result["openings"] == {"兵十": [4]}  # caption 5 carries both, so it is no evidence
