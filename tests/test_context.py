"""Regression locks for SessionContext's within-session learning contract (D-015).

The property worth protecting is not "it learns terms" but the two bounds that
keep the translator glossary to what this session keeps saying:

1. Support must come from CONTEXT_TERM_SUPPORT distinct segments, and a candidate
   is forgotten once CONTEXT_TERM_MEMORY segments pass without a sighting, so one
   caption or a stale sighting promotes nothing.
2. Trust is a lease each sighting renews. A term the session stops saying expires
   on schedule and has to re-earn support; without expiry it holds a slot forever.

Both are invisible in ordinary use and easy to delete during a refactor. The
session boundary is locked here too: the picture lives in the object, so a new
object knows nothing.
"""

from __future__ import annotations

import pytest

from live_stt import (
    CONTEXT_EN_SUPPORT,
    CONTEXT_MAX_TERMS,
    CONTEXT_TERM_LEASE,
    CONTEXT_TERM_MEMORY,
    CONTEXT_TERM_SUPPORT,
    SessionContext,
)

DRUG = "プレドニン"
DEPT = "神経内科"
ONLY_DRUG = f"{DRUG}です"  # one trusted term; "投与" in _see's sentence is a second
RENDERED = "The nurse gave Predonine at eight."
FILLER = "それではこちらをごらんください"  # hiragana alone: no candidate, no renewal


def _pair(ctx: SessionContext, times: int, en: str = RENDERED) -> None:
    for _ in range(times):
        ctx.observe_ja(ONLY_DRUG)
        ctx.observe_en(ONLY_DRUG, en)


def _see(ctx: SessionContext, term: str, times: int) -> None:
    for _ in range(times):
        ctx.observe_ja(f"{term}を投与しました")


def _pass(ctx: SessionContext, times: int) -> None:
    for _ in range(times):
        ctx.observe_ja(FILLER)


def test_promotes_only_at_the_support_threshold():
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT - 1)
    assert DRUG not in ctx.terms()
    _see(ctx, DRUG, 1)
    assert DRUG in ctx.terms()


def test_one_caption_cannot_support_itself():
    """Repetition inside a single segment is one sighting, not many."""
    ctx = SessionContext()
    ctx.observe_ja((f"{DRUG}、" * (CONTEXT_TERM_SUPPORT + 2)) + "を投与")
    assert DRUG not in ctx.terms()


def test_trust_expires_without_a_fresh_sighting():
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pass(ctx, CONTEXT_TERM_LEASE - 1)
    assert DRUG in ctx.terms()
    _pass(ctx, 1)
    assert DRUG not in ctx.terms()


def test_a_sighting_renews_the_lease():
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pass(ctx, CONTEXT_TERM_LEASE - 1)
    _see(ctx, DRUG, 1)
    _pass(ctx, CONTEXT_TERM_LEASE - 1)
    assert DRUG in ctx.terms()


def test_expired_term_can_be_re_earned():
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pass(ctx, CONTEXT_TERM_LEASE)
    assert DRUG not in ctx.terms()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    assert DRUG in ctx.terms()


def test_stale_partial_support_is_forgotten():
    """A sighting older than the memory window cannot complete a promotion."""
    ctx = SessionContext()
    _see(ctx, DRUG, 1)
    for _ in range(CONTEXT_TERM_MEMORY + 1):
        ctx.observe_ja("今日はいい天気です")
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT - 1)
    assert DRUG not in ctx.terms()


def test_partial_support_dies_exactly_at_the_memory_boundary():
    """CONTEXT_TERM_MEMORY segments without a sighting forget a candidate; one fewer does not."""
    for gap, trusted in ((CONTEXT_TERM_MEMORY - 1, True), (CONTEXT_TERM_MEMORY, False)):
        ctx = SessionContext()
        _see(ctx, DRUG, CONTEXT_TERM_SUPPORT - 1)
        _pass(ctx, gap)
        _see(ctx, DRUG, 1)
        assert (DRUG in ctx.terms()) is trusted, gap


def test_a_seed_term_leaves_the_rest_of_its_caption_as_evidence():
    """The seed is excluded from support accounting; the caption around it is not."""
    ctx = SessionContext(DEPT)
    for _ in range(CONTEXT_TERM_SUPPORT):
        ctx.observe_ja(f"{DEPT}で{DRUG}を投与しました")
    assert DRUG in ctx.terms()


def test_seed_is_trusted_at_once_and_never_evicted():
    ctx = SessionContext(f"{DEPT}の申し送り")
    assert DEPT in ctx.terms()
    for i in range(CONTEXT_TERM_LEASE + CONTEXT_MAX_TERMS * 3):
        ctx.observe_ja(f"患者{i}番の血圧を測定しました")
    assert DEPT in ctx.terms()


def test_seed_term_never_enters_the_learned_pool():
    """Seed terms are trusted by authorship, so they must not consume support."""
    ctx = SessionContext(DEPT)
    _see(ctx, DEPT, CONTEXT_TERM_SUPPORT * 2)
    assert ctx.terms().count(DEPT) == 1


def test_learned_pool_is_bounded_and_keeps_the_recent():
    ctx = SessionContext()
    for i in range(CONTEXT_MAX_TERMS * 2):
        _see(ctx, f"Term{i:03d}", CONTEXT_TERM_SUPPORT)
    learned = ctx.terms()
    assert len(learned) == CONTEXT_MAX_TERMS
    assert f"Term{CONTEXT_MAX_TERMS * 2 - 1:03d}" in learned
    assert "Term000" not in learned


def test_overflow_evicts_the_least_recently_proved_term():
    """Recency, not spelling: terms learned in reverse order catch a key that sorts names."""
    ctx = SessionContext()
    for i in range(CONTEXT_MAX_TERMS, -1, -1):  # Term012 first, Term000 last
        _see(ctx, f"Term{i:03d}", CONTEXT_TERM_SUPPORT)
    assert f"Term{CONTEXT_MAX_TERMS:03d}" not in ctx.terms()
    assert "Term000" in ctx.terms()


def test_learned_terms_list_longest_first_after_the_seed():
    ctx = SessionContext(DEPT)
    for term in ("ゴン", DRUG):  # the shorter one learned first, so recency cannot pass for length
        for _ in range(CONTEXT_TERM_SUPPORT):
            ctx.observe_ja(term)
    assert ctx.terms() == [DEPT, DRUG, "ゴン"]


def test_seed_is_nfkc_normalized():
    ctx = SessionContext("ＭＲＩ検査")
    assert ctx.seed == "MRI検査"


def test_empty_context_produces_nothing():
    ctx = SessionContext()
    assert ctx.terms() == []
    assert ctx.translator_brief() == ""


def test_grammar_fragments_are_not_terms():
    """Script runs, not n-grams: hiragana carries grammar, never session identity."""
    ctx = SessionContext()
    for _ in range(CONTEXT_TERM_SUPPORT * 2):
        ctx.observe_ja("それではこちらをごらんください")
    assert ctx.terms() == []


def test_a_two_character_katakana_term_is_learned():
    """The katakana floor matches the kanji one at 2 (M12.4).

    A short native name is what the recogniser writes in katakana, and support,
    not the floor, is what keeps ordinary vocabulary out: over 215 real captions
    a 2-character floor admitted 10 forms and exactly one reached support.
    """
    ctx = SessionContext()
    _see(ctx, "ゴン", CONTEXT_TERM_SUPPORT)
    assert "ゴン" in ctx.terms()


def test_the_floor_still_rejects_a_lone_katakana_character():
    """Lowering the floor to 2 is not removing it; one character carries no identity."""
    ctx = SessionContext()
    for _ in range(CONTEXT_TERM_SUPPORT):
        ctx.observe_ja("コの字にゴンが曲がる")
    assert ctx.terms() == ["ゴン"]


def test_nothing_carries_into_a_new_session():
    ctx = SessionContext(DEPT)
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    assert ctx.terms()
    assert SessionContext().terms() == []


def test_translator_brief_names_the_terms_once_known():
    ctx = SessionContext(f"{DEPT}の申し送り")
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    brief = ctx.translator_brief()
    assert DEPT in brief and DRUG in brief


def test_rendering_reaches_the_brief_only_at_its_support_threshold():
    """The term list says which names matter; the rendering says how to spell them."""
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pair(ctx, CONTEXT_EN_SUPPORT - 1)
    assert f"{DRUG} = Predonine" not in ctx.translator_brief()
    _pair(ctx, 1)
    assert f"{DRUG} = Predonine" in ctx.translator_brief()


def test_disagreeing_renderings_never_reach_support():
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    for spelling in ("Predonine", "Prednisolone", "Predonin", "Prednisone"):
        _pair(ctx, 1, f"The nurse gave {spelling} at eight.")
    assert ctx.renderings == {}


def test_an_ambiguous_turn_is_not_evidence():
    """Two trusted terms or two proper nouns leave the alignment a guess."""
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)  # trusts both プレドニン and 投与
    for _ in range(CONTEXT_EN_SUPPORT * 2):
        ctx.observe_ja(f"{DRUG}を投与しました")
        ctx.observe_en(f"{DRUG}を投与しました", RENDERED)
        ctx.observe_en(ONLY_DRUG, "The nurse gave Predonine to Tanaka.")
    assert ctx.renderings == {}


def test_a_sentence_opening_capital_is_not_a_name():
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pair(ctx, CONTEXT_EN_SUPPORT * 2, "Predonine was given at eight.")
    assert ctx.renderings == {}


@pytest.mark.parametrize(
    "en",
    [
        "The nurse said I gave the dose.",
        "The nurse said I’m late with the dose.",
        "The nurse said I've given the dose.",
        "The nurse said I’d given the dose.",
        "The nurse said I’ll give the dose.",
    ],
)
def test_the_first_person_pronoun_is_never_a_name(en: str):
    """The one word the positional rule cannot judge: English capitalizes "I" anywhere.

    Over 215 real translator turns it was the commonest sole proper noun in the
    stream — 22 of the 63 single-name turns against `Gon`'s 17 — and it pinned
    `標柱 = I` into the brief for 13 sightings (M12.5). The contractions match as
    single tokens because `_EN_NAME`'s class swallows the apostrophe, and both
    apostrophe characters have to fold.
    """
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pair(ctx, CONTEXT_EN_SUPPORT * 2, en)
    assert ctx.renderings == {}
    _pair(ctx, CONTEXT_EN_SUPPORT)  # same open gate, a real mid-sentence name: learned
    assert ctx.renderings[DRUG] == "Predonine"


@pytest.mark.parametrize("opened,closed", [('"', '"'), ("“", "”"), ("‘", "’")])
def test_a_quotation_opens_a_sentence(opened: str, closed: str):
    """A quotation's first word is capitalized by convention, exactly like a sentence's.

    Dropping it is also what leaves the rest of the quote unambiguous: before
    this rule the quoted opener counted as a second proper noun and shut the
    gate on the real one beside it.
    """
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pair(ctx, CONTEXT_EN_SUPPORT * 2, f"The nurse said, {opened}Predonine is late.{closed}")
    assert ctx.renderings == {}
    _pair(ctx, CONTEXT_EN_SUPPORT, f"The nurse said, {opened}It is Predonine again.{closed}")
    assert ctx.renderings[DRUG] == "Predonine"


@pytest.mark.parametrize(
    "en",
    [
        "The nurse said, “The dose is late.” Predonine was given anyway.",
        "The nurse hesitated… Predonine was given anyway.",
    ],
)
def test_a_terminator_ends_its_sentence_behind_a_quote_or_an_ellipsis(en: str):
    """A bare `[.!?]` lookbehind cannot see either boundary, so the next sentence's
    first word reads as a mid-sentence capital -- exactly the thing that IS evidence.

    Measured cost on the committed pairing trace (P-020): n=182 read `…their
    doing.” Hyōjun was startled and looked at Kasuke's face.` as one sentence, so
    two names shut the gate and the correct `カスケ = Kasuke` never landed.
    """
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pair(ctx, CONTEXT_EN_SUPPORT * 2, en)
    assert ctx.renderings == {}
    _pair(ctx, CONTEXT_EN_SUPPORT)  # same open gate, a real mid-sentence name: learned
    assert ctx.renderings[DRUG] == "Predonine"


def test_a_name_after_quoted_speech_is_still_evidence():
    """The straight `"` closes with the character it opens with, so only the opener splits."""
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pair(ctx, CONTEXT_EN_SUPPORT, '"It is late," Predonine was given anyway.')
    assert ctx.renderings[DRUG] == "Predonine"


def test_a_possessive_folds_to_the_bare_name():
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pair(ctx, CONTEXT_EN_SUPPORT, "We raised Predonine’s dose today.")
    assert ctx.renderings[DRUG] == "Predonine"


def test_a_rendering_expires_with_the_term_that_carries_it():
    """Bounds both dicts: a spelling is only ever read for a term the brief lists."""
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pair(ctx, CONTEXT_EN_SUPPORT)
    assert DRUG in ctx.renderings
    _pass(ctx, CONTEXT_TERM_LEASE)
    assert DRUG not in ctx.terms()
    assert ctx.renderings == {}


def test_untranslated_captions_teach_nothing():
    """source-only degradation (D-009) must not feed an empty rendering into the brief."""
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _pair(ctx, CONTEXT_EN_SUPPORT * 2, "")
    assert ctx.renderings == {}
    assert " = " not in ctx.translator_brief()
