"""Regression locks for SessionContext's within-session learning contract (D-015).

The property worth protecting is not "it learns terms" but the two bounds that
keep learning from poisoning the recognizer it conditions:

1. Support must come from segments the prompt did not already contain the term
   in. Conditioning on a mis-recognition reproduces it, so without that
   exclusion an error promotes itself on evidence it manufactured.
2. Trust is a lease that only an un-prompted sighting renews. A trusted term is
   normally in the prompt, so it expires on schedule and has to re-earn support
   unaided; without expiry a term the prompt keeps producing is never dislodged.

Both are invisible in ordinary use and easy to delete during a refactor. The
session boundary is locked here too: the picture lives in the object, so a new
object knows nothing.
"""

from __future__ import annotations

from live_stt import (
    CONTEXT_MAX_TERMS,
    CONTEXT_PROMPT_MAX_CHARS,
    CONTEXT_TERM_LEASE,
    CONTEXT_TERM_MEMORY,
    CONTEXT_TERM_SUPPORT,
    SessionContext,
)

DRUG = "プレドニン"
DEPT = "神経内科"


def _see(ctx: SessionContext, term: str, times: int) -> None:
    for _ in range(times):
        ctx.observe_ja(f"{term}を投与しました")


def _see_prompted(ctx: SessionContext, term: str, times: int) -> None:
    """Sight a term the way a biased recognizer would: hotwords, then caption."""
    for _ in range(times):
        _, prompted = ctx.asr_hotwords()
        ctx.observe_ja(f"{term}を投与しました", prompted)


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


def test_prompted_sightings_are_not_evidence():
    """The anti-feedback bound: a term the prompt supplied proves nothing."""
    ctx = SessionContext()
    prompted = frozenset({DRUG})
    for _ in range(CONTEXT_TERM_SUPPORT * 3):
        ctx.observe_ja(f"{DRUG}を投与しました", prompted)
    assert DRUG not in ctx.terms()


def test_trust_expires_without_unprompted_proof():
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    assert DRUG in ctx.terms()
    _see_prompted(ctx, DRUG, CONTEXT_TERM_LEASE)
    assert DRUG not in ctx.terms()


def test_expired_term_can_be_re_earned_unaided():
    ctx = SessionContext()
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    _see_prompted(ctx, DRUG, CONTEXT_TERM_LEASE)
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


def test_hotwords_stay_within_budget():
    ctx = SessionContext("これは" + "長い前置き" * 20)
    for i in range(CONTEXT_MAX_TERMS):
        _see(ctx, f"Candidate{i:03d}", CONTEXT_TERM_SUPPORT)
    terms, _ = ctx.asr_hotwords()
    assert len(terms) <= CONTEXT_PROMPT_MAX_CHARS


def test_hotwords_list_each_term_once():
    """Repetition is what makes a <|startofprev|> payload loop, so spend budget once."""
    ctx = SessionContext(f"{DEPT}の申し送り")
    _see(ctx, DEPT, CONTEXT_TERM_SUPPORT)
    terms, _ = ctx.asr_hotwords()
    assert terms.count(DEPT) == 1


def test_hotwords_report_the_terms_they_carry():
    """The returned set is what observe_ja needs to discount; it must be exact."""
    ctx = SessionContext(DEPT)
    _see(ctx, DRUG, CONTEXT_TERM_SUPPORT)
    terms, biased = ctx.asr_hotwords()
    assert biased == frozenset(ctx.terms())  # nothing trusted is left undeclared
    assert DEPT in biased and DRUG in biased
    assert all(term in terms for term in biased)


def test_seed_is_nfkc_normalized():
    ctx = SessionContext("ＭＲＩ検査")
    assert ctx.seed == "MRI検査"


def test_empty_context_produces_nothing():
    ctx = SessionContext()
    assert ctx.terms() == []
    assert ctx.asr_hotwords() == ("", frozenset())
    assert ctx.translator_brief() == ""


def test_grammar_fragments_are_not_terms():
    """Script runs, not n-grams: hiragana and 2-char kana carry no session identity."""
    ctx = SessionContext()
    for _ in range(CONTEXT_TERM_SUPPORT * 2):
        ctx.observe_ja("それではこちらをごらんください")
    assert ctx.terms() == []


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
