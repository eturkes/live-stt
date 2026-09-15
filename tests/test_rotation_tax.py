"""Locks for the EN-leg thread-rotation tax, a closed `.agent/deferred.md` row.

A glossary change rotates the codex thread inline, and that part is right: a newly
trusted term reaches the model through `developerInstructions` alone. What was
wrong is rotating for a glossary whose CONTENT never moved. The brief listed terms
in `terms()` order, which recency reshuffles on every sighting, so 26 of the 39
rotations over the committed 215-caption trace bought a fresh thread for a
reordered copy of the same glossary, at 3.900 s against 2.085 s steady.

Both halves are locked here: the brief is a pure function of glossary CONTENT, and
every real change -- a new term, a changed rendering, a dropped term -- still
rotates. Recency keeps its two real jobs, ordering the recogniser's bounded prompt
and driving eviction, and the turn cadence arm is untouched.
"""

from __future__ import annotations

import asyncio
import json

import live_stt
from tests.eval_en_pairing import TURNS

TERM_RENDERINGS = {"兵十": "Hyoju", "加助": "Kasuke"}


def _promote(context: live_stt.SessionContext, *terms: str) -> None:
    for term in terms:
        for _ in range(live_stt.CONTEXT_TERM_SUPPORT):
            context.observe_ja(term)


def _pair(context: live_stt.SessionContext, term: str, rendering: str) -> None:
    for _ in range(live_stt.CONTEXT_EN_SUPPORT):
        context.observe_en(term, f"The visitor was {rendering}.")


def test_the_brief_is_a_pure_function_of_glossary_content():
    first = live_stt.SessionContext()
    second = live_stt.SessionContext()
    _promote(first, *TERM_RENDERINGS)
    _promote(second, *reversed(TERM_RENDERINGS))
    for term, rendering in TERM_RENDERINGS.items():
        _pair(first, term, rendering)
        _pair(second, term, rendering)

    assert set(first.terms()) == set(second.terms())
    assert first.terms() != second.terms()  # same content, different sighting recency
    assert first.renderings == second.renderings == TERM_RENDERINGS
    assert first.translator_brief() == second.translator_brief()


def test_a_new_term_still_changes_the_brief():
    context = live_stt.SessionContext()
    _promote(context, "兵十")
    _pair(context, "兵十", "Hyoju")
    before = context.translator_brief()

    _promote(context, "加助")
    after = context.translator_brief()

    assert "加助" in after
    assert after != before


def test_a_changed_rendering_still_changes_the_brief():
    context = live_stt.SessionContext()
    _promote(context, "兵十")
    before = context.translator_brief()

    _pair(context, "兵十", "Hyoju")
    after = context.translator_brief()

    assert "兵十 = Hyoju" in after
    assert after != before


def test_a_dropped_term_still_changes_the_brief():
    context = live_stt.SessionContext()
    _promote(context, "兵十")
    _pair(context, "兵十", "Hyoju")
    before = context.translator_brief()

    for _ in range(live_stt.CONTEXT_TERM_LEASE):
        context.observe_ja("あ")
    after = context.translator_brief()

    assert context.terms() == []
    assert context.renderings == {}
    assert after != before


def test_the_committed_trace_replays_under_fifteen_brief_changes():
    """The row's own bar, replayed over the artifact it was written against.

    Production order, which is what makes the count real: `observe_ja` at
    publication, then `_translate` comparing the brief against the live thread's,
    then `observe_en` on the English that came back.
    """
    turns = json.loads(TURNS.read_text())["turns"]
    context = live_stt.SessionContext()
    brief = ""
    changes = 0
    for turn in turns:
        context.observe_ja(turn["ja"])
        current = context.translator_brief()
        if current != brief:
            changes += 1
            brief = current
        if turn["en"]:
            context.observe_en(turn["ja"], turn["en"])

    assert len(turns) == 215
    assert changes < 15


def test_recency_still_orders_terms_and_hotwords():
    context = live_stt.SessionContext()
    _promote(context, "兵十", "加助")  # equal length, so recency alone breaks the tie
    brief = context.translator_brief()

    assert context.terms() == ["加助", "兵十"]
    assert context.asr_hotwords() == ("加助、兵十", frozenset({"加助", "兵十"}))

    context.observe_ja("兵十")  # an un-prompted sighting renews trust and reorders

    assert context.terms() == ["兵十", "加助"]
    assert context.asr_hotwords()[0] == "兵十、加助"
    assert context.translator_brief() == brief  # the thread it is on survives it


def test_recency_still_evicts_the_least_recently_seen_term():
    context = live_stt.SessionContext()
    kana = "アイウエオカキクケコサシスセソタチツテトナニヌネノハ"
    terms = [kana[i : i + 2] for i in range(0, len(kana), 2)]
    assert len(terms) == live_stt.CONTEXT_MAX_TERMS + 1

    _promote(context, *terms)

    assert terms[0] not in context.terms()
    assert set(context.terms()) == set(terms[1:])


def test_the_turn_cadence_rotation_is_untouched():
    """`_turns % TRANSLATE_ROTATE_TURNS` is tested BEFORE the increment, so the
    101st turn is the one that pays the cadence rotation."""

    async def scenario():
        context = live_stt.SessionContext()
        translator = live_stt.CodexTranslator(context)
        translator.enabled = True
        translator._thread_id = "th-1"
        rotations = []

        async def fake_new_thread():
            rotations.append(translator._turns)
            return "th-2"

        async def fake_turn(ja):
            return "ok"

        translator._new_thread = fake_new_thread  # type: ignore[method-assign]
        translator._turn = fake_turn  # type: ignore[method-assign]

        assert await translator._translate("こんにちは") == "ok"
        assert rotations == []  # the first turn never rotates on the cadence

        translator._turns = live_stt.TRANSLATE_ROTATE_TURNS - 1
        assert await translator._translate("あ") == "ok"
        assert rotations == []  # the 100th turn still rides the thread it has
        assert await translator._translate("い") == "ok"

        assert rotations == [live_stt.TRANSLATE_ROTATE_TURNS]  # the 101st pays it
        assert translator._turns == live_stt.TRANSLATE_ROTATE_TURNS + 1

    asyncio.run(scenario())
