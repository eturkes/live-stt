"""The repetition probe owns its translator even when a turn fails before timing starts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tests import eval_translate_repeat as repeat
from tests.evaluator_fakes import hermetic_evaluator  # noqa: F401

pytestmark = pytest.mark.usefixtures("hermetic_evaluator")


def test_failed_probe_reaps_its_translator(monkeypatch):
    closed = []

    async def new_thread():
        raise RuntimeError("thread startup failed")

    class Translator:
        def __init__(self):
            self._notes = asyncio.Queue()
            self._legs = {"ja": SimpleNamespace(thread_id=None)}

        async def start(self):
            return True

        async def close(self):
            closed.append(self)

        _new_thread = staticmethod(new_thread)

    monkeypatch.setattr(repeat.live_stt, "CodexTranslator", Translator)
    monkeypatch.setattr(repeat, "controls", lambda: {n: "日本語" for n in repeat.SPANS})
    with pytest.raises(RuntimeError, match="thread startup failed"):
        asyncio.run(repeat.run(None))
    assert len(closed) == 1
