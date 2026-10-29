"""Hermetic evaluator probes: every external boundary must be explicitly faked."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import urllib.request

import pytest

import live_stt
import replay


@pytest.fixture
def hermetic_evaluator(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("unmocked evaluator boundary reached")

    # asyncio needs a local self-pipe; allocate it before socket boundaries close.
    with asyncio.Runner() as runner:
        runner.get_loop()
        monkeypatch.setattr(asyncio, "run", runner.run)
        for module, names in (
            (subprocess, ("Popen", "run", "check_output", "check_call", "call")),
            (urllib.request, ("urlopen", "urlretrieve")),
            (socket, ("socket", "socketpair", "create_connection", "getaddrinfo")),
            (live_stt, ("load_recognizer", "WhisperEngine", "make_vad")),
            (replay, ("_recognizer", "replay_wav", "replay_recognizer", "_run_recognizer")),
        ):
            for name in names:
                monkeypatch.setattr(module, name, forbidden)
        yield
