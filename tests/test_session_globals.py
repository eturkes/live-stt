"""Tests that drive the real CLI or session restore the process state it writes."""

import os
import sys

import pytest

import live_stt
from tests.session_globals import keep_session_globals
from tests.test_language_detector import run_cli_to_session
from tests.test_playback import DEFAULT, HARDWARE, PIPEWIRE, _run_session
from tests.test_shipped_path import run_cli


@pytest.mark.parametrize("drive", [run_cli_to_session, run_cli], ids=["detector", "shipped"])
def test_a_cli_harness_restores_the_source_language(monkeypatch, drive):
    # Leaked, `en` rerouted test_shipped_path's model-path test to turbo whenever it ran next.
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", "ja")
    with pytest.MonkeyPatch.context() as session:
        drive(session, "--source-lang", "en")
        assert live_stt.ASR_LANGUAGE == "en"
    assert live_stt.ASR_LANGUAGE == "ja"


def test_a_playback_session_restores_the_capture_property(monkeypatch, tmp_path):
    monkeypatch.setenv("PIPEWIRE_ALSA", "outer")  # registers this test's own undo
    monkeypatch.delenv("PIPEWIRE_ALSA")
    with pytest.MonkeyPatch.context() as session:
        _run_session(session, tmp_path, [HARDWARE, PIPEWIRE, DEFAULT], playback=True)
    assert "PIPEWIRE_ALSA" not in os.environ


def test_the_first_cli_run_leaves_the_logger_as_it_found_it(monkeypatch, capsys):
    logger = live_stt.logger
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    with pytest.MonkeyPatch.context() as session:
        keep_session_globals(session)
        session.setattr(sys, "argv", ["live-stt", "--help"])
        with pytest.raises(SystemExit):
            live_stt.main()
        assert logger.handlers and logger.propagate is False
    assert (logger.handlers, logger.propagate) == ([], True)
