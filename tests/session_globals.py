"""Undo what the real CLI and session write process-wide, for tests that drive them.

`_run_cli` rebinds `live_stt.ASR_LANGUAGE`, `run_session` sets `PIPEWIRE_ALSA` under
`--playback`, and `main()` configures the `live_stt` logger once per process. Unrestored,
each leaked into whichever test ran next: an `en` default rerouted a later model-path test,
and the first logger configuration turned propagation off for every later caplog reader.
"""

import pytest

import live_stt


def keep_session_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", live_stt.ASR_LANGUAGE)
    # setenv first registers the undo that also removes a value the session sets.
    monkeypatch.setenv("PIPEWIRE_ALSA", "")
    monkeypatch.delenv("PIPEWIRE_ALSA")
    logger = live_stt.logger
    monkeypatch.setattr(logger, "handlers", list(logger.handlers))
    monkeypatch.setattr(logger, "level", logger.level)
    monkeypatch.setattr(logger, "propagate", logger.propagate)
