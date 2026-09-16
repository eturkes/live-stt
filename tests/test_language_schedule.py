"""U2 schedule: when the detector is asked, what freezes, and what a held label publishes.

The grading check for queue row 3 unit (2). Stub seeded by the lead: fill each body in
place, never rename a case and never delete one. Add a case with the next free name where
the contract needs more.

Contract, in the order the cases sit below:

- **Never before `LID_MIN_SECONDS`=2.0 s of voiced buffer**, then retried on every later VAC
  update until one accepts, then FROZEN for the rest of the utterance and reset at VAD
  close. `label is None` is the whole "still deciding" state, and at most one
  `detector.decide()` call happens per update.
- **The LID input is the untrimmed prefix from utterance start**, pre-pad included --
  `ring.slice(utterance_start, consumed)`, the same construction the census scored. Never
  `processor.audio`, which `streaming.py` trims past `VAC_TRIM_S` so it stops being a prefix.
- **`held`** is the session-level fallback label, `ja` at startup, replaced by each accepted
  label. An utterance that never accepts decodes and translates under it.
- **Before acceptance the decode renders WHOLLY DIM and the published line commits nothing**:
  the status line's normal-intensity run is empty and the provisional run carries everything.
  After acceptance, and whenever the flag is off, the split is today's.
- **A label differing from the token that produced the text resets the `StreamingProcessor`**:
  a fresh processor, the utterance audio re-inserted from its start, the accumulated text
  cleared. Unit (3) owns handing the token to `generate()`; nothing here touches
  `ASR_LANGUAGE`.
- **No caption is ever withheld.** 409 of 1,926 utterances never accept, so an unaccepted one
  publishes under `held` and its SOURCE line carries the mark `SRC n <!>: text` -- space,
  `<!>`, colon. The `TGT` line is never marked. `session_report.py` parses that line and
  counts held utterances; today's `_EVENT` pattern drops it whole.
- **`--two-way` is default OFF**, and with the flag absent every path above is today's code.

Weights-free by construction: stub the detector, never load `models/lid/d2-ecapa/`.

Import `live_stt` and `session_report` as the filled bodies need them; the initial
skeleton leaves them out so its placeholder-only form still passes `ruff check`.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import live_stt
import session_report
from streaming import Segment, StreamingProcessor


class _StubVad:
    def __init__(self, script: Sequence[bool]):
        self.script = list(script)
        self.calls = 0
        self.queued = 0

    def accept_waveform(self, _block: np.ndarray) -> None:
        was_speech = self.is_speech_detected()
        self.calls += 1
        if was_speech and not self.is_speech_detected():
            self.queued += 1

    def is_speech_detected(self) -> bool:
        index = min(self.calls, len(self.script)) - 1
        return self.script[index] if index >= 0 else False

    def empty(self) -> bool:
        return self.queued == 0

    def pop(self) -> None:
        assert self.queued
        self.queued -= 1


class _ScriptedRecognizer:
    def __init__(self, hypotheses: Sequence[str] = ("こんにちは",)):
        self.hypotheses = tuple(hypotheses)
        self.buffers: list[np.ndarray] = []

    def decode_segments(
        self, samples: np.ndarray, language: str | None = None
    ) -> tuple[str, list[Segment]]:
        index = min(len(self.buffers), len(self.hypotheses) - 1)
        text = self.hypotheses[index]
        self.buffers.append(samples.copy())
        duration = len(samples) / live_stt.SAMPLE_RATE
        segments = [] if not text else [Segment(0.0, duration, text)]
        return text, segments


class _StubDetector:
    def __init__(self, outcomes: Sequence[str | None]):
        self.outcomes = tuple(outcomes)
        self.calls: list[np.ndarray] = []

    def decide(self, pcm: np.ndarray) -> str | None:
        index = len(self.calls)
        self.calls.append(pcm.copy())
        if index >= len(self.outcomes):
            raise AssertionError("detector was called after its scripted decisions ended")
        return self.outcomes[index]


class _RecordingProcessor(StreamingProcessor):
    def __init__(self, *, decode: Any, buffer_trim_s: float):
        super().__init__(decode=decode, buffer_trim_s=buffer_trim_s)
        self.insertions: list[np.ndarray] = []

    def insert_audio(self, chunk: np.ndarray) -> None:
        self.insertions.append(chunk.copy())
        super().insert_audio(chunk)


@dataclass(frozen=True)
class _Update:
    buffer_s: float
    partial: str
    provisional: str
    commit: str
    final: bool
    detector_calls: int
    published: int


@dataclass(frozen=True)
class _VacRun:
    lines: list[tuple[str, int, str, bool]]
    updates: list[_Update]
    state: live_stt.State
    rec: Any
    detector: _StubDetector | None
    processors: list[_RecordingProcessor]
    blocks: list[np.ndarray]


def _drive_vac(
    monkeypatch: pytest.MonkeyPatch,
    script: Sequence[bool],
    *,
    window: int = live_stt.SAMPLE_RATE // 2,
    detector: _StubDetector | None = None,
    rec: Any | None = None,
    pre_pad_s: float = 0.0,
    through_worker: bool = False,
) -> _VacRun:
    monkeypatch.setattr(live_stt, "VAD_PRE_PAD_S", pre_pad_s)
    recognizer = rec or _ScriptedRecognizer()
    vad = _StubVad(script)
    state = live_stt.State()
    lines: list[tuple[str, int, str, bool]] = []
    updates: list[_Update] = []
    processors: list[_RecordingProcessor] = []
    blocks = [np.full(window, i + 1, dtype=np.float32) for i in range(len(script))]

    def make_processor(*, decode: Any, buffer_trim_s: float) -> _RecordingProcessor:
        processor = _RecordingProcessor(decode=decode, buffer_trim_s=buffer_trim_s)
        processors.append(processor)
        return processor

    def capture_line(
        tag: str, seq: int, text: str, _output_file: object, *, held: bool = False
    ) -> None:
        lines.append((tag, seq, text, held))

    def capture_update(
        buffer_s: float,
        _buffer_end_s: float,
        _commit_audio_s: float | None,
        commit: str,
        final: bool,
        _decode_s: float,
    ) -> None:
        updates.append(
            _Update(
                buffer_s=buffer_s,
                partial=state.partial,
                provisional=state.provisional,
                commit=commit,
                final=final,
                detector_calls=0 if detector is None else len(detector.calls),
                published=len(lines),
            )
        )

    monkeypatch.setattr(live_stt, "StreamingProcessor", make_processor)
    monkeypatch.setattr(live_stt, "emit_line", capture_line)
    queue: asyncio.Queue = asyncio.Queue()
    for block in blocks:
        queue.put_nowait(block)
    queue.put_nowait(None)

    async def scenario() -> None:
        args = (recognizer, vad, window, queue, state, None)
        if through_worker:
            await live_stt.worker(
                *args,
                on_update=capture_update,
                detector=detector,
            )
        else:
            await live_stt._vac_segments(
                *args,
                on_update=capture_update,
                detector=detector,
            )

    asyncio.run(scenario())
    return _VacRun(lines, updates, state, recognizer, detector, processors, blocks)


def test_no_lid_call_before_two_seconds_of_voiced_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector([None])

    run = _drive_vac(monkeypatch, [True, True, False], detector=detector)

    assert detector.calls == [], "LID ran on an utterance whose longest prefix was only 1.5 s"
    assert all(update.buffer_s < live_stt.LID_MIN_SECONDS for update in run.updates)


def test_the_first_lid_call_happens_once_the_buffer_reaches_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector(["ja"])

    _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
        through_worker=True,
    )

    assert [call.size for call in detector.calls] == [
        round(live_stt.LID_MIN_SECONDS * live_stt.SAMPLE_RATE)
    ], "the shipped worker did not make its first LID decision on the 2.0 s prefix"


def test_an_abstaining_update_is_retried_on_the_next_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector([None, None])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    assert [call.size for call in detector.calls] == [
        2 * live_stt.SAMPLE_RATE,
        int(2.5 * live_stt.SAMPLE_RATE),
    ], "an abstention was not retried on the VAD-final update"
    call_counts = [update.detector_calls for update in run.updates]
    previous_counts = [0, *call_counts[:-1]]
    assert all(
        after - before <= 1 for before, after in zip(previous_counts, call_counts, strict=True)
    )


def test_the_first_accepted_label_freezes_for_the_rest_of_the_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector(["en", "ja", "ja"])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, True, True, False],
        detector=detector,
    )

    assert len(run.processors) == 2, "a later prefix replaced the first accepted label"
    assert run.lines and run.lines[0][3] is False


def test_the_frozen_label_is_never_rescored(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _StubDetector(["en", "ja", "ja"])

    _drive_vac(
        monkeypatch,
        [True, True, True, True, True, True, False],
        detector=detector,
    )

    assert len(detector.calls) == 1, "the detector was called again after accepting a label"


def test_the_label_resets_at_vad_close(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _StubDetector(["ja", "en"])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False, True, True, True, True, False],
        detector=detector,
    )

    assert len(detector.calls) == 2, "the first utterance's accepted label leaked past VAD close"
    assert [call.size for call in detector.calls] == [2 * live_stt.SAMPLE_RATE] * 2
    assert len(run.lines) == 2


def test_the_held_label_starts_japanese(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _StubDetector(["ja"])

    run = _drive_vac(
        monkeypatch,
        [True, True, False, True, True, True, True, False],
        detector=detector,
    )

    assert len(detector.calls) == 1, "the second utterance never reached its LID gate"
    assert len(run.processors) == 2, (
        "accepting ja reset a processor that should start under held=ja"
    )


def test_the_held_label_becomes_the_last_accepted_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector(["en", "en"])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False, True, True, True, True, False],
        detector=detector,
    )

    assert len(detector.calls) == 2, "both utterances did not reach their LID gates"
    assert len(run.processors) == 3, "the second utterance did not start under the accepted en hold"


def test_the_detector_scores_the_untrimmed_prefix_from_utterance_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector(["ja"])
    script = [False] * 5 + [True] * 16 + [False]

    run = _drive_vac(
        monkeypatch,
        script,
        window=live_stt.SAMPLE_RATE // 10,
        detector=detector,
        pre_pad_s=0.4,
    )

    assert len(detector.calls) == 1, "the 2.0 s utterance prefix was not scored exactly once"
    # Speech opens at block 5. Its 0.4 s pre-pad begins at block 1, and the
    # second cadence update ends after block 20: exactly ring.slice(start, consumed).
    expected = np.concatenate(run.blocks[1:21])
    np.testing.assert_array_equal(detector.calls[0], expected)


def test_the_detector_is_never_handed_the_trimmed_processor_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrimmingRecognizer:
        def __init__(self) -> None:
            self.buffers: list[np.ndarray] = []

        def decode_segments(
            self, samples: np.ndarray, language: str | None = None
        ) -> tuple[str, list[Segment]]:
            texts = ("ABold", "ABnew", "newer", "newer")
            index = min(len(self.buffers), len(texts) - 1)
            text = texts[index]
            self.buffers.append(samples.copy())
            duration = len(samples) / live_stt.SAMPLE_RATE
            if index < 2:
                return text, [Segment(0.0, 0.5, "AB"), Segment(0.5, duration, text[2:])]
            return text, [Segment(0.0, duration, text)]

    monkeypatch.setattr(live_stt, "VAC_TRIM_S", 0.5)
    detector = _StubDetector([None, None, None])
    rec = TrimmingRecognizer()

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, True, True, False],
        detector=detector,
        rec=rec,
    )

    assert run.processors[0].trims >= 2, "the positive control never trimmed processor.audio"
    assert len(detector.calls) >= 2, "the detector never received a prefix after processor trim"
    np.testing.assert_array_equal(detector.calls[1], np.concatenate(run.blocks[:6]))
    assert rec.buffers[2].size < detector.calls[1].size


def test_pre_acceptance_text_renders_wholly_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _StubDetector([None, None])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    visible = [(update.partial, update.provisional) for update in run.updates if not update.final]
    assert visible == [("", "こんにちは"), ("", "こんにちは")], (
        "pre-acceptance text escaped the dim provisional run",
        visible,
    )


def test_pre_acceptance_commits_nothing_to_the_published_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector([None, None])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    committed = [update for update in run.updates if not update.final and update.commit]
    assert committed, "the positive control never reached a LocalAgreement commit"
    assert committed[0].partial == "", "a pre-acceptance commit reached the normal-intensity run"
    assert committed[0].provisional == "こんにちは"


def test_an_accepted_label_matching_the_token_keeps_the_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector(["ja"])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    assert len(detector.calls) == 1, "the matching-label path never reached its LID gate"
    assert len(run.processors) == 1, "accepting the processor's existing ja token reset it"


def test_an_accepted_label_differing_from_the_token_resets_the_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector(["en"])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    assert len(detector.calls) == 1, "the differing-label path never reached its LID gate"
    assert len(run.processors) == 2, "accepting en left text decoded under the held ja token"


def test_a_reset_re_inserts_the_utterance_audio_from_its_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector(["en"])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    assert len(run.processors) == 2, "the differing accepted label did not reset the processor"
    assert run.processors[1].insertions, "the fresh processor received no audio"
    expected = np.concatenate(run.blocks[:4])
    np.testing.assert_array_equal(run.processors[1].insertions[0], expected)


def test_an_utterance_that_never_accepts_still_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector([None, None])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    assert len(detector.calls) == 2, "the abstaining path was never exercised through VAD-final"
    assert len(run.lines) == 1, "an unaccepted utterance was withheld"
    assert run.lines[0][:3] == ("SRC", 1, "こんにちは")


def test_an_utterance_that_never_accepts_is_marked_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector([None, None])

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    assert run.lines == [("SRC", 1, "こんにちは", True)], "the held source line was unmarked"


def test_the_target_line_is_never_marked(monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = io.StringIO()
    monkeypatch.setattr(live_stt, "write_stdout", lambda _text: None)

    live_stt.emit_line("SRC", 4, "こんにちは", transcript, held=True)
    # held=True on a target line is what the guard exists for: production never
    # passes it, so without this argument the case cannot see `tag == "SRC"` go.
    live_stt.emit_line("TGT", 4, "Hello.", transcript, held=True)

    bodies = [line.split("] ", 1)[1] for line in transcript.getvalue().splitlines()]
    assert bodies[-1] == "TGT 4: Hello."
    assert "<!>" not in bodies[-1]


def test_the_mark_sits_between_the_number_and_the_colon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    displayed: list[str] = []
    monkeypatch.setattr(live_stt, "write_stdout", displayed.append)

    live_stt.emit_line("SRC", 12, "保留", None, held=True)

    assert displayed == ["  SRC 12 <!>: 保留\n"], "the held mark is not ` n <!>:`"


def test_the_transcript_records_the_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = io.StringIO()
    monkeypatch.setattr(live_stt, "write_stdout", lambda _text: None)

    live_stt.emit_line("SRC", 12, "保留", transcript, held=True)

    saved = transcript.getvalue().split("] ", 1)[1]
    assert saved == "SRC 12 <!>: 保留\n", "the durable line omitted the held mark"


def test_the_session_report_parses_a_marked_source_line(tmp_path: Path) -> None:
    path = tmp_path / "held.txt"
    path.write_text(
        "[2026-09-16T10:00:00+09:00] SRC 7 <!>: 保留された発話\n",
        encoding="utf-8",
    )

    session = session_report.read_session(str(path))

    assert sorted(session.src) == [7], "session_report dropped a marked source line"
    assert session.src[7].text == "保留された発話"


def test_the_session_report_counts_held_utterances(tmp_path: Path) -> None:
    path = tmp_path / "held-count.txt"
    path.write_text(
        "".join(
            [
                "[2026-09-16T10:00:00+09:00] SRC 1 <!>: 保留一\n",
                "[2026-09-16T10:00:01+09:00] SRC 2: 確定\n",
                "[2026-09-16T10:00:02+09:00] TGT 2: Settled.\n",
            ]
        ),
        encoding="utf-8",
    )
    session = session_report.read_session(str(path))

    report = session_report.build([session], [])

    assert report["sessions"][0].get("held") == 1, "the per-session held count is absent or wrong"
    assert report["totals"].get("held") == 1, "the total held count is absent or wrong"


def test_the_flag_off_path_calls_no_detector_and_renders_as_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_decide(_self: object, _pcm: np.ndarray) -> str | None:
        pytest.fail("flag-off path called LanguageDetector.decide")

    monkeypatch.setattr(live_stt.LanguageDetector, "decide", unexpected_decide)

    run = _drive_vac(
        monkeypatch,
        [True, True, True, True, False],
        detector=None,
        through_worker=True,
    )

    visible = [(update.partial, update.provisional) for update in run.updates if not update.final]
    assert visible == [("", "こんにちは"), ("こんにちは", "")]
    assert run.lines == [("SRC", 1, "こんにちは", False)]
