"""U3 token: which language every decode is issued under, and when that can change.

The grading check for queue row 3 unit (3). Stub seeded by the lead: fill each body in
place, never rename a case and never delete one. Add a case with the next free name where
the contract needs more.

Contract, in the order the cases sit below:

- **Every decode names its language.** `Recognizer.generate()` takes the language for THIS
  decode and falls back to the module default when the caller names none;
  `Recognizer.decode_segments()` forwards it, being the callable `StreamingProcessor` holds.
  The rest of the `generate()` call is untouched: `task="transcribe"`,
  `repetition_penalty`, `hotwords` and `return_timestamps` keep today's values.
- **`held` is the SOURCE LANGUAGE at startup, not the literal `ja`.** One-way
  `--source-lang en` constructs no detector, so `token` stays `held` for the whole run --
  a literal would decode English under `<|ja|>` and silently break transcribe-only mode.
  Under `--two-way` the two are equal, `--source-lang` being a parse error there.
- **The open processor decodes under `token`**, which changes ONLY together with a
  `StreamingProcessor` rebuild. Before acceptance that is the held token; an accepted label
  equal to it changes nothing; an accepted label differing from it rebuilds, and the
  re-inserted prefix and every later buffer of that utterance decode under the NEW token.
  After acceptance the token is frozen for the rest of the utterance, and the next utterance
  opens on the last accepted label.
- **An utterance that never accepts decodes wholly under the held token** and still
  publishes (unit (2) owns its `<!>` mark).
- **`--two-way` is default OFF**, and with the flag absent every decode is issued exactly as
  today: `replay.py` and every existing lock call `worker()` unchanged, so the committed
  replay goldens must not move.

Weights-free by construction: stub the recognizer and record the language each decode was
issued under. Never load `models/`.

Import `live_stt` as the filled bodies need it; the skeleton leaves it out so an all-unfilled
file still passes `ruff check`.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import live_stt
from streaming import Segment, StreamingProcessor


class _Pipeline:
    instances: list[_Pipeline] = []

    def __init__(self, _model_dir: str, _device: str, **_kwargs: object) -> None:
        self.calls: list[dict[str, object]] = []
        self.instances.append(self)

    def generate(self, samples: object, **kwargs: object) -> object:
        self.calls.append({"samples": np.asarray(samples), **kwargs})
        chunk = types.SimpleNamespace(start_ts=0.0, end_ts=1.0, text="あ")
        return types.SimpleNamespace(texts=["あ"], chunks=[chunk])


class _StubVad:
    def __init__(self, script: Sequence[bool]) -> None:
        self.script = tuple(script)
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


@dataclass(frozen=True)
class _Decode:
    samples: np.ndarray
    language: str | None


class _RecordingRecognizer:
    def __init__(self, hypotheses: Sequence[str] = ("こんにちは",)) -> None:
        self.hypotheses = tuple(hypotheses)
        self.calls: list[_Decode] = []

    def decode_segments(
        self, samples: np.ndarray, language: str | None = None
    ) -> tuple[str, list[Segment]]:
        index = min(len(self.calls), len(self.hypotheses) - 1)
        text = self.hypotheses[index]
        self.calls.append(_Decode(samples.copy(), language))
        duration = len(samples) / live_stt.SAMPLE_RATE
        segments = [] if not text else [Segment(0.0, duration, text)]
        return text, segments


class _StubDetector:
    def __init__(self, outcomes: Sequence[str | None]) -> None:
        self.outcomes = tuple(outcomes)
        self.calls: list[np.ndarray] = []

    def decide(self, pcm: np.ndarray) -> str | None:
        index = len(self.calls)
        self.calls.append(pcm.copy())
        if index >= len(self.outcomes):
            raise AssertionError("detector was called after its scripted decisions ended")
        return self.outcomes[index]


class _RecordingProcessor(StreamingProcessor):
    def __init__(
        self,
        *,
        decode: Callable[[np.ndarray], tuple[str, list[Segment]]],
        buffer_trim_s: float,
        calls: list[_Decode],
    ) -> None:
        self.call_indices: list[int] = []
        self.insertions: list[np.ndarray] = []

        def capture(samples: np.ndarray) -> tuple[str, list[Segment]]:
            before = len(calls)
            result = decode(samples)
            self.call_indices.extend(range(before, len(calls)))
            return result

        super().__init__(decode=capture, buffer_trim_s=buffer_trim_s)

    def insert_audio(self, chunk: np.ndarray) -> None:
        self.insertions.append(chunk.copy())
        super().insert_audio(chunk)


@dataclass(frozen=True)
class _TokenRun:
    lines: list[tuple[str, int, str, bool]]
    rec: _RecordingRecognizer
    detector: _StubDetector | None
    processors: list[_RecordingProcessor]
    blocks: list[np.ndarray]


def _drive_tokens(
    monkeypatch: pytest.MonkeyPatch,
    script: Sequence[bool],
    *,
    detector: _StubDetector | None = None,
    language: str = "ja",
    window: int = live_stt.SAMPLE_RATE // 2,
) -> _TokenRun:
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    rec = _RecordingRecognizer()
    vad = _StubVad(script)
    lines: list[tuple[str, int, str, bool]] = []
    processors: list[_RecordingProcessor] = []
    blocks = [np.full(window, value + 1, dtype=np.float32) for value in range(len(script))]

    def make_processor(
        *,
        decode: Callable[[np.ndarray], tuple[str, list[Segment]]],
        buffer_trim_s: float,
    ) -> _RecordingProcessor:
        processor = _RecordingProcessor(decode=decode, buffer_trim_s=buffer_trim_s, calls=rec.calls)
        processors.append(processor)
        return processor

    def capture_line(
        tag: str, seq: int, text: str, _output_file: object, *, held: bool = False
    ) -> None:
        lines.append((tag, seq, text, held))

    monkeypatch.setattr(live_stt, "StreamingProcessor", make_processor)
    monkeypatch.setattr(live_stt, "emit_line", capture_line)
    queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
    for block in blocks:
        queue.put_nowait(block)
    queue.put_nowait(None)
    asyncio.run(
        live_stt.worker(
            rec,
            vad,
            window,
            queue,
            live_stt.State(),
            None,
            detector=detector,
        )
    )
    return _TokenRun(lines, rec, detector, processors, blocks)


def _languages(run: _TokenRun) -> list[str | None]:
    return [call.language for call in run.rec.calls]


def _processor_calls(run: _TokenRun, index: int) -> list[_Decode]:
    return [run.rec.calls[i] for i in run.processors[index].call_indices]


def _whisper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, language: str = "ja"
) -> tuple[live_stt.WhisperEngine, _Pipeline]:
    _Pipeline.instances.clear()
    monkeypatch.setitem(
        sys.modules, "openvino_genai", types.SimpleNamespace(WhisperPipeline=_Pipeline)
    )
    monkeypatch.setattr(live_stt, "OPENVINO_CACHE_DIR", tmp_path / "openvino-cache")
    monkeypatch.setattr(live_stt, "ASR_LANGUAGE", language)
    engine = live_stt.WhisperEngine(tmp_path / "model", "CPU")
    (pipeline,) = _Pipeline.instances
    return engine, pipeline


def _call_with_language(
    call: Callable[..., Any], samples: np.ndarray, language: str, **kwargs: object
) -> Any:
    assert "language" in inspect.signature(call).parameters, (
        f"{call.__name__} must accept a per-decode language"
    )
    return call(samples, language=language, **kwargs)


def _run_one_way(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, language: str
) -> list[dict[str, object]]:
    engine, pipeline = _whisper(monkeypatch, tmp_path, language=language)
    script = [True] * 6 + [False]
    window = live_stt.SAMPLE_RATE // 2
    queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
    for value in range(len(script)):
        queue.put_nowait(np.full(window, value + 1, dtype=np.float32))
    queue.put_nowait(None)
    asyncio.run(
        live_stt.worker(
            engine,
            _StubVad(script),
            window,
            queue,
            live_stt.State(),
            None,
        )
    )
    return pipeline.calls


def test_one_way_decodes_every_buffer_under_the_japanese_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _run_one_way(monkeypatch, tmp_path, "ja")

    assert len(calls) > 1
    assert [call["language"] for call in calls] == ["<|ja|>"] * len(calls)


def test_source_lang_en_decodes_every_buffer_under_the_english_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _run_one_way(monkeypatch, tmp_path, "en")

    assert len(calls) > 1
    assert [call["language"] for call in calls] == ["<|en|>"] * len(calls)


def test_the_generate_call_keeps_task_penalty_and_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, pipeline = _whisper(monkeypatch, tmp_path)
    engine.set_hotwords("東京、タワー")

    _call_with_language(
        engine.generate,
        np.zeros(live_stt.SAMPLE_RATE, dtype=np.float32),
        "en",
        timestamps=True,
    )

    (call,) = pipeline.calls
    assert call["language"] == "<|en|>"
    assert call["task"] == "transcribe"
    assert call["repetition_penalty"] == live_stt.ASR_REPETITION_PENALTY
    assert call["hotwords"] == "東京、タワー"
    assert call["return_timestamps"] is True


def test_decode_segments_forwards_the_language_it_was_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, pipeline = _whisper(monkeypatch, tmp_path)

    text, segments = _call_with_language(
        engine.decode_segments,
        np.zeros(live_stt.SAMPLE_RATE, dtype=np.float32),
        "en",
    )

    assert text == "あ"
    assert [(segment.start_s, segment.end_s, segment.text) for segment in segments] == [
        (0.0, 1.0, "あ")
    ]
    assert pipeline.calls[0]["language"] == "<|en|>"


def test_decode_segments_without_a_language_keeps_the_module_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, pipeline = _whisper(monkeypatch, tmp_path, language="en")

    engine.decode_segments(np.zeros(live_stt.SAMPLE_RATE, dtype=np.float32))

    assert pipeline.calls[0]["language"] == "<|en|>"


def test_two_way_decodes_the_pre_acceptance_buffers_under_the_held_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector([None, "en"])

    run = _drive_tokens(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    languages = _languages(run)
    assert len(detector.calls) == 2
    assert languages[:2] == ["ja", "ja"]
    assert languages[2:] and set(languages[2:]) == {"en"}


def test_the_held_token_at_startup_is_the_source_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _drive_tokens(
        monkeypatch,
        [True, True, True, True, False],
        language="en",
    )

    languages = _languages(run)
    assert len(languages) > 1
    assert languages == ["en"] * len(languages)


def test_a_label_matching_the_token_leaves_the_decode_token_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _drive_tokens(
        monkeypatch,
        [True, True, True, True, False],
        detector=_StubDetector(["ja"]),
    )

    assert len(run.processors) == 1
    assert _languages(run) == ["ja"] * len(run.rec.calls)


def test_a_label_differing_from_the_token_decodes_the_rest_under_the_new_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _drive_tokens(
        monkeypatch,
        [True, True, True, True, False],
        detector=_StubDetector(["en"]),
    )

    languages = _languages(run)
    assert len(run.processors) == 2
    assert languages[0] == "ja"
    assert languages[1:] and set(languages[1:]) == {"en"}


def test_the_re_inserted_prefix_is_decoded_under_the_accepted_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _drive_tokens(
        monkeypatch,
        [True, True, True, True, False],
        detector=_StubDetector(["en"]),
    )

    assert len(run.processors) == 2
    accepted_calls = _processor_calls(run, 1)
    assert accepted_calls and accepted_calls[0].language == "en"
    np.testing.assert_array_equal(
        accepted_calls[0].samples,
        np.concatenate(run.blocks[:4]),
    )


def test_the_token_is_frozen_for_the_rest_of_the_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector(["en", "ja"])

    run = _drive_tokens(
        monkeypatch,
        [True] * 8 + [False],
        detector=detector,
    )

    languages = _languages(run)
    assert "en" in languages
    first_english = languages.index("en")
    assert len(detector.calls) == 1
    assert languages[:first_english] == ["ja"] * first_english
    assert languages[first_english:] == ["en"] * (len(languages) - first_english)


def test_the_next_utterance_opens_on_the_last_accepted_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector(["en"])

    run = _drive_tokens(
        monkeypatch,
        [True, True, True, True, False, True, True, False],
        detector=detector,
    )

    assert len(detector.calls) == 1
    assert len(run.processors) == 3
    second_utterance = _processor_calls(run, 2)
    assert second_utterance
    assert [call.language for call in second_utterance] == ["en"] * len(second_utterance)


def test_an_utterance_that_never_accepts_decodes_wholly_under_the_held_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = _StubDetector([None, None])

    run = _drive_tokens(
        monkeypatch,
        [True, True, True, True, False],
        detector=detector,
    )

    languages = _languages(run)
    assert len(detector.calls) == 2
    assert languages and languages == ["ja"] * len(languages)
    assert run.lines == [("SRC", 1, "こんにちは", True)]


def test_the_flag_off_path_decodes_exactly_as_today(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _drive_tokens(
        monkeypatch,
        [True, True, True, True, False],
    )

    assert [call.samples.size for call in run.rec.calls] == [
        live_stt.SAMPLE_RATE,
        2 * live_stt.SAMPLE_RATE,
        int(2.5 * live_stt.SAMPLE_RATE),
    ]
    assert _languages(run) == ["ja", "ja", "ja"]
    assert run.lines == [("SRC", 1, "こんにちは", False)]


def test_no_decode_is_ever_issued_without_a_language_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = [True, True, True, True, False]
    runs = [
        _drive_tokens(monkeypatch, script),
        _drive_tokens(monkeypatch, script, language="en"),
        _drive_tokens(monkeypatch, script, detector=_StubDetector([None, None])),
        _drive_tokens(monkeypatch, script, detector=_StubDetector(["en"])),
    ]

    languages = [language for run in runs for language in _languages(run)]
    assert languages
    assert None not in languages
    assert set(languages) == {"ja", "en"}


def test_a_token_change_never_reaches_an_already_open_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _drive_tokens(
        monkeypatch,
        [True, True, True, True, False],
        detector=_StubDetector(["en"]),
    )

    assert len(run.processors) == 2
    old_processor = run.processors[0]
    old_processor.decode(np.zeros(live_stt.SAMPLE_RATE, dtype=np.float32))

    assert run.rec.calls[-1].language == "ja"
