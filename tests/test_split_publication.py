"""Queue: Publish long utterances at settled segments — worker contract a/b/c."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import live_stt
from streaming import Segment, StreamingProcessor

WINDOW = live_stt.SAMPLE_RATE // 2
CLEAN = tuple("あいうえおかきく")
BAD = "I think you said the HDMI is broken, right?"
GOOD = "今日は会議で新しい計画について話しました。"


class _Vad:
    def __init__(self, script: Sequence[bool]):
        self.script = tuple(script)
        self.calls = 0
        self.queued = 0

    def accept_waveform(self, _samples: np.ndarray) -> None:
        was_speech = self.is_speech_detected()
        self.calls += 1
        if was_speech and not self.is_speech_detected():
            self.queued += 1

    def is_speech_detected(self) -> bool:
        return self.calls > 0 and self.script[min(self.calls, len(self.script)) - 1]

    def empty(self) -> bool:
        return self.queued == 0

    def pop(self) -> None:
        assert self.queued, "VAD popped a segment that did not close"
        self.queued -= 1


class _Recognizer:
    """PCM labels bind each span to retained audio; cuts cannot replay old text."""

    def __init__(self, labels: Sequence[str], *, reverse: Sequence[str] | None = None):
        self.labels = {i + 1: text for i, text in enumerate(labels)}
        self.reverse = None if reverse is None else dict(enumerate(reverse, 1))
        self.languages: list[str | None] = []
        self.hotwords = ""

    def set_hotwords(self, terms: str) -> None:
        self.hotwords = terms

    def decode_segments(
        self, samples: np.ndarray, language: str | None = None
    ) -> tuple[str, list[Segment]]:
        self.languages.append(language)
        mapping = self.reverse if language == "en" and self.reverse is not None else self.labels
        changes = np.flatnonzero(np.diff(samples)) + 1
        edges = [0, *changes.tolist(), len(samples)]
        spans = []
        for start, stop in zip(edges, edges[1:], strict=False):
            text = mapping.get(int(samples[start]), "")
            if text:
                spans.append(
                    Segment(start / live_stt.SAMPLE_RATE, stop / live_stt.SAMPLE_RATE, text)
                )
        return "".join(span.text for span in spans), spans


class _ForcedRecognizer:
    def __init__(self) -> None:
        self.calls = 0

    def decode_segments(
        self, samples: np.ndarray, language: str | None = None
    ) -> tuple[str, list[Segment]]:
        self.calls += 1
        # LA2 alone commits; the stable prefix leaves with its audio on a forced cut.
        prefix = "確定" if np.any(samples == 1) else ""
        return prefix + chr(ord("あ") + self.calls), []


class _Detector:
    def __init__(self, outcomes: Sequence[str | None], vad: _Vad):
        self.outcomes = tuple(outcomes)
        self.vad = vad
        self.calls: list[int] = []
        self.accepted_at: int | None = None

    def decide(self, _pcm: np.ndarray) -> str | None:
        index = len(self.calls)
        assert index < len(self.outcomes), "detector ran beyond the scripted acceptance schedule"
        self.calls.append(self.vad.calls)
        label = self.outcomes[index]
        if label is not None:
            self.accepted_at = self.vad.calls
        return label


@dataclass(frozen=True)
class _Line:
    seq: int
    text: str
    held: bool
    window: int
    accepted_at: int | None


@dataclass
class _Run:
    state: live_stt.State
    vad: _Vad
    rec: Any
    lines: list[_Line] = field(default_factory=list)
    submitted: list[tuple[int, str, str | None]] = field(default_factory=list)
    observed: list[tuple[str, int]] = field(default_factory=list)
    raw: list[str] = field(default_factory=list)
    commits: list[str] = field(default_factory=list)
    trimmed: list[str] = field(default_factory=list)
    forced_at: list[int] = field(default_factory=list)
    processors: list[StreamingProcessor] = field(default_factory=list)
    detector: _Detector | None = None


class _Translator:
    def __init__(self, run: _Run):
        self.run = run

    def submit(self, seq: int, text: str, source: str | None = None) -> None:
        assert self.run.lines[-1].seq == seq, "translator submission preceded SRC publication"
        self.run.submitted.append((seq, text, source))


class _Context:
    def __init__(self, run: _Run):
        self.run = run

    def asr_hotwords(self) -> tuple[str, frozenset[str]]:
        return "", frozenset()

    def observe_ja(self, text: str, prompted: frozenset[str] = frozenset()) -> None:
        self.run.observed.append((text, self.run.vad.calls))


def _drive(
    monkeypatch: pytest.MonkeyPatch,
    labels: Sequence[str],
    *,
    close: bool = True,
    trim_s: float = 1.0,
    script: Sequence[bool] | None = None,
    outcomes: Sequence[str | None] | None = None,
    reverse: Sequence[str] | None = None,
    recognizer: Any | None = None,
    tail: str = "",
    translate: bool = True,
) -> _Run:
    speech = list(script) if script is not None else [True] * len(labels)
    blocks = [np.full(WINDOW, i + 1, dtype=np.float32) for i in range(len(labels))]
    assert len(speech) == len(blocks)
    if close:
        speech.append(False)
        blocks.append(np.zeros(WINDOW, dtype=np.float32))
    if tail:
        assert not close
        blocks.append(np.full(WINDOW // 2, len(labels) + 1, dtype=np.float32))
    vad = _Vad(speech)
    rec: Any = (
        recognizer if recognizer is not None else _Recognizer([*labels, tail], reverse=reverse)
    )
    run = _Run(live_stt.State(), vad, rec)
    if outcomes is not None:
        run.detector = _Detector(outcomes, vad)
    trim_text = ""
    trim_at = 0

    def capture_line(tag: str, seq: int, text: str, _file: object, *, held: bool = False) -> None:
        assert tag == "SRC"
        accepted = None if run.detector is None else run.detector.accepted_at
        run.lines.append(_Line(seq, text, held, vad.calls, accepted))

    def capture_segment(_start: int, _n: int, _seg_len: int, _decode_s: float, text: str) -> None:
        run.raw.append(text)

    def capture_update(
        _buffer_s: float,
        _buffer_end_s: float,
        _commit_audio_s: float | None,
        text: str,
        _final: bool,
        _decode_s: float,
    ) -> None:
        nonlocal trim_text, trim_at
        run.commits.append(text)
        trim_text += text
        cut = len(trim_text) - len(run.processors[-1].emitted)
        if not _final and cut > trim_at:
            run.trimmed.append(trim_text[trim_at:cut])
            trim_at = cut
        forced = sum(processor.forced_trims for processor in run.processors)
        if forced > len(run.forced_at):
            run.forced_at.append(vad.calls)

    def make_processor(*, decode: Any, buffer_trim_s: float) -> StreamingProcessor:
        nonlocal trim_text, trim_at
        trim_text, trim_at = "", 0
        processor = StreamingProcessor(decode=decode, buffer_trim_s=buffer_trim_s)
        run.processors.append(processor)
        return processor

    monkeypatch.setattr(live_stt, "VAC_CHUNK_S", 0.5)
    monkeypatch.setattr(live_stt, "VAC_TRIM_S", trim_s)
    monkeypatch.setattr(live_stt, "VAD_PRE_PAD_S", 0.0)
    monkeypatch.setattr(live_stt, "StreamingProcessor", make_processor)
    monkeypatch.setattr(live_stt, "emit_line", capture_line)
    queue: asyncio.Queue = asyncio.Queue()
    for block in blocks:
        queue.put_nowait(block)
    queue.put_nowait(None)
    translator: Any = _Translator(run) if translate else None
    context: Any = _Context(run)
    detector: Any = run.detector
    asyncio.run(
        live_stt.worker(
            rec,
            vad,
            WINDOW,
            queue,
            run.state,
            None,
            translator=translator,
            context=context,
            on_segment=capture_segment,
            on_update=capture_update,
            detector=detector,
        )
    )
    assert not run.state.stopping, "real worker stopped on a harness/implementation exception"
    return run


def _texts(run: _Run) -> list[str]:
    return [line.text for line in run.lines]


def _trimmed(run: _Run) -> None:
    assert sum(processor.trims for processor in run.processors) > 0, "trim control did not fire"


@pytest.mark.parametrize("translate", [True, False], ids=["translator", "source-only"])
def test_trim_publishes_before_speech_end_and_submits_the_same_piece(
    monkeypatch: pytest.MonkeyPatch, translate: bool
) -> None:
    run = _drive(monkeypatch, CLEAN, translate=translate)

    _trimmed(run)
    assert run.lines[0].window == 3, "the first trimmed piece waited past its non-final VAC update"
    assert run.lines[0].text == "あい"
    assert run.lines[-1].window == len(CLEAN) + 1
    assert "".join(_texts(run)) == "".join(CLEAN)
    expected = [(line.seq, line.text) for line in run.lines] if translate else []
    assert [(seq, text) for seq, text, _ in run.submitted] == expected


def test_generated_raw_pieces_reconstruct_the_untrimmed_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = random.Random(3210)  # noqa: S311 — repeatable generated inputs
    for case in range(24):
        labels = tuple(
            rng.choice(tuple("あいうえおかきくけこ")) for _ in range(rng.randrange(8, 29))
        )
        trim_s = rng.choice((0.5, 1.0, 1.5, 2.5))
        close = bool(case % 2)
        control = _drive(monkeypatch, labels, trim_s=60.0, close=close)
        split = _drive(monkeypatch, labels, trim_s=trim_s, close=close)

        expected = "".join(labels)
        _trimmed(split)
        assert control.raw == split.raw == [expected], (case, trim_s, close)
        assert "".join(split.commits) == "".join(control.commits) == expected
        assert "".join(_texts(split)) == expected


def test_on_segment_keeps_one_whole_raw_observation_per_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _drive(monkeypatch, CLEAN)

    _trimmed(run)
    assert run.raw == ["".join(CLEAN)]
    assert "".join(run.commits) == run.raw[0]


def test_numbers_are_gapless_once_only_across_pieces_and_utterances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [*CLEAN[:4], "", "", *CLEAN[4:]]
    run = _drive(monkeypatch, labels, script=[True] * 4 + [False] * 2 + [True] * 4)

    _trimmed(run)
    assert len(run.lines) >= 4, "two trimmed utterances did not each publish a piece and remainder"
    assert [line.seq for line in run.lines] == list(range(1, len(run.lines) + 1))
    assert [(seq, text) for seq, text, _ in run.submitted] == [
        (line.seq, line.text) for line in run.lines
    ]
    assert run.raw == ["".join(CLEAN[:4]), "".join(CLEAN[4:])]
    assert "".join(_texts(run)) == "".join(CLEAN)


def test_forced_trim_releases_only_committed_text(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _drive(monkeypatch, [""] * 61, trim_s=0.5, recognizer=_ForcedRecognizer())

    assert sum(processor.forced_trims for processor in run.processors) > 0
    assert run.trimmed == ["確定"]
    early = [line for line in run.lines if line.window <= 61]
    assert [line.text for line in early] == ["確定"], (
        "forced cut published a provisional tail or waited"
    )
    assert early[0].window == run.forced_at[0]
    assert "".join(_texts(run)) == run.raw[0]
    assert run.lines[-1].window == 62


def test_punctuation_only_piece_joins_the_next_word_piece(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = ("、", "", "次", "の", "話", "です")
    run = _drive(monkeypatch, labels, trim_s=0.5)

    _trimmed(run)
    assert run.trimmed[:2] == ["、", "次"], "punctuation never formed its own deferred piece"
    assert run.lines[0].window == 4, "punctuation did not join the next trimmed word piece"
    assert run.lines[0].text == "、次"
    assert all(any(char.isalnum() or char == "_" for char in line.text) for line in run.lines)
    assert "".join(_texts(run)) == "".join(labels)


def test_a_screened_piece_burns_no_number_and_keeps_a_clean_neighbour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = (BAD, "", GOOD, "次の話です。", "終わります。", "また明日。")
    run = _drive(monkeypatch, labels, trim_s=0.5)

    assert run.trimmed[0] == BAD, "screening probe did not isolate the defective piece"

    assert live_stt.caption_defect(BAD)
    assert live_stt.caption_defect("".join(labels)) is None
    _trimmed(run)
    assert BAD not in "".join(_texts(run)), "the combined utterance bypassed per-piece screening"
    assert "".join(_texts(run)) == "".join(labels[2:])
    assert [line.seq for line in run.lines] == list(range(1, len(run.lines) + 1))
    assert run.state.dropped_captions == 1
    assert run.raw == ["".join(labels)]


def test_a_clean_piece_survives_a_screened_neighbour(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = ("短い話。", "続き。", BAD, "", "終わり。", "また明日。")
    run = _drive(monkeypatch, labels, trim_s=0.5)

    _trimmed(run)
    assert BAD in run.trimmed, "screening probe did not isolate the defective piece"
    assert _texts(run)[:2] == ["短い話。", "続き。"]
    assert [line.window for line in run.lines[:2]] == [2, 3]
    assert BAD not in "".join(_texts(run))
    assert "".join(_texts(run)) == "短い話。続き。終わり。また明日。"
    assert run.state.dropped_captions == 1


def test_learner_observes_once_at_close_over_published_pieces_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = (BAD, "", GOOD, "次の話です。", "終わります。", "また明日。")
    run = _drive(monkeypatch, labels, trim_s=0.5)

    assert run.trimmed[0] == BAD, "screening probe did not isolate the defective piece"

    _trimmed(run)
    assert run.observed == [("".join(labels[2:]), len(labels) + 1)], (
        "learner received screened text, one observation per piece, or pre-close evidence",
        run.observed,
    )
    assert run.observed[0][0] == "".join(_texts(run))


@pytest.mark.parametrize("tail", ["", "け"], ids=["whole-window", "sub-window-tail"])
def test_sentinel_flushes_the_remainder_once(monkeypatch: pytest.MonkeyPatch, tail: str) -> None:
    run = _drive(monkeypatch, CLEAN, close=False, tail=tail)

    _trimmed(run)
    expected = "".join(CLEAN) + tail
    assert run.raw == [expected]
    assert "".join(_texts(run)) == expected
    assert run.observed == [(expected, len(CLEAN) + bool(tail))]
    assert len(run.lines) > 1, (
        "sentinel published the whole utterance instead of the split remainder"
    )
    assert [(seq, text) for seq, text, _ in run.submitted] == [
        (line.seq, line.text) for line in run.lines
    ]


def test_two_way_withholds_every_piece_until_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _drive(monkeypatch, CLEAN, outcomes=[None, "ja"])

    _trimmed(run)
    assert run.detector is not None and run.detector.accepted_at == 5
    assert run.lines
    assert all(line.accepted_at is not None and line.window >= 5 for line in run.lines)
    assert not any(line.held for line in run.lines)


def test_two_way_releases_pre_acceptance_settled_text_at_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _drive(monkeypatch, CLEAN, outcomes=[None, "ja"])

    _trimmed(run)
    assert run.detector is not None and run.detector.accepted_at == 5
    assert run.lines[0].window == 5, (
        "accepted label left already-trimmed text waiting for speech end"
    )
    assert run.lines[0].text.startswith("あい")
    assert "".join(_texts(run)) == "".join(CLEAN)
    assert [(seq, text) for seq, text, _ in run.submitted] == [
        (line.seq, line.text) for line in run.lines
    ]
    assert {source for _, _, source in run.submitted} == {"ja"}


def test_a_token_rebuild_discards_unpublished_text(monkeypatch: pytest.MonkeyPatch) -> None:
    reverse = tuple("甲乙丙丁戊己庚辛壬")
    run = _drive(monkeypatch, CLEAN, outcomes=[None, "en"], reverse=reverse)

    assert len(run.processors) == 2, "different accepted token did not rebuild the processor"
    assert "ja" in run.rec.languages and "en" in run.rec.languages
    assert all(line.accepted_at is not None for line in run.lines)
    expected = "".join(reverse[: len(CLEAN)])
    assert run.raw == [expected], "raw utterance retained text decoded under the abandoned token"
    assert "".join(_texts(run)) == expected
    assert all(char not in "".join(_texts(run)) for char in CLEAN)
    assert {source for _, _, source in run.submitted} == {"en"}


@pytest.mark.parametrize("close", [True, False], ids=["vad-close", "sentinel"])
def test_never_accepted_utterance_publishes_whole_with_the_held_mark(
    monkeypatch: pytest.MonkeyPatch, close: bool
) -> None:
    run = _drive(monkeypatch, CLEAN, close=close, outcomes=[None] * 6)

    _trimmed(run)
    assert run.detector is not None and len(run.detector.calls) >= 2
    assert [(line.seq, line.text, line.held, line.window) for line in run.lines] == [
        (1, "".join(CLEAN), True, len(CLEAN) + close)
    ]
    assert run.raw == ["".join(CLEAN)]
    assert run.observed == [("".join(CLEAN), len(CLEAN) + close)]
    assert run.submitted == [(1, "".join(CLEAN), "ja")]


@pytest.mark.parametrize("speech", [False, True], ids=["silence", "empty-speech"])
def test_silence_and_empty_decodes_publish_nothing(
    monkeypatch: pytest.MonkeyPatch, speech: bool
) -> None:
    run = _drive(monkeypatch, [""] * 8, script=[speech] * 8)

    assert run.lines == run.submitted == run.observed == []
    assert run.raw == ([""] if speech else [])


def test_exact_trim_boundary_keeps_one_final_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    run = _drive(monkeypatch, CLEAN[:2], close=False)

    assert all(processor.trims == 0 for processor in run.processors)
    assert [(line.seq, line.text, line.window) for line in run.lines] == [(1, "あい", 2)]
    assert run.raw == ["あい"]
    assert run.observed == [("あい", 2)]


@pytest.mark.parametrize("prior", [False, True], ids=["first-utterance", "after-clean-utterance"])
def test_punctuation_only_utterance_keeps_one_raw_observation(
    monkeypatch: pytest.MonkeyPatch, prior: bool
) -> None:
    punctuation = ("、", "。", "！", "？", "、", "。")
    labels = ("あ", "い", "", "", *punctuation) if prior else punctuation
    script = [True] * 2 + [False] * 2 + [True] * len(punctuation) if prior else None
    run = _drive(monkeypatch, labels, script=script, trim_s=0.5)

    _trimmed(run)
    expected_raw = ["あい", "".join(punctuation)] if prior else ["".join(punctuation)]
    assert run.raw == expected_raw
    assert "".join(run.commits) == "".join(expected_raw)
    assert (run.lines[-1].text, run.lines[-1].window) == (
        "".join(punctuation),
        len(labels) + 1,
    )
    assert sum(line.text == "".join(punctuation) for line in run.lines) == 1
    assert [line.seq for line in run.lines] == list(range(1, len(run.lines) + 1))
    assert run.submitted[-1] == (run.lines[-1].seq, "".join(punctuation), "ja")
    expected_observed = [("あい", 3)] if prior else []
    assert run.observed == [*expected_observed, ("".join(punctuation), len(labels) + 1)]


@pytest.mark.parametrize("close", [True, False], ids=["vad-close", "sentinel"])
def test_punctuation_only_remainder_after_published_pieces_reaches_no_consumer(
    monkeypatch: pytest.MonkeyPatch, close: bool
) -> None:
    labels = ("話", "し", "ます", "、", "。", "！")
    run = _drive(monkeypatch, labels, close=close, trim_s=0.5)

    _trimmed(run)
    assert run.trimmed[:3] == list(labels[:3])
    assert run.raw == ["".join(labels)]
    assert "".join(run.commits) == "".join(labels)
    assert _texts(run) == list(labels[:3])
    assert [(seq, text) for seq, text, _ in run.submitted] == [
        (line.seq, line.text) for line in run.lines
    ]
    assert run.observed == [("".join(labels[:3]), len(labels) + close)]


def test_a_fully_screened_utterance_leaves_the_next_number_for_clean_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = [BAD, "", BAD, "", "", "", *CLEAN[:4]]
    run = _drive(monkeypatch, labels, script=[True] * 4 + [False] * 2 + [True] * 4)

    _trimmed(run)
    assert run.lines and run.lines[0].seq == 1
    assert "".join(_texts(run)) == "".join(CLEAN[:4])
    assert [line.seq for line in run.lines] == list(range(1, len(run.lines) + 1))
    assert run.observed == [("".join(CLEAN[:4]), 11)]
    assert run.raw == [BAD * 2, "".join(CLEAN[:4])]
    assert run.state.dropped_captions > 0


def test_a_repetitive_piece_does_not_drop_the_clean_remainder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repetition = "繰り返し" * 12
    labels = (repetition, "", GOOD, "次の話です。", "終わります。", "また明日。")
    run = _drive(monkeypatch, labels, trim_s=0.5)

    assert run.trimmed[0] == repetition, "screening probe did not isolate the repetitive piece"
    assert live_stt.caption_defect(repetition)
    _trimmed(run)
    assert "".join(_texts(run)) == "".join(labels[2:])
    assert run.raw == ["".join(labels)]
    assert run.observed == [("".join(labels[2:]), len(labels) + 1)]
    assert run.state.dropped_captions == 1


def test_whitespace_matched_engine_spans_do_not_publish_before_a_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spans whose join outruns the stripped text let process() commit past emitted.

    The raw-length identity then grows a gap no trim made; only a trim may publish.
    (reviewer-1's witness, ported.)
    """
    from types import SimpleNamespace

    class Rec(live_stt.WhisperEngine):
        def __init__(self) -> None:
            self.hotwords = ""
            self.supports_hotwords = False

        def generate(self, samples, *, timestamps=False, language=None):  # type: ignore[override]
            return SimpleNamespace(
                texts=["会議"],
                chunks=[
                    SimpleNamespace(start_ts=0.0, end_ts=0.4, text=" 会議"),
                    SimpleNamespace(start_ts=0.4, end_ts=0.5, text=" "),
                ],
            )

    text, spans = Rec().decode_segments(np.zeros(16000, dtype=np.float32), language="ja")
    assert "".join(s.text for s in spans).strip() == text == "会議"
    run = _drive(monkeypatch, CLEAN[:4], trim_s=0.5, recognizer=Rec())
    assert all(p.trims == p.forced_trims == 0 for p in run.processors)
    assert all(line.window == len(CLEAN[:4]) + 1 for line in run.lines), (
        "SRC published without a normal or forced trim",
        [(line.window, line.text) for line in run.lines],
    )


def test_two_way_publishes_english_decoded_under_en(monkeypatch: pytest.MonkeyPatch) -> None:
    """The latin screen guards the JA pin; text decoded under <|en|> is supposed to be latin.

    Keyed on ASR_LANGUAGE, which --two-way leaves at ja, it dropped every English caption.
    """
    english = ("Hello ", "there. ", "We ", "moved ", "the ", "meeting ", "to ", "Friday.")
    run = _drive(monkeypatch, CLEAN, outcomes=[None, "en"], reverse=english)

    assert live_stt.ASR_LANGUAGE == "ja"
    assert run.state.dropped_captions == 0
    assert "".join(_texts(run)) == "".join(english[: len(CLEAN)]).strip()
    assert {source for _, _, source in run.submitted} == {"en"}


def test_a_latin_dominant_piece_decoded_under_ja_is_still_screened() -> None:
    assert live_stt.caption_defect(BAD, "ja")
    assert live_stt.caption_defect(BAD, "en") is None
    assert live_stt.caption_defect("繰り返し" * 12, "en")  # the loop rule spans languages


def test_two_way_still_screens_a_latin_piece_decoded_under_ja(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decode token, never the mode, picks the rule (reviewer-5/6's witness, ported)."""
    labels = (BAD, "", "", "", GOOD, "次の話です。")
    run = _drive(monkeypatch, labels, trim_s=0.5, outcomes=["ja"])

    assert set(run.rec.languages) == {"ja"}
    assert run.state.dropped_captions == 1
    assert BAD not in "".join(_texts(run))
    assert "".join(_texts(run)) == "".join(labels[4:])
