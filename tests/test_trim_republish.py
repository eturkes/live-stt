"""Early trim spans: retained text separates re-transcribed tails from real repeats."""

import asyncio
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import live_stt  # noqa: E402
import streaming  # noqa: E402
from streaming import SAMPLE_RATE, Segment, StreamingProcessor  # noqa: E402

Audio = NDArray[np.float32]
PUBLISHED = "歯の根っこが溶けてしまって歯がガタガタになるとかそういうことがあるらしいんですよね。"
OVERLAP = "そういうことがあるらしいんですよね。"
RETAINED = "それを事前に予測するAIを作ってほしいという。"
CONTINUATION = "それを事前に予測するAIを作ってほしいという共同研究。"


@dataclass(frozen=True)
class _Frame:
    text: str
    cut: int = 0


class _Script:
    def __init__(self, frames: list[_Frame]):
        self.frames = frames
        self.calls = 0

    def __call__(self, samples: Audio) -> tuple[str, list[Segment]]:
        if (
            self.calls
            and len(samples) == int(0.75 * SAMPLE_RATE)
            and self.frames[self.calls - 1].cut
        ):
            # `_echo` re-decodes the audio up to the first span: a scripted span was really
            # spoken, so that audio holds exactly its text -- every repeat here is a real one.
            spoken = self.frames[self.calls - 1]
            return spoken.text[: spoken.cut], [Segment(0.0, 0.75, spoken.text[: spoken.cut])]
        assert self.calls < len(self.frames), "unscripted decode"
        frame = self.frames[self.calls]
        self.calls += 1
        end_s = len(samples) / SAMPLE_RATE
        if not frame.text:
            return "", []
        if frame.cut:
            spans = [Segment(0.0, 0.75, frame.text[: frame.cut])]
            if frame.cut < len(frame.text):
                spans.append(Segment(0.75, end_s, frame.text[frame.cut :]))
            return frame.text, spans
        return frame.text, [Segment(0.0, end_s, frame.text)]

    def decode_segments(
        self, samples: Audio, language: str | None = None
    ) -> tuple[str, list[Segment]]:
        return self(samples)


def _frames(
    published: str, retained: str, following: list[_Frame], *, committed: int = 0
) -> list[_Frame]:
    # The second hypothesis agrees past the cut by exactly `committed` characters.
    assert 0 <= committed <= len(retained)
    return [
        _Frame(published + retained[:committed] + "※"),
        _Frame(published + retained, len(published)),
        *following,
    ]


def _step(processor: StreamingProcessor) -> str:
    processor.insert_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    return processor.process()[0]


def _start(
    published: str, retained: str, following: list[_Frame], *, committed: int = 0
) -> tuple[StreamingProcessor, list[str]]:
    processor = StreamingProcessor(
        decode=_Script(_frames(published, retained, following, committed=committed)),
        buffer_trim_s=1.5,
    )
    commits = [_step(processor), _step(processor)]
    assert commits == ["", published + retained[:committed]]
    assert processor.trims == 1 and processor.forced_trims == 0
    assert processor.offset_s == pytest.approx(0.75)
    assert processor.emitted == retained[:committed]
    return processor, commits


def _run(published: str, retained: str, following: list[_Frame], *, committed: int = 0) -> str:
    processor, commits = _start(published, retained, following, committed=committed)
    commits.extend(_step(processor) for _ in following)
    tail = processor.finish()
    assert processor.finish() == ""
    return "".join(commits) + tail


def test_live_1173_1174_publishes_retranscribed_tail_nowhere():
    actual = _run(
        PUBLISHED,
        RETAINED,
        [_Frame(OVERLAP + CONTINUATION), _Frame(OVERLAP + CONTINUATION)],
        committed=1,
    )
    assert actual == PUBLISHED + CONTINUATION


def test_live_307_308_discards_head_with_inserted_comma():
    published = "録画されるんでただそんなに内容しっかりはしていないので"
    overlap = "ただ、そんなに内容しっかりはしていないので"
    continuation = "まだ何もアウトカムがないような状態なので。"
    actual = _run(
        published,
        "まだ何もアウトカムがないように。",
        [_Frame(overlap + continuation), _Frame(overlap + continuation)],
    )
    assert actual == published + continuation


@pytest.mark.parametrize(
    ("published", "retained", "overlap", "continuation"),
    [
        (
            "ポックをどこかで止めるという",
            "のをやるんですけど。",
            "という",
            "のをやるんですけど誤解できない。",
        ),
        (
            "終わらないけどだから",
            "バッチリとダッチなやつがまた",
            "だから",
            "バッチングを打つんですがまだ答えが。",
        ),
    ],
    ids=["877-878", "402-403"],
)
def test_live_short_retranscribed_head_is_not_new_speech(
    published: str, retained: str, overlap: str, continuation: str
):
    actual = _run(
        published, retained, [_Frame(overlap + continuation), _Frame(overlap + continuation)]
    )
    assert actual == published + continuation


def test_persistent_head_stays_unpublished_through_the_next_trim():
    following = [
        _Frame(OVERLAP + "それを事前に予測する"),
        _Frame(OVERLAP + "それを事前に予測するAIを"),
        _Frame(OVERLAP + CONTINUATION),
        _Frame(OVERLAP + CONTINUATION + "次の計画です。", len(OVERLAP + CONTINUATION)),
        _Frame("次の計画です。"),
    ]
    processor, commits = _start(PUBLISHED, RETAINED, following, committed=1)
    commits.extend(_step(processor) for _ in following)
    assert processor.trims >= 2, "persistent-head script never reached its second trim"
    assert "".join(commits) + processor.finish() == PUBLISHED + CONTINUATION + "次の計画です。"


def test_empty_post_trim_hypothesis_does_not_erase_retained_evidence():
    actual = _run(
        PUBLISHED,
        RETAINED,
        [_Frame(""), _Frame(OVERLAP + CONTINUATION), _Frame(OVERLAP + CONTINUATION)],
    )
    assert actual == PUBLISHED + CONTINUATION


@pytest.mark.parametrize("committed", [0, 1, 2])
def test_generated_retranscribed_heads_preserve_every_new_character(committed: int):
    rng = random.Random(0x7A1)  # noqa: S311 — deterministic generated contract inputs
    alphabet = [chr(code) for code in range(0x4E00, 0x4F00)]
    failures = []
    for size in (1, 2, 3, 4, 8, 23, 24, 25, 64):
        for retained_size in (3, 4, 8, 32):
            chars = rng.sample(alphabet, 8 + size + retained_size + 5)
            published = "".join(chars[: 8 + size])
            overlap = published[-size:]
            retained = "".join(chars[8 + size : -5])
            continuation = retained + "".join(chars[-5:])
            actual = _run(
                published,
                retained,
                [_Frame(overlap + continuation), _Frame(overlap + continuation)],
                committed=committed,
            )
            if actual != published + continuation:
                failures.append((size, retained_size, published + continuation, actual))
    assert not failures, f"{len(failures)}/36 violations; first={failures[:2]}"


@pytest.mark.parametrize("committed", [0, 1])
def test_finish_with_retranscribed_head_flushes_only_new_speech(committed: int):
    actual = _run(PUBLISHED, RETAINED, [_Frame(OVERLAP + CONTINUATION)], committed=committed)
    assert actual == PUBLISHED + CONTINUATION


@pytest.mark.parametrize("phrase", ["そう", "はい。", "だから", "という"])
@pytest.mark.parametrize("committed", [0, 1])
def test_real_repeat_at_retained_head_keeps_both_copies(phrase: str, committed: int):
    published = "前の説明は" + phrase
    retained = phrase + "次の説明です。"
    actual = _run(published, retained, [_Frame(retained), _Frame(retained)], committed=committed)
    assert actual == published + retained


def test_generated_real_repeats_keep_every_copy():
    rng = random.Random(0xB07)  # noqa: S311 — deterministic generated contract inputs
    alphabet = "あいうえおかきくけこABCDEFGHIJ0123456789漢字資料会議計画"
    for length in (1, 2, 3, 8, 24, 32):
        phrase = "".join(rng.sample(alphabet, length))
        for copies in (2, 3, 5):
            published = "前段。" + phrase
            retained = phrase * (copies - 1) + "後段。"
            actual = _run(published, retained, [_Frame(retained), _Frame(retained)])
            assert actual == published + retained, (length, copies, actual)


def test_real_repeat_after_retained_content_is_preserved():
    published = "説明が終わったので、はい。"
    retained = "次の話ですが、"
    continuation = retained + "はい。続きを話します。"
    assert _run(published, retained, [_Frame(continuation), _Frame(continuation)]) == (
        published + continuation
    )


@pytest.mark.parametrize("retained", ["続き", "続きです。", "そういう別の話", "A new topic."])
@pytest.mark.parametrize("committed", [0, 1])
def test_unchanged_retained_head_keeps_existing_publication(retained: str, committed: int):
    continuation = retained + "さらに次へ。"
    assert (
        _run(PUBLISHED, retained, [_Frame(continuation), _Frame(continuation)], committed=committed)
        == PUBLISHED + continuation
    )


@pytest.mark.parametrize("overlap", ["", "はい。", OVERLAP])
def test_empty_retained_text_keeps_existing_publication(overlap: str):
    published = "前の説明ははい。" + OVERLAP
    continuation = overlap + "次の話です。"
    assert _run(published, "", [_Frame(continuation), _Frame(continuation)]) == (
        published + continuation
    )


def test_a_doubled_trimming_decode_whose_audio_holds_both_copies_keeps_both():
    retained = OVERLAP + CONTINUATION
    assert _run(PUBLISHED, retained, [_Frame(retained), _Frame(retained)], committed=1) == (
        PUBLISHED + retained
    )


class _Vad:
    def __init__(self, speech_windows: int):
        self.speech_windows = speech_windows
        self.calls = 0
        self.queued = 0

    def accept_waveform(self, _samples: Audio) -> None:
        self.calls += 1
        if self.calls == self.speech_windows + 1:
            self.queued += 1

    def is_speech_detected(self) -> bool:
        return 0 < self.calls <= self.speech_windows

    def empty(self) -> bool:
        return self.queued == 0

    def pop(self) -> None:
        assert self.queued
        self.queued -= 1


@pytest.mark.parametrize("translate", [False, True], ids=["source-only", "translator"])
def test_worker_src_lines_never_republish_1173_tail(
    monkeypatch: pytest.MonkeyPatch, translate: bool
):
    frames = _frames(
        PUBLISHED,
        RETAINED,
        [_Frame(OVERLAP + CONTINUATION), _Frame(OVERLAP + CONTINUATION)],
        committed=1,
    )
    recognizer = _Script(frames)
    lines: list[tuple[int, str]] = []
    submitted: list[tuple[int, str]] = []

    def capture(tag: str, seq: int, text: str, _file: object, *, held: bool = False) -> None:
        assert tag == "SRC" and not held
        lines.append((seq, text))

    class Translator:
        def submit(self, seq: int, text: str, source: str | None = None) -> None:
            submitted.append((seq, text))

    monkeypatch.setattr(live_stt, "VAC_CHUNK_S", 1.0)
    monkeypatch.setattr(live_stt, "VAC_TRIM_S", 1.5)
    monkeypatch.setattr(live_stt, "VAD_PRE_PAD_S", 0.0)
    monkeypatch.setattr(live_stt, "emit_line", capture)
    state = live_stt.State()
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(4):
        queue.put_nowait(np.zeros(SAMPLE_RATE, dtype=np.float32))
    queue.put_nowait(None)
    rec: Any = recognizer
    vad: Any = _Vad(3)
    translator: Any = Translator() if translate else None
    asyncio.run(live_stt.worker(rec, vad, SAMPLE_RATE, queue, state, None, translator=translator))
    assert not state.stopping, "worker failed before publication could be checked"
    assert recognizer.calls == len(frames)
    assert lines == [(1, PUBLISHED), (2, CONTINUATION)]
    assert submitted == (lines if translate else [])


# Reviewer reproducers (reviewers 7 + 8) against the first fix shape, which took evidence
# from `previous`, required two exact opener characters and ranked the longest agreement.


@pytest.mark.parametrize("stop", [5, len(OVERLAP)])
def test_a_decode_stopping_inside_the_head_cannot_vouch_for_it(stop: int):
    following = [
        _Frame(OVERLAP[:stop]),
        _Frame(OVERLAP + CONTINUATION),
        _Frame(OVERLAP + CONTINUATION),
    ]
    assert _run(PUBLISHED, RETAINED, following, committed=1) == PUBLISHED + CONTINUATION


def test_empty_retained_text_stays_no_evidence_after_a_later_decode():
    following = [
        _Frame(CONTINUATION),
        _Frame(OVERLAP + CONTINUATION),
        _Frame(OVERLAP + CONTINUATION),
    ]
    assert _run(PUBLISHED, "", following) == PUBLISHED + OVERLAP + CONTINUATION


def test_one_retained_character_is_evidence():
    following = [_Frame("だから次の話です。"), _Frame("だから次の話です。")]
    assert _run("前の説明だから", "次", following) == "前の説明だから次の話です。"


def test_a_later_recurrence_of_the_retained_opener_keeps_the_speech_before_it():
    continuation = "次の件で、次の話です。"
    following = [_Frame(OVERLAP + continuation), _Frame(OVERLAP + continuation)]
    assert _run(PUBLISHED, "次の話です。", following) == PUBLISHED + continuation


def test_a_heard_repeat_survives_a_re_spelled_opener():
    following = "明です。説明です。続きます。"
    frames = [_Frame(following), _Frame(following)]
    assert (
        _run("前の説明です。", "説明です。説明です。続きます。", frames)
        == "前の説明です。" + following
    )


def test_head_costs_match_one_alignment_per_head():
    rng = random.Random(3)  # noqa: S311
    for _ in range(2000):
        told = "".join(rng.choice("abcd") for _ in range(rng.randint(1, 10)))
        text = "".join(rng.choice("abcd") for _ in range(rng.randint(1, 14)))
        costs = streaming.head_costs(told, text)
        for h in range(1, len(text) + 1):
            assert costs[h] == min(streaming.tail_costs(text[h - 1 :: -1], told[::-1]))


def test_a_repeated_opener_over_a_long_cut_stays_inside_one_update():
    # 440 trimmed characters and an opener recurring at every position: one alignment per
    # candidate head took 6.5-7.3 CPU s here against the 1 s update cadence (reviewer-7).
    frames = [
        _Frame("あ" * 440 + "※"),
        _Frame("あ" * 440 + "ああ後段", 440),
        _Frame("い" + "あ" * 443),
    ]
    processor = StreamingProcessor(decode=_Script(frames), buffer_trim_s=1.5)
    _step(processor)
    _step(processor)
    assert processor.trims == 1
    started = time.process_time()
    _step(processor)
    assert time.process_time() - started < live_stt.VAC_CHUNK_S
