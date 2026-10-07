"""A dropped published mark has no audio and must not spend the next word character."""

import random
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming import SAMPLE_RATE, Segment, StreamingProcessor  # noqa: E402

Audio = NDArray[np.float32]
Decode = Callable[[Audio], tuple[str, list[Segment]]]
WORD_CLASSES = {
    "hiragana": "".join(chr(code) for code in range(0x3041, 0x3097) if chr(code).isalpha()),
    "katakana": "".join(chr(code) for code in range(0x30A1, 0x30FB) if chr(code).isalpha()),
    "kanji": "".join(chr(code) for code in range(0x4E00, 0x4E80)),
    "latin": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "digit": "0123456789",
}
MARKS = tuple(
    chr(code)
    for code in range(sys.maxunicode + 1)
    if unicodedata.category(chr(code))[0] in {"P", "Z"} or chr(code).isspace()
)


def _scripted(*texts: str) -> Decode:
    remaining = list(texts)

    def decode(samples: Audio) -> tuple[str, list[Segment]]:
        text = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return text, [Segment(0.0, len(samples) / SAMPLE_RATE, text)]

    return decode


def _step(processor: StreamingProcessor) -> str:
    processor.insert_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    return processor.process()[0]


def _run(*texts: str) -> tuple[list[str], str]:
    processor = StreamingProcessor(decode=_scripted(*texts), buffer_trim_s=64.0)
    commits = [_step(processor) for _ in texts]
    tail = processor.finish()
    assert processor.finish() == ""
    return commits, tail


@pytest.mark.parametrize(
    ("prefix", "first", "last"),
    [
        ("仕様書を書いて。", "それどおりにあって。", "それどおりにあってればうまくやってます。"),
        ("コースについて。", "コースの3人", "コースの3人で進めます。"),
    ],
    ids=["src-1137", "katakana-first-mora"],
)
def test_live_dropped_period_keeps_first_word_character(prefix: str, first: str, last: str):
    commits, tail = _run(prefix, prefix, prefix[:-1] + first, prefix[:-1] + last)
    assert commits[:2] == ["", prefix]
    assert "".join(commits) + tail == prefix + last


def test_live_dropped_period_keeps_ja_of_jaa():
    prefix = "そういうことですね。"
    continuation = "じゃあ次の説明を始めます。"
    commits, tail = _run(
        prefix,
        prefix,
        prefix[:-1] + "じゃあ次の説明を",
        prefix[:-1] + continuation,
    )
    assert commits[:2] == ["", prefix]
    assert "".join(commits) + tail == prefix + continuation


@pytest.mark.parametrize("mark", ["。", "、", "?", " ", "　"], ids=repr)
def test_finish_dropped_mark_keeps_first_word_character(mark: str):
    words = "仕様書を書いて"
    continuation = "それどおりに進めます"
    processor = StreamingProcessor(
        decode=_scripted(words + mark + "!", words + mark + "?", words + continuation),
        buffer_trim_s=64.0,
    )
    commits = [_step(processor), _step(processor)]
    assert commits == ["", words + mark]
    commits.append(_step(processor))
    assert commits[-1] == ""
    tail = processor.finish()
    assert "".join(commits) + tail == words + mark + continuation
    assert processor.finish() == ""


@pytest.mark.parametrize("word_class", list(WORD_CLASSES))
def test_seeded_dropped_mark_preserves_whole_published_stream(word_class: str):
    rng = random.Random(0xD09 + list(WORD_CLASSES).index(word_class))  # noqa: S311
    alphabet = "".join(WORD_CLASSES.values())
    failures: list[str] = []
    lengths = [1, 2, 3, 8, 16, 23, 24, 25, 32, 64]
    for case, mark in enumerate(MARKS):
        continuation = "".join(rng.sample(WORD_CLASSES[word_class], [1, 2, 3, 8][case % 4]))
        available = [char for char in alphabet if char not in continuation]
        # Distinct words + punctuation-only prior tails exclude ambiguous word-end rewrites.
        words = "".join(rng.sample(available, lengths[case % len(lengths)]))
        published = words + mark
        commits, tail = _run(
            published + "!", published + "?", words + continuation, words + continuation
        )
        actual = "".join(commits) + tail
        expected = published + continuation
        if commits[:2] != ["", published] or actual != expected:
            failures.append(
                f"case={case} class={word_class} mark={mark!r} "
                f"category={unicodedata.category(mark)} prefix={words!r} "
                f"continuation={continuation!r} expected={expected!r} actual={actual!r} "
                f"commits={commits!r} tail={tail!r}"
            )
    assert not failures, f"{len(failures)}/{len(MARKS)} violations; first={failures[:3]}"


@pytest.mark.parametrize(
    ("old", "new"),
    [("。", "、"), ("、", "。"), ("!", "?"), (".", ","), (" ", "。"), (" ", "　")],
    ids=["period-comma", "comma-period", "bang-question", "dot-comma", "space-period", "spaces"],
)
def test_mark_respelling_keeps_existing_boundary_control(old: str, new: str):
    words = "仕様書を書いて"
    continuation = "それどおりに進めます"
    published = words + old
    revised = words + new
    commits, tail = _run(
        published + ":", published + ";", revised + continuation, revised + continuation
    )
    assert commits[:2] == ["", published]
    assert "".join(commits) + tail == published + continuation


@pytest.mark.parametrize("word", ["あ", "コ", "漢", "A", "3"], ids=list(WORD_CLASSES))
def test_word_respelt_as_mark_keeps_existing_boundary_control(word: str):
    words = "仕様書を書いて"
    continuation = "次の説明を始めます"
    published = words + word
    revised = words + "。"
    commits, tail = _run(
        published + ":", published + ";", revised + continuation, revised + continuation
    )
    assert commits[:2] == ["", published]
    assert "".join(commits) + tail == published + continuation


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (("。", "。", "AB", "AB"), "。AB"),
        (("説明。", "説明。こ", "説明ここから先", "説明ここから先に"), "説明。ここから先に"),
    ],
    ids=["lone-mark", "evidenced-repeat"],
)
def test_review_shapes_keep_the_first_word_character(texts: tuple[str, ...], expected: str):
    # reviewer-3 + reviewer-4: a one-character published tail reached _thin first, and a
    # repeated character evidenced both tied ends, so nearness spent it on the mark.
    commits, tail = _run(*texts)
    assert "".join(commits) + tail == expected


def test_a_mark_the_record_respelled_but_never_published_gives_nothing_back():
    # reviewer-3: one decode re-spells the published C as 。 in the record; restoring the
    # original spelling must not hand C back as new speech.
    commits, tail = _run("ABC", "ABCD", "AB。XY", "ABCDE", "ABCDEF")
    assert "".join(commits) + tail == "ABCDEF"


def test_a_trim_that_leaves_only_the_published_mark_keeps_the_next_character():
    # reviewer-4: the trim cuts after the first segment, so the record holds the lone 。.
    calls = iter(
        [
            ("仕様書を書いて。!", [Segment(0.0, 1.0, "仕様書を書いて"), Segment(1.0, 2.0, "。!")]),
            ("仕様書を書いて。?", [Segment(0.0, 1.0, "仕様書を書いて"), Segment(1.0, 2.0, "。?")]),
            ("それ", [Segment(0.0, 1.0, "それ")]),
            ("それ", [Segment(0.0, 2.0, "それ")]),
        ]
    )
    processor = StreamingProcessor(decode=lambda _audio: next(calls), buffer_trim_s=1.5)
    commits = [_step(processor) for _ in range(4)]
    assert processor.trims >= 1
    assert "".join(commits) + processor.finish() == "仕様書を書いて。それ"
