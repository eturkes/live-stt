"""Published text stays fixed when its decode spelling or segment timing changes."""

import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming import SAMPLE_RATE, Segment, StreamingProcessor  # noqa: E402

Audio = NDArray[np.float32]
Decode = Callable[[Audio], tuple[str, list[Segment]]]
PREFIX = "今日は新しいビジュアルについて話します"
CONTINUATION = "続いて説明を始めます"


def _audio(seconds: float = 1.0) -> Audio:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def _scripted(*texts: str) -> Decode:
    remaining = list(texts)

    def decode(samples: Audio) -> tuple[str, list[Segment]]:
        text = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return text, [Segment(0.0, len(samples) / SAMPLE_RATE, text)]

    return decode


def _step(processor: StreamingProcessor) -> str:
    processor.insert_audio(_audio())
    return processor.process()[0]


def _edit(prefix: str, kind: str, position: int, replacement: str = "、") -> str:
    if kind == "insert":
        return prefix[:position] + replacement + prefix[position:]
    if kind == "delete":
        return prefix[:position] + prefix[position + 1 :]
    assert kind == "substitute"
    return prefix[:position] + replacement + prefix[position + 1 :]


def _revised_run(prefix: str, revised: str, continuation: str) -> tuple[list[str], str]:
    processor = StreamingProcessor(
        decode=_scripted(
            prefix + "甲", prefix + continuation[:1], revised + continuation, revised + continuation
        ),
        buffer_trim_s=64.0,
    )
    commits = [_step(processor) for _ in range(4)]
    assert processor.trims == processor.forced_trims == 0
    return commits, processor.finish()


@pytest.mark.parametrize("kind", ["insert", "delete", "substitute"])
@pytest.mark.parametrize(
    "position",
    [0, len(PREFIX) // 2, len(PREFIX) - 2, len(PREFIX) - 1],
    ids=["start", "middle", "penultimate", "last"],
)
def test_revised_published_prefix_resumes_at_its_end(kind: str, position: int):
    revised = _edit(PREFIX, kind, position)
    commits, tail = _revised_run(PREFIX, revised, CONTINUATION)
    assert commits == ["", PREFIX, "", CONTINUATION]
    assert tail == ""


def test_same_length_katakana_respelling_control():
    prefix = "このパックについて話します"
    commits, tail = _revised_run(prefix, prefix.replace("パック", "バック"), CONTINUATION)
    assert commits == ["", prefix, "", CONTINUATION]
    assert tail == ""


@pytest.mark.parametrize("short_length", [0, len(PREFIX) // 2, len(PREFIX) - 1])
@pytest.mark.parametrize("resumption", ["unchanged", "insert", "delete"])
def test_short_decode_keeps_publication_and_resumes(short_length: int, resumption: str):
    revised = PREFIX if resumption == "unchanged" else _edit(PREFIX, resumption, 3)
    short = PREFIX[:short_length]
    processor = StreamingProcessor(
        decode=_scripted(
            PREFIX + "甲",
            PREFIX + "乙",
            short,
            short,
            revised + CONTINUATION,
            revised + CONTINUATION,
        ),
        buffer_trim_s=64.0,
    )
    assert _step(processor) == ""
    assert _step(processor) == PREFIX
    for _ in range(2):
        assert _step(processor) == ""
        assert processor.emitted == PREFIX
    resumed = _step(processor) + _step(processor)
    assert resumed == CONTINUATION
    assert processor.finish() == ""
    assert processor.trims == processor.forced_trims == 0


def test_segment_timestamp_jitter_does_not_change_commits_control():
    texts = [PREFIX + "甲", PREFIX + "乙", PREFIX + CONTINUATION, PREFIX + CONTINUATION]

    def run(jitter: bool) -> tuple[list[str], str]:
        remaining = list(enumerate(texts))

        def decode(_samples: Audio) -> tuple[str, list[Segment]]:
            index, text = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            delta = [0.0, 0.09, -0.07, 0.13][index] if jitter else 0.0
            split = len(PREFIX) // 2
            return text, [
                Segment(0.0, 0.5 + delta, text[:split]),
                Segment(0.5 + delta, float(index + 1) + delta, text[split:]),
            ]

        processor = StreamingProcessor(decode=decode, buffer_trim_s=64.0)
        return [_step(processor) for _ in texts], processor.finish()

    expected = (["", PREFIX, "", CONTINUATION], "")
    assert run(False) == expected
    assert run(True) == expected


@pytest.mark.parametrize("padding", ["", " ", "\t", "  ", " \t", "\n   "], ids=repr)
def test_final_segment_offsets_use_stripped_text(padding: str):
    # Decode the audio that remains after a cut, not the original full buffer again.
    def decode(samples: Audio) -> tuple[str, list[Segment]]:
        if len(samples) > 5 * SAMPLE_RATE:
            return "会議中終", [
                Segment(0.0, 4.0, padding + "会議"),
                Segment(4.0, 6.0, "中"),
                Segment(6.0, 8.0, "終"),
            ]
        if len(samples) > 3 * SAMPLE_RATE:
            return "中終", [Segment(0.0, 2.0, "中"), Segment(2.0, 4.0, "終")]
        return "終", [Segment(0.0, 2.0, "終")]

    processor = StreamingProcessor(decode=decode, buffer_trim_s=8.0)
    processor.insert_audio(_audio(9.0))
    commit, _ = processor.process()
    tail = processor.finish()
    assert commit + tail == "会議中終"
    assert commit == "会議中"
    assert tail == "終"
    assert processor.forced_trims == 0


@pytest.mark.parametrize("kind", ["unchanged", "insert", "delete", "substitute"])
def test_finish_returns_only_unpublished_remainder(kind: str):
    revised = PREFIX if kind == "unchanged" else _edit(PREFIX, kind, len(PREFIX) - 2)
    processor = StreamingProcessor(
        decode=_scripted(PREFIX, PREFIX, revised + CONTINUATION), buffer_trim_s=64.0
    )
    assert _step(processor) == ""
    assert _step(processor) == PREFIX
    assert _step(processor) == ""
    assert processor.finish() == CONTINUATION
    assert processor.finish() == ""
    assert processor.trims == processor.forced_trims == 0


@pytest.mark.parametrize("kind", ["insert", "delete", "substitute"])
def test_seeded_single_edit_preserves_committed_stream(kind: str):
    rng = random.Random(0xB0A4D + ["insert", "delete", "substitute"].index(kind))  # noqa: S311
    punctuation = "、。！？（）「」・，,.;:"
    continuation = "続稿完成"
    excluded = set(continuation + "甲乙")
    alphabet = [
        chr(code)
        for start, stop in [(0x3041, 0x3097), (0x30A1, 0x30F7), (0x4E00, 0x4E80)]
        for code in range(start, stop)
        if chr(code) not in excluded
    ]
    failures: list[str] = []
    for case in range(192):
        length = [2, 3, 8, 16, 32, 64][case % 6]
        # Distinct characters make the edit location identifiable independently of the policy.
        prefix = "".join(rng.sample(alphabet, length))
        position = [0, length // 2, max(0, length - 2), length - 1][case % 4]
        if case % 5 == 4:
            position = rng.randrange(length)
        replacements = punctuation if case % 4 else "".join(c for c in alphabet if c not in prefix)
        revised = _edit(prefix, kind, position, rng.choice(replacements))
        commits, tail = _revised_run(prefix, revised, continuation)
        expected = prefix + continuation
        actual = "".join(commits)
        if actual != expected or tail != "" or commits[:2] != ["", prefix] or commits[2] != "":
            failures.append(
                f"case={case} {kind}@{position} prefix={prefix!r} revised={revised!r} "
                f"expected={expected!r} actual={actual!r} tail={tail!r}"
            )
    assert not failures, f"{len(failures)}/192 violations; first={failures[:3]}"
