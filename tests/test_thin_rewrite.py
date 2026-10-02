"""Thin published-prefix rewrites need located evidence and two-decode confirmation."""

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
ASCII = "ABCDEFGH"
JAPANESE = "今日の状況を詳しくお話します"


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
    assert processor.trims == processor.forced_trims == 0
    assert processor.offset_s == 0.0
    return commits, tail


def _output(*texts: str) -> str:
    commits, tail = _run(*texts)
    return "".join(commits) + tail


def _distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (source != target)))
        row = current
    return row[-1]


def test_reviewer_7_probe_keeps_new_speech():
    assert _output(ASCII, ASCII + "I", "XXGHIJ", "XXGHIJK") == ASCII + "IJK"


def test_terminal_kana_kanji_respelling_keeps_new_speech():
    assert (
        _output(JAPANESE, JAPANESE + "次です", "概要を説明致す次です", "概要を説明致す次ですね")
        == JAPANESE + "次ですね"
    )


def test_two_decode_head_drop_then_restoration_keeps_new_speech():
    prefix = "りごとを言いま"
    assert (
        _output(
            prefix,
            prefix + "した。一体誰が",
            "ました。一体誰が",
            "ました。一体誰がイ",
            "りごとを言いました。一体誰がイ",
        )
        == "りごとを言いました。一体誰がイ"
    )


@pytest.mark.parametrize("blip", ["IJ", "XXGHIJ"], ids=["total-drop", "with-continuation"])
def test_total_rewrite_then_restoration_control(blip: str):
    assert _output(ASCII, ASCII + "I", blip, ASCII + "IJK", ASCII + "IJKL") == "ABCDEFGHIJKL"


@pytest.mark.parametrize(
    ("prefix", "continuation", "garbage"),
    [(ASCII, "IJKL", "XXXXXXXXXX"), (JAPANESE, "次ですね", "雑" * 20)],
    ids=["ascii", "japanese"],
)
def test_one_decode_garbage_control(prefix: str, continuation: str, garbage: str):
    assert (
        _output(
            prefix, prefix + continuation[:2], garbage, prefix + continuation, prefix + continuation
        )
        == prefix + continuation
    )


@pytest.mark.parametrize(
    ("prefix", "revised", "continuation"),
    [(ASCII, "XXXXXXGH", "IJK"), ("あいうえおかきく", "さしすせそたきく", "けこさ")],
    ids=["ascii", "japanese"],
)
def test_same_length_thin_rewrite_control(prefix: str, revised: str, continuation: str):
    assert len(prefix) == len(revised)
    assert (
        _output(prefix, prefix + continuation[:2], revised + continuation, revised + continuation)
        == prefix + continuation
    )


@pytest.mark.parametrize("matching", [1, 2, 3], ids=["one-match", "two-matches", "below-half"])
@pytest.mark.parametrize("alphabet", ["ABCDEFGH", "あいうえおかきく"], ids=["ascii", "japanese"])
def test_below_half_adopts_after_one_held_decode(matching: int, alphabet: str):
    continuation = "IJK" if alphabet == ASCII else "けこさ"
    revised = "§" + alphabet[-matching:]
    processor = StreamingProcessor(
        decode=_scripted(
            alphabet,
            alphabet + continuation[:2],
            revised + continuation[:2],
            revised + continuation,
        ),
        buffer_trim_s=64.0,
    )
    commits = [_step(processor), _step(processor)]
    assert commits == ["", alphabet]
    commits.append(_step(processor))
    assert commits[-1] == ""
    assert processor.emitted == alphabet
    commits.append(_step(processor))
    assert commits[-1] == continuation[:2]
    tail = processor.finish()
    assert "".join(commits) + tail == alphabet + continuation


def test_same_length_first_evidenced_decode_leaves_record():
    processor = StreamingProcessor(
        decode=_scripted(ASCII, ASCII + "IJ", "XXXXXXGHIJ", "XXXXXXGHIJK"), buffer_trim_s=64.0
    )
    commits = [_step(processor), _step(processor), _step(processor)]
    assert commits == ["", ASCII, ""]
    assert processor.emitted == ASCII
    commits.append(_step(processor))
    assert "".join(commits) + processor.finish() == ASCII + "IJK"


@pytest.mark.parametrize("intervening", ["", "Q", "ABC"], ids=["empty", "garbage", "short-prefix"])
def test_evidence_comes_from_last_located_decode(intervening: str):
    assert _output(ASCII, ASCII + "IJ", intervening, "XXGHIJ", "XXGHIJK") == ASCII + "IJK"


def test_alignment_located_record_refreshes_evidence():
    assert _output(ASCII, ASCII + "IJ", "ABCXDEFGHIJ", "XXGHIJK", "XXGHIJKL") == ASCII + "IJKL"


def test_adopted_thin_end_refreshes_evidence():
    assert (
        _output(ASCII, ASCII + "IJ", "UVWXYZGHIJ", "UVWXYZGHIJKL", "QHIJKLM", "QHIJKLMN")
        == ASCII + "IJKLMN"
    )


@pytest.mark.parametrize("matching", [1, 2])
def test_thin_end_after_nonzero_tail_start(matching: int):
    prefix = "abcdefghijklmnopqrstuvwx" + ASCII
    retained_head = prefix[:8]
    revised = retained_head + "§" + prefix[-matching:]
    assert _output(prefix, prefix + "IJ", revised + "IJ", revised + "IJK") == prefix + "IJK"


def test_lowest_alignment_cost_wins_before_nearest_end():
    revised = "XGHIJXXHIJK"
    assert _distance(ASCII, revised[:3]) == 6
    assert _distance(ASCII, revised[:8]) == 7
    assert _output(ASCII, ASCII + "IJ", revised, revised + "L") == ASCII + revised[3:] + "L"


def test_equal_alignment_cost_uses_end_nearest_old_one():
    revised = "XGIJXGIJK"
    assert _distance(ASCII, revised[:2]) == _distance(ASCII, revised[:6]) == 7
    assert _output(ASCII, ASCII + "IJ", revised, revised + "L") == ASCII + revised[6:] + "L"


@pytest.mark.parametrize(
    "second", ["YYGHIJK", "XXGHKIJK"], ids=["different-prefix", "different-end"]
)
def test_mismatched_second_decode_keeps_count(second: str):
    processor = StreamingProcessor(
        decode=_scripted(ASCII, ASCII + "IJ", "XXGHIJ", second, ASCII + "IJKL", ASCII + "IJKLM"),
        buffer_trim_s=64.0,
    )
    commits = [_step(processor), _step(processor), _step(processor)]
    assert commits == ["", ASCII, ""]
    assert processor.emitted == ASCII
    commits.append(_step(processor))
    assert commits[-1] == ""
    commits.extend([_step(processor), _step(processor)])
    assert "".join(commits) + processor.finish() == ASCII + "IJKLM"


def test_count_fallback_record_has_no_evidence_control():
    assert _output(ASCII, ASCII + "IJ", "XXXXXXGHKL", "ZZGHKL", "ZZGHKLM") == ASCII


def test_count_fallback_invalidates_old_located_evidence_control():
    assert _output(ASCII, ASCII + "IJ", "ZZZZZZGHKL", "XXGHIJ", "XXGHIJK") == ASCII


@pytest.mark.parametrize("fallback", ["ZZZZZZGH", "XXXXXXGH"], ids=["thin-lock", "half-control"])
def test_count_fallback_record_regains_evidence_only_when_located(fallback: str):
    assert (
        _output(
            ASCII,
            ASCII + "IJ",
            fallback + "KL",
            "YYGHKL",
            "YYGHKLM",
            fallback + "IJ",
            "XXGHIJ",
            "XXGHIJK",
        )
        == ASCII + "IJK"
    )


@pytest.mark.parametrize(
    ("prefix", "continuation", "revised"),
    [(ASCII, "IJ", "XXGHIJ"), (ASCII, "I", "XXGHIJ"), (JAPANESE, "次です", "概要を説明致す次です")],
    ids=["two-evidence-chars", "one-available-char", "kana-kanji"],
)
def test_finish_adopts_unconfirmed_evidenced_end(prefix: str, continuation: str, revised: str):
    commits, tail = _run(prefix, prefix + continuation, revised)
    assert commits == ["", prefix, ""]
    expected = "次です" if prefix == JAPANESE else "IJ"
    assert tail == expected
    assert "".join(commits) + tail == prefix + expected


@pytest.mark.parametrize(
    "revised",
    ["§§§§IJK", "§§§§§§§§IJK", "XXGHIK", "XXGHJI", "XXGHI", "XXGH", ""],
    ids=[
        "zero-match-short",
        "zero-match-long",
        "second-char-mismatch",
        "first-char-mismatch",
        "partial-evidence",
        "no-continuation",
        "empty",
    ],
)
def test_no_evidence_count_stands_exactly_control(revised: str):
    expected_tail = revised[len(ASCII) :]
    assert _output(ASCII, ASCII + "IJ", revised, revised) == ASCII + expected_tail


def test_end_at_or_before_tail_start_is_not_evidence_control():
    prefix = "abcdefghijklmnopqrstuvwx" + ASCII
    assert _output(prefix, prefix + "IJ", "abcdefGIJ", "abcdefGIJK") == prefix


@pytest.mark.parametrize("matching", [4, 5, 8], ids=["exact-half", "above-half", "prefix-match"])
def test_at_least_half_agreement_control(matching: int):
    revised = ASCII if matching == len(ASCII) else "§" + ASCII[-matching:]
    assert _output(ASCII, ASCII + "IJ", revised + "IJ", revised + "IJK") == ASCII + "IJK"


def test_seeded_confirmed_thin_rewrite_keeps_whole_continuation():
    rng = random.Random(0x7A1B)  # noqa: S311
    failures: list[str] = []
    for case in range(192):
        length = [3, 4, 7, 8, 23, 24, 25, 32, 64][case % 9]
        start = 0x21 if case % 2 else 0x3041
        prefix = "".join(rng.sample([chr(start + i) for i in range(96)], length))
        tail_length = min(24, length)
        matching = rng.randint(1, (tail_length - 1) // 2)
        revised = prefix[: length - tail_length] + "§" + prefix[-matching:]
        continuation = "甲乙丙丁"
        evidence = continuation[: 1 + case % 2]
        actual = _output(prefix, prefix + evidence, revised + evidence, revised + continuation)
        expected = prefix + continuation
        if actual != expected:
            failures.append(
                f"case={case} matching={matching}/{tail_length} prefix={prefix!r} "
                f"revised={revised!r} evidence={evidence!r} expected={expected!r} actual={actual!r}"
            )
    assert not failures, f"{len(failures)}/192 violations; first={failures[:3]}"


@pytest.mark.parametrize("kind", ["garbage", "total-drop", "head-drop"])
def test_seeded_one_decode_blip_preserves_clean_stream(kind: str):
    rng = random.Random(0x7A1A + ["garbage", "total-drop", "head-drop"].index(kind))  # noqa: S311
    failures: list[str] = []
    for case in range(192):
        length = [8, 16, 24, 32, 64][case % 5]
        start = 0x41 if case % 2 else 0x3041
        alphabet = [chr(start + i) for i in range(96) if chr(start + i) not in "甲乙丙丁"]
        prefix = "".join(rng.sample(alphabet, length))
        continuation = "甲乙丙丁"
        clean = [
            prefix,
            prefix + continuation[:2],
            prefix + continuation[:3],
            prefix + continuation,
        ]
        if kind == "garbage":
            blip = "§" * [1, length, length + 2][case % 3]
        elif kind == "total-drop":
            blip = continuation[: 1 + case % 3]
        else:
            blip = prefix[-(1 + case % 3) :] + continuation[:2]
        expected = prefix + continuation
        clean_output = _output(*clean)
        with_blip = _output(*clean[:2], blip, *clean[2:])
        if clean_output != expected or with_blip != expected:
            failures.append(
                f"case={case} {kind} prefix={prefix!r} blip={blip!r} "
                f"expected={expected!r} clean={clean_output!r} actual={with_blip!r}"
            )
    assert not failures, f"{len(failures)}/192 violations; first={failures[:3]}"
