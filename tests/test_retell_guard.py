"""Published-tail retells: exact suffix guard, repeat vetoes and buffer-local evidence."""

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
PREFIX = "前半の説明について"


def _scripted(*texts: str) -> Decode:
    pending = iter(texts)
    latest = ""

    def decode(samples: Audio) -> tuple[str, list[Segment]]:
        nonlocal latest
        latest = next(pending, latest)
        return latest, [Segment(0.0, len(samples) / SAMPLE_RATE, latest)]

    return decode


def _step(processor: StreamingProcessor) -> str:
    processor.insert_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    return processor.process()[0]


def _run(texts: list[str], final: bool) -> tuple[list[str], str]:
    processor = StreamingProcessor(decode=_scripted(*texts), buffer_trim_s=64.0)
    commits = [_step(processor) for _ in texts]
    if not final:
        commits.append(_step(processor))
    tail = processor.finish()
    assert processor.finish() == ""
    assert processor.trims == processor.forced_trims == 0
    return commits, tail


@pytest.mark.parametrize("final", [False, True], ids=["process", "finish"])
@pytest.mark.parametrize(
    ("published", "heard", "rewrites", "continuation"),
    [
        (
            PREFIX + "やりやすくっていう",
            "現状は。",
            [PREFIX + "やりやすくて現状は。", PREFIX + "やりやすくっていう現状は。"],
            "現状は。",
        ),
        (
            "対象のそういう",
            "乙",
            ["そして、対象のそういう結果は。"],
            "結果は。",
        ),
        (
            "対象の位置に入れるっていう",
            "作業は。",
            ["対象の説明します。対象の説明です。", "対象の説明します。入れるっていう作業は。"],
            "作業は。",
        ),
        (
            PREFIX + "ちょっとい",
            "じります。",
            [PREFIX + "いじります。", PREFIX + "ちょっといじります。"],
            "じります。",
        ),
    ],
    ids=["A1-reverted", "A2-restored-head", "A3-count-record", "A4-midword"],
)
def test_retold_suffix_is_published_once(
    published: str, heard: str, rewrites: list[str], continuation: str, final: bool
):
    # Only the defect's short suffixes come from speech; surrounding text is synthetic.
    commits, tail = _run([published + "甲", published + heard, *rewrites], final)
    assert commits[:2] == ["", published]
    assert "".join(commits) + tail == published + continuation
    assert (tail if final else commits[-1]) == continuation
    assert (commits[-1] if final else tail) == ""


def _oracle(text: str, end: int, shown: str, heard: str, found: bool = True) -> int:
    # Enumerate literal suffix/prefix intersections, independently of anchor alignment.
    matches = {
        shown[-size:] for size in range(3, len(shown) + 1) if text[end:].startswith(shown[-size:])
    }
    if not found or not matches:
        return end
    run = max(matches, key=len)
    repeated = run + run in text
    continuation = bool(heard) and run.startswith(heard[: len(run)])
    return end if repeated or continuation else end + len(run)


def _boundary(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    end: int,
    shown: str,
    heard: str = "",
    *,
    found: bool = True,
    stable: int | None = None,
) -> StreamingProcessor:
    processor = StreamingProcessor(decode=_scripted(text), buffer_trim_s=64.0)
    processor.emitted = text[:end]
    processor.previous = text if stable is None else text[:stable] + "※"
    # A re-spelled anchor record and the as-published record can legitimately differ.
    monkeypatch.setattr(processor, "shown", shown, raising=False)
    monkeypatch.setattr(processor, "heard", heard, raising=False)
    monkeypatch.setattr(processor, "_anchor", lambda _text, final=False: (end, found))
    return processor


def _publish(processor: StreamingProcessor, final: bool) -> str:
    if not final:
        return _step(processor)
    processor.insert_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    return processor.finish()


@pytest.mark.parametrize("final", [False, True], ids=["process", "finish"])
@pytest.mark.parametrize("where", ["before-anchor", "at-anchor", "after-anchor"])
def test_current_decode_spelling_a_real_repeat_keeps_it(
    monkeypatch: pytest.MonkeyPatch, final: bool, where: str
):
    run = "ごちゃ"
    head = run * 2 + "境" if where == "before-anchor" else "境"
    tail = {
        "before-anchor": run + "続き",
        "at-anchor": run * 2 + "続き",
        "after-anchor": run + "続き" + run * 2,
    }[where]
    processor = _boundary(monkeypatch, head + tail, len(head), "前" + run)
    assert _publish(processor, final) == tail


@pytest.mark.parametrize("final", [False, True], ids=["process", "finish"])
@pytest.mark.parametrize("heard", ["ご", "ごち", "ごちゃ", "ごちゃ続き"])
def test_publishing_decode_hearing_even_one_repeat_character_keeps_it(
    monkeypatch: pytest.MonkeyPatch, final: bool, heard: str
):
    # The later spelling has one copy, so this veto cannot borrow the doubled-text veto.
    processor = _boundary(monkeypatch, "境ごちゃ続き", 1, "前ごちゃ", heard)
    assert _publish(processor, final) == "ごちゃ続き"


@pytest.mark.parametrize("final", [False, True], ids=["process", "finish"])
@pytest.mark.parametrize("heard", ["ご", "ごち", "ごちゃ"])
def test_heard_repeat_survives_respelling_the_first_copy(heard: str, final: bool):
    published = PREFIX + "ごちゃ"
    revised = PREFIX + "ゴチャごちゃ続き"
    commits, tail = _run([published + "甲", published + heard, revised], final)
    assert "".join(commits) + tail == published + "ごちゃ続き"


@pytest.mark.parametrize("final", [False, True], ids=["process", "finish"])
def test_longest_suffix_is_selected_before_applying_repeat_veto(
    monkeypatch: pytest.MonkeyPatch, final: bool
):
    # The 3-character candidate repeats; the 6-character candidate occurs only once.
    processor = _boundary(monkeypatch, "境abcabc続き", 1, "前abcabc")
    assert _publish(processor, final) == "続き"


@pytest.mark.parametrize("final", [False, True], ids=["process", "finish"])
@pytest.mark.parametrize("run", ["", "い", "はい"])
def test_runs_shorter_than_three_are_never_dropped(
    monkeypatch: pytest.MonkeyPatch, final: bool, run: str
):
    processor = _boundary(monkeypatch, "境" + run + "続き", 1, "前" + run)
    assert _publish(processor, final) == run + "続き"


@pytest.mark.parametrize("final", [False, True], ids=["process", "finish"])
@pytest.mark.parametrize("end", [1, 2, 4, 10], ids=["count", "thin", "held", "stops-short"])
def test_not_found_never_uses_the_suffix_guard(
    monkeypatch: pytest.MonkeyPatch, final: bool, end: int
):
    text = "境あいうえおかきく"
    processor = _boundary(monkeypatch, text, end, "前" + text[end:], found=False)
    assert _publish(processor, final) == text[end:]


@pytest.mark.parametrize("final", [False, True], ids=["process", "finish"])
def test_generated_guard_matches_literal_suffix_oracle(
    monkeypatch: pytest.MonkeyPatch, final: bool
):
    rng = random.Random(0xA710)  # noqa: S311 — generated contract cases, fixed seed
    failures = []
    for case in range(256):
        size = [0, 1, 2, 3, 4, 8, 24, 65][case % 8]
        run = "".join(rng.choices("あいうえおかきくけこ", k=size))
        shown = "旧稿" + run
        head = "新稿" + "境" * rng.randrange(1, 5)
        tail = run + ("続き" if case % 7 else "")
        heard = ["", "別", run[:1], run[:2], run, run + "続き"][case % 6]
        if case % 11 == 0:
            head = run * 2 + head
        if case % 13 == 0:
            tail = tail[:-1]
        text, end = head + tail, len(head)
        expected = text[_oracle(text, end, shown, heard) :]
        processor = _boundary(monkeypatch, text, end, shown, heard)
        actual = _publish(processor, final)
        if actual != expected:
            failures.append((case, shown, heard, tail, expected, actual))
    assert not failures, f"{len(failures)}/256 mismatches; first={failures[:3]}"


@pytest.mark.parametrize("stable_extra", [0, 1, 2, 3, 4])
def test_emitted_record_covers_the_guarded_run_and_any_further_agreement(
    monkeypatch: pytest.MonkeyPatch, stable_extra: int
):
    text, end = "境あいう次へ", 1
    processor = _boundary(monkeypatch, text, end, "前あいう", "別", stable=end + stable_extra)
    commit = _step(processor)
    assert commit == text[end + 3 : end + stable_extra]
    assert processor.emitted == text[: end + max(3, stable_extra)]


@pytest.mark.parametrize("stable_extra", [0, 1, 2])
def test_next_decode_cannot_recommit_a_guarded_run_remainder(
    monkeypatch: pytest.MonkeyPatch, stable_extra: int
):
    processor = _boundary(monkeypatch, "境あいう次へ", 1, "前あいう", "別", stable=1 + stable_extra)
    _step(processor)
    # A record stopping inside the run leaves a suffix below the guard's 3-character floor.
    monkeypatch.setattr(processor, "_anchor", StreamingProcessor._anchor.__get__(processor))
    assert _step(processor) == "次へ"


def test_as_published_record_and_last_heard_tail_survive_empty_commits():
    published = PREFIX + "やりやすくっていう"
    processor = StreamingProcessor(
        decode=_scripted(
            published + "甲",
            published + "現状は。",
            PREFIX + "やりやすくて現状は。",
            "",
            published + "現状は。",
            published + "現状は。",
        ),
        buffer_trim_s=64.0,
    )
    assert getattr(processor, "shown", None) == ""
    assert getattr(processor, "heard", None) == ""
    assert _step(processor) == ""
    assert _step(processor) == published
    assert getattr(processor, "shown", None) == published
    assert getattr(processor, "heard", None) == "現状は。"
    for _ in range(3):
        assert _step(processor) == ""
        assert getattr(processor, "shown", None) == published
        assert getattr(processor, "heard", None) == "現状は。"
    assert _step(processor) == "現状は。"
    assert getattr(processor, "shown", None) == published + "現状は。"
    assert getattr(processor, "heard", None) == ""


def _trimmed(monkeypatch: pytest.MonkeyPatch, remaining: str) -> StreamingProcessor:
    cut, tail = "ABCDE", remaining + "XYZ次へ"

    def decode(samples: Audio) -> tuple[str, list[Segment]]:
        text = cut if len(samples) <= SAMPLE_RATE * 0.75 else tail
        return text, [Segment(0.0, len(samples) / SAMPLE_RATE, text)]

    processor = StreamingProcessor(decode=decode, buffer_trim_s=1.5)
    processor.insert_audio(np.zeros(3 * SAMPLE_RATE, dtype=np.float32))
    processor.emitted = cut + remaining
    processor.previous = cut + tail
    monkeypatch.setattr(processor, "shown", "oldXYZ", raising=False)
    monkeypatch.setattr(processor, "heard", "XYZ次へ", raising=False)
    processor._trim([Segment(0.0, 0.75, cut), Segment(0.75, 3.0, tail)])
    assert processor.trims == 1
    assert processor.emitted == remaining
    processor.buffer_trim_s = 64.0
    return processor


@pytest.mark.parametrize("remaining", ["", "U", "UV", "UVW"])
def test_trim_keeps_exactly_the_remaining_record_length(
    monkeypatch: pytest.MonkeyPatch, remaining: str
):
    processor = _trimmed(monkeypatch, remaining)
    expected = "oldXYZ"[-len(remaining) :] if remaining else ""
    assert getattr(processor, "shown", None) == expected
    assert getattr(processor, "heard", None) == "XYZ次へ"


@pytest.mark.parametrize("remaining", ["", "U", "UV"])
def test_trimmed_away_record_cannot_drop_new_speech(
    monkeypatch: pytest.MonkeyPatch, remaining: str
):
    processor = _trimmed(monkeypatch, remaining)
    # No heard veto: test that buffer-local publication evidence alone limits the guard.
    monkeypatch.setattr(processor, "heard", "")
    processor.previous = remaining + "XYZ次へ"
    assert _step(processor) == "XYZ次へ"
    assert processor.finish() == ""


def test_force_trim_clears_both_new_records(monkeypatch: pytest.MonkeyPatch):
    processor = _boundary(monkeypatch, "境あいう次へ", 1, "前あいう", "あいう")
    processor.buffer_trim_s = 1.0
    processor.insert_audio(np.zeros(4 * SAMPLE_RATE, dtype=np.float32))
    processor._force_trim()
    assert processor.forced_trims == 1
    assert getattr(processor, "shown", None) == ""
    assert getattr(processor, "heard", None) == ""
