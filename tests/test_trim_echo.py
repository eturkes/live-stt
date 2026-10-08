"""Audio-confirmed trim echoes: retain speech, remove only an unheard pre-cut copy."""

import asyncio
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import live_stt  # noqa: E402
from streaming import ANCHOR_DRIFT, SAMPLE_RATE, Segment, StreamingProcessor  # noqa: E402

Audio = NDArray[np.float32]
Spans = list[tuple[float, float, str]]


@dataclass
class _Result:
    committed: str
    tail: str
    calls: list[Audio]
    processor: StreamingProcessor

    @property
    def text(self) -> str:
        return self.committed + self.tail


def _once(spans: Spans, precut: str, *, samples: int = 9 * SAMPLE_RATE) -> _Result:
    calls: list[Audio] = []
    audio = np.arange(samples, dtype=np.float32)
    text = "".join(t for _, _, t in spans).strip()

    def decode(data: Audio) -> tuple[str, list[Segment]]:
        calls.append(data.copy())
        if len(data) == samples:
            return text, [Segment(a, b, t) for a, b, t in spans]
        return precut, []

    processor = StreamingProcessor(decode=decode, buffer_trim_s=8.0)
    processor.insert_audio(audio)
    committed, _ = processor.process()
    tail = processor.finish()
    assert processor.finish() == ""
    return _Result(committed, tail, calls, processor)


# Short excerpts at the four measured cuts; surrounding session speech stays private.
# (SRC, penultimate prefix, doubled run, pre-cut hypothesis, samples, cut, retained decode)
_LIVE = [
    (
        117,
        "意味でよい、私はベータテスト中なので",
        "開発に協力してくれる方。",
        "意味でよい、私はベータテスト中なので。",
        130816,
        4.32,
        "開発に協力してくれる方。",
    ),
    (
        162,
        "システムにはベータってつけときますじゃないですか",
        "そう認識されないんで。",
        "システムにはベータってつけときます、じゃないですか何から?",
        130816,
        6.64,
        "そう認識されないんで。",
    ),
    (
        166,
        "佐藤さんに、印刷したのを印刷した渡して",
        "一緒に貼ってもらう隣に。",
        "佐藤さんに印刷したのを渡して。",
        140288,
        6.36,
        "一緒に貼ってもらおう隣に。",
    ),
    (
        322,
        "ここですね、ちょこっとこの文を追加しましたけど",
        "えっとチェーン先生がこの。",
        "ここですね、ちょこっとこの文を追加しましたけど。",
        130816,
        5.26,
        "えっと千年先生がこのどういう証拠をと思って",
    ),
]


@pytest.mark.parametrize(
    ("src", "prefix", "run", "precut", "samples", "cut", "retained"),
    _LIVE,
    ids=["L1-src117", "L2-src162", "L3-src166", "L4-src322"],
)
def test_live_trim_publishes_only_the_retained_copy(
    src: int, prefix: str, run: str, precut: str, samples: int, cut: float, retained: str
):
    calls: list[int] = []
    after_cut = False

    def decode(data: Audio) -> tuple[str, list[Segment]]:
        calls.append(len(data))
        if after_cut:
            return retained, [Segment(0.0, len(data) / SAMPLE_RATE, retained)]
        if len(data) != samples:
            return precut, []
        return prefix + run * 2, [
            Segment(0.0, cut, prefix + run),
            Segment(cut, samples / SAMPLE_RATE, run),
        ]

    processor = StreamingProcessor(decode=decode, buffer_trim_s=8.0)
    processor.insert_audio(np.zeros(samples, dtype=np.float32))
    utterance, _ = processor.process()
    boundary = live_stt.settled_boundary(utterance, 0, processor.emitted)
    first = utterance[:boundary]
    after_cut = True
    processor.insert_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    utterance += processor.process()[0] + processor.finish()
    pieces = [first, utterance[boundary:]]
    assert pieces == [prefix, retained], f"SRC {src}: unheard copy reached publication"
    assert calls[:2] == [samples, int(cut * SAMPLE_RATE)]
    assert len(calls) == 3 and processor.trims == 1 and processor.forced_trims == 0


_REAL_REPEATS = [
    ("本当にめっちゃ使うはずなんですよ", "本当にめっちゃ使うはずなんですよ"),
    (
        "これまで登録した人いったやつも1回メール流して",
        "これまで登録した人いったやつも1回メール流して",
    ),
    (
        "のSQLを何か読み込もうとするんですよレンズがそう",
        "のSQLを何か読み込もうとするんですよレンズが。",
    ),
    ("これここでいじることもできるし", "これここでいじることもできるし"),
    ("中で温めるっす。", "中で温める。"),
    ("そうすると2台でV-ラボ", "そうすると2台でV-愛撫を。"),
    (
        "クラウドシッション使わないんだよなスライドを作って",
        "クラウドフィッション使わないんだよなスライドを作って",
    ),
]


@pytest.mark.parametrize(("run", "heard"), _REAL_REPEATS, ids=[f"R1-{i}" for i in range(1, 8)])
def test_real_repeat_keeps_both_copies(run: str, heard: str):
    result = _once([(0.0, 4.0, "前段。" + run), (4.0, 9.0, run)], "前段。" + heard)
    assert result.text == "前段。" + run * 2


@pytest.mark.parametrize("samples", [0, 1, 8 * SAMPLE_RATE - 1, 8 * SAMPLE_RATE])
def test_c1_at_or_below_trim_threshold_never_probes(samples: int):
    result = _once([(0.0, 0.0, "前段確認"), (0.0, 0.0, "確認")], "前段", samples=samples)
    assert len(result.calls) == 1
    assert result.text == "前段確認確認"


@pytest.mark.parametrize("text", ["", "確認確認", "前段確認確認続き"])
def test_c2_one_segment_never_probes(text: str):
    result = _once([(0.0, 9.0, text)], "前段")
    assert len(result.calls) == 1
    assert result.text == text


@pytest.mark.parametrize(
    ("left", "right"), [("前段", "確認"), ("前段あ", "あ続き"), ("", "確認"), ("確認", "")]
)
def test_c3_fewer_than_two_overlapping_characters_never_probes(left: str, right: str):
    result = _once([(0.0, 4.0, left), (4.0, 9.0, right)], "")
    assert len(result.calls) == 1
    assert result.text == left + right


def test_c3_an_overlap_before_the_last_pair_does_not_trigger():
    result = _once([(0.0, 2.0, "前段確認"), (2.0, 4.0, "確認途中"), (4.0, 9.0, "結論")], "")
    assert len(result.calls) == 1
    assert result.text == "前段確認確認途中結論"


@pytest.mark.parametrize("samples", [8 * SAMPLE_RATE + 1, 9 * SAMPLE_RATE])
@pytest.mark.parametrize("cut", [2.00009, 4.32])
@pytest.mark.parametrize("precut", ["", "前段確認"])
def test_d1_trigger_probes_exactly_one_prefix(samples: int, cut: float, precut: str):
    result = _once(
        [(0.0, cut, "前段確認"), (cut, samples / SAMPLE_RATE, "確認")], precut, samples=samples
    )
    assert len(result.calls) == 2, "trigger must make exactly one pre-cut decode"
    np.testing.assert_array_equal(result.calls[1], result.calls[0][: int(cut * SAMPLE_RATE)])


@pytest.mark.parametrize("samples", [8 * SAMPLE_RATE + 1, 9 * SAMPLE_RATE])
def test_c4_a_boundary_at_zero_seconds_decodes_nothing_more(samples: int):
    # _trim cannot cut at 0 s, so no trim could resolve the echo there; the decode stands.
    result = _once(
        [(0.0, 0.0, "前段確認"), (0.0, samples / SAMPLE_RATE, "確認")], "", samples=samples
    )
    assert len(result.calls) == 1
    assert result.text == "前段確認確認"


@pytest.mark.parametrize("run", ["確認", "ああああ", "。」", "確認しました。"])
@pytest.mark.parametrize("prefix", ["", "前段"])
def test_f1_final_process_and_finish_drop_exactly_one_copy(prefix: str, run: str):
    result = _once([(0.0, 4.0, prefix + run), (4.0, 9.0, run + "続き")], prefix)
    assert result.text == prefix + run + "続き"
    assert result.committed == prefix
    assert result.tail == run + "続き"


def test_c2_missing_segments_never_probes():
    result = _once([], "確認確認")
    assert len(result.calls) == 1 and result.text == ""


def test_d1_outer_whitespace_normalization_precedes_the_trigger():
    result = _once([(0.0, 4.0, " 前段確認"), (4.0, 9.0, "確認続き \n")], "前段")
    assert result.text == "前段確認続き"
    assert result.committed == "前段" and result.tail == "確認続き"


def test_d1_echo_is_removed_before_agreement_with_an_earlier_hypothesis():
    raw = "前段確認確認続き"
    calls: list[int] = []

    def decode(data: Audio) -> tuple[str, list[Segment]]:
        calls.append(len(data))
        if len(data) == 4 * SAMPLE_RATE:
            return "前段", []
        if len(data) == 8 * SAMPLE_RATE:
            return raw, [Segment(0.0, 8.0, raw)]
        return raw, [Segment(0.0, 4.0, "前段確認"), Segment(4.0, 9.0, "確認続き")]

    processor = StreamingProcessor(decode=decode, buffer_trim_s=8.0)
    processor.insert_audio(np.zeros(8 * SAMPLE_RATE, dtype=np.float32))
    first = processor.process()[0]
    processor.insert_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    text = first + processor.process()[0] + processor.finish()
    assert text == "前段確認続き"
    assert calls == [8 * SAMPLE_RATE, 9 * SAMPLE_RATE, 4 * SAMPLE_RATE]


def test_d1_retold_head_and_unheard_copy_are_both_removed():
    calls: list[int] = []

    def decode(data: Audio) -> tuple[str, list[Segment]]:
        calls.append(len(data))
        if len(data) == 9 * SAMPLE_RATE:
            return "初段返し本編", [Segment(0.0, 0.75, "初段返し"), Segment(0.75, 9.0, "本編")]
        if len(data) == 4 * SAMPLE_RATE:
            return "返し本編", []
        return "返し本編確認確認", [Segment(0.0, 4.0, "返し本編確認"), Segment(4.0, 8.25, "確認")]

    processor = StreamingProcessor(decode=decode, buffer_trim_s=8.0)
    processor.insert_audio(np.zeros(9 * SAMPLE_RATE, dtype=np.float32))
    first = processor.process()[0]
    text = first + processor.process()[0] + processor.finish()
    assert text == "初段返し本編確認"
    assert calls == [9 * SAMPLE_RATE, int(8.25 * SAMPLE_RATE), 4 * SAMPLE_RATE]


def _distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        following = [i]
        for j, b in enumerate(right, 1):
            following.append(min(row[j] + 1, following[j - 1] + 1, row[j - 1] + (a != b)))
        row = following
    return row[-1]


def _heard(run: str, precut: str) -> bool:
    # Exhaustive substring enumeration is independent of the production alignment primitive.
    tail = precut[-len(run) - ANCHOR_DRIFT :]
    budget = len(run) // 4
    return any(
        _distance(run, tail[start:stop]) <= budget
        for start in range(len(tail) + 1)
        for stop in range(start, len(tail) + 1)
        if abs(stop - start - len(run)) <= budget
    )


@pytest.mark.parametrize(
    ("run", "precut", "keep"),
    [
        ("abcd", "前abcX後", True),
        ("abcd", "前abXd後", True),
        ("abcd", "前abXcd後", True),
        ("abcd", "前acd後", True),
        ("abcd", "前abXY後", False),
        ("abc", "前abX後", False),
        ("ab", "前ab後", True),
        ("ab", "前aX後", False),
        ("確認", "", False),
        ("確認", "確認" + "外" * ANCHOR_DRIFT, True),
        ("確認", "確認" + "外" * (ANCHOR_DRIFT + 1), False),
        ("あいうえおかきく", "あいうえおか", True),
        ("あいうえおかきく", "あいうえお", False),
    ],
)
def test_d1_substring_distance_and_tail_boundaries(run: str, precut: str, keep: bool):
    assert _heard(run, precut) is keep
    result = _once([(0.0, 4.0, "前段" + run), (4.0, 9.0, run + "続き")], precut)
    assert result.text == "前段" + run * (2 if keep else 1) + "続き"


def test_d1_longest_overlap_wins_and_earlier_occurrences_stay():
    run = "あいうあいう"
    result = _once(
        [(0.0, 2.0, run + "先行。"), (2.0, 4.0, "前段" + run), (4.0, 9.0, run + "続き")], "あいう"
    )
    assert result.text == run + "先行。前段" + run + "続き"
    assert result.committed == run + "先行。前段"


@pytest.mark.parametrize("seed", range(6))
def test_d1_generated_inputs_match_substring_oracle(seed: int):
    rng = random.Random(seed)  # noqa: S311 — reproducible generated contract inputs
    for _ in range(30):
        run = "".join(rng.choices("あいうえおかきくけこ", k=rng.randrange(2, 17)))
        observed = list(run)
        for _ in range(rng.randrange(len(run) // 4 + 3)):
            if observed:
                at = rng.randrange(len(observed))
                operation = rng.randrange(3)
                if operation == 0:
                    observed[at] = "異"
                elif operation == 1:
                    observed.pop(at)
                else:
                    observed.insert(at, "異")
        precut = "前段" + "".join(observed) + "外" * rng.randrange(ANCHOR_DRIFT + 3)
        keep = _heard(run, precut)
        result = _once([(0.0, 4.0, "前段" + run), (4.0, 9.0, run + "続き")], precut)
        assert result.text == "前段" + run * (2 if keep else 1) + "続き", (run, precut, keep)
        assert len(result.calls) == 2


class _Vad:
    def __init__(self):
        self.calls = 0
        self.queued = 0

    def accept_waveform(self, _samples: Audio) -> None:
        self.calls += 1
        if self.calls == 4:
            self.queued += 1

    def is_speech_detected(self) -> bool:
        return 0 < self.calls < 4

    def empty(self) -> bool:
        return self.queued == 0

    def pop(self) -> None:
        assert self.queued
        self.queued -= 1


@pytest.mark.parametrize("translate", [False, True], ids=["source-only", "translator"])
def test_w1_worker_publishes_the_run_once(monkeypatch: pytest.MonkeyPatch, translate: bool):
    prefix, run = "ベータテスト中なので", "開発に協力してくれる方。"
    frames = [prefix + "※", prefix + run * 2, run, run]
    calls: list[int] = []
    probes: list[int] = []
    lines: list[tuple[int, str]] = []
    submitted: list[tuple[int, str]] = []

    class Recognizer:
        def decode_segments(
            self, samples: Audio, language: str | None = None
        ) -> tuple[str, list[Segment]]:
            if len(samples) == int(0.75 * SAMPLE_RATE):
                probes.append(len(samples))
                return prefix, []
            i = len(calls)
            calls.append(len(samples))
            assert i < len(frames), "unscripted full-buffer decode"
            text = frames[i]
            end = len(samples) / SAMPLE_RATE
            if i == 1:
                return text, [Segment(0.0, 0.75, prefix + run), Segment(0.75, end, run)]
            return text, [Segment(0.0, end, text)]

    class Translator:
        def submit(self, seq: int, text: str, source: str | None = None) -> None:
            submitted.append((seq, text))

    def capture(tag: str, seq: int, text: str, _file: object, *, held: bool = False) -> None:
        assert tag == "SRC" and not held
        lines.append((seq, text))

    monkeypatch.setattr(live_stt, "VAC_CHUNK_S", 1.0)
    monkeypatch.setattr(live_stt, "VAC_TRIM_S", 1.5)
    monkeypatch.setattr(live_stt, "VAD_PRE_PAD_S", 0.0)
    monkeypatch.setattr(live_stt, "emit_line", capture)
    state = live_stt.State()
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(4):
        queue.put_nowait(np.zeros(SAMPLE_RATE, dtype=np.float32))
    queue.put_nowait(None)
    rec: Any = Recognizer()
    vad: Any = _Vad()
    translator: Any = Translator() if translate else None
    asyncio.run(live_stt.worker(rec, vad, SAMPLE_RATE, queue, state, None, translator=translator))
    assert not state.stopping, "worker failed before publication could be checked"
    assert lines == [(1, prefix), (2, run)]
    assert submitted == (lines if translate else [])
    assert len(calls) == len(frames) and probes == [int(0.75 * SAMPLE_RATE)]
