"""LocalAgreement-2 policy tests.

The policy decides what reaches the screen and where audio is cut, so the
properties under test are the two that make it safe: output is append-only, and
a trim never drops text that was not already emitted.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming import SAMPLE_RATE, Segment, StreamingProcessor, common_prefix  # noqa: E402


def audio(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def scripted(*results):
    """Decoder returning each (text, segments) in turn, then repeating the last."""
    calls = list(results)

    def decode(_samples):
        return calls.pop(0) if len(calls) > 1 else calls[0]

    return decode


def segs(*spans):
    return [Segment(a, b, t) for a, b, t in spans]


TRANSCRIPT = "あいうえおかきくけこさしすせそたちつてと"


class Corpus:
    """Decoder over a ground truth of one character per second of audio.

    A fixed-text stub cannot exercise trimming: after a cut the buffer holds
    different audio, so a decoder that ignores the buffer reports text the audio
    no longer contains and every trim looks like a duplication bug. Reading
    `offset_s` and the buffer length instead makes the emitted stream comparable
    against TRANSCRIPT, which is what the lossless property actually claims.
    """

    def __init__(self, processor_ref, segment_chars=4):
        self.ref = processor_ref
        self.segment_chars = segment_chars

    def __call__(self, samples):
        p = self.ref[0]
        start = int(round(p.offset_s))
        stop = min(len(TRANSCRIPT), start + int(round(len(samples) / SAMPLE_RATE)))
        text = TRANSCRIPT[start:stop]
        spans, at = [], 0.0
        for i in range(0, len(text), self.segment_chars):
            piece = text[i : i + self.segment_chars]
            spans.append(Segment(at, at + len(piece), piece))
            at += len(piece)
        return text, spans


def corpus_processor(**kwargs):
    ref: list = [None]
    p = StreamingProcessor(decode=Corpus(ref), **kwargs)
    ref[0] = p
    return p


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("", "", 0), ("abc", "abd", 2), ("abc", "abc", 3), ("", "abc", 0), ("abc", "", 0)],
)
def test_common_prefix(a, b, expected):
    assert common_prefix(a, b) == expected


def test_first_decode_commits_nothing():
    """Nothing has agreed yet, so a lone hypothesis must stay unpublished."""
    p = StreamingProcessor(decode=scripted(("あいうえお", segs((0.0, 1.0, "あいうえお")))))
    p.insert_audio(audio(1.0))
    commit, _ = p.process()
    assert commit == ""


def test_agreement_commits_common_prefix():
    p = StreamingProcessor(
        decode=scripted(
            ("あいうえお", segs((0.0, 1.0, "あいうえお"))),
            ("あいうXX", segs((0.0, 2.0, "あいうXX"))),
        )
    )
    p.insert_audio(audio(1.0))
    assert p.process()[0] == ""
    p.insert_audio(audio(1.0))
    assert p.process()[0] == "あいう"


def test_output_is_append_only_when_hypothesis_retracts():
    """A later decode that disagrees cannot un-emit what already agreed."""
    p = StreamingProcessor(
        decode=scripted(
            ("あいうえお", segs((0.0, 1.0, "あいうえお"))),
            ("あいうえお", segs((0.0, 2.0, "あいうえお"))),
            ("あい", segs((0.0, 3.0, "あい"))),
        )
    )
    for _ in range(2):
        p.insert_audio(audio(1.0))
        p.process()
    assert p.emitted == "あいうえお"
    p.insert_audio(audio(1.0))
    commit, _ = p.process()
    assert commit == ""
    assert p.emitted == "あいうえお"


def test_no_trim_below_threshold():
    p = StreamingProcessor(
        decode=scripted(("あいうえお", segs((0.0, 1.0, "あいうえお")))), buffer_trim_s=8.0
    )
    for _ in range(2):
        p.insert_audio(audio(1.0))
        p.process()
    assert p.trims == 0
    assert p.offset_s == 0.0
    assert len(p.audio) == 2 * SAMPLE_RATE


def test_trim_cuts_at_fully_emitted_segment_and_advances_offset():
    """The cut lands at the end of the last segment the emitted text covers."""
    p = corpus_processor(buffer_trim_s=8.0)
    p.insert_audio(audio(9.0))
    commit, _ = p.process()
    assert commit == "あいうえおかきく"  # every segment but the last is final
    assert p.trims == 1
    assert p.offset_s == pytest.approx(8.0)  # end of the last fully covered segment
    assert len(p.audio) == pytest.approx(1.0 * SAMPLE_RATE, abs=1)


def test_trim_is_lossless_across_repeated_cuts():
    """Concatenated commits reconstruct the reference exactly: no loss, no repeat."""
    p = corpus_processor(buffer_trim_s=8.0)
    seen = ""
    for _ in range(len(TRANSCRIPT)):
        p.insert_audio(audio(1.0))
        seen += p.process()[0]
    seen += p.finish()
    assert seen == TRANSCRIPT
    assert p.forced_trims == 0


def test_force_trim_past_hard_limit_counts_and_shrinks():
    p = StreamingProcessor(
        decode=scripted(("あ", segs((0.0, 1.0, "あ")))), buffer_trim_s=8.0
    )
    p.insert_audio(audio(29.0))
    p.process()
    assert p.forced_trims == 1
    assert len(p.audio) == 8 * SAMPLE_RATE
    assert p.offset_s == pytest.approx(21.0)


def test_finish_flushes_tail_once():
    p = StreamingProcessor(decode=scripted(("あいうえお", segs((0.0, 1.0, "あいうえお")))))
    p.insert_audio(audio(1.0))
    p.process()
    assert p.finish() == "あいうえお"
    assert p.finish() == ""


def test_audio_time_interpolates_inside_segment_and_adds_offset():
    p = StreamingProcessor(decode=scripted(("", [])))
    p.offset_s = 10.0
    spans = segs((0.0, 4.0, "あいうえ"))
    assert p._audio_time_at(spans, 2) == pytest.approx(12.0)
    assert p._audio_time_at(spans, 4) == pytest.approx(14.0)


def test_audio_time_is_none_past_the_last_segment():
    p = StreamingProcessor(decode=scripted(("", [])))
    assert p._audio_time_at(segs((0.0, 1.0, "あ")), 5) is None


def test_missing_segments_disable_trimming_but_not_commits():
    """Without spans there is no anchor, so the policy must emit and keep the audio."""
    p = StreamingProcessor(decode=scripted(("あいうえお", [])), buffer_trim_s=8.0)
    p.insert_audio(audio(9.0))
    p.process()
    commit, when = p.process()
    assert commit == "あいうえお"
    assert when is None
    assert p.trims == 0
