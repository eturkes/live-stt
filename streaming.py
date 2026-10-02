"""LocalAgreement-2 streaming policy over an offline Whisper pipeline, for Japanese.

Ported from ufal/whisper_streaming (Macháček et al., IJCNLP 2023). The policy is
unchanged — a unit is emitted only once two consecutive decodes of the growing
buffer agree on it, so output is append-only and never rewritten — but two
mechanisms had to be rebuilt because this stack cannot supply what upstream uses, and a
third keeps re-spelled hypotheses from moving the published boundary.

1. COMMIT UNIT = CHARACTER, not word. Reference Whisper splits ja/zh/th/lo/my/yue
   on unicode code points rather than spaces (`Tokenizer.split_to_word_tokens`),
   because these languages do not delimit words. openvino.genai applies the space
   rule for every language, so a Japanese sentence comes back as ONE word and its
   word timings are unusable. Characters are the finest honest unit available.

2. TRIM ANCHOR = FULLY-EMITTED SEGMENT. Upstream trims the audio buffer by word
   timestamp. Measured here, neither available anchor survives on its own: the
   same sentence moved from [0.00, 8.80] to [1.00, 9.00] between two decodes, and
   its text gained and lost 。 and 、 while 棲 alternated with 住. So the cut point
   is neither a timestamp nor a text match but the end of the last segment whose
   text the emitted prefix already covers — a point both decodes agree on by
   construction. Everything before it is emitted, nothing after it is, which is
   what makes the cut lossless in both directions.

3. PUBLISHED BOUNDARY = ALIGNED, not counted. A later decode re-spells the published
   prefix (an inserted 、, パック for バック), so its count no longer marks what was shown;
   `_anchor` locates the published tail by edit distance instead. A decode that re-spells
   most of that tail moves the boundary only once a second decode agrees, or at end of audio
   (`_thin`).

Nothing here prompts the model. Feeding recent transcript back as prev-text made
the recogniser loop: CER 1.8919 on the pause-free clip, 2,126 insertions against
1,166 reference characters, with the tail repeating one clause seven times.
Session terms reach the model through `hotwords` instead, which the NPU rejects
outright, so on the default device this policy runs unconditioned.

AlignAtt would be the stronger policy (SimulStreaming, best of IWSLT 2025): it
stops decoding when the last token's most-attended mel frame comes within
`frame_threshold` frames of the buffer end. It needs cross-attention and a forced
decoder prefix, and this pipeline exposes neither.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
# Whisper's window is 30 s. Enforced AFTER a decode, so a catch-up fold can hand it more
# once trims have already failed (asr-pipeline.md).
HARD_TRIM_S = 28.0
# Boundary re-anchoring (StreamingProcessor._anchor): how much published tail is located,
# and how far past the old boundary it may be found above half agreement (_thin searches the
# whole decode).
ANCHOR_TAIL = 24
ANCHOR_DRIFT = 4
# Buffer seconds past buffer_trim_s after which a cut no longer waits for a commit.
ANCHOR_STALL_S = 4.0


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def tail_costs(tail: str, window: str) -> list[int]:
    """cost[j]: edit distance of `tail` against window[:j] -- start pinned, end free."""
    cost = list(range(len(window) + 1))
    for i, char in enumerate(tail, 1):
        row = [i]
        for j, other in enumerate(window, 1):
            row.append(min(cost[j - 1] + (char != other), cost[j] + 1, row[j - 1] + 1))
        cost = row
    return cost


@dataclass
class Segment:
    start_s: float
    end_s: float
    text: str


# `WhisperEngine.decode_segments`; the trim rule needs the spans, not just text.
Decoder = Callable[[np.ndarray], tuple[str, list[Segment]]]


@dataclass
class StreamingProcessor:
    """Growing-buffer processor: decode, agree, emit, trim.

    Every decode covers audio from the same buffer start, so two hypotheses are
    directly comparable as strings and no cross-decode stitching is needed. The
    only place a boundary must be located is the trim, which is why the trim rule
    carries the whole correctness burden.
    """

    decode: Decoder
    buffer_trim_s: float = 8.0
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    offset_s: float = 0.0
    emitted: str = ""  # text already output for the CURRENT buffer
    previous: str = ""  # previous hypothesis for the current buffer
    found_in: str = ""  # decode `emitted` was last LOCATED in; "" after a count fallback
    held: str | None = None  # re-spelled published prefix a thin decode proposed
    forced_trims: int = 0
    trims: int = 0

    def insert_audio(self, chunk: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, chunk])

    def process(self) -> tuple[str, float | None]:
        text, segments = self.decode(self.audio)
        if segments:
            # Whisper chunks carry the whitespace `text` stripped (' Hello', ' '), which
            # put every count one character off the audio it named: a cut after ' ABC'
            # retained D yet counted it trimmed (ABCDDEFG, reviewer-8).
            first, last = segments[0], segments[-1]
            segments = [
                Segment(first.start_s, first.end_s, first.text.lstrip()),
                *segments[1:],
            ]
            segments[-1] = Segment(last.start_s, last.end_s, segments[-1].text.rstrip())
        agreed = common_prefix(text, self.previous)
        published = len(self.emitted)
        anchor, found = self._anchor(text)
        self.previous = text
        stable = max(agreed, anchor)
        buffered_s = len(self.audio) / SAMPLE_RATE
        final_s = 0
        if buffered_s > self.buffer_trim_s and len(segments) >= 2:
            # A segment that is no longer the last one is final: its audio has
            # stopped growing, so no later decode can extend it and waiting for a
            # second agreement only adds latency. Without this the policy
            # deadlocks -- a lagging commit point cannot trim, the buffer grows,
            # a longer buffer makes agreement slower still, and measured lag ran
            # to 5.62 s median / 25.38 s max, worse than the shipped VAD policy.
            # Clamped: a decoder whose spans still outrun the text would otherwise commit
            # past it and leave emitted behind the commit.
            final_s = min(len(text), sum(len(segment.text) for segment in segments[:-1]))
            stable = max(stable, final_s)
        commit = text[anchor:stable]
        # `emitted` records what was PUBLISHED, so it may only grow inside a buffer.
        # A decode that retracts below it (shorter hypothesis) would otherwise shrink
        # the record, and the next decode would re-commit characters already on
        # screen -- the doubled-character artefact seen in live output.
        if stable <= len(text):
            self.emitted = text[:stable]
            # A counted record names no place in `text`, so what follows it there is no
            # evidence for _thin: garbage counted into the record, then a repeated phrase
            # (チンチロリン、チンチロリン) after it, confirmed a wrong end in the scenario harness.
            self.found_in = text if found else ""
        # Absolute audio time the commit reaches. LocalAgreement holds text back
        # until a second decode confirms it, so this trails the buffer end and is
        # the only honest reference point for latency.
        commit_audio_s = self._audio_time_at(segments, stable)
        # A decode that only re-spells published text commits nothing, yet its emitted
        # prefix may already cover a final segment; gating that cut on this commit alone
        # starved _trim into a forced trim at 29 s (reviewer-8) -- the earlier startswith
        # guard's failure mode. So it also cuts wherever count slicing WOULD have
        # committed, which keeps every recorded trim schedule (tests/eval_latency.py).
        # Past the stall bound any available cut is taken, commit or not: the traced
        # buffers never pass 11.25 s, so recorded schedules keep, and a punctuation
        # flip-flop in a published segment can no longer starve the buffer.
        counted = max(agreed, published, final_s) > published
        stalled = buffered_s >= self.buffer_trim_s + ANCHOR_STALL_S
        if (commit or counted or stalled) and stable <= len(text):
            if buffered_s > self.buffer_trim_s:
                self._trim(segments)
        if len(self.audio) / SAMPLE_RATE > HARD_TRIM_S:
            self._force_trim()
        return commit, commit_audio_s

    def _anchor(self, text: str, final: bool = False) -> tuple[int, bool]:
        """Where the published `emitted` ends inside `text`, and whether it was found there.

        A later decode re-spells the published prefix -- an inserted 、, a dropped
        particle, パック for バック -- and slicing it by len(emitted) lands the boundary
        a character early, re-committing a published one (らら, のの: 10 of
        whisper-ja-760M's 21 retention insertions), or late, dropping one. The last
        ANCHOR_TAIL published characters are aligned by edit distance against the text
        from where they should start, the text's end free: the cheapest end is the
        boundary. Longest-block matching locked onto a LATER repeat of a short phrase
        (ああい → あい + new ああい) and swallowed new speech (reviewers 7, 8).
        Past the end of `text` means the decode no longer reaches the published end:
        process() then commits nothing and keeps the record, as before. Not found = the
        old count, which names no place in `text`.
        """
        emitted = self.emitted
        n = len(emitted)
        held, self.held = self.held, None
        if text.startswith(emitted):
            return n, True
        tail = emitted[-ANCHOR_TAIL:]
        start = n - len(tail)
        cost = tail_costs(tail, text[start : n + ANCHOR_DRIFT])
        best = min(cost)
        if 2 * best > len(tail):
            return self._thin(text, tail, held, final)
        # Equal costs are an edit at the very end of the published tail -- dropped,
        # inserted or re-spelled -- which text alone cannot separate. The last decode
        # can: the end after which this text continues as that one did right after the
        # published end is the boundary; else the end nearest the old one. Evidence never
        # outbids cost: repeated short phrases (そうそう) put a confirming continuation
        # after a LATER repeat, and taking it swallowed new speech (reviewer-8).
        ends = [start + j for j, c in enumerate(cost) if c == best]
        follow = self.previous[n : n + 2]
        evidenced = [end for end in ends if follow and text.startswith(follow, end)]
        if evidenced:
            return min(evidenced, key=lambda end: (abs(end - n), -end)), True
        if ends[-1] == len(text) < n:
            return max(n, len(text) + 1), True  # the text stops short of the published end
        return min(ends, key=lambda end: (abs(end - n), -end)), True

    def _thin(self, text: str, tail: str, held: str | None, final: bool) -> tuple[int, bool]:
        """Below half agreement: the boundary moves only once two decodes agree on it.

        Keeping the count lost the speech after a re-spelling of most of the tail
        (ABCDEFGH → XXGH: IJK dropped, reviewer-7). Taking the end the continuation marks
        at once moved the record into a one-decode spelling, so the reversion, head drop
        or total drop after it re-committed published text (total-drop exact output 99.9 →
        50.6 % on the scenario harness, consultants 1 + 2). So that end is HELD for one
        decode, which commits nothing and keeps the record (a HARD_TRIM_S force trim still
        fires), and the next decode adopts it
        by proposing the same re-spelled prefix; anything else keeps the count, which
        self-heals across a blip because every count names the same published length.
        Evidence = what followed the published end in the decode the record was last
        FOUND in, since a held or counted decode's own continuation proves nothing. It
        may sit at any end the tail reaches with at least one matching character
        (cost < max(len(tail), j): a head the decoder restores costs insertions past
        ANCHOR_DRIFT); cost, then nearness, picks among several. finish() adopts
        unconfirmed: no decode is left to confirm it, and the count would drop the tail.
        """
        n = len(self.emitted)
        follow = self.found_in[n : n + 2]
        if not follow:
            return n, False
        start = n - len(tail)
        cost = tail_costs(tail, text[start:])
        seen = [
            start + j
            for j, c in enumerate(cost)
            if c < max(len(tail), j) and text.startswith(follow, start + j)
        ]
        if not seen:
            return n, False
        end = min(seen, key=lambda end: (cost[end - start], abs(end - n), -end))
        if final or held == text[:end]:
            return end, True
        if held is None:
            self.held = text[:end]
            return max(n, len(text) + 1), True  # past the text: no commit, record kept
        return n, False

    def _audio_time_at(self, segments: list[Segment], index: int) -> float | None:
        """Absolute audio time of character `index`, interpolated inside its segment."""
        covered = 0
        for segment in segments:
            length = len(segment.text)
            if covered + length >= index and length > 0:
                share = (index - covered) / length
                within = segment.start_s + share * (segment.end_s - segment.start_s)
                return self.offset_s + within
            covered += length
        return None

    def _trim(self, segments: list[Segment]) -> None:
        """Cut at the end of the last segment wholly covered by emitted text."""
        covered = 0
        cut_s = 0.0
        cut_chars = 0
        for segment in segments:
            covered += len(segment.text)
            if covered > len(self.emitted):
                break
            if segment.end_s > 0:
                cut_s = segment.end_s
                cut_chars = covered
        if cut_s <= 0 or cut_s * SAMPLE_RATE >= len(self.audio):
            return
        self.emitted = self.emitted[cut_chars:]
        self.previous = self.previous[cut_chars:]
        self.found_in = self.found_in[cut_chars:]
        self.audio = self.audio[int(cut_s * SAMPLE_RATE) :]
        self.offset_s += cut_s
        self.trims += 1

    def _force_trim(self) -> None:
        """Last resort when no segment boundary is emitted yet.

        Drops audio whose text may not have been emitted, so it can lose content.
        A nonzero count means the trim rule failed, not that the run merely ran long.
        """
        keep = int(self.buffer_trim_s * SAMPLE_RATE)
        self.offset_s += (len(self.audio) - keep) / SAMPLE_RATE
        self.audio = self.audio[-keep:]
        self.emitted = ""
        self.previous = ""
        self.found_in = ""
        self.held = None
        self.forced_trims += 1

    def finish(self) -> str:
        """Emit the unconfirmed tail; at end of audio there is nothing left to confirm."""
        end, _ = self._anchor(self.previous, final=True)
        tail = self.previous[end:]
        self.emitted = self.previous
        return tail
