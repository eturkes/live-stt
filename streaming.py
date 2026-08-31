"""LocalAgreement-2 streaming policy over an offline Whisper pipeline, for Japanese.

Ported from ufal/whisper_streaming (Macháček et al., IJCNLP 2023). The policy is
unchanged — a unit is emitted only once two consecutive decodes of the growing
buffer agree on it, so output is append-only and never rewritten — but two
mechanisms had to be rebuilt because this stack cannot supply what upstream uses.

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

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
HARD_TRIM_S = 28.0  # Whisper's window is 30 s; never let the buffer reach it


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


@dataclass
class Segment:
    start_s: float
    end_s: float
    text: str


@dataclass
class StreamingProcessor:
    """Growing-buffer processor: decode, agree, emit, trim.

    Every decode covers audio from the same buffer start, so two hypotheses are
    directly comparable as strings and no cross-decode stitching is needed. The
    only place a boundary must be located is the trim, which is why the trim rule
    carries the whole correctness burden.
    """

    decode: object  # samples -> (text, [Segment, ...])
    buffer_trim_s: float = 8.0
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    offset_s: float = 0.0
    emitted: str = ""  # text already output for the CURRENT buffer
    previous: str = ""  # previous hypothesis for the current buffer
    forced_trims: int = 0
    trims: int = 0

    def insert_audio(self, chunk: np.ndarray) -> None:
        self.audio = np.concatenate([self.audio, chunk])

    def process(self) -> tuple[str, float | None]:
        text, segments = self.decode(self.audio)
        agreed = common_prefix(text, self.previous)
        self.previous = text
        stable = max(agreed, len(self.emitted))
        buffered_s = len(self.audio) / SAMPLE_RATE
        if buffered_s > self.buffer_trim_s and len(segments) >= 2:
            # A segment that is no longer the last one is final: its audio has
            # stopped growing, so no later decode can extend it and waiting for a
            # second agreement only adds latency. Without this the policy
            # deadlocks -- a lagging commit point cannot trim, the buffer grows,
            # a longer buffer makes agreement slower still, and measured lag ran
            # to 5.62 s median / 25.38 s max, worse than the shipped VAD policy.
            stable = max(stable, sum(len(segment.text) for segment in segments[:-1]))
        commit = text[len(self.emitted) : stable]
        # `emitted` records what was PUBLISHED, so it may only grow inside a buffer.
        # A decode that retracts below it (shorter hypothesis) would otherwise shrink
        # the record, and the next decode would re-commit characters already on
        # screen -- the doubled-character artefact seen in live output.
        if stable <= len(text):
            self.emitted = text[:stable]
        # Absolute audio time the commit reaches. LocalAgreement holds text back
        # until a second decode confirms it, so this trails the buffer end and is
        # the only honest reference point for latency.
        commit_audio_s = self._audio_time_at(segments, stable)
        if commit and buffered_s > self.buffer_trim_s:
            self._trim(segments)
        if len(self.audio) / SAMPLE_RATE > HARD_TRIM_S:
            self._force_trim()
        return commit, commit_audio_s

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
        self.forced_trims += 1

    def finish(self) -> str:
        """Emit the unconfirmed tail; at end of audio there is nothing left to confirm."""
        tail = self.previous[len(self.emitted) :]
        self.emitted = self.previous
        return tail
