"""Proof that the shipped-path suite is not vacuous.

`test_shipped_path.py` passed on its first run, which on its own says nothing: a
test that asserts what the code already spells cannot tell a regression from a
rename. So each production decision the shipped path turns on is mutated in a
throwaway tree and the suite must go red. A surviving mutant means the surface
is described but not defended.

The mutations are behavioural, not cosmetic: every one changes where audio is
sent, what the model is allowed to receive, or which decode loop runs. The
control case reruns the same tree unmutated, because a harness that fails for
its own reasons would score every mutant as killed.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["live_stt.py", "streaming.py"]
SUITES = ["test_shipped_path.py", "test_streaming.py"]

# id -> (surface, exact source text, replacement). Each must appear exactly once
# in live_stt.py; an ambiguous or stale anchor would silently mutate nothing.
MUTATIONS = {
    "default-engine": (
        "the engine a bare `live-stt` run loads",
        'default="whisper",',
        'default="k2v2",',
    ),
    "default-device": (
        "the accelerator the default engine compiles for",
        'ASR_DEVICE = "NPU"',
        'ASR_DEVICE = "CPU"',
    ),
    "hotword-devices": (
        "which devices may receive the session term list",
        'ASR_HOTWORDS_DEVICES = frozenset({"GPU", "CPU"})',
        'ASR_HOTWORDS_DEVICES = frozenset({"GPU", "CPU", "NPU"})',
    ),
    "whisper-marker": (
        "the file that proves the whisper model is present",
        'marker = "openvino_encoder_model.xml"',
        'marker = "openvino_decoder_model.xml"',
    ),
    "duck-dispatch": (
        "the predicate that chooses the streaming policy over VAD segments",
        'hasattr(rec, "decode_segments")',
        'hasattr(rec, "decode")',
    ),
    "engine-choice-order": (
        "the order `--help` lists the selectable engines in",
        "choices=sorted(ENGINE_DIRS),",
        "choices=sorted(ENGINE_DIRS, reverse=True),",
    ),
    "decode-language": (
        "the language every decode is pinned to",
        'language="<|ja|>"',
        'language="<|en|>"',
    ),
    "partial-growth": (
        "whether the status line tracks the utterance or freezes at its first commit",
        "state.partial = utterance",
        "state.partial = state.partial or utterance",
    ),
    "seq-on-publish": (
        "whether a caption number counts captions or utterances",
        "        if utterance:\n            seq += 1\n            emit_line",
        "        seq += 1\n        if utterance:\n            emit_line",
    ),
    "decode-input": (
        "the audio a decode actually sends to the pipeline",
        'cast("Sequence[SupportsFloat]", samples),',
        'cast("Sequence[SupportsFloat]", np.zeros_like(samples)),',
    ),
    "hotword-omission": (
        "sending an empty term list instead of omitting the keyword",
        'keywords = {"hotwords": self.hotwords} if self.hotwords else {}',
        'keywords = {"hotwords": self.hotwords}',
    ),
}


def build(tmp: Path, mutation: str | None) -> None:
    """Copy the smallest tree the suites need, then apply one mutation to it."""
    (tmp / "tests").mkdir()
    for name in SOURCES:
        shutil.copy(ROOT / name, tmp / name)
    for name in SUITES:
        shutil.copy(ROOT / "tests" / name, tmp / "tests" / name)
    # Own testpaths, so the mutant run cannot re-collect this file and recurse.
    (tmp / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    if mutation is None:
        return
    _, old, new = MUTATIONS[mutation]
    source = (tmp / "live_stt.py").read_text()
    assert source.count(old) == 1, f"{mutation}: anchor {old!r} matched {source.count(old)} times"
    (tmp / "live_stt.py").write_text(source.replace(old, new))


def run(tmp: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=tmp,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_unmutated_tree_is_green(tmp_path):
    """Positive control: without it, a broken harness would score every mutant killed."""
    build(tmp_path, None)
    done = run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_changing_a_shipped_decision_breaks_the_suite(mutation, tmp_path):
    build(tmp_path, mutation)
    done = run(tmp_path)
    surface = MUTATIONS[mutation][0]
    assert done.returncode != 0, f"{mutation} survived: nothing defends {surface}\n{done.stdout}"
    # A mutant that only broke collection would be scored killed while defending
    # nothing, so require a real assertion failure and no import/collection error.
    summary = done.stdout.strip().splitlines()[-1]
    assert "failed" in summary, f"{mutation}: no test failed\n{done.stdout}"
    assert "error" not in summary, f"{mutation}: killed by a broken tree\n{done.stdout}"
