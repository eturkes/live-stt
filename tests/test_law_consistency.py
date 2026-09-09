"""Locks the two deferral-queue invariants a tool can decide.

`.agent/spec.md`'s `Deferred` spine and `.agent/deferred.md`'s rows are one list written twice, so
they drift apart silently; and a rank number moves whenever an earlier row dies, which retargets
every `rank N` reference onto a different unit without touching the referring line. Name the row.

Rank and title are checked as ORDERED PAIRS, not as two independent sets — set membership passes a
swap of two titles between their ranks, which is the drift most likely to survive a reading.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / ".agent" / "deferred.md"
SPEC = ROOT / ".agent" / "spec.md"

_QUEUE_ROW = re.compile(r"^(\d+)\. \*\*(.+?)\*\*", re.MULTILINE)
_SPINE_RANK = re.compile(r"\*\*(\d+)\*\*")
_RANK_REFERENCE = re.compile(r"\branks?\s+\d", re.IGNORECASE)
# A title runs to the next spine entry or to the sentence that closes the list.
_TITLE_END = {"", "·", "."}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _spine() -> str:
    return SPEC.read_text(encoding="utf-8").split("## Deferred", 1)[1].split("\n## ", 1)[0]


def _scanned_files() -> list[Path]:
    return [*sorted((ROOT / ".claude" / "rules").glob("*.md")), SPEC, ROOT / "README.md"]


def test_spine_pairs_every_queue_row_with_its_rank():
    queue = _QUEUE_ROW.findall(QUEUE.read_text(encoding="utf-8"))
    assert queue, "no numbered rows parsed out of the queue"
    assert [rank for rank, _ in queue] == [str(n) for n in range(1, len(queue) + 1)]

    spine = _spine()
    marks = list(_SPINE_RANK.finditer(spine))
    assert [m.group(1) for m in marks] == [rank for rank, _ in queue]

    for i, (rank, title) in enumerate(queue):
        stop = marks[i + 1].start() if i + 1 < len(marks) else len(spine)
        named = _squash(spine[marks[i].end() : stop])
        wanted = _squash(title).rstrip(".")
        assert named.startswith(wanted), (
            f"spine rank {rank} does not name {title!r}: {named[:60]!r}"
        )
        assert named[len(wanted) :].lstrip()[:1] in _TITLE_END, f"spine rank {rank} runs on"


def test_no_rank_reference_outside_the_queue():
    offenders = [
        f"{path.relative_to(ROOT)}: {hit}"
        for path in _scanned_files()
        for hit in _RANK_REFERENCE.findall(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"name the row, not its moving rank: {offenders}"
