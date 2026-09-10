"""Locks the law-consistency invariants a tool can decide.

`.agent/spec.md`'s `Deferred` spine and `.agent/deferred.md`'s rows are one list written twice, so
they drift apart silently; and a rank number moves whenever an earlier row dies, which retargets
every `rank N` reference onto a different unit without touching the referring line. Name the row.

Rank and title are checked as ORDERED PAIRS, not as two independent sets — set membership passes a
swap of two titles between their ranks, which is the drift most likely to survive a reading.

`upstream-sync.md`'s override table is re-read against the template after every refresh, which is
only executable if every row sits in that one table and keys on bytes a reader can match: a row that
drifts out of the block goes unchecked, an unquoted prose key cannot be rechecked at all, and a
reworded or deleted clause leaves a quoted key overriding nothing while still reading as live law.
All three are silent in prose and decidable here.

Membership is LITERAL, so this decides that an anchor still occurs, never that it is still its own
row's clause — a phrase surviving elsewhere in the template reads as live, and the length floor only
keeps a key off a word that identifies nothing. The global half of the haystack needs
`~/.claude/CLAUDE.md`, which every session on this machine holds.
"""

from __future__ import annotations

import re
from itertools import takewhile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / ".agent" / "deferred.md"
SPEC = ROOT / ".agent" / "spec.md"
SYNC = ROOT / ".claude" / "rules" / "upstream-sync.md"
TEMPLATE = ROOT / "CLAUDE.md"
GLOBAL_TEMPLATE = Path.home() / ".claude" / "CLAUDE.md"

_QUEUE_ROW = re.compile(r"^(\d+)\. \*\*(.+?)\*\*", re.MULTILINE)
_SPINE_RANK = re.compile(r"\*\*(\d+)\*\*")
_RANK_REFERENCE = re.compile(r"\branks?\s+\d", re.IGNORECASE)
_OVERRIDE_HEADER = "| template clause | repo ruling |"
_OVERRIDE_SEPARATOR = "| --- | --- |"
_ANCHOR = re.compile(r'"([^"]+)"')
# Shortest live anchor is 17 chars. A floor keeps a key from resolving on a word like "rev", which
# occurs all over both templates and identifies no clause.
_ANCHOR_MIN = 12
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


def test_every_override_row_keys_on_a_live_template_anchor():
    lines = SYNC.read_text(encoding="utf-8").splitlines()
    assert _OVERRIDE_HEADER in lines, f"override table must open on {_OVERRIDE_HEADER!r}"
    head = lines.index(_OVERRIDE_HEADER)
    assert lines[head + 1] == _OVERRIDE_SEPARATOR
    rows = list(takewhile(lambda line: line.startswith("|"), lines[head + 2 :]))
    assert rows, "override table parsed no rows"

    # Scoped to the contiguous block: a row below the prose or in a second table escapes both this
    # check and the re-read the file asks for. ANY pipe outside the block counts, since GFM builds a
    # table out of `a|b` with no spaces too ⇒ matching row SHAPE alone leaves that form a hole. This
    # file is prose plus one table, so a pipe wanted in prose later fails loudly instead.
    block = range(head, head + 2 + len(rows))
    strays = [line for i, line in enumerate(lines) if "|" in line and i not in block]
    assert not strays, f"every override row belongs in the one table: {strays}"

    haystack = TEMPLATE.read_text(encoding="utf-8")
    if GLOBAL_TEMPLATE.exists():
        haystack += GLOBAL_TEMPLATE.read_text(encoding="utf-8")
    for row in rows:
        key = row.split("|")[1].strip()
        anchors = [a for a in _ANCHOR.findall(key) if len(a) >= _ANCHOR_MIN]
        assert anchors, f"key needs a quoted anchor of {_ANCHOR_MIN}+ chars, not prose: {key!r}"
        for anchor in anchors:
            assert anchor in haystack, f"override row keys on a dead clause: {anchor!r}"


def test_no_rank_reference_outside_the_queue():
    offenders = [
        f"{path.relative_to(ROOT)}: {hit}"
        for path in _scanned_files()
        for hit in _RANK_REFERENCE.findall(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"name the row, not its moving rank: {offenders}"
