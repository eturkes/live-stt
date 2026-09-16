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

`.agent/spec.md`'s `Intent` is the user's alone, so law that quotes it cannot repair its own
citation when a unit renames the thing quoted: the quote goes false while the sentence around it
still reads as the ask. Both texts sit in the tree, so the comparison is literal here.

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
# How law names a queue row: ``.agent/deferred.md` → *Title*`, which prose wrapping can split across
# a line, so match against whitespace-squashed text rather than the raw file.
_QUEUE_LINK = re.compile(r"`\.agent/deferred\.md` → \*([^*]+)\*")
_SPINE_RANK = re.compile(r"\*\*(\d+)\*\*")
_RANK_REFERENCE = re.compile(r"\branks?\s+\d", re.IGNORECASE)
_OVERRIDE_HEADER = "| template clause | repo ruling |"
_OVERRIDE_SEPARATOR = "| --- | --- |"
_ANCHOR = re.compile(r'"([^"]+)"')
# How law cites the user-owned section: a backticked `Intent`. A bare one listing the five section
# names attributes no wording and is not a citation. Everything backticked AFTER the citation in
# that sentence is the quotation — what comes before it names the file the section lives in.
_INTENT_CITATION = "`Intent`"
_FRAGMENT = re.compile(r"`([^`]+)`")
_SENTENCE = re.compile(r"(?<=[.;]) ")
# Shortest live anchor is 17 chars. A floor keeps a key from resolving on a word like "rev", which
# occurs all over both templates and identifies no clause.
_ANCHOR_MIN = 12
# A title runs to the next spine entry or to the sentence that closes the list.
_TITLE_END = {"", "·", "."}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _spine() -> str:
    return SPEC.read_text(encoding="utf-8").split("## Deferred", 1)[1].split("\n## ", 1)[0]


def _intent() -> str:
    # The heading rides the haystack, so a citation naming the section resolves on the section.
    spec = SPEC.read_text(encoding="utf-8")
    return "## Intent" + spec.split("## Intent", 1)[1].split("\n## ", 1)[0]


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


def test_every_queue_link_resolves_to_a_live_row():
    """A row dies at its close; a pointer at it does not, and still reads as live law.

    Naming by rank is already banned (below) because a rank retargets. A TITLE cannot retarget, so
    it fails the other way: `upstream-sync.md` spent this phase naming *Maintenance + security pass*
    after that row closed, describing queued work that no longer existed. Rank references and dead
    titles are the same hole in two directions, and both are decidable.
    """
    rows = _QUEUE_ROW.findall(QUEUE.read_text(encoding="utf-8"))
    live = {_squash(title).rstrip(".") for _, title in rows}
    assert live, "no queue rows parsed"
    dead = [
        f"{path.relative_to(ROOT)}: {title.strip()}"
        for path in _scanned_files()
        for title in _QUEUE_LINK.findall(re.sub(r"\s+", " ", path.read_text(encoding="utf-8")))
        if _squash(title).rstrip(".") not in live
    ]
    assert not dead, f"law points at queue rows that no longer exist: {dead}"


def test_no_rank_reference_outside_the_queue():
    offenders = [
        f"{path.relative_to(ROOT)}: {hit}"
        for path in _scanned_files()
        for hit in _RANK_REFERENCE.findall(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"name the row, not its moving rank: {offenders}"


def test_every_intent_citation_quotes_live_intent_text():
    """A citation of the user-owned `Intent` must quote wording that section actually carries.

    A unit renames what law quotes and rewrites its own prose with it; `Intent` is the user's, so
    the rename stops at its edge and the citation then attributes the new wording to a section that
    never said it. The SRC/TGT rename did that to the lag bullet, inside the sentence forbidding a
    relabel of the ask, and every gate step stayed green because the two texts were never compared.

    The queue joins the scan here: its rows are acceptance contracts and the EN-lag row cites
    `Intent` the same way, while the rank and title locks read it as their source of truth instead.
    """
    intent = _intent()
    misquotes = [
        f"{path.relative_to(ROOT)}: {fragment}"
        for path in [*_scanned_files(), QUEUE]
        for sentence in _SENTENCE.split(re.sub(r"\s+", " ", path.read_text(encoding="utf-8")))
        if _INTENT_CITATION in sentence
        for fragment in _FRAGMENT.findall(sentence.split(_INTENT_CITATION, 1)[1])
        if fragment not in intent
    ]
    assert not misquotes, f"law quotes Intent wording that Intent does not carry: {misquotes}"
