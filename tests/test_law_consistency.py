"""Locks the law-consistency invariants a tool can decide.

`.agent/spec.md`'s open `Tasks` rows and `.agent/deferred.md`'s rows are one list written twice, so
they drift apart silently; and a rank number moves whenever an earlier row dies, which retargets
every `rank N` reference onto a different unit without touching the referring line. Name the row.

Rank and title are checked as ORDERED PAIRS, not as two independent sets — set membership passes a
swap of two titles between their ranks, which is the drift most likely to survive a reading.

A repo ruling on a template clause is re-read against the template after every refresh, which is
only executable if every ruling keys on bytes a reader can match: a ruling written without its key
goes unchecked, an unquoted prose key cannot be rechecked at all, and a reworded or deleted clause
leaves a quoted key overriding nothing while still reading as live law. All three are silent in
prose and decidable here.

`.agent/spec.md`'s `Intent` outranks the rest, so law that quotes it must quote what it carries: a
unit that renames the thing quoted and leaves `Intent` behind turns the quote false while the
sentence around it still reads as the ask. Both texts sit in the tree, so the comparison is literal
here.

Membership is LITERAL, so this decides that an anchor still occurs, never that it is still its own
ruling's clause — a phrase surviving elsewhere in the template reads as live, and the length floor
only keeps a key off a word that identifies nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / ".agent" / "deferred.md"
SPEC = ROOT / ".agent" / "spec.md"
RULES = ROOT / ".claude" / "rules"
# The template copies a refresh overwrites whole: rulings key on these alone ⇒ the lock is hermetic.
TEMPLATE_FILES = [ROOT / "CLAUDE.md", *sorted((ROOT / ".claude" / "agents").glob("*.md"))]

_QUEUE_ROW = re.compile(r"^(\d+)\. \*\*(.+?)\*\*", re.MULTILINE)
# How law names a queue row: ``.agent/deferred.md` → *Title*`, which prose wrapping can split across
# a line, so match against whitespace-squashed text rather than the raw file.
_QUEUE_LINK = re.compile(r"`\.agent/deferred\.md` → \*([^*]+)\*")
_RANK_MARK = re.compile(r"\*\*(\d+)\*\*")
# Every checklist item, at any indent, marker or separator. An open row is `- [ ] **N** Title` alone
# on its line; a ticked one is `- [x] <sha> Title` and carries no rank, the queue having renumbered
# once its row died.
_CHECKLIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+\[.\].*$", re.MULTILINE)
_OPEN_ROW = re.compile(r"- \[ \] \*\*(\d+)\*\* (.+)")
_TICKED_ROW = re.compile(r"- \[x\] [0-9a-f]{7,40} \S.*")
# The open items that are no queue rows, one of each at most: the resume note `pause.md` and
# `pause-now.md` write at the head of the open unit, top-level or as its first sub-bullet, and
# `resume.md` deletes at close; and the finish line `route.md` writes at the head of `Tasks` and
# deletes before its last commit, its close-out commits running this lock through the pre-commit
# hook. The statusline reads the resume note to colour the phase; neither carries a rank, so neither
# pairs.
_NOTE_ROW = re.compile(r"[ \t]*- \[ \] (RESUME|ROUTE): \S.*")
_RANK_REFERENCE = re.compile(r"\branks?\s+\d", re.IGNORECASE)
# A ruling = its tag, the quoted clauses it keys on, its verdict. The opener is found loosely and
# parsed strictly, so a near-miss (curly quotes, a missing arrow, an unknown verdict) fails by name
# instead of escaping the check; prose must not restate the opener, since it would parse as one.
_RULING_OPENER = re.compile(r"template[\s-]*ruling", re.IGNORECASE)
_RULING_TAG = re.compile(
    r'Template ruling: ("[^"]+"(?: \+ "[^"]+")*) → \*\*(RETIRED|ADAPTED|INAPPLICABLE)\*\*'
)
# A verdict outside a tag is a ruling with no key.
_VERDICT = re.compile(r"\*\*(?:RETIRED|ADAPTED|INAPPLICABLE)\b[^*]*\*\*")
_ANCHOR = re.compile(r'"([^"]+)"')
# How law cites the section that outranks it: a backticked `Intent`. A bare one listing the five
# section names attributes no wording and is not a citation. Everything backticked AFTER the
# citation in that sentence is the quotation — what comes before it names the file the section
# lives in.
_INTENT_CITATION = "`Intent`"
_FRAGMENT = re.compile(r"`([^`]+)`")
_SENTENCE = re.compile(r"(?<=[.;]) ")
# Shortest live anchor is 17 chars. A floor keeps a key from resolving on a word like "review",
# which occurs all over both templates and identifies no clause.
_ANCHOR_MIN = 12


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _tasks() -> str:
    return SPEC.read_text(encoding="utf-8").split("\n## Tasks\n", 1)[1].split("\n## ", 1)[0]


def _intent() -> str:
    # The heading rides the haystack, so a citation naming the section resolves on the section.
    spec = SPEC.read_text(encoding="utf-8")
    return "## Intent" + spec.split("## Intent", 1)[1].split("\n## ", 1)[0]


def _scanned_files() -> list[Path]:
    return [*sorted((ROOT / ".claude" / "rules").glob("*.md")), SPEC, ROOT / "README.md"]


def test_open_tasks_pair_every_queue_row_with_its_rank():
    queue = _QUEUE_ROW.findall(QUEUE.read_text(encoding="utf-8"))
    assert queue, "no numbered rows parsed out of the queue"
    assert [rank for rank, _ in queue] == [str(n) for n in range(1, len(queue) + 1)]

    tasks = _tasks()
    items = _CHECKLIST_ITEM.findall(tasks)
    notes = [m.group(1) for i in items if (m := _NOTE_ROW.fullmatch(i))]
    assert len(notes) == len(set(notes)), f"one RESUME and one ROUTE note at most: {notes}"
    stray = [
        i
        for i in items
        if not (_OPEN_ROW.fullmatch(i) or _TICKED_ROW.fullmatch(i) or _NOTE_ROW.fullmatch(i))
    ]
    assert not stray, (
        "a Tasks row is `- [ ] **N** Title`, `- [x] <sha> Title`, `- [ ] RESUME: …` or"
        f" `- [ ] ROUTE: …`: {stray}"
    )
    rows = [m for i in items if (m := _OPEN_ROW.fullmatch(i))]
    # Every rank mark in the section must head an open row: one on a ticked row, a sub-bullet or
    # the pointer paragraph would otherwise pair by position with a unit it does not name.
    assert _RANK_MARK.findall(tasks) == [m.group(1) for m in rows]
    assert [m.group(1) for m in rows] == [rank for rank, _ in queue]

    for row, (rank, title) in zip(rows, queue, strict=True):
        assert _squash(row.group(2)).rstrip(".") == _squash(title).rstrip("."), (
            f"Tasks rank {rank} does not name {title!r}: {row.group(2)!r}"
        )


def test_every_template_ruling_keys_on_a_live_template_anchor():
    haystack = re.sub(r"\s+", " ", "\n".join(p.read_text(encoding="utf-8") for p in TEMPLATE_FILES))
    parsed = 0
    for path in sorted(RULES.glob("*.md")):
        name = path.relative_to(ROOT)
        # Squashed: prose wrapping splits a tag, and a quoted clause, across lines.
        text = re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))
        rulings = list(_RULING_TAG.finditer(text))
        starts = {m.start() for m in rulings}
        broken = [
            text[m.start() : m.start() + 80]
            for m in _RULING_OPENER.finditer(text)
            if m.start() not in starts
        ]
        assert not broken, f"{name}: a ruling tag must parse in full: {broken}"
        verdicts = {m.start(2) - 2 for m in rulings}
        strays = [v.group() for v in _VERDICT.finditer(text) if v.start() not in verdicts]
        assert not strays, f"{name}: a verdict outside a ruling tag keys on nothing: {strays}"
        for m in rulings:
            for anchor in _ANCHOR.findall(m.group(1)):
                assert len(anchor) >= _ANCHOR_MIN, (
                    f"{name}: key needs a quoted anchor of {_ANCHOR_MIN}+ chars: {anchor!r}"
                )
                assert anchor in haystack, f"{name}: ruling keys on a dead clause: {anchor!r}"
        parsed += len(rulings)
    assert parsed, "no ruling tags parsed out of .claude/rules/"


def test_every_queue_link_resolves_to_a_live_row():
    """A row dies at its close; a pointer at it does not, and still reads as live law.

    Naming by rank is already banned (below) because a rank retargets. A TITLE cannot retarget, so
    it fails the other way: a pointer naming a closed row describes queued work that no longer
    exists. Rank references and dead titles are the same hole in two directions, and both are
    decidable.
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
    """A citation of `Intent` must quote wording that section actually carries.

    A unit renames what law quotes and rewrites its own prose with it; a rename that stops at the
    edge of `Intent` leaves the citation attributing the new wording to a section that never said
    it. The SRC/TGT rename did that to the lag bullet, inside the sentence forbidding a relabel of
    the ask, and every gate step stayed green because the two texts were never compared.

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
