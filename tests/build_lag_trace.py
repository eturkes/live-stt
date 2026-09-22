#!/usr/bin/env python3
"""Reduce live session transcripts to the committed, text-free lag trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import session_report  # noqa: E402

TRACE = ROOT / "tests" / "lag_sessions.json"
SOURCES = (
    ("2026-09-15T16-48-03", "transcripts/2026-09-15T16-48-03.txt", "JA/EN"),
    ("2026-09-18T12-00-26", "transcripts/2026-09-18T12-00-26.txt", "SRC/TGT"),
)


def _source_root() -> Path:
    """Find transcripts in-tree or in the primary tree around a nested worktree."""
    for root in (ROOT, *ROOT.parents):
        if all((root / relative).is_file() for _, relative, _ in SOURCES):
            return root
    missing = ", ".join(relative for _, relative, _ in SOURCES)
    raise FileNotFoundError(f"missing transcript inputs: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _offset(at: datetime, start: datetime) -> int:
    seconds = (at - start).total_seconds()
    if not seconds.is_integer() or seconds < 0:
        raise ValueError(f"event offset is not a nonnegative integer: {seconds}")
    return int(seconds)


def _reduce(source_root: Path, session_id: str, relative: str, grammar: str) -> dict[str, Any]:
    path = source_root / relative
    session = session_report.read_session(str(path))
    orphan_targets = sorted(set(session.tgt) - set(session.src))
    if orphan_targets:
        raise ValueError(f"{relative}: target without source: {orphan_targets}")

    events = [caption.at for caption in session.src.values()]
    events += [caption.at for caption in session.tgt.values()]
    events += [at for at, _ in session.notes]
    if not events:
        raise ValueError(f"{relative}: no parsed events")
    start = min(events)

    pairs = []
    for n in sorted(session.src):
        source = session.src[n]
        target = session.tgt.get(n)
        pairs.append(
            {
                "n": n,
                "src_s": _offset(source.at, start),
                "tgt_s": None if target is None else _offset(target.at, start),
                "src_chars": len(source.text),
                "tgt_chars": None if target is None else len(target.text),
                "held": source.held,
            }
        )

    return {
        "id": session_id,
        "source_file": relative,
        "source_sha256": _sha256(path),
        "grammar": grammar,
        "start": start.isoformat(timespec="seconds"),
        "captions": len(session.src),
        "notes": [{"at_s": _offset(at, start), "text": text} for at, text in session.notes],
        "pairs": pairs,
    }


def build() -> dict[str, Any]:
    source_root = _source_root()
    return {
        "schema": 1,
        "generator": "tests/build_lag_trace.py",
        "note": "Text-free live-session timing and length reduction for translation-lag analysis.",
        "sessions": [
            _reduce(source_root, session_id, relative, grammar)
            for session_id, relative, grammar in SOURCES
        ],
    }


def _bytes() -> bytes:
    return (json.dumps(build(), indent=2, ensure_ascii=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Require byte-identical regeneration.")
    args = parser.parse_args()

    try:
        rendered = _bytes()
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)

    if args.check:
        if not TRACE.is_file() or TRACE.read_bytes() != rendered:
            print(f"error: {TRACE.relative_to(ROOT)} is not byte-identical", file=sys.stderr)
            sys.exit(1)
        print(f"checked {TRACE.relative_to(ROOT)}")
        return

    TRACE.write_bytes(rendered)
    print(f"wrote {TRACE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
