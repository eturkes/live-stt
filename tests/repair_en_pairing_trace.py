#!/usr/bin/env python3
"""Repair only the pairing sample's source pointer; preserve every measured byte.

    uv run python tests/repair_en_pairing_trace.py [--check]

The frozen turbo raw rows must match the entire ordered turn input before repair.
The historical pointer names the current 760M trace instead. Byte replacement keeps
sampled English, timing, formatting and every other field intact; reruns are idempotent.
"""

import argparse
import json
from pathlib import Path

TESTS = Path(__file__).resolve().parent
TURNS = TESTS / "en_pairing_trace.json"
SOURCE = TESTS / "caption_trace_turbo.json"


def repaired_bytes(path: Path = TURNS, source: Path = SOURCE) -> bytes:
    raw = path.read_bytes()
    trace = json.loads(raw)
    captions = json.loads(source.read_bytes())["captions"]
    expected = [(caption["idx"], caption["text"]) for caption in captions if caption["text"]]
    if [(turn["idx"], turn["ja"]) for turn in trace["turns"]] != expected:
        raise ValueError("pairing turns do not match the frozen ordered source inputs")
    previous = trace["source"]["trace"]
    if previous not in {"caption_trace.json", source.name}:
        raise ValueError(f"unexpected pairing source pointer: {previous}")
    before = f'"trace": "{previous}"'.encode()
    after = f'"trace": "{source.name}"'.encode()
    if raw.count(before) != 1:
        raise ValueError("expected exactly one source pointer in the pairing trace")
    return raw.replace(before, after, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="Refuse an unrepaired pointer.")
    args = parser.parse_args()
    repaired = repaired_bytes()
    if TURNS.read_bytes() == repaired:
        print("pairing source pointer matches the frozen inputs")
        return 0
    if args.check:
        print("pairing source pointer needs repair")
        return 1
    TURNS.write_bytes(repaired)
    print("repaired pairing source pointer; measured bytes preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
