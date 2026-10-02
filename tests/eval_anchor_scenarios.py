#!/usr/bin/env python3
"""Scripted decode runs through StreamingProcessor: exact output per re-spelling scenario.

Each script grows one utterance from the committed narration captions by 2-6 characters per
decode, then disturbs it the way Whisper re-decodes a buffer: re-spells part of the published
text (persistently, for one decode or two), drops its head, drops everything published, or
returns unrelated text. The truth is the utterance itself: no re-spelling or drop touches the
text past the published end, so a correct processor outputs it exactly; the garbage scenarios
replace that text too, so their exact rate is no processor's to win. `--baseline` replays the
same scripts through another streaming.py and counts the scripts only one of them gets exact.
Report-only: the shipped `_thin` loses a few scripts the old count wins (seed 11: 0; seed 0: 1;
`--scripts 30000 --seed 7`: 18 of 315,805 against ~64,000 the other way, user-accepted).
Synthetic by construction: it ranks two processors, and its rates describe no live session.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CAPTIONS = ROOT / "tests" / "caption_trace.json"
SCENARIOS = (
    "persist",  # the published region keeps its re-spelling to the end
    "flip1",  # re-spelled for one decode, then the original spelling returns
    "flip2",  # re-spelled for two decodes
    "headdrop1",  # the head of the published text missing for one decode
    "headdrop2",
    "totaldrop1",  # everything published missing for one decode
    "garbage1",  # one decode of unrelated text
    "lastflip",  # the same disturbances on the FINAL decode, which finish() flushes
    "lasthead",
    "lasttotal",
    "lastgarbage",
)


def load(path: Path) -> ModuleType:
    name = f"streaming_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass resolves its module by name
    spec.loader.exec_module(module)
    return module


def respell(text: str, rng: random.Random, alphabet: list[str]) -> str:
    """Replace random spans with spans of other characters, 2 shorter to 2 longer."""
    out, i = [], 0
    while i < len(text):
        if rng.random() < 0.6:
            span = rng.randint(1, 3)
            width = max(1, span + rng.choice([-2, -1, 0, 1, 1, 2]))
            out.append("".join(rng.choice(alphabet) for _ in range(width)))
            i += span
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def script(
    kind: str, rng: random.Random, corpus: str, alphabet: list[str]
) -> tuple[str, list[str]] | None:
    length = rng.randint(14, 44)
    at = rng.randrange(len(corpus) - length - 1)
    truth = corpus[at : at + length]
    lengths, reach = [], rng.randint(3, 8)
    while reach < length:
        lengths.append(reach)
        reach += rng.randint(2, 6)
    lengths += [length, length]
    if len(lengths) < 5:
        return None
    texts = [truth[:reach] for reach in lengths]
    k = rng.randrange(2, len(texts) - 2)
    published = lengths[k - 2]  # what decodes k-2 and k-1 agreed on
    if published < 2:
        return None
    start = max(0, published - rng.randint(2, 24))
    region = respell(truth[start:published], random.Random(rng.random()), alphabet)  # noqa: S311
    head = rng.randint(2, max(2, min(8, published - 1)))

    def disturbed(reach: int, how: str) -> str:
        if how == "flip":
            return truth[:start] + region + truth[published:reach]
        if how == "head":
            return truth[head:reach]
        if how == "total":
            return truth[published:reach]
        other = rng.randrange(len(corpus) - 60)
        return corpus[other : other + rng.randint(3, reach + 6)]

    if kind.startswith("last"):
        texts[-1] = disturbed(lengths[-1], kind[4:])
        return truth, texts
    how = {"persist": "flip", "flip": "flip", "headdrop": "head", "totaldrop": "total"}
    span = len(texts) if kind == "persist" else int(kind[-1])
    for step in range(k, min(len(texts), k + span)):
        texts[step] = disturbed(lengths[step], how.get(kind.rstrip("12"), "garbage"))
    return truth, texts


def run(module: ModuleType, texts: list[str]) -> str:
    decodes = iter(texts)
    processor = module.StreamingProcessor(
        decode=lambda _audio: (next(decodes), []), buffer_trim_s=64.0
    )
    processor.insert_audio(np.zeros(module.SAMPLE_RATE, dtype=np.float32))
    return "".join(processor.process()[0] for _ in texts) + processor.finish()


def verdict(output: str, truth: str) -> str:
    if output == truth:
        return "exact"
    return "dup" if len(output) > len(truth) else "loss" if len(output) < len(truth) else "subst"


def evaluate(scripts: int, seed: int, baseline: Path | None) -> dict[str, dict[str, int]]:
    captions = json.loads(CAPTIONS.read_text(encoding="utf-8"))["captions"]
    corpus = "".join(caption["text"] for caption in captions)
    alphabet = sorted(set(corpus) - set("、。"))
    current = load(ROOT / "streaming.py")
    other = load(baseline) if baseline else None
    rng = random.Random(seed)  # noqa: S311
    report: dict[str, dict[str, int]] = {}
    for kind in SCENARIOS:
        tally: Counter[str] = Counter()
        for _ in range(scripts):
            drawn = script(kind, rng, corpus, alphabet)
            if drawn is None:
                continue
            truth, texts = drawn
            tally["scripts"] += 1
            mine = verdict(run(current, texts), truth)
            tally[mine] += 1
            if other is not None:
                theirs = verdict(run(other, texts), truth)
                tally[f"baseline_{theirs}"] += 1
                tally["baseline_only_exact"] += theirs == "exact" != mine
                tally["current_only_exact"] += mine == "exact" != theirs
        report[kind] = dict(tally)
    return report


def render(report: dict[str, dict[str, int]]) -> str:
    lines = []
    for kind, tally in report.items():
        n = tally["scripts"]
        line = f"{kind:12s} n={n:5d} exact {tally.get('exact', 0) / n:6.1%}"
        line += "".join(f" {key} {tally.get(key, 0):5d}" for key in ("dup", "loss", "subst"))
        if "baseline_only_exact" in tally:
            line += f" | baseline exact {tally.get('baseline_exact', 0) / n:6.1%}"
            line += f" baseline-only {tally['baseline_only_exact']:4d}"
            line += f" current-only {tally['current_only_exact']:4d}"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--scripts", type=int, default=3000, help="scripts drawn per scenario")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--baseline", type=Path, help="another streaming.py, e.g. `git show REV:streaming.py`"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.scripts, args.seed, args.baseline)
    print(json.dumps(report, indent=2) if args.json else render(report))


if __name__ == "__main__":
    main()
