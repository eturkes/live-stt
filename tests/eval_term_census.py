#!/usr/bin/env python3
"""Census one reference name across the shipped path's caption stream (M12.1).

D-015's `observe_en` keys a learned English spelling on the JA string the
RECOGNISER produced, so the key is a hypothesis, not the name. Two failure modes
follow, and both are decided by how the recogniser spells one recurring name:
a **split key** (the recogniser alternates forms, each carrying its own
rendering) and a **dead pairing** (a mis-recognised form pins a correct spelling
to a key that never recurs).

This derives the deciding statistic from `caption_trace.json` alone -- no model,
no accelerator, no audio, under a second -- the way `eval_vac_lag.py` derives
caption lag, so a fresh clone can rerun it:

    uv run python tests/eval_term_census.py [--term 兵十] [--json]

Method. The reference and the captions are aligned character by character with
the shipped scorer (`cer.alignment`, one DP, one tie order), so each occurrence
of the term is located in the hypothesis by the alignment rather than by
searching for forms someone guessed in advance. Each aligned position is then
widened to the `_TERM_RUN` candidate that covers it, because a candidate is
exactly what the learner sees: a form that is not one -- a lone kanji, a run
below the length floor -- can never reach support, however often it recurs.

Support is then counted by replaying the captions through a real `SessionContext`
with an empty `prompted` set. That set is empty in production too on the shipped
device: `ASR_HOTWORDS_DEVICES` excludes "NPU", so no sighting is ever prompted
and D-015's anti-feedback exclusion discounts nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

from cer import alignment, normalize  # noqa: E402
from live_stt import (  # noqa: E402
    _TERM_RUN,
    CONTEXT_EN_SUPPORT,
    CONTEXT_TERM_SUPPORT,
    SessionContext,
)

TRACE = TESTS / "caption_trace.json"
MANIFEST = TESTS / "long_form.json"
DEFAULT_TERM = "兵十"
CONTEXT_CHARS = 6  # raw characters shown either side of an aligned occurrence


def norm_map(text: str) -> tuple[str, list[int]]:
    """normalize(text) plus, per normalized character, its index in `text`.

    Normalizing per character is what makes an aligned position reportable as
    the raw substring a reader can check. NFKC may compose across a character
    boundary, which would break that correspondence, so the reassembly is
    compared against the real normalize() rather than assumed.
    """
    pieces = [normalize(ch) for ch in text]
    joined = "".join(pieces)
    if joined != normalize(text):
        raise SystemExit(
            "per-character normalization diverges from normalize() on this text; "
            "the raw-offset map would be wrong, so the census refuses to report"
        )
    return joined, [k for k, piece in enumerate(pieces) for _ in piece]


def hyp_map(captions: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Normalized caption stream plus, per character, its (caption idx, raw offset)."""
    chunks, provenance = [], []
    for caption in captions:
        norm, src = norm_map(caption["text"])
        chunks.append(norm)
        provenance += [(caption["idx"], k) for k in src]
    return "".join(chunks), provenance


def _candidate_at(text: str, offsets: list[int]) -> str | None:
    """The _TERM_RUN candidate covering any of `offsets`, or None if none does."""
    for match in _TERM_RUN.finditer(text):
        if any(match.start() <= k < match.end() for k in offsets):
            return match.group()
    return None


def census(term: str, ref_text: str, captions: list[dict]) -> dict:
    """Locate every reference occurrence of `term` in the caption stream."""
    ref_norm, _ = norm_map(ref_text)
    hyp_norm, provenance = hyp_map(captions)
    by_idx = {c["idx"]: c["text"] for c in captions}
    # Deletions and insertions carry no counterpart, so dropping them leaves
    # exactly the reference positions that landed somewhere in the hypothesis.
    ref_to_hyp = {i: j for i, j in alignment(ref_norm, hyp_norm) if i is not None and j is not None}

    term_norm = normalize(term)
    occurrences = []
    start = ref_norm.find(term_norm)
    while start != -1:
        landed = [ref_to_hyp[k] for k in range(start, start + len(term_norm)) if k in ref_to_hyp]
        row: dict = {"ref_pos": start, "caption": None, "hyp": "", "form": None, "context": ""}
        if landed:
            hits = [provenance[j] for j in landed]
            # An occurrence can straddle a caption boundary; the caption holding
            # most of it is the one the learner would have seen it in.
            idx = max({i for i, _ in hits}, key=lambda i: sum(1 for j, _ in hits if j == i))
            offsets = sorted(k for i, k in hits if i == idx)
            text = by_idx[idx]
            row["caption"] = idx
            row["hyp"] = text[offsets[0] : offsets[-1] + 1]
            row["form"] = _candidate_at(text, offsets)
            lead = max(0, offsets[0] - CONTEXT_CHARS)
            row["context"] = text[lead : offsets[-1] + 1 + CONTEXT_CHARS]
        occurrences.append(row)
        start = ref_norm.find(term_norm, start + 1)

    forms: dict[str, list[int]] = {}
    for row in occurrences:
        if row["form"] is not None:
            forms.setdefault(row["form"], []).append(row["caption"])
    return {
        "occurrences": occurrences,
        "forms": [
            {
                "form": form,
                "captions": sorted(set(seen)),
                "n_captions": len(set(seen)),
                "reaches_support": len(set(seen)) >= CONTEXT_TERM_SUPPORT,
            }
            for form, seen in sorted(forms.items(), key=lambda kv: -len(set(kv[1])))
        ],
        "n_occurrences": len(occurrences),
        "n_recognized_as_candidate": sum(1 for r in occurrences if r["form"] is not None),
        "n_dropped": sum(1 for r in occurrences if r["caption"] is None),
    }


def learner(captions: list[dict]) -> dict:
    """Replay the caption stream through SessionContext; report what it trusts.

    `openings` counts observe_en's JA-side gate -- captions where exactly one
    trusted term is still unpaired -- because a term with fewer than
    CONTEXT_EN_SUPPORT of them can never acquire a rendering however well it is
    recognised. The real call runs a translator turn later, on a state that can
    only hold more terms, so these are upper bounds.
    """
    context = SessionContext()
    first_trusted: dict[str, int] = {}
    openings: dict[str, list[int]] = {}
    for caption in captions:
        if not caption["text"]:
            continue  # production observes published utterances only
        context.observe_ja(caption["text"])
        for term in context.terms():
            first_trusted.setdefault(term, caption["idx"])
        unpaired = [t for t in context.terms() if t in caption["text"]]
        if len(unpaired) == 1:
            openings.setdefault(unpaired[0], []).append(caption["idx"])
    return {
        "n_published": sum(1 for c in captions if c["text"]),
        "first_trusted": first_trusted,
        "trusted_at_end": context.terms(),
        "en_support": CONTEXT_EN_SUPPORT,
        "openings": openings,
    }


def pinned_section(manifest: dict, wav: str) -> dict:
    """The corpus section a trace was built from, located by its own WAV path.

    Naming the section here rather than fixing one keeps the census bound to
    whichever section the trace records, so a trace and its reference can never
    come from different chapters of the narration.
    """
    for section in manifest["sections"].values():
        if section["build"]["wav"] == wav:
            return section
    raise SystemExit(
        f"trace WAV {wav} is not a pinned corpus section\n"
        "rebuild the corpus: uv run --with soundfile python tests/eval_long_form.py"
    )


def run(term: str) -> dict:
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    section = pinned_section(manifest, trace["source"]["wav"])
    pinned = section["build"]["wav_sha256"]
    if trace["source"]["wav_sha256"] != pinned:
        raise SystemExit(
            f"trace was built from a different WAV\n  trace:  "
            f"{trace['source']['wav_sha256']}\n  pinned: {pinned}\n"
            "rebuild it: tests/build_caption_trace.py"
        )
    captions = trace["captions"]
    result = {
        "term": term,
        "trace": {
            "wav_sha256": trace["source"]["wav_sha256"],
            "requested_device": trace["run"]["requested_device"],
            "hotwords_reachable": trace["run"]["hotwords_reachable"],
            "n_captions": len(captions),
        },
        "support_threshold": CONTEXT_TERM_SUPPORT,
        **census(term, section["reference"]["text"], captions),
    }
    result["learner"] = learner(captions)
    result["target_forms_trusted"] = [
        f["form"] for f in result["forms"] if f["form"] in result["learner"]["first_trusted"]
    ]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--term", default=DEFAULT_TERM, help=f"Reference term ({DEFAULT_TERM})")
    ap.add_argument("--json", action="store_true", help="Emit the census as JSON.")
    args = ap.parse_args()

    result = run(args.term)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    trace, lea = result["trace"], result["learner"]
    print(
        f"census of {result['term']} over {trace['n_captions']} captions "
        f"({lea['n_published']} published) from {trace['requested_device']}, "
        f"hotwords_reachable={trace['hotwords_reachable']}"
    )
    print(
        f"  reference occurrences={result['n_occurrences']}  "
        f"as candidate={result['n_recognized_as_candidate']}  dropped={result['n_dropped']}"
    )
    for row in result["occurrences"]:
        where = f"caption {row['caption']}" if row["caption"] is not None else "(dropped)"
        print(f"    ref@{row['ref_pos']:>4} {where:>12}  form={row['form']}  …{row['context']}…")
    print(f"  distinct forms={len(result['forms'])} (support={result['support_threshold']})")
    for form in result["forms"]:
        mark = "REACHES SUPPORT" if form["reaches_support"] else "below support"
        print(f"    {form['form']}  captions={form['n_captions']} {form['captions']}  {mark}")
    print(
        f"  learner: {len(lea['first_trusted'])} terms trusted over the session "
        f"{lea['first_trusted']}, {len(lea['trusted_at_end'])} live at the end; "
        f"target forms trusted={result['target_forms_trusted'] or 'none'}"
    )
    print(f"  pairing openings (need {lea['en_support']} for a rendering):")
    for term, seen in lea["openings"].items():
        print(f"    {term}  n={len(seen)} {seen}")


if __name__ == "__main__":
    main()
