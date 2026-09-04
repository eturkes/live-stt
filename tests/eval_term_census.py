#!/usr/bin/env python3
"""Census the shipped path's caption stream for what the learner keys on (M12.1, M12.3).

D-015's `observe_en` keys a learned English spelling on the JA string the
RECOGNISER produced, so the key is a hypothesis, not the name. Two questions
follow and this answers both from `caption_trace.json` alone -- no model, no
accelerator, no audio, under a second -- the way `eval_vac_lag.py` derives
caption lag, so a fresh clone can rerun it:

    uv run python tests/eval_term_census.py [--term 兵十] [--floor 3] [--json]

**How one name is recognised** (M12.1, `--term`). The reference and the captions
are aligned character by character with the shipped scorer (`cer.alignment`, one
DP, one tie order), so each occurrence is located in the hypothesis by the
alignment rather than by searching for forms someone guessed in advance. Each
aligned position is then widened to the `_TERM_RUN` candidate that covers it,
because a candidate is exactly what the learner sees: a form that is not one --
a lone kanji, a run below the length floor -- can never reach support, however
often it recurs. Alignment runs per section, so an occurrence is always matched
inside the audio it was read from.

**What the candidate floor costs** (M12.4, `--floor`). Two filters stand between
a script run and the translator brief, and only the second one costs anything:
the floor decides what is a candidate, `CONTEXT_TERM_SUPPORT` decides which
candidates are ever briefed. So an arm reports both stages -- the forms one floor
admits and the other does not, then the trust episodes each one actually opens.

**Whether a pairing dies** (M12.3, always). A dead pairing is a trusted key that
acquires an English rendering and then stops recurring: its trust lapses, the
rendering is discarded with it, and the translator is left to re-invent a
spelling it was already told. Whether a key stops recurring is a property of the
caption stream alone, so the screen below finds every candidate offline and a
translator arm is needed only to confirm one.

Support is counted by replaying the captions through a real `SessionContext`
with an empty `prompted` set. That set is empty in production too on the shipped
device: `ASR_HOTWORDS_DEVICES` excludes "NPU", so no sighting is ever prompted
and D-015's anti-feedback exclusion discounts nothing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
sys.path[:0] = [str(TESTS), str(ROOT)]

import live_stt  # noqa: E402
from cer import alignment, normalize  # noqa: E402
from live_stt import (  # noqa: E402
    CONTEXT_EN_SUPPORT,
    CONTEXT_TERM_LEASE,
    CONTEXT_TERM_SUPPORT,
    SessionContext,
)

# The candidate rule is read through the module, never bound here, so `--floor`
# moves it in one place and no arm can report one floor's candidates under
# another's trust.

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
    for match in live_stt._TERM_RUN.finditer(text):
        if any(match.start() <= k < match.end() for k in offsets):
            return match.group()
    return None


def _summary(occurrences: list[dict]) -> dict:
    """Fold located occurrences into the per-form view the learner acts on."""
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


def census(term: str, ref_text: str, captions: list[dict]) -> dict:
    """Locate every reference occurrence of `term` in one section's captions."""
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
    return _summary(occurrences)


def story_census(term: str, sections: list[tuple[str, str, list[dict]]]) -> dict:
    """Census `term` across the whole story, one alignment per section."""
    occurrences = []
    for key, ref_text, captions in sections:
        occurrences += [
            {"section": key, **row} for row in census(term, ref_text, captions)["occurrences"]
        ]
    return _summary(occurrences)


def _placeholder(n: int) -> str:
    """A unique capitalized proper noun for the nth term the screen pairs.

    Letters only: `_EN_NAME` stops at a digit, so a numbered placeholder would
    collapse to one shared spelling and let two terms confirm each other.
    """
    letters, k = "", n + 1
    while k:
        k, remainder = divmod(k - 1, 26)
        letters = chr(ord("a") + remainder) + letters
    return "N" + letters


def best_case() -> Callable[[dict, list[str]], str]:
    """English at the learner's BEST case: every opening answered with a proper noun.

    A rendering unobtainable even here is not one a translator could strand, and
    supplying it is also what keeps the gate's state true -- a paired term stops
    blocking its neighbours' openings, which a never-pairing replay would hide.
    """
    names: dict[str, str] = {}

    def supply(caption: dict, unpaired: list[str]) -> str:
        if len(unpaired) != 1:
            return ""  # observe_en's JA-side gate is shut, so any English is a no-op
        names.setdefault(unpaired[0], _placeholder(len(names)))
        return f"the story mentions {names[unpaired[0]]}."

    return supply


def learner(captions: list[dict], english: Callable[[dict, list[str]], str] | None = None) -> dict:
    """Replay the caption stream through SessionContext; report every trust episode.

    An episode is one term's whole trust lifetime -- promoted, sighted, paired,
    expired. That is the unit because `renderings` is discarded with the term it
    belongs to, so an episode is exactly one chance to acquire an English
    spelling and lose it, and a term re-earning support later starts over.

    `english` supplies each caption's English side, given the caption and the
    terms observe_en's JA-side gate leaves unpaired; it defaults to `best_case`.
    M12.5 passes the REAL translator's captions through the same bookkeeping, so
    one definition of an episode serves both the simulation and the live run.
    """
    english = english or best_case()
    context = SessionContext()
    episodes: list[dict] = []
    live: dict[str, dict] = {}
    seq = 0
    for caption in captions:
        text = caption["text"]
        if not text:
            continue  # production observes published utterances only
        seq += 1
        idx = caption["idx"]
        context.observe_ja(text)
        trusted = context.terms()
        for term in trusted:
            if term not in live:
                live[term] = {
                    "term": term,
                    "trusted_at": idx,
                    "n_sightings": 0,
                    "last_sighting": idx,
                    "last_sighting_seq": seq,
                    "openings": [],
                    "paired_at": None,
                    "rendering": None,
                    "sightings_after_paired": 0,
                    "expired_at": None,
                    "mechanism": None,
                }
                episodes.append(live[term])
        for term in [t for t in live if t not in trusted]:
            episode = live.pop(term)
            episode["expired_at"] = idx
            # Only the lease and CONTEXT_MAX_TERMS drop a trusted term, and the
            # two are separable from outside: the lease fires exactly
            # CONTEXT_TERM_LEASE published captions after the last sighting, so
            # anything earlier is the capacity bound evicting the stalest term.
            episode["mechanism"] = (
                "lease" if seq - episode["last_sighting_seq"] >= CONTEXT_TERM_LEASE else "eviction"
            )
        # The lease renews on a CANDIDATE sighting, which is what observe_ja
        # folds in, while observe_en's gate is a plain substring of the caption:
        # a term swallowed by a longer kanji run opens a pairing without
        # renewing its own trust.
        for term in set(live_stt._TERM_RUN.findall(text)) & live.keys():
            live[term]["n_sightings"] += 1
            live[term]["last_sighting"] = idx
            live[term]["last_sighting_seq"] = seq
            if live[term]["paired_at"] is not None:
                live[term]["sightings_after_paired"] += 1
        # observe_en's own gate. A paired term drops out of it, so an episode
        # records at most CONTEXT_EN_SUPPORT openings: the count answers "could a
        # rendering be acquired", never "how much evidence was available".
        unpaired = [t for t in trusted if t in text and t not in context.renderings]
        if len(unpaired) == 1 and unpaired[0] in live:
            live[unpaired[0]]["openings"].append(idx)
        en = english(caption, unpaired)
        if en:
            context.observe_en(text, en)
            for term, episode in live.items():
                if episode["paired_at"] is None and term in context.renderings:
                    episode["paired_at"] = idx
                    episode["rendering"] = context.renderings[term]
    for episode in episodes:
        # Everything the dead-pairing verdict rests on, in the lease's own unit:
        # how much session was left to spend after the key went quiet, and how
        # many chances to pair it had while it was still being said.
        episode["published_after"] = seq - episode["last_sighting_seq"]
        episode["openings_before_quiet"] = sum(
            1 for i in episode["openings"] if i <= episode["last_sighting"]
        )
        episode["dead_pairing"] = (
            episode["paired_at"] is not None and episode["expired_at"] is not None
        )
    return {
        "n_published": seq,
        "lease": CONTEXT_TERM_LEASE,
        "en_support": CONTEXT_EN_SUPPORT,
        "episodes": episodes,
        "trusted_at_end": context.terms(),
        "dead_pairings": [e["term"] for e in episodes if e["dead_pairing"]],
    }


_KATAKANA_FLOOR = re.compile(r"(?<=\[ァ-ヺー\]\{)\d+")


def shipped_floor() -> int:
    """`_TERM_RUN`'s katakana floor, read off the shipped pattern."""
    found = _KATAKANA_FLOOR.findall(live_stt._TERM_RUN.pattern)
    if len(found) != 1:
        raise SystemExit(
            "_TERM_RUN no longer carries exactly one katakana floor, so no arm can move "
            f"it: {live_stt._TERM_RUN.pattern!r}"
        )
    return int(found[0])


@contextmanager
def candidate_floor(floor: int) -> Iterator[None]:
    """Run the block with the candidate rule's katakana floor moved to `floor`.

    Rewriting the shipped pattern, rather than restating it, is what keeps an arm
    differing from production in the floor and in nothing else.
    """
    shipped = live_stt._TERM_RUN
    shipped_floor()  # refuse before rewriting a pattern this cannot locate a floor in
    live_stt._TERM_RUN = re.compile(_KATAKANA_FLOOR.sub(str(floor), shipped.pattern))
    try:
        yield
    finally:
        live_stt._TERM_RUN = shipped


def floor_arm(captions: list[dict], floor: int) -> dict:
    """The shipped katakana floor against `floor`, in candidates and in trust.

    A candidate that never reaches support never enters a brief, so the count
    that decides a floor is the second one: how many of the forms it uniquely
    admits the learner ends up trusting. Shared episodes are compared whole,
    because admitting a term also spends a capacity slot and blocks its
    neighbours' pairing openings while it is unpaired -- interference the
    per-form counts cannot show.
    """
    arms = {}
    for name, value in (("shipped", shipped_floor()), ("alt", floor)):
        with candidate_floor(value):
            forms: dict[str, set[int]] = {}
            for caption in captions:
                for term in set(live_stt._TERM_RUN.findall(caption["text"])):
                    forms.setdefault(term, set()).add(caption["idx"])
            arms[name] = {"floor": value, "forms": forms, "learner": learner(captions)}
    episodes = {n: {e["term"]: e for e in a["learner"]["episodes"]} for n, a in arms.items()}

    def only(name: str, other: str) -> list[dict]:
        return [
            {
                "form": form,
                "captions": sorted(seen),
                "n_captions": len(seen),
                "reaches_support": len(seen) >= CONTEXT_TERM_SUPPORT,
                "trusted": form in episodes[name],
            }
            for form, seen in sorted(
                arms[name]["forms"].items(), key=lambda kv: (-len(kv[1]), kv[0])
            )
            if form not in arms[other]["forms"]
        ]

    shared = episodes["shipped"].keys() & episodes["alt"].keys()

    def behaviour(episode: dict) -> dict:
        # `best_case` names placeholders in first-opening order, so an arm that
        # pairs a different term first spells every later one differently. That
        # is the simulation's counter, not the learner's behaviour.
        return {k: v for k, v in episode.items() if k != "rendering"}

    return {
        "shipped_floor": arms["shipped"]["floor"],
        "alt_floor": floor,
        "only_shipped": only("shipped", "alt"),
        "only_alt": only("alt", "shipped"),
        "n_candidates": {n: len(a["forms"]) for n, a in arms.items()},
        "n_episodes": {n: len(e) for n, e in episodes.items()},
        "dead_pairings": {n: a["learner"]["dead_pairings"] for n, a in arms.items()},
        "evictions": {
            n: [e["term"] for e in a["learner"]["episodes"] if e["mechanism"] == "eviction"]
            for n, a in arms.items()
        },
        "n_shared_episodes": len(shared),
        "shared_episodes_identical": all(
            behaviour(episodes["shipped"][t]) == behaviour(episodes["alt"][t]) for t in shared
        ),
    }


def pinned_sections(manifest: dict, trace: dict) -> list[tuple[str, str, list[dict]]]:
    """Each traced section with its pinned reference text and its own captions.

    Resolving sections through the trace's own records rather than a fixed list
    keeps a trace and its reference from ever coming from different chapters.
    """
    sections = []
    for key, recorded in sorted(trace["sections"].items()):
        section = manifest["sections"].get(key)
        if section is None:
            raise SystemExit(
                f"traced section {key} is not in the corpus manifest\n"
                "rebuild the corpus: uv run --with soundfile python tests/eval_long_form.py"
            )
        pinned = section["build"]["wav_sha256"]
        if recorded["wav_sha256"] != pinned:
            raise SystemExit(
                f"section {key} was traced from a different WAV\n  trace:  "
                f"{recorded['wav_sha256']}\n  pinned: {pinned}\n"
                "rebuild it: tests/build_caption_trace.py"
            )
        captions = [c for c in trace["captions"] if c["section"] == key]
        sections.append((key, section["reference"]["text"], captions))
    return sections


def run(term: str, floor: int | None = None) -> dict:
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sections = pinned_sections(manifest, trace)
    captions = trace["captions"]
    result = {
        "term": term,
        "trace": {
            "sections": [key for key, _, _ in sections],
            "requested_device": trace["run"]["requested_device"],
            "hotwords_reachable": trace["run"]["hotwords_reachable"],
            "audio_s": trace["source"]["audio_s"],
            "n_captions": len(captions),
        },
        "support_threshold": CONTEXT_TERM_SUPPORT,
        **story_census(term, sections),
    }
    result["learner"] = learner(captions)
    # `reaches_support` counts sightings anywhere in the story, while the learner
    # also has to hold a candidate for CONTEXT_TERM_MEMORY segments -- so a form
    # can meet the count and still never be trusted. Only the replay decides.
    trusted_terms = {e["term"] for e in result["learner"]["episodes"]}
    for form in result["forms"]:
        form["trusted"] = form["form"] in trusted_terms
    result["target_forms_trusted"] = [f["form"] for f in result["forms"] if f["trusted"]]
    if floor is not None:
        result["floor"] = floor_arm(captions, floor)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--term", default=DEFAULT_TERM, help=f"Reference term ({DEFAULT_TERM})")
    ap.add_argument("--floor", type=int, help="Compare the shipped katakana floor against this.")
    ap.add_argument("--json", action="store_true", help="Emit the census as JSON.")
    args = ap.parse_args()

    result = run(args.term, args.floor)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    trace, lea = result["trace"], result["learner"]
    print(
        f"census of {result['term']} over {trace['n_captions']} captions "
        f"({lea['n_published']} published) from {len(trace['sections'])} sections, "
        f"{trace['audio_s']:.1f}s on {trace['requested_device']}, "
        f"hotwords_reachable={trace['hotwords_reachable']}"
    )
    print(
        f"  reference occurrences={result['n_occurrences']}  "
        f"as candidate={result['n_recognized_as_candidate']}  dropped={result['n_dropped']}"
    )
    for row in result["occurrences"]:
        where = f"caption {row['caption']}" if row["caption"] is not None else "(dropped)"
        print(
            f"    {row['section']} ref@{row['ref_pos']:>4} {where:>12}  "
            f"form={row['form']}  …{row['context']}…"
        )
    print(f"  distinct forms={len(result['forms'])} (support={result['support_threshold']})")
    for form in result["forms"]:
        if form["trusted"]:
            mark = "TRUSTED"
        elif form["reaches_support"]:
            mark = "sighted enough, never trusted (candidate memory expired between sightings)"
        else:
            mark = "below support"
        print(f"    {form['form']}  captions={form['n_captions']} {form['captions']}  {mark}")
    print(
        f"  target forms trusted={result['target_forms_trusted'] or 'none'}; "
        f"{len(lea['trusted_at_end'])} terms live at the end: {lea['trusted_at_end']}"
    )
    print(
        f"  trust episodes={len(lea['episodes'])} "
        f"(lease={lea['lease']} published captions, pairing needs {lea['en_support']} openings)"
    )
    print(
        f"    {'term':<10} {'trusted@':>8} {'sight':>5} {'last@':>5} {'after':>5} "
        f"{'open':>4} {'pre':>3} {'paired@':>7} {'used':>4} {'expired@':>8}  how"
    )
    for episode in lea["episodes"]:
        print(
            f"    {episode['term']:<10} {episode['trusted_at']:>8} {episode['n_sightings']:>5} "
            f"{episode['last_sighting']:>5} {episode['published_after']:>5} "
            f"{len(episode['openings']):>4} {episode['openings_before_quiet']:>3} "
            f"{str(episode['paired_at']):>7} {episode['sightings_after_paired']:>4} "
            f"{str(episode['expired_at']):>8}  {episode['mechanism'] or 'live at end'}"
            f"{'  DEAD PAIRING' if episode['dead_pairing'] else ''}"
        )
    dead = [e for e in lea["episodes"] if e["dead_pairing"]]
    if not dead:
        print("  dead pairings: NONE — no trusted key acquired a rendering and then lost it")
    else:
        print(f"  dead pairings: {len(dead)}")
        for episode in dead:
            print(
                f"    {episode['term']}  paired@{episode['paired_at']}  "
                f"expired@{episode['expired_at']} ({episode['mechanism']}), "
                f"{episode['sightings_after_paired']} sightings used the rendering, "
                f"{episode['published_after']} published captions after its last sighting"
            )
    if "floor" in result:
        _print_floor(result["floor"])


def _print_floor(arm: dict) -> None:
    print(
        f"  katakana floor {arm['shipped_floor']} (shipped) vs {arm['alt_floor']}: candidates "
        f"{arm['n_candidates']['shipped']} vs {arm['n_candidates']['alt']}, episodes "
        f"{arm['n_episodes']['shipped']} vs {arm['n_episodes']['alt']}, dead pairings "
        f"{len(arm['dead_pairings']['shipped'])} vs {len(arm['dead_pairings']['alt'])}, evictions "
        f"{len(arm['evictions']['shipped'])} vs {len(arm['evictions']['alt'])}"
    )
    for key, floor in (("only_shipped", arm["shipped_floor"]), ("only_alt", arm["alt_floor"])):
        rows = arm[key]
        print(
            f"    admitted by floor {floor} alone: {len(rows)} forms, "
            f"{sum(1 for r in rows if r['reaches_support'])} reach support, "
            f"{sum(1 for r in rows if r['trusted'])} trusted"
        )
        for row in rows:
            if row["trusted"]:
                mark = "TRUSTED"
            elif row["reaches_support"]:
                mark = "sighted enough, never trusted"
            else:
                mark = "below support"
            print(f"      {row['form']}  captions={row['n_captions']} {row['captions']}  {mark}")
    print(
        f"    {arm['n_shared_episodes']} episodes open in both arms; "
        f"identical={arm['shared_episodes_identical']}"
    )


if __name__ == "__main__":
    main()
