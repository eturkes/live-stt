#!/usr/bin/env python3
"""Re-derive what a live session did, from the files that session already wrote.

A live run leaves two artifacts: `transcripts/<start>.txt` (numbered JA/EN lines
plus `--` markers) and whatever stderr was redirected to. Everything else --
the meter, the scrollback, the counters -- dies with the terminal. So the
questions a soak actually has to answer (`live-smoke.md`) were answerable only by
hand-reading, which is how a 47-turn JA-only tail sat in one session for weeks
before anyone named its cause.

This reads those files and answers them mechanically. It needs no hardware, no
weights and no network, and it imports the SHIPPED screen rather than restating
it, so a threshold change moves this report with it.

    uv run python session_report.py [TRANSCRIPT ...] [--log FILE ...] [--json]

The screen sweep rebinds `live_stt.CAPTION_REPEAT_UNIT_CHARS` around the real
`repeat_span`, the same way `tests/eval_term_census.py --floor` rewrites the
shipped `_TERM_RUN`: one implementation, measured at several settings.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime

import live_stt as app

# `[2026-09-04T14:38:41+09:00] JA 263: text` and the unnumbered `-- note` form.
_EVENT = re.compile(r"^\[([^\]]+)\] (JA|EN) (\d+): (.*)$")
_NOTE = re.compile(r"^\[([^\]]+)\] -- (.*)$")
# `[2026-09-04 14:38:49,538] ERROR message`; the comma is logging's msec sep.
_LOG = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\] (\w+) (.*)$")

_DECLINED = re.compile(r"^caption (\d+) not translated: (.*)$")
_BLOCK_FAIL = re.compile(r"^translation failed \((.*)\); JA-only for this block$")
# The prefix is optional: sessions predating the `_disable` rewrite logged the
# bare reason, so a corpus spans both spellings of the same event.
_DISABLED = re.compile(r"^(?:translation disabled: )?(.*?); JA-only for the rest of the session$")
_RESTORED = re.compile(r"^translation restored: (.*)$")
_PEAK = re.compile(r"^backlog peak:(.*)$")
# `session: <transcript path>`, the run's own start, or `not saved (--no-save)`.
_SESSION = re.compile(r"^session: (.*)$")
# `caption dropped (<defect>): <=24 chars>…`. Non-greedy, so the split lands on
# the FIRST `): ` -- a defect string may carry a bare `)` and a caption may carry
# the whole separator, and only this anchor survives both.
_DROPPED = re.compile(r"^caption dropped \((.*?)\): (.*)$")

# Why a published caption never got an EN line. Ordered by how much the evidence
# pins it down: a logged decline is certain, a trailing gap is an inference.
DECLINED = "declined"  # the caption's own text; the screen refused it
STRIKE = "strike"  # a failure inside the run that disabled the leg
DISABLED = "disabled"  # after a permanent degrade, so JA-only by design
SHUTDOWN = "shutdown"  # last caption of the session; the drain did not finish
FAILED = "failed"  # isolated transient failure, leg survived


@dataclass
class Caption:
    n: int
    at: datetime
    text: str


@dataclass
class LogEvent:
    at: datetime
    level: str
    body: str


@dataclass
class Session:
    path: str
    ja: dict[int, Caption] = field(default_factory=dict)
    en: dict[int, Caption] = field(default_factory=dict)
    notes: list[tuple[datetime, str]] = field(default_factory=list)
    events: list[LogEvent] = field(default_factory=list)
    marker_start: datetime | None = None  # `session:` in the log, set by attach_logs

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def start(self) -> datetime | None:
        """When the run began -- its own log marker, else the filename.

        `live_stt` logs `session: <path>` at startup whenever stderr is redirected,
        and that is the only exact answer. Without it the filename still covers the
        startup window where a degrade can already fire, because a transcript is
        named for its start time -- but `-o PATH` names the file freely, so an
        unmarked log falls back to the earliest line and understates the start by
        exactly that window.
        """
        if self.marker_start:
            return self.marker_start
        try:
            return datetime.strptime(self.name.removesuffix(".txt"), "%Y-%m-%dT%H-%M-%S")
        except ValueError:
            times = [c.at for c in self.ja.values()] + [c.at for c in self.en.values()]
            return min(times) if times else None


def read_session(path: str) -> Session:
    s = Session(path=path)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if m := _EVENT.match(line):
                at = datetime.fromisoformat(m.group(1)).replace(tzinfo=None)
                cap = Caption(int(m.group(3)), at, m.group(4))
                (s.ja if m.group(2) == "JA" else s.en)[cap.n] = cap
            elif m := _NOTE.match(line):
                s.notes.append(
                    (datetime.fromisoformat(m.group(1)).replace(tzinfo=None), m.group(2))
                )
    return s


def read_log(path: str) -> list[LogEvent]:
    """Parse one redirected stderr log. Unparseable lines are native noise.

    sherpa's `circular-buffer.cc … Overflow!` is written by C++ straight to the
    stream and carries no logging prefix; it is expected (>60 s of unbroken
    speech) and not an event.
    """
    out: list[LogEvent] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if m := _LOG.match(line.rstrip("\n")):
                at = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                out.append(LogEvent(at, m.group(2), m.group(3)))
    return out


def attach_logs(sessions: list[Session], events: list[LogEvent]) -> list[LogEvent]:
    """Give each event to the session that was running when it was written.

    Runs are sequential, so a session owns wall-clock time from its own start
    until the next one starts. Bounding ownership by the last CAPTION instead
    loses exactly the events that matter most: `codex app-server exited` fires
    after the final caption by construction, and one live session's whole cause
    of death sat 1 s past the end of its own span.

    A `session: <path>` marker names the run that wrote it, so a marked log opens
    the window at the real process start rather than at the first caption -- the
    startup drops `-o PATH` sessions lost. A marker naming a transcript this
    report was not given -- `--no-save`, or a file the user did not pass -- opens
    a window owned by NO session, so its events stay unclaimed instead of landing
    on the run before it.
    """
    by_name = {s.name: s for s in sessions}
    marked: list[tuple[datetime, Session | None]] = []
    for e in events:
        if m := _SESSION.match(e.body):
            owner = by_name.get(m.group(1).rsplit("/", 1)[-1])
            if owner:
                owner.marker_start = e.at
            marked.append((e.at, owner))
    claimed = {id(s) for _, s in marked if s}
    started = sorted(
        marked + [(s.start, s) for s in sessions if s.start and id(s) not in claimed],
        key=lambda p: p[0],
    )
    unclaimed = []
    for e in events:
        owner = None
        for i, (begin, s) in enumerate(started):
            nxt = started[i + 1][0] if i + 1 < len(started) else None
            if begin <= e.at and (nxt is None or e.at < nxt):
                owner = s
        if owner:
            owner.events.append(e)
        else:
            unclaimed.append(e)
    return unclaimed


def repeat_span_at(text: str, bound: int) -> int:
    """The shipped `repeat_span` measured at one unit bound."""
    original = app.CAPTION_REPEAT_UNIT_CHARS
    app.CAPTION_REPEAT_UNIT_CHARS = bound
    try:
        return app.repeat_span(text)
    finally:
        app.CAPTION_REPEAT_UNIT_CHARS = original


def find_degrade(s: Session) -> dict | None:
    """When the EN leg died for the rest of the session, and on which mechanism.

    Three sources, strongest first. A `--` marker is the shipped record and names
    its own reason. A log line is nearly as good. Absent both -- every session
    saved before the marker existed -- the transcript still shows it: EN stops
    and never resumes, with more captions behind it than the strike budget.
    """

    def framed(at: datetime, reason: str, source: str) -> dict:
        # Only the strike path burns captions on the way down; an EOF kills the
        # leg between turns, so nothing before it is a strike.
        return {
            "at": at,
            "reason": reason,
            "source": source,
            "three_strike": "consecutive failures" in reason,
        }

    for at, note in s.notes:
        if note.startswith("translation disabled: "):
            return framed(at, note.split(": ", 1)[1], "marker")
    for e in s.events:
        if m := _DISABLED.match(e.body):
            return framed(e.at, m.group(1), "log")
    if not s.en:
        return None
    last_en = max(s.en)
    trailing = sorted(n for n in s.ja if n > last_en)
    if len(trailing) < app.TRANSLATE_MAX_FAILURES:
        return None  # too short to be the 3-strike path; read as shutdown loss
    strike_at = s.ja[trailing[app.TRANSLATE_MAX_FAILURES - 1]].at
    return framed(strike_at, f"{app.TRANSLATE_MAX_FAILURES} consecutive failures", "inferred")


def find_restores(s: Session) -> list[dict]:
    out = [
        {"at": at, "reason": n.split(": ", 1)[1], "source": "marker"}
        for at, n in s.notes
        if n.startswith("translation restored: ")
    ]
    out += [
        {"at": e.at, "reason": m.group(1), "source": "log"}
        for e in s.events
        if (m := _RESTORED.match(e.body))
    ]
    return sorted(out, key=lambda d: d["at"])


def _counter(peak: str, name: str) -> int:
    """One counter off a rendered `_backlog` string; absent means zero.

    `_backlog` renders a counter only when it is nonzero, so a session that has
    dropped nothing shows no `drop=` at all. The word boundary is what keeps
    `drop` off `tdrop=` and `skip` off `tskip=`, which count different things.
    """
    m = re.search(rf"\b{name}=(\d+)", peak)
    return int(m.group(1)) if m else 0


def _at_caption(c: Caption | None) -> dict | None:
    return None if c is None else {"n": c.n, "at": c.at.isoformat(sep=" "), "text": c.text[:40]}


def attribute_drops(s: Session) -> list[dict]:
    """What surrounded each `backlog peak:` drop increase.

    `drop=` counts backend-sized callback BLOCKS of captured audio, never samples
    and never speech, so every figure here is blocks. The meter re-logs the peak
    whenever the rendered string CHANGES, for any counter, so most peak lines
    carry no drop increase and produce no row.

    A drop means ingestion fell behind, and the live evidence says the correlate
    is a SCREENED caption rather than a long utterance (`asr-pipeline.md`): 16 of
    the 20 publication gaps over 20 s in the one captured session dropped nothing.
    So each increase is placed against the captions that bracket it, the gap
    between them, and the `caption dropped (…)` lines inside that gap -- which is
    the only record a screened caption leaves, having no number and no transcript
    line. Separating the repetition arm from the latin arm is the whole point:
    only the first costs more than real time.

    A drop BEFORE the first caption reaches this function whenever the log carries
    the run's own `session: <path>` marker, which is every capture a current
    live-stt writes. Coverage limit: a log predating that marker still starts the
    session at its filename, so under `-o PATH` -- a free name, no time in it --
    those events stay unclaimed there.
    """
    peaks = [(e.at, m.group(1)) for e in s.events if (m := _PEAK.match(e.body))]
    screened = [
        # `removesuffix`, not `rstrip`: the logger appends exactly one ellipsis
        # after a 24-char head, so a head that already ended in one logs `……`
        # and `rstrip` would eat the caption's own character too.
        (e.at, m.group(1), m.group(2).removesuffix("…"))
        for e in s.events
        if (m := _DROPPED.match(e.body))
    ]
    captions = sorted(s.ja.values(), key=lambda c: (c.at, c.n))
    last_event = max((e.at for e in s.events), default=None)

    rows: list[dict] = []
    drop = skip = 0
    for at, body in peaks:
        now, skip_now = _counter(body, "drop"), _counter(body, "skip")
        if now > drop:
            before = next((c for c in reversed(captions) if c.at <= at), None)
            after = next((c for c in captions if c.at > at), None)
            # The window is the publication gap the increase sits in. Open at
            # either end it runs to the session's own edge, because a drop before
            # the first caption or after the last one still has screened captions
            # to answer for.
            lo = before.at if before else (s.start or at)
            hi = after.at if after else max(at, last_event or at)
            rows.append(
                {
                    "at": at.isoformat(sep=" "),
                    "drop": now,
                    "delta": now - drop,
                    "skip_delta": skip_now - skip,
                    "prev": _at_caption(before),
                    "next": _at_caption(after),
                    "gap_s": (
                        None if not (before and after) else (after.at - before.at).total_seconds()
                    ),
                    "screened": [
                        {"at": w.isoformat(sep=" "), "defect": d, "text": t}
                        for w, d, t in screened
                        if lo <= w <= hi
                    ],
                }
            )
        drop, skip = now, skip_now
    return rows


def explain_missing(s: Session) -> list[dict]:
    """Why each published caption has no EN line."""
    logged = {int(m.group(1)): m.group(2) for e in s.events if (m := _DECLINED.match(e.body))}
    degrade = find_degrade(s)
    last_en = max(s.en) if s.en else 0
    last_ja = max(s.ja) if s.ja else 0
    strikes = set()
    if degrade and degrade["three_strike"]:
        after = sorted(n for n in s.ja if n > last_en)
        strikes = set(after[: app.TRANSLATE_MAX_FAILURES])

    # `translation failed (<type>)` names the exception a turn died on. Second
    # resolution, so it is matched to the caption it followed, not to an exact
    # timestamp: the EN would have landed after its JA line, never before.
    failures = [
        (e.at, m.group(1) or "TimeoutError") for e in s.events if (m := _BLOCK_FAIL.match(e.body))
    ]

    def failure_near(n: int) -> str | None:
        after = s.ja[n].at
        nxt = min((s.ja[k].at for k in s.ja if k > n), default=None)
        for at, cause in failures:
            if at >= after and (nxt is None or at <= nxt):
                return cause
        return None

    rows = []
    for n in sorted(set(s.ja) - set(s.en)):
        text = s.ja[n].text
        defect = app.caption_defect(text)
        cause = failure_near(n)
        # Precedence is causal, not textual. Once the leg is down every caption
        # lacks an EN whatever its text, so a screen verdict there is a
        # counterfactual and rides `screened_now` instead of explaining the loss.
        if n in logged:
            why, detail = DECLINED, logged[n]
        elif n in strikes:
            why, detail = STRIKE, f"failure that spent a strike ({cause or 'cause not logged'})"
        elif degrade and n > last_en:
            why, detail = DISABLED, str(degrade["reason"])
        elif defect:
            why, detail = DECLINED, defect
        elif cause:
            why, detail = FAILED, f"transient failure ({cause}), leg recovered"
        elif n == last_ja:
            why, detail = SHUTDOWN, "last caption; EN did not drain before exit"
        else:
            why, detail = FAILED, "transient failure, leg recovered"
        rows.append(
            {
                "n": n,
                "at": s.ja[n].at.isoformat(sep=" "),
                "why": why,
                "detail": detail,
                # A strike today's screen would refuse never reaches the
                # translator, so it cannot spend one of TRANSLATE_MAX_FAILURES.
                "screened_now": defect is not None,
                "chars": len(text),
                "text": text[:40],
            }
        )
    return rows


def lags(s: Session) -> list[tuple[int, float]]:
    return [(n, (s.en[n].at - s.ja[n].at).total_seconds()) for n in sorted(s.en) if n in s.ja]


def sweep(caps: list[tuple[str, int, str]], lo: int, hi: int) -> list[dict]:
    """What the repetition screen catches as the unit bound widens.

    Reported as the bound RANGES that share a verdict, each naming the captions
    the widening newly catches. The count alone cannot answer the question the
    bound exists for -- which loop was escaping -- and a 9-character unit
    escaping a bound of 8 is the live defect that set the shipped 13.
    """
    rows: list[dict] = []
    previous: set[tuple[str, int]] = set()
    for bound in range(lo, hi + 1):
        caught = {
            (s, n) for s, n, t in caps if repeat_span_at(t, bound) >= app.CAPTION_REPEAT_MAX_CHARS
        }
        if rows and len(caught) == rows[-1]["caught"]:
            rows[-1]["to"] = bound
            continue
        rows.append(
            {
                "from": bound,
                "to": bound,
                "caught": len(caught),
                "new": [{"session": s, "n": n} for s, n in sorted(caught - previous)],
            }
        )
        previous = caught
    return rows


def build(sessions: list[Session], unclaimed: list[LogEvent]) -> dict:
    caps = [(s.name, c.n, c.text) for s in sessions for c in s.ja.values()]
    texts = [t for _, _, t in caps]
    survivors = sorted(
        ((app.repeat_span(t), t) for t in texts if not app.caption_defect(t)), reverse=True
    )
    lengths = sorted(len(t) for t in texts)
    all_lags = [x for s in sessions for _, x in lags(s)]
    # Split the drops off the SHIPPED verdict rather than restating the latin
    # rule: `caption_defect` returns the repetition reason first and is gated on
    # ASR_LANGUAGE, so the subtraction follows --source-lang where a copy of the
    # rule silently reported a latin screen an `en` session never ran.
    repetition_drops = sum(1 for t in texts if app.repeat_span(t) >= app.CAPTION_REPEAT_MAX_CHARS)
    combined_drops = sum(1 for t in texts if app.caption_defect(t))
    latin_only = combined_drops - repetition_drops

    per_session = []
    for s in sessions:
        missing = explain_missing(s)
        degrade = find_degrade(s)
        sl = [x for _, x in lags(s)]
        peaks = [e.body for e in s.events if _PEAK.match(e.body)]
        per_session.append(
            {
                "session": s.name,
                "captions": len(s.ja),
                "translated": len(s.en),
                "missing": missing,
                "missing_by_reason": {
                    w: sum(1 for r in missing if r["why"] == w)
                    for w in (DECLINED, STRIKE, DISABLED, SHUTDOWN, FAILED)
                    if any(r["why"] == w for r in missing)
                },
                "degrade": (
                    None
                    if not degrade
                    else {
                        "at": degrade["at"].isoformat(sep=" "),
                        "reason": degrade["reason"],
                        "source": degrade["source"],
                        "last_en": max(s.en) if s.en else None,
                        "ja_only_after": sum(1 for n in s.ja if n > max(s.en, default=0)),
                    }
                ),
                "restores": [
                    {"at": r["at"].isoformat(sep=" "), "reason": r["reason"], "source": r["source"]}
                    for r in find_restores(s)
                ],
                # A peak never clears, so the LAST line is the session's worst.
                "backlog_peak": peaks[-1] if peaks else None,
                "drops": attribute_drops(s),
                "lag_s": (
                    None
                    if not sl
                    else {
                        "pairs": len(sl),
                        "p50": statistics.median(sl),
                        "p90": sorted(sl)[int(len(sl) * 0.9)],
                        "max": max(sl),
                    }
                ),
                # _translate tests `_turns % TRANSLATE_ROTATE_TURNS` BEFORE the
                # increment, so the rotation is paid by the 101st turn, not the
                # 100th: tagging n % 100 made this read the wrong caption and
                # "0 of 75 slow turns at a boundary" (f398818) constrained nothing.
                # Limit: n is a caption number, and a declined caption never became
                # a turn, so the two drift apart by the session's decline count.
                "slow_turns": [
                    {
                        "n": n,
                        "lag_s": x,
                        "at_rotation": n > 1 and (n - 1) % app.TRANSLATE_ROTATE_TURNS == 0,
                    }
                    for n, x in lags(s)
                    if x >= 6
                ],
            }
        )

    return {
        "sessions": per_session,
        "totals": {
            "sessions": len(sessions),
            "captions": len(texts),
            "translated": sum(len(s.en) for s in sessions),
            "missing": sum(len(p["missing"]) for p in per_session),
            "unclaimed_log_events": len(unclaimed),
        },
        "screen": {
            "unit_chars": app.CAPTION_REPEAT_UNIT_CHARS,
            "max_chars": app.CAPTION_REPEAT_MAX_CHARS,
            "latin_ratio": app.CAPTION_LATIN_RATIO,
            "source_lang": app.ASR_LANGUAGE,
            "repetition_drops": repetition_drops,
            "latin_drops": latin_only,
            "combined_drops": combined_drops,
            "sweep": sweep(caps, 1, 24),
            "longest_surviving": [{"span": sp, "text": t[:40]} for sp, t in survivors[:5] if sp],
        },
        "length": (
            {}
            if not lengths
            else {
                "p50": lengths[len(lengths) // 2],
                "p90": lengths[int(len(lengths) * 0.9)],
                "p99": lengths[int(len(lengths) * 0.99)],
                "max": lengths[-1],
            }
        ),
        "lag_s": (
            {}
            if not all_lags
            else {
                "pairs": len(all_lags),
                "p50": statistics.median(all_lags),
                "p90": sorted(all_lags)[int(len(all_lags) * 0.9)],
                "max": max(all_lags),
            }
        ),
    }


def render(rep: dict) -> str:
    t = rep["totals"]
    out = [
        f"{t['sessions']} sessions, {t['captions']} captions, "
        f"{t['translated']} translated, {t['missing']} without EN",
        "",
    ]
    for p in rep["sessions"]:
        out.append(f"== {p['session']}  {p['captions']} captions, {p['translated']} EN")
        if p["lag_s"]:
            g = p["lag_s"]
            out.append(
                f"   EN behind JA: p50 {g['p50']:.0f}s  p90 {g['p90']:.0f}s  max {g['max']:.0f}s"
            )
        if p["degrade"]:
            d = p["degrade"]
            out.append(
                f"   DEGRADE [{d['source']}] {d['at']}  {d['reason']}"
                f"  (last EN n={d['last_en']}, {d['ja_only_after']} JA-only after)"
            )
        for r in p["restores"]:
            out.append(f"   RESTORE [{r['source']}] {r['at']}  {r['reason']}")
        out.append(f"   backlog peak: {p['backlog_peak'] or 'none logged'}")
        if p["drops"]:
            out.append(f"   drops: {len(p['drops'])} increases, {p['drops'][-1]['drop']} blocks")
        for d in p["drops"]:
            begin = f"n={d['prev']['n']}" if d["prev"] else "session start"
            end = f"n={d['next']['n']}" if d["next"] else "session end"
            gap = "gap open" if d["gap_s"] is None else f"gap {d['gap_s']:.1f}s"
            skipped = f"  skip +{d['skip_delta']}" if d["skip_delta"] else ""
            out.append(
                f"     {d['at'][11:]}  +{d['delta']} blocks (drop={d['drop']})"
                f"  {gap}  {begin} -> {end}{skipped}"
            )
            for w in d["screened"]:
                out.append(f"       screened {w['at'][11:]}  {w['defect']}  {w['text']}")
        if p["missing_by_reason"]:
            summary = ", ".join(f"{k}={v}" for k, v in p["missing_by_reason"].items())
            out.append(f"   no EN: {summary}")
        # A degrade's tail is one event, not N. Listing every caption behind it
        # buries the handful that each lost their EN for their own reason.
        held = [r for r in p["missing"] if r["why"] == DISABLED]
        if held:
            screened = sum(1 for r in held if r["screened_now"])
            out.append(
                f"     n={held[0]['n']}..{held[-1]['n']}  {DISABLED:<9} {len(held)} behind the "
                f"degrade ({screened} the screen would refuse anyway)"
            )
        for row in p["missing"]:
            if row["why"] == DISABLED:
                continue
            flag = " [screened now]" if row["screened_now"] and row["why"] != DECLINED else ""
            out.append(
                f"     n={row['n']:<4} {row['why']:<9} {row['chars']:>4}ch  {row['detail']}{flag}"
            )
        rot = [s for s in p["slow_turns"] if s["at_rotation"]]
        out.append(
            f"   slow turns (>=6s): {len(p['slow_turns'])}"
            f"   at a {app.TRANSLATE_ROTATE_TURNS}-turn boundary: {len(rot)}"
        )
        out.append("")

    s = rep["screen"]
    out.append(
        f"screen: unit<={s['unit_chars']} span>={s['max_chars']} "
        f"latin>{s['latin_ratio']}x ({s['source_lang']})  ->  "
        f"{s['repetition_drops']} repetition + {s['latin_drops']} latin"
        f" = {s['combined_drops']} drops"
    )
    out.append("   unit-bound sweep:")
    for r in s["sweep"]:
        span = f"{r['from']}" if r["from"] == r["to"] else f"{r['from']}-{r['to']}"
        new = ", ".join(f"{d['session'].removesuffix('.txt')} n={d['n']}" for d in r["new"])
        out.append(
            f"     bound {span:<7} catches {r['caught']:>3}"
            + (f"   newly: {new}" if r["from"] > 1 and new else "")
        )
    out.append("   longest surviving repetitions:")
    for r in s["longest_surviving"]:
        out.append(f"     {r['span']:>4}ch  {r['text']}")
    if rep["length"]:
        n = rep["length"]
        out.append(
            f"caption length: p50 {n['p50']} p90 {n['p90']} p99 {n['p99']} max {n['max']} chars"
        )
    if rep["lag_s"]:
        g = rep["lag_s"]
        out.append(
            f"EN behind JA overall: {g['pairs']} pairs, "
            f"p50 {g['p50']:.0f}s p90 {g['p90']:.0f}s max {g['max']:.0f}s"
        )
    out.append(
        "note: thread rotation is not separable from transcripts alone -- a slow turn is "
        "reported with whether it sits on a rotation boundary, never asserted as one."
    )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-derive a live session from its saved files.")
    ap.add_argument("transcripts", nargs="*", help="Transcript files (default: transcripts/*.txt).")
    ap.add_argument("--log", action="append", default=[], help="Redirected stderr log; repeatable.")
    ap.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    ap.add_argument(
        "--source-lang",
        choices=["ja", "en"],
        default=app.ASR_LANGUAGE,
        help="Source language the session ran with (default: %(default)s).",
    )
    args = ap.parse_args()

    # The transcript records captions, not the flags that produced them, so the
    # language rides the command line. Rebinding the module global is how the
    # shipped screen is re-run rather than re-implemented: without it an `en`
    # session is re-derived under a latin screen the live run never applied.
    app.ASR_LANGUAGE = args.source_lang

    paths = args.transcripts or sorted(glob.glob("transcripts/*.txt"))
    if not paths:
        # A fresh clone has no transcripts/ and this is not an error: there is
        # simply nothing to report yet.
        print("no transcripts found", file=sys.stderr)
        return 0

    sessions = sorted((read_session(p) for p in paths), key=lambda s: s.name)
    logs = args.log or [p for p in ["stt.log"] if glob.glob(p)]
    events: list[LogEvent] = []
    for p in logs:
        events += read_log(p)
    unclaimed = attach_logs(sessions, sorted(events, key=lambda e: e.at))

    report = build(sessions, unclaimed)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
