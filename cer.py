#!/usr/bin/env python3
"""Character-level alignment for JA STT evaluation: normalize + Levenshtein S/D/I.

Shared eval primitive (sibling to replay.py). tests/build_stressor.py (T9.1)
uses it to score excess deletion; the CER harness + its fixed-vector unit tests
(T9.2) build on top. Pure stdlib, no third-party deps.

normalize(): NFKC -> casefold -> drop Unicode categories P/S/Z/M (punctuation,
symbols, separators incl. whitespace, combining marks). One aggressive recipe,
mirroring kotoba-whisper / Whisper BasicTextNormalizer, so "空が青いです。." and
"空が青いです" collapse identically. Numerals are NOT folded on purpose (七 vs 7
scores as a substitution) -> the primary metric stays strict; a lenient
numeral-folded figure, if ever wanted, is a separate report, never a silent one.

align(): edit distance with an explicit backtrace and a documented tie order.
On an equal-cost tie the diagonal (match/substitute) is preferred over the
delete/insert L-shape, so a transposition (ab <-> ba) scores as two
substitutions rather than a delete + an insert. This keeps deletions -- the
headline "missed content" signal for long-form collapse -- from being inflated
by reorderings. Returns (S, D, I) counts.
"""

from __future__ import annotations

import unicodedata

_DROP = frozenset("PSZM")  # Unicode major categories stripped before scoring


def normalize(text: str) -> str:
    """NFKC -> casefold -> drop punctuation/symbol/separator/mark characters."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in folded if unicodedata.category(c)[0] not in _DROP)


def align(ref: str, hyp: str) -> tuple[int, int, int]:
    """Align `hyp` to `ref`, returning (substitutions, deletions, insertions).

    Deletion = a ref character absent from hyp (dropped content -- the headline
    for long-form collapse). Insertion = a hyp character with no ref counterpart
    (hallucination/duplication). Substitution = a mismatched pair. Inputs are
    taken as-is; callers normalize() first.

    The DP records one move per cell, preferring the diagonal on cost ties so a
    transposition counts as substitutions, not delete+insert (keeps D honest).
    """
    m, n = len(ref), len(hyp)
    # cost[i][j] = edit distance ref[:i] -> hyp[:j]; move = how cell (i,j) was
    # reached: 0 diagonal (match/sub), 1 deletion (consume ref), 2 insertion.
    cost = [[0] * (n + 1) for _ in range(m + 1)]
    move = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        cost[i][0] = i
        move[i][0] = 1
    for j in range(1, n + 1):
        cost[0][j] = j
        move[0][j] = 2
    for i in range(1, m + 1):
        ri = ref[i - 1]
        for j in range(1, n + 1):
            diag = cost[i - 1][j - 1] + (ri != hyp[j - 1])
            best, mv = diag, 0  # diagonal wins ties
            drop = cost[i - 1][j] + 1
            if drop < best:
                best, mv = drop, 1
            add = cost[i][j - 1] + 1
            if add < best:
                best, mv = add, 2
            cost[i][j] = best
            move[i][j] = mv
    s = d = ins = 0
    i, j = m, n
    while i > 0 or j > 0:
        mv = move[i][j]
        if mv == 0:
            s += ref[i - 1] != hyp[j - 1]
            i -= 1
            j -= 1
        elif mv == 1:
            d += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, d, ins


def cer(ref: str, hyp: str) -> float:
    """(S + D + I) / len(normalized ref); 0.0 when the ref normalizes to empty."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0
    s, d, ins = align(r, h)
    return (s + d + ins) / len(r)
