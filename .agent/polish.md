# Polish — live-stt

Deferred-perfection register. `/session-polish` is its sole consumer: pick by `pri`, size-fit the
remaining window, run one item at a time under the gate identity in `memory.md`, prune the row in the
commit that lands it. Off-spine improvements are born here at deferral time with the acceptance check
written while the evidence is fresh. Milestone/unit state stays in `roadmap.md`; nothing here gates a
milestone.

Row shape: `P-<n>` monotonic and never reused (pruning leaves gaps) · `pri` 1 = do first … 3 =
whenever · `size` S ≈ ≤15 % of a window, M ≈ ≤35 %, L = the session. An item whose evidence pointer
or acceptance check stops holding takes `stale(<why>)` in place. A finding that implies spine work
goes under Spine flags and to the user instead of running here.

## Open

- **P-009 — Rule on the VAC repetition artifact pinned in the `whisper/long` golden.** `pri=3`
  `size=M` — **ROOT CAUSE NAMED; the remaining half is spine work, see Spine flags.**
  - why: the committed row's second utterance carries a real LocalAgreement-2 repetition —
    `…ジェミニAPIに送ってに送って、…`. M11.3c characterized it and deliberately did not fix it (D-014
    makes goldens characterization snapshots; pinning is what makes a future commit-policy change
    visible), so the golden asserts a known-wrong transcript as correct.
  - **root cause (measured, do not re-derive):** `emitted` is a character COUNT re-derived from the
    latest hypothesis, but it must denote what was PUBLISHED. `process()` pins
    `stable = max(agreed, len(self.emitted))` and then assigns `self.emitted = text[:stable]`, so a
    decode that RE-SPELLS an already-published prefix at the same character length silently
    re-anchors the boundary to a different point in the audio. Instrumented replay of
    `spike/backends/cache/long.wav` on NPU, 13 updates, **0 trims and 0 forced trims** — so this is
    not a trim effect and not the `_trim`/`common_prefix` path the row originally suspected:
      - `[9]` publishes `マイクから音声を取り込みジェミニAPIに送って` (23 chars, screen state).
      - `[12]` (the LAST `process()`, `commit=''`) rewrites the record to `text[:23]` of a hypothesis
        that re-spelled the prefix as `マイクから音声を取り込み、Gemini API` — also 23 chars, but
        covering ~4 characters (~0.7 s) LESS audio.
      - `update(final=True)` then appends `finish()`, which returns `previous[len(emitted):]` =
        `に送って、日本語の…` — audio already on screen — giving the golden's exact string.
    D-016(d)'s append-only fix stopped `emitted` from SHRINKING; this is the length-PRESERVING case
    it does not cover. `commit = text[len(self.emitted):stable]` reads the same index, so the
    re-anchor can duplicate mid-stream too; `long` only happens to hit it at the final flush.
    **Verdict: NOT inherent to LocalAgreement-2** — the policy is fine, the bookkeeping is wrong, so
    the row's "written ruling that it is inherent" branch is factually unavailable.
  - evidence: `tests/replay_goldens.json` `whisper/long` segment 2; `streaming.py` `process()`
    (`stable`/`self.emitted = text[:stable]`) and `finish()`.
  - acceptance (unchanged, fix branch only): a fix that removes the duplication with
    `tests/eval_cer.py` showing no CER regression and the golden regenerated.
  - **tried + rejected, both measured on NPU — do not re-run either as-is:**
    - v1, guard `commit` on `text.startswith(self.emitted)`: duplication gone, but `_trim` fires only
      on a nonempty commit ⇒ trimming starves. `max_buffer` 11.248 → 23.9/25.5 s, retention decode
      117 → 210 s on 182 s of audio = **RTF 1.15, above real time**. Unshippable.
    - v2, audio-time cut at `finish()`: `emitted_s` = furthest published audio time (monotone),
      `reanchored` set when `commit_audio_s < emitted_s`, cut at `max(start, _index_at_time(…))`.
      `process()` left byte-identical, so trim behaviour and per-update cost are untouched — the
      M11.4 backpressure arm reproduces the committed trace at `divergences == 0`. Duplication gone
      on `whisper/long` (`…ジェミニAPIに送って、日本語の…`, tail intact). **Retention CER 0.0583 →
      0.0635**: N=1166, S=33, **D 35 → 41**, I=0, hyp 1131 → 1125 chars. Retention carries no
      duplication (I=0 in both arms) ⇒ there the cut only costs. Code + test + regenerated golden on
      branch `wt/p009-attempt` @ `bd37bf7`.
  - **what a third attempt must solve first:** `reanchored` cannot separate a genuine re-spelling
    from ordinary span jitter. P-011 measured 6 of 157 commits moving backward, median 0.084 s / max
    0.452 s, while the real re-spelling moved ~0.7 s — so a detector that fires on any backward move
    drops characters clip-wide, and `_index_at_time`'s `round()` interpolation can cut one early
    each time it fires. Separate the two cases before applying any cut, and re-measure retention CER
    on every arm. Both directions change shipped decode output ⇒ roadmap unit, not polish.

- **P-002 — Decide whether the session context should learn EN renderings, not just JA terms.**
  `pri=3` `size=M`
  - why: `SessionContext` (D-015) holds a JA term list and tells the translator to render each one
    consistently, but never learns *which* EN rendering a term received. A first-draft `observe_en`
    pairing JA terms with capitalized EN runs was cut before shipping: aligning a rendering out of
    unaligned JA/EN caption pairs is guesswork, its capitalization and title heuristics were
    untestable, and no failure mode was named that the plain term list does not already prevent.
    Repeating the term list every turn already pins a name to one spelling across thread rotations,
    so the learner has to beat that, not beat nothing.
  - evidence: `live_stt.py` `SessionContext.translator_brief` carries the term list and the
    consistency instruction; the cut is why it lists terms unpaired.
  - acceptance: on a JA corpus with a proper noun recurring across ≥10 turns, count distinct EN
    renderings of that noun per session with and without the learner. The learner ships only if it
    strictly reduces the distinct-rendering count on a corpus it was not tuned on, and a paired
    judged comparison on the same items shows no adequacy regression against the current brief.
    Either arm failing keeps the term list unpaired and closes this row.

## Spine flags

- spine? `streaming.py`'s `emitted` can re-anchor to earlier audio at an unchanged character count,
  duplicating already-published text on the shipped whisper/VAC path | why: instrumented NPU replay
  of `long.wav` (13 updates, 0 trims) shows `process()` rewriting the 23-character record from
  `…ジェミニAPIに送って` to `…、Gemini API` at update `[12]`, after which `finish()` flushes
  `に送って、…` a second time — reproducing `whisper/long` segment 2 exactly. Live decode output, so
  a fix needs a CER re-measurement and a golden regeneration. Two fixes were built and measured;
  both cost more than the artifact (v1 RTF 1.15, v2 retention CER 0.0583 → 0.0635) and neither
  shipped. Full derivation + both measurements in P-009; user rules on funding a third attempt.
