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
      branch `attempt/p009-audio-time-cut` @ `bd37bf7` — keep it; it is the only copy.
  - **ruled by user: leave the artifact.** The golden keeps pinning the duplication as a
    characterization snapshot (D-014). Do not re-attempt without a funded roadmap unit.
  - **what a third attempt must solve first:** `reanchored` cannot separate a genuine re-spelling
    from ordinary span jitter. P-011 measured 6 of 157 commits moving backward, median 0.084 s / max
    0.452 s, while the real re-spelling moved ~0.7 s — so a detector that fires on any backward move
    drops characters clip-wide, and `_index_at_time`'s `round()` interpolation can cut one early
    each time it fires. Separate the two cases before applying any cut, and re-measure retention CER
    on every arm. Both directions change shipped decode output ⇒ roadmap unit, not polish.

- **P-013 — `--context` is a shipped CLI flag with no README row.** `pri=3` `size=S`
  - why: README's option table (`--engine`, `--asr-device`, `--no-translate`, `-o`, `--no-save`,
    `--device`, `--list-devices`) skips `--context TEXT` entirely, and `--asr-device`'s row is the
    only place "session term biasing" is named — so the one user-facing correction channel for a
    mis-recognised name is undiscoverable from the doc. D-015's memory entry claimed README cited it;
    that citation never existed and is now removed.
  - evidence: `live_stt.py:1528` defines `--context`; `/usr/bin/rg -- '--context' README.md` is empty.
  - acceptance: README's option table carries a `--context TEXT` row in ASD-STE100 register (L-021),
    naming what the seed does (trusted at once, never evicted, reaches both the recogniser term list
    and the translator glossary) without restating D-015's internals; `rg -- '--context' README.md`
    is non-empty and `rg -nP '[\x{2013}\x{2014}]' README.md` stays clean.

- **P-012 — Re-measure the EN rendering learner on ASR output, not clean text.** `pri=3` `size=M`
  - why: `observe_en` (D-015, shipped by P-002) keys a learned English spelling on the JA term the
    recogniser produced. Every P-002 arm ran on clean Aozora text, which is the learner's BEST case:
    live, a mis-recognised name pairs a correct spelling to a wrong key, and the key never recurs, so
    the pairing is dead weight — or worse, the recogniser alternates two spellings of one name and
    each keeps its own rendering. Neither was measured.
  - evidence: `live_stt.py` `SessionContext.observe_en` pairs on `t in ja` against the caption text;
    `CodexTranslator.run` feeds it the accepted caption, recognition errors included.
  - acceptance: replay a WAV corpus through the shipped whisper/VAC path so the JA carries real
    recognition errors, then count both distinct EN renderings per session (with/without the learner,
    ≥3 sessions each) and pairings whose key never recurs. The learner stands as shipped if it still
    reduces the rendering count and dead pairings stay near zero; otherwise gate pairing on a term
    seen un-prompted since it was trusted.
  - prerequisite (first cost of the row): no committed corpus has both audio and a proper noun in
    ≥10 utterances — `tests/long_form.json`'s chapter-一 WAV puts 兵十 in 8. The Kokoro alignment
    covers the whole recording, so extending the build past section 一 is the cheap route.

## Spine flags

- spine? `streaming.py`'s `emitted` can re-anchor to earlier audio at an unchanged character count,
  duplicating already-published text on the shipped whisper/VAC path | why: instrumented NPU replay
  of `long.wav` (13 updates, 0 trims) shows `process()` rewriting the 23-character record from
  `…ジェミニAPIに送って` to `…、Gemini API` at update `[12]`, after which `finish()` flushes
  `に送って、…` a second time — reproducing `whisper/long` segment 2 exactly. Live decode output, so
  a fix needs a CER re-measurement and a golden regeneration. Two fixes were built and measured;
  both cost more than the artifact (v1 RTF 1.15, v2 retention CER 0.0583 → 0.0635) and neither
  shipped. Full derivation + both measurements in P-009; user rules on funding a third attempt.
