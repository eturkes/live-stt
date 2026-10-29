# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). A `steered/maintain.md` Queue body names
the rows it funds, and the `auto/maintain.md` Queue body funds every row in rank order; closing a
row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Live-mic validation pass** — user-only (L-004): M14's `_respawn` arm, the degrade marker and
   the SIGHUP shutdown have never met a mic (`_probe` has); standing debt = `live-smoke.md` items 1,
   3, 4 (latency feel), 7 (Ctrl+C mid-decode), 8 (`-o`), 9 (quiet far speech + empty room), 10 (GPU
   mode), 11 (`--playback` in a real call) and a multi-hour soak.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.

2. **Playback leveling across digital silence** — `--playback` hears EXACT zeros whenever nothing
   plays, and `LeveledVad` admits no window ≤ `VAD_LEVEL_FLOOR_DBFS`, so its 10 s history never
   ages and a quiet talker after a loud one is heard at the loud one's gain 1.0: live, 5 quiet CV
   clips (speech p90 −37..−43 dBFS) between loud ones with 10 s zero gaps, 3 plays each — cv489 lost
   whole, cv894 lost `先月`, cv904 gained `はい。`; loud clips kept every caption, none landing past a clip's 4 s tail. Every arm tried
   trades quiet against loud (`asr-pipeline.md`); `LeveledVad` is a user ruling ⇒ the fix
   needs the user's ruling first. Tier `kernel` (decode-quality mechanism, no downstream re-check).
   **Re-open condition (user ruling):** a real call recorded with `--save-audio` (`live-smoke.md`
   item 11) holds exact-zero runs between talkers — the regime this row fixes; until then the row
   waits, unfunded.
   **Accept:** a playback-mode-only change (mic path byte-identical by construction: the replay
   goldens and retention CER 0.0532 unmoved) recovers cv489, cv894 and cv904 in
   `tests/eval_playback_silence.py`'s shipped arm with no loud clip losing a character against
   today's shipped arm, confirmed by one `tests/eval_playback_routing.py --e2e probe` and one
   `--e2e pinned` run; the user rules the shape before it ships.

3. **Python 3.15 interpreter** — `.python-version` pins `3.14` while CPython 3.15.1 is the latest
   release: the wheel-only hard dependencies openvino, openvino-genai, onnxruntime and sherpa-onnx
   publish no cp315 wheel at their latest releases.
   **Accept:** every package in `uv.lock` resolves a cp315 wheel → `.python-version` 3.15, `.venv`
   re-synced, `tests/fetch_fleurs_parallel.py` rerun so its Python stamp re-qualifies on a
   byte-identical rebuild, the full gate green under the accelerator prelude with zero skips,
   retention CER 0.0532 re-derived on the NPU.

4. **Export recipe at the latest transformers and Python** — `models/README.md`'s whisper-ja-760M
   export pins `transformers==4.57.6` in a Python 3.13 venv. optimum-intel 2.2.0 caps
   `transformers<5.6`, and the newest it admits, 5.5.4, exports a model whose retention hypothesis
   differs from the shipped weights' by one inserted `。` (CER 0.0532 either way). Under Python 3.14
   optimum 2.3.0's exporter raises `TypeError: NormalizedConfig.__init__() got multiple values for
   argument 'allow_new'` (a `functools.partial` class attribute binds as a method from 3.14 on).
   The pinned recipe decodes byte-identically to the shipped weights on 150 + 150 FLEURS clips and
   on retention.
   **Accept:** each unblock moves its own pin — an optimum-intel release admitting the latest
   transformers moves `transformers==`, an optimum stack running under the latest Python moves
   `--python` — once an export under the moved pin into scratch, run through
   `tests/eval_asr_screen.py --arms fleurs,retention --device NPU` (private `OPENVINO_CACHE_DIR`),
   decodes every hypothesis byte-identically to the shipped weights; any differing hypothesis → a
   re-export ruling for the user with the D-016 JA legs re-measured.

5. **Ruling-tag lock reports every dead anchor** — `tests/test_law_consistency.py`'s
   `test_every_template_ruling_keys_on_a_live_template_anchor` asserts inside its per-file loop, so a
   template refresh that kills anchors in two rules files names only the first: the refresh that
   reworded the closing-diff review and portability clauses failed on `assurance-posture.md` alone,
   `toolchain.md`'s dead anchor surfacing only through a scratch run of the test's own regexes. Each
   hidden anchor costs one more gate run to discover. Tier `kernel` (a gate lock; nothing downstream
   re-checks it).
   **Accept:** one run of the lock names every broken opener, stray verdict, short anchor and dead
   anchor across all `.claude/rules/` files — a dead anchor planted in two files at once, both named
   in one failure — and each of the four firings still fires alone on its own planted input (L-022).

6. **Queue rows declare their tier** — `assurance-posture.md` makes a unit's queue row its acceptance
   contract, "its tier declared there", and `CLAUDE.md`'s `Session flow` keys a MAINTAIN unit's
   teammate depth and advisor bar on that tier; three open rows carry none, so a session funding one
   opens without the depth decision. Tier `docs` (queue prose; a consistency pass re-checks it).
   **Accept:** every open row in this file declares `Tier kernel|data|docs` with the reason that
   decides it (where a defect is re-checked, at what cost); a user-only row states its tier or that
   none applies.

7. **Fresh-session baseline re-measured** — `assurance-posture.md` records a fresh MAIN session
   opening at ≈47K over 99.8 KB attached (user-level instruction file 10.9 KB, `CLAUDE.md` 34.8 KB,
   `.agent/spec.md` 22.3 KB, the three bare rules files 31.9 KB) and orders a re-measure whenever
   that set moves; the set now reads 12.1 / 35.6 / 24.4 / 33.2 KB (`wc -c`) while the recorded
   figures still describe the old set, so the baseline a work figure subtracts from a `main=` total
   is unmeasured for today's set. Tier `docs` (rule prose; the next re-measure re-checks it).
   **Accept:** the rule's own recipe runs on a clean tree at the repo root — `claude -p "Reply with
   the single word OK." --output-format stream-json --verbose`, assistant `usage` summed, `wc -c`
   over the set — and `assurance-posture.md`'s baseline figures equal that run's, command and numbers
   in the commit body.

8. **Landed teammate worktrees removed** — `git worktree list` holds 45 linked worktrees under
   `.scratch/worktrees/` past their close, which `CLAUDE.md`'s `Teammates` ends by removing worktrees
   + branches: `wt/reviewer-1..35` (each tip an ancestor of `main`), `wt/general-purpose-1..9` (every
   commit patch-equivalent on `main` by `git cherry`, save `d40fdba`, landed conflict-resolved as
   `37a398f`) and a detached `main-gate` (`d7b329a` ≡ `ee387e4`). 30 reviewer worktrees hold 35
   untracked review-probe paths alone and no tracked edit: test files under `tests/`, one
   `.reviewer27/` scratch dir, `reviewer34-neutralize.py`. Tier `data`
   (the landing proof re-validates each branch at removal, and is the last check: a deleted branch
   takes its reflog with it).
   **Accept:** each removal follows its own landing proof (ancestor, `git cherry` all `-`, or the
   named equivalent) and an inspection of its worktree's untracked files; then `git worktree list`
   names the primary tree alone, `git branch --list 'wt/*'` prints nothing, and
   `attempt/p009-audio-time-cut` stays (user ruling, `b906ce0`).
