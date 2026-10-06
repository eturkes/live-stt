# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). The session body the user pastes names the
rows it funds, and a `maintain.md` Queue body naming none funds every row in rank order; closing a
row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Run the wheels at the host's OpenVINO 2026.4.1** — the host and container GenAI tarballs moved
   2026.4.0 → 2026.4.1 (core: NPU compiled-blob import hardening, GPU/CPU fixes; genai: a version
   bump alone) while `uv.lock` pins the 2026.4.0 wheels. The container NPU path stays green on the
   old wheels (whisper golden warm 14.5 s, tarball selftest CPU/GPU/NPU OK), so this is alignment,
   not a repair — and bumping re-arms the soname match (`.2641` wheel = tarball) that `.envrc` and
   `_session_env` strip. Data tier, funded by the 10-07 request.
   **Accept:** `uv.lock` carries `openvino` / `openvino-genai` / `openvino-tokenizers` 2026.4.1,
   lock-only, floors untouched; under the prelude, `/proc/self/maps` of an NPU decode maps every
   `libopenvino.so*`, `libopenvino_genai.so*` and `libtbb*` from the venv, a no-`.envrc` control
   mapping the tarball's genai instead; every replay golden passes, the whisper NPU one with its
   cold-compile time recorded; retention CER re-derived on the NPU, no worse than 0.0532; paired
   per-update decode cost old vs new, no regression whose CI excludes 0; the version-specific prose
   (`.envrc`, `openvino-accel.md`) names the current pair; `pip-audit` clean; the gate stays green.

2. **A dropped trailing mark swallows the next spoken character** — `_anchor` ties deleting the
   published tail's final `。`/`、` against substituting it with the next decoded character and takes
   the nearer end, which spends that character on the mark: the first mora of the next word vanishes
   (`。ゃあ`, `。れどおりに`, `。ースの3人`). 25 of 441 sentence joins plus 9 line starts in the 10-02
   session; the shipped processor reproduces `SRC 1137` byte-identically (`仕様書を書いて。` ×2, then
   `仕様書を書いてそれどおりに…` publishes `仕様書を書いて。れどおりに…`), and the old count rule cut
   at the same place. Kernel.
   **Accept:** while the published tail ends in punctuation or a space, no equal-cost end that spends
   a word character on that mark is chosen over the end that drops it — a mark carries no audio, so
   a word character in its slot is new speech; a mark re-spelled as another mark (`。` → `、`) keeps
   today's boundary. Locks red on the unfixed processor for the live shapes, plus that control.
   Every existing streaming lock unchanged; the whisper NPU replay golden and the retention CER
   re-derived on the NPU (a moved golden moves only where a published mark meets a decode that drops
   it, recorded in the commit body); `eval_anchor_scenarios.py --baseline` against the old processor
   reported. The gate stays green.

3. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and M14's `_respawn` arm have never met a mic (`_probe` has); standing
   debt = latency feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.

4. **Reconcile the human-facing doc set** — `.agent/spec.md` `Artifacts` and `orientation.md` call
   `README.md` the only human-facing doc, while `human-docs.md`, L-021's owner, names `README.md`,
   `models/README.md` and the CLI strings ⇒ an agent reading the first two can write
   `models/README.md` in agent register. Docs tier.
   **Accept:** `rg -n --hidden 'human-facing' .agent/spec.md .claude/rules/` → every hit names the
   same surface set as `human-docs.md` or points at it; the gate stays green.

5. **Screen each segment inside a released piece** — one trim can release several whisper segments
   as one piece, and the screen drops that piece whole, so a loop segment takes its clean
   neighbours with it (tester-1 trace: `繰り返し`×12 + one clean sentence, both dropped). Kernel.
   **Accept:** a piece is screened per released segment (segment counts from the trimming decode,
   never hypothesis text), the surviving segments publish as one line, a lock shows a loop segment
   dropped alone with its clean neighbour published, red on the unfixed worker; the gate stays
   green.

6. **Session report reads two-way transcripts** — a transcript records no per-line decode language,
   so `session_report.py` re-screens every source line under one `--source-lang`: an English line a
   two-way session published reads as a latin `declined` caption (reviewer-6, authored mixed
   transcript: `latin_drops=1`). Data tier.
   **Accept:** a two-way session's English source lines are never reported as screen declines — the
   transcript carries what the report needs, or the report takes a flag that skips the latin rule
   for two-way sessions; a lock over an authored mixed transcript, red before; the gate stays green.

7. **A trim cut at an early segment end re-publishes speech** — `_trim` cuts the audio at the last
   covered segment's `end_s`, and whisper places that end up to ~2 s early, so the retained audio
   still holds the tail of the piece just published and the next decode re-transcribes it into the
   next line (10-02: `SRC 1173` → `1174` repeats `そういうことがあるらしいんですよね。`; `9`/`10`,
   `290`/`291`, `307`/`308` the same shape, typescript-confirmed on `1173`). Kernel, unfunded.
   **Accept:** a post-trim decode whose head re-spells the published tail publishes that head
   nowhere; a phrase the speaker genuinely repeats across a segment boundary (`そうそう`,
   `はい。はい。`) keeps both copies; a lock replaying the `1173`/`1174` shape, red before; retention
   CER and the long-form replay no worse than recorded; the gate stays green.

8. **Re-measure the fresh-session baseline** — `assurance-posture.md` states a fresh MAIN session
   opens at 47K over 89 KB attached (global `CLAUDE.md` 27.0 KB, `.agent/spec.md` 14.2 KB, three
   bare rules files 28.7 KB); `wc -c` now reads 98.3 KB (28.0 / 21.9 / 28.9 KB), so the literal the
   teammate-budget paragraph quotes no longer describes a fresh session. Docs tier.
   **Accept:** a fresh session on a clean tree, measured with `context-gauge`, plus `wc -c` over
   the attached set → the paragraph carries those numbers, or drops the literal for the recipe;
   the gate stays green.
