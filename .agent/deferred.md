# live-stt — deferral queue

The funding menu, read on demand. Unattached ⇒ `.agent/spec.md` stays the sole attached state, and
its `Tasks` = one open `- [ ]` row per row below, same rank and title, then the pointer here. Rank =
funding order. Acceptance is written at deferral time while the evidence is fresh, and the funded row
is that unit's whole contract (`assurance-posture.md`). The session body the user pastes names the
rows it funds, and a `maintain.md` Queue body naming none funds every row in rank order; closing a
row deletes it from this file and from `.agent/spec.md`'s `Tasks` in one
commit. `tests/test_law_consistency.py` locks that pairing and rejects naming a row by `rank N`
anywhere else, since a rank retargets onto a different unit the moment an earlier row dies.

1. **Live-mic validation pass** — user-only (L-004), the largest untested surface: M13.2, the four
   2026-09-06 polish fixes and M14's `_respawn` arm have never met a mic (`_probe` has); standing
   debt = latency feel, `-o`, soak, sustained cadence, Ctrl+C-mid-decode, VAC partial cadence.
   **Accept:** the user runs `live-smoke.md` and reports; each item lands verified or defective.

2. **Reconcile the human-facing doc set** — `.agent/spec.md` `Artifacts` and `orientation.md` call
   `README.md` the only human-facing doc, while `human-docs.md`, L-021's owner, names `README.md`,
   `models/README.md` and the CLI strings ⇒ an agent reading the first two can write
   `models/README.md` in agent register. Docs tier.
   **Accept:** `rg -n --hidden 'human-facing' .agent/spec.md .claude/rules/` → every hit names the
   same surface set as `human-docs.md` or points at it; the gate stays green.

3. **Screen each segment inside a released piece** — one trim can release several whisper segments
   as one piece, and the screen drops that piece whole, so a loop segment takes its clean
   neighbours with it (tester-1 trace: `繰り返し`×12 + one clean sentence, both dropped). Kernel.
   **Accept:** a piece is screened per released segment (segment counts from the trimming decode,
   never hypothesis text), the surviving segments publish as one line, a lock shows a loop segment
   dropped alone with its clean neighbour published, red on the unfixed worker; the gate stays
   green.

4. **Session report reads two-way transcripts** — a transcript records no per-line decode language,
   so `session_report.py` re-screens every source line under one `--source-lang`: an English line a
   two-way session published reads as a latin `declined` caption (reviewer-6, authored mixed
   transcript: `latin_drops=1`). Data tier.
   **Accept:** a two-way session's English source lines are never reported as screen declines — the
   transcript carries what the report needs, or the report takes a flag that skips the latin rule
   for two-way sessions; a lock over an authored mixed transcript, red before; the gate stays green.

5. **Thin rewrite keeps new speech** — `StreamingProcessor._anchor` falls back to the old count
   whenever a decode re-spells more than half of the last `ANCHOR_TAIL` published characters, so
   speech right after the published end is lost or re-committed (reviewer-7: published `ABCDEFGH`,
   previous `ABCDEFGHI`, decodes `XXGHIJ`, `XXGHIJK` → `IJK` lost). Letting the continuation rule
   run before the half bar alone is REFUSED (consultant-1 + consultant-2 BLOCK): it moves the
   record into a one-decode spelling, so a reversion, head drop or total drop duplicates published
   text — total-drop exact output 99.9 % → 50.6 % on the scenario harness. Kernel.
   **Accept:** below half agreement — (a) evidence = the first two characters that followed the
   published end in the decode the record was last LOCATED in (prefix match, alignment or an
   adopted thin end, never a count fallback), at an end after the tail's start where the published
   tail aligns with at least one matching character; lowest alignment cost wins, then the end
   nearest the old one; (b) the first decode carrying evidence commits nothing and leaves the
   record, and the next decode adopts the end when it proposes the same re-spelled published
   prefix, else the count stands; (c) a record set by the count fallback carries no evidence until
   a decode locates it again; (d) `finish()` adopts an evidenced end unconfirmed; (e) with no
   evidence the count stands exactly as on HEAD. Locks red on HEAD: the reviewer-7 probe, a
   terminal kana↔kanji re-spelling (`今日の状況を詳しくお話します` → `概要を説明致す次です`), a head
   drop over two decodes then restoration (`りごとを言いま` published, `ました。一体誰が`,
   `ました。一体誰がイ`, then `りごとを言いました。一体誰がイ`). Controls green on both: total rewrite with and without the continuation,
   each followed by restoration (`IJ` / `XXGHIJ` then `ABCDEFGHIJK…` → `ABCDEFGHIJKL`), one-decode
   garbage, a same-length thin rewrite. Every existing anchor lock passes unchanged; the four
   recorded clips replay byte-identical commit streams (`eval_latency.py` `commit_changes=0`); a
   committed scenario evaluator finds no script where HEAD's output is exact and the new one's is
   not; retention CER on the NPU recorded; the gate stays green.
