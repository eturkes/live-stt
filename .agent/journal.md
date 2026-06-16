# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

---

## 2026-06-16 — CLAUDE.md sync #4: human-facing prose polish (dashes + de-smell)

**Trigger:** `/session-prompt` override: "I updated the `CLAUDE.md`. If there is any work
you need to do in response, use this session to do so." 4th consecutive CLAUDE.md-sync
session. Diff (2+/2−): (1) the biases/"pink-elephant" rule rephrased + scope-broadened to
"writing text you will later read, especially if interpreted as a prompt" (no actionable
surface — governs how I phrase agent-facing text); (2) the UI/UX rule gained a concrete
directive — *"When writing for a human audience, prefer hyphens over other kinds of dashes,
enumerate flexibly, and vary comparative constructions"* (+ "user-facing"→"human-facing",
"+smells"). User chose **"Full human-facing polish"** over no-op / dashes-only.

**Shipped (README.md + `live_stt.py:659` string; no logic change):** swept all 13 em/en-dash
sites from README (the only human-facing doc) — pipeline steps `**X** —` → `**X.**`, em-dash
asides → parens/commas/sentence-splits, range en-dashes (`0.2–0.7`, `7–60`) → hyphens — plus
one de-cliché ("degrades gracefully"→"falls back"). `live_stt.py`'s lone human-facing dash
(`unavailable — JA-only` startup status) → `unavailable (JA-only, see log)`, mirroring the
sibling `disabled (--no-translate)`; README L36's code-literal synced to it. The other two
sub-directives (enumerate flexibly, vary comparatives) yielded ~nothing — README was already
clean of "not just X but Y"/forced triads. Plan in `.agent/scratch/2026-06-16_claudemd-sync4-human-polish.md`.

**Exempt (rule's own carve-out: "underlying code, including comments"):** all `.agent/*`,
`PLAN.md`, `CLAUDE.md`, every `live_stt.py` comment, the Codex `developerInstructions` prompt
(`:68`, model-facing + benched D-011), and the `__doc__`/argparse help (clean already).
Mass-sweeping those would fight L-015 (em-dashes deliberate in agent docs) → new **L-021**.

**Verified (agent):** README `grep -P '[\x{2013}\x{2014}]'` → 0; `import live_stt` OK; 50
tests green; ruff clean; `uvx pyright@1.1.410` 0/0/0 (confirms the in-session
`time_info`-unused note at `:663` is the known stale-LSP false-positive — `audio_callback`'s
sounddevice-signature param, unrelated to the string edit). The `:659` change has no test
coupling (grep) and touches no logic.

**Did not verify (user smoke, L-004):** none — no mic/`--device`/latency/Ctrl+C/multi-hour
surface touched (the `:659` status print is off those paths). Standing live-mic/soak debt
(T8.2 checklist) unchanged.

**Memory:** new **L-021** (apply human-facing style rules to human-facing surfaces only;
agent docs/comments/prompt exempt). No ADR — a style-rule application, not an architecture
choice (prior syncs' "record state, not an always" reasoning, L-008 spirit). Journal pruned
oldest (sync #2 → git). Roadmap untouched — **T8.3** remains the lowest open task next
session.

---

## 2026-06-16 — T8.2: live-mic smoke + soak checklist (`.agent/smoke.md`)

**Trigger:** `/session-prompt` (no override) → roadmap. T8.2 was the lowest open task;
user confirmed "Proceed with T8.2" over redirecting to an agent-verifiable T8 code task.

**Shipped (docs only, `live_stt.py` untouched):** new `.agent/smoke.md` — the standing
L-004 live-path debt made *runnable*. A 7-item numbered live-mic pass (`--list-devices`,
capture+meter, `--device N`, latency+VAD endpointing, translation cadence/interleave,
Ctrl+C mid-utterance flush+persist, `-o` persistence) + a soak section; each item states
its pass criterion **and** the backing `live_stt.py` observable. Every observable was
grounded against live code first (L-020): meter `{rms:.4f}{ q=}{ drop=}`
(`audio_callback`→`state.latest_ms`/`state.dropped`, `AUDIO_QUEUE_MAX`), endpoint
latency = `VAD_MIN_SILENCE_S` + ~0.1 s decode (D-010), shutdown flush in `worker`'s
`finally` + translator-drain-last (T1.4/T8.1), `emit_line` per-line flush, rotation at
`_turns % TRANSLATE_ROTATE_TURNS`, quota via the out-of-band `account/rateLimits/read`
RPC (D-011 — not surfaced in code). Soak = the 3 in-code signals + an external RSS check
with its code rationale (fixed-cap ring, `_RESAMPLE_CACHE`≤8, draining
`_notes`/`_pending`); no invented metric, no new code. Linked from PLAN T4.3 +
orientation "Smoke-test constraints" so the recurring "Did not verify (L-004)" disclaimer
points at a fixed list; indexed in `.agent/README.md`.

**Verified (agent):** 50 tests green (doc-only; pre-commit gate unaffected). No
code/CLI/output surface touched.

**Did not verify (user smoke, L-004):** the checklist is the deliverable but is
*user-executed* — T8.2 makes the live-path debt runnable, it does **not** close it. The
live path (mic, `--device`/`--list-devices`, latency feel, Ctrl+C flush+persist, `-o`,
multi-hour soak) still awaits an actual user pass through `.agent/smoke.md`; that standing
debt is unchanged, now with a fixed procedure to execute.

**Memory:** PLAN T8.2 → SHIPPED with a shipped-summary; orientation + PLAN T4.3 linked;
README indexed; journal pruned oldest (CLAUDE.md sync #1 → git). No new lesson/ADR —
authoring a smoke procedure from already-decided observables is not a new failure mode or
architecture choice (the observables trace to existing D-/T-records; L-020 already covers
"ground against live code"). **T8.3** is the lowest open task next session — its read-loop
test ideally lands with T8.4's scaffold (T8.3 synergy note).

---

## 2026-06-16 — T8.1: non-blocking worker-stop sentinel (shutdown deadlock fix)

**Trigger:** `/session-prompt` (no override) → roadmap. T8.1 was the lowest open
task; user confirmed "Proceed with T8.1" over redirecting to another T8 item.

**Shipped (live_stt.py + tests):** `run_session`'s shutdown `finally` replaced its
blocking `await audio_q.put(None)` with the evict-then-put idiom (`put_nowait`; on
`QueueFull`, `get_nowait` one stale block, retry) — a verbatim transplant of
`submit_sentinel`. Fixes the named deadlock: `worker()` is audio_q's only consumer
and returns on an in-worker exception (after `request_stop`); if the mic callback
filled audio_q to `AUDIO_QUEUE_MAX`, the blocking put parked the loop forever, and
since Ctrl+C routes to `request_stop` (not `KeyboardInterrupt`) the only escape was
SIGKILL with `-o` left unclosed. New `test_shutdown_sentinel_lands_on_full_audio_queue_without_blocking`
(tests/test_audio.py) drives the idiom on a synthetic full `asyncio.Queue(maxsize=4)`
under a 1 s `wait_for` (must not fire) + asserts the sentinel lands while the oldest
block is evicted (1,2,3 survive). No new symbol.

**Verified (agent):** 50 tests green (+1 over 49), ruff clean, `uvx pyright@1.1.410`
0 errors/warnings/informations. Normal Ctrl+C path is effect-identical (a put with
spare capacity lands immediately then breaks); only the dead-worker + full-queue case
changes (drops one block vs hangs to SIGKILL). Replay path structurally untouched —
replay.py drives `worker` directly, never `run_session`. (In-session LSP flagged
stale `capsys`/`time_info` "unused" notes from line shifts; the pyright CLI gate is
authoritative and clean.)

**Did not verify (user smoke, L-004):** live Ctrl+C-in-terminal mid-utterance flush
+ `-o` persistence — structurally unchanged but on the shutdown/signal path, so it
stays user-smoke (now itself the open T8.2 checklist task).

**Memory:** PLAN T8.1 → SHIPPED with a shipped-summary; journal pruned oldest (T8
roadmap entry → git). No new lesson/ADR — a clean transplant of an existing,
already-reasoned idiom is not a new failure mode or architecture choice (L-008 spirit:
the idiom + its rationale already live in code comments + the submit_sentinel
precedent; nothing generalizable beyond L-005/L-020). **T8.2** is the lowest open task
next session.

---

## 2026-06-16 — CLAUDE.md sync #3: cosmetic UI/UX-rule prose polish (no-op)

**Trigger:** User (`/session-prompt` override): "I updated the `CLAUDE.md`. If there is
any work you need to do in response, use this session to do so." Third consecutive
CLAUDE.md-sync session today; unlike the prior two syncs (substantive `.serena`/gitignore),
this diff (1+/1−) is a single-sentence **cosmetic prose polish** of the UI/UX rule: clause
reorder ("As it is largely human-authored" fronted) + de-parenthesized "...forced
optimizations present for LLM steering...". No new directive/constraint/fact/structural
requirement.

**Checked every surface the edit could touch (→ no work):** code/config — none (the UI/UX
rule has no code/config surface); live memory docs — no orientation/lessons/decisions entry
quotes the changed sentence (L-001 quotes a *different* CLAUDE.md rule, unchanged; journal
#2's "largely human-authored" is a historical diff description, left as-is); user-facing
README/CLI — unaffected (what compliance means is unchanged).

**Action:** committed the user's prose polish alone + this journal note. Confirmed scope
with the user first (chose "commit polish, end here" over recover-from-siblings / pivot-to-
T8.1). No code/config/doc change.

**Did not verify (user smoke, L-004):** none — no mic/`--device`/latency/Ctrl+C/multi-hour
surface touched. Standing live-mic/soak debt (T8.2) unchanged.

**Memory:** journal pruned oldest (T7 → git). No new lesson/ADR — a cosmetic doc tweak is
not a new failure mode or architecture choice (L-008 spirit: record state, not an "always").
Roadmap untouched — **T8.1** remains the lowest open task next session.

