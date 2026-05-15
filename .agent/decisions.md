# Decisions

Architectural/design choices with rationale. ADR-style but compact. Append-only; supersede via new entries pointing back at older IDs.

---

## D-001 — Backend = Gemini Live API (`gemini-3.1-flash-live-preview`)

**Date:** ~April 2026 (commit `57d7634`).
**Source:** `SPIKE_REPORT.md` (T3.1 spike).

**Decision:** Use `client.aio.live.connect` with bidirectional streaming; raw PCM16 at 16 kHz; native VAD for turn boundaries.

**Rationale:** TTFT ~1.0 s post-audio vs ~6–10 s for the prior REST chunked pipeline. Code shrank from 270 → 240 lines (since grown back). Concurrency simplified from "1 capture + N worker threads + two queues" to "single async loop + bounded queue."

**Cost:** ~10× more expensive per minute at list price (~$1.40/hr) due to billed audio-output tokens. Acceptable for personal-tool usage.

**Revisit if:** Google ships a text-only Live SKU (cost drops ~3×); or volume becomes high enough that REST behind a `--rest` flag pays for itself.

---

## D-002 — Single-file architecture in `live_stt.py`

**Decision:** All app logic lives in `live_stt.py`. No package layout, no module split.

**Rationale:** Personal tool, one developer (agent), one user. Module splitting buys reuse and isolation we don't need and costs agent context per file load. Single-file fits comfortably in one read.

**Constraint:** When the file passes ~1000 lines, revisit — at that scale, navigation cost may exceed split cost.

---

## D-003 — Persistent session + reconnect + resumption handle

**Date:** Commit `aa5a466`.
**Source:** `PLAN.md` T3.2 shipped.

**Decision:** Outer reconnect loop around `client.aio.live.connect`, with `SessionResumptionConfig(handle=...)` and `ContextWindowCompressionConfig(sliding_window=...)`. Audio queue is bounded (`AUDIO_QUEUE_MAX=100`) so reconnect-gap buffering is bounded too.

**Rationale:** Live audio-only sessions cap at 15 min and the underlying WS times out at ~10 min. Without reconnect, sessions die. Resumption handle preserves conversation context across reconnects (2 h handle TTL).

**Deferred:** Client-side transcript replay (Approach B); entity-dict glossary injection (Approach C). Both wait on evidence of drift modes we haven't observed.

---

## D-004 — `.agent/` directory shape for memory system

**Date:** 2026-05-16.
**Trigger:** `CLAUDE.md` #8 mandates a memory/notetaking/scratchpad system; user opted for `.agent/` with structured files.

**Decision:** Six top-level files (`README.md`, `SESSION_PROMPT.md`, `orientation.md`, `journal.md`, `lessons.md`, `decisions.md`) + two subdirs (`scratch/`, `archive/`). All committed (per user choice).

**Rationale:**
- Separation lets fresh agents load only what's relevant: `orientation.md` for facts, `lessons.md` for "what not to do," `decisions.md` for "why it's this way," `journal.md` for recent history.
- Committing everything means a fresh remote clone reproduces the agent's full context — critical given CLAUDE.md #7 ("decomposed into steps with an unlimited number of fresh agent sessions").
- Scratch committed too: makes per-task reasoning visible in git history; cost is some clutter, which we mitigate by descriptive filenames + periodic pruning.

**Alternatives considered:**
- Single `NOTES.md`: rejected — flat append-only file makes targeted retrieval harder, conflates ephemeral and durable content.
- Hybrid (lessons grow into CLAUDE.md): rejected — CLAUDE.md is user-controlled and short-context; volume scales better in `.agent/`.

---

## D-005 — Spike reports stay at root, not archived

**Date:** 2026-05-16.

**Decision:** `SPIKE_REPORT.md` and `SPIKE_REPORT_BACKENDS.md` remain at repo root. Indexed from `.agent/orientation.md`.

**Rationale:** They are historical decision records (the "why" behind D-001, D-003). Moving them into `.agent/archive/` would scatter context and disrupt git blame trails. Root-level visibility is appropriate for foundational evidence.

**Revisit if:** More spike reports accumulate (>4) — then move to `spike/reports/` or `.agent/archive/spike-reports/`.

---

## D-006 — Don't refactor `live_stt.py` for "LLM-readability"

**Date:** 2026-05-16.

**Decision:** Leave `live_stt.py` as-is during this CLAUDE.md adoption pass.

**Rationale:** See `lessons.md` L-001. The code's optimization comments encode irreplaceable rationale; denser naming saves tokens at the cost of call-site readability. No specific failure mode is prevented by the proposed edits.

**Revisit if:** Specific agent-confusion incidents arise. Then target the confusing line, not the whole file.
