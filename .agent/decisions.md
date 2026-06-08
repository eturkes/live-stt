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
**Trigger:** CLAUDE.md (memory-system rule) mandates a memory/notetaking/scratchpad system; user opted for `.agent/` with structured files.

**Decision:** Six top-level files (`README.md`, `SESSION_PROMPT.md`, `orientation.md`, `journal.md`, `lessons.md`, `decisions.md`) + one subdir (`scratch/`). All committed (per user choice). *(2026-06-04: dropped the `archive/` subdir — git history is the archive; see `journal.md` cap policy.)*

**Rationale:**
- Separation lets fresh agents load only what's relevant: `orientation.md` for facts, `lessons.md` for "what not to do," `decisions.md` for "why it's this way," `journal.md` for recent history.
- Committing everything means a fresh remote clone reproduces the agent's full context — critical given CLAUDE.md (long-time-horizon rule: "decomposed into steps with an unlimited number of fresh agent sessions").
- Scratch committed too: makes per-task reasoning visible in git history; cost is some clutter, which we mitigate by descriptive filenames + periodic pruning.

**Alternatives considered:**
- Single `NOTES.md`: rejected — flat append-only file makes targeted retrieval harder, conflates ephemeral and durable content.
- Hybrid (lessons grow into CLAUDE.md): rejected — CLAUDE.md is short-context meta-instructions; volume scales better in `.agent/`.

---

## D-005 — Spike reports stay at root, not archived

**Date:** 2026-05-16.

**Decision:** `SPIKE_REPORT.md` and `SPIKE_REPORT_BACKENDS.md` remain at repo root. Indexed from `.agent/orientation.md`.

**Rationale:** They are historical decision records (the "why" behind D-001, D-003). Moving them into `.agent/archive/` would scatter context and disrupt git blame trails. Root-level visibility is appropriate for foundational evidence.

**Revisit if:** More spike reports accumulate (>4) — then move to `spike/reports/`.

---

## D-006 — Don't refactor `live_stt.py` for "LLM-readability"

**Date:** 2026-05-16.

**Decision:** Leave `live_stt.py` as-is during this CLAUDE.md adoption pass.

**Rationale:** See `lessons.md` L-001. The code's optimization comments encode irreplaceable rationale; denser naming saves tokens at the cost of call-site readability. No specific failure mode is prevented by the proposed edits.

**Revisit if:** Specific agent-confusion incidents arise. Then target the confusing line, not the whole file.

---

## D-007 — Pre-commit hook via `.githooks/` + `core.hooksPath`, not the `pre-commit` framework

**Date:** 2026-05-18.

**Decision:** Pytest runs on every `git commit` via a project-local shell script at `.githooks/pre-commit`. Each clone opts in once with `git config --local core.hooksPath .githooks`. No `pre-commit` (Python framework) dep.

**Rationale:**
- The hook is one line of logic (`exec uv run pytest -q`). The `pre-commit` framework adds a config schema, a virtualenv-per-hook caching layer, and a network bootstrap — all of which buy nothing at this scope.
- Project-local script is committed → reproducible across clones. The `core.hooksPath` step is the only manual piece, called out in `README.md` and `.agent/orientation.md`.
- Keeps the dep tree shallow (CLAUDE.md local-installation rule: prefer project-scoped install/config). The dev group already has `pytest` and `ruff`; adding `pre-commit` would be a third tool.
- Bypassable with `git commit --no-verify` for genuine emergencies; both `CLAUDE.md` and the hook's own comment discourage it.

**Alternatives considered:**
- `pre-commit` framework — rejected (overkill, see above).
- Native `.git/hooks/pre-commit` only — rejected (not committed, not reproducible).
- Wrap the `core.hooksPath` config into `uv sync` somehow — rejected (no clean uv extension point; explicit one-time step is fine).

**Revisit if:** Hook grows beyond one or two commands, or we want hook chaining / language-specific hooks. Then `pre-commit` framework starts paying for itself.

**Side effect on PLAN.md:** Decision #2 ("test discipline: hook or aspirational?") resolved → hook. Removed from pending decisions table.

---

## D-008 — `.claude/settings.json` deny-list scope

**Date:** 2026-06-08.
**Trigger:** CLAUDE.md sentence (synced from sibling projects, see L-011): maintain `permissions.deny` `Read()` rules.

**Decision:** Deny Read on `.git/**`, `.venv/**` (137 M), `.env*` (secret — existence checks via Bash keep the key out of transcripts), `uv.lock` (160 K ≈ ~50 K tokens; `pyproject.toml` is the dependency surface), `LICENSE`, `spike/backends/cache/**` (auto-generated bench artifacts), and `**/__pycache__/** | .pytest_cache | .ruff_cache`.

**Kept readable:** `spike/` prototypes + `results.json/md` (needed when T-BACKENDS-001 unblocks), `SPIKE_REPORT*.md`, all of `.agent/` including `scratch/`.

**Notes:** Deny rules gate Read/Grep/Glob; Bash `cat`/`grep` stays available as the deliberate escape hatch (L-009-style `.venv/bin` shebang forensics remain possible). `ckc`'s settings additionally carry `env` (`CLAUDE_CODE_SUBAGENT_MODEL=opus`, `CLAUDE_CODE_EFFORT_LEVEL=max`) — intentionally not imported; outside the instruction's scope, flagged to user.

**Revisit if:** A denied path is needed repeatedly via Bash → narrow the rule rather than deleting the block.

**Amendment 2026-06-08:** The Bash escape hatch is dead in practice — Bash commands referencing deny-listed paths were refused twice (`grep` on `.env`, `ls` on `spike/backends/cache/`). Treat deny-listed paths as fully off-limits via **every** tool; ask the user instead of probing. Runtime reads by the app itself (python-dotenv loading `.env`) are unaffected.

---

## D-009 — No-API-key architecture: local STT + Codex-subscription translation, Gemini replaced outright

**Date:** 2026-06-08.
**Trigger:** User: "I don't want to use API keys, I want to use a Codex subscription." Supersedes **D-001** (Gemini Live backend) and the premise of **D-003** (its session machinery); kills T-BACKENDS-001 (was blocked on keys that will never arrive).

**Decision:** Replace the Gemini Live backend entirely (user chose outright replacement over keep-until-proven, accepting a regression window) with:
- **STT leg:** local open-source streaming JA engine, CPU-only (8-core/30 GB, no GPU). Engine chosen via spike-harness bench (T4.1).
- **Translation leg:** Codex subscription text surface (ChatGPT Pro-tier OAuth, ~1,600 msgs/5 h). Surface chosen via research (T4.2): prefer sanctioned persistent CLI surface over the undocumented `chatgpt.com/backend-api/codex/responses` endpoint.

**Rationale:**
- Eliminates all metered cost (Gemini path was ~$1.40/hr, L-003; flat-rate plan already paid for).
- Research findings (2026-06): Codex subscription auth is **text-only** programmatically — no clean audio path. Codex CLI `[realtime]` voice sessions exist (subscription-only, Whisper-backed) but are agentic-loop-wrapped, 60 s-clip-capped, unfit as a continuous JA transcriber — rejected as the engine ("option C").
- Fully-local option B (local MT) rejected by user in favor of GPT-5.5-quality translation at zero marginal cost.

**Consequences:**
- Endpointing/VAD responsibility returns to the app (Gemini's native VAD is gone) — engine-native preferred, silero-vad fallback.
- New runtime dependency on codex CLI + OAuth state (`~/.codex/auth.json`); interactive login is user-performed.
- Translation can lag or exhaust quota → graceful JA-only degradation is a hard requirement (T4.4).
- `google-genai`, reconnect/resumption machinery, `list_live_models.py` removed at T4.5.

**Revisit if:** a subscription-auth realtime audio surface becomes programmatically sanctioned (re-evaluate one-leg architecture); or local-engine JA quality on CPU proves insufficient at T4.1 bench (fallback: revisit option B variants or larger local models before any metered API).
