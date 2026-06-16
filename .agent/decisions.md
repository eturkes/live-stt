# Decisions

Architectural/design choices with rationale. ADR-style but compact. Append-only; supersede via new entries pointing back at older IDs.

---

## D-001 — Backend = Gemini Live API (`gemini-3.1-flash-live-preview`)

**STATUS: SUPERSEDED by D-009 (2026-06-08).** Gemini backend removed at T4.5; kept for trajectory.

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

**STATUS: SUPERSEDED by D-009 (2026-06-08).** Its session machinery left with the Gemini backend at T4.5; kept for trajectory.

**Date:** Commit `aa5a466`.
**Source:** `PLAN.md` T3.2 shipped.

**Decision:** Outer reconnect loop around `client.aio.live.connect`, with `SessionResumptionConfig(handle=...)` and `ContextWindowCompressionConfig(sliding_window=...)`. Audio queue is bounded (`AUDIO_QUEUE_MAX=100`) so reconnect-gap buffering is bounded too.

**Rationale:** Live audio-only sessions cap at 15 min and the underlying WS times out at ~10 min. Without reconnect, sessions die. Resumption handle preserves conversation context across reconnects (2 h handle TTL).

**Deferred:** Client-side transcript replay (Approach B); entity-dict glossary injection (Approach C). Both wait on evidence of drift modes we haven't observed.

---

## D-004 — `.agent/` directory shape for memory system

**Date:** 2026-05-16.
**Trigger:** CLAUDE.md (memory-system rule) mandates a memory/notetaking/scratchpad system; user opted for `.agent/` with structured files.

**Decision:** Five top-level files (`README.md`, `orientation.md`, `journal.md`, `lessons.md`, `decisions.md`) + one subdir (`scratch/`). All committed (per user choice). *(2026-06-04: dropped the `archive/` subdir — git history is the archive; see `journal.md` cap policy.)* *(2026-06-08: `SESSION_PROMPT.md` removed — the bootstrap prompt became the `/session-prompt [TASK]` slash command at `.claude/commands/session-prompt.md`; see D-012.)*

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

**Amendment 2026-06-08 (b) — env-block question CLOSED:** `CLAUDE_CODE_SUBAGENT_MODEL=opus` + `CLAUDE_CODE_EFFORT_LEVEL=max` are already set **globally** (`~/.claude/settings.json` env block, container HOME; symlinked from `~/agents/claude/settings.json`) and apply to every project — per-project import is moot. ckc's project copy is a redundant no-op (identical values; project env only matters to override global). Future sessions: the max-model/max-effort rules are mechanically enforced; stop flagging this. Same session: `enabledPlugins.pyright-lsp@claude-plugins-official` added to project settings (user opted in; ckc parity). Server dep `pyright-langserver` 1.1.410 is user-level (`~/.local/bin`, pnpm tree at `~/.local/share/lsp-node`, predates this project). Plugin tools attach at session start → functional verification falls to the next session.

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

---

## D-010 — STT engine: reazonspeech-k2-v2 primary, parakeet-ja A/B alternate (both via sherpa-onnx)

**Date:** 2026-06-08. **Source:** T4.1 bench, `spike/backends/prototype_local.py` (engine kwarg serves both).

**Measured (5 clips, 8-core CPU, int8-enc+fp32-dec for k2):** both engines TTFT ≤0.10 s post-audio-end (Gemini baseline: 1.21 s mean), $0/hr, decode totals 0.03–0.21 s. Transcripts near-exact; k2 errors: 文→分 homophone (paused), ジェミニ→ゼミニ; parakeet: ジェミニ→`jeミinapi`, numeral style. Comparable to Gemini's own `paused` drift (採寸分).

**Decision:** default **k2-v2** (RTF 0.054 vs 0.106, 148 MB vs 625 MB, Apache-2.0, JA-specialist, kanji-rich output); keep parakeet selectable (slightly better published CER, won the 文 homophone) — same integration, near-zero cost to retain.

**Two findings that bind T4.3:**
1. **Gemini-TTS boundary artifact:** multi-sentence single-segment TTS clips collapse continuous decode (each sentence decodes perfectly in isolation; model's own 13.4 s natural-speech test WAV is flawless). Bench clips re-rendered per-sentence with 0.7 s real silences (scenarios.py comment). Production (real mic, real pauses + VAD) is unaffected; synthetic continuous TTS is not a valid robustness test for these models.
2. **Silero VAD onset clipping:** segments open 0.2–0.7 s late (こんにちは→はい); sherpa exposes no pad field. Fix: re-slice each segment from the fed-sample stream with `VAD_PRE_PAD_S = 0.4` lead-in (prototype_local.py). T4.3 needs a bounded ring buffer for the same.

**Revisit if:** live-mic smoke tests show accuracy/latency regressions vs these synthetic results; or Reazon ships its planned native-streaming JA model (drop chunking); or proper-noun drift (foreign names) proves disruptive in practice.

---

## D-011 — Translation leg: codex app-server, Spark+low, developerInstructions, tool features off

**Date:** 2026-06-08. **Source:** T4.2 bench, `spike/backends/codex_client.py` (carries the T4.4 pattern).

**Surface:** persistent `codex app-server` subprocess, newline-delimited JSON-RPC/stdio: `initialize` → `thread/start` (one thread/session, `ephemeral:true`, `sandbox:"read-only"`, `approvalPolicy:"never"`, `personality:"none"`) → `turn/start` per block → `item/agentMessage/delta` → `turn/completed`. Usage from `thread/tokenUsage/updated`; quota from `account/rateLimits/read`.

**Config (all four bind):**
1. **Model `gpt-5.3-codex-spark` + `effort:"low"`** (Spark floor; `minimal` rejected). Fallback **`gpt-5.4-mini` + `effort:"none"`** — p50 1.18 s, also 8/8 clean.
2. **Disable tool-injecting features at spawn** — `web_search="disabled"`, `features.{image_generation,browser_use,browser_use_external,computer_use,apps}=false`. THE latency lever: tool schemas were ~15 K tokens/turn; p50 3.15 s → **0.99 s** (ttft 0.94 s; Gemini baseline 1.21 s). Each enabled tool also 400s at low/minimal effort.
3. **Instructions via `developerInstructions` on thread/start** (content: codex_ws/AGENTS.md text). 4/4 injection-resistant (imperatives/role-reassignment in speech → translated, not obeyed). AGENTS.md-in-cwd mode REJECTED: Spark answered "delete all files…" as a request and suggested `rm -rf`. `baseInstructions` pinned server-side (stock prompt stays, ≈18 K in).
4. **Per-thread marginal cost ≈ 180 uncached in + 7–60 out tokens/turn** (prefix 5.1 K cached from turn 2; turn 1 ≈ 2.7 s uncached). ~50 bench turns moved primary 5 h window 0 → 0 %. Plan reports **`prolite`** (user had said Pro) — Spark entitled regardless; headroom ample.

**T4.4 notes:** thread grows ~30 tok/turn — rotate thread every ~100 turns (one 2.7 s uncached turn) or on `modelContextWindow` pressure; cwd → any empty dir (sandbox read-only makes it inert); errors arrive as `error` notifications with `willRetry` — degrade to JA-only per D-009.

**Revisit if:** Spark latency/entitlement changes on plan change; or sustained-session quota burn contradicts the ~0 % observation; or OpenAI sanctions a leaner instructions channel on subscription auth.

**Amendment 2026-06-15 (T6):** Re-verified non-breaking at codex CLI **0.139.0** (bench
was 0.137.0) — config (`gpt-5.3-codex-spark`+`low`+features-off+`developerInstructions`)
still valid; clean JA->EN via a synthetic `CodexTranslator` turn (agent-verifiable per
orientation; `auth.json` present). Also hardened `_read_loop`: `readline()` is now
wrapped so a >64 KiB line (asyncio `StreamReader` default limit) or broken transport
routes into the existing EOF cleanup -> immediate clean JA-only instead of waiting out
the per-turn `wait_for` timeout (D-009 strengthened at the input boundary). Security
audit verdict: no remotely-exploitable surface.

---

## D-012 — Bootstrap prompt is the `/session-prompt [TASK]` slash command

**Date:** 2026-06-08. **Trigger:** user — convert the reusable session prompt to a native slash command with per-session roadmap override.

**Decision:** The bootstrap prompt lives at `.claude/commands/session-prompt.md` (invoked `/session-prompt [TASK]`), not as `.agent/SESSION_PROMPT.md` (deleted, D-004 amended). Body is the former prompt verbatim minus the human-paste preamble; the trailing "USER STEERING" HTML comment is replaced by an `$ARGUMENTS` slot in § "What to do right now":
- **`/session-prompt` (blank):** follow `PLAN.md` — lowest-numbered open task.
- **`/session-prompt <TASK>`:** `<TASK>` (task ID or free text) overrides the roadmap for that session; bootstrap reads still run.

**Rationale:** Slash commands are Claude Code's native reusable-prompt mechanism — discoverable in-session, no copy/paste, no manual comment-appending for steering. One `$ARGUMENTS` arg unifies the previous two modes (roadmap vs. hand-edited steering) into the documented `<TASK>` contract.

**Refs updated:** `.agent/README.md` (pointer + table row), D-004 (file list). Historical `journal.md` mentions of `SESSION_PROMPT.md` left as-is (accurate at their dates).

**Amendment 2026-06-08 — renamed `/session` → `/session-prompt`:** command file `session.md` → `session-prompt.md` (`git mv`, history preserved); the title/body/verify lines above already reflect the new name. The original name was `/session` (created earlier the same day). Live refs repointed (`.agent/README.md`, D-004 amendment); L-014's context line keeps its dated `session.md` path as accurate-at-date history.

**Verify (user-side):** `/session-prompt` and `/session-prompt T2.2` must appear/expand correctly in a fresh session — slash-command registration is not agent-verifiable.

---

## D-013 — `.serena/` (Headroom) tracking split + deny-list; scoped-commit convention

**Date:** 2026-06-15. **Trigger:** CLAUDE.md edit (synced template, L-011) adding two rules: (a) Headroom wraps the session and introduces `.serena/`, where `project.yml` is the tracked LSP config while `cache/`, `project.local.yml`, `memories/` "should be ignored both by you and Git"; (b) end-of-turn commits use the scoped-commit form (scopedcommits.com).

**Decisions:**
1. **Track `.serena/project.yml` + `.serena/.gitignore`**, leave `cache/`/`project.local.yml`/`memories/` untracked. The nested Serena-generated `.serena/.gitignore` ignores all three — `memories/` was missing and was added this session.
2. **"Ignored by you" → `permissions.deny` `Read()`** on `.serena/cache/**`, `.serena/memories/**`, `.serena/project.local.yml` — same mechanism CLAUDE.md prescribes for low-value paths (D-008). `project.yml` + `.serena/.gitignore` stay readable.
3. **Project memory stays `.agent/`, not Serena memories.** `.serena/memories/` is Serena's default store, unused here (D-004); deny+ignore keep it out of git and agent context. Avoid `mcp__serena__*_memory` tools for project notes.
4. **Scoped commits** (`Scope: summary`): already the de-facto style (`Tooling:`/`Maintenance:`/`Settings:` + co-author trailer already match scopedcommits.com); now documented in orientation step 8. No retroactive rewrite.

**Rationale:** `project.yml` is portable shared LSP config (no secrets/abs paths — verified before tracking); committing it gives fresh clones working LSP without re-bootstrapping Serena. The other three are machine-local / regenerable / redundant-with-`.agent/`. The deny-list mirrors git-ignore so neither git nor agent context carries them.

**Revisit if:** Headroom changes `.serena/` layout; or a denied `.serena` path is needed repeatedly via Bash (narrow the rule per D-008).

**Amendment 2026-06-16 (CLAUDE.md sync) — memories→root gitignore + `ignored_paths` sync:** CLAUDE.md was updated to (a) home the `.serena/memories/` ignore in the **repo-root** `.gitignore` and (b) sync Serena's `ignored_paths` with the non-gitignored "do-not-read" set. Two changes vs. decisions 1–2 above:
- **Memories ignore moved nested→root, sole home.** `.serena/memories/` now lives in the repo-root `.gitignore`; the `/memories` line was removed from the Serena-generated `.serena/.gitignore` (which still carries `/cache` + `/project.local.yml`). Rationale: Serena regenerates the nested file and once dropped `memories` entirely (the original D-013 trigger) — the root entry is durable across regenerations. Verified: root `.gitignore` now matches `.serena/memories/`; nothing was tracked there.
- **`.serena/project.yml` `ignored_paths` = `[uv.lock, LICENSE]`.** Exactly the deny-listed paths (D-008) that git does *not* ignore, so Serena's file tools (which honor `.gitignore` but not the Claude `permissions.deny`) would otherwise surface them. Gitignored deny entries need no listing (`ignore_all_files_in_gitignore: true` covers them). **Keep these three surfaces in sync:** a new non-gitignored deny entry must also land in `ignored_paths`.

**Amendment 2026-06-16 (CLAUDE.md sync #2) — `.serena/` committed as is; memories un-ignored:** CLAUDE.md replaced the prior "home `.serena/memories/` in the repo-root `.gitignore`" instruction with: `.serena/` "comes with its own gitignore file and can be committed as is", plus the new fact "Serena contains a memory system, but it has been disabled globally". Verified: `~/.serena/serena_config.yml` `excluded_tools:` lists all six memory tools (`read_memory`/`write_memory`/`list_memories`/`delete_memory`/`edit_memory`/`rename_memory`) → Serena never writes `.serena/memories/`. Reverses the prior amendment's nested→root move (user chose "fully as is"):
- **Removed `.serena/memories/` from the repo-root `.gitignore`** (left a one-line breadcrumb comment there to stop the entry oscillating nested↔root across sessions). The nested `.serena/.gitignore` is unchanged (`/cache` + `/project.local.yml`; no `/memories`) and is now the *only* `.serena/` gitignore — root no longer reaches in.
- **`.serena/memories/` is empty and now un-ignored by git.** Safe: the memory system is disabled globally (the dir stays empty) and `.agent/` is the sole store; a memory file appearing later would show as untracked, not silently commit. It stays **Read-denied** in `.claude/settings.json` — a deliberate do-not-read-but-not-gitignored split, which CLAUDE.md sanctions ("some committed files are not worth reading, some gitignored ones are").
- **Unchanged:** `.serena/project.yml` + `.serena/.gitignore` stay tracked; `ignored_paths: [uv.lock, LICENSE]` + the 3-surface sync rule for non-gitignored deny entries still stand.

---

## D-014 — Deterministic WAV replay as the local-pipeline regression harness; bench harness retired

**Date:** 2026-06-15. **Trigger:** user (`/session-prompt` override) — make live-stt regression-testable (not add features): a deterministic WAV replay/eval path through the exact VAD + RingBuffer + sherpa decode pipeline, live-mic path unchanged. Three design choices were confirmed with the user before coding.

**Decisions:**
1. **Replay drives the real `live_stt.worker`, not a copy.** Added an optional `on_segment=None` instrumentation hook to `worker()`; the live-mic path passes nothing → behavior byte-for-byte unchanged. `replay.py` passes a collector capturing `(start, n, seg_len, decode_s, text)` per popped VAD segment (incl. empty-text, for segmentation fidelity). Chosen over freeze-and-reimplement precisely because a reimplemented loop is what drifted in the old `prototype_local.py`.
2. **Retire the whole bench harness.** Deleted the runnable `.py` (`harness`/`bench`/`scenarios`/`prototype_local` + dead metered prototypes + `codex_client`/`translate`). `prototype_local.py` was a superseded copy of the production loop (its `fed_slice` was marked "production wants RingBuffer instead"); `codex_client.py`'s pattern lives in `CodexTranslator` (D-011); the metered prototypes died with T-BACKENDS-001 (D-009). Kept the gitignored WAV corpus (`cache/`), the historical `*.md` records (D-005 precedent), and `codex_ws/AGENTS.md` (D-011 instructions source).
3. **Corpus stays gitignored; goldens are characterization snapshots.** The 5 bench WAVs stay in the deny-listed `spike/backends/cache/` (not promoted to tracked fixtures); `tests/test_replay.py` skips when they (or the models) are absent. Expected values = the *current* real-pipeline output in tracked `tests/replay_goldens.json` (engines carry stable known quirks per D-010 — ジェミニ→ゼミニ, 文→分 — so golden-master beats asserting idealized refs). Asserted surface = segment count + per-segment text + boundary (±0.1 s); CPU-variable decode latency/RTF is reported only.

**Rationale:** a regression test of a *copy* of the pipeline tests the copy, not the product. The hook makes replay exercise the identical loop the mic feeds, so a future agent editing VAD/worker/ring code gets an immediate deterministic signal; retiring the drift-prone harness removes the second copy entirely.

**Practical note:** the cache is deny-listed, so its path cannot appear on a Bash command line, but a script's *runtime* file reads are unaffected (D-008 amendment) — `gen_replay_goldens.py` and the test construct the cache path internally and read the WAVs at runtime.

**Coverage split (the deliverable):** replay covers WAV→resample→VAD segmentation→ring re-slice→decode→transcript + per-segment latency + JA-only degradation; mic capture, device selection, live-latency feel, Ctrl+C flush, multi-hour quota, and live Codex cadence stay user-only. Tabulated in `PLAN.md` § T5 "Coverage split"; authoritative user-only list in `orientation.md` § Smoke-test constraints.

**Revisit if:** parakeet goldens added (T5.2 → key goldens by engine); a real-recorded corpus lands (T5.3); or `worker()` grows further instrumentation needs (consider a small event object over positional hook args).

**Amendment 2026-06-15 (T5.2):** First revisit-if resolved — parakeet goldens added. Shape chosen: **engine-first** (`engine → clip_id → {n_segments, segments, …}`), the redundant per-clip `engine` field dropped (engine is the key). `gen_replay_goldens.py` loops `ENGINES` with a per-engine `check_models` skip-warn; `test_replay.py` parametrizes over `(engine, clip_id)` with per-engine gating. 35 tests green (+5 parakeet). Parakeet snapshot confirms the D-010 quirks (`jeミinapi`, `2つ目`, lowercase `api`) and the `文`-homophone win. T5.3 (real-recorded corpus) remains the open revisit-if.

**Amendment 2026-06-15 (T5.3):** Second revisit-if resolved — a real-recorded
corpus landed, via **web-fetch, not mic capture**: L-004 blocks only the microphone,
so the agent downloaded real clips (CLAUDE.md network access). 7 Common Voice 8.0 JA
clips (CC0; 5 single + 2 concatenated-with-real-silence, mirroring the D-010
per-utterance+silence method with real voices) fetched from the ungated Parquet
mirror `japanese-asr/ja_asr.common_voice_8_0` via the HF datasets-server `/rows` API
(few labeled samples, no multi-GB pull; MP3 decoded by soundfile, no ffmpeg). Wiring:
`tests/fetch_real_clips.py` (committed provenance tool; pinned revision + row indices)
writes WAVs to the deny-listed cache (internal path, L-016) and a tracked
`tests/real_clips.json` manifest; `gen_replay_goldens.py` merges that manifest with
the inline synthetic `CLIPS` (chosen over inlining fetched multibyte transcripts:
single source, drift-free, reproducible). 49 tests green (+14). Real-acoustic goldens
expose engine divergence the TTS corpus could not (松井/松居, バック/パック,
午後七時/午後7時) and confirm katakana フィリピン decodes cleanly. T5 revisit-ifs:
none open (T5.1-T5.3 all shipped).
