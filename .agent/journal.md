# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

---

## 2026-06-08 — Env-block question closed (already global); pyright-lsp enabled project-scope

**Trigger:** User picked "resolve env-settings question" from session-start options, then clarified: "I thought those settings were set globally?" — correct. Global `~/.claude/settings.json` (container HOME; symlink → `~/agents/claude/settings.json`) already carries `CLAUDE_CODE_SUBAGENT_MODEL=opus` + `CLAUDE_CODE_EFFORT_LEVEL=max` (plus MAX_OUTPUT_TOKENS=128000, agent-teams flag, etc.). Two sessions of "import ckc's env block?" flagging were moot — nobody had checked the global file. CLOSED → D-008 amendment (b). ckc's project copy = redundant no-op.

**pyright-lsp (user opted Enable):** `enabledPlugins.pyright-lsp@claude-plugins-official` now in project `.claude/settings.json`. First attempt used the CLI's default **user** scope and silently edited the GLOBAL settings → reverted (`uninstall -s user`, then `install -s project`; leftover empty key hand-cleaned at the symlink target) → **L-012**. Server dep already satisfied user-level: `~/.local/bin/pyright-langserver` (pyright 1.1.410, pnpm tree from ckc setup). Global file diff vs pre-session: key order only (CLI rewrites reorder).

**Verify next session:** pyright LSP tools attach + give diagnostics on `live_stt.py` (plugins load at session start — unverifiable from the session that flips the flag).

**Flagged for user:** (1) ckc's `installed_plugins.json` record still points at pre-move `~/Documents/pro/ckc` — ckc may need `install -s project` re-run from its new path. (2) ckc's project env block duplicates the global one; prune at will. (3) T4.3/T4.4 live-mic smoke test still pending (L-004).

---

## 2026-06-08 — T4.2–T4.5 shipped: re-architecture complete, Gemini fully removed

**Trigger:** User: "OK I authenticated Codex" — unblocked T4.2; ran the remaining T4 series to completion in one session.

**T4.2 (→ D-011):** `codex app-server` JSON-RPC/stdio surface benched via `spike/backends/codex_client.py`. Binding config: Spark+`low` (`minimal` rejected), tool-features off at spawn (`web_search="disabled"` + 5 `features.*=false`) — THE latency lever, p50 3.15→**0.99 s**/turn (Gemini baseline 1.21 s); instructions via `developerInstructions` (4/4 injection-resistant; AGENTS.md-in-cwd mode REJECTED — obeyed "delete all files" as a request). Marginal ~180 uncached in + 7–60 out tok/turn; ~50 turns moved the 5 h window 0→0%. Plan reports `prolite` (user said Pro); Spark entitled regardless. Fallback: mini+`none`, p50 1.18 s.

**T4.3 + T4.4:** `live_stt.py` rewritten — mic → resample → silero VAD + 60 s `RingBuffer` (absolute indexing; pre-pad 0.4 s re-slice) → executor decode → numbered `JA n:`/`EN n:` lines; `CodexTranslator` (warm-up turn absorbs ~3 s uncached cost; thread rotation @100 turns; degradation: missing CLI → JA-only at start, 3 consecutive failures → JA-only for session, backlog >50 drops oldest). RingBuffer phase bug caught by tests (oversized append ignored ring phase; fixed with phase-aligned two-segment write). 22 tests green; synthetic E2E: STT reproduces T4.1 bench exactly, 9/9 EN ordered @~1 s cadence.

**T4.5:** `google-genai` + `python-dotenv` removed (25 pkgs; spike `load_dotenv` lines dropped), `list_live_models.py` + `.env` deleted, README/orientation/SESSION_PROMPT rewritten for the new architecture, D-001/D-003 superseded, L-002/L-003 historical, L-004 rescoped.

**Did not verify (user smoke-test, L-004):** live mic capture, `--device`/`--list-devices`, Ctrl+C mid-utterance flush + translator drain, real-time latency feel, multi-hour quota burn. Note: secondary weekly Codex window already at 54% from user's other usage.

**Carried open question:** import `ckc`'s `env` settings (`CLAUDE_CODE_SUBAGENT_MODEL=opus`, `CLAUDE_CODE_EFFORT_LEVEL=max`)? Still unanswered.

---

## 2026-06-08 — T4 pivot (no API keys) + T4.1 shipped: local STT engine selected

**Trigger:** User rejected the T-BACKENDS-001 premise mid-bootstrap: "I don't want to use API keys, I want to use a Codex subscription." Locked via AskUserQuestion: local STT + Codex-subscription translation (A), Gemini replaced outright, ChatGPT Pro tier. → **D-009**; T-BACKENDS-001 superseded; T4.1–T4.5 added to PLAN.

**Research (2 subagents, compressed → `.agent/scratch/2026-06-08_T4-research-notes.md`):** no native frame-sync streaming open JA model exists (June 2026) — all paths are VAD-chunked offline decode. Codex leg: persistent `codex app-server` (JSON-RPC/stdio) + `gpt-5.3-codex-spark` (Pro-only, separate rate pool); raw backend-api POST rejected (Cloudflare TLS fingerprinting).

**T4.1 shipped (→ D-010):** `prototype_local.py` (sherpa-onnx offline + silero VAD, harness contract). Bench: **k2-v2** primary / parakeet A/B; TTFT ≤0.10 s vs Gemini 1.21 s; near-exact JA; $0/hr. Two T4.3-binding findings: Gemini-TTS boundary artifact (scenarios.py medium/long re-rendered per-sentence + 0.7 s silences) and silero onset clipping (`VAD_PRE_PAD_S=0.4` re-slice; production wants ring buffer). Models in gitignored `models/`.

**Env quirks:** `uv add sherpa-onnx` skipped its declared dep `sherpa-onnx-core` (carries libonnxruntime) — explicit add fixed import. Deny-list now verified enforced (Read(uv.lock) refused) **including via Bash** (twice) → D-008 amended: deny paths are off-limits via every tool; ask, don't probe.

**Next (T4.2, user action required):** codex CLI 0.137.0 installed (`~/.local/bin/codex`), not yet authenticated. User must run `codex login --device-auth` interactively (may need "Allow device code login" at chatgpt.com Settings→Security). Then: latency bench, Spark entitlement check, instruction-control verification, quota accounting.

---

## 2026-06-08 — CLAUDE.md sync (permissions.deny sentence) + `.claude/settings.json` created

**Trigger:** User: "I updated the CLAUDE.md" — but live-stt's copy was byte-identical to HEAD (mtime 06-03, predating the claim; no stash/reflog trace). Forensics: sibling projects `ckc` (06-07), `lean-cds`+`rehab` (06-08) all carry one added sentence → "Maintain `permissions.deny` `Read()` rules in `.claude/settings.json`…". live-stt had been missed in the user's project-by-project propagation. Lesson → L-011.

**Changes:** Synced the sentence into CLAUDE.md (byte parity with siblings). Created `.claude/settings.json` denying Read on `.git/.venv/.env*/uv.lock/LICENSE/spike cache/tool caches` (scope rationale → D-008; `ckc`-style `./`-anchored rules). Gitignored `.claude/settings.local.json`. Orientation file-map updated (+`.claude/settings.json` row, `.env` read-denied note).

**Verified:** `jq` parses settings.json; pre-commit pytest on commit. **Did not verify:** deny-rule enforcement — rules load for new sessions; next session should confirm `Read(.venv/…)`/`Read(uv.lock)` are refused and Bash fallback works.

**Flagged for user:** `ckc` settings also carry `env` `CLAUDE_CODE_SUBAGENT_MODEL=opus` + `CLAUDE_CODE_EFFORT_LEVEL=max` (mechanically enforces CLAUDE.md's max-model-subagents rule). Not imported — say the word and it lands here too.

