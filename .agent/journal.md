# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

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

---

## 2026-06-04 — Relocation re-verify: prior fix held, nothing broken

**Trigger:** User steering — "project recently relocated, fix anything that broke." Expected an L-009 recurrence (stale venv shebangs).

**Finding:** Nothing broken; last session's `rm -rf .venv && uv sync` held. Project is now reached at `/run/host/home/eturkes/Projects/live-stt` (host fs mounted into the Debian/Distrobox container; container HOME=`/var/home/eturkes/debian`, interpreter from the uv cache under it). `.venv/bin/*` shebangs correctly point at this `/run/host/...` path → console scripts spawn; L-009's diagnostic (`uv run <script>` fails but `python -m` works) did **not** trigger.

**Verified green:** `pytest -q` → 23 passed, `uv sync` clean (36 pkgs), `ruff check` clean, `import live_stt` / `import sounddevice 0.5.5` OK, `.env`+GEMINI_API_KEY present, `core.hooksPath=.githooks` survived the move, `live-stt --help` parses, no tracked file embeds a stale abs path.

**ldconfig false-alarm → L-010:** `ldconfig -p | grep portaudio` is empty (container `ld.so.cache` unpopulated; `sudo ldconfig` doesn't repopulate) though `libportaudio.so.2` is installed at `/usr/lib/x86_64-linux-gnu/` and dlopens fine — use the Python import / `dpkg`, not `ldconfig`, to check native-lib presence here.

**Did not verify (user smoke-test, L-004):** live mic capture, `--device`/`--list-devices` enumeration, real-time latency, Ctrl+C. A post-move failure, if any was observed, most likely lives in one of these agent-unverifiable paths — report the symptom.

---

## 2026-06-04 — Token-efficiency overhaul of the memory system

**Trigger:** User steering — "make working in this codebase more token-efficient." Measured per-session bootstrap read cost at ~14.6K tokens; `journal.md` was 43% of it (25K chars, 9 entries) and the only unbounded sink. `archive/` had never been used in 9 sessions; 5 of 9 entries were CLAUDE.md-sync / `compaction.sh` churn whose durable facts already live in `lessons.md` / `decisions.md` / `orientation.md`.

**Changes (user-approved scope — Aggressive + git-as-archive):**
- **Journal:** deleted 7 churn/sync entries; kept the 2 most-recent real-work entries (audit, T2.3), compressed to nugget + open follow-ups. Added this header cap.
- **Removed `.agent/archive/`** entirely — it duplicated git. Updated `README.md` (Files table + Pruning), `decisions.md` D-004 (shape) and D-005 (stale archive pointer).
- **Standing cap:** journal keeps ≤4 entries; enforced at `orientation.md` how-to-work step 7 + stated in `README.md` Pruning + `SESSION_PROMPT.md` working agreement.
- **De-dup:** smoke-test-constraints list is now canonical in `orientation.md` only; `SESSION_PROMPT.md` references it instead of restating.
- **Trimmed** superseded `L-007` to a one-line pointer to `L-008` (kept ID + trajectory note; full text in git).
- **`CLAUDE.md` untouched** (user declined that option).

**Result:** bootstrap read cost ~14.6K → ~9.6K tokens/session (−35%; 58.6K→38.3K chars, ~5.1K tok saved/session). Journal alone cut 82% (25K→4.4K chars) and is now self-limiting at ≤4 entries instead of growing ~2–3K chars/session unbounded.

**Also fixed (env regression, no tracked diff):** `uv run pytest` and `uv run live-stt` both failed to spawn — the project had been moved `~/Documents/pro/live-stt → ~/Projects/live-stt`, leaving every `.venv/bin/` console-script shebang pointing at the dead old path. `.venv/bin/python` (a symlink to the uv-cached interpreter) still resolved, so `python -m pytest` masked it. Rebuilt: `rm -rf .venv && uv sync`. Lesson → L-009.

**Verified:** `git show HEAD~1:.agent/journal.md` recovers all 9 pre-prune entries (git-as-archive confirmed). After venv rebuild: `uv run pytest -q` → 23 passed; `uv run python -c "import live_stt"` → OK. No app code touched.

**Did not verify (user smoke-test):** `uv run live-stt` against a live mic — the entry-point shebang is now fixed and the launcher regenerated, but mic capture / device enumeration stay agent-unverifiable (L-004). The dir move had broken the launcher, so a real run is the end-to-end confirmation it's back.
