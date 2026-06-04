# Session Journal

Chronological log of agent sessions. Most recent at the top. One section per session, prefixed with ISO date and a one-line topic.

**Cap: keep the ≤4 most-recent entries.** Each session, after appending its own entry, deletes the oldest entries beyond 4. Pruned entries are not moved anywhere — git is the archive: recover any past entry with `git log -p -- .agent/journal.md` or `git show <commit>:.agent/journal.md`. See `.agent/README.md` § Pruning.

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

---

## 2026-05-18 — Proactive audit: doc-drift, spike lint

No PLAN tasks were actionable (only blocked T-BACKENDS-001), so ran a self-audit: fixed line-anchor drift (`live_stt.py:248-300`), made `spike/` ruff-clean (15→0), gitignored `spike/backends/results.md`, documented the sentinel-on-full-queue recovery at `live_stt.py:440-442`. Finding: the `.agent/` system held up under second-look; the only durable drift was line-anchors/counts (cheap to mop up, no process change warranted). Full detail: git `fd7e3d4`.

**Open:** T-BACKENDS-001 blocked on `DEEPGRAM_API_KEY` / `OPENAI_API_KEY`.

---

## 2026-05-18 — T2.3 shipped: structured logging for errors

Routed 5 runtime `sys.stderr.write` sites through a module `logger` (INFO=go_away reconnect, WARNING=audio-status from the PortAudio thread, ERROR=send/recv/session). Custom `_StderrFormatter` prepends `_LINE_CLEAR` only when `sys.stderr.isatty()`, so the stdout level meter coexists with terminal logs while redirected stderr stays ANSI-free (`2> errors.log` gets clean `[ts] LEVEL msg`). Rationale → `L-006`. GEMINI_API_KEY preflight `print` left as-is (out of scope). Full detail: git `dd1a505`.

**Open (user smoke-test):** level meter clears cleanly when a real log record fires mid-session; `live-stt > /dev/null 2> errors.log` stays ANSI-clean under a live mic.
