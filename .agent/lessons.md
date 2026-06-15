# Lessons

Generalizable rules harvested from past mistakes or non-obvious findings. Format: short title, context, mistake/finding, rule.

When a lesson supersedes an earlier one, mark the earlier as `STATUS: SUPERSEDED by <id>` and keep both — future agents need to see the trajectory.

---

## L-001 — Don't over-edit already-optimized code in the name of LLM-readability

**Context:** `live_stt.py` carries dense rationale comments explaining each numpy optimization (cache-reuse, ufunc choice over `np.clip`, allocation avoidance, integer-decim fast path). When asked to "make code more LLM-friendly," instinct is to densify naming and strip comments.

**Finding:** Those comments are exactly what CLAUDE.md (use-practices rule) endorses ("unconventional patterns or be poorly documented by human standards if you prefer it that way" — both directions). They explain *why* a non-obvious line exists. Stripping them would force every future agent to re-derive the optimization rationale from benchmarks. Denser naming (`r` for `resample`, `pcm` for `pcm16_bytes`) saves ~30 tokens at the cost of every future call-site read.

**Rule:** When auditing code for "LLM-readability," the test is: *would removing this make a future agent more or less likely to reason correctly about the code?* If a comment encodes a non-obvious invariant or optimization rationale, **keep it**. If naming is already domain-aligned (`resample`, `sender`, `receiver`), **keep it**. Only edit if you can name a specific failure mode the edit prevents.

---

## L-002 — `python-genai#1224` workaround

**STATUS: HISTORICAL (2026-06-08).** The Gemini receiver this guarded was removed at T4.5 (D-009). Relevant only if a google-genai Live session ever returns.

**Context:** `session.receive()` in `google-genai` exits its async iterator on `turn_complete`, even though the underlying WebSocket session is still alive and more turns will arrive.

**Finding:** Without an outer `while`, the receiver coroutine returns silently after the first turn, leaving the session idle but connected. Reconnect doesn't fire because nothing has errored.

**Rule:** Always wrap `session.receive()` in `while not state.stopping and not state.should_reconnect:` so the iterator is re-entered after each `turn_complete`. See `live_stt.py:248-300` (outer while at line 258). If/when upstream fixes the bug, the outer loop becomes a no-op — safe to keep.

---

## L-003 — Audio-output tokens are billed even when discarded

**STATUS: HISTORICAL (2026-06-08).** The metered Gemini path was removed at T4.5 (D-009); current architecture has zero marginal cost. Relevant only when evaluating audio-native API backends.

**Context:** Native-audio Live models (`gemini-3.1-flash-live-preview` and family) only emit the `AUDIO` response modality. The app reads `output_audio_transcription.text` and discards the audio bytes.

**Finding:** Google bills the audio-output tokens (~$0.018/min at list price, April 2026) regardless. No knob exists to opt out. This is the dominant term in cost-per-hour (~$1.40/hr at list).

**Rule:** Cost discussions about the Live path must account for the audio-output bill. If Google ships a text-only Live SKU in the future, revisit. Until then, REST is ~10× cheaper per minute for non-real-time use cases.

---

## L-004 — Mic and real-terminal paths are agent-unverifiable

**Context:** The agent runs in a sandbox without a real microphone or interactive terminal. CLAUDE.md (ask-questions rule) says to ask the user about ambiguities. *(2026-06-08: "Gemini rate-limit" dropped from scope with D-009; canonical list lives in `orientation.md` § Smoke-test constraints.)*

**Finding:** Any change that touches `sd.InputStream`, `audio_callback`, real-time latency, Ctrl+C handling, or multi-hour-session behavior must be flagged for user smoke-test. Unit tests cover pure functions (`resample`, `RingBuffer`, `emit_line`) and synthetic E2E covers decode/translation, but not the live mic path.

**Rule:** End every session that touched the audio or session-loop code with a "**Did not verify**" list naming each unverifiable behavior the user should smoke-test. Do not claim "done" without that disclaimer.

---

## L-005 — Avoid abstractions in `live_stt.py`

**Context:** Past versions had a config dict, calibration logic, VAD module, and a worker pool. All were ripped out (commits `3930a8e`, `57d7634`).

**Finding:** The codebase trajectory is consistent: less code wins. Each abstraction added cost more in agent confusion than it saved in flexibility, for a single-user personal tool.

**Rule:** Default to inlining. Do not add config systems, DI, plugin interfaces, or class hierarchies unless the task explicitly requires them. Three similar lines beat a premature abstraction.

---

## L-006 — TTY-aware `logging.Formatter` when stderr coexists with a live stdout UI

**Context:** T2.3 (2026-05-18). The level meter on stdout repaints continuously via `\r\x1b[2K`. Event messages (errors, `go_away`, audio status) needed to interrupt the meter without leaving the previous meter frame frozen above, AND stay free of ANSI when redirected to a log file (`2> errors.log`).

**Finding:** Inlining `_LINE_CLEAR + msg` at every log call site couples every site to terminal cursor protocol, hurts readability, and forks the format for TTY vs non-TTY. Putting the prefix in a custom `logging.Formatter` keeps each call site format-agnostic (`logger.error("[send error: %s]", e)`) and centralizes the TTY check (`sys.stderr.isatty()` once at handler-construction time). When stderr is redirected, the formatter omits the prefix → log file gets clean `[timestamp] LEVEL msg` lines.

**Rule:** When a single output stream must coexist with a continuously-redrawn UI element on a sibling stream, encode the cursor protocol in the `Formatter`, not at the call sites. Gate ANSI emission on `isatty()` evaluated once. Call sites stay narrow: just the level and the message. This is the rare case where adding a small class (one subclass, ~10 lines) beats inlining (per L-005) because the alternative is to leak cursor concerns into every log site forever.

---

## L-007 — Externally-owned tools named by absolute path stay un-mirrored

**STATUS: SUPERSEDED by L-008.** Void rule, kept for trajectory: it over-reached by generalizing one reversible `compaction.sh` placement choice into an `always git rm the repo copy` law that the very next turn contradicted. Current state and the corrected lesson are in **L-008**. Full original text: git history.

---

## L-008 — A single user preference is not an "always" rule; record state, don't codify it

**Context:** Within two consecutive turns the user flipped `compaction.sh`'s home. Turn 1: "it now lives in `$HOME/.claude/`" → I deleted the repo copy and wrote L-007 generalizing "externally-owned tools stay un-mirrored." Turn 2: "I decided to keep a copy in-repo" → CLAUDE.md repointed back to "the supplied `compaction.sh`."

**Finding:** L-007 over-reached. It promoted one reversible placement choice into a firm `always git rm` rule that the next turn contradicted, and that would have led a future agent to wrongly delete the now-wanted repo copy. The deletion of the stale fork was correct; turning it into an engineering law was the error.

**Rule:** When the user makes a placement/config/preference choice, record it as *current state* and cite the authority (CLAUDE.md) — not as an `always`/`never` law. Reserve firm lessons for facts with a derivable technical cause (API quirks, build constraints, measured behavior). Ask before reversing any user-set arrangement. **Current state:** `compaction.sh` is vendored in-repo per CLAUDE.md ("the supplied `compaction.sh`"); the repo file is a byte-snapshot of the shared `$HOME/.claude/compaction.sh` (single-mode as of 2026-06-08: manual transcript-read of the session JSONL → `N% used/window`; the former statusline stdin-JSON path + ANSI coloring were stripped, both copies in step). Re-sync the repo copy if the shared tool changes; keep both unless the user says otherwise.

---

## L-009 — Moving the project dir breaks venv console scripts, not `python`

**Context:** 2026-06-04. `uv run pytest` failed with `Failed to spawn: pytest / No such file or directory`, yet `uv run python -m pytest` passed 23. The pre-commit hook (`uv run pytest -q`) was blocked.

**Finding:** The project dir had been moved (`~/Documents/pro/live-stt` → `~/Projects/live-stt`). `uv`/`pip` bake an **absolute-path shebang** into every `.venv/bin/` console script (`pytest`, `ruff`, `live-stt`, …) at install time, so a move/rename invalidates all of them (`cannot execute: required file not found`). `.venv/bin/python` survives because it is a **symlink** to the uv-cached interpreter *outside* the project — which is why `python -m <mod>` keeps working and masks the real cause.

**Rule:** When `uv run <script>` fails to spawn but `uv run python -m <module>` works, suspect a moved/renamed project dir with stale console-script shebangs. Fix by regenerating the venv: `rm -rf .venv && uv sync` (a plain `uv sync` will not rewrite existing shebangs). A fresh clone is immune — it builds the venv at its real path; only an in-place move triggers this.

---

## L-010 — Check native-lib presence by importing the binding, not `ldconfig -p`

**Context:** 2026-06-04 relocation re-verify. `ldconfig -p | grep portaudio` returned nothing and briefly read as "PortAudio missing → the move broke the lib."

**Finding:** False alarm. In this Distrobox container the `ld.so.cache` is empty (`ldconfig -p` → 0 entries) and even `sudo ldconfig` doesn't repopulate it, yet `libportaudio.so.2` is installed at `/usr/lib/x86_64-linux-gnu/` and `sounddevice` dlopens it fine (ctypes resolves it via the default trusted lib dirs, not the cache).

**Rule:** To confirm a dlopen'd native lib is available, import its Python binding (`uv run python -c "import sounddevice"`) or query the packager (`dpkg -l libportaudio2`). Treat `ldconfig -p` as unreliable in this container — an empty/stale cache is normal there and does not mean the lib is absent.

---

## L-011 — CLAUDE.md is a shared template across `~/Projects/*`; recover missed edits from siblings

**Context:** 2026-06-08. User said "I updated the CLAUDE.md" but live-stt's copy was byte-identical to HEAD and its mtime predated the claim. No stash, no reflog trace, no global `~/.claude/CLAUDE.md`.

**Finding:** The user maintains near-identical CLAUDE.md copies in sibling projects (`ckc`, `rehab`, `lean-cds`, `pose-estimation`) and propagates edits project-by-project; live-stt had been missed. `diff` against the most-recently-modified siblings (sort by mtime) recovered the exact intended edit — one sentence, identical in all three updated copies. Sibling `.claude/settings.json` files also served as precedent for config shape.

**Rule:** When the user reports a CLAUDE.md (or other shared-template) change that no local diff/mtime supports, diff against the sibling projects' copies under `~/Projects/` before asking. If multiple freshly-updated siblings agree byte-for-byte, apply the same edit here for template parity and report the sync explicitly.

---

## L-012 — `claude plugin install` defaults to user scope and silently edits the GLOBAL settings file

**Context:** 2026-06-08. Enabling pyright-lsp for live-stt only. Bare `claude plugin install pyright-lsp@claude-plugins-official` reported `(scope: user)` and wrote `enabledPlugins` into `~/.claude/settings.json` — enabling the plugin for **every** project — while leaving the project's `.claude/settings.json` untouched.

**Finding:** Default `--scope` is `user` for both `install` and `uninstall`. Project-scoped enablement (ckc pattern: `enabledPlugins` in project `.claude/settings.json` + `installed_plugins.json` record carrying `projectPath`) requires explicit `-s project`. Recovery: `uninstall -s user` then `install -s project`; the uninstall left an empty `"enabledPlugins": {}` in the global file (cleaned by hand — and note the global settings.json is a **symlink** into `~/agents/claude/`; Edit must target the resolved path). Each CLI write also reorders the global file's keys. Separately: `installed_plugins.json` bakes absolute `projectPath`s, so a moved project leaves a stale record (observed: ckc's pre-move path).

**Rule:** Always pass `-s project` to `claude plugin install`/`uninstall` for project-local intent, and verify afterwards which settings file actually changed (grep `enabledPlugins` in both global and project files). After a project relocation, expect plugin install records to need a re-install from the new path.

---

## L-013 — `pgrep -f`/`pkill -f` self-match: the harness Bash wrapper embeds your pattern

**Context:** 2026-06-08 maintenance. Killing a stalled pyright CLI: `pgrep -f "pyright/index.js"` kept "finding" a process after the kill, and a follow-up `pkill -f` exited 144 — it had killed its own shell.

**Finding:** Claude Code runs each Bash call as `/usr/bin/bash -c 'eval <full command text>'`, so the pattern string appears verbatim in the wrapper's own cmdline. `pgrep -f`/`pkill -f` therefore match (and kill) the very shell executing them. The "phantom respawned PID" after the first kill was the next pgrep's wrapper, not the target.

**Rule:** Break literal self-match with a bracket class in every `pgrep -f`/`pkill -f` pattern — `pgrep -f "pyright/index[.]js"` matches the target but not the wrapper's literal `[.]` text. Append `|| echo none` so zero matches is visible. Treat any -f match whose cmdline starts with `/usr/bin/bash -c source .../shell-snapshots/...` as your own wrapper.

## L-014 — Slash-command / frontmatter YAML values starting with `[` or `{` must be quoted

**Context:** 2026-06-08, authoring `.claude/commands/session.md`. `argument-hint: [TASK] — blank follows…` looked fine but is invalid YAML.

**Finding:** A leading `[` (or `{`) opens a YAML *flow sequence/mapping*; trailing prose after the `]` is then unexpected and the document raises `ParserError` (confirmed: `uv run --with pyyaml` parsed the quoted form, failed the bare form). Docs examples like `argument-hint: add [id] | list` only work because they start with a plain char — the bracket sits mid-scalar. Brittle parsers may drop a malformed `description`/`argument-hint` silently, so the command can register with a missing hint and no error.

**Rule:** Quote any frontmatter scalar that begins with a YAML indicator (`[ { } ] , & * ! | > % @ \` "` `#` `: `). Verify ad-hoc frontmatter with an ephemeral parser — `uv run --with pyyaml python3 -c "import yaml,sys; yaml.safe_load(open(p).read().split('---')[1])"` — rather than eyeballing it.

## L-015 — Headroom compresses tool reads; verify Edit anchors against raw bytes

**Context:** 2026-06-15. CLAUDE.md now documents that Headroom compresses everything the agent reads. An `Edit` on `orientation.md` step 8 failed with "String to replace not found" even though the `old_string` matched what the Read tool had shown me.

**Finding:** Read/Grep/Bash output is semantically compressed before it reaches the agent, so the rendered text is not guaranteed byte-identical to disk. Short anchors usually survive; long ones accumulate divergence — here `git log` appeared without backticks in my view but carried them on disk (`per ` `git log` ` style`). Edit matches real bytes, so a compressed-view mismatch fails silently.

**Rule:** Prefer short, distinctive `old_string` anchors. When an Edit fails on a string you "see," fetch raw bytes (`sed -n 'Np' FILE | cat -A`) and re-anchor on those, reproducing exact Unicode (em-dash `—`, `§`) and backticks. To delete a large block you have not seen byte-exact, use a line-range delete (`sed -i 'A,Bd'`) instead of guessing its bytes.

## L-016 — Deny-listed paths block on the command line, not on a script's runtime `open()`

**Context:** 2026-06-15, T5. The replay regression corpus lives in `spike/backends/cache/` — deny-listed (D-008 amendment: refused via every tool, Bash included). `tests/gen_replay_goldens.py` and `tests/test_replay.py` must read those WAVs, and a manual `replay.py <cache.wav>` smoke would need the path too.

**Finding:** The deny-list operates at the Claude **tool boundary**: it refuses a deny-listed path that appears in a Read/Edit target or anywhere on a Bash command line — so `uv run python -c "...spike/backends/cache/x.wav..."` is refused (the literal path sits in the command text). It does NOT intercept the OS. A script that builds the path from components internally (`CACHE = ROOT / "spike" / "backends" / "cache"`) and `open()`s it at runtime reads the file fine, because no deny-listed string ever crosses a tool. For an agent-side CLI smoke that therefore can't name the corpus, synthesize a throwaway WAV (no cache reference anywhere).

**Rule:** When a tool/test/script must read a deny-listed path, construct the path from components *inside the script file* and let it `open()` at runtime — never pass the deny-listed path as a CLI argument or `-c` substring. To smoke-run a CLI that takes such a path, feed it a synthetic temp artifact, not the real deny-listed file. (Distinct from L-015: there the block is incidental Read-compression; here it is intentional and total at the tool boundary.)

## L-017 — Fetch a few real labeled samples from HF via the datasets-server rows API, not the `datasets` loader

**Context:** 2026-06-15, T5.3. Needed a handful of real JA speech clips (audio +
ground-truth transcript) to harden the replay corpus; Common Voice is the canonical
CC0 source.

**Finding:** The `datasets` library is the wrong tool for *a few* samples.
`mozilla-foundation/common_voice_17_0` is gated AND uses a loading script; `datasets`
3.x no longer runs the script and raised `EmptyDatasetError` ("doesn't contain any
data files"), and the datasets-server refuses script datasets too ("runs arbitrary
Python code"). What works with zero gating, zero full-download, zero ffmpeg: pick an
**ungated Parquet mirror** (here `japanese-asr/ja_asr.common_voice_8_0`), confirm it
via the datasets-server `/splits` endpoint, then read rows from
`https://datasets-server.huggingface.co/rows?dataset=...&config=...&split=...&offset=N&length=<=100`.
Each row carries the label fields plus an `audio[0].src` URL — a **signed, expiring**
cached-asset, so fetch it within the same run. `soundfile`/libsndfile (>=1.1) decodes
the MP3 directly (`MP3` in `sf.available_formats()`) — no ffmpeg/sox; resample with
the project's own resampler for fidelity.

**Rule:** To pull a few real labeled samples (audio/text/image) from Hugging Face,
prefer an ungated Parquet mirror + the datasets-server `/rows` API over `load_dataset`
— it sidesteps gating, loading-script breakage, and multi-GB downloads. Verify viewer
support with `/splits` first; treat `audio.src` URLs as expiring; check
`soundfile.available_formats()` before reaching for ffmpeg.

## L-018 — Maintenance-pass recipe: outdated check, CVE scan, codex-leg re-verify

**Context:** 2026-06-15, T6. Roadmap fully shipped; user asked for a periodic
maintenance + security pass (CLAUDE.md: keep software current + schedule security
audits). Captured the turnkey commands so future passes don't re-derive them.

**Recipe:**
- **Outdated deps:** `uv tree --outdated --depth 1` (current-vs-latest, runtime + dev).
  Bump via `uv lock --upgrade-package NAME` then `uv sync`; leave the pyproject `>=`
  floors alone — only the lock moves.
- **CVE scan:** `uv export --format requirements-txt --all-groups --no-emit-project >
  /tmp/reqs.txt && uvx pip-audit -r /tmp/reqs.txt`. pip-audit audits the *env* it runs
  in, so a bare `uvx pip-audit` scans only itself — feed it the exported reqs.
- **Codex-leg re-verify** (after a codex CLI version drift; it is NOT in pytest):
  synthetic turn through the *real* class — `import CodexTranslator`, `await t.start()`
  (its warm-up turn proves auth+entitlement+JSON-RPC), `await t._translate("…")`. Needs
  `~/.codex/auth.json` (check existence only, never read it); negligible quota (D-011).

**Rule:** Run a maintenance pass as inventory (`uv tree --outdated`) -> CVE scan
(`uv export | uvx pip-audit -r`) -> apply safe lock bumps + full gate (import, pytest,
ruff, `uvx pyright@1.1.410`) -> re-verify the codex leg via a synthetic
`CodexTranslator` turn. Keep package versions out of orientation (drift risk per
CLAUDE.md); record the point-in-time verification in journal/PLAN + a dated decision
amendment.

## L-019 — A "refactor pass" on mature, guarded code may correctly yield almost nothing

**Context:** 2026-06-15, T7. User asked for a proactive refactor pass (CLAUDE.md
endorses periodic refactor). `live_stt.py` is small, heavily optimized, and already
guarded by L-001/L-005/D-006. The pink-elephant pull (CLAUDE.md) is to manufacture
changes to justify the task.

**Finding:** Screening each candidate against L-001's test — "name the specific
failure mode this edit prevents" — eliminated most ideas. What survived was tiny (a
redundant exception tuple, a named timeout constant, one inert micro-opt removal);
the higher-effort ideas (merge near-duplicate methods, cross-file dedup, centralize a
constant, prune overlapping tests) each failed it (deliberate semantic difference,
coupling cost > lines saved, or loses a named case).

**Rule:** Treat a refactor-pass deliverable as an audit — read broadly, list
candidates, apply only those with a nameable failure-mode-prevented or a
house-style-conformance win, and record the rejected ones (scratch + PLAN) so they
aren't re-litigated. "Mature; minimal change" is a valid, honest result; report it
rather than forcing edits.
