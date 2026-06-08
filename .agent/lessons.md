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

**Rule:** When the user makes a placement/config/preference choice, record it as *current state* and cite the authority (CLAUDE.md) — not as an `always`/`never` law. Reserve firm lessons for facts with a derivable technical cause (API quirks, build constraints, measured behavior). Ask before reversing any user-set arrangement. **Current state:** `compaction.sh` is vendored in-repo per CLAUDE.md ("the supplied `compaction.sh`"); the repo file is a byte-snapshot of the shared `$HOME/.claude/compaction.sh` (dual-mode: manual transcript-read + statusline stdin-JSON). Re-sync the repo copy if the shared tool changes; keep both unless the user says otherwise.

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
