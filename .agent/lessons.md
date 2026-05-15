# Lessons

Generalizable rules harvested from past mistakes or non-obvious findings. Format: short title, context, mistake/finding, rule.

When a lesson supersedes an earlier one, mark the earlier as `STATUS: SUPERSEDED by <id>` and keep both — future agents need to see the trajectory.

---

## L-001 — Don't over-edit already-optimized code in the name of LLM-readability

**Context:** `live_stt.py` carries dense rationale comments explaining each numpy optimization (cache-reuse, ufunc choice over `np.clip`, allocation avoidance, integer-decim fast path). When asked to "make code more LLM-friendly," instinct is to densify naming and strip comments.

**Finding:** Those comments are exactly what `CLAUDE.md` rule #15 endorses ("unconventional patterns or be poorly documented by human standards if you prefer it that way" — both directions). They explain *why* a non-obvious line exists. Stripping them would force every future agent to re-derive the optimization rationale from benchmarks. Denser naming (`r` for `resample`, `pcm` for `pcm16_bytes`) saves ~30 tokens at the cost of every future call-site read.

**Rule:** When auditing code for "LLM-readability," the test is: *would removing this make a future agent more or less likely to reason correctly about the code?* If a comment encodes a non-obvious invariant or optimization rationale, **keep it**. If naming is already domain-aligned (`resample`, `sender`, `receiver`), **keep it**. Only edit if you can name a specific failure mode the edit prevents.

---

## L-002 — `python-genai#1224` workaround

**Context:** `session.receive()` in `google-genai` exits its async iterator on `turn_complete`, even though the underlying WebSocket session is still alive and more turns will arrive.

**Finding:** Without an outer `while`, the receiver coroutine returns silently after the first turn, leaving the session idle but connected. Reconnect doesn't fire because nothing has errored.

**Rule:** Always wrap `session.receive()` in `while not state.stopping and not state.should_reconnect:` so the iterator is re-entered after each `turn_complete`. See `live_stt.py:222-262`. If/when upstream fixes the bug, the outer loop becomes a no-op — safe to keep.

---

## L-003 — Audio-output tokens are billed even when discarded

**Context:** Native-audio Live models (`gemini-3.1-flash-live-preview` and family) only emit the `AUDIO` response modality. The app reads `output_audio_transcription.text` and discards the audio bytes.

**Finding:** Google bills the audio-output tokens (~$0.018/min at list price, April 2026) regardless. No knob exists to opt out. This is the dominant term in cost-per-hour (~$1.40/hr at list).

**Rule:** Cost discussions about the Live path must account for the audio-output bill. If Google ships a text-only Live SKU in the future, revisit. Until then, REST is ~10× cheaper per minute for non-real-time use cases.

---

## L-004 — Mic and Gemini-rate-limit paths are agent-unverifiable

**Context:** The agent runs in a sandbox without a real microphone or sustained API budget. CLAUDE.md #3 says to ask the user about ambiguities.

**Finding:** Any change that touches `sd.InputStream`, `audio_callback`, real-time latency, or sustained-session behavior must be flagged for user smoke-test. Unit tests cover pure functions (`resample`, `pcm16_bytes`, `emit_block`) but not the live path.

**Rule:** End every session that touched the audio or session-loop code with a "**Did not verify**" list naming each unverifiable behavior the user should smoke-test. Do not claim "done" without that disclaimer.

---

## L-005 — Avoid abstractions in `live_stt.py`

**Context:** Past versions had a config dict, calibration logic, VAD module, and a worker pool. All were ripped out (commits `3930a8e`, `57d7634`).

**Finding:** The codebase trajectory is consistent: less code wins. Each abstraction added cost more in agent confusion than it saved in flexibility, for a single-user personal tool.

**Rule:** Default to inlining. Do not add config systems, DI, plugin interfaces, or class hierarchies unless the task explicitly requires them. Three similar lines beat a premature abstraction.
