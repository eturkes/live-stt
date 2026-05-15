# 2026-05-16 — Code Audit for LLM-Readability

**Trigger:** User opted into "touch code as well" during CLAUDE.md adoption.

**Scope:** `live_stt.py`, `list_live_models.py`, `tests/test_audio.py`.

## Findings

### `live_stt.py` (518 lines)

Code already optimized along multiple axes:

- **Constants at the top**, name-explicit (`SEND_RATE`, `METER_FULL_SCALE_RMS`, `RECONNECT_BACKOFF_MIN_S`).
- **Comments encode rationale, not behavior.** Examples worth preserving:
  - Lines 33-38: meter-bar precomputation explained ("avoids two string allocations per tick").
  - Lines 58-64: `_RESAMPLE_CACHE` contract documented ("Returned buffer is reused across calls — callers must consume before the next call").
  - Lines 73-76: integer-decim fast-path explained ("strided slice beats np.interp by ~20x").
  - Lines 82-86: rationale for skipping `np.interp`'s binary search.
  - Lines 100-102: `take(out=)` rejected with cost number ("~40% slower on small arrays").
  - Lines 134-138: `np.minimum`/`np.maximum` over `np.clip` rationale ("~30% faster on small arrays").
  - Lines 197-198: queue coalescing race-freeness justified.
  - Lines 222-227: `python-genai#1224` workaround called out by issue number.
  - Lines 263-273: shutdown vs reconnect partial-buffer flush asymmetry justified.
- **Naming is domain-aligned.** `resample`, `sender`, `receiver`, `meter`, `emit_block` map 1:1 to the audio-pipeline mental model.
- **No dead code.** Past sessions ripped out config, VAD, calibration, worker pool. No leftover scaffolding.

### Candidate edits considered and rejected

1. **Denser variable names** (`r` for `resample`, `pcm` for `pcm16_bytes`, `s` for `session`). Rejected: saves ~30 tokens of file size at the cost of every call-site read. CLAUDE.md #15 endorses unconventional patterns *only when they prevent a failure mode* — none identified.
2. **Strip optimization comments.** Rejected: those comments are the *highest-value* comments in the file — they encode irreproducible benchmark findings. A future agent without them would re-derive them or, worse, "simplify" `audio[::decim]` back to `np.interp`.
3. **Inline `build_config()` / `_install_signal_handlers()` / `_wait_for_stop_or_reconnect()`.** Rejected: each isolates a noisy concern (pydantic config, OS-specific signal handling quirks, two-event wait) from the main flow. Inlining would crowd `run_session()` without reducing total LoC.
4. **Replace `sys.stderr.write` with `logging`.** Out of scope here — this is already tracked as PLAN.md T2.3.
5. **Replace `class State` with `@dataclass(slots=True)`.** Rejected: the class has lazy-init event fields (`stop_event`, `reconnect_event`) and method behavior (`request_stop`, `request_reconnect`). A dataclass would split the state shape from the helpers without simplifying either.

### `list_live_models.py` (small fix applied)

Reflowed the long `methods = ... or ... or []` line to satisfy ruff E501. Functional behavior unchanged. Smoke-tested: prints current Live-capable models.

### `tests/test_audio.py` (no changes)

Comprehensive coverage of pure functions. Already documents the `_RESAMPLE_CACHE` buffer-reuse contract in test docstrings (e.g. `test_resample_returns_shared_output_buffer`). Tests pass: 23/23.

## Conclusion

**No changes to `live_stt.py`.** Promoted finding to `.agent/lessons.md` L-001 and `.agent/decisions.md` D-006. Single-line fix to `list_live_models.py:19` (long line, originally flagged in `SPIKE_REPORT.md`).

## Verification

```sh
uv run ruff check list_live_models.py live_stt.py tests/     # clean
uv run pytest -q                                              # 23 passed
uv run python -c "import live_stt"                            # ok
uv run python list_live_models.py                             # ok (4 models)
```

**Did not verify (user smoke-test):** No audio paths or session loops touched. Nothing for user to smoke-test from this session.
