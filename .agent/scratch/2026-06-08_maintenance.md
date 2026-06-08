# Scratch — 2026-06-08 maintenance pass

User picked "maintenance pass" (no open PLAN tasks; live-mic smoke test pending on user).
Side finding at session start: pyright-lsp attached (D-008b verify CLOSED) and its first diagnostics show `.venv` not resolved.

## Items

1. **Pyright venv config.** numpy/sherpa_onnx/sounddevice `reportMissingImports` → pyright lacks `venvPath`/`venv`. Add `[tool.pyright]` to pyproject.toml (no separate pyrightconfig.json — fewer files). Re-triage after: L545 `_proc.close()` Optional-access (suspect: stdin close in `close()`), L718/720 `translator_task` Optional in shutdown path, L654 unused `time_info` (sounddevice callback signature — required positional, not noise we can drop).
   - Edit policy: L-001/D-006 — fix only what a future agent would misread or what is a real None-path; prefer config-level suppression for callback-signature noise.
2. **Dependency sweep** (CLAUDE.md update rule): `uv lock --upgrade` → review delta → `uv sync` → verify: import smoke, pytest, synthetic E2E (harness loaders) if sherpa/numpy moved. Keep BOTH sherpa-onnx pins (uv dep-skip quirk). `uv cache prune` after.
3. **codex CLI**: 0.137.0 → check latest; if update: one synthetic app-server turn re-verify (D-011 config still valid: Spark+low, tool-features off, developerInstructions).
4. **Hygiene**: orientation.md says "~750 lines" (actual 794 — fine, "~" holds; update if drifted past honesty), root `__pycache__` gitignore check, journal append+prune, commit.

## Verify ladder (after any dep move)
`uv run python -c "import live_stt"` → `uv run pytest` → harness synthetic decode (only if sherpa/onnx/numpy changed) → codex synthetic turn (only if codex changed).

## Out of scope
live_stt.py refactors beyond diagnostic-driven minimal fixes; anything touching mic paths (L-004 — user smoke test still pending).
