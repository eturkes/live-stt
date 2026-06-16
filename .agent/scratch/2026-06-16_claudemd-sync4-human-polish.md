# 2026-06-16 — CLAUDE.md sync #4: human-facing prose polish

**Trigger:** `/session-prompt` override — "I updated the CLAUDE.md. If there is any work
you need to do in response, use this session to do so." Uncommitted diff (2+/2−).

## Diff analysis

- **Change 1 (biases/"pink elephant" rule):** rephrase + scope broadening ("constructing
  systems" → "writing text you will later read, especially if interpreted as a prompt").
  Core directive (positive framing) unchanged. **No actionable surface** — governs how I
  phrase agent-facing text.
- **Change 2 (UI/UX rule):** drops the "CLAUDE.md is a good example of user-facing
  language" sentence; adds a concrete directive: *"When writing for a human audience,
  prefer to use hyphens over other kinds of dashes, enumerate flexibly, and vary
  comparative constructions."* Also "user-facing"→"human-facing", "+smells". **This is the
  only change implying work.**

## Scope (user chose "Full human-facing polish")

In scope (human-facing): `README.md` prose + user-facing CLI strings in `live_stt.py`.
Exempt (rule carves out "underlying code, including comments, are tailored towards your
ease of use"): all `.agent/*`, `PLAN.md`, `CLAUDE.md`, every `live_stt.py` comment, and
the Codex `developerInstructions` prompt string (`:68`, model-facing + benched, D-011).
L-015 notes em-dashes are deliberate in agent docs.

## live_stt.py findings

- Only one human-facing dash: `:659` `print("Translation: unavailable — JA-only (see log)")`.
- Module docstring (`__doc__` → argparse `--help`): clean (no dashes/tics/cliches) → leave.
- argparse help strings (`:773-796`), startup prints: terse, clean → leave.
- No test references the status string (grep) → safe to edit.

## Edits

**live_stt.py:659** — em-dash → parens, mirroring the sibling `disabled (--no-translate)`:
`unavailable — JA-only (see log)` → `unavailable (JA-only, see log)`.

**README.md** (13 dash sites + de-cliche; comparative/enumeration tics: README was already
clean of "not just X but Y" / forced triads, so those directives yielded little):
- L6: em-dash → parenthetical; "degrades gracefully" → "falls back" (de-jargon/cliche).
- L36: code-literal synced to the new `:659` string.
- L52-56: pipeline steps `**X** — ` → `**X.** ` (kills the rigid em-dash separator; a real
  sequence stays a uniform numbered list — "enumerate flexibly" targets manufactured triads,
  not legit sequences). L54 en-dash `0.2–0.7` → `0.2-0.7`.
- L60: long sentence's inner em-dash aside split into its own short sentence (varies rhythm).
- L114: em-dash → sentence split.
- L118: em-dash → comma ("...decode, no mic or translation").
- L125: em-dash → "since" clause.
- L129: em-dash → comma.
- L150: en-dash `~7–60` → `~7-60`.

## Verify

`grep -P '[\x{2013}\x{2014}]' README.md` → 0 (human-facing); `import live_stt`; `uv run
pytest` (50 green, non-functional change); ruff + `uvx pyright@1.1.410`. The `:659` string
is a startup status char-swap off the mic/signal paths → no new L-004 smoke surface.

## Memory

Journal entry (sync #4) + prune to ≤4. No new lesson (L-011/L-015 already cover
shared-template + Headroom anchors; this is applying an existing rule, not a new failure
mode). No new ADR (a style-rule application, not an architecture choice). Commit:
CLAUDE.md + README.md + live_stt.py + this scratch + journal.
