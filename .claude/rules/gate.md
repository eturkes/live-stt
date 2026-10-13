---
paths: ["gate.py", "tests/test_gate.py", "tests/test_law_consistency.py", ".agent/spec-lint"]
---

# Gate checks — mechanics, coverage limits, firing records

The law each check enforces lives in `toolchain.md`; this file holds how each check fires and where
it stops seeing.

## Steps

**Every step ships the input that fires it, and the gate runs those inputs on itself.**
`tests/test_gate.py` seeds one defect of each step's own class into a throwaway tree and asserts
`gate FAILED: <step>` — pytest an `assert False`, ruff-check an `F401`, ruff-format bad spacing,
pyright and pyright-tests `x: int = "s"`, secrets an AWS-shaped key, import a module that raises.
The seeds ride the pytest step, so a green gate is simultaneously its own positive control: a step
that stopped being able to fail turns the gate red instead of quietly green. The skip rule seeds
four more pytest trees: `@pytest.mark.skip("flaky")`, the same tree under `addopts = "--no-summary
--runxfail"`, a `PYTEST_ADDOPTS`-suppressed summary the tally still catches, and a
declared-`absent:` twin as the positive control — without that twin, a check that flagged EVERY skip
would pass the seed. A path holding `": "` is locked too, since the location ends at its line
number. One scope hole that seeding cannot see gets its own lock — the seeded tree holds a single
root-level file, so an over-grown `SCAN_SKIP_DIRS` or `tests/` exclusion would still catch it while
the real tree went unscanned ⇒ `test_secret_scan_reaches_the_real_tree` pins the walk to production,
a nested file and a dotdir file.

The pytest step runs `-rsxX` and `gate.py`'s `undeclared_demotions` reads the summary. **The step
owns its effective options, because `--no-summary` and `--runxfail` each empty that input in
silence:** `-o addopts=` drops project-configured ones, `PYTEST_ADDOPTS` is stripped from the step
env, and the closing tally is cross-checked against what the summary named, so whatever else
suppresses the detail turns the step red instead of quiet — the tally is read off the LAST tallying
line, so a plugin trailer printed after it cannot displace it. **Coverage limits, all three loud or
out of reach rather than silently green:** the checker reads emitted `SKIPPED`/`XFAIL`/`XPASS`
lines, so a DELETED case and a demoted tier leave no trace and stay under the approval law alone;
in-tree pytest configuration outranks it, since a `conftest.py` setting `config.option.runxfail`
reconfigures pytest itself and this repo ships no conftest at all; and a directory named with a
`:<digits>: ` inside it false-REDS a declared skip, pytest's `location: reason` line being genuinely
ambiguous there.

Two locks own the `ruff-format` Markdown scope: `test_the_format_step_reaches_a_markdown_code_block`
seeds a misformatted block and demands `gate FAILED: ruff-format`, and
`test_the_upstream_files_stay_out_of_the_format_step` copies production's own `pyproject.toml` into
the tree, so the carve-out under test is the shipped one, with a `doc.md` twin as the positive
control against an exclusion that swallowed every `.md`.

`spec-lint` runs `.agent/spec-lint` through `sh`, a byte copy of the template's shape check
(`upstream-sync.md`). Its seed ends a minimal spec on the phase line `MAINTAIN: …`, and
`test_the_spec_lint_step_reads_the_phase_line` demands the finding text, with a clean twin as the
positive control. Neutralized from a `cp` snapshot of `gate.py`, the step as `true` reddens both
tests and as `sh -c 'exit 1'` reddens the twin. The pre-refresh `.agent/spec.md` drew 6 findings:
the Phase line-1 form, 13 Phase history lines, and the three-line `Tasks` pointer (three non-row
lines plus a last line without the pointer).

## Law locks (`tests/test_law_consistency.py`)

Pairing is checked ORDERED because independent rank/title membership passes a swap of two titles;
that swap and the `at rank N` evasion form are both proven red by mutation, restoring from a `cp`
snapshot rather than `git checkout` (L-022). The `Tasks` port is proven red the same way on a swap,
a dropped row, a renamed or run-on title, a stray rank mark, a ranked or sha-less ticked row, an
unranked, indented or two-space/tab-separated open row, a second, ticked, empty, colon-less or
ranked resume note and a missing header, with a legal `- [x] <sha>` row and a top-level and an
indented resume note as the positive controls. The two notes, planted from a `cp` snapshot of
`.agent/spec.md`: a top-level and an indented `- [ ] ROUTE: …` row and one beside a resume note
pass; a second, ticked, empty, colon-less or ranked ROUTE row stays red, as do a second resume note
and a stray open row.

The third, the queue-link lock, is mutation-proven red by restoring the pointer `upstream-sync.md`
kept at *Maintenance + security pass* after that row closed, from a `cp` snapshot (L-022).

None of the five locks needs a seeded fixture — the tree itself was the firing input, red on two
renamed spine titles, three rank references (one already retargeted onto the wrong unit), one
prose-keyed row whose clause upstream had long dropped, and one pointer at a closed queue row. The
fifth fires on a citation of role tags that Intent no longer carries, restored from a `cp` snapshot:
the lag bullet of `translation-leg.md` as it stood before Intent took the role grammar quotes the
two pre-role tags, and it reddens naming both. Prose must not restate that example either, this file
being scanned too. Mutation-proven red for the fourth: a dead anchor, a generic short anchor, curly
quotes, a changed header cell, a parse that yields no rows, and a row moved below the prose, into a
pipe-less table or into a second table — each escape in both the spaced and the no-space pipe form.
Every mutant is restored from a `cp` snapshot (L-022). **Coverage limit: the global half of the
haystack needs `~/.claude/CLAUDE.md`** — present in every session here, so the check reports absence
as a failure naming the anchor rather than skipping.
