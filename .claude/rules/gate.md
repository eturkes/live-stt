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
root-level file, so an over-grown exclusion would still catch it while the real tree went unscanned
⇒ `test_secret_scan_reaches_the_real_tree` pins production, a nested file, a dotdir file and, of
`models/`, the tracked README alone (user ruling 10-16). **The scan reads git's list (tracked + untracked-unignored)
only at a repository's top level** (`git rev-parse --show-prefix` empty) and walks minus
`SCAN_SKIP_DIRS` anywhere else: below the top level git lists nothing ignored, so a seed tree under
`.scratch/` would scan nothing and its seed would go vacuous. Three locks hold that split — a
throwaway repository's list (deleted, ignored and evidence-JSON files out), the seeded key reddening
through git's list beside an ignored copy that does not, and a seed tree under an ignored directory
still reddening — each red under its neutralization (always walk, git below the top level, deleted
or ignored files kept, untracked dropped, evidence JSON kept). **Coverage limit unchanged in kind:
an ignored file holding a key is out of scope by design — a commit cannot carry it.**

The pytest step runs on `PYTEST_WORKERS`=6 pytest-xdist workers (serial ≈39 s → ≈24 s on this host;
per-test outcomes compared as junit maps, identical: 1229 pass + 1 skip, before the GPU golden) — seed trees included, so
every seed fires through the parallel runner. It runs `-rsxX` and `gate.py`'s
`undeclared_demotions` reads the summary. **The step
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

Each step prints its wall-clock and the run its total; the pytest step runs `--durations=5` and
`slowest_tests` prints that block on a green run (`-v` and a failure print the whole output).
`test_a_green_run_prints_each_wall_clock_and_the_slowest_tests` locks all three, proven red by
neutralization from a `cp` snapshot of `gate.py`: the step clock, the slowest block, the total and
the `--durations` flag each reddened it, restored green. **Coverage limit: pytest hides durations
under its 5 ms `--durations-min` floor**, so an all-fast suite prints no block — the seed sleeps
50 ms for that reason.

`spec-lint` runs `.agent/spec-lint` through `sh`, the template's shape check, a copy each refresh
replaces whole (`orientation.md`). Its seed ends a minimal spec on the phase line `MAINTAIN: …`, and
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
and a stray open row. An empty queue pairs with pointer-only `Tasks`; the two queue readers reject
numbered malformed rows rather than using nonemptiness as a parse witness. The committed empty-tree
controls still reject ranked and unranked stray open tasks and a dangling link. Restoring the two
nonempty assertions from a `cp` snapshot reddens both empty-tree readers; restored, both pass.
Title matching squashes whitespace alone: case and punctuation stay significant in both Tasks and
queue links. Committed near-misses change one letter's case or append one period beside an exact
positive twin; restoring case-folding + trailing-period stripping from a `cp` snapshot reddens all
four, restored green.

The third, the queue-link lock, is mutation-proven red by planting a pointer at *Maintenance +
security pass*, a row closed long before, into a rules file from a `cp` snapshot (L-022).

None of the five locks needs a seeded fixture — the tree itself was the firing input, red on two
renamed spine titles, three rank references (one already retargeted onto the wrong unit), one
prose-keyed row whose clause upstream had long dropped, and one pointer at a closed queue row. The
fifth fires on a citation of role tags that Intent no longer carries, restored from a `cp` snapshot:
the lag bullet of `translation-leg.md` as it stood before Intent took the role grammar quotes the
two pre-role tags, and it reddens naming both. Prose must not restate that example either, this file
being scanned too. Citation scope crosses semicolons and ends at `.`, `!` or `?` followed by a
space, outside backticked fragments; punctuation within a fragment stays inside its literal quote.
Committed seeds put an absent fragment after a semicolon and across an internal period, beside live
semicolon + separate-sentence twins. Restoring the old delimiter split from a `cp` snapshot reddens
four controls, restored green. Abbreviations remain lexical sentence boundaries, not a natural-language
parser; this check decides the explicit backtick citation grammar alone.

The fourth holds the ruling tags. Mutation-proven red, each mutant planted from a `cp` snapshot and
restored (L-022): a dead anchor, a short anchor (`review`), curly quotes, an unknown verdict, a
suffixed verdict (`ADAPTED — split`), a missing arrow, a capitalised and a hyphenated opener, a
keyless bold verdict, the opener restated in prose, an anchor only the user-level file once carried,
the pre-move capitalisation of the prototype clause, and a rules dir yielding no tags. The
haystack is the repo's own template copies alone, so no anchor rests on a user-level file that
changes outside the repo.
