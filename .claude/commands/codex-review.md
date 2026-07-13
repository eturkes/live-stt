Run a non-interactive Codex review of this session and act on its findings.

Prompt: focus below non-empty ⇒ review exactly that; empty ⇒ adversarially review this session's work per AGENTS.md's review criteria. Either way, AGENTS.md sets the review bar → add only what it omits, positively framed: output = the review only → the caller applies accepted fixes (verify freely via reads + non-destructive checks — tests, build, `git diff`); each finding → severity, `file:line`, flaw, impact, concrete fix.

**Parallel-safe: one run = one tag.** ≥1 `/codex-review` may be live across unrelated projects → give each run its own path — a fixed one like `/tmp/codex-review-prompt.txt` gets clobbered by parallel runs → you track the wrong codex thread. `RUN` = unique path under session scratchpad (`codex-review-<cwd-basename>-<short random>`); this run's prompt = `$RUN.prompt`, review = `$RUN.review`.

Deliver the prompt via stdin from a file: Write it to `$RUN.prompt`, then redirect into `codex exec`. (Prompts are backtick-heavy → the inline-argument form `codex exec "…"` runs backticks as command substitution, and an argument that ends up empty makes `codex exec` silently fall back to stdin — backgrounded or redirected, it then blocks forever at 0 CPU until killed. A `"$(cat <<'EOF'…)"` argument passes the perm layer yet embeds the whole prompt in the command text → transcript bloat + terminator-collision risk. Stdin-from-file sidesteps shell quoting and preserves backticks verbatim.) `-o` → final review to `$RUN.review`. Let `codex exec` run to completion — review length scales with scope, a `timeout` guard's kill wastes the whole run, and a second clock invites wait-vs-guard misreckoning (bit once); detect stalls from the probe below:

```bash
RUN="<session-scratchpad>/codex-review-myproj-7f3a"   # unique/run → parallel-safe
# $RUN.prompt already written via the Write tool (shell heredocs risk backtick/terminator mangling):
codex exec --dangerously-bypass-approvals-and-sandbox \
  -o "$RUN.review" < "$RUN.prompt" 2>/dev/null
```

Launch with `run_in_background: true`; wait via `TaskOutput(task_id, block=true, timeout=600000)` loops — timeout is MS, cap 600000 (a seconds-misread makes every block look like an instant timeout while codex runs on). Reviews commonly take 10–40+ min; keep looping while progress shows. Progress probe (suspected stall): this cwd's rollout JSONL (located per the fallback below) mtime-advances while codex thinks — flat mtime across two full loops + an idle `codex exec` process = real stall → `TaskStop`, relaunch once.

Review = `$RUN.review` (also echoed to stdout). Both lost → fall back to rollout JSONL under `~/.codex/sessions/<Y>/<M>/<D>/` (date-nested → recurse the whole tree), but disambiguate by cwd over mtime: rollout's 1st line = `session_meta` w/ `payload.cwd` + `payload.session_id` (id also in filename) → newest rollout where `payload.cwd` == `$PWD` → read its last assistant message.

Relay the findings, say which you accept or reject and why, and fix the accepted ones before closing.

Review focus (may be empty): $ARGUMENTS
