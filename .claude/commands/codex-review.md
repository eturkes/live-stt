Run a non-interactive Codex review of this session and act on its findings.

Prompt: focus below non-empty ⇒ review exactly that; empty ⇒ adversarially review this session's work per AGENTS.md's review criteria.

**Parallel-safe: one run = one tag.** ≥1 `/codex-review` may be live across unrelated projects → never share a fixed path like `/tmp/codex-review-prompt.txt`: parallel runs clobber it → you track the wrong codex thread. `RUN` = unique path under session scratchpad (`codex-review-<cwd-basename>-<short random>`); this run's prompt = `$RUN.prompt`, review = `$RUN.review`.

Deliver the prompt via stdin from a file: Write it to `$RUN.prompt`, then redirect into `codex exec`. (The inline-argument form `codex exec "…"` and the `"$(cat <<'EOF'…)"` form both fail — prompts are backtick-heavy: inline backticks run as command substitution and the nested `cat` is denied as a file-dump, so the argument empties, `codex exec` silently falls back to stdin and, backgrounded or redirected, blocks forever at 0 CPU until killed. Stdin-from-file sidesteps shell quoting and preserves backticks verbatim.) Model = `~/.codex/config.toml` (always your latest); effort forced `xhigh`; `-o` → final review to `$RUN.review`; `timeout` guards an upstream stall:

```bash
RUN="<session-scratchpad>/codex-review-myproj-7f3a"   # unique/run → no parallel-run collision
# $RUN.prompt already written via the Write tool, not a heredoc/`cat`:
timeout 1500 codex exec --dangerously-bypass-approvals-and-sandbox \
  -c model_reasoning_effort="xhigh" \
  -o "$RUN.review" < "$RUN.prompt" 2>/dev/null
```

Review = `$RUN.review` (also echoed to stdout). Both lost → fall back to rollout JSONL `~/.codex/sessions/<date>/`, but disambiguate by cwd NOT mtime: rollout's 1st line = `session_meta` w/ `payload.cwd` + `payload.session_id` (id also in filename) → newest rollout where `payload.cwd` == `$PWD` → read its last assistant message.

Relay the findings, say which you accept or reject and why, and fix the accepted ones before closing.

Review focus (may be empty): $ARGUMENTS
