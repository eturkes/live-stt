Run a non-interactive Codex review of this session and hold yourself accountable
to it; Codex runs with full autonomy, the latest model, and the highest reasoning
effort.

Build the prompt: focus below non-empty ⇒ review exactly that; empty ⇒
adversarially review this session's work — correctness and logic, soundness of
the core claims, and any gap between what the code and outputs actually guarantee
and what the session claims. Weigh honesty and overreach above style.

Deliver via stdin from a file: Write the prompt to a temp file, then redirect it
into `codex exec`. The inline-argument form (`codex exec "...prompt..."`) and the
`"$(cat <<'EOF'...)"` form both fail — prompts are full of backticks, which
mangle into an empty argument (inline backticks run as command substitution; the
nested `cat` is denied as a file-dump), whereupon `codex exec` silently falls
back to stdin and, backgrounded or redirected, blocks forever at 0 CPU until
killed. Stdin-from-file sidesteps shell quoting and preserves backticks verbatim.

Model comes from `~/.codex/config.toml` (always your latest); effort is forced to
`xhigh`; `timeout` guards an upstream stall:

```bash
# prompt already written to a temp file via the Write tool, not a heredoc/`cat`:
timeout 1500 codex exec --dangerously-bypass-approvals-and-sandbox \
  -c model_reasoning_effort="xhigh" < /tmp/codex-review-prompt.txt 2>/dev/null
```

Codex also writes the full session to a rollout JSONL under
`~/.codex/sessions/<date>/`; if stdout is lost, recover the review from the last
assistant message of the newest rollout.

Relay the findings, say which you accept or reject and why, and fix the accepted
ones before closing.

Review focus (may be empty): $ARGUMENTS
