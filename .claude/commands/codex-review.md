Run a non-interactive Codex review of this session and act on its findings.

Prompt: focus below non-empty ⇒ review exactly that; empty ⇒ adversarially review this session's work per AGENTS.md's review criteria.

Deliver the prompt via stdin from a file: Write it to a temp file, then redirect into `codex exec`. (The inline-argument form `codex exec "…"` and the `"$(cat <<'EOF'…)"` form both fail — prompts are backtick-heavy: inline backticks run as command substitution and the nested `cat` is denied as a file-dump, so the argument empties, `codex exec` silently falls back to stdin and, backgrounded or redirected, blocks forever at 0 CPU until killed. Stdin-from-file sidesteps shell quoting and preserves backticks verbatim.) Model = `~/.codex/config.toml` (always your latest); effort forced `xhigh`; `timeout` guards an upstream stall:

```bash
# prompt already written to a temp file via the Write tool, not a heredoc/`cat`:
timeout 1500 codex exec --dangerously-bypass-approvals-and-sandbox \
  -c model_reasoning_effort="xhigh" < /tmp/codex-review-prompt.txt 2>/dev/null
```

Codex also writes the full session to a rollout JSONL under `~/.codex/sessions/<date>/`; if stdout is lost, recover the review from the newest rollout's last assistant message.

Relay the findings, say which you accept or reject and why, and fix the accepted ones before closing.

Review focus (may be empty): $ARGUMENTS
