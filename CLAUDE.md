@AGENTS.md

# Claude Code
- Adapt `/session-prompt` + `/codex-review` to the project as it evolves; keep them token-efficient + agent-only (skip human UX / usage hints). I trigger them.
- I usually cap you at 200K context — it sharpens you but demands token discipline: prune redundant/obsolete info + structures often. `Read`/`Bash` bypass `.gitignore` → enforce the do-not-read set (`AGENTS.md`) via `permissions.deny` `Read()` rules in `.claude/settings.json`, covering gitignored + non-gitignored paths alike, kept synced with the set.
- Past 80% context (watch via `.agent/context.sh`): drive ongoing work to a clean state, hold new tasks, and close out cleanly when you confidently can. Otherwise I compact manually, then direct you to continue from where you halted or to revert and retry at smaller scope.
