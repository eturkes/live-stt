# Session prompt

Execute the supplied task exactly with full project context. State arrives attached: @.agent/roadmap.md @.agent/memory.md; an unexpanded path → `/session-roadmap`'s `ls .agent/` check. Roadmap MODE dispatch remains `/session-roadmap`-owned; this session's scope = supplied task.

- Run the task MAIN-direct; teammates fan out on it. Machinery = session-roadmap execution map + roles, briefs, roster, worktree isolation, hygiene, verification, Close order + commit convention.
- Keep roadmap, memory + related files consistent with task changes. Route adjacent improvements to `.agent/polish.md`, with an acceptance check + `pri` written at deferral.
- A requirement-changing design reaches the user before any scope-source edit; the supplied task is what authorizes edits inside its own scope.
- Context policy (project `CLAUDE.md`): user-requested tasks run across coherent checkpoints as the harness manages the window; a user-stated bound overrides.
- Close: changed state ⇒ one scoped commit per cohesive piece; read-only result ⇒ read-only close.

Task: $ARGUMENTS
