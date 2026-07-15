---
name: session-prompt
description: >-
  Continue live-stt from a fresh Codex session using its roadmap workflow. Invoke explicitly as
  $session-prompt, optionally followed by a task override.
---

1. Resolve paths relative to this `SKILL.md`, then read `../../../.codex/prompts/session.md`
   completely.
2. Treat text accompanying `$session-prompt` as `TASK`; an invocation containing only the skill
   name has an empty `TASK`.
3. Follow that prompt as the session workflow.
4. Keep this skill and `.codex/prompts/session.md` synchronized whenever either side changes.
