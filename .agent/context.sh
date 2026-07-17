#!/bin/sh
# Context gauge → "N% used/window" (tokens) from the live Claude Code transcript = headroom. Window = 272K; auto-compaction triggers at 90% (~245K).
# Sums the last assistant turn's real API tokens (input+cache_creation+cache_read+output) = that request's
# occupancy floor for the NEXT turn — the dominant, authoritative headroom signal. It far exceeds the visible
# conversation: sys-prompt/tools/CLAUDE.md + injected reminders + prior-turn redacted extended-thinking ride in
# the cached input, none shown in the .jsonl. A high reading is REAL occupancy, not inflated accounting.
transcript_root="$HOME/.claude/projects"
f=$(find "$transcript_root" -mindepth 2 -maxdepth 2 -type f -name "$CLAUDE_CODE_SESSION_ID.jsonl" -print -quit 2>/dev/null)
# fallback (no session id): newest transcript in THIS project's dir only, never another project's
project_transcripts="$transcript_root/$(pwd -P | tr '/.' '-')"
[ -n "$f" ] || f=$(find "$project_transcripts" -maxdepth 1 -type f -name '*.jsonl' -printf '%T@ %p\n' 2>/dev/null | sort -nr | cut -d ' ' -f 2- | head -1)
u=$(jq -n 'last(inputs|select(.type=="assistant" and .isSidechain!=true and .message.model!="<synthetic>" and (.message.usage|type)=="object")|.message.usage|.input_tokens+.cache_creation_input_tokens+.cache_read_input_tokens+.output_tokens)//empty' "$f" 2>/dev/null)
w=272000
awk -v u="$u" -v w="$w" '
function h(n){ if(n>=1000000){s=sprintf("%.1fM",n/1000000);sub(/\.0M$/,"M",s);return s}
              return sprintf("%dK",int(n/1000+0.5)) }
BEGIN{ if(u==""){ print "? ?/" h(w); exit }
       print int(u*100/w+0.5) "% " h(u) "/" h(w) }'
