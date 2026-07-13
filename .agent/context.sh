#!/bin/sh
# Context gauge → "N% used/window" (tokens) from the live Claude Code transcript = headroom. Window = 1M.
f=$(ls "$HOME"/.claude/projects/*/"$CLAUDE_CODE_SESSION_ID".jsonl 2>/dev/null)
[ -n "$f" ] || f=$(ls -t "$HOME"/.claude/projects/*/*.jsonl 2>/dev/null | head -1)
u=$(jq -n 'last(inputs|select(.type=="assistant" and .isSidechain!=true and .message.model!="<synthetic>" and (.message.usage|type)=="object")|.message.usage|.input_tokens+.cache_creation_input_tokens+.cache_read_input_tokens)//empty' "$f" 2>/dev/null)
w=1000000
awk -v u="$u" -v w="$w" '
function h(n){ if(n>=1000000){s=sprintf("%.1fM",n/1000000);sub(/\.0M$/,"M",s);return s}
              return sprintf("%dK",int(n/1000+0.5)) }
BEGIN{ if(u==""){ print "? ?/" h(w); exit }
       print int(u*100/w+0.5) "% " h(u) "/" h(w) }'
