#!/bin/sh
# Context gauge: prints "N% used/window" (tokens) from the current Claude Code session transcript, to judge headroom. Window 1M, or 200K when CLAUDE_CODE_DISABLE_1M_CONTEXT is set.
f=$(ls "$HOME"/.claude/projects/*/"$CLAUDE_CODE_SESSION_ID".jsonl 2>/dev/null)
[ -n "$f" ] || f=$(ls -t "$HOME"/.claude/projects/*/*.jsonl 2>/dev/null | head -1)
u=$(jq -s 'map(select(.type=="assistant" and .isSidechain!=true and .message.model!="<synthetic>" and (.message.usage|type)=="object"))
  | if length>0 then (.[-1].message.usage|.input_tokens+.cache_creation_input_tokens+.cache_read_input_tokens) else empty end' "$f" 2>/dev/null)
case $CLAUDE_CODE_DISABLE_1M_CONTEXT in 1|true|yes|on) w=200000 ;; *) w=1000000 ;; esac
awk -v u="$u" -v w="$w" '
function h(n){ if(n>=1000000){s=sprintf("%.1fM",n/1000000);sub(/\.0M$/,"M",s);return s}
              return sprintf("%dK",int(n/1000+0.5)) }
BEGIN{ if(u==""){ print "? ?/" h(w); exit }
       print int(u*100/w+0.5) "% " h(u) "/" h(w) }'
