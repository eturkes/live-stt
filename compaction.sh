#!/bin/sh
# Claude Code context gauge. CLAUDE_CODE_SESSION_ID set => manual (read transcript); unset => statusline (stdin JSON).
if [ -n "$CLAUDE_CODE_SESSION_ID" ]; then
  f=$(ls "$HOME"/.claude/projects/*/"$CLAUDE_CODE_SESSION_ID".jsonl 2>/dev/null)
  [ -n "$f" ] || f=$(ls -t "$HOME"/.claude/projects/*/*.jsonl 2>/dev/null | head -1)
  u=$(jq -s 'map(select(.type=="assistant" and .isSidechain!=true and .message.model!="<synthetic>" and (.message.usage|type)=="object"))
    | if length>0 then (.[-1].message.usage|.input_tokens+.cache_creation_input_tokens+.cache_read_input_tokens) else empty end' "$f" 2>/dev/null)
  case $CLAUDE_CODE_DISABLE_1M_CONTEXT in 1|true|yes|on) w=200000 ;; *) w=1000000 ;; esac
  c=0
else
  j=$(cat)
  u=$(printf '%s' "$j" | jq -r '.context_window.total_input_tokens // empty' 2>/dev/null)
  w=$(printf '%s' "$j" | jq -r '.context_window.context_window_size // empty' 2>/dev/null)
  c=1
fi
[ "$w" -gt 0 ] 2>/dev/null || w=200000
awk -v u="$u" -v w="$w" -v c="$c" '
function h(n){ if(n>=1000000){s=sprintf("%.1fM",n/1000000);sub(/\.0M$/,"M",s);return s}
              return sprintf("%dK",int(n/1000+0.5)) }
BEGIN{ if(u==""){ print "? ?/" h(w); exit }
       p=int(u*100/w+0.5); s=p "% " h(u) "/" h(w)
       if(c&&p>=80) s="\033[31m" s "\033[0m"; else if(c&&p>=60) s="\033[33m" s "\033[0m"
       print s }'
