#!/bin/sh
# My own Claude Code context gauge. Prints e.g. `15% 30K/200K` (percent first =
# the number I act on; the pair reveals the active window). `? ?/200K` when usage
# isn't readable yet (before the first turn, or right after /compact).
#
# WRAP UP at >=90%: finish the current step at a clean boundary so the user can
#   run /compact (auto-compact is off), or just end if the work is essentially done.
#
# used  = input + cache_read + cache_creation of the latest MAIN-thread assistant
#         turn; subagent (sidechain), synthetic/error, and ai-title turns skipped.
# window= 200K, or 1M when the 1M beta is live. It's always Opus, so the only knob
#         is the launcher-injected CLAUDE_CODE_DISABLE_1M_CONTEXT (set => 200K).
#         CLAUDE_CONTEXT_WINDOW=<n> forces it; a safety net bumps the denominator
#         if usage ever exceeds it, so the percent can't read past 100.
# Read-only. Needs jq.

r="$HOME/.claude/projects"
f=$(ls "$r"/*/"$CLAUDE_CODE_SESSION_ID".jsonl 2>/dev/null)   # this session, ...
[ -n "$f" ] || f=$(ls -t "$r"/*/*.jsonl 2>/dev/null | head -1)   # ... else newest

u=
[ -n "$f" ] && u=$(jq -s '
  [ .[] | select(.type=="assistant" and .isSidechain!=true
                 and .message.model!="<synthetic>"
                 and (.message.usage|type)=="object") ]
  | if length>0 then (.[-1].message.usage
      | .input_tokens + .cache_creation_input_tokens + .cache_read_input_tokens)
    else empty end' "$f" 2>/dev/null)

case $CLAUDE_CODE_DISABLE_1M_CONTEXT in 1|true|yes|on) w=200000 ;; *) w=1000000 ;; esac
case $CLAUDE_CONTEXT_WINDOW in ''|*[!0-9]*) ;; *) w=$CLAUDE_CONTEXT_WINDOW ;; esac

awk -v u="$u" -v w="$w" '
function h(n){ if(n>=1000000){s=sprintf("%.1fM",n/1000000);sub(/\.0M$/,"M",s);return s}
              return sprintf("%dK",int(n/1000+0.5)) }
BEGIN{ if(u==""){ printf "? ?/%s\n",h(w); exit }
       if(u>w) w=(u<=1000000)?1000000:u
       printf "%d%% %s/%s\n", int(u*100/w+0.5), h(u), h(w) }'
