#!/bin/sh
# Context gauge → "N% used/240K" from an assistant turn's usage sum (input+cache_creation+cache_read
# +output) = the CLI's own compaction input. High reads = normal: sys/tools/CLAUDE.md + redacted thinking
# bill from cached input the .jsonl omits; server-tool turns (ToolSearch) bill per internal iteration.
# 240K = auto-compaction point (ACW 273K − 33K; raw 1M = informational); warn 220K. Teammates share it.
# POSIX sh + utility options throughout (jq aside): globs pin transcript depth, `ls -t` ranks by mtime.
# Usage: context-gauge.sh [-p] [<teammate>]
#   (no arg)   → MAIN, last turn = live occupancy: this session id, else this project's newest transcript
#   <teammate> → spawned role name (`name` in agent-*.meta.json) or raw agent id, under this project's
#                sessions (plain + workflow nestings) — newest match wins, so a live teammate always
#                resolves. Reports the HIGH-WATER turn, the number unit sizing needs: compaction resets
#                occupancy, and a stopped/dead teammate trails stripped
#                `{input_tokens:0,output_tokens:0}` turns that read as 0%.
#   -p         → print the resolved transcript path instead of the gauge; marker polling reads that
#                transcript's LAST assistant text (a raw grep also hits the spawn prompt + every
#                `SendMessage` body carrying the marker)
[ "$1" = "-p" ] && { path_only=1; shift; }
transcript_root="$HOME/.claude/projects"
project_transcripts="$transcript_root/$(pwd -P | tr '/.' '--')"
f=""
if [ -n "$1" ]; then
  agent=true # every subagent turn carries isSidechain=true
  # newest-first scan, first name match wins: filename carries a raw agent id, meta.json carries the role
  # shellcheck disable=SC2012
  f=$(ls -td "$project_transcripts"/*/subagents/*.jsonl \
              "$project_transcripts"/*/subagents/workflows/*/*.jsonl 2>/dev/null |
    while IFS= read -r p; do
      [ -f "$p" ] || continue
      case ${p##*/} in *"$1"*) printf '%s\n' "$p"; break;; esac
      m=${p%.jsonl}.meta.json
      [ -f "$m" ] && [ "$(jq -r '.name//""' "$m" 2>/dev/null)" = "$1" ] &&
        { printf '%s\n' "$p"; break; }
    done)
else
  agent=false
  for p in "$transcript_root"/*/"$CLAUDE_CODE_SESSION_ID.jsonl"; do
    [ -f "$p" ] && { f=$p; break; }
  done
  # fallback (no session id): newest regular transcript in THIS project's dir alone; UUID names → ls output parses cleanly
  # shellcheck disable=SC2012
  [ -n "$f" ] || f=$(ls -td "$project_transcripts"/*.jsonl 2>/dev/null |
    while IFS= read -r p; do [ -f "$p" ] && { printf '%s\n' "$p"; break; }; done)
fi
[ -n "$path_only" ] && { [ -n "$f" ] && printf '%s\n' "$f"; exit; }
u=$(jq -n --argjson a "$agent" '[inputs|select(.type=="assistant" and ($a or .isSidechain!=true) and .message.model!="<synthetic>" and (.message.usage|type)=="object" and (.message.usage.cache_read_input_tokens|type)=="number")|.message.usage|.input_tokens+.cache_creation_input_tokens+.cache_read_input_tokens+.output_tokens]|select(length>0)|if $a then max else .[-1] end' "$f" 2>/dev/null)
w=240000
awk -v u="$u" -v w="$w" '
function h(n){ if(n>=1000000){s=sprintf("%.1fM",n/1000000);sub(/\.0M$/,"M",s);return s}
              return sprintf("%dK",int(n/1000+0.5)) }
BEGIN{ if(u==""){ print "? ?/" h(w); exit }
       print int(u*100/w+0.5) "% " h(u) "/" h(w) }'
