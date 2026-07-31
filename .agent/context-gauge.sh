#!/bin/sh
# Context gauge → "N% used/240K": last assistant turn's usage sum (input+cache_creation+cache_read+output)
# = the CLI's own compaction input. High reads = normal: sys/tools/CLAUDE.md + redacted thinking bill from
# cached input the .jsonl omits; server-tool turns (ToolSearch) bill per internal iteration.
# 240K = auto-compaction point (ACW 273K − 33K; raw 1M = informational); warn 220K. Subagents same.
# POSIX sh + utility options throughout (jq aside): globs pin transcript depth, `ls -t` ranks by mtime.
transcript_root="$HOME/.claude/projects"
f=""
for p in "$transcript_root"/*/"$CLAUDE_CODE_SESSION_ID.jsonl"; do
  [ -f "$p" ] && { f=$p; break; }
done
# fallback (no session id): newest regular transcript in THIS project's dir alone; UUID names → ls output parses cleanly
project_transcripts="$transcript_root/$(pwd -P | tr '/.' '--')"
# shellcheck disable=SC2012
[ -n "$f" ] || f=$(ls -td "$project_transcripts"/*.jsonl 2>/dev/null |
  while IFS= read -r p; do [ -f "$p" ] && { printf '%s\n' "$p"; break; }; done)
u=$(jq -n 'last(inputs|select(.type=="assistant" and .isSidechain!=true and .message.model!="<synthetic>" and (.message.usage|type)=="object")|.message.usage|.input_tokens+.cache_creation_input_tokens+.cache_read_input_tokens+.output_tokens)//empty' "$f" 2>/dev/null)
w=240000
awk -v u="$u" -v w="$w" '
function h(n){ if(n>=1000000){s=sprintf("%.1fM",n/1000000);sub(/\.0M$/,"M",s);return s}
              return sprintf("%dK",int(n/1000+0.5)) }
BEGIN{ if(u==""){ print "? ?/" h(w); exit }
       print int(u*100/w+0.5) "% " h(u) "/" h(w) }'
