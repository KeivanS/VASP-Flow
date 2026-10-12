#!/bin/bash
# SLURM history (sacct) and current queue entries of jobs whose name contains a pattern.
#   bash show_job_history.sh mp-20352                    # last 3 days
#   bash show_job_history.sh mp-20352 mp-6430 -S 2026-10-01
since=$(date -d '3 days ago' +%F 2>/dev/null); pats=()
while [ $# -gt 0 ]; do case $1 in -S) since=$2; shift 2 ;; *) pats+=("$1"); shift ;; esac; done
[ ${#pats[@]} -gt 0 ] || { echo "usage: show_job_history.sh PATTERN... [-S YYYY-MM-DD]"; exit 1; }
re=$(IFS='|'; echo "${pats[*]}")
echo "== sacct since $since"
sacct -u "$USER" -S "$since" -X -o JobID,JobName%28,State,Elapsed,Timelimit,ExitCode,End | awk -v re="$re" 'NR <= 2 || $2 ~ re'
echo "== in the queue now"
squeue -h -u "$USER" -o '%i %j %T %M %r' | awk -v re="$re" '$2 ~ re' | grep . || echo "   (none)"
