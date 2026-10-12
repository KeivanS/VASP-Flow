#!/bin/bash
# Last lines of the newest job log (.out and .err) of one step of a material.
#   bash show_job_log.sh mp-20352 lobster          # newest log, 15 lines each
#   bash show_job_log.sh mp-20352 lobster 40 21281654
id=${1:?usage: show_job_log.sh ID|MATERIAL_DIR STEP [LINES] [JOBID]}; step=${2:?step: relax scf bands lobster ...}
n=${3:-15}; jid=${4:-}
if [ -d "$id" ]; then dirs=$id; else dirs=$(ls -d /scratch/$USER/VASP-Flow/*/materials/"$id" 2>/dev/null); fi
[ -n "$dirs" ] || { echo "no folder for $id"; exit 1; }
for m in $dirs; do
    b=$(basename "$m")
    out=$(ls -t "$m"/*"${b}_${step}-${jid}"*.out 2>/dev/null | head -1)
    [ -n "$out" ] || { echo "== $m: no ${b}_${step} log"; continue; }
    for f in "$out" "${out%.out}.err"; do
        [ -s "$f" ] && { echo "== $f"; tail -n "$n" "$f"; }
    done
done
