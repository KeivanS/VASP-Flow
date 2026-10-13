#!/bin/bash
# One repair round of a highthroughput run: diagnose every material, show what failed and
# why, and (after you answer y) submit the repairs.  Run it from the HT folder (the one
# with materials/ and material_list.txt), after the jobs of the previous round finished:
#
#   cd /scratch/$USER/VASP-Flow/HT_ins_1
#   bash ~/VASP-Flow/ht_tools/next_round.sh            # asks before submitting
#   bash ~/VASP-Flow/ht_tools/next_round.sh -y         # submits without asking
#   bash ~/VASP-Flow/ht_tools/next_round.sh -n         # diagnosis only
#
# -> diagnosis_round<N>/  report.txt (details per material), diagnosis.tsv (table),
#    failed_list.txt, review_list.txt, fix/<material>.sh, submit_all_fixes.sh(.log)
# Extra arguments go to vasp_diagnose.py (e.g. --max-rounds 4, --since 2026-10-01).
set -u
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIAG="$TOOLS/../vasp_diagnose.py"
ask=1
case "${1:-}" in -y) ask=0; shift ;; -n) ask=2; shift ;; esac
[ -d "$HERE/materials" ] && [ -f "$HERE/material_list.txt" ] \
    || { echo "run from the HT folder (no materials/ or material_list.txt in $HERE)"; exit 1; }
cd "$HERE"

# the previous round may still be submitting (throttled by MAX_QUEUED)
if pgrep -u "${USER:-$(whoami)}" -f "$HERE/diagnosis_round[0-9]*/submit_all_fixes.sh" >/dev/null 2>&1; then
    echo "The previous round is still submitting its repairs (see diagnosis_round*/submit_all_fixes.log)."
    echo "Run next_round.sh again when it has finished."; exit 1
fi
# jobs of this run still in the queue: the diagnosis skips those materials
ids=" $(awk '{print $1}' material_list.txt | grep -v '^#' | tr '\n' ' ') "
nq=$(squeue -h -u "${USER:-$(whoami)}" -o %j 2>/dev/null | awk -v ids="$ids" '
    { id = $1; sub(/_[a-z]+$/, "", id) } index(ids, " " id " ") { n++ } END { print n + 0 }')
if [ "$nq" -gt 0 ]; then
    echo "$nq job(s) of this run are still queued or running; their materials are skipped this round."
    if [ $ask = 1 ]; then read -r -p "Diagnose the others now? [y/N] " a; case $a in y|Y) ;; *) exit 0 ;; esac; fi
fi

n=$(command ls -d diagnosis_round[0-9]* 2>/dev/null | sed 's/diagnosis_round//' | sort -n | tail -1)
out=diagnosis_round$(( ${n:-0} + 1 ))
echo "== $out"
python3 "$DIAG" material_list.txt --out "$out" "$@" || exit 1
echo
bash "$TOOLS/show_problems.sh" "$out"
echo
echo "Details (cause, evidence, certainty, changes):  less $HERE/$out/report.txt"
nfix=$(grep -c '^run_fix ' "$out/submit_all_fixes.sh")
nheld=$(grep -c '^# run_fix ' "$out/submit_all_fixes.sh")
[ "$nheld" -gt 0 ] && echo "$nheld material(s) held for review - see report.txt; set_params.sh changes their settings."
if [ "$nfix" -eq 0 ]; then echo "Nothing to submit."; exit 0; fi
[ $ask = 2 ] && { echo "Submit later:  nohup bash $out/submit_all_fixes.sh > $out/submit_all_fixes.log 2>&1 &"; exit 0; }
if [ $ask = 1 ]; then
    read -r -p "Submit the $nfix repair(s) now? [y/N] " a
    case $a in y|Y) ;; *) echo "Submit later:  nohup bash $out/submit_all_fixes.sh > $out/submit_all_fixes.log 2>&1 &"; exit 0 ;; esac
fi
nohup bash "$HERE/$out/submit_all_fixes.sh" > "$HERE/$out/submit_all_fixes.log" 2>&1 &
echo "Submitting in the background (log: $out/submit_all_fixes.log).  When the jobs are done, run next_round.sh again."
