#!/bin/bash
# List the materials vasp_diagnose.py found broken, and what each fix script will do.
#   bash show_problems.sh diag_round3          # broken ones (spilling_high counted, not listed)
#   bash show_problems.sh diag_round3 --all    # also list the spilling_high ones
d=${1:?usage: show_problems.sh DIAG_DIR [--all]}; all=${2:-}
t=$d/diagnosis.tsv; [ -f "$t" ] || { echo "no $t"; exit 1; }
awk -F'\t' -v all="$all" 'NR > 1 && $5 != "ok" && ($5 != "spilling_high" || all != "") { printf "%-32s %-55s %s\n", $1, $4, $5 }
    $5 == "spilling_high" { s++ } END { if (s && all == "") print "(" s " spilling_high not listed: --all)" }' "$t"
f=$d/submit_all_fixes.sh; [ -f "$f" ] || exit 0
echo; echo "== will run (run_fix) / skipped (# run_fix):"
awk '/^ *#? *run_fix +\// { n = split($NF, a, "/"); sub(/\.sh$/, "", a[n]); print (/^ *#/ ? "  skipped  " : "  RUN      ") a[n] }' "$f"
echo; echo "== changes per fix script:"
for s in "$d"/fix/*.sh; do
    [ -f "$s" ] || continue
    echo "-- $(basename "$s" .sh)"
    sed -n -E 's/^echo "  (.*)"$/   \1/p' "$s" | grep -v '^   submitted'
done
