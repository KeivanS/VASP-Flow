#!/bin/bash
# Comment out (skip) fix scripts in DIAG_DIR/submit_all_fixes.sh whose name contains a pattern.
#   bash skip_fixes.sh diagnosis_round2 HT_1-500__mp-6430 HT_1-500__mp-7889
#   bash skip_fixes.sh diagnosis_round2 --undo HT_1-500__mp-6430      # put it back
d=${1:?usage: skip_fixes.sh DIAG_DIR [--undo] PATTERN...}; shift
f=$d/submit_all_fixes.sh; [ -f "$f" ] || { echo "no $f"; exit 1; }
undo=; [ "${1:-}" = --undo ] && { undo=1; shift; }
for p in "$@"; do
    q=$(printf '%s' "$p" | sed 's/[.[\*^$/|]/\\&/g')
    if [ -n "$undo" ]; then e="/fix\/[^ ]*$q[^ ]*\.sh/s/^# *run_fix/run_fix/"
    else e="/fix\/[^ ]*$q[^ ]*\.sh/s/^run_fix/# run_fix/"; fi
    sed -E "$e" "$f" > "$f.tmp" && cat "$f.tmp" > "$f" && rm -f "$f.tmp"
done
awk '/^ *#? *run_fix +\// { n = split($NF, a, "/"); sub(/\.sh$/, "", a[n]); print (/^ *#/ ? "  skipped  " : "  RUN      ") a[n] }' "$f"
