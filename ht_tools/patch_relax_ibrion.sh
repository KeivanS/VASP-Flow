#!/bin/bash
# Switch the relaxations of an existing highthroughput run from IBRION = 2 (conjugate
# gradient) to IBRION = 1 (quasi-Newton) wherever the relax has not converged yet.
#
# Why: with ISIF = 3, IBRION = 2 does not stop once the cell is converged; its line
# search (ZBRENT) then jumps back to the input cell and aborts with "ZBRENT: fatal error
# in bracketing" (21 of 500 materials in HT_1-500).  Tests on mp-8 (Re, no free
# coordinates) and mp-404 (MnTe, Pnma, 16 free coordinates) converged cleanly with
# IBRION = 1.
#
# For every material whose 01_relax has not printed "reached required accuracy":
#   * 01_relax/INCAR: IBRION -> 1  (VASP reads the INCAR when a job starts, so this
#     also applies to relax jobs that are already queued)
#   * after a ZBRENT abort the CONTCAR is the reverted input cell: 01_relax/POSCAR is
#     kept (the input cell), the aborted OUTCAR/OSZICAR/CONTCAR are moved to
#     01_relax/zbrent_<date>/ so the rerun starts clean
# Materials with a RUNNING relax are skipped.  Idempotent; dry run by default.
#
#   cd ~/VASP-Flow/HT_1-500 && bash ../ht_tools/patch_relax_ibrion.sh [--apply]
set -u
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"
cd "$HERE/materials" || { echo "no materials/ under $HERE"; exit 1; }
STAMP=$(date +%Y%m%d_%H%M)
RUNNING=$(squeue -h -u "${USER:-$(whoami)}" -t R -o '%j' 2>/dev/null)

n_set=0; n_zb=0; n_run=0; n_ok=0; zb_list=""
for d in mp-*/; do
    m=${d%/}
    r="$m/01_relax"
    [ -f "$r/INCAR" ] || continue
    if grep -q "reached required accuracy" "$r/OUTCAR" 2>/dev/null; then n_ok=$((n_ok + 1)); continue; fi
    if printf '%s\n' "$RUNNING" | grep -qx "${m}_relax"; then n_run=$((n_run + 1)); continue; fi
    if grep -qE '^[[:space:]]*IBRION[[:space:]]*=[[:space:]]*2' "$r/INCAR"; then
        n_set=$((n_set + 1))
        [ $APPLY = 1 ] && sed -i -E 's/^([[:space:]]*IBRION[[:space:]]*=[[:space:]]*)2([^0-9].*)?$/\11   ! quasi-Newton (IBRION=2 hit ZBRENT failures)/' "$r/INCAR"
    fi
    if [ -f "$r/OUTCAR" ] && grep -qs "ZBRENT: fatal" "$m"/*_relax-*.out; then
        n_zb=$((n_zb + 1)); zb_list="$zb_list $m"
        if [ $APPLY = 1 ]; then
            mkdir -p "$r/zbrent_$STAMP"
            for f in OUTCAR OSZICAR CONTCAR XDATCAR .vf_resume .vf_restarts; do
                [ -e "$r/$f" ] && mv "$r/$f" "$r/zbrent_$STAMP/"
            done
        fi
    fi
done
echo "relax already converged: $n_ok   running now (skipped): $n_run"
echo "01_relax/INCAR IBRION 2 -> 1: $n_set materials"
echo "ZBRENT-aborted relaxes reset to the input cell: $n_zb${zb_list:+  ($zb_list )}"
if [ $APPLY = 1 ]; then echo "Done.  Restore the full material_list.txt if you shortened it, then ./submit_all.sh"
else echo "Dry run - nothing changed. Re-run with --apply."; fi
