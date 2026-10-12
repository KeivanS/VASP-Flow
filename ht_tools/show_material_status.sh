#!/bin/bash
# Key INCAR tags, end of OSZICAR and (relax) forces/pressure of each step of a material.
#   bash show_material_status.sh mp-20352                 # every run under /scratch/$USER/VASP-Flow
#   bash show_material_status.sh mp-20352 lobster bands   # only these steps
#   bash show_material_status.sh /path/to/materials/mp-20352
id=${1:?usage: show_material_status.sh ID|MATERIAL_DIR [relax scf bands dos lobster]}; shift
steps=${*:-relax scf bands dos lobster}
if [ -d "$id" ]; then dirs=$id; else dirs=$(ls -d /scratch/$USER/VASP-Flow/*/materials/"$id" $HOME/VASP-Flow/*/materials/"$id" 2>/dev/null); fi
[ -n "$dirs" ] || { echo "no folder for $id"; exit 1; }
for m in $dirs; do
    [ -f "$m/../../MOVED_TO_SCRATCH.txt" ] && continue
    echo "######## $m"
    for s in $steps; do
        case $s in relax) sd=01_relax ;; scf) sd=02_scf ;; bands) sd=03_bands ;; dos) sd=04_dos ;; lobster) sd=08_lobster ;; *) sd=$s ;; esac
        [ -f "$m/$sd/INCAR" ] || continue
        st="not finished"; grep -q "General timing and accounting" "$m/$sd/OUTCAR" 2>/dev/null && st=finished
        [ -f "$m/$sd/OUTCAR" ] || st="no OUTCAR"
        echo "== $sd ($st)"
        awk '/^ *(ALGO|KPAR|NCORE|NBANDS|NELM|NSW|EDIFFG?|ISYM|ICHARG|ISTART|LDAU) *=/ {sub(/!.*/, ""); printf "   %s\n", $0}' "$m/$sd/INCAR"
        tail -3 "$m/$sd/OSZICAR" 2>/dev/null | sed 's/^/   /'
        if [ $sd = 01_relax ] && [ -f "$m/$sd/OUTCAR" ]; then
            grep -q "reached required accuracy" "$m/$sd/OUTCAR" && echo "   relax converged" || echo "   relax NOT converged"
            awk '/FORCES: max atom/ {f=$0} /external pressure/ {p=$0} END {print "  " f; print "  " p}' "$m/$sd/OUTCAR"
        fi
    done
done
