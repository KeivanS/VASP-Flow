#!/bin/bash
# Per step of a material: key INCAR tags; every ionic step of OSZICAR (F = free energy, E0 = energy
# at sigma -> 0) with the last 5 electronic steps (DAV/RMM/CG) before it; relax forces/pressure;
# and, when VASP did not finish, the last 10 lines of OUTCAR and when it was last written.
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
        [ -s "$m/$sd/OSZICAR" ] && awk '
            /^ *(DAV|RMM|CG|SDA):/ { b[++n % 5] = $0; ne++; next }
            / F= / { for (i = n - 4; i <= n; i++) if (i > 0) print "   " b[i % 5]
                     print "   " $0 "   (" ne " electronic steps)"; n = 0; ne = 0; ni++ }
            END { if (n) { for (i = n - 4; i <= n; i++) if (i > 0) print "   " b[i % 5]
                           print "   ... ionic step " ni + 1 " still running (" ne " electronic steps so far)" } }' "$m/$sd/OSZICAR"
        [ -s "$m/$sd/OSZICAR" ] || echo "   OSZICAR empty or missing: no electronic step done"
        if [ "$st" = "not finished" ]; then
            echo "   -- OUTCAR last written $(date -r "$m/$sd/OUTCAR" '+%F %T'); its last 10 lines:"
            tail -10 "$m/$sd/OUTCAR" | sed 's/^/   | /'
        fi
        if [ $sd = 01_relax ] && [ -f "$m/$sd/OUTCAR" ]; then
            grep -q "reached required accuracy" "$m/$sd/OUTCAR" && echo "   relax converged" || echo "   relax NOT converged"
            awk '/FORCES: max atom/ {f=$0} /external pressure/ {p=$0} END {print "  " f; print "  " p}' "$m/$sd/OUTCAR"
        fi
    done
done
