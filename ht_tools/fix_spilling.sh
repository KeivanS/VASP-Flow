#!/bin/bash
# Redo the LOBSTER step of materials whose abs. charge spilling is too high, with more
# bands and a basis scan.  Run from the HT folder (materials/, env.sh):
#
#   bash ~/VASP-Flow/ht_tools/fix_spilling.sh                    # list materials above 5 %, what would change
#   bash ~/VASP-Flow/ht_tools/fix_spilling.sh --submit           # do it
#   bash ~/VASP-Flow/ht_tools/fix_spilling.sh --above 3 --submit mp-11713 mp-7988
#
# Per material (skipped while it has jobs in the queue):
#   1. the old LOBSTER run goes to materials/<id>/round_<i>_<date>/08_lobster/ (inputs copied,
#      OUTCAR/lobsterout/ICO*LIST moved, WAVECAR and the *CAR.lobster curves deleted) with WHY.txt;
#      repair_history.tsv gets a line "lobster spilling_high";
#   2. 08_lobster/INCAR: NBANDS = --factor (2) x the largest pbeVaspFit2015 basis of the cell
#      (never fewer than before); KPAR made a divisor of the cores;
#   3. one job (08_lobster/basis_scan.sbatch): the symmetry-off NSCF, then LOBSTER once per
#      basis (min/standard/max per element); the lowest charge spilling is kept
#      (08_lobster/basis_scan.tsv, basis_scan_best.txt); WAVECAR deleted, plots redrawn.
# Options: --above X (5), --factor F (2), R_PARTITION (standard), R_NTASKS (40), R_TIME (04:00:00).
set -u
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD="$TOOLS/../modules"
[ -d "$HERE/materials" ] || { echo "run from the HT folder (no materials/ in $HERE)"; exit 1; }
above=5; factor=2; submit=0; ids=()
while [ $# -gt 0 ]; do
    case $1 in
        --above) above=$2; shift 2 ;; --factor) factor=$2; shift 2 ;;
        --submit) submit=1; shift ;; -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) ids+=("$1"); shift ;;
    esac
done
NT=${R_NTASKS:-40}
spill() { grep -i "abs. charge spilling" "$1" 2>/dev/null | awk '{gsub("%","",$NF); print $NF}' | sort -g | tail -1; }
if [ ${#ids[@]} -eq 0 ]; then
    for f in "$HERE"/materials/*/08_lobster/lobsterout; do
        s=$(spill "$f"); [ -n "$s" ] || continue
        awk -v s="$s" -v t="$above" 'BEGIN{exit !(s > t)}' && ids+=("$(basename "$(dirname "$(dirname "$f")")")")
    done
fi
[ ${#ids[@]} -gt 0 ] || { echo "no material with abs. charge spilling above $above %"; exit 0; }
Q=" $(squeue -h -u "${USER:-$(whoami)}" -o %j 2>/dev/null | tr '\n' ' ') "
part=${R_PARTITION:-standard}
[ $submit = 1 ] || echo "(dry run - add --submit to do it)"
for id in "${ids[@]}"; do
    m=$HERE/materials/$id; L=$m/08_lobster
    s=$(spill "$L/lobsterout")
    echo "== $id   abs. charge spilling ${s:-?} %"
    [ -f "$L/INCAR" ] && [ -s "$m/02_scf/CHGCAR" ] || { echo "   no 08_lobster/INCAR or 02_scf/CHGCAR - skipped"; continue; }
    case "$Q" in *" ${id}_"*) echo "   jobs queued - skipped"; continue ;; esac
    cp -p "$MOD/lobster_basis.py" "$MOD/lobster_basis_table.json" "$L/"
    [ -f "$L/POSCAR" ] || cp -p "$m/02_scf/POSCAR" "$L/POSCAR"
    [ -e "$L/POTCAR" ] || ln -s ../POTCAR "$L/POTCAR"
    ( cd "$L" && python3 lobster_basis.py info . ) | sed 's/^/   /'
    old=$(awk -F= 'toupper($1) ~ /^[ \t]*NBANDS[ \t]*$/ {print $2+0}' "$L/INCAR" | tail -1)
    new=$(cd "$L" && python3 lobster_basis.py nbands . --factor "$factor") || { echo "   could not count basis functions - skipped"; continue; }
    [ -n "$old" ] && [ "$old" -gt "$new" ] && new=$old
    echo "   NBANDS ${old:-default} -> $new"
    [ $submit = 1 ] || continue

    cd "$m" || continue
    i=$(( $(command ls -d round_[0-9]*_* stale_* 2>/dev/null | wc -l) + 1 ))
    stamp=$(date +%Y%m%d_%H%M); RB=round_${i}_$stamp/08_lobster
    mkdir -p "$RB"
    for f in INCAR KPOINTS POSCAR lobsterin; do [ -f "08_lobster/$f" ] && cp -p "08_lobster/$f" "$RB/"; done
    for f in OUTCAR OSZICAR EIGENVAL vasp.out lobsterout lobster.out ICOHPLIST.lobster ICOBILIST.lobster \
             ICOOPLIST.lobster CHARGE.lobster basis_scan.tsv basis_scan_best.txt; do
        [ -e "08_lobster/$f" ] && mv -f "08_lobster/$f" "$RB/"
    done
    rm -rf 08_lobster/basis_scan
    for f in WAVECAR CHG CHGCAR vasprun.xml DOSCAR PROCAR IBZKPT PCDAT REPORT STOPCAR .vf_resume .vf_restarts; do rm -f "08_lobster/$f"; done
    rm -f 08_lobster/*.lobster
    for f in "$id-${id}_lobster"-*.out "$id-${id}_lobster"-*.err; do [ -e "$f" ] && mv -f "$f" "$RB/"; done
    printf 'fix_spilling.sh %s: abs. charge spilling %s %% > %s %%\nNBANDS %s -> %s, then LOBSTER basis scan (min/standard/max per element).\n' \
        "$(date '+%F %T')" "${s:-?}" "$above" "${old:-default}" "$new" > "round_${i}_$stamp/WHY.txt"
    sed -i.bak -E "/^[[:space:]]*NBANDS[[:space:]]*=/d" 08_lobster/INCAR && rm -f 08_lobster/INCAR.bak
    printf 'NBANDS = %s   ! fix_spilling.sh: %s x largest LOBSTER basis\n' "$new" "$factor" >> 08_lobster/INCAR
    kp=$(awk -F= 'toupper($1) ~ /^[ \t]*KPAR[ \t]*$/ {print $2+0}' 08_lobster/INCAR | tail -1)
    if [ -n "$kp" ] && [ "$kp" -gt 0 ] && [ $((NT % kp)) -ne 0 ]; then
        k=$kp; while [ $((NT % k)) -ne 0 ]; do k=$((k - 1)); done
        sed -i.bak -E "s/^[[:space:]]*KPAR[[:space:]]*=.*/KPAR = $k/" 08_lobster/INCAR && rm -f 08_lobster/INCAR.bak
    fi
    acct=$(sed -n 's/^#SBATCH --account=//p' job.sbatch)
    cat > 08_lobster/basis_scan.sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${id}_lobster
#SBATCH --partition=$part
#SBATCH --account=${acct:-elmgroup}
#SBATCH --nodes=1
#SBATCH --ntasks=$NT
#SBATCH --time=${R_TIME:-04:00:00}
#SBATCH --mem=0
#SBATCH --output=$m/$id-${id}_lobster-%j.out
#SBATCH --error=$m/$id-${id}_lobster-%j.err
# one job: the symmetry-off NSCF, then LOBSTER once per basis (lowest spilling kept), then
# WAVECAR deleted and the plots redrawn (job.sbatch would delete the WAVECAR before a scan)
VASPFLOW_NO_CHECK=1 source "$HERE/env.sh" >/dev/null 2>&1
cd "$L" || exit 1
rm -f STOPCAR
[ -f copy_from_scf.sh ] && bash ./copy_from_scf.sh
[ -s CHGCAR ] || cp "$m/02_scf/CHGCAR" .
echo "--- LOBSTER NSCF (NBANDS $new): \$(date)"
export OMP_NUM_THREADS=1
\$VASP_LAUNCH "\$VASP_STD"
grep -q "General timing and accounting" OUTCAR || { echo "!!! NSCF did not finish"; exit 1; }
echo "--- LOBSTER basis scan: \$(date)"
export OMP_NUM_THREADS=$NT
python3 lobster_basis.py scan . --lobster "\$LOBSTER_BIN" --clean || exit 1
rm -f CHGCAR CHG
cd "$m"
if [ -f analyze.sh ]; then
    [ -n "\${ANALYSIS_PYTHON:-}" ] && export PATH="\$(dirname "\$ANALYSIS_PYTHON"):\$PATH"
    mkdir -p analysis && MPLBACKEND=Agg bash analyze.sh > analysis/analyze.log 2>&1
fi
echo "--- done: \$(date)"
EOF
    j1=$(sbatch --parsable 08_lobster/basis_scan.sbatch) || { echo "   !!! sbatch failed"; cd "$HERE"; continue; }
    printf '%s\t%s\tlobster\tspilling_high\t1.00\t%s\tfix_spilling.sh\n' "$i" "$stamp" \
        "NBANDS ${old:-default} -> $new; LOBSTER basis scan" >> repair_history.tsv
    echo "   submitted NSCF + LOBSTER basis scan = $j1 ($part, 1 node x $NT); backup in $m/round_${i}_$stamp/"
    cd "$HERE"
done
