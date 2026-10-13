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
#      (never fewer than before); KPAR made a divisor of the cores; 08_lobster/KPOINTS: the mesh
#      reduced to the v2.5 limits (<= 10000 points, k-points x plane waves <= 6e6) if larger;
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
cap_mesh() {   # MATERIAL_DIR WRITE(0|1): LOBSTER mesh within the v2.5 limits (<= 10000 points, k x PW <= 6e6)
    python3 - "$1" "${R_MAX_KPTS:-10000}" "${R_MAX_KPW:-6e6}" "$2" <<'PY'
import sys, os, math
m, maxk, budget, write = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == '1'
kp = open(os.path.join(m, '02_scf/KPOINTS')).read().split('\n')
mesh = [int(x) for x in kp[3].split()[:3]]
pos = os.path.join(m, '02_scf/POSCAR')
L = open(pos).read().split('\n')
sc = float(L[1].split()[0])
A = [[float(x) for x in L[i].split()[:3]] for i in (2, 3, 4)]
det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
       + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
vol = -sc if sc < 0 else abs(det) * sc ** 3
la, lb = (math.sqrt(sum(x * x for x in v)) for v in A[:2])
cosg = sum(a * b for a, b in zip(A[0], A[1])) / (la * lb)
hexa = abs(la - lb) < 1e-3 * la and abs(abs(cosg) - 0.5) < 1e-3
encut = 500.0
for l in open(os.path.join(m, '08_lobster/INCAR')):
    if l.split('=')[0].strip().upper() == 'ENCUT':
        encut = float(l.split('=')[1].split()[0])
npw = vol * (encut / 3.80998) ** 1.5 / (6 * math.pi ** 2)
def nk(g):
    n = g[0] * g[1] * g[2]; t = 2 ** sum(1 for x in g if x % 2 == 0)
    return (n - t) // 2 + t
def fits(g):
    return (not maxk or g[0] * g[1] * g[2] <= maxk) and (not budget or nk(g) * npw <= budget)
def snap(n, i):
    if n <= 1: return 1
    if hexa and i < 2 and n >= 6: return max(6, int(round(n / 6.0)) * 6)
    return max(2, int(round(n / 2.0)) * 2)
new, f = list(mesh), 1.0
while not fits(new) and f > 0.05 and max(new) > 2:
    f *= 0.95
    new = [snap(int(round(n * f)) if n > 1 else 1, i) for i, n in enumerate(mesh)]
tag = '' if new == mesh else ' reduced from %dx%dx%d' % tuple(mesh)
if write and new != mesh:
    open(os.path.join(m, '08_lobster/KPOINTS'), 'w').write(
        "Automatic Gamma mesh (LOBSTER, SCF mesh%s: %d k x ~%d PW; ISYM=0)\n0\nGamma\n  %d  %d  %d\n  0  0  0\n"
        % (tag, nk(new), npw, new[0], new[1], new[2]))
print('   LOBSTER mesh %dx%dx%d -> %dx%dx%d  (%d k-points x ~%d plane waves = %.2g)'
      % tuple(mesh + new + [nk(new), npw, nk(new) * npw]))
PY
}
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
    [ $submit = 1 ] || { cap_mesh "$m" 0; continue; }

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
    cap_mesh "$m" 1 || echo "   could not check the k-mesh"
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
