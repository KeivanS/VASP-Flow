#!/bin/bash
# Redo the LOBSTER step of high-throughput materials on ONE node, with the LOBSTER
# k-mesh set by the v2.5 rule: the SCF mesh, reduced (same ratios, even N_i) until
# it has <= 10000 points and k-points x plane waves <= 6e6 (ISYM=0 memory limit).
#
#   cd /scratch/$USER/VASP-Flow/HT_ins_1
#   bash ~/VASP-Flow/ht_tools/redo_lobster.sh mp-972937 mp-20352
#
# Per material: cancels its queued/running jobs, rewrites 08_lobster/KPOINTS,
# makes KPAR a divisor of the cores, moves the old 08_lobster outputs to
# stale_<date>/08_lobster/, submits  job.sbatch lobster.
# Override: R_PARTITION (standard), R_NTASKS (40), R_TIME (03:00:00),
#           R_MAX_KPTS (10000), R_MAX_KPW (6e6).
set -u
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"
[ -d "$HERE/materials" ] || { echo "run from the highthroughput folder (no materials/ in $HERE)"; exit 1; }
[ $# -gt 0 ] || { echo "usage: redo_lobster.sh ID [ID ...]"; exit 1; }
NT=${R_NTASKS:-40}
shopt -s nullglob

for id in "$@"; do
    m="$HERE/materials/$id"
    echo "== $id"
    [ -f "$m/08_lobster/INCAR" ] && [ -f "$m/02_scf/KPOINTS" ] || { echo "   no 08_lobster/INCAR or 02_scf/KPOINTS - skipped"; continue; }
    grep -q "General timing and accounting" "$m/02_scf/OUTCAR" 2>/dev/null \
        || { echo "   02_scf has not finished - skipped"; continue; }
    [ -s "$m/02_scf/CHGCAR" ] || { echo "   02_scf/CHGCAR missing - skipped"; continue; }
    cd "$m" || continue

    q=$(squeue -h -u "$USER" -o '%i %j' 2>/dev/null | awk -v n="${id}_" 'index($2, n) == 1 {print $1}')
    if [ -n "$q" ]; then echo "   cancelling jobs: $q"; scancel $q; sleep 5; fi

    python3 - "$m" "${R_MAX_KPTS:-10000}" "${R_MAX_KPW:-6e6}" <<'PY' || { echo "   could not set the mesh - skipped"; continue; }
import sys, os, math
m, maxk, budget = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
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
open(os.path.join(m, '08_lobster/KPOINTS'), 'w').write(
    "Automatic Gamma mesh (LOBSTER, SCF mesh%s: %d k x ~%d PW; ISYM=0)\n0\nGamma\n  %d  %d  %d\n  0  0  0\n"
    % (tag, nk(new), npw, new[0], new[1], new[2]))
print('   LOBSTER mesh %dx%dx%d -> %dx%dx%d  (%d k-points x ~%d plane waves = %.2g)'
      % tuple(mesh + new + [nk(new), npw, nk(new) * npw]))
PY

    kp=$(awk -F= 'toupper($1) ~ /^[ \t]*KPAR[ \t]*$/ {print $2+0}' 08_lobster/INCAR | tail -1)
    if [ -n "$kp" ] && [ "$kp" -gt 0 ] && [ $((NT % kp)) -ne 0 ]; then
        k=$kp; while [ $((NT % k)) -ne 0 ]; do k=$((k - 1)); done
        sed -i -E "s/^[[:space:]]*KPAR[[:space:]]*=.*/KPAR = $k/I" 08_lobster/INCAR; echo "   KPAR $kp -> $k"
    fi

    st="stale_$(date +%Y%m%d_%H%M)/08_lobster"; moved=0
    for f in 08_lobster/{OUTCAR,OSZICAR,vasprun.xml,WAVECAR,EIGENVAL,DOSCAR,CONTCAR,XDATCAR,IBZKPT,PCDAT,REPORT,CHGCAR,CHG,lobsterout,lobster.out} \
             08_lobster/*.lobster 08_lobster/ICO*LIST* 08_lobster/slurm-*.out 08_lobster/*.timeout*; do
        [ -e "$f" ] || continue
        mkdir -p "$st"; mv "$f" "$st/"; moved=$((moved + 1))
    done
    [ $moved -gt 0 ] && echo "   $moved old output file(s) -> $st/"

    part=${R_PARTITION:-standard}
    jid=$(sbatch --parsable --job-name="${id}_lobster" --partition="$part" --nodes=1 \
          --ntasks-per-node="$NT" --time="${R_TIME:-03:00:00}" job.sbatch lobster) \
        && echo "   submitted ${id}_lobster = $jid ($part, 1 node x $NT)" \
        || echo "   !!! sbatch failed"
done
