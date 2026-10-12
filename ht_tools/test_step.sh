#!/bin/bash
# Run a test of one material's SCF / bands / LOBSTER-NSCF step with modified INCAR tags,
# outside materials/ (the production chain and the collection scripts never see it).
# The relaxation has its own tool, test_relax.sh.
#
#   cd /scratch/$USER/VASP-Flow/HT_redo_plainRe
#   bash ../ht_tools/test_step.sh mp-7889 bands fast "ALGO=Fast"
#   bash ../ht_tools/test_step.sh mp-7889 bands kpar4 "KPAR=4; NCORE=2"
#   bash ../ht_tools/test_step.sh --report tests/mp-7889_bands_*
#
# -> tests/<id>_<step>_<tag>/ with the step's INCAR and KPOINTS, the material's POTCAR,
#    the step's POSCAR (the structure the production run used) and, for bands/lobster,
#    02_scf/CHGCAR.  No WAVECAR is copied: the test starts from random wavefunctions.
#    Each "TAG=VALUE" replaces that tag (or is appended); "TAG=" removes it.
#    KPAR is lowered to a divisor of the core count if needed.  Only VASP runs - the
#    LOBSTER binary is not started.  Submitted with the run's env.sh, partition, account.
#
# Defaults: 40 cores, 1 h.  Override with T_NTASKS=.. T_TIME=.. T_PARTITION=..
set -u
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"

if [ "${1:-}" = "--report" ]; then
    shift
    printf '%-34s %-9s %-6s %5s %16s %16s  %s\n' test status EDIFF steps first_E last_E max_abs_E
    for d in "$@"; do
        [ -d "$d" ] || continue
        st=running; [ -f "$d/OUTCAR" ] || st=queued
        grep -q "General timing and accounting" "$d/OUTCAR" 2>/dev/null && st=finished
        grep -qE "DUE TO TIME LIMIT" "$d"/slurm-*.out 2>/dev/null && st=timeout
        ed=no; grep -q "aborting loop because EDIFF is reached" "$d/OUTCAR" 2>/dev/null && ed=yes
        awk -v d="$(basename "$d")" -v st=$st -v ed=$ed '
            /^ *(DAV|RMM|CG):/ { n++; e=$3+0; if (n==1) f=e; l=e; a=(e<0?-e:e); if (a>m) m=a }
            END { printf "%-34s %-9s %-6s %5d %16.6g %16.6g  %.3g\n", d, st, ed, n, f, l, m }' "$d/OSZICAR" 2>/dev/null \
            || printf '%-34s %-9s\n' "$(basename "$d")" "$st"
    done
    echo "(a max |E| above ~1e5 eV means the run diverged)"
    exit 0
fi

id=$1; step=$2; tag=$3; mods=${4:-}
case $step in
    scf) sd=02_scf ;; bands) sd=03_bands ;; lobster) sd=08_lobster ;;
    *) echo "step must be scf, bands or lobster (relax: test_relax.sh)"; exit 1 ;;
esac
src="$HERE/materials/$id"
[ -f "$src/$sd/INCAR" ] || { echo "no $src/$sd/INCAR"; exit 1; }
dst="$HERE/tests/${id}_${step}_$tag"
[ -e "$dst" ] && { echo "$dst exists - pick another tag"; exit 1; }
mkdir -p "$dst"
cp "$src/$sd/INCAR" "$src/$sd/KPOINTS" "$dst/"
cp -L "$src/POTCAR" "$dst/POTCAR"
if [ -s "$src/$sd/POSCAR" ]; then cp "$src/$sd/POSCAR" "$dst/"; else cp "$src/02_scf/POSCAR" "$dst/"; fi
if [ "$step" != scf ]; then
    cp "$src/02_scf/CHGCAR" "$dst/" || { echo "no 02_scf/CHGCAR - cannot test $step"; rm -rf "$dst"; exit 1; }
else
    mods="ISTART=0; ICHARG=2; $mods"
fi
NT=${T_NTASKS:-40}
IFS=';' read -ra kv <<< "$mods"
for x in ${kv[@]+"${kv[@]}"}; do
    t=$(echo "${x%%=*}" | tr -d ' '); v=$(echo "${x#*=}" | sed 's/^ *//')
    [ -z "$t" ] && continue
    if [ -z "$v" ]; then
        sed -i -E "/^[[:space:]]*$t[[:space:]]*=/Id" "$dst/INCAR"; continue
    fi
    if grep -qiE "^[[:space:]]*$t[[:space:]]*=" "$dst/INCAR"; then
        sed -i -E "s|^[[:space:]]*$t[[:space:]]*=.*|$t = $v   ! test_step $tag|I" "$dst/INCAR"
    else
        echo "$t = $v   ! test_step $tag" >> "$dst/INCAR"
    fi
done
kp=$(awk -F= 'toupper($1) ~ /^[ \t]*KPAR[ \t]*$/ {print $2+0}' "$dst/INCAR" | tail -1)
if [ -n "$kp" ] && [ "$kp" -gt 0 ] && [ $((NT % kp)) -ne 0 ]; then
    k=$kp; while [ $((NT % k)) -ne 0 ]; do k=$((k - 1)); done
    sed -i -E "s|^[[:space:]]*KPAR[[:space:]]*=.*|KPAR = $k   ! test_step: divisor of $NT cores|I" "$dst/INCAR"
fi
part=${T_PARTITION:-$(sed -n 's/^#SBATCH --partition=//p' "$src/job.sbatch")}
acct=$(sed -n 's/^#SBATCH --account=//p' "$src/job.sbatch")
cat > "$dst/run.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=test_${id}_${step}_$tag
#SBATCH --partition=${part:-standard}
#SBATCH --account=${acct:-elmgroup}
#SBATCH --nodes=1
#SBATCH --ntasks=$NT
#SBATCH --time=${T_TIME:-01:00:00}
#SBATCH --output=slurm-%j.out
VASPFLOW_NO_CHECK=1 source "$HERE/env.sh" >/dev/null 2>&1
export OMP_NUM_THREADS=1
cd "$dst"
\$VASP_LAUNCH "\$VASP_STD"
EOF
echo "$(basename "$dst"): INCAR changes:"; grep "test_step" "$dst/INCAR" || echo "  (none - reference run)"
(cd "$dst" && sbatch run.sbatch)
