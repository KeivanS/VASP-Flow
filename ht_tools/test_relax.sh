#!/bin/bash
# Run a test relaxation of one material with modified INCAR tags, outside materials/
# (the production chain and the collection scripts never see it).
#
#   cd ~/VASP-Flow/HT_1-500
#   bash ../ht_tools/test_relax.sh mp-8 ibrion1 "IBRION=1"
#   bash ../ht_tools/test_relax.sh mp-404 ibrion1 "IBRION=1; NSW=60"
#
# -> tests/<mp-id>_<tag>/ with the material's 01_relax INCAR/KPOINTS, its POTCAR and the
#    ORIGINAL input cell (materials/<id>/POSCAR); each "TAG=VALUE" replaces that tag in the
#    INCAR (or is appended).  Submitted with the run's env.sh, partition and account.
#    Afterwards:  bash ../ht_tools/test_relax.sh --report tests/<mp-id>_<tag>
#
# Small, short jobs start sooner (SLURM backfills them into gaps): defaults are
# 8 cores on a shared node, 1 h (an 8-atom magnetic cell needed ~40 min).  Override with
#   T_NTASKS=40 T_TIME=01:00:00 T_PARTITION=standard bash ../ht_tools/test_relax.sh ...
# KPAR is set to T_NTASKS (NCORE = 1) so the parallel layout fits the core count.
set -u
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"

if [ "${1:-}" = "--report" ]; then
    d=$2
    echo "== $d"
    grep -E "reached required accuracy|ZBRENT|BRIONS|General timing" "$d/OUTCAR" "$d"/slurm-*.out 2>/dev/null | sort -u | head
    paste <(grep " F=" "$d/OSZICAR" | awk '{print $1, $3}') \
          <(grep "volume of cell" "$d/OUTCAR" | tail -n +2 | awk '{print $5}') \
          <(grep "external pressure" "$d/OUTCAR" | awk '{print $4}') \
          <(grep "FORCES: max atom" "$d/OUTCAR" | awk '{print $5}') |
        awk 'BEGIN{print "step  F(eV)        V(A3)    P(kB)  maxF(eV/A)"} {printf "%3d %.7f %8.4f %7.2f %8.4f\n",$1,$2,$3,$4,$5}'
    exit 0
fi

id=$1; tag=$2; mods=${3:-}
src="$HERE/materials/$id"
[ -d "$src/01_relax" ] || { echo "no $src/01_relax"; exit 1; }
dst="$HERE/tests/${id}_$tag"
[ -e "$dst" ] && { echo "$dst exists - pick another tag"; exit 1; }
mkdir -p "$dst"
cp "$src/01_relax/INCAR" "$src/01_relax/KPOINTS" "$dst/"
cp -L "$src/POTCAR" "$dst/POTCAR"
cp "$src/POSCAR" "$dst/POSCAR"
NT=${T_NTASKS:-8}
mods="KPAR=$NT; NCORE=1; $mods"
IFS=';' read -ra kv <<< "$mods"
for x in "${kv[@]}"; do
    t=$(echo "${x%%=*}" | tr -d ' '); v=$(echo "${x#*=}" | sed 's/^ *//')
    [ -z "$t" ] && continue
    if grep -qiE "^[[:space:]]*$t[[:space:]]*=" "$dst/INCAR"; then
        sed -i -E "s|^[[:space:]]*$t[[:space:]]*=.*|$t = $v   ! test_relax $tag|I" "$dst/INCAR"
    else
        echo "$t = $v   ! test_relax $tag" >> "$dst/INCAR"
    fi
done
part=${T_PARTITION:-$(sed -n 's/^#SBATCH --partition=//p' "$src/job.sbatch")}
acct=$(sed -n 's/^#SBATCH --account=//p' "$src/job.sbatch")
cat > "$dst/run.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=test_${id}_$tag
#SBATCH --partition=${part:-standard}
#SBATCH --account=${acct:-elmgroup}
#SBATCH --nodes=1
#SBATCH --ntasks=$NT
#SBATCH --time=${T_TIME:-01:00:00}
#SBATCH --output=slurm-%j.out
VASPFLOW_NO_CHECK=1 source "$HERE/env.sh" >/dev/null 2>&1
cd "$dst"
\$VASP_LAUNCH "\$VASP_STD"
EOF
echo "INCAR changes:"; grep "test_relax" "$dst/INCAR"
(cd "$dst" && sbatch run.sbatch)
