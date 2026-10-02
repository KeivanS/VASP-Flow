#!/bin/bash
# Collect the data needed for the descriptor database from a VASP-Flow
# highthroughput run (ht-semimetals.py layout) into one small tarball.
#
# Per material it packs only data -- no plot scripts, no images:
#   POSCAR                      input structure (MP primitive cell)
#   01_relax/CONTCAR            relaxed structure
#   02_scf/INCAR                settings (ENCUT, GGA+U LDAU*, ISPIN) for energy comparisons
#   08_lobster/POSCAR           structure LOBSTER actually used (site order of the labels)
#   08_lobster/ICOHPLIST.lobster ICOBILIST.lobster ICOOPLIST.lobster
#   08_lobster/COHPCAR.lobster  COBICAR.lobster  COOPCAR.lobster   (needed for the
#                               antibonding integrals ACOHP/ACOBI/ACOOP; skip with NO_CARS=1)
#   08_lobster/lobsterin lobsterout lobster.out CHARGE.lobster MadelungEnergies.lobster
#   08_lobster/lobster_summary.csv  analysis/analyze.log
#   03_bands/EIGENVAL           band gap / band edges (skip with NO_EIGENVAL=1)
#   <step>/OUTCAR.trim          key OUTCAR lines only (E-fermi, NELECT, ISPIN, energies, ...)
# and two tables at the top level:
#   _status.tsv   per material (ALL materials): which steps finished (same tests as
#                 submit_all.sh) and whether this archive contains it (packed=1)
#   _skipped.tsv  files larger than MAX_FILE_SIZE that were left out
#
# Incremental: only materials whose LOBSTER step is finished AND that changed since
# they were last collected are packed (marker <material>/.vf_collected, touched after
# a successful pack; reset_stale.sh removes it).  ALL=1 packs every finished material.
#
# Usage (run from the HT root, e.g. cd ~/VASP-Flow/HT_1-500):
#   bash ../ht_tools/collect_lobster.sh [ROOT] [MAX_FILE_SIZE]
#   ROOT           default: ./materials
#   MAX_FILE_SIZE  default: 50M
#   ALL=1                     repack everything finished (ignore .vf_collected)
#   NO_CARS=1 NO_EIGENVAL=1   for a minimal archive
set -u
ROOT=${1:-$PWD/materials}
MAX=${2:-50M}
STAMP=$(date +%Y%m%d_%H%M)
OUT=$HOME/lobster_collect_$STAMP.tar.gz
WORK=$(mktemp -d)
STAGE="$WORK/lobster_collect_$STAMP"
mkdir -p "$STAGE"

cd "$ROOT" || { echo "Cannot cd to $ROOT"; exit 1; }

OUTCAR_KEYS='NIONS|NBANDS=|ISPIN|NELECT|E-fermi|energy  without entropy|free  energy|number of electron|magnetization|reached required accuracy|aborting loop because EDIFF|General timing|Elapsed time'
LOB_FILES="POSCAR ICOHPLIST.lobster ICOBILIST.lobster ICOOPLIST.lobster lobsterin lobsterout lobster.out CHARGE.lobster MadelungEnergies.lobster lobster_summary.csv"
[ "${NO_CARS:-0}" = 1 ] || LOB_FILES="$LOB_FILES COHPCAR.lobster COBICAR.lobster COOPCAR.lobster"

done_relax()   { [ ! -d "$1/01_relax" ] || grep -q "reached required accuracy" "$1/01_relax/OUTCAR" 2>/dev/null; }
done_vasp()    { grep -q "General timing and accounting" "$1/$2/OUTCAR" 2>/dev/null; }
done_lobster() { grep -q "finished in" "$1/08_lobster/lobster.out" 2>/dev/null; }
yn() { "$@" && echo 1 || echo 0; }

# copy SRC (relative to ROOT) into the stage if it exists and is not too big
take() {
    [ -f "$1" ] || return 0
    if [ -n "$(find "$1" -size +"$MAX")" ]; then
        printf '%s\t%s\n' "$(stat -c %s "$1")" "$1" >> "$STAGE/_skipped.tsv"
        return 0
    fi
    mkdir -p "$STAGE/$(dirname "$1")" && cp -p "$1" "$STAGE/$1"
}

printf 'material\trelax\tscf\tbands\tlobster\tall_done\tn_icohp\tpacked\n' > "$STAGE/_status.tsv"
printf 'bytes\tpath\n' > "$STAGE/_skipped.tsv"

# changed since last collection? (any key output newer than the .vf_collected marker)
changed() {
    [ "${ALL:-0}" = 1 ] || [ ! -f "$1/.vf_collected" ] && return 0
    [ -n "$(find "$1/01_relax/OUTCAR" "$1/03_bands/EIGENVAL" "$1/08_lobster/lobster.out" \
              -newer "$1/.vf_collected" 2>/dev/null)" ]
}

n=0; np=0; PACKED="$WORK/packed.txt"; : > "$PACKED"
for d in mp-*/; do
    m=${d%/}
    r=$(yn done_relax "$m"); s=$(yn done_vasp "$m" 02_scf)
    b=$(yn done_vasp "$m" 03_bands); l=$(yn done_lobster "$m")
    a=$([ -f "$m/.done" ] && echo 1 || echo 0)
    ni=$(grep -c '^ *[0-9]' "$m/08_lobster/ICOHPLIST.lobster" 2>/dev/null); ni=${ni:-0}
    n=$((n + 1))
    p=0; [ "$l" = 1 ] && changed "$m" && p=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$m" "$r" "$s" "$b" "$l" "$a" "$ni" "$p" >> "$STAGE/_status.tsv"
    [ "$p" = 1 ] || continue
    echo "$m" >> "$PACKED"; np=$((np + 1))

    take "$m/POSCAR"
    take "$m/01_relax/CONTCAR"
    take "$m/02_scf/INCAR"
    take "$m/analysis/analyze.log"
    for f in $LOB_FILES; do take "$m/08_lobster/$f"; done
    [ "${NO_EIGENVAL:-0}" = 1 ] || take "$m/03_bands/EIGENVAL"
    for st in 01_relax 02_scf 03_bands 08_lobster; do
        [ -f "$m/$st/OUTCAR" ] || continue
        mkdir -p "$STAGE/$m/$st"
        grep -E "$OUTCAR_KEYS" "$m/$st/OUTCAR" > "$STAGE/$m/$st/OUTCAR.trim"
    done
done

if tar -czf "$OUT" -C "$WORK" "lobster_collect_$STAMP"; then
    while read -r m; do touch "$m/.vf_collected"; done < "$PACKED"
else
    echo "ERROR: tar failed - markers not updated"; rm -rf "$WORK"; exit 1
fi
rm -rf "$WORK"

echo "Materials scanned: $n   packed (new or changed, LOBSTER finished): $np"
awk -F'\t' 'NR>1{r+=$2;s+=$3;b+=$4;l+=$5;a+=$6} END{printf "Finished steps: relax %d, scf %d, bands %d, lobster %d, all (.done) %d\n",r,s,b,l,a}' \
    <(tar -xzOf "$OUT" "lobster_collect_$STAMP/_status.tsv")
echo "Archive: $OUT  ($(du -h "$OUT" | cut -f1))"
echo "On your Mac:  scp ke4c@login.hpc.virginia.edu:$OUT ~/PROJECTS/AI_ML/MYDATA/"
