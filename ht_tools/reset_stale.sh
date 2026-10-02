#!/bin/bash
# Prepare materials of a VASP-Flow highthroughput run (ht-semimetals layout) so that
# ./submit_all.sh redoes exactly what is invalid, in the right order.
#
#   A  relax_unconverged : 01_relax never reached "required accuracy" but SCF/bands/
#      LOBSTER already ran on that geometry.  submit_all.sh alone would only rerun
#      the relax (from 01_relax/POSCAR = the *first-pass* CONTCAR) and keep the old
#      SCF/bands/LOBSTER.  Fix: POSCAR <- newest complete CONTCAR, and the downstream
#      outputs are moved aside so the whole chain relax->scf->bands->lobster reruns.
#   B  lobster_truncated : lobster.out says "finished" but a COHPCAR/COBICAR/COOPCAR
#      is empty or truncated (WAVECAR is already deleted) -> rerun the 08_lobster step.
#   C  bands_blank       : 03_bands/EIGENVAL missing or all-zero -> rerun bands (+ LOBSTER
#      is untouched).
#   D  relax_after_downstream : the relax finished (converged) AFTER SCF/bands/LOBSTER
#      ran, e.g. a relax added or resubmitted later -- they used the unrelaxed cell and
#      submit_all.sh would never redo them.  The converged relax is kept; SCF, bands and
#      LOBSTER outputs are moved aside so they rerun from its CONTCAR.
#
# Nothing is deleted: moved files go to <material>/stale_<date>/<step>/.
# Materials with jobs in the queue are skipped (cancel them first, see below).
#
# Usage (run from the HT root, e.g. cd ~/VASP-Flow/HT_1-500):
#   bash ../ht_tools/reset_stale.sh            dry run: list what would be done
#   bash ../ht_tools/reset_stale.sh --apply    do it; then run ./submit_all.sh
#   HT_DIR=~/VASP-Flow/HT_1-500 bash reset_stale.sh ...   from anywhere
#
# If relax-only jobs from an earlier ./submit_all.sh are still queued for case-A
# materials, cancel them first:   scancel --name=<mp-id>_relax   (the dry run lists them)
set -u
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"
STAMP=$(date +%Y%m%d_%H%M)
USER_NAME="${USER:-$(whoami)}"
QUEUE=$(squeue -h -u "$USER_NAME" -o '%j' 2>/dev/null)

relaxed()  { grep -q "reached required accuracy" "$1/01_relax/OUTCAR" 2>/dev/null; }
finished() { grep -q "General timing and accounting" "$1/$2/OUTCAR" 2>/dev/null; }
lob_done() { grep -q "finished in" "$1/08_lobster/lobster.out" 2>/dev/null; }
car_ok() {      # all three curve files present with the line count of the first one
    local n="" f m
    for f in COHPCAR COBICAR COOPCAR; do
        [ -s "$1/08_lobster/$f.lobster" ] || return 1
        m=$(wc -l < "$1/08_lobster/$f.lobster")
        [ -z "$n" ] && n=$m
        [ "$m" = "$n" ] || return 1
    done
}
eig_ok() {      # EIGENVAL present and not all-zero eigenvalues
    [ -s "$1/03_bands/EIGENVAL" ] || return 1
    awk 'NR>7 && NF>=2 && $2+0!=0 {found=1; exit} END{exit !found}' "$1/03_bands/EIGENVAL"
}
stash() {       # $1 material, $2 step, files...
    local m=$1 st=$2; shift 2
    local dst="$m/stale_$STAMP/$st"
    for f in "$@"; do
        [ -e "$m/$st/$f" ] || continue
        if [ $APPLY = 1 ]; then mkdir -p "$dst" && mv "$m/$st/$f" "$dst/"; fi
    done
}
newest_contcar() {   # print the path of the newest CONTCAR with all atom lines
    local d=$1/01_relax nat c
    nat=$(sed -n 7p "$d/POSCAR" | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')
    for c in $(ls -t "$d"/CONTCAR* 2>/dev/null); do
        [ "$(wc -l < "$c")" -ge $((8 + nat)) ] && { echo "$c"; return 0; }
    done
    return 1
}

nA=0; nB=0; nC=0; nD=0; busyA=""
cd "$HERE/materials" || { echo "no materials/ under $HERE"; exit 1; }
for d in mp-*/; do
    m=${d%/}
    [ -d "$m/01_relax" ] || continue
    if printf '%s\n' "$QUEUE" | grep -q "^${m}_"; then
        # case-A materials with a (stale) relax-only job queued: must be cancelled first
        if ! relaxed "$m" && finished "$m" 02_scf; then busyA="$busyA ${m}_relax"; fi
        # (a running relax for such a material shows up as case D once it has finished)
        continue
    fi

    if ! relaxed "$m" && finished "$m" 02_scf; then
        c=$(newest_contcar "$m") || c=""
        echo "A $m: relax unconverged; restart from ${c:-POSCAR}; redo scf, bands, lobster"
        if [ $APPLY = 1 ]; then
            mkdir -p "$m/stale_$STAMP/01_relax"
            cp -p "$m/01_relax/POSCAR" "$m/stale_$STAMP/01_relax/POSCAR.before"
            [ -n "$c" ] && cp "$c" "$m/01_relax/POSCAR"
            rm -f "$m/01_relax/.vf_resume" "$m/01_relax/.vf_restarts"
        fi
        stash "$m" 01_relax OUTCAR OSZICAR OUTCAR.pass1 OSZICAR.pass1
        stash "$m" 02_scf   OUTCAR OSZICAR CONTCAR vasprun.xml
        stash "$m" 03_bands OUTCAR EIGENVAL vasprun.xml
        stash "$m" 08_lobster OUTCAR lobster.out lobsterout ICOHPLIST.lobster ICOBILIST.lobster \
              ICOOPLIST.lobster COHPCAR.lobster COBICAR.lobster COOPCAR.lobster lobster_summary.csv
        [ $APPLY = 1 ] && rm -f "$m/.done" "$m/.vf_collected"
        nA=$((nA + 1)); continue
    fi
    if relaxed "$m" && finished "$m" 02_scf && [ "$m/01_relax/OUTCAR" -nt "$m/02_scf/OUTCAR" ]; then
        echo "D $m: relax finished after SCF/bands/LOBSTER ran (they used the unrelaxed cell); redo them"
        stash "$m" 02_scf   OUTCAR OSZICAR CONTCAR vasprun.xml
        stash "$m" 03_bands OUTCAR EIGENVAL vasprun.xml
        stash "$m" 08_lobster OUTCAR lobster.out lobsterout ICOHPLIST.lobster ICOBILIST.lobster \
              ICOOPLIST.lobster COHPCAR.lobster COBICAR.lobster COOPCAR.lobster lobster_summary.csv
        [ $APPLY = 1 ] && rm -f "$m/.done" "$m/.vf_collected" "$m"/0[238]_*/.vf_resume
        nD=$((nD + 1)); continue
    fi
    if lob_done "$m" && ! car_ok "$m"; then
        echo "B $m: LOBSTER curve files truncated; redo the 08_lobster step"
        stash "$m" 08_lobster OUTCAR lobster.out lobsterout ICOHPLIST.lobster ICOBILIST.lobster \
              ICOOPLIST.lobster COHPCAR.lobster COBICAR.lobster COOPCAR.lobster lobster_summary.csv
        [ $APPLY = 1 ] && rm -f "$m/.done" "$m/.vf_collected" "$m/08_lobster/.vf_resume"
        nB=$((nB + 1))
    fi
    if finished "$m" 03_bands && ! eig_ok "$m"; then
        echo "C $m: 03_bands/EIGENVAL blank; redo bands"
        stash "$m" 03_bands OUTCAR EIGENVAL vasprun.xml
        [ $APPLY = 1 ] && rm -f "$m/.done" "$m/.vf_collected" "$m/03_bands/.vf_resume"
        nC=$((nC + 1))
    fi
done

echo ""
echo "A relax unconverged: $nA   D relax newer than SCF: $nD   B LOBSTER truncated: $nB   C bands blank: $nC"
if [ -n "$busyA" ]; then
    echo ""
    echo "$(echo $busyA | wc -w) unconverged-relax material(s) have a relax-only job queued (would"
    echo "leave SCF/bands/LOBSTER on the old geometry). Cancel them, then run this script again:"
    echo "  scancel --name=$(echo $busyA | tr ' ' ',')"
fi
if [ $APPLY = 1 ]; then echo "Done. Now run ./submit_all.sh"
else echo "Dry run - nothing changed. Re-run with --apply."; fi
