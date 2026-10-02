#!/bin/bash
# Free disk space in a VASP-Flow highthroughput run (ht-semimetals layout) by deleting
# large VASP files that no remaining step of the chain will read:
#
#   01_relax/{WAVECAR,CHGCAR,CHG}    once 02_scf finished   (SCF copies them at its start)
#   02_scf/{WAVECAR,CHG}             once 02_scf finished   (bands/LOBSTER read only CHGCAR;
#                                                            the SCF WAVECAR is used by DFPT only)
#   03_bands/{WAVECAR,CHGCAR,CHG}    once 03_bands finished
#   08_lobster/{WAVECAR,CHGCAR,CHG}  once LOBSTER finished ("finished in" in lobster.out)
#   02_scf/CHGCAR                    only with --deep, and only when bands AND LOBSTER are
#                                    finished (it is needed to redo bands/LOBSTER later)
#
# Materials with jobs in the queue are never touched. Dry run by default.
#
# Usage (from the HT root, e.g. cd ~/VASP-Flow/HT_1-500):
#   bash ../ht_tools/cleanup_large.sh                   dry run: what would be deleted, how much
#   bash ../ht_tools/cleanup_large.sh --apply           delete
#   bash ../ht_tools/cleanup_large.sh --deep [--apply]  also 02_scf/CHGCAR of complete materials
set -u
APPLY=0; DEEP=0
for a in "$@"; do
    case "$a" in --apply) APPLY=1 ;; --deep) DEEP=1 ;; *) echo "unknown option $a"; exit 2 ;; esac
done
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"
cd "$HERE/materials" || { echo "no materials/ under $HERE"; exit 1; }
QUEUE=$(squeue -h -u "${USER:-$(whoami)}" -o '%j' 2>/dev/null)

finished() { grep -q "General timing and accounting" "$1/$2/OUTCAR" 2>/dev/null; }
lob_done() { grep -q "finished in" "$1/08_lobster/lobster.out" 2>/dev/null; }

LIST=$(mktemp)
add() { local m=$1 st=$2; shift 2; for f in "$@"; do [ -f "$m/$st/$f" ] && echo "$m/$st/$f" >> "$LIST"; done; }
for d in mp-*/; do
    m=${d%/}
    printf '%s\n' "$QUEUE" | grep -q "^${m}_" && continue
    if finished "$m" 02_scf; then
        add "$m" 01_relax WAVECAR CHGCAR CHG
        add "$m" 02_scf   WAVECAR CHG
    fi
    finished "$m" 03_bands && add "$m" 03_bands WAVECAR CHGCAR CHG
    lob_done "$m"          && add "$m" 08_lobster WAVECAR CHGCAR CHG
    [ "$DEEP" = 1 ] && finished "$m" 03_bands && lob_done "$m" && add "$m" 02_scf CHGCAR
done

nfiles=$(wc -l < "$LIST")
kb=$( [ "$nfiles" -gt 0 ] && tr '\n' '\0' < "$LIST" | xargs -0 du -ck 2>/dev/null | tail -1 | cut -f1 || echo 0)
nm=$(cut -d/ -f1 "$LIST" | sort -u | wc -l)
echo "Files: $nfiles in $nm materials, $(awk -v k="${kb:-0}" 'BEGIN{printf "%.1f GB", k/1048576}')"
echo "By type:"; sed 's|^[^/]*/||' "$LIST" | sort | uniq -c | sort -rn | head -12
echo "Skipped (jobs in queue): $(printf '%s\n' "$QUEUE" | sed -n 's/^\(mp-[0-9]*\)_.*/\1/p' | sort -u | grep -c .) materials"
if [ "$APPLY" = 1 ]; then
    tr '\n' '\0' < "$LIST" | xargs -0 rm -f
    echo "Deleted."
else
    echo "Dry run - nothing deleted. Re-run with --apply."
fi
rm -f "$LIST"
