#!/bin/bash
# Audit a highthroughput run (ht_runs.py layout) before its results go into the
# database: was every material of material_list.txt run, did every step converge,
# were the results extracted (collect_lobster.sh archives) and do the backups hold them?
#
# Per material (first problem wins):
#   missing         no materials/<id>/ directory
#   queued          jobs of this material still in squeue
#   relax_unconv    01_relax/OUTCAR lacks "reached required accuracy"
#   scf_missing     02_scf did not reach the final timing block
#   scf_unconv      SCF hit NELM (no "aborting loop because EDIFF is reached")
#   lobster_missing 08_lobster/lobster.out lacks "finished in"
#   lobster_empty   ICOHPLIST.lobster has no bond lines
#   spilling_high   abs. charge spilling > SPILL_MAX % (default 5)
#   ok
# Band-structure problems (bands_missing / bands_unconv / bands_diverged) are listed
# in their own column: LOBSTER does not depend on them.
#
# Extraction: "extracted" = the material's ICOHPLIST is in one of the
# $HOME/lobster_collect_*.tar.gz archives (or ARCHIVES="glob ...").
# Backups: each BACKUP argument is a directory holding a copy of the run
# (<B>/<id>/, <B>/{materials,results}/<id>/ or <B>/<HT name>/{materials,results}/<id>/) or a .tar.gz;
# a material is backed up there when its ICOHPLIST.lobster exists with the same size.
#
# Usage (from the HT root, e.g. cd ~/VASP-Flow/HT_1-500):
#   bash ../ht_tools/audit_run.sh [BACKUP ...]
#   bash ../ht_tools/audit_run.sh /scratch/$USER/HT_1-500_backup ~/HT_1-500_backup
#   SPILL_MAX=3 ARCHIVES="$HOME/lobster_collect_2026*.tar.gz" bash ...
# Writes audit_<date>.tsv in the HT root; read-only otherwise.
set -u
SPILL_MAX=${SPILL_MAX:-5}
HERE=$PWD
HT=$(basename "$HERE")
[ -d materials ] || { echo "Run from the HT root (the folder with materials/)"; exit 1; }
LIST=material_list.txt                 # older runs have none: audit every materials/mp-*
[ -f "$LIST" ] || { LIST=$(mktemp); ls -d materials/mp-*/ 2>/dev/null | xargs -n1 basename > "$LIST"
                    echo "No material_list.txt - auditing the $(wc -l < "$LIST") materials/mp-* directories"; }
OUT=$HERE/audit_$(date +%Y%m%d_%H%M).tsv
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

QUEUE=$(squeue -h -u "${USER:-$(whoami)}" -o '%j' 2>/dev/null)

# materials whose ICOHPLIST is inside a tarball -> one id per line in $2
tar_ids() { tar -tzvf "$1" 2>/dev/null | awk '$NF ~ /mp-[^\/]*\/08_lobster\/ICOHPLIST\.lobster$/ {
                n = split($NF, p, "/"); for (i = 1; i <= n; i++) if (p[i] ~ /^mp-/) print p[i] "\t" $3 }' >> "$2"; }

: > "$WORK/extracted"
for a in ${ARCHIVES:-$HOME/lobster_collect_*.tar.gz}; do [ -f "$a" ] && tar_ids "$a" "$WORK/extracted"; done

NB=$#; i=0
for b in "$@"; do
    i=$((i + 1)); : > "$WORK/bk$i"
    if [ -f "$b" ]; then tar_ids "$b" "$WORK/bk$i"; continue; fi
    [ -d "$b" ] || { echo "WARNING: backup $b not found"; continue; }
    for base in "$b" "$b/materials" "$b/results" "$b/$HT/materials" "$b/$HT/results"; do
        [ -d "$base" ] || continue
        for f in "$base"/mp-*/08_lobster/ICOHPLIST.lobster; do
            [ -f "$f" ] && printf '%s\t%s\n' "$(basename "$(dirname "$(dirname "$f")")")" "$(stat -c %s "$f")" >> "$WORK/bk$i"
        done
    done
done

T=$'\t'
has() { grep -q "^$1$T" "$2"; }                    # id in list (any size)
same() { grep -q "^$1$T$2\$" "$3"; }                # id with this size

hdr='material\tstatus\tbands\tspilling\tn_icohp\textracted'
for i in $(seq 1 $NB); do hdr="$hdr\tbackup$i"; done
printf "$hdr\n" > "$OUT"

while read -r m _; do
    [ -z "$m" ] || [ "${m:0:1}" = "#" ] && continue
    d=materials/$m; st=ok; bands=ok; spill=; ni=0; size=
    if [ ! -d "$d" ]; then st=missing; bands=-
    else
        ni=$(grep -c '^ *[0-9]' "$d/08_lobster/ICOHPLIST.lobster" 2>/dev/null); ni=${ni:-0}
        size=$(stat -c %s "$d/08_lobster/ICOHPLIST.lobster" 2>/dev/null)
        spill=$(grep -i 'abs. charge spilling' "$d/08_lobster/lobster.out" 2>/dev/null \
                | grep -oE '[0-9]+\.[0-9]+' | sort -g | tail -1)
        if   grep -qE "^${m}(_|\$)" <<< "$QUEUE"; then st=queued
        elif [ -d "$d/01_relax" ] && ! grep -q "reached required accuracy" "$d/01_relax/OUTCAR" 2>/dev/null; then st=relax_unconv
        elif ! grep -q "General timing and accounting" "$d/02_scf/OUTCAR" 2>/dev/null; then st=scf_missing
        elif ! grep -q "aborting loop because EDIFF is reached" "$d/02_scf/OUTCAR"; then st=scf_unconv
        elif ! grep -q "finished in" "$d/08_lobster/lobster.out" 2>/dev/null; then st=lobster_missing
        elif [ "$ni" = 0 ]; then st=lobster_empty
        elif [ -n "$spill" ] && awk -v s="$spill" -v x="$SPILL_MAX" 'BEGIN{exit !(s > x)}'; then st=spilling_high
        fi
        if ! grep -q "General timing and accounting" "$d/03_bands/OUTCAR" 2>/dev/null; then bands=bands_missing
        elif awk '/^ *(DAV|RMM|CG):/ && ($3 > 1e6 || $3 < -1e6) {f=1} END{exit !f}' "$d/03_bands/OSZICAR" 2>/dev/null; then bands=bands_diverged
        elif ! grep -q "aborting loop because EDIFF is reached" "$d/03_bands/OUTCAR"; then bands=bands_unconv
        fi
    fi
    ex=0; has "$m" "$WORK/extracted" && ex=1
    line="$m\t$st\t$bands\t${spill:--}\t$ni\t$ex"
    for i in $(seq 1 $NB); do
        if [ -n "$size" ] && same "$m" "$size" "$WORK/bk$i"; then b=1
        elif has "$m" "$WORK/bk$i"; then b=differs
        else b=0; fi
        line="$line\t$b"
    done
    printf "$line\n" >> "$OUT"
done < "$LIST"

echo "Audit of $HT: $(($(wc -l < "$OUT") - 1)) materials -> $OUT"
echo; echo "Status:"; awk -F'\t' 'NR>1{c[$2]++} END{for (k in c) printf "  %-16s %5d\n", k, c[k]}' "$OUT" | sort
echo; echo "Bands:";  awk -F'\t' 'NR>1{c[$3]++} END{for (k in c) printf "  %-16s %5d\n", k, c[k]}' "$OUT" | sort
echo
awk -F'\t' -v nb=$NB 'NR>1 && $2=="ok" { ok++; if ($6==1) ex++; for (i=1;i<=nb;i++) if ($(6+i)==1) b[i]++ }
    END { printf "Finished OK: %d   extracted: %d   not yet extracted: %d\n", ok, ex, ok-ex
          for (i=1;i<=nb;i++) printf "  backup%d holds %d of the %d OK materials\n", i, b[i], ok }' "$OUT"
i=0; for b in "$@"; do i=$((i + 1)); echo "  backup$i = $b"; done
echo
list() { awk -F'\t' -v c="$1" -v k="$2" 'NR>1 && $c==k {printf " %s", $1}' "$OUT"; }
for s in missing relax_unconv scf_missing scf_unconv lobster_missing lobster_empty spilling_high; do
    ids=$(list 2 $s); [ -n "$ids" ] && echo "$s:$ids"
done
ids=$(awk -F'\t' 'NR>1 && $2=="ok" && $6==0 {printf " %s", $1}' "$OUT"); [ -n "$ids" ] && echo "not extracted:$ids"
echo
echo "Next: not ok -> bash ~/VASP-Flow/ht_tools/next_round.sh; not extracted -> collect_lobster.sh"
