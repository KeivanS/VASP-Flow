#!/bin/bash
# Move a highthroughput run (ht-semimetals layout) to /scratch and resubmit every
# material that did not finish, continuing from where it stopped.
#
# Run it from the HT root in $HOME.  With --apply it
#   1. copies the whole run to <dest>/<HT name> (rsync -a --update: files already in
#      the scratch copy are kept unless the home copy is newer) and leaves a
#      MOVED_TO_SCRATCH.txt note in the home copy - from then on work in scratch;
#   2. per material finds the first step that has to be (re)done and prepares it:
#        relax    not "reached required accuracy": continue from CONTCAR (or the last
#                 XDATCAR frame; the input cell if neither is complete); input kept as
#                 01_relax/POSCAR.orig
#        scf      not finished, hit NELM (-> ALGO = All), or its CHGCAR is gone while
#                 bands/LOBSTER still need it.  Starts from the relaxed CHGCAR when that
#                 file is complete; otherwise from scratch (ISTART = 0, ICHARG = 2)
#        bands    not finished (the SCF CHGCAR is there)
#        lobster  not finished; when the symmetry-off NSCF finished and its WAVECAR is
#                 there, only the LOBSTER binary is rerun
#      Every step that is redone gets its old outputs moved to <material>/stale_<date>/
#      (nothing is deleted); when the relax or SCF is redone, all later steps are too;
#   3. submits  <first> -> ... -> scf -> bands -(afterany)-> lobster  like submit_all.sh
#      (walltimes TIME_* from env.sh, --time to change; at most MAX_QUEUED jobs queued).
# Materials with queued jobs, .done, or every step finished are left alone.
#
# Usage (from the HT root, e.g. cd ~/VASP-Flow/HT_ins_1):
#   bash ../ht_tools/resume_in_scratch.sh                     dry run: what would be done
#   nohup bash ../ht_tools/resume_in_scratch.sh --apply > ~/resume_HT_ins_1.log 2>&1 &
#   options:  --dest DIR             scratch parent (default /scratch/$USER/VASP-Flow)
#             --time STEP=HH:MM:SS   walltime of one step (relax|scf|bands|lobster), repeatable
#             --skip ID[,ID...]      leave these materials alone
#             mp-... mp-...          only these materials
#             MAX_QUEUED=300 bash ...  (default 200; the submission waits while the queue is full)
# Run it under nohup (or tmux/screen): with a full queue it waits for room, and a
# closed terminal would stop it half-way.
set -u
APPLY=0; DEST_ROOT=/scratch/${USER:-$(whoami)}/VASP-Flow; SKIP=" "; IDS=()
T_relax=; T_scf=; T_bands=; T_lobster=
while [ $# -gt 0 ]; do
    case "$1" in
        --apply) APPLY=1 ;;
        --dest)  DEST_ROOT=${2%/}; shift ;;
        --skip)  SKIP="$SKIP${2//,/ } "; shift ;;
        --time)  case "${2%%=*}" in
                     relax|scf|bands|lobster) eval "T_${2%%=*}=\${2#*=}" ;;
                     *) echo "--time takes relax|scf|bands|lobster=HH:MM:SS"; exit 2 ;;
                 esac; shift ;;
        -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
        mp-*)    IDS+=("$1") ;;
        *)       echo "unknown argument: $1 (see --help)"; exit 2 ;;
    esac
    shift
done
MAX_QUEUED=${MAX_QUEUED:-200}
USER_NAME=${USER:-$(whoami)}
STAMP=$(date +%Y%m%d)
SRC=$PWD; HT=$(basename "$SRC"); DEST=$DEST_ROOT/$HT
[ -d materials ] && [ -f env.sh ] || { echo "Run from the HT root (the folder with materials/ and env.sh)"; exit 1; }

# ── 1. copy to scratch ───────────────────────────────────────────────────────
W=$SRC
if [ "$APPLY" = 1 ] && [ "$SRC" != "$DEST" ]; then
    mkdir -p "$DEST" || exit 1
    echo "Copying $SRC -> $DEST (rsync, keeps newer scratch files) ..."
    rsync -a --update "$SRC/" "$DEST/" || { echo "ERROR: rsync failed - nothing submitted"; exit 1; }
    printf 'This run continues in %s (resumed %s).\nCheck, collect and resubmit there, not here.\n' \
        "$DEST" "$(date)" > "$SRC/MOVED_TO_SCRATCH.txt"
    W=$DEST
fi
cd "$W" || exit 1
VASPFLOW_NO_CHECK=1 source ./env.sh >/dev/null 2>&1
[ "$APPLY" = 1 ] || echo "DRY RUN in $W (the --apply run copies to $DEST first and works there)"

if [ ${#IDS[@]} = 0 ]; then
    if [ -f material_list.txt ]; then IDS=($(awk 'NF && $1 !~ /^#/ {print $1}' material_list.txt))
    else IDS=($(cd materials && for x in mp-*/; do echo "${x%/}"; done)); fi
fi
QUEUE=$(squeue -h -u "$USER_NAME" -o '%j' 2>/dev/null)

# ── helpers ──────────────────────────────────────────────────────────────────
relaxed() { [ ! -d "$1/01_relax" ] || grep -q "reached required accuracy" "$1/01_relax/OUTCAR" 2>/dev/null; }
fin()     { grep -q "General timing and accounting" "$1/OUTCAR" 2>/dev/null; }
conv()    { grep -q "aborting loop because EDIFF is reached" "$1/OUTCAR" 2>/dev/null; }
lob_done() { grep -q "finished in" "$1/08_lobster/lobster.out" 2>/dev/null; }
nat()     { sed -n 7p "$1" | awk '{s=0; for (i=1;i<=NF;i++) s+=$i; print s}'; }
# complete CHGCAR: at least ceil(NGX*NGY*NGZ/5) data lines after the grid line
chg_ok()  { awk 'NR==7 {for (i=1;i<=NF;i++) n+=$i}
                 NR==8 && $1 ~ /^[sS]/ {sel=1}
                 NR==10+sel+n {g=$1*$2*$3; need=NR+int((g+4)/5)}
                 END {exit !(g>0 && NR>=need)}' "$1" 2>/dev/null; }
set_tag() { sed -i.bak -E "/^[[:space:]]*$2[[:space:]]*=/d" "$1" && rm -f "$1.bak"; printf '%s = %s\n' "$2" "$3" >> "$1"; }

OUTS="OUTCAR OSZICAR vasprun.xml CONTCAR XDATCAR CHG CHGCAR WAVECAR EIGENVAL DOSCAR PROCAR
      IBZKPT PCDAT REPORT vaspout.h5 ELFCAR LOCPOT STOPCAR .vf_resume .vf_restarts lobsterout lobster.out"
stash() {   # $1 material dir, $2 step dir, [$3 = lobster-only: keep the NSCF]
    local d=$1/$2 s=$1/stale_$STAMP/$2 f list=$OUTS
    [ "${3:-}" = lobster-only ] && list="STOPCAR .vf_resume .vf_restarts lobsterout lobster.out"
    for f in $list "$d"/*.lobster; do
        f=${f#"$d/"}
        [ -e "$d/$f" ] || continue
        mkdir -p "$s" && mv "$d/$f" "$s/"
    done
}
poscar_from_xdatcar() {   # $1 = relax dir: last XDATCAR frame -> POSCAR
    ( cd "$1" && python3 - <<'PYX'
lines = open('XDATCAR').read().splitlines()
idx = [i for i, l in enumerate(lines) if l.strip().lower().startswith('direct configuration')]
if not idx:
    raise SystemExit(1)
last = idx[-1]
nat = sum(int(x) for x in lines[6].split())
if len(lines) < last + 1 + nat:
    raise SystemExit(1)
head = lines[last - 7:last] if last >= 7 and lines[last - 7].strip() == lines[0].strip() else lines[:7]
open('POSCAR', 'w').write('\n'.join(head + ['Direct'] + lines[last + 1:last + 1 + nat]) + '\n')
PYX
    )
}
walltime() {     # env.sh is re-read, so editing TIME_* there affects the rest of a running submission
    local v; eval "v=\${T_$1:-}"
    VASPFLOW_NO_CHECK=1 source ./env.sh >/dev/null 2>&1
    [ -n "$v" ] && { echo "$v"; return; }
    case "$1" in
        relax) echo "${TIME_RELAX:-08:00:00}" ;;  scf)     echo "${TIME_SCF:-04:00:00}" ;;
        bands) echo "${TIME_BANDS:-02:00:00}" ;;  lobster) echo "${TIME_LOBSTER:-04:00:00}" ;;
    esac
}
queued() { squeue -h -u "$USER_NAME" -o '%j' 2>/dev/null | wc -l | tr -d ' '; }

# ── 2.+3. per material ───────────────────────────────────────────────────────
n_sub=0; n_skip=0; C_relax=0; C_scf=0; C_bands=0; C_lobster=0
for m in "${IDS[@]}"; do
    d=materials/$m
    case "$SKIP" in *" $m "*) continue ;; esac
    [ -d "$d" ] || { echo "?? $m: no directory"; continue; }
    [ -f "$d/.done" ] && continue
    if grep -qE "^${m}(_|\$)" <<< "$QUEUE"; then echo "-- $m: jobs queued, skipped"; n_skip=$((n_skip+1)); continue; fi

    need_b=0; fin "$d/03_bands" || need_b=1
    need_l=0; lob_done "$d" || need_l=1
    nscf_ok=0; fin "$d/08_lobster" && [ -s "$d/08_lobster/WAVECAR" ] && nscf_ok=1
    why=; algo=0
    if ! relaxed "$d"; then first=relax
    elif ! fin "$d/02_scf"; then first=scf; why="not finished"
    elif ! conv "$d/02_scf"; then first=scf; why="hit NELM -> ALGO = All"; algo=1
    elif [ ! -s "$d/02_scf/CHGCAR" ] && { [ $need_b = 1 ] || { [ $need_l = 1 ] && [ $nscf_ok = 0 ]; }; }; then
        first=scf; why="CHGCAR gone, needed by bands/LOBSTER"
    elif [ $need_b = 1 ]; then first=bands
    elif [ $need_l = 1 ]; then first=lobster
    else continue; fi

    case $first in
        relax) steps="relax scf bands lobster" ;;
        scf)   steps="scf bands lobster" ;;
        bands) steps="bands"; [ $need_l = 1 ] && steps="bands lobster" ;;
        lobster) steps="lobster" ;;
    esac
    eval "C_$first=\$((C_$first + 1))"

    # describe (and with --apply: prepare) the restart
    note=
    case $first in
    relax)
        r=$d/01_relax; na=$(nat "$r/POSCAR")
        if [ -s "$r/CONTCAR" ] && [ "$(wc -l < "$r/CONTCAR")" -ge $((8 + na)) ]; then geo=CONTCAR
        elif [ -s "$r/XDATCAR" ]; then geo=XDATCAR
        else geo=input; fi
        note="relax from $geo"
        if [ "$APPLY" = 1 ]; then
            [ -f "$r/POSCAR.orig" ] || cp -p "$r/POSCAR" "$r/POSCAR.orig"
            case $geo in
                CONTCAR) cp "$r/CONTCAR" "$r/POSCAR" ;;
                XDATCAR) poscar_from_xdatcar "$r" || note="relax from input (XDATCAR unreadable)" ;;
            esac
            for st in 01_relax 02_scf 03_bands 08_lobster; do stash "$d" $st; done
        fi ;;
    scf)
        r=$d/01_relax; start="relaxed CHGCAR"
        { [ -d "$r" ] && [ -s "$r/CHGCAR" ] && chg_ok "$r/CHGCAR"; } || start="scratch (relax CHGCAR missing or incomplete)"
        [ -d "$r" ] || start="scratch"
        note="scf ($why) from $start"
        if [ "$APPLY" = 1 ]; then
            for st in 02_scf 03_bands 08_lobster; do stash "$d" $st; done
            if [ "$start" != "relaxed CHGCAR" ]; then
                [ -e "$r/CHGCAR" ] && { mkdir -p "$d/stale_$STAMP/01_relax"; mv "$r/CHGCAR" "$r/WAVECAR" "$d/stale_$STAMP/01_relax/" 2>/dev/null; }
                set_tag "$d/02_scf/INCAR" ISTART 0
                set_tag "$d/02_scf/INCAR" ICHARG 2
            fi
            [ $algo = 1 ] && set_tag "$d/02_scf/INCAR" ALGO All
        fi ;;
    bands)
        note="bands"
        if [ "$APPLY" = 1 ]; then
            stash "$d" 03_bands
            if [ $need_l = 1 ]; then
                if [ $nscf_ok = 1 ]; then stash "$d" 08_lobster lobster-only; else stash "$d" 08_lobster; fi
            fi
        fi ;;
    lobster)
        if [ $nscf_ok = 1 ]; then note="LOBSTER binary only (NSCF kept)"; else note="lobster NSCF + LOBSTER"; fi
        if [ "$APPLY" = 1 ]; then
            if [ $nscf_ok = 1 ]; then stash "$d" 08_lobster lobster-only; else stash "$d" 08_lobster; fi
        fi ;;
    esac

    if [ "$APPLY" != 1 ]; then
        echo "   $m: $note  | chain: ${steps// / -> }"
        continue
    fi
    if [ ! -f "$d/POTCAR" ]; then echo "SKIP $m: no POTCAR"; continue; fi
    while [ "$(queued)" -ge "$MAX_QUEUED" ]; do sleep 60; done
    prev=; prevst=; line="$m: $note |"
    for st in $steps; do
        t=$(walltime $st); dep=
        if [ -n "$prev" ]; then
            if [ $st = lobster ] && [ "$prevst" = bands ]; then dep="--dependency=afterany:$prev"
            else dep="--dependency=afterok:$prev"; fi
        fi
        jid=$(cd "$d" && sbatch --parsable --job-name="${m}_$st" --time="$t" $dep job.sbatch $st) \
            || { line="$line FAILED at $st"; break; }
        prev=$jid; prevst=$st; line="$line $st=$jid($t)"
    done
    echo "submitted $line"
    n_sub=$((n_sub + 1))
done

echo
echo "First step to redo:"; for k in relax scf bands lobster; do eval "v=\$C_$k"; printf '  %-8s %4d\n' $k "$v"; done
[ "$n_skip" -gt 0 ] && echo "  (queued, skipped: $n_skip)"
if [ "$APPLY" = 1 ]; then
    echo "Submitted $n_sub material(s) from $W.  Monitor: squeue -u $USER_NAME"
    echo "Later, from $W:  bash ../ht_tools/audit_run.sh ; bash ../ht_tools/collect_lobster.sh"
else
    echo "Dry run - nothing copied or submitted.  Re-run with --apply (under nohup)."
fi
