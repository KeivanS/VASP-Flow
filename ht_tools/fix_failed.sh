#!/bin/bash
# Repair materials of a VASP-Flow highthroughput run (ht-semimetals layout) whose
# step chain broke -- a step ended FAILED / TIMEOUT / OUT_OF_MEMORY and the rest of
# the chain sits in the queue as (DependencyNeverSatisfied) -- and resubmit them.
#
# Per material: find the first unfinished step, look up its last job in sacct,
# classify the failure from sacct + the job's .out/.err + OUTCAR/OSZICAR, apply the
# fix, cancel the dead pending jobs, move partial outputs of that step and every
# later one aside (<material>/stale_<date>/<step>/, nothing is deleted) and submit
#     <step> -> ... -> scf -> bands -(afterany)-> lobster
# LOBSTER only needs the SCF CHGCAR, so a failed bands step no longer blocks it.
#
#   timeout         walltime x2 of the limit it had (a relax continues from its CONTCAR)
#   oom             (incl. UCX "Cannot allocate memory") whole-node memory (--mem=0) and KPAR capped at 8 in the INCARs
#                   (with KPAR = #ranks every rank holds a full copy of the problem)
#   wavecar         SCF died reading the relax WAVECAR (NBANDS / lattice changed):
#                   the SCF starts from the relaxed CHGCAR alone
#   scf_unconv      bands/LOBSTER sit on an SCF that hit NELM: SCF rerun, ALGO = All
#   bands_diverged  NSCF Davidson blew up (|E| > 1e6 eV): NBANDS >= 1.3 x SCF NBANDS,
#                   ALGO = Normal (EDIFF is never changed: 1E-6 is the loosest allowed)
#   unknown         error lines are printed; resubmitted unchanged only with
#                   --retry-unknown (a crash that is not one of the above usually
#                   needs a look first)
#
# Usage (from the HT root, e.g. cd ~/VASP-Flow/HT_1-500):
#   bash ../ht_tools/fix_failed.sh                   dry run, auto-detect broken chains
#   bash ../ht_tools/fix_failed.sh mp-7889 mp-6430   dry run, just these
#   bash ../ht_tools/fix_failed.sh --apply [ids]     do it
#   SINCE=2026-10-01 bash ...                        sacct window (default: 7 days)
set -u
APPLY=0; RETRY_UNKNOWN=0; IDS=()
for a in "$@"; do
    case "$a" in
        --apply) APPLY=1 ;;
        --retry-unknown) RETRY_UNKNOWN=1 ;;
        mp-*|*-*) IDS+=("$a") ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"
[ -d "$HERE/materials" ] || { echo "no materials/ under $HERE (run from the HT root)"; exit 1; }
VASPFLOW_NO_CHECK=1 source "$HERE/env.sh" >/dev/null 2>&1
STAMP=$(date +%Y%m%d_%H%M)
USER_NAME="${USER:-$(whoami)}"
SINCE="${SINCE:-$(date -d '7 days ago' +%Y-%m-%d 2>/dev/null || date -v-7d +%Y-%m-%d)}"
STEPS=(relax scf bands lobster)
declare -A DIR=([relax]=01_relax [scf]=02_scf [bands]=03_bands [lobster]=08_lobster)
QUEUE=$(squeue -h -u "$USER_NAME" -o '%i|%j|%t|%r' 2>/dev/null)

# ── helpers ──────────────────────────────────────────────────────────────────
step_done() {   # same tests as submit_all.sh
    case "$2" in
        relax)   [ ! -d "$1/01_relax" ] || grep -q "reached required accuracy" "$1/01_relax/OUTCAR" 2>/dev/null ;;
        scf)     grep -q "General timing and accounting" "$1/02_scf/OUTCAR" 2>/dev/null ;;
        bands)   grep -q "General timing and accounting" "$1/03_bands/OUTCAR" 2>/dev/null ;;
        lobster) grep -q "finished in" "$1/08_lobster/lobster.out" 2>/dev/null ;;
    esac
}
active() {      # a job of this material is running, or pending on something other than a dependency
    printf '%s\n' "$QUEUE" | awk -F'|' -v p="$1_" 'index($2,p)==1 && ($3!="PD" || $4 !~ /^Dependency/) {f=1} END{exit !f}'
}
set_incar_tag() { # file TAG value  (replace any existing line, then append)
    sed -i -E "/^[[:space:]]*$2[[:space:]]*=/d" "$1"
    printf '%s = %s\n' "$2" "$3" >> "$1"
}
incar_val() {   # file TAG -> value (comments dropped)
    sed 's/[!#].*//' "$1" 2>/dev/null | awk -F= -v t="$2" \
        'toupper($1) ~ "^[ \t]*" t "[ \t]*$" {gsub(/[ \t]/,"",$2); v=$2} END{print v}'
}
outcar_nbands() { grep -m1 -oE 'NBANDS= *[0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+'; }
to_sec() {      # [D-]HH:MM:SS -> seconds
    local t=$1 d=0
    [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r h m s <<< "$t"
    echo $(( d*86400 + 10#$h*3600 + 10#$m*60 + 10#${s:-0} ))
}
from_sec() { printf '%02d:%02d:%02d' $(($1/3600)) $(($1%3600/60)) $(($1%60)); }
step_time() {
    case "$1" in
        relax) echo "${TIME_RELAX:-08:00:00}" ;;  scf)     echo "${TIME_SCF:-04:00:00}" ;;
        bands) echo "${TIME_BANDS:-02:00:00}" ;;  lobster) echo "${TIME_LOBSTER:-04:00:00}" ;;
    esac
}
scf_unconverged() {   # electronic steps of the SCF reached NELM
    local o="$1/02_scf/OUTCAR" z="$1/02_scf/OSZICAR" nelm n
    [ -f "$o" ] && [ -f "$z" ] || return 1
    grep -q "EDIFF was not reached" "$o" && return 0
    nelm=$(grep -m1 -oE 'NELM *= *[0-9]+' "$o" | grep -oE '[0-9]+$')
    n=$(awk '/^(DAV|RMM|CG|DMP|SDA):/{n=$2} END{print n+0}' "$z")
    [ -n "$nelm" ] && [ "$n" -ge "$nelm" ]
}
bands_diverged() { awk '/^(DAV|RMM):/{e=$3+0; if (e>1e6||e<-1e6) f=1} END{exit !f}' "$1/03_bands/OSZICAR" 2>/dev/null; }
newest_contcar() {
    local d=$1/01_relax nat c
    nat=$(sed -n 7p "$d/POSCAR" | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')
    for c in $(ls -t "$d"/CONTCAR* 2>/dev/null); do
        [ "$(wc -l < "$c")" -ge $((8 + nat)) ] && { echo "$c"; return 0; }
    done
    return 1
}
stash() {       # $1 material, $2 step dir, files...
    local m=$1 st=$2; shift 2
    local dst="$m/stale_$STAMP/$st" f
    for f in "$@"; do
        [ -e "$m/$st/$f" ] || continue
        mkdir -p "$dst" && mv "$m/$st/$f" "$dst/"
    done
}
STEP_OUT="OUTCAR OSZICAR vasprun.xml EIGENVAL PROCAR lobster.out lobsterout ICOHPLIST.lobster ICOBILIST.lobster ICOOPLIST.lobster COHPCAR.lobster COBICAR.lobster COOPCAR.lobster lobster_summary.csv .vf_resume .vf_restarts STOPCAR"

# ── which materials ──────────────────────────────────────────────────────────
if [ ${#IDS[@]} -eq 0 ]; then
    mapfile -t IDS < <( {
        printf '%s\n' "$QUEUE" | awk -F'|' '$4=="DependencyNeverSatisfied"{print $2}'
        sacct -u "$USER_NAME" -S "$SINCE" -X -n -P -o JobName,State 2>/dev/null \
            | awk -F'|' '$2 ~ /^(FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)/{print $1}'
    } | sed -E 's/_(relax|scf|bands|lobster)$//' | grep '^mp-' | sort -u )
    # sacct/squeue list jobs of EVERY highthroughput directory; keep this directory's materials
    mine=(); for m in "${IDS[@]}"; do [ -d "$HERE/materials/$m" ] && mine+=("$m"); done
    IDS=("${mine[@]}")
fi
cd "$HERE/materials" || exit 1

declare -A NFIX=()
for m in "${IDS[@]}"; do
    [ -d "$m" ] || { echo "?? $m: no directory"; continue; }
    [ -f "$m/.done" ] && continue
    if active "$m"; then echo "-- $m: has a running/queued job, skipped"; continue; fi
    first=""
    for st in "${STEPS[@]}"; do step_done "$m" "$st" || { first=$st; break; }; done
    [ -z "$first" ] && continue

    # last job of that step
    job=$(sacct -u "$USER_NAME" -S "$SINCE" -X -n -P --name="${m}_$first" \
          -o JobID,State,Elapsed,Timelimit,NodeList 2>/dev/null | tail -n 1)
    IFS='|' read -r jid state elapsed tlim node <<< "$job"
    steps_state=$( [ -n "$jid" ] && sacct -j "$jid" -n -P -o State,MaxRSS 2>/dev/null | tr '\n' ' ')
    out="$m/${m}-${m}_$first-$jid.out"; err="$m/${m}-${m}_$first-$jid.err"

    # classify
    cls=unknown
    if [[ $state == OUT_OF_MEMORY* || $steps_state == *OUT_OF_MEMORY* ]] \
            || grep -qiE 'oom[-_ ]kill|out of memory|cannot allocate memory' "$err" "$out" 2>/dev/null; then cls=oom
    elif [ "$first" = scf ] && grep -qE 'number of bands has changed|different cutoff or change in lattice' "$out" 2>/dev/null; then cls=wavecar
    elif { [ "$first" = bands ] || [ "$first" = lobster ]; } && scf_unconverged "$m"; then cls=scf_unconv
    elif [ "$first" = bands ] && bands_diverged "$m"; then cls=bands_diverged
    elif [[ $state == TIMEOUT* ]]; then cls=timeout
    fi
    # bands that ran out of time while diverging is still a divergence
    [ "$cls" = timeout ] && [ "$first" = bands ] && bands_diverged "$m" && cls=bands_diverged

    start=$first; [ "$cls" = scf_unconv ] && start=scf
    # bands/LOBSTER restart from the SCF CHGCAR; if it is gone the SCF must run again
    { [ "$start" = bands ] || [ "$start" = lobster ]; } && [ ! -s "$m/02_scf/CHGCAR" ] && start=scf
    echo "== $m  step=$first  job=${jid:-none} $state $elapsed/$tlim on ${node:-?}  -> $cls"
    [ -n "$steps_state" ] && echo "   sacct steps (State MaxRSS): $steps_state"
    if [ "$cls" = unknown ] || [ "$cls" = oom ]; then
        grep -hiE 'error|bad news|oom|memory|killed|severe|abort|quota|segmentation|cannot|not found|exit code' \
             "$out" "$err" 2>/dev/null \
          | grep -vE 'IBZKPT|Often results|\(SIGTERM\)|kinetic energy error|^Image|Unknown|Stack trace' \
          | sort | uniq -c | sort -rn | head -n 8 | sed 's/^/   | /'
    fi
    if [ "$cls" = unknown ] && [ $RETRY_UNKNOWN = 0 ]; then
        echo "   not touched (unknown cause; inspect, or re-run with --retry-unknown)"
        NFIX[unknown]=$(( ${NFIX[unknown]:-0} + 1 )); continue
    fi
    NFIX[$cls]=$(( ${NFIX[$cls]:-0} + 1 ))

    # plan the fix
    mem=""; declare -A T=()
    for st in "${STEPS[@]}"; do T[$st]=$(step_time "$st"); done
    case "$cls" in
        timeout)
            T[$first]=$(from_sec $(( $(to_sec "${tlim:-$(step_time "$first")}") * 2 )))
            echo "   fix: walltime $first ${T[$first]}"
            if [ "$first" = relax ] && c=$(newest_contcar "$m"); then
                echo "   fix: relax continues from $(basename "$c")"
                [ $APPLY = 1 ] && { mkdir -p "$m/stale_$STAMP/01_relax"
                    cp -p "$m/01_relax/POSCAR" "$m/stale_$STAMP/01_relax/POSCAR.before"
                    cp "$c" "$m/01_relax/POSCAR"; }
            fi ;;
        oom)
            mem="--mem=0"
            nt=$(( $(awk -F= '/^#SBATCH --nodes=/{print $2}' "$m/job.sbatch") \
                 * $(awk -F= '/^#SBATCH --ntasks-per-node=/{print $2}' "$m/job.sbatch") ))
            k=8; while [ $k -gt 1 ] && [ $((nt % k)) -ne 0 ]; do k=$((k - 1)); done
            echo "   fix: --mem=0 (whole node), KPAR <= $k for $nt ranks"
            if [ $APPLY = 1 ]; then
                for st in "${STEPS[@]}"; do
                    f="$m/${DIR[$st]}/INCAR"; [ -f "$f" ] || continue
                    kp=$(incar_val "$f" KPAR)
                    [ -n "$kp" ] && [ "$kp" -gt "$k" ] && set_incar_tag "$f" KPAR "$k"
                done
            fi ;;
        wavecar)
            echo "   fix: SCF from the relaxed CHGCAR only (relax WAVECAR moved aside)"
            [ $APPLY = 1 ] && stash "$m" 01_relax WAVECAR && stash "$m" 02_scf WAVECAR ;;
        scf_unconv)
            echo "   fix: SCF rerun with ALGO = All (it reached NELM); bands + LOBSTER redone"
            [ $APPLY = 1 ] && set_incar_tag "$m/02_scf/INCAR" ALGO All ;;
        bands_diverged)
            nb_scf=$(outcar_nbands "$m/02_scf/OUTCAR"); nb_cur=$(incar_val "$m/03_bands/INCAR" NBANDS)
            nb=$(( (${nb_scf:-${nb_cur:-0}} * 13 + 9) / 10 ))
            [ -n "$nb_cur" ] && [ "$nb_cur" -gt "$nb" ] && nb=$nb_cur
            echo "   fix: 03_bands NBANDS ${nb_cur:-default} -> $nb (SCF had ${nb_scf:-?}), ALGO Normal (EDIFF unchanged)"
            [[ $state == TIMEOUT* ]] && { T[bands]=$(from_sec $(( $(to_sec "${tlim:-$(step_time bands)}") * 2 ))); echo "   fix: walltime bands ${T[bands]}"; }
            if [ $APPLY = 1 ]; then
                f="$m/03_bands/INCAR"
                [ "$nb" -gt 0 ] && set_incar_tag "$f" NBANDS "$nb"
                set_incar_tag "$f" ALGO Normal
            fi ;;
    esac

    # clean up, cancel the dead chain, resubmit
    chain=(); on=0
    for st in "${STEPS[@]}"; do
        [ "$st" = "$start" ] && on=1
        [ $on = 1 ] || continue
        # bands and LOBSTER are independent: redoing bands keeps a finished LOBSTER
        if [ "$st" != "$start" ] && [ "$start" = bands ] && step_done "$m" "$st"; then continue; fi
        chain+=("$st")
    done
    dead=$(printf '%s\n' "$QUEUE" | awk -F'|' -v p="${m}_" 'index($2,p)==1{print $1}' | xargs)
    echo "   resubmit: ${chain[*]}${dead:+   (cancel: $dead)}"
    [ $APPLY = 1 ] || continue
    [ -n "$dead" ] && scancel $dead
    for st in "${chain[@]}"; do
        if [ "$st" = relax ]; then stash "$m" 01_relax OUTCAR OSZICAR XDATCAR .vf_resume .vf_restarts STOPCAR
        else stash "$m" "${DIR[$st]}" $STEP_OUT; fi
    done
    rm -f "$m/.done" "$m/.vf_collected"
    prev=""; prev_st=""; line=""
    for st in "${chain[@]}"; do
        dep=""
        if [ -n "$prev" ]; then
            if [ "$prev_st" = bands ]; then dep="--dependency=afterany:$prev"   # LOBSTER needs only the SCF
            else dep="--dependency=afterok:$prev"; fi
        fi
        jid=$(cd "$m" && sbatch --parsable --job-name="${m}_$st" --time="${T[$st]}" $mem $dep job.sbatch "$st") \
            || { echo "   !!! sbatch failed for $st"; break; }
        prev=${jid%%;*}; prev_st=$st; line="$line $st=$prev(${T[$st]})"
    done
    echo "   submitted:$line"
done

echo ""
for c in "${!NFIX[@]}"; do printf '  %-15s %s\n' "$c" "${NFIX[$c]}"; done
if [ $APPLY = 1 ]; then echo "Done.  Monitor: squeue -u $USER_NAME"
else echo "Dry run - nothing changed. Re-run with --apply."; fi
