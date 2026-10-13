#!/bin/bash
# Set your own parameters for one step of one or more materials; the next repair round
# (next_round.sh) redoes that step - and the steps after it - with them.  Run from the
# HT folder (or give material folders as paths):
#
#   bash ~/VASP-Flow/ht_tools/set_params.sh mp-20352 lobster "KPAR=4; KMESH=10 10 14; TIME=06:00:00"
#   bash ~/VASP-Flow/ht_tools/set_params.sh mp-542623,mp-972937 scf "NODES=1"
#   bash ~/VASP-Flow/ht_tools/set_params.sh mp-7889 scf "ALGO=All; NELM=200; SYMPREC="   # SYMPREC= removes it
#   bash ~/VASP-Flow/ht_tools/set_params.sh mp-7889 --show
#   bash ~/VASP-Flow/ht_tools/set_params.sh mp-7889 --clear [scf]
#
# STEP: relax scf bands dos lobster, or all (every step's INCAR; redoes the whole chain).
# TAG=VALUE: any INCAR tag.  Special keys: KMESH=n1 n2 n3 (Gamma mesh of that step),
# TIME=HH:MM:SS (walltime), NODES=n (nodes; KPAR is made a divisor of the cores).
# Stored in materials/<id>/user_params.txt; your values win over the automatic fixes
# and are kept for later rounds.  Nothing is submitted here.
set -u
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"
usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }
[ $# -ge 2 ] || usage
ids=$1; step=$2; mods=${3:-}
for id in ${ids//,/ }; do
    if [ -d "$id" ]; then m=$(cd "$id" && pwd); else m=$HERE/materials/$id; fi
    [ -d "$m" ] || { echo "!! $id: no folder $m"; continue; }
    f=$m/user_params.txt
    case $step in
        --show) echo "== $id"; [ -s "$f" ] && sed 's/^/   /' "$f" || echo "   (no settings)"; continue ;;
        --clear)
            if [ -n "$mods" ]; then [ -f "$f" ] && awk -v s="$mods" '$1 != s' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
            else rm -f "$f"; fi
            echo "== $id: settings ${mods:-of all steps} removed"; continue ;;
        relax) sd=01_relax ;; scf) sd=02_scf ;; bands) sd=03_bands ;; dos) sd=04_dos ;; lobster) sd=08_lobster ;;
        all) sd= ;;
        *) echo "STEP must be relax, scf, bands, dos, lobster or all"; exit 1 ;;
    esac
    [ -z "$sd" ] || [ -d "$m/$sd" ] || { echo "!! $id: no $sd folder"; continue; }
    [ -n "$mods" ] || usage
    now=$(date '+%Y-%m-%d %H:%M:%S')
    echo "== $id $step"
    IFS=';' read -ra kv <<< "$mods"
    for x in "${kv[@]}"; do
        [ -n "${x// /}" ] || continue
        case $x in *=*) ;; *) echo "   !! '$x' is not TAG=VALUE - skipped"; continue ;; esac
        t=$(echo "${x%%=*}" | tr -d ' ' | tr a-z A-Z); v=$(echo "${x#*=}" | sed 's/^ *//; s/ *$//')
        case $t in
            KMESH) echo "$v" | grep -qE '^[0-9]+ +[0-9]+ +[0-9]+$' || { echo "   !! KMESH needs three numbers"; continue; } ;;
            TIME)  echo "$v" | grep -qE '^([0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}$' || { echo "   !! TIME needs HH:MM:SS"; continue; } ;;
            NODES) echo "$v" | grep -qE '^[0-9]+$' || { echo "   !! NODES needs a number"; continue; } ;;
        esac
        printf '%s %s = %s   # set %s\n' "$step" "$t" "$v" "$now" >> "$f"
        [ -n "$v" ] && echo "   $t = $v" || echo "   $t removed"
    done
done
echo "Applied by the next round:  bash ~/VASP-Flow/ht_tools/next_round.sh"
