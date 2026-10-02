#!/bin/bash
# Make an existing highthroughput run (ht-semimetals layout) leave fewer large files:
#
#   1. 02_scf/INCAR: LWAVE = .TRUE. -> .FALSE. for every material whose SCF has not
#      finished yet (bands and LOBSTER read only the SCF CHGCAR; the SCF WAVECAR is
#      used by DFPT/ELF only, which this screen does not run).  INCARs are read when a
#      job starts, so this also applies to jobs that are already queued.
#      Skipped if the INCAR has LELF = .TRUE. (ELF needs the WAVECAR).
#   2. job.sbatch: the final clean-up (do_final, end of the lobster step) also deletes
#      02_scf/WAVECAR 02_scf/CHG and the empty 03_bands/WAVECAR 03_bands/CHG
#      08_lobster/CHG.  SLURM keeps its own copy of the script for jobs already
#      queued, so this applies to submissions made from now on; for those, run
#      cleanup_large.sh from time to time.
#
# Idempotent; dry run by default.
#   cd ~/VASP-Flow/HT_1-500 && bash ../ht_tools/patch_lean_outputs.sh [--apply]
set -u
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
HERE="$(cd "${HT_DIR:-$PWD}" && pwd)"
cd "$HERE/materials" || { echo "no materials/ under $HERE"; exit 1; }

OLD='rm -f 03_bands/CHGCAR 08_lobster/CHGCAR$'
NEW='rm -f 03_bands/CHGCAR 08_lobster/CHGCAR 02_scf/WAVECAR 02_scf/CHG 03_bands/WAVECAR 03_bands/CHG 08_lobster/CHG'
n_incar=0; n_job=0; n_elf=0
for d in mp-*/; do
    m=${d%/}
    inc="$m/02_scf/INCAR"
    if [ -f "$inc" ] && ! grep -q "General timing and accounting" "$m/02_scf/OUTCAR" 2>/dev/null; then
        if grep -qiE '^[[:space:]]*LELF[[:space:]]*=[[:space:]]*\.?T' "$inc"; then
            n_elf=$((n_elf + 1))
        elif grep -qiE '^[[:space:]]*LWAVE[[:space:]]*=[[:space:]]*\.?T' "$inc"; then
            n_incar=$((n_incar + 1))
            [ $APPLY = 1 ] && sed -i -E 's/^([[:space:]]*LWAVE[[:space:]]*=[[:space:]]*)\.?[Tt][A-Za-z]*\.?/\1.FALSE.   ! lean outputs: bands\/LOBSTER read only CHGCAR/' "$inc"
        fi
    fi
    job="$m/job.sbatch"
    if [ -f "$job" ] && grep -qE "$OLD" "$job"; then
        n_job=$((n_job + 1))
        [ $APPLY = 1 ] && sed -i -E "s|$OLD|$NEW|" "$job"
    fi
done
echo "02_scf/INCAR LWAVE -> .FALSE.: $n_incar materials (skipped, LELF on: $n_elf)"
echo "job.sbatch final clean-up extended: $n_job materials"
if [ $APPLY = 1 ]; then echo "Done."; else echo "Dry run - nothing changed. Re-run with --apply."; fi
