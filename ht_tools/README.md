# ht_tools — check, repair and collect a highthroughput run (ht_runs.py layout)

Run them from the HT folder (the one with `materials/`, `material_list.txt`, `submit_all.sh`, `env.sh`):

```bash
cd /scratch/$USER/VASP-Flow/HT_ins_1
bash ~/VASP-Flow/ht_tools/next_round.sh        # diagnose all materials, show what failed and why, submit the repairs (asks y/N)
bash ~/VASP-Flow/ht_tools/set_params.sh mp-20352 lobster "KPAR=4; KMESH=10 10 14"   # your own settings; next round applies them
bash ~/VASP-Flow/ht_tools/audit_run.sh         # read-only: run / converged / extracted / backed up, per material
bash ~/VASP-Flow/ht_tools/collect_lobster.sh   # tarball of new/changed finished materials
```

**The repair cycle:** `./submit_all.sh` once, then `next_round.sh` (answer `y`) each time the queue is empty, until it
reports nothing to submit; then `audit_run.sh` and `collect_lobster.sh`. Each round writes `diagnosis_round<N>/`:
`report.txt` (per material: problem, cause in words, evidence, certainty of the cause and of the fix, changes, backup),
`diagnosis.tsv` (one line per material), `failed_list.txt`, `review_list.txt`, `fix/<material>.sh`, `submit_all_fixes.sh`.
Repairs are made in place; the failed run goes to `materials/<id>/round_<i>_<date>/` (with `WHY.txt`), and
`materials/<id>/repair_history.tsv` records every repair. A material is held for review after the same cause twice, or 5 repairs.

**Quick look-up tools** (read-only except `skip_fixes.sh`; an ID is searched in `/scratch/$USER/VASP-Flow/*/materials/`, a path is used as given):

```bash
bash ~/VASP-Flow/ht_tools/show_problems.sh diagnosis_round2      # broken materials, certainty, what each fix changes, which fixes will run
bash ~/VASP-Flow/ht_tools/skip_fixes.sh diagnosis_round2 HT_1-500__mp-6430   # don't submit this one (--undo puts it back)
bash ~/VASP-Flow/ht_tools/show_material_status.sh mp-20352 [lobster ...]   # INCAR key tags, every ionic step with its last 5 electronic steps, relax forces/pressure, OUTCAR tail if VASP did not finish
bash ~/VASP-Flow/ht_tools/show_job_history.sh mp-20352 [mp-6430 ...] [-S 2026-10-01]   # sacct (state, elapsed, limit) + queue
bash ~/VASP-Flow/ht_tools/show_job_log.sh mp-20352 lobster [LINES] [JOBID]   # end of the newest .out/.err of that step
```

| Script | Purpose |
|---|---|
| `next_round.sh [-y\|-n] [vasp_diagnose options]` | One repair round: runs `vasp_diagnose.py material_list.txt --out diagnosis_round<N>`, prints `show_problems.sh`, asks before starting `submit_all_fixes.sh` under nohup (`-y` without asking, `-n` diagnosis only). Warns when jobs of the run are still queued (those materials are skipped). |
| `set_params.sh ID[,ID] STEP "TAG=VAL; ..."` | Your own settings for a step (relax/scf/bands/dos/lobster/all): INCAR tags (`TAG=` removes one), `KMESH=n1 n2 n3`, `TIME=HH:MM:SS`, `NODES=n`. Stored in `materials/<id>/user_params.txt`; the next round redoes that step (and the later ones) with them, after its own fixes, so yours win. `--show`, `--clear [STEP]`. |
| `show_problems.sh DIAG_DIR [--all]` | The broken materials of a round with cause/fix certainty and repair round, which fix scripts will run, and what each changes. |
| `skip_fixes.sh DIAG_DIR PATTERN...` | Comment fix scripts out of `submit_all_fixes.sh` (`--undo` puts them back). |
| `show_material_status.sh`, `show_job_history.sh`, `show_job_log.sh` | One material: how far each step got; its SLURM jobs; the end of a job log. |
| `test_step.sh` | Test the SCF, bands or LOBSTER-NSCF step of one material with modified INCAR tags in `tests/<id>_<step>_<tag>/` (outside `materials/`; 1 node, 40 cores, 1 h; `T_NTASKS`/`T_TIME`/`T_PARTITION`); `--report tests/...` lists finished / time limit, EDIFF reached, electronic steps, first/last/max energy (max \|E\| > 1e5 eV = diverged). |
| `test_relax.sh` | The same for the relaxation, from the original input cell; `--report` prints energy/volume/pressure/max force per step and ZBRENT/convergence lines. |
| `audit_run.sh [BACKUP ...]` | Read-only check before adding a run to the database. Per material of `material_list.txt`: missing, queued, relax not converged, SCF not finished or hit NELM, LOBSTER not finished / empty ICOHPLIST / charge spilling > `SPILL_MAX` % (default 5); bands status in its own column; whether it is in a `$HOME/lobster_collect_*.tar.gz` archive (`ARCHIVES=` to change); and, for each BACKUP argument, whether its ICOHPLIST is there with the same size. Writes `audit_<date>.tsv`. |
| `collect_lobster.sh` | Packs only data (structures, `02_scf/INCAR`, raw LOBSTER `ICO*LIST`/`*CAR` files, `EIGENVAL`, trimmed OUTCARs) plus `_status.tsv`. Incremental: a material is packed once its LOBSTER step has finished and again only if it changed (marker `<material>/.vf_collected`); `ALL=1` repacks everything. |
| `cleanup_large.sh` | Frees disk space: deletes WAVECAR/CHG(CAR) files that no remaining step of a material's chain reads. `02_scf/CHGCAR` is kept (needed to redo bands/LOBSTER) unless `--deep` and both are finished. Materials with queued jobs are skipped; dry run by default. |
