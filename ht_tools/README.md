# ht_tools — maintenance scripts for a highthroughput run (ht-semimetals layout)

Run them from the HT root (the folder with `materials/`, `submit_all.sh`, `env.sh`):

```bash
cd ~/VASP-Flow/HT_1-500
bash ../ht_tools/reset_stale.sh            # dry run
bash ../ht_tools/reset_stale.sh --apply    # then ./submit_all.sh
bash ../ht_tools/collect_lobster.sh        # tarball of new/changed finished materials
bash ../ht_tools/cleanup_large.sh          # dry run; --apply frees WAVECAR/CHGCAR no step still needs
bash ../ht_tools/fix_failed.sh             # dry run; --apply repairs + resubmits broken chains
bash ../ht_tools/audit_run.sh [BACKUP ...] # read-only: run / converged / extracted / backed up, per material
bash ../ht_tools/resume_in_scratch.sh      # dry run; --apply copies the run to /scratch and resubmits unfinished materials
```

**Quick look-up tools** (read-only except `skip_fixes.sh`; run from anywhere — an ID is searched in `/scratch/$USER/VASP-Flow/*/materials/`, a path is used as given):

```bash
bash ht_tools/show_problems.sh diag_round3            # broken materials, what each fix changes, which fixes will run
bash ht_tools/skip_fixes.sh diag_round3 HT_1-500__mp-6430 HT_1-500__mp-7889   # don't submit these (--undo puts them back)
bash ht_tools/show_material_status.sh mp-20352 [lobster ...]   # INCAR key tags, every ionic step with its last 5 electronic steps, relax forces/pressure, OUTCAR tail if VASP did not finish
bash ht_tools/show_job_history.sh mp-20352 [mp-6430 ...] [-S 2026-10-01]   # sacct (state, elapsed, limit) + queue
bash ht_tools/show_job_log.sh mp-20352 lobster [LINES] [JOBID]   # end of the newest .out/.err of that step
```

| Script | Purpose |
|---|---|
| `reset_stale.sh` | Before resubmitting: (A) relax never reached "required accuracy" but SCF/bands/LOBSTER already ran on that geometry — restart the relax from the newest complete CONTCAR and redo the downstream steps; (B) LOBSTER curve files truncated — redo `08_lobster`; (C) blank `03_bands/EIGENVAL` — redo bands. Moves invalid outputs to `<material>/stale_<date>/` (nothing deleted); skips materials with queued jobs and prints the `scancel` line for stale relax-only jobs. |
| `collect_lobster.sh` | Packs only data (structures, `02_scf/INCAR`, raw LOBSTER `ICO*LIST`/`*CAR` files, `EIGENVAL`, trimmed OUTCARs) plus `_status.tsv`. Incremental: a material is packed once its LOBSTER step has finished and again only if it changed (marker `<material>/.vf_collected`); `ALL=1` repacks everything. |
| `cleanup_large.sh` | Frees disk space: deletes WAVECAR/CHG(CAR) files that no remaining step of a material's chain reads (relax files once SCF finished, SCF WAVECAR/CHG once SCF finished, bands/LOBSTER files once those finished). `02_scf/CHGCAR` is kept (needed to redo bands/LOBSTER) unless `--deep` and both are finished. Materials with queued jobs are skipped; dry run by default. |
| `patch_lean_outputs.sh` | For a run generated before the lean-output fix: sets `LWAVE = .FALSE.` in `02_scf/INCAR` of materials whose SCF has not run (applies to queued jobs too) and extends the final clean-up in each `job.sbatch` to delete `02_scf/WAVECAR`, `02_scf/CHG` and the empty bands/LOBSTER placeholders (applies to new submissions). Idempotent; dry run by default. |
| `test_relax.sh` | Test relaxation of one material with modified INCAR tags in `tests/<id>_<tag>/` (outside `materials/`), from the original input cell; `--report` prints energy/volume/pressure/max force per step and ZBRENT/convergence lines. |
| `test_step.sh` | Same for the SCF, bands or LOBSTER-NSCF step: `test_step.sh <id> scf|bands|lobster <tag> "TAG=VAL; ..."` runs the step's inputs (plus `02_scf/CHGCAR` for an NSCF, no WAVECAR) in `tests/<id>_<step>_<tag>/` (40 cores, 1 h; `T_NTASKS`/`T_TIME`/`T_PARTITION`); `--report tests/...` lists per test: finished / time limit, EDIFF reached, number of electronic steps, first/last/max energy (max \|E\| > 1e5 eV = diverged). |
| `redo_lobster.sh ID ...` | Redo the LOBSTER step on one node (`standard`, 40 cores; `R_PARTITION`/`R_NTASKS`/`R_TIME`): cancels the material's jobs, sets `08_lobster/KPOINTS` to the SCF mesh reduced to ≤ 10000 points and k-points × plane waves ≤ 6e6 (the v2.5 rule; `R_MAX_KPTS`, `R_MAX_KPW`), makes KPAR divide the cores, moves old LOBSTER outputs to `stale_<date>/08_lobster/`, submits `job.sbatch lobster`. Needs a finished SCF with CHGCAR. |
| `patch_relax_ibrion.sh` | Switches unconverged relaxations of an existing run from IBRION = 2 to 1 (applies to queued jobs too) and resets ZBRENT-aborted relaxes to the input cell (aborted files kept in `01_relax/zbrent_<date>/`). Skips converged and running relaxes; dry run by default. |
| `fix_failed.sh` | Materials whose chain broke (a step FAILED/TIMEOUT/OUT_OF_MEMORY, the rest pending as `DependencyNeverSatisfied`; auto-detected, or pass ids): classifies the failure from sacct + logs + OUTCAR/OSZICAR — timeout (walltime ×2, relax continues from CONTCAR), oom (`--mem=0`, KPAR ≤ 8), wavecar (SCF crashed reading the relax WAVECAR → CHGCAR-only start), scf_unconv (SCF hit NELM → rerun with ALGO = All), bands_diverged (NBANDS ≥ 1.3× SCF, ALGO Normal; EDIFF is never changed) — cancels the dead jobs, stashes partial outputs in `stale_<date>/`, resubmits from that step with LOBSTER `afterany` on bands. Unknown causes are only printed (`--retry-unknown` resubmits them unchanged). Dry run by default. |
| `audit_run.sh` | Read-only check before adding a run to the database. Per material of `material_list.txt`: missing, queued, relax not converged, SCF not finished or hit NELM, LOBSTER not finished / empty ICOHPLIST / charge spilling > `SPILL_MAX` % (default 5); bands status in its own column; whether it is in a `$HOME/lobster_collect_*.tar.gz` archive (`ARCHIVES=` to change); and, for each BACKUP argument (a copied run directory or a .tar.gz), whether its ICOHPLIST is there with the same size. Writes `audit_<date>.tsv` and prints the counts and problem ids. |
| `resume_in_scratch.sh` | Moves a run out of the home quota: `--apply` rsyncs the HT folder to `/scratch/$USER/VASP-Flow/<HT>` (`--dest`), leaves `MOVED_TO_SCRATCH.txt` in the home copy, and resubmits every unfinished material from its first unfinished step — relax from CONTCAR (or the last XDATCAR frame), SCF from the relaxed CHGCAR when it is complete (else `ICHARG = 2`; `ALGO = All` after NELM), LOBSTER binary only when the NSCF and its WAVECAR are there. Old outputs of redone steps go to `stale_<date>/`. Chain relax → scf → bands -(afterany)→ lobster; `--time STEP=HH:MM:SS`, `--skip ID,…`, `MAX_QUEUED`. Run under `nohup`. Dry run by default. |
