# ht_tools — maintenance scripts for a highthroughput run (ht-semimetals layout)

Run them from the HT root (the folder with `materials/`, `submit_all.sh`, `env.sh`):

```bash
cd ~/VASP-Flow/HT_1-500
bash ../ht_tools/reset_stale.sh            # dry run
bash ../ht_tools/reset_stale.sh --apply    # then ./submit_all.sh
bash ../ht_tools/collect_lobster.sh        # tarball of new/changed finished materials
bash ../ht_tools/cleanup_large.sh          # dry run; --apply frees WAVECAR/CHGCAR no step still needs
```

| Script | Purpose |
|---|---|
| `reset_stale.sh` | Before resubmitting: (A) relax never reached "required accuracy" but SCF/bands/LOBSTER already ran on that geometry — restart the relax from the newest complete CONTCAR and redo the downstream steps; (B) LOBSTER curve files truncated — redo `08_lobster`; (C) blank `03_bands/EIGENVAL` — redo bands. Moves invalid outputs to `<material>/stale_<date>/` (nothing deleted); skips materials with queued jobs and prints the `scancel` line for stale relax-only jobs. |
| `collect_lobster.sh` | Packs only data (structures, `02_scf/INCAR`, raw LOBSTER `ICO*LIST`/`*CAR` files, `EIGENVAL`, trimmed OUTCARs) plus `_status.tsv`. Incremental: a material is packed once its LOBSTER step has finished and again only if it changed (marker `<material>/.vf_collected`); `ALL=1` repacks everything. |
| `cleanup_large.sh` | Frees disk space: deletes WAVECAR/CHG(CAR) files that no remaining step of a material's chain reads (relax files once SCF finished, SCF WAVECAR/CHG once SCF finished, bands/LOBSTER files once those finished). `02_scf/CHGCAR` is kept (needed to redo bands/LOBSTER) unless `--deep` and both are finished. Materials with queued jobs are skipped; dry run by default. |
| `patch_lean_outputs.sh` | For a run generated before the lean-output fix: sets `LWAVE = .FALSE.` in `02_scf/INCAR` of materials whose SCF has not run (applies to queued jobs too) and extends the final clean-up in each `job.sbatch` to delete `02_scf/WAVECAR`, `02_scf/CHG` and the empty bands/LOBSTER placeholders (applies to new submissions). Idempotent; dry run by default. |
| `test_relax.sh` | Test relaxation of one material with modified INCAR tags in `tests/<id>_<tag>/` (outside `materials/`), from the original input cell; `--report` prints energy/volume/pressure/max force per step and ZBRENT/convergence lines. |
| `patch_relax_ibrion.sh` | Switches unconverged relaxations of an existing run from IBRION = 2 to 1 (applies to queued jobs too) and resets ZBRENT-aborted relaxes to the input cell (aborted files kept in `01_relax/zbrent_<date>/`). Skips converged and running relaxes; dry run by default. |
