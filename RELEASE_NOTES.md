# VASP-Flow release notes

## v2.5 — 2026-10-12 · LOBSTER k-mesh within memory

- k-point density: `fine` = 6000 kpra (was 5000), `coarse` = 1000; `ht-semimetals.py` uses `--kpra 8000` by default (the old default, `--kspacing 0.10`, gave very dense meshes, e.g. 24×24×24 = 110592 kpra for an 8-atom cell; it is now only used when given).
- The LOBSTER NSCF uses the SCF mesh (was 2× SCF in the agents) and is reduced, same ratios, above 10000 mesh points (`LOBSTER_MAX_KPTS`) or above the memory limit below.
- Memory: the LOBSTER NSCF runs without symmetry, so VASP keeps every k-point of its mesh (24×24×24 → 6916 k-points) and dense meshes stopped at set-up with `insufficient virtual memory`, whatever the memory per rank. The generators now estimate k-points × plane waves (plane waves from cell volume and ENCUT) and reduce the LOBSTER mesh, keeping its ratios, until it is below `LOBSTER_MAX_KPW` (default 6e6; `LOBSTER_MAX_KPW: <n>` in `instructions.txt`, `ht-semimetals.py --lobster-max-kpw`, 0 = no cap). The KPOINTS comment records the reduction; KPAR follows the reduced count.
- `vasp_diagnose.py`: an out-of-memory LOBSTER step also gets its k-mesh halved per axis.
- `ht_tools/test_step.sh`: one-node tests go to `standard` when the run uses the multi-node `parallel` partition; a failed `sbatch` removes the test folder.

## v2.4 — 2026-10-12 · Checking and repairing runs

Includes the v2.3 changes of 2026-10-05, which were not released separately.

- **`vasp_diagnose.py`** (new): checks every run of a `*_list.txt` (material names of any kind — mp-IDs, formulas, folder names — or folders), steps `00_convergence` … `08_lobster`, for high-throughput, SLURM and local layouts. It names the cause of each failure (timeout, out of memory, SCF/NSCF not converged or diverged, EDDDAV, ZBRENT, symmetry, WAVECAR/CHGCAR, LOBSTER, quota, …), writes one repair + resubmission script per material and `submit_all_fixes.sh` (queue-throttled, resumable). Read-only until you run the scripts; a repair already tried is held back for review.
- **Fixed-density steps use `ALGO = Fast`**: bands, DOS and the LOBSTER NSCF (Davidson start, then RMM-DIIS). Pure Davidson diverged to |E| ~ 1e10 eV for Ba2YReO6/Sr2YReO6; `ALGO = All` is not allowed with ICHARG = 11. Relax and SCF keep `ALGO = Normal`.
- **`ht_tools/`** new tools:
  - `test_step.sh`: test the SCF/bands/LOBSTER NSCF of one material with changed INCAR tags in `tests/` (`--report` to compare).
  - `show_problems.sh`, `skip_fixes.sh`: what the diagnosis found and what each fix changes; leave materials out of the resubmission.
  - `show_material_status.sh`: per step, how long VASP ran, key INCAR tags, every ionic step with its last 5 electronic steps, relax forces/pressure, OUTCAR tail when VASP did not finish.
  - `show_job_history.sh`, `show_job_log.sh`: SLURM states/times of a material's jobs, end of the newest logs.
- `vasp-agent-slurm.py`: convergence-test scripts locate their folder by absolute path (not `$0`, which is SLURM's spool copy).
- From v2.3: `ht-semimetals.py` uses the Materials Project POTCAR choices by default (`--potcar-set mp`); `NBANDS` = 1.2 × VASP default in relax/SCF/bands, bands never below the SCF; `#SBATCH --mem=0`; relaxation with IBRION = 1; `ht_tools/` maintenance scripts (`fix_failed.sh`, `test_relax.sh`, `collect_lobster.sh`, `reset_stale.sh`, `cleanup_large.sh`, `audit_run.sh`, `resume_in_scratch.sh`).
- **Manual**: one file (`manual.tex`/`manual.pdf`) organised by task — getting started, single calculations, high-throughput screens, checking and repairing runs — with how things work in the appendices; the `vasp_diagnose.py` manual is merged in. `QUICK_REFERENCE_AGENT.txt` gains a "Checking & repairing runs" box.

## v2.2 — 2026-09-30 · Relax-first high throughput; results viewer

- `ht-mp-scf.py`: each material now starts with `01_relax` (IBRION = 2, ISIF = 3: cell shape, volume and ions), since MP structures come from a different functional/setup. Same k-density as the SCF by default, so the SCF reuses CONTCAR, CHGCAR and WAVECAR. Options `--no-relax`, `--relax-kpra`.
- `ht-mp-scf.py`, SLURM chained mode: every step of a material (relax → SCF → LOBSTER) is submitted, not only the SCF.
- `ht-semimetals.py`: DFPT step removed (ill-defined for zero-gap systems, Davidson convergence problems). Each job runs `analyze.sh` at the end (`ANALYSIS_PYTHON` in `env.sh`); new `analyze_all.sh`; `collect_results.sh` keeps the per-step folder layout with trimmed OUTCARs.
- GUI: **Open results folder** — view a finished job, a whole highthroughput directory, or an unpacked `results/` tarball; plots missing from `analysis/` are drawn from the raw outputs. SCF/NSCF steps are shown as done when VASP finished.
- GGA+U flag `GGA_U: ON|OFF` in `instructions.txt` (GUI: auto/on/off; HT drivers: `--gga_u`). Without it the U_eff of `hubbard_u_defaults.csv` (now the only source of U values; LDAUU = U_eff, LDAUJ = 0) is applied only when the POSCAR contains a chalcogen or halogen (O, S, Se, Te, F, Cl, Br, I); same U block in every step.
- `ht-semimetals.py`: relaxation first (`01_relax`, ISIF = 3), one SLURM job per step chained with afterok.
- SLURM walltime per step (relax 8 h, SCF 4 h, bands/DOS 2 h, LOBSTER 4 h; `<STEP>_WALLTIME` / `WALLTIME` / `TIME_*` in env.sh override). A step that reaches its limit is stopped cleanly 15 min before the end and requeued (same job id): relaxations continue from CONTCAR or the last XDATCAR frame, LOBSTER reruns only the binary once its NSCF is done.
- HT drivers never re-download a structure: `_ht_inputs/<id>/POSCAR` is reused, and `--poscar-dir <folder>` takes POSCARs from another (e.g. older highthroughput) folder; no MP key needed then.
- `ht-semimetals.py --jobs N`: materials prepared in parallel (default min(8, CPUs)); walltimes only in `env.sh` (not in each `instructions.txt`).
- NumPy 2 compatibility: `np.trapz` → `np.trapezoid` in `cohp_plot.py` and `lobster_postprocess.py`.

## v2.1 — 2026-09-30 · Simpler k-mesh and parallel rules; two-step ELF

- k-mesh from one formula: N_i = nearest even of s·|b_i*| (hexagonal in-plane: multiple of 6), s set by the kpra target.
- KPAR = largest divisor of the ranks ≤ irreducible k-points; NCORE = largest divisor of ranks-per-k-group ≤ its square root. Wannier90 and phonons default to KPAR = 1; LOBSTER uses its own (ISYM = 0) k-count.
- `ELF: separate`: SCF at full KPAR, then a short KPAR = 1 restart writes ELFCAR (default in `ht-mp-scf.py`).
- `ht-mp-scf.py`: `--kpra` (default coarse), `--kpar`, `--ncore`. `ht-semimetals.py`: `--kpar`, `--ncore`, POTCARs built automatically.

## v2.0 — 2026-09-28 · Input-consistency release

- `MAGMOM:` accepts negative, per-element and per-atom values; switches ISPIN on; SOC moments along x/y/z.
- k-point density (coarse/fine/integer kpra) reflected in every KPOINTS file, uniform spacing, hexagonal multiples of 6, `RELAX_KMESH_DENSITY`.
- Relaxation starts from scratch; the SCF reads its CHGCAR (and WAVECAR when the mesh matches).
- Explicit `INCAR <step>:` and `KPOINTS <step>:` blocks override everything (GUI: Setup → Advanced).
- `check_agent_consistency.py` compares GUI, workstation and SLURM outputs. New `ht-semimetals.py`.
