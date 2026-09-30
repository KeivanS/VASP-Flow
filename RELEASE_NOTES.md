# VASP-Flow release notes

## v2.2 — 2026-09-30 · Relax-first high throughput; results viewer

- `ht-mp-scf.py`: each material now starts with `01_relax` (IBRION = 2, ISIF = 3: cell shape, volume and ions), since MP structures come from a different functional/setup. Same k-density as the SCF by default, so the SCF reuses CONTCAR, CHGCAR and WAVECAR. Options `--no-relax`, `--relax-kpra`.
- `ht-mp-scf.py`, SLURM chained mode: every step of a material (relax → SCF → LOBSTER) is submitted, not only the SCF.
- `ht-semimetals.py`: DFPT step removed (ill-defined for zero-gap systems, Davidson convergence problems). Each job runs `analyze.sh` at the end (`ANALYSIS_PYTHON` in `env.sh`); new `analyze_all.sh`; `collect_results.sh` keeps the per-step folder layout with trimmed OUTCARs.
- GUI: **Open results folder** — view a finished job, a whole highthroughput directory, or an unpacked `results/` tarball; plots missing from `analysis/` are drawn from the raw outputs. SCF/NSCF steps are shown as done when VASP finished.
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
