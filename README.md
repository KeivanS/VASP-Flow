# VASP Workflow GUI

A browser-based interface for setting up and running VASP density-functional-theory (DFT) calculations. You describe what you want; the tool generates all input files, runs the calculations, and produces plots.

---

## What you need before you start

| Requirement | Notes |
|---|---|
| **VASP** | Licensed separately. You need compiled binaries: `vasp_std` (standard), `vasp_ncl` (spin-orbit), optionally `vasp_gam` (gamma-only). |
| **POTCAR library** | A folder of PAW pseudopotential files, one sub-folder per element (e.g. `potpaw_PBE.54/Ga/POTCAR`). Provided with your VASP license. |
| **Python ≥ 3.8** | Comes with most Linux/Mac systems. A conda environment is recommended. |
| **MPI** | `mpirun`, `mpiexec`, or `srun` (SLURM). |
| **sumo** | For band structure and DOS plots. |
| **LOBSTER** | Optional — only for the COHP/COBI/COOP bonding-analysis task. Free for academic use from [cohp.de](http://www.cohp.de). Set the binary path as `LOBSTER_X` in `site.env` (or `$LOBSTER_BIN` at runtime). |

> **Detailed references:** every Setup option and instructions-file keyword is documented in
> [`QUICK_REFERENCE_GUI.txt`](QUICK_REFERENCE_GUI.txt) (browser GUI) and
> [`QUICK_REFERENCE_AGENT.txt`](QUICK_REFERENCE_AGENT.txt) (command-line agent);
> the full manual is [`manual.pdf`](manual.pdf). This README is the quick tour.

---

## Installation (one time)

```bash
# 1. Get the code
git clone https://github.com/KeivanS/VASP-Flow.git
cd VASP-Flow

# 2. Install Python dependencies
pip install flask sumo

# 3. Optional: phonon and Wannier support
conda install -c conda-forge phonopy wannier90
```

---

## Starting the GUI

```bash
cd /path/to/VASP-Flow
python vasp-gui.py
```

Open your browser at **http://localhost:5001**

> **On a cluster?** Use SSH port forwarding so you can open the GUI on your laptop:
> ```bash
> ssh -L 5001:localhost:5001 username@cluster
> ```
> Then open http://localhost:5001 on your laptop.

---

## First-time configuration

The first time you open the GUI you will see a **System Setup** section at the top of the page. Fill it in once — it is saved automatically and will be loaded every time you start the GUI.

| Field | What to enter | Example |
|---|---|---|
| VASP standard binary | Full path to `vasp_std` | `/opt/vasp/6.4/bin/vasp_std` |
| VASP non-collinear binary | Full path to `vasp_ncl` — needed for spin-orbit calculations | `/opt/vasp/6.4/bin/vasp_ncl` |
| VASP gamma-only binary | Full path to `vasp_gam` — optional, faster for large supercells | `/opt/vasp/6.4/bin/vasp_gam` |
| POTCAR library directory | Folder containing one sub-folder per element | `/opt/vasp/potpaw_PBE.54` |
| MPI launch command | How MPI is invoked on your system | `mpirun -np` or `srun` |
| Default MPI cores | Number of CPU cores per job | `16` |

Click **💾 Save Settings**.

---

## Running a calculation — step by step

### 1. Prepare your structure

You need a **POSCAR** file (crystal structure in VASP format). Sources:
- [Materials Project](https://materialsproject.org) — search for your material, export as POSCAR
- VESTA — export from a CIF file
- ASE: `ase convert structure.cif POSCAR`

### 2. Set up the calculation

In the **Setup** tab:

**Project Name** — give it a descriptive name (e.g. `GaAs_PBEsol`). A folder with this name will be created in your working directory.

**POSCAR** — paste the file contents, or click **📁 Load file**.

**Functional** — choose your exchange-correlation approximation:
- `PBEsol` — best for structural properties of solids (recommended starting point)
- `PBE` — standard, widely used
- `R2SCAN` — meta-GGA, good balance of accuracy and cost
- `HSE06` — hybrid functional, more accurate band gaps but significantly slower

**Spin** — leave as *Non-magnetic* unless you know your material has magnetic order or you need spin-orbit coupling (SOC).

**Tasks** — check the calculations you want to run:

| Task | When to use |
|---|---|
| ☑ Structure relaxation | Almost always — optimises atomic positions first |
| ☑ SCF | Self-consistent charge density — needed before bands, DOS, or LOBSTER. The ELF checkbox inside it (on by default) also writes an ELFCAR for electron-localization plots |
| ☑ Band structure | Electronic band dispersion along a k-path |
| ☑ DOS | Density of states, optionally projected on orbitals |
| ☐ DFPT | Born effective charges and dielectric tensor |
| ☐ Phonons | Lattice dynamics via phonopy |
| ☐ Wannier90 | Maximally-localised Wannier functions |
| ☐ LOBSTER | COHP / COBI / COOP chemical-bonding analysis (`08_lobster`): a symmetry-off (ISYM=0, or −1 with SOC) NSCF from the SCF charge density, then the LOBSTER run. The bond range automatically covers 1st **and** 2nd neighbour shells. Needs the LOBSTER binary |

**POTCAR** — the directory is pre-filled from System Setup. If your element has multiple PAW variants (e.g. `Ga` vs `Ga_d`), the GUI will let you choose.

Click **▶ Generate Workflow**.

### 3. Run the calculations

Switch to the **Workflow** tab.

- Click **▶ Run** on each step in order — or **▶ Run All Steps** to chain them automatically.
- A live terminal shows VASP output as it runs.
- A green ✓ appears when a step finishes successfully.

**Normal order:** relaxation → SCF → bands → DOS

> If a step shows *WARNING: may not have converged*, open the OUTCAR via **✏ Edit files** and check whether `reached required accuracy` appears near the bottom. You may need to increase NSW (ionic steps) or NELM (electronic steps) in INCAR.

### 4. View results

Switch to the **Results** tab:
- Band structure and DOS plots appear automatically after the respective steps.
- **Fat bands** (orbital-weighted band structure) render next to the raw bands from `03_bands/PROCAR` (LORBIT=11 is always set): colored markers per element and orbital channel, marker size ∝ projection weight.
- Click **Run sumo-bandplot** or **Run sumo-dosplot** to regenerate a plot after editing settings.
- **ELF plots** (from the SCF's ELFCAR): 1D profiles along the non-equivalent 1st- and 2nd-shell bonds, plus a 2D ELF map on the plane through an atom and its 1st/2nd neighbours.
- **LOBSTER plots**: −COHP, COBI and COOP vs energy — total over all bonds plus per-pair 1st/2nd-shell overlays, annotated with the integral to E_F (ICOHP/ICOBI/ICOOP), the antibonding integral, the antibonding fraction f_AB, and the bonding→antibonding crossing energy.
- Born effective charges and phonon spectra appear if you ran DFPT / phonons.
- Bands, DOS and LOBSTER cards have an **Energy window** bar — enter min/max (eV) and click Zoom to re-render the plot (and its PDF download) on that window.

---

## Setting any VASP keyword explicitly

Nothing is hard-wired: **every VASP INCAR tag can be specified explicitly**, three ways (in increasing precedence):

**1. Instructions file** — add an `INCAR: … END_INCAR` block, global or per-step:

```
INCAR:                  # applies to every step
   LREAL   = .FALSE.
   ADDGRID = .TRUE.
END_INCAR

INCAR scf:              # this step only (relax/scf/bands/dos/wannier/dfpt/phonons/lobster)
   NBANDS = 64
END_INCAR
```

**2. Agent command line** — the repeatable `--incar` option (both `vasp-agent.py` and `vasp-agent-slurm.py`):

```bash
./vasp-agent.py -i instructions.txt -s POSCAR \
    --incar "ALGO=Fast; NELMIN=6" --incar "scf:NBANDS=64"
```

**3. GUI** — every generated INCAR is editable per step (see below).

Tags you set replace the generated defaults in place; unknown tags are appended under a `# User INCAR overrides` comment. The only exceptions are physics-critical guards that are re-imposed (ISYM=2 and KPAR=NCORE=1 in `06_dfpt`; symmetry-off ISYM in `08_lobster`). See `examples/` for complete working files.

---

## Editing input files

Every step has an **✏ Edit files** button. Click it to open INCAR, KPOINTS, or POSCAR in the browser, make changes, and save — then re-run that step. Any VASP keyword can be added or changed this way.

A `.bak` backup is created automatically the first time you edit a file.

---

## Resuming a previous project

Use the **Select project** dropdown at the top of the page:

- **→ Workflow** — jump straight to the Workflow tab for a project you already generated.
- **✏ Edit & Regenerate** — go back to Setup with all previous settings pre-filled, make changes, and regenerate input files.

---

## Convergence testing

Check **Convergence tests** in Setup to run a sweep of ENCUT and/or k-point mesh values before the main calculation. The test structure has atom 1 shifted by (0.01, 0.02, 0.03) Å so the force on it is non-zero, and the GUI plots total energy, pressure, **and forces** vs. each parameter (production steps keep the original positions).

The converged values are also **selected automatically** — the smallest ENCUT / k-mesh whose successive force change stays below 5 meV/Å and pressure change below 0.5 kbar — and pre-filled into the Phase-2 form; edit them to override before running the production steps.

---

## Running on a supercomputer (SLURM)

1. Start `python vasp-gui.py` on the login node (use SSH port forwarding to view in your browser).
2. In **System Setup**, set MPI launch command to `srun` and MPI cores to `nodes × cores_per_node`.
3. Select the **SLURM HPC** profile in the Execution Profile dropdown.
4. Edit `profiles/slurm.json` to match your cluster:

```json
{
  "slurm": {
    "partition": "compute",
    "nodes": 4,
    "ntasks_per_node": 32,
    "time": "24:00:00",
    "account": "myproject"
  },
  "modules": ["vasp/6.4.1", "intel-mpi/2021.8"],
  "mpi_cmd": "srun",
  "mpi_np": 128
}
```

Generated `run.sh` files will contain the correct `#SBATCH` headers and `module load` lines. You can submit them directly with `sbatch run.sh` from the terminal, or run them through the GUI if you have an interactive allocation.

---

## What gets created

After generating a workflow for a project named `GaAs_PBEsol`:

```
GaAs_PBEsol/
├── 00_convergence/  (if convergence tests enabled) encut/ kpoints/ choose_params.py
├── 01_relax/     INCAR  KPOINTS  POSCAR  POTCAR  run.sh
├── 02_scf/       INCAR  KPOINTS  POSCAR  POTCAR  run.sh
├── 03_bands/     INCAR  KPOINTS         POTCAR  run.sh
├── 04_dos/       INCAR  KPOINTS         POTCAR  run.sh
├── 05_wannier/   06_dfpt/   07_phonons/          (if those tasks are enabled)
├── 08_lobster/   INCAR  KPOINTS  lobsterin  POTCAR  run.sh   (LOBSTER task)
├── analysis/     plots collected by analyze.sh and the Results tab
├── POTCAR        (shared PAW file, symlinked into each step)
├── instructions.txt
└── project.json  (saved GUI state for “Edit & Regenerate”)
```

Each `run.sh` can also be submitted manually:

```bash
cd GaAs_PBEsol/02_scf
sbatch run.sh      # SLURM
bash run.sh        # workstation / interactive node
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| *POTCAR not found* | Wrong POTCAR directory | System Setup → POTCAR library directory |
| *vasp_std: command not found* | Wrong binary path | System Setup → VASP standard binary |
| *WARNING: may not have converged* | Ran out of ionic/electronic steps | In INCAR, increase NSW or NELM; re-run |
| Band structure looks wrong | SCF not run before bands | Make sure 02_scf completed before 03_bands |
| *sumo-bandplot: command not found* | sumo not installed | `pip install sumo` |
| GUI unreachable from laptop | No SSH tunnel | `ssh -L 5001:localhost:5001 user@cluster` |
| Project not visible in dropdown | Folder missing POSCAR or settings.json | Check that the folder is in the working directory |
| LOBSTER plots empty / "run 08_lobster first" | The lobster binary did not run — check `08_lobster/lobster.out` | Install LOBSTER and set `LOBSTER_X` in `site.env` (full path; `~` is expanded) |

---

## License

Copyright (C) 2024-2025 VASP Workflow GUI Contributors

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See the [LICENSE](LICENSE) file for the full text.

Note: VASP itself is proprietary software licensed separately by the VASP
Software GmbH; this project only generates inputs for and post-processes
outputs from VASP.
