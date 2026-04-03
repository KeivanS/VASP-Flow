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
| ☑ SCF | Self-consistent charge density — needed before bands or DOS |
| ☑ Band structure | Electronic band dispersion along a k-path |
| ☑ DOS | Density of states, optionally projected on orbitals |
| ☐ DFPT | Born effective charges and dielectric tensor |
| ☐ Phonons | Lattice dynamics via phonopy |
| ☐ Wannier90 | Maximally-localised Wannier functions |

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
- Click **Run sumo-bandplot** or **Run sumo-dosplot** to regenerate a plot after editing settings.
- Born effective charges and phonon spectra appear if you ran DFPT / phonons.

---

## Editing input files

Every step has an **✏ Edit files** button. Click it to open INCAR, KPOINTS, or POSCAR in the browser, make changes, and save — then re-run that step.

A `.bak` backup is created automatically the first time you edit a file.

---

## Resuming a previous project

Use the **Select project** dropdown at the top of the page:

- **→ Workflow** — jump straight to the Workflow tab for a project you already generated.
- **✏ Edit & Regenerate** — go back to Setup with all previous settings pre-filled, make changes, and regenerate input files.

---

## Convergence testing

Check **Convergence tests** in Setup to run a sweep of ENCUT and/or k-point mesh values before the main calculation. The GUI will plot the total energy vs. each parameter so you can choose well-converged values before committing to the full workflow.

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
├── 01_relax/     INCAR  KPOINTS  POSCAR  POTCAR  run.sh
├── 02_scf/       INCAR  KPOINTS  POSCAR  POTCAR  run.sh
├── 03_bands/     INCAR  KPOINTS         POTCAR  run.sh
├── 04_dos/       INCAR  KPOINTS         POTCAR  run.sh
├── POTCAR        (shared PAW file, symlinked into each step)
├── instructions.txt
└── settings.json
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
