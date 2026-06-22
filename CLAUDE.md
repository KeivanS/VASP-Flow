# VASP Workflow GUI — Claude Code Guide

## Project Overview

Browser-based GUI for setting up, running, and analyzing DFT calculations with VASP. Flask REST API backend with a single-page JavaScript frontend.

## Architecture

```
vasp-gui.py                  # Flask server + embedded HTML/CSS/JS SPA (main entry point)
vasp-agent.py                # Workflow orchestration agent (CLI and library)
vasp-agent-slurm.py          # SLURM edition: SBATCH scripts + dependency-chained submission
ht-mp-scf.py                 # High-throughput driver: MP primitive cells → SCF + ELF batch
modules/
  instruction_parser.py      # Parses natural language instructions → settings dict
  vasp_input_generator.py    # Generates INCAR, KPOINTS, POTCAR, run.sh per step
```

**High-throughput (`ht-mp-scf.py`):** reads `highthrouput_list` (one mp-ID per line), downloads each **primitive** cell from Materials Project (`mp-api`/`pymatgen`, needs `MP_API_KEY`), stages `_ht_inputs/<id>/{POSCAR,instructions.txt}` (SCF task + `INCAR scf:` block with `LELF=.TRUE.` for ELF), and writes `runall.sh` that calls `vasp-agent.py` (local, sequential) or `vasp-agent-slurm.py` (SLURM). SLURM runs are chained across materials via `--dependency=afterok` by default (`--no-chain` to submit independently).

**Data flow:** Setup form → POST /api/generate → vasp-agent.py → InstructionParser → VASPInputGenerator → ProjectName/{00_convergence, 01_relax, 02_scf, 03_bands, 04_dos, 05_wannier, 06_dfpt, 07_phonons}

## Running the Project

```bash
make setup    # First time: create site.env from site.env.example
make run      # Start Flask server on http://localhost:5001
make snaps    # Alternate GUI on port 5050 (sc-snaps-gui.py)
```

**CLI agent:**
```bash
./vasp-agent.py -i instructions.txt -s POSCAR
```

## Configuration

- `site.env` — Platform-specific config (NOT committed, created from `site.env.example`)
  - `PYTHON`, `VASP_STD`, `VASP_NCL`, `VASP_GAM`, `MPI_LAUNCH`, `MPI_NP`, `WANNIER90_X`, `VASP_POTCAR_DIR`
- `<Project>/settings.json` — GUI state for "Edit & Regenerate"
- `<Project>/instructions.txt` — Natural language workflow definition

## Dependencies

```bash
pip install flask sumo
conda install -c conda-forge phonopy wannier90   # optional
```

External binaries required: VASP (std/ncl/gam), MPI, wannier90.x, phonopy

## Key Flask Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/generate` | POST | Generate workflow from Setup form |
| `/api/run` | POST | Execute a workflow step |
| `/api/stream/<job_key>` | GET (SSE) | Stream live job output |
| `/api/status/<slug>` | GET | Project status & step completion |
| `/api/plot/<slug>/<ptype>` | GET | Generate band/DOS plots via sumo |
| `/api/outcar/<slug>/<step>` | GET | Parse OUTCAR analysis |
| `/api/born_charges/<slug>` | GET | Extract Born effective charges |
| `/api/phonon_plot/<slug>/<ptype>` | GET | Phonon band/DOS plots |
| `/api/files/<slug>/<step>` | GET | List editable files in step |
| `/api/file/<slug>/<step>/<filename>` | GET/POST | Read/edit file contents |

## Workflow Steps

| Directory | Description |
|-----------|-------------|
| `00_convergence` | ENCUT/k-mesh convergence tests |
| `01_relax` | Geometry optimization (IBRION=2) |
| `02_scf` | Self-consistent field (IBRION=-1) |
| `03_bands` | Band structure (line-mode KPOINTS) |
| `04_dos` | Density of states (dense mesh, LORBIT=11) |
| `05_wannier` | Wannier90 NSCF interface |
| `06_dfpt` | Born charges + dielectric (IBRION=8) |
| `07_phonons` | Phonopy finite-displacement supercells |

## instruction_parser.py

Regex-based extraction from natural language instruction files. Supported parameters: functional (PBE/PBEsol/R2SCAN/HSE06/VV10/LDA), SOC + magnetization direction, GGA+U per element/orbital, task list, convergence test ranges, k-point path, Wannier90 projections and energy windows, DFPT flags, phonopy settings (supercell dim, mesh, displacement, NAC), DOS projections, explicit k-mesh override (KMESH), MPI settings (KPAR, NCORE, np).

**Constant pressure:** `PRESSURE = 10 GPa` (or `kbar`) triggers a constant-pressure relaxation — forces `ISIF=3`, `IBRION=2`, and emits `PSTRESS` (converted to kBar; GPa assumed if no unit). Parsed into the `pressure` dict.

**Raw INCAR passthrough:** an `INCAR: … END_INCAR` block in the instructions file injects literal INCAR tags into the generated INCAR(s). Per-step blocks use `INCAR <step>:` (relax/scf/bands/dos/wannier/dfpt/phonons); an unqualified block applies to all steps. Parsed into `incar_raw` ({'all'|step: ['TAG = val', …]}); merged by `VASPInputGenerator._apply_incar_overrides()`, which overwrites matching generated tags in place and appends the rest under a "User INCAR overrides" comment.

## vasp_input_generator.py

`VASPInputGenerator` class. Key methods: `generate_relax_input()`, `generate_scf_input()`, `generate_bands_input()`, `generate_dos_input()`, `generate_wannier_input()`, `generate_dfpt_input()`, `generate_phonons_input()`.

**Smart file copying:** `_write_copy_if_newer()` generates bash code that copies source→dest only if source is newer, preserving user edits.

**ENCUT defaults by functional:** PBE=400, PBEsol=450, R2SCAN=680, HSE06=400, VV10=450, LDA=350

## Testing

No formal test suite. Sample project directories: `GaAs_test/`, `AgCrPS3/`, `GaAs-phonons/`, `gaAs_wan/`, `GaAs_wannier2/`. Manual testing via browser GUI or CLI agent.
