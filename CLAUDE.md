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

**High-throughput (`ht-mp-scf.py`):** reads `highthrouput_list` (one mp-ID per line), downloads each **primitive** cell from Materials Project (`mp-api`/`pymatgen`, needs `MP_API_KEY`), stages `_ht_inputs/<id>/{POSCAR,instructions.txt}` (relax task with an `INCAR relax:` block IBRION=1/ISIF=3 (quasi-Newton; IBRION=2 hit ZBRENT failures on converged cells) and `RELAX_KMESH_DENSITY` = SCF density unless `--no-relax`; SCF task with `ELF: separate`; optional LOBSTER), and writes `runall.sh` that calls `vasp-agent.py` (local, sequential) or `vasp-agent-slurm.py` (SLURM). `ht-semimetals.py` also starts with `01_relax` (ISIF=3, restarted once from CONTCAR if unconverged; `--no-relax`). SLURM runs chain every step of every material (relax → scf → lobster, then the next material) via `--dependency=afterok` by default (`--no-chain` to submit independently).

**Data flow:** Setup form → POST /api/generate → vasp-agent.py → InstructionParser → VASPInputGenerator → ProjectName/{00_convergence, 01_relax, 02_scf, 03_bands, 04_dos, 05_wannier, 06_dfpt, 07_phonons, 08_lobster}

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
| `/api/open_results` | POST | Register an existing results folder (one job, a highthroughput dir, or an unpacked `results/`) for viewing; stored in `CONFIG['opened_results']` (slug → path), resolved by `_pd()` |

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
| `08_lobster` | LOBSTER bonding analysis: symmetry-off (ISYM=0) NSCF from `02_scf` CHGCAR + lobster run (COHP/COBI/COOP) |

## instruction_parser.py

Regex-based extraction from natural language instruction files. Supported parameters: functional (PBE/PBEsol/R2SCAN/HSE06/VV10/LDA), SOC + magnetization direction, GGA+U per element/orbital, task list, convergence test ranges, k-point path, Wannier90 projections and energy windows, DFPT flags, phonopy settings (supercell dim, mesh, displacement, NAC), DOS projections, explicit k-mesh override (KMESH), LOBSTER/COHP/COBI bonding analysis (auto-adds an `scf` dependency), MPI settings (KPAR, NCORE, np).

**LOBSTER step (`08_lobster`):** triggered by a `lobster`/`COHP`/`COBI`/`COOP`/`bonding analysis` task. `generate_lobster_input()` emits a symmetry-off NSCF (`ISYM=0`, `ICHARG=11` reading `02_scf/CHGCAR`, `NBANDS` ≥ number of LOBSTER basis functions via `_lobster_nbands()`, `LMAXMIX` from `_lmaxmix()`, `LWAVE=.TRUE.`); `run.sh` then builds `lobsterin` from the DOSCAR energy window and runs the lobster binary (`$LOBSTER_BIN`, default `lobster-5.1.0-OSX`). Aggregate per-bond ICOHP/ICOBI/ICOOP + antibonding integrals into a CSV with `lobster_postprocess.py` (reads `08_lobster/`, falling back to `02_scf/`).

**Constant pressure:** `PRESSURE = 10 GPa` (or `kbar`) triggers a constant-pressure relaxation — forces `ISIF=3`, `IBRION=2`, and emits `PSTRESS` (converted to kBar; GPa assumed if no unit). Parsed into the `pressure` dict.

**MAGMOM:** `MAGMOM:` takes a uniform number (negative allowed), per-element values (`Fe=4.0, O=0.6`, unlisted elements 0) or a per-atom list (`2*4.0 2*-4.0`); any `MAGMOM` line turns spin polarisation on. Parsed into `magmom` ({kind, value/values}); resolved to one number per atom by `VASPInputGenerator._magmom_values()` (raises ValueError on a length/element mismatch). With SOC the values are rotated onto the requested x/y/z direction (3N `MAGMOM` components); `ISPIN` is not written.

**k-mesh density:** `KMESH_DENSITY`/`KPOINTS_DENSITY`/`KPRA` = `coarse` (1000) | `fine` (5000) | integer kpra; `RELAX_KMESH_DENSITY` (default coarse). `_kpoints_from_kpra()` is one formula: s = (kpra/N_atoms/(|b1*||b2*||b3*|))^(1/3), N_i = nearest even of s·|b_i*| (min 2); hexagonal lattices (geometric test `_is_hex_lattice()`, not the GUI flag) round in-plane N to multiples of 6; 2-D slabs use the in-plane formula and Nz=1. `kpoints_text(step)` is the single source of truth for every step's KPOINTS (relax own density; DOS/LOBSTER = 2× SCF; DFPT/Wannier = SCF; phonons coarse) and is what the GUI's phase-2 regeneration calls.

**Explicit blocks:** `INCAR [step]: … END_INCAR` and `KPOINTS [step]: … END_KPOINTS` (also `INCAR_SCF:`), steps relax/scf/bands/dos/wannier/dfpt/phonons/lobster, unnamed = all (KPOINTS: all but bands). They override everything else. KPOINTS blocks are written verbatim and marked with `<step>/.explicit_kpoints` so convergence phase-2 patching (both agents, GUI) skips them; derived steps follow an explicit SCF mesh. KPAR/NCORE in an INCAR block steer the companion value. The parser blanks the blocks before scanning keywords, so tags inside a block never leak into global settings.

**Relax → SCF:** relax INCAR has `ISTART=0`, `ICHARG=2`, `LWAVE/LCHARG=.TRUE.`. `02_scf/copy_from_relax.sh` (run at RUN time by both run.sh flavours) copies CONTCAR (if newer) and CHGCAR; WAVECAR too and `ISTART=1, ICHARG=1` when KPOINTS (mesh), ISPIN, LSORBIT and ENCUT match the relax, else `ISTART=0, ICHARG=1`. `ISTART`/`ICHARG` in an `INCAR scf:` block disable it.

**KPAR/NCORE:** `_auto_kpar_ncore(np, n_k, kpar=None, ncore=None)`: KPAR = largest divisor of np ≤ n_k (irreducible k of the step's own KPOINTS); NCORE = largest divisor of np/KPAR ≤ its sqrt. A given value is snapped to a divisor and the other derived. Priority: ELF pin > INCAR block > per-step key > global key > rule. DFPT hard-codes 1/1; phonons and wannier default KPAR=1.

**ELF:** `ELF: on` (default) = LELF in the SCF, KPAR pinned to 1; `ELF: separate` (default in `ht-mp-scf.py`) = SCF at auto KPAR, then `02_scf/run_elf.sh` runs a short restart in `02_scf/elf/` (ISTART=1, ICHARG=1, LELF, KPAR=1, NBANDS from the SCF OUTCAR) and copies ELFCAR to `02_scf/`. Both run.sh flavours call run_elf.sh after VASP.

**GGA+U:** parser `gga_u_mode` = `auto` (no flag, or `GGA_U: AUTO`) | `on` (`GGA_U: ON`, or a bare `GGA+U` in Methods) | `off` (`GGA_U: OFF`, `no GGA+U`); key GGA_U/GGA-U/GGA+U/GGAU and value case-insensitive. `_u_lines()`: explicit `GGA+U with U=…` wins; else table U when mode=on, or mode=auto and the POSCAR has O/S/Se/Te/F/Cl/Br/I (`_U_ANIONS`); never automatic under R2SCAN/HSE06; same block in every step. `load_u_defaults()` reads ONLY `hubbard_u_defaults.csv` (whitespace table, `#` comments; element, orbital 3d/4f/…, U_eff) → LDAUU=U_eff, LDAUJ=0; no built-in values (warning + no U if the file is unreadable). GUI select `u_mode`; HT drivers `--gga_u auto|on|off` (aliases `--gga-u`, `--gga+u`, upper case; value case-insensitive).

**SLURM walltime + auto-continue:** `STEP_WALLTIME` (relax 8 h, scf 4 h, bands/dos 2 h, lobster 4 h) in `vasp_input_generator.py`; `vasp-agent-slurm.py._step_time()`: `<STEP>_WALLTIME` (parser `step_walltime`) > `WALLTIME` > STEP_WALLTIME > profile. Step scripts carry `RESUME_SBATCH` (`--signal=B:USR1@900 --requeue --open-mode=append`) and source `RESUME_LIB`: USR1 → STOPCAR (LSTOP relax / LABORT else), `scontrol requeue` (fallback: resubmit + re-point dependents); on restart `vf_resume_prepare` continues a relax from CONTCAR/last XDATCAR frame and skips a finished LOBSTER NSCF. `ht-semimetals.py` job.sbatch takes a step argument; `submit_all.sh` chains relax→scf→bands→lobster with `TIME_*` from env.sh.

**Raw INCAR passthrough:** an `INCAR: … END_INCAR` block in the instructions file injects literal INCAR tags into the generated INCAR(s). Per-step blocks use `INCAR <step>:` (relax/scf/bands/dos/wannier/dfpt/phonons); an unqualified block applies to all steps. Parsed into `incar_raw` ({'all'|step: ['TAG = val', …]}); merged by `VASPInputGenerator._apply_incar_overrides()`, which overwrites matching generated tags in place and appends the rest under a "User INCAR overrides" comment.

## vasp_input_generator.py

`VASPInputGenerator` class. Key methods: `generate_relax_input()`, `generate_scf_input()`, `generate_bands_input()`, `generate_dos_input()`, `generate_wannier_input()`, `generate_dfpt_input()`, `generate_phonons_input()`.

**Smart file copying:** `_write_copy_if_newer()` generates bash code that copies source→dest only if source is newer, preserving user edits.

**ENCUT defaults by functional:** PBE=400, PBEsol=450, R2SCAN=680, HSE06=400, VV10=450, LDA=350

## Testing

`python3 check_agent_consistency.py` generates a set of cases (MAGMOM forms, densities, explicit blocks, KPAR/NCORE, relax→SCF) through the GUI (Flask test client), `vasp-agent.py` and `vasp-agent-slurm.py` with a stub POTCAR library and requires identical INCAR/KPOINTS/copy scripts plus the expected values. Otherwise no formal test suite. Sample project directories: `GaAs_test/`, `AgCrPS3/`, `GaAs-phonons/`, `gaAs_wan/`, `GaAs_wannier2/`. Manual testing via browser GUI or CLI agent.
