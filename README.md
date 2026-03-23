# VASP Workflow GUI

A browser-based graphical interface for setting up, running, and analysing
[VASP](https://www.vasp.at/) density-functional-theory calculations.
It generates all input files, manages multi-step workflows, streams live
output, and produces publication-quality plots via
[sumo](https://github.com/SMTG-Bham/sumo).

---

## Features

| Area | Details |
|------|---------|
| **Setup** | Paste or load a POSCAR; choose functional (PBE, PBEsol, LDA, HSE06, R2SCAN, …), spin / SOC, GGA+U, 2D/slab flag |
| **POTCAR** | Browse available PAW variants per element and select with one click |
| **Workflows** | Convergence tests (ENCUT + k-mesh) → Relaxation → SCF → Band structure → DOS; two-phase or single-shot |
| **Parameters** | Convergence-test mode or manual ENCUT / k-mesh entry |
| **File editor** | View and edit INCAR, KPOINTS, POSCAR, OUTCAR, OSZICAR, vasp.out, run.sh per step directly in the browser |
| **Smart copying** | Files copied between steps (CONTCAR → POSCAR, SCF → bands/DOS) use a *newer-wins* rule so manual edits are preserved |
| **NBANDS** | Automatically set for band-structure runs from `ceil(NELECT/2) + 10` (rounded to even) using the SCF OUTCAR |
| **Plots** | Band structure and DOS generated with sumo; Fermi level taken from the dense-mesh DOS OUTCAR for accuracy |
| **Resume** | Reload any existing project from a dropdown on page open — no need to regenerate |

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.8 | |
| [Flask](https://flask.palletsprojects.com/) | `pip install flask` |
| [sumo](https://sumo.readthedocs.io/) | `pip install sumo` — for band/DOS plots |
| VASP | Licensed separately; set `$VASP_POTCAR_DIR` and `$VASP_EXEC` |

```bash
pip install flask sumo
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/vasp-workflow-gui.git
cd vasp-workflow-gui
pip install flask sumo
```

Set environment variables (add to `~/.bash_profile` or `~/.zshrc`):

```bash
export VASP_POTCAR_DIR="/path/to/PAW_PBE"   # directory containing element sub-folders
export VASP_EXEC="vasp_std"                  # or vasp_ncl for SOC runs
```

---

## Usage

```bash
python3 vasp-gui7.py
# Opens http://localhost:5001 in your browser automatically
```

### Workflow

1. **Setup tab** — enter project name, paste POSCAR, choose method and tasks, select POTCAR variants, click *Generate Workflow*
2. **Workflow tab** — run steps individually or all at once; view/edit any input file; monitor live output in log boxes
3. **Results tab** — click *Run sumo-bandplot* / *Run sumo-dosplot* to generate plots; view convergence charts

### Resuming a project

After restarting the server, use the **Resume project** dropdown at the top of the page to reload any existing project without regenerating.

---

## File structure

```
vasp-workflow-gui/
├── vasp-gui7.py            # Main GUI (Flask server + single-page app)
├── vasp-agent.py           # Workflow agent — generates VASP input files
├── modules/
│   ├── instruction_parser.py      # Parses instruction text into a settings dict
│   └── vasp_input_generator.py   # Generates INCAR, KPOINTS, run scripts, …
├── examples/
│   ├── example1_basic.txt         # Basic workflow instruction example
│   ├── example2_advanced.txt      # Advanced example (SOC, GGA+U, …)
│   └── POSCAR_MoS2                # Example POSCAR
├── manual.pdf              # Full user manual
├── INSTALL.md              # Detailed installation notes
└── QUICK_REFERENCE.txt     # One-page cheat-sheet
```

### Customising defaults

All GUI defaults (energy window, k-path, NSW, EDIFFG, POTCAR directory, …)
are collected in the `CONFIG` dict at the top of `vasp-gui7.py`:

```python
CONFIG = {
    'band_ymin':    '-4',
    'band_ymax':    '4',
    'dos_xmin':     '-6',
    'dos_xmax':     '6',
    'kpath':        'G-M-K-G',
    'nkpts_bands':  60,
    ...
}
```

The set of file buttons shown per workflow step is controlled by the
`STEP_FILES` list near the top of the JavaScript section — add or remove
one entry to change every step row at once.

---

## Notes on VASP licensing

VASP is commercial software. This tool only generates input files and
calls the VASP binary you supply — it does not distribute or include any
VASP source code or binaries. You must hold a valid VASP licence to use
this workflow.

---

## License

Copyright (C) 2024–2025  VASP Workflow GUI Contributors

This program is free software: you can redistribute it and/or modify it
under the terms of the **GNU General Public License version 3** as
published by the Free Software Foundation.

See [LICENSE](LICENSE) for the full text.
