# Examples

| File | Shows |
|---|---|
| `instructions_LaP_full.txt` | Full workflow: convergence tests (auto-selected ENCUT/k-mesh), SCF with ELF, band structure, projected DOS, DFPT (Born charges + dielectric), LOBSTER COHP/COBI bonding — plus explicit VASP keywords via INCAR blocks |
| `POSCAR_LaP` | Rocksalt LaP structure for the full-workflow example |
| `instructions_pressure.txt` | Constant-pressure relaxation (`PRESSURE:` → PSTRESS/ISIF=3) and the raw INCAR passthrough |
| `POSCAR_MoS2` | 2D layered structure for the pressure example |
| `highthrouput_list` | Input list (one Materials Project ID per line) for `ht-mp-scf.py` batch SCF+ELF runs |

Run an example from the repo root:

```bash
./vasp-agent.py -i examples/instructions_pressure.txt -s examples/POSCAR_MoS2
```

## Setting any VASP keyword explicitly

Every VASP INCAR tag can be specified — none are hard-wired. Three routes,
in increasing precedence:

1. **Instructions file** — `INCAR: … END_INCAR` blocks, global or per-step
   (`INCAR scf:`). See both example files.
2. **Command line** — repeatable `--incar` option, optionally step-prefixed:
   ```bash
   ./vasp-agent.py -i instructions.txt -s POSCAR \
       --incar "ALGO=Fast; NELMIN=6" --incar "scf:NBANDS=64"
   ```
   (also accepted by `vasp-agent-slurm.py`)
3. **GUI** — every generated INCAR is editable per step: Workflow tab →
   **✏ Edit files** → INCAR → save → re-run the step. A `.bak` backup is
   kept automatically.

Tags you set override the generated defaults (matching tags are replaced in
place; new ones are appended under a "User INCAR overrides" comment). The
only exceptions are physics-critical guards, which are re-imposed: `ISYM=2`
and `KPAR=NCORE=1` in `06_dfpt`, and symmetry-off `ISYM` in `08_lobster`.
