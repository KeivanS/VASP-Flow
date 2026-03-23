# VASP Workflow Agent - Installation & Getting Started

## Quick Installation

1. **Extract the archive:**
```bash
tar -xzf vasp-workflow.tar.gz
cd vasp-workflow
```

2. **Set permissions:**
```bash
chmod +x vasp-agent.py test_install.sh
```

3. **Test the installation:**
```bash
./test_install.sh
```

## First Steps

### 1. Configure POTCAR Path

You need to set the path to your VASP pseudopotentials. This can be done in two ways:

**Option A: Set environment variable (recommended)**
```bash
export VASP_POTCAR_DIR="$HOME/SIMULATIONS/VASP"
# Add this line to your ~/.bashrc to make it permanent
```

**Option B: Edit paths manually**
After generating a workflow, edit the `make_potcar.sh` file in each calculation directory.

### 2. Your First Calculation

Create a simple instruction file `my_instructions.txt`:
```
Project: My first VASP calculation

Methods: PBEsol functional

Tasks: Relaxation
       Band structure along G-M-K-G
       DOS

Convergence: Test k-points 8x8 to 12x12
```

Run the agent:
```bash
./vasp-agent.py -p my_first_project \
                -i my_instructions.txt \
                -s /path/to/your/POSCAR
```

Navigate to generated files:
```bash
cd projects/my_first_project/generated
```

Review and edit if needed:
```bash
cat 01_relax/INCAR          # Check INCAR settings
cat 01_relax/KPOINTS        # Check k-points
nano 01_relax/make_potcar.sh   # Edit POTCAR path if needed
```

Run calculations:
```bash
./submit_all.sh              # Run all calculations
# OR
cd 01_relax && ./run.sh     # Run step by step
```

## Documentation

- **Quick Reference:** `QUICK_REFERENCE.txt` - Print and keep handy
- **Full Manual:** `manual.pdf` - Complete documentation
- **Examples:** `examples/` - Example instruction files
- **README:** `README.md` - Overview and features

## Instruction File Syntax

### Basic Template:
```
Project: [Your project name]

Methods: [Functional]
         [Additional methods like SOC, GGA+U]

Tasks: [Calculation 1]
       [Calculation 2]
       [...]

Convergence: [Convergence tests]

Goals: [Optional description]
```

### Available Options:

**Functionals:**
- PBE, PBEsol, R2SCAN, HSE06, RVV10

**Special Methods:**
- `SOC with magnetization in z-direction`
- `GGA+U with U=3.0 on Mo-d orbitals`

**Tasks:**
- Relaxation, SCF, Band structure, DOS, Fat bands, Wannierization, Transport

**Convergence:**
- `Test k-points 8x8 to 16x16`
- `ENCUT 400-600 eV`

## Examples

See `examples/example1_basic.txt` and `examples/example2_advanced.txt` for complete examples.

## Directory Structure After Generation

```
projects/YOUR_PROJECT/
├── instructions.txt          # Your instructions (backup)
├── POSCAR                   # Your structure (backup)
└── generated/               # All generated files
    ├── 00_convergence/      # Convergence test scripts
    ├── 01_relax/            # Relaxation
    │   ├── INCAR
    │   ├── KPOINTS
    │   ├── POSCAR
    │   ├── make_potcar.sh
    │   └── run.sh
    ├── 02_scf/              # (if requested)
    ├── 03_bands/            # (if requested)
    ├── 04_dos/              # (if requested)
    ├── analysis/            # Analysis scripts
    ├── submit_all.sh        # Master submission script
    └── analyze_all.sh       # Master analysis script
```

## For SLURM Clusters

To use on a SLURM cluster, modify the `run.sh` scripts:

```bash
#!/bin/bash
#SBATCH --job-name=vasp_calc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --partition=normal

module load vasp/6.3.0

# Create POTCAR if not present
if [ ! -f POTCAR ]; then
    ./make_potcar.sh
fi

# Run VASP
srun vasp_std > vasp.out 2>&1
```

Then submit with:
```bash
sbatch run.sh
```

## Checking Results

After calculations:
```bash
# Check if converged
grep "reached required accuracy" 01_relax/OUTCAR

# Extract energy
grep "energy  without" 01_relax/OUTCAR | tail -1

# Check for errors
grep "ERROR" 01_relax/OUTCAR
```

## Troubleshooting

**Problem:** POTCAR not found  
**Solution:** Edit `make_potcar.sh` with correct path or set VASP_POTCAR_DIR

**Problem:** vasp_std command not found  
**Solution:** Add VASP to PATH or use full path in run.sh

**Problem:** Calculation not converging  
**Solution:** Check OUTCAR for errors, increase NSW or adjust convergence criteria

**Problem:** Memory error  
**Solution:** Reduce k-point density or ENCUT, or use more resources

## Getting Help

1. Read `QUICK_REFERENCE.txt` for syntax
2. Check `manual.pdf` for detailed documentation
3. Review `examples/` for working examples
4. Examine generated files to understand the workflow

## Tips for Success

✓ Always review generated INCAR files before running
✓ Start with convergence tests before expensive calculations
✓ Use relaxed structures for band structure and DOS
✓ Keep instruction files for reproducibility
✓ Backup CHGCAR and WAVECAR for restarting calculations

## What's Next?

1. **Run the test:** `./test_install.sh`
2. **Read the manual:** Open `manual.pdf`
3. **Try an example:** Use `examples/example1_basic.txt`
4. **Create your workflow:** Write your own instruction file
5. **Customize:** Edit generated files as needed

## System Requirements

- VASP 5.4 or later
- Python 3.6 or later
- Linux/Unix environment
- VASP pseudopotentials (POTCAR files)
- Sufficient disk space and memory for calculations

## Version

Version 1.0 (2026-01-27)
Created by Claude (Anthropic)

---

For complete documentation, see `manual.pdf`
