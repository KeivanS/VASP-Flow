#!/usr/bin/env python3
"""
VASP Input Generator Module
Generates INCAR, KPOINTS, and job scripts for VASP calculations
"""

import os, shutil
from typing import Dict, List, Any
import numpy as np

class VASPInputGenerator:
    """Generate VASP input files based on calculation type and parameters"""
    
    def __init__(self, structure_file: str, instructions: Dict):
        self.poscar = structure_file
        self.instructions = instructions
        self.elements = self._read_elements_from_poscar()
    
    def _read_elements_from_poscar(self) -> List[str]:
        """Read element names from POSCAR"""
        with open(self.poscar, 'r') as f:
            lines = f.readlines()
            # Line 5 contains element names in VASP5 format
            if len(lines) > 5:
                elements = lines[5].split()
                # Check if line 5 is actually elements (contains letters)
                if any(c.isalpha() for c in lines[5]):
                    return elements
        return []
    
    # Default binary locations — change here if your installation differs
    VASP_BIN_DIR = "~/BIN"
    VASP_STD     = f"{VASP_BIN_DIR}/vasp_std"
    VASP_NCL     = f"{VASP_BIN_DIR}/vasp_ncl"
    VASP_GAM     = f"{VASP_BIN_DIR}/vasp_gam"

    def _get_vasp_exec(self) -> str:
        """Return the correct VASP binary path for this calculation.

        ~/BIN/vasp_ncl  — noncollinear / spin-orbit coupling (SOC)
        ~/BIN/vasp_gam  — Gamma-only (explicit gamma_only flag)
        ~/BIN/vasp_std  — everything else (default)
        """
        if self.instructions.get('soc', False):
            return self.VASP_NCL
        if self.instructions.get('gamma_only', False):
            return self.VASP_GAM
        return self.VASP_STD

    def _get_mpi_cmd(self, vasp_exec: str) -> str:
        """Build the mpirun launch line.

        Reads 'mpi_np' (number of MPI ranks) from instructions.
        On macOS with Open MPI the standard flags are:
            mpirun -np N --bind-to core vasp_xxx
        NCORE in the INCAR should be set so that NCORE * KPAR = N,
        with NCORE ~ sqrt(N) as a good starting point.
        """
        np = self.instructions.get('mpi_np', 1)
        if np <= 1:
            return vasp_exec
        # --bind-to core improves NUMA locality on multi-socket Macs
        return f"mpirun -np {np} {vasp_exec}"

    @staticmethod
    def _write_copy_if_newer(f, src_var, src_file, dst_file, label):
        """Write bash snippet: copy src to dst only if src is newer."""
        src = f'"{src_var}/{src_file}"'
        dst = f'"$HERE/{dst_file}"'
        f.write(f'if [ -f {src} ] && [ -s {src} ]; then\n')
        f.write(f'    if [ {src} -nt {dst} ]; then\n')
        f.write(f'        cp {src} {dst}\n')
        f.write(f'        echo "  {dst_file} updated from {label} ({src_file} is newer)"\n')
        f.write(f'    else\n')
        f.write(f'        echo "  Keeping existing {dst_file} (local copy is newer than {label})"\n')
        f.write(f'    fi\n')
        f.write(f'else\n')
        f.write(f'    echo "  WARNING: {label}/{src_file} not found; keeping existing {dst_file}"\n')
        f.write(f'fi\n')

    def generate_relax_input(self, output_dir: str):
        """Generate input files for structure relaxation"""
        os.makedirs(output_dir, exist_ok=True)
        
        # INCAR
        incar_content = self._generate_incar_relax()
        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(incar_content)
        
        # KPOINTS
        kpoints_content = self._generate_kpoints_auto(density='medium')
        with open(f"{output_dir}/KPOINTS", 'w') as f:
            f.write(kpoints_content)
        
        # Copy POSCAR
        os.system(f"cp {self.poscar} {output_dir}/POSCAR")
        
        # Job script
        job_script = self._generate_job_script('relax', self._get_vasp_exec())
        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(job_script)
        os.chmod(f"{output_dir}/run.sh", 0o755)
    
    def generate_scf_input(self, output_dir: str, from_relax: str = None):
        """Generate input files for self-consistent calculation"""
        os.makedirs(output_dir, exist_ok=True)
        
        # INCAR
        incar_content = self._generate_incar_scf()
        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(incar_content)
        
        # KPOINTS
        kpoints_content = self._generate_kpoints_auto(density='high')
        with open(f"{output_dir}/KPOINTS", 'w') as f:
            f.write(kpoints_content)
        
        # POSCAR: copy the input structure as a placeholder.
        # At runtime, run.sh will overwrite it with CONTCAR from 01_relax
        # if that directory exists and the relaxation completed.
        shutil.copy(self.poscar, f"{output_dir}/POSCAR")

        # Runtime copy script: pulls relaxed geometry from 01_relax
        if from_relax:
            rel_relax = os.path.relpath(from_relax, output_dir)
            with open(f"{output_dir}/copy_from_relax.sh", 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f'HERE="$(cd "$(dirname "$0")" && pwd)"\n')
                f.write(f'RELAX_DIR="$HERE/{rel_relax}"\n')
                self._write_copy_if_newer(f, '$RELAX_DIR', 'CONTCAR', 'POSCAR', '01_relax')
            os.chmod(f"{output_dir}/copy_from_relax.sh", 0o755)
        
        # Job script
        job_script = self._generate_job_script('scf', self._get_vasp_exec())
        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(job_script)
        os.chmod(f"{output_dir}/run.sh", 0o755)
    
    def generate_bands_input(self, output_dir: str, from_scf: str):
        """Generate input files for band structure calculation"""
        os.makedirs(output_dir, exist_ok=True)
        
        # INCAR
        incar_content = self._generate_incar_bands()
        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(incar_content)
        
        # KPOINTS
        kpath = self.instructions.get('kpath', ['G', 'X', 'M', 'G'])
        kpoints_content = self._generate_kpoints_linemode(kpath)
        with open(f"{output_dir}/KPOINTS", 'w') as f:
            f.write(kpoints_content)
        
        # Copy CHGCAR and POSCAR from SCF at runtime (relative path)
        rel_scf = os.path.relpath(from_scf, output_dir)
        with open(f"{output_dir}/copy_from_scf.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f'HERE="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write(f'SCF_DIR="$HERE/{rel_scf}"\n')
            f.write('# CHGCAR: always take the latest from SCF (not user-editable)\n')
            f.write('cp "$SCF_DIR/CHGCAR" "$HERE/"\n')
            f.write('echo "  CHGCAR copied from 02_scf"\n')
            self._write_copy_if_newer(f, '$SCF_DIR', 'POSCAR', 'POSCAR', '02_scf')
            # Set NBANDS from NELECT in SCF OUTCAR: ceil(NELECT/2) + 10, rounded to even
            f.write('if [ -f "$SCF_DIR/OUTCAR" ]; then\n')
            f.write('    NELECT=$(grep "^ *NELECT" "$SCF_DIR/OUTCAR" | head -1 | awk \'{print int($3)}\')\n')
            f.write('    if [ -n "$NELECT" ] && [ "$NELECT" -gt 0 ]; then\n')
            f.write('        NBANDS=$(( (NELECT / 2 + 10 + 1) / 2 * 2 ))\n')
            f.write('        if grep -q "^NBANDS" "$HERE/INCAR"; then\n')
            f.write('            sed -i.bak "s/^NBANDS.*/NBANDS = $NBANDS/" "$HERE/INCAR"\n')
            f.write('        else\n')
            f.write('            echo "NBANDS = $NBANDS" >> "$HERE/INCAR"\n')
            f.write('        fi\n')
            f.write('        echo "  NBANDS = $NBANDS  (NELECT = $NELECT, occupied = $((NELECT/2)))"\n')
            f.write('    fi\n')
            f.write('fi\n')
        os.chmod(f"{output_dir}/copy_from_scf.sh", 0o755)

        # Job script
        job_script = self._generate_job_script('bands', self._get_vasp_exec())
        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(job_script)
        os.chmod(f"{output_dir}/run.sh", 0o755)
    
    def generate_dos_input(self, output_dir: str, from_scf: str):
        """Generate input files for DOS calculation"""
        os.makedirs(output_dir, exist_ok=True)
        
        # INCAR
        incar_content = self._generate_incar_dos()
        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(incar_content)
        
        # KPOINTS - denser mesh
        kpoints_content = self._generate_kpoints_auto(density='very_high')
        with open(f"{output_dir}/KPOINTS", 'w') as f:
            f.write(kpoints_content)
        
        # Copy from SCF at runtime (relative path)
        rel_scf = os.path.relpath(from_scf, output_dir)
        with open(f"{output_dir}/copy_from_scf.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f'HERE="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write(f'SCF_DIR="$HERE/{rel_scf}"\n')
            f.write('# CHGCAR: always take the latest from SCF (not user-editable)\n')
            f.write('cp "$SCF_DIR/CHGCAR" "$HERE/"\n')
            f.write('echo "  CHGCAR copied from 02_scf"\n')
            self._write_copy_if_newer(f, '$SCF_DIR', 'POSCAR', 'POSCAR', '02_scf')
        os.chmod(f"{output_dir}/copy_from_scf.sh", 0o755)

        # Job script
        job_script = self._generate_job_script('dos', self._get_vasp_exec())
        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(job_script)
        os.chmod(f"{output_dir}/run.sh", 0o755)
    
    def _get_parallel_lines(self, calc_type: str = 'scf') -> list:
        """Return INCAR lines for MPI parallelization.

        Hardwired defaults for a 16-core machine (KPAR * NCORE = N):

          relax / scf / dos  →  KPAR=2, NCORE=8
            Dense k-mesh: split cores into 2 k-groups of 8 cores each.

          bands              →  KPAR=1, NCORE=16
            Line-mode k-points cannot be parallelised over k; give all
            cores to band-orbital groups.

          ncl (SOC, vasp_ncl) → KPAR=2, NCORE=8
            NCL doubles memory per band so keep NCORE moderate.

        Any value explicitly set in the instruction file (kpar / ncore)
        overrides the hardwired defaults for every calc type.
        """
        np = self.instructions.get('mpi_np', 1) or 1
        if np <= 1:
            return []

        # Hardwired per-type defaults
        defaults = {
            'relax': (2, np // 2),
            'scf':   (4, np // 4),
            'bands': (1, np),        # KPAR=1 for line-mode
            'dos':   (4, np // 4),
        }
        # If SOC (vasp_ncl), cap NCORE at np//2 regardless of calc type
        soc = self.instructions.get('soc', False)
        if soc:
            defaults = {k: (2, np // 2) for k in defaults}
            defaults['bands'] = (1, np // 2)  # still KPAR=1 for bands

        kpar_def, ncore_def = defaults.get(calc_type, (2, np // 2))

        # Instruction-file overrides win if explicitly set
        kpar  = self.instructions.get('kpar')  or kpar_def
        ncore = self.instructions.get('ncore') or ncore_def

        # Safety: ensure KPAR * NCORE never exceeds np
        kpar  = max(1, min(kpar,  np))
        ncore = max(1, min(ncore, np // kpar))

        return [
            "",
            "# MPI parallelization",
            f"KPAR  = {kpar}",
            f"NCORE = {ncore}",
        ]

    def _encut(self) -> int:
        """Return ENCUT: instruction-specified value, or 500 default."""
        return self.instructions.get('encut_val') or 500

    def _functional_lines(self) -> list:
        """Return INCAR lines for the chosen functional."""
        f = self.instructions.get('functional', 'PBE')
        if f == 'PS':
            return ["GGA = PS"]
        if f == 'LDA':
            return ["# LDA — no GGA tag; VOSKOWN for Vosko-Wilk-Nusair interpolation",
                    "VOSKOWN = 1"]
        if f == 'AM':
            return ["GGA = AM"]
        if f == 'R2SCAN':
            return ["METAGGA = R2SCAN", "LASPH = .TRUE."]
        if f == 'HSE06':
            return ["# HSE06 hybrid functional",
                    "LHFCALC = .TRUE.", "HFSCREEN = 0.2",
                    "ALGO = Damped", "TIME = 0.4"]
        return []   # PBE is the default — no tag needed

    def _soc_lines(self) -> list:
        if not self.instructions.get('soc', False):
            return []
        mag = self.instructions.get('magnetization', {})
        dir_map = {'x': '1 0 0', 'y': '0 1 0', 'z': '0 0 1'}
        saxis = dir_map.get(mag.get('direction', 'z'), '0 0 1')
        return ["# Spin-orbit coupling",
                "LSORBIT = .TRUE.", "LNONCOLLINEAR = .TRUE.",
                f"SAXIS = {saxis}", ""]

    def _mag_lines(self) -> list:
        mag = self.instructions.get('magnetization', {})
        if not mag.get('enabled') or self.instructions.get('soc', False):
            return []
        return ["# Collinear magnetization", "ISPIN = 2",
                "MAGMOM = 2*0.6", ""]

    def _u_lines(self) -> list:
        """Generate GGA+U INCAR lines with actual per-element U values."""
        u_info = self.instructions.get('gga_u', {})
        if not u_info.get('enabled'):
            return []
        els = u_info.get('elements', {})
        if not els:
            return []
        # Build LDAUL, LDAUU, LDAUJ arrays in element order
        orb_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        ldaul, ldauu, ldauj = [], [], []
        for el in (self.elements or list(els.keys())):
            if el in els:
                orb = els[el].get('orbital', 'd')
                ldaul.append(str(orb_map.get(orb, 2)))
                ldauu.append(str(els[el].get('U', 0.0)))
                ldauj.append('0.0')
            else:
                ldaul.append('-1')
                ldauu.append('0.0')
                ldauj.append('0.0')
        return ["# GGA+U (Dudarev, LDAUTYPE=2)",
                "LDAU = .TRUE.", "LDAUTYPE = 2",
                f"LDAUL = {' '.join(ldaul)}",
                f"LDAUU = {' '.join(ldauu)}",
                f"LDAUJ = {' '.join(ldauj)}",
                "LDAUPRINT = 2", ""]

    def _generate_incar_relax(self) -> str:
        """Generate INCAR for relaxation."""
        encut  = self._encut()
        isif   = self.instructions.get('isif')   or 3
        nsw    = self.instructions.get('nsw')    or 100
        ediffg = self.instructions.get('ediffg') or -0.01

        lines = [
            "# Relaxation calculation",
            "SYSTEM = " + self.instructions.get('project_name', 'Relaxation'),
            "",
            "# Electronic structure",
            "PREC = Accurate",
            f"ENCUT = {encut}",
            "EDIFF = 1E-6",
            "NELM = 100",
            "ALGO = Normal",
            "",
            "# Ionic relaxation",
            "IBRION = 2",
            f"ISIF = {isif}",
            f"NSW = {nsw}",
            f"EDIFFG = {ediffg}",
            "",
            "# Electronic smearing",
            "ISMEAR = 0",
            "SIGMA = 0.05",
            "",
        ]

        fl = self._functional_lines()
        if fl: lines += fl + [""]
        lines += self._mag_lines()
        lines += self._soc_lines()
        lines += self._u_lines()
        lines.extend(self._get_parallel_lines('relax'))
        lines.extend(["# Output", "LWAVE = .TRUE.", "LCHARG = .TRUE.", "LORBIT = 11"])
        return '\n'.join(lines) + '\n'

    def _generate_incar_scf(self) -> str:
        """Generate INCAR for SCF calculation."""
        encut = self._encut()
        lines = [
            "# Self-consistent field calculation",
            "SYSTEM = " + self.instructions.get('project_name', 'SCF'),
            "",
            "# Electronic structure",
            "PREC = Accurate",
            f"ENCUT = {encut}",
            "EDIFF = 1E-6",
            "NELM = 100",
            "ALGO = Normal",
            "",
            "# Static calculation",
            "IBRION = -1",
            "NSW = 0",
            "",
            "# Electronic smearing",
            "ISMEAR = 0",
            "SIGMA = 0.05",
            "",
        ]
        fl = self._functional_lines()
        if fl: lines += fl + [""]
        lines += self._mag_lines()
        lines += self._soc_lines()
        lines += self._u_lines()
        
        lines.extend(self._get_parallel_lines('scf'))
        lines.extend(["# Output", "LWAVE = .TRUE.", "LCHARG = .TRUE.", "LORBIT = 11"])
        return '\n'.join(lines) + '\n'

    def _generate_incar_bands(self) -> str:
        """Generate INCAR for band structure (NSCF)."""
        encut = self._encut()
        lines = [
            "# Band structure (NSCF)",
            "SYSTEM = " + self.instructions.get('project_name', 'Bands'),
            "",
            "# Electronic structure",
            "PREC = Accurate",
            f"ENCUT = {encut}",
            "EDIFF = 1E-6",
            "NELM = 100",
            "",
            "# Non-self-consistent from CHGCAR",
            "ICHARG = 11",
            "IBRION = -1",
            "NSW = 0",
            "",
            "# Smearing",
            "ISMEAR = 0",
            "SIGMA = 0.05",
            "",
        ]
        fl = self._functional_lines()
        if fl: lines += fl + [""]
        lines += self._mag_lines()
        lines += self._soc_lines()
        lines += self._u_lines()
        lines.extend(self._get_parallel_lines('bands'))
        lines.extend(["# Output", "LWAVE = .FALSE.", "LCHARG = .FALSE.", "LORBIT = 11"])
        return '\n'.join(lines) + '\n'

    def _generate_incar_dos(self) -> str:
        """Generate INCAR for DOS (NSCF with tetrahedron smearing)."""
        encut = self._encut()
        lines = [
            "# Density of states (NSCF)",
            "SYSTEM = " + self.instructions.get('project_name', 'DOS'),
            "",
            "# Electronic structure",
            "PREC = Accurate",
            f"ENCUT = {encut}",
            "EDIFF = 1E-6",
            "NELM = 100",
            "",
            "# Non-self-consistent from CHGCAR",
            "ICHARG = 11",
            "IBRION = -1",
            "NSW = 0",
            "",
            "# Tetrahedron method — best for DOS",
            "ISMEAR = -5",
            "",
        ]
        fl = self._functional_lines()
        if fl: lines += fl + [""]
        lines += self._mag_lines()
        lines += self._soc_lines()
        lines += self._u_lines()
        lines.extend(self._get_parallel_lines('dos'))
        lines.extend([
            "# DOS output",
            "LORBIT = 11",
            "NEDOS = 3000",
            "EMIN = -15",
            "EMAX = 15",
            "",
            "LWAVE = .FALSE.",
            "LCHARG = .FALSE.",
        ])
        return '\n'.join(lines) + '\n'
    
    def _generate_kpoints_auto(self, density: str = 'medium') -> str:
        """Generate automatic Gamma-centred KPOINTS file.

        The VASP format after the 'Gamma' keyword requires three integers
        (N1 N2 N3) for the mesh subdivisions along each reciprocal lattice
        vector, followed by a shift line (S1 S2 S3).  A single integer is
        not a valid specification and VASP will reject or misinterpret it.

        For 2-D slab calculations (e.g. MoS2 monolayer) the out-of-plane
        direction is already sampled by a single k-point, so kz = 1.
        Adjust is_2d via the instructions dict if needed.
        """
        # In-plane mesh density for each tier
        density_map = {
            'low':       6,
            'medium':   10,
            'high':     14,
            'very_high': 18,
        }

        n_xy = density_map.get(density, 10)

        # Detect 2-D slab: long c-axis (vacuum layer) → kz = 1
        is_2d = self.instructions.get('is_2d', False)
        n_z = 1 if is_2d else n_xy

        return (
            f"Automatic Gamma mesh\n"
            f"0\n"
            f"Gamma\n"
            f"  {n_xy}  {n_xy}  {n_z}\n"
            f"  0    0    0\n"
        )
    
    def _generate_kpoints_linemode(self, kpath: List[str], npoints: int = 40) -> str:
        """Generate KPOINTS file for band structure.
        npoints is overridden by 'nkpts_bands' in instructions if set.
        """
        npoints = self.instructions.get('nkpts_bands') or npoints
        # High-symmetry k-point coordinates in reduced reciprocal coordinates.
        # M differs between hexagonal and cubic/tetragonal BZs:
        #   Hexagonal  (MoS2, graphene): M = (1/2, 0, 0)
        #   Cubic/Tetrag.             : M = (1/2, 1/2, 0)
        # Set 'is_hex': False in instructions for non-hexagonal structures.
        is_hex = self.instructions.get('is_hex', True)

        kpoint_coords = {
            'G': [0.0,       0.0,       0.0],
            'Z': [0.0,       0.0,       0.5],
            'M': [0.5,       0.0,       0.0] if is_hex else [0.5, 0.5, 0.0],
            'K': [1/3,       1/3,       0.0],
            'H': [1/3,       1/3,       0.5],
            'A': [0.0,       0.0,       0.5],
            'L': [0.5,       0.0,       0.5],
            'X': [0.5,       0.0,       0.0],
            'Y': [0.0,       0.5,       0.0],
            'R': [0.5,       0.5,       0.5],
        }
        
        lines = ["Line-mode KPOINTS\n", f"{npoints}\n", "Line-mode\n", "rec\n"]
        
        for i in range(len(kpath) - 1):
            start_label = kpath[i]
            end_label = kpath[i + 1]
            
            start = kpoint_coords.get(start_label, [0.0, 0.0, 0.0])
            end = kpoint_coords.get(end_label, [0.0, 0.0, 0.0])
            
            lines.append(f"  {start[0]:8.4f} {start[1]:8.4f} {start[2]:8.4f}  ! {start_label}\n")
            lines.append(f"  {end[0]:8.4f} {end[1]:8.4f} {end[2]:8.4f}  ! {end_label}\n")
            lines.append("\n")
        
        return ''.join(lines)
    
    def _generate_potcar_script(self) -> str:
        """Generate script to create POTCAR"""
        elements_str = ' '.join(self.elements) if self.elements else "Si"
        
        script = f"""#!/bin/bash
# Script to generate POTCAR file
# Modify POTCAR_DIR to point to your pseudopotential directory

POTCAR_DIR="$HOME/potcar/PBE"  # MODIFY THIS PATH

# Elements in your structure
ELEMENTS="{elements_str}"

# Create POTCAR
rm -f POTCAR
for elem in $ELEMENTS; do
    if [ -f "$POTCAR_DIR/$elem/POTCAR" ]; then
        cat "$POTCAR_DIR/$elem/POTCAR" >> POTCAR
        echo "Added $elem to POTCAR"
    else
        echo "ERROR: POTCAR not found for $elem at $POTCAR_DIR/$elem/POTCAR"
        exit 1
    fi
done

echo "POTCAR created successfully"
"""
        return script
    
    def _generate_job_script(self, calc_type: str, vasp_exec: str) -> str:
        """Generate job execution script with correct MPI invocation."""
        launch_cmd = self._get_mpi_cmd(vasp_exec)
        np         = self.instructions.get('mpi_np', 1)

        script = f"""#!/bin/bash
# run.sh for {calc_type}
# VASP binary : {vasp_exec}
# MPI ranks   : {np}
# Launch      : {launch_cmd}

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

# POTCAR is a symlink to the project-level POTCAR built by vasp-agent.py.
if [ ! -f "$HERE/POTCAR" ]; then
    echo "ERROR: POTCAR not found in $HERE"
    echo "  Re-run vasp-agent.py to rebuild it."
    exit 1
fi

# SCF: pull relaxed geometry from 01_relax/CONTCAR
if [ -f "$HERE/copy_from_relax.sh" ]; then
    bash "$HERE/copy_from_relax.sh"
fi

# Bands/DOS: pull CHGCAR + POSCAR from 02_scf
if [ -f "$HERE/copy_from_scf.sh" ]; then
    bash "$HERE/copy_from_scf.sh"
fi

# Run VASP
echo "Starting {calc_type} ({vasp_exec}) at $(date)"
{launch_cmd} > vasp.out 2>&1

# Convergence check
if grep -q "reached required accuracy" OUTCAR 2>/dev/null; then
    echo "  OK: converged at $(date)"
else
    echo "  WARNING: may not have converged -- check OUTCAR"
fi
"""
        return script
