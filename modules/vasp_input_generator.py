#!/usr/bin/env python3
"""
VASP Input Generator Module
Generates INCAR, KPOINTS, and job scripts for VASP calculations
"""

import os, shutil
from typing import Dict, List, Any
import numpy as np

# ── Platform settings ────────────────────────────────────────────────────────
# Read from environment (set via site.env + Makefile export).
# These can also be set as shell environment variables directly.
_VASP_STD    = os.environ.get('VASP_STD',    '~/BIN/vasp_std')
_VASP_NCL    = os.environ.get('VASP_NCL',    '~/BIN/vasp_ncl')
_VASP_GAM    = os.environ.get('VASP_GAM',    '~/BIN/vasp_gam')
_MPI_LAUNCH  = os.environ.get('MPI_LAUNCH',  'mpirun -np')
_MPI_NP      = int(os.environ.get('MPI_NP',  '1'))
_WANNIER90_X = os.environ.get('WANNIER90_X', 'wannier90.x')
_LOBSTER_X   = os.environ.get('LOBSTER_X',   'lobster-5.1.0-OSX')

# ── Default k-paths per Bravais lattice type (Setyawan & Curtarolo 2010) ─────
# All coordinates are fractional reciprocal (VASP "rec" format) in the
# PRIMITIVE cell reciprocal basis for cF/cI/hP/tP/oP/rP/mP, or the
# simple-cubic reciprocal basis for cP (conventional cubic).
_KPATH_LIBRARY = {
    'cF': {   # FCC primitive  (cell angles ≈ 60°)
        'name': 'FCC',
        'path':    ['G','X','W','K','G','L','U','W'],
        'path_2d': ['G','X','W','K','G'],
        'coords': {
            'G': [0.000, 0.000, 0.000],
            'X': [0.500, 0.000, 0.500],
            'W': [0.500, 0.250, 0.750],
            'K': [0.375, 0.375, 0.750],
            'L': [0.500, 0.500, 0.500],
            'U': [0.625, 0.250, 0.625],
        },
    },
    'cI': {   # BCC primitive  (cell angles ≈ 109.47°)
        'name': 'BCC',
        'path':    ['G','H','N','G','P','H'],
        'path_2d': ['G','H','N','G'],
        'coords': {
            'G': [ 0.000,  0.000,  0.000],
            'H': [ 0.500, -0.500,  0.500],
            'N': [ 0.000,  0.000,  0.500],
            'P': [ 0.250,  0.250,  0.250],
        },
    },
    'cP': {   # Simple cubic / conventional cubic (right angles, a=b=c)
        'name': 'Cubic',
        'path':    ['G','X','M','G','R','X'],
        'path_2d': ['G','X','M','G'],
        'coords': {
            'G': [0.000, 0.000, 0.000],
            'X': [0.000, 0.500, 0.000],
            'M': [0.500, 0.500, 0.000],
            'R': [0.500, 0.500, 0.500],
        },
    },
    'hP': {   # Hexagonal  (a=b, γ=120°)
        'name': 'Hexagonal',
        'path':    ['G','M','K','G','A','L','H','A'],
        'path_2d': ['G','M','K','G'],
        'coords': {
            'G': [0.000,       0.000,       0.000],
            'M': [0.500,       0.000,       0.000],
            'K': [1/3,         1/3,         0.000],
            'A': [0.000,       0.000,       0.500],
            'L': [0.500,       0.000,       0.500],
            'H': [1/3,         1/3,         0.500],
        },
    },
    'tP': {   # Simple tetragonal  (a=b≠c, 90°)
        'name': 'Tetragonal',
        'path':    ['G','X','M','G','Z','R','A','Z'],
        'path_2d': ['G','X','M','G'],
        'coords': {
            'G': [0.000, 0.000, 0.000],
            'X': [0.500, 0.000, 0.000],
            'M': [0.500, 0.500, 0.000],
            'Z': [0.000, 0.000, 0.500],
            'R': [0.500, 0.000, 0.500],
            'A': [0.500, 0.500, 0.500],
        },
    },
    'oP': {   # Simple orthorhombic  (a≠b≠c, 90°)
        'name': 'Orthorhombic',
        'path':    ['G','X','S','Y','G','Z','U','R','T','Z'],
        'path_2d': ['G','X','S','Y','G'],
        'coords': {
            'G': [0.000, 0.000, 0.000],
            'X': [0.500, 0.000, 0.000],
            'Y': [0.000, 0.500, 0.000],
            'Z': [0.000, 0.000, 0.500],
            'S': [0.500, 0.500, 0.000],
            'T': [0.000, 0.500, 0.500],
            'U': [0.500, 0.000, 0.500],
            'R': [0.500, 0.500, 0.500],
        },
    },
    'rP': {   # Rhombohedral primitive  (a=b=c, equal angles ≠ 90°)
        'name': 'Rhombohedral',
        'path':    ['G','T','F','G','L'],
        'path_2d': ['G','T','F','G'],
        'coords': {
            'G': [0.000, 0.000, 0.000],
            'T': [0.500, 0.500, 0.500],
            'F': [0.500, 0.500, 0.000],
            'L': [0.500, 0.000, 0.000],
        },
    },
    'mP': {   # Simple monoclinic  (α=γ=90°, β≠90°) — simplified path
        'name': 'Monoclinic',
        'path':    ['G','Y','A','Z','G','B','D'],
        'path_2d': ['G','Y','B','G'],
        'coords': {
            'G': [0.000, 0.000, 0.000],
            'Y': [0.000, 0.500, 0.000],
            'A': [0.000, 0.500, 0.500],
            'Z': [0.000, 0.000, 0.500],
            'B': [0.500, 0.000, 0.000],
            'D': [0.500, 0.500, 0.000],
        },
    },
}


class VASPInputGenerator:
    """Generate VASP input files based on calculation type and parameters"""

    def __init__(self, structure_file: str, instructions: Dict, profile: Dict = None):
        self.poscar = structure_file
        self.instructions = instructions
        self.profile = profile or {}
        self.elements = self._read_elements_from_poscar()

    def _profile_get(self, key: str, env_fallback: str) -> str:
        """Return profile value if set, otherwise the environment-variable fallback."""
        val = self.profile.get(key, '')
        return val if val else env_fallback
    
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
    
    def _get_vasp_exec(self) -> str:
        """Return the correct VASP binary (profile overrides site.env/environment)."""
        if self.instructions.get('soc', False):
            return self._profile_get('vasp_ncl', _VASP_NCL)
        if self.instructions.get('gamma_only', False):
            return self._profile_get('vasp_gam', _VASP_GAM)
        return self._profile_get('vasp_std', _VASP_STD)

    def _get_lobster_exec(self) -> str:
        """LOBSTER binary (profile 'lobster_x' overrides site.env LOBSTER_X)."""
        return self._profile_get('lobster_x', _LOBSTER_X)

    def _get_mpi_cmd(self, vasp_exec: str) -> str:
        """Build the MPI launch line. Profile mpi_cmd overrides site.env MPI_LAUNCH.

        mpi_cmd may or may not contain '{np}':
          'mpirun -np {np}' → 'mpirun -np 16 vasp_std'   (workstation)
          'srun'            → 'srun vasp_std'              (SLURM: task count from #SBATCH)
          ''                → falls back to env MPI_LAUNCH
        """
        np = self.instructions.get('mpi_np') or self.profile.get('mpi_np') or _MPI_NP
        if not np or np <= 1:
            return vasp_exec
        mpi_cmd = self.profile.get('mpi_cmd', '')
        if not mpi_cmd:
            return f"{_MPI_LAUNCH} {np} {vasp_exec}"
        if '{np}' in mpi_cmd:
            return f"{mpi_cmd.format(np=np)} {vasp_exec}"
        return f"{mpi_cmd} {vasp_exec}"

    def _run_sh_preamble(self, job_name: str = 'vasp') -> str:
        """Return SBATCH header + module loads for SLURM profiles, else empty string."""
        slurm   = self.profile.get('slurm')
        modules = self.profile.get('modules', [])
        lines   = []

        if slurm:
            s = slurm
            lines.append(f'#SBATCH --job-name={job_name}')
            lines.append(f'#SBATCH --partition={s.get("partition", "standard")}')
            lines.append(f'#SBATCH --nodes={s.get("nodes", 1)}')
            lines.append(f'#SBATCH --ntasks-per-node={s.get("ntasks_per_node", 1)}')
            lines.append(f'#SBATCH --time={s.get("time", "24:00:00")}')
            if s.get('account'):
                lines.append(f'#SBATCH --account={s["account"]}')
            lines.append(f'#SBATCH --output={s.get("output", "slurm-%j.out")}')
            lines.append(f'#SBATCH --error={s.get("error", "slurm-%j.err")}')
            lines.append('')

        for mod in modules:
            lines.append(f'module load {mod}')
        if modules:
            lines.append('')

        return '\n'.join(lines) + ('\n' if lines else '')

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
    
    def _ncore_for(self, step: str) -> int:
        """Parse the NCORE value the parallel block would emit for `step`."""
        import re
        for ln in self._get_parallel_lines(step):
            m = re.match(r'\s*NCORE\s*=\s*(\d+)', ln)
            if m:
                return int(m.group(1))
        return 1

    def _lobster_nbands(self, potcar_path: str, ncore: int = 1):
        """NBANDS large enough for a later LOBSTER analysis, or None.

        LOBSTER requires NBANDS >= the number of local basis functions
        (sum over atoms of s/p/d/f orbital multiplicities for the
        pbeVaspFit2015 basis), which usually exceeds the occupied-band count.
        Returns None if pymatgen / the POTCAR are unavailable so SCF generation
        never fails just because the band count could not be computed.
        """
        if not potcar_path or not os.path.isfile(potcar_path):
            return None
        try:
            import math
            from pymatgen.core import Structure
            from pymatgen.io.vasp.inputs import Potcar
            from pymatgen.io.lobster import Lobsterin
        except Exception:
            return None
        try:
            mult = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
            st = Structure.from_file(self.poscar)
            pot = Potcar.from_file(potcar_path)
            symbols = [p.symbol for p in pot]
            zval = {p.symbol.split('_')[0]: float(p.zval) for p in pot}
            counts = {}
            for site in st:
                el = site.specie.symbol
                counts[el] = counts.get(el, 0) + 1
            nelect = sum(zval[el] * n for el, n in counts.items())
            per_el = {}
            for entry in Lobsterin.get_basis(st, potcar_symbols=symbols):
                toks = entry.split()
                el = toks[0].split('_')[0]
                per_el[el] = sum(mult[o[-1]] for o in toks[1:])
            n_basis = sum(per_el[el] * counts[el] for el in counts)
            nb = max(n_basis, math.ceil(nelect / 2))
            if ncore and ncore > 1:
                nb = int(math.ceil(nb / ncore) * ncore)
            return nb
        except Exception:
            return None

    def _lmaxmix(self) -> int:
        """LMAXMIX from the elements present: 6 if any f-block, 4 if any d, else 2."""
        try:
            from pymatgen.core import Element
            blocks = {Element(e).block for e in self.elements}
            if 'f' in blocks:
                return 6
            if 'd' in blocks:
                return 4
        except Exception:
            pass
        return 2

    def generate_scf_input(self, output_dir: str, from_relax: str = None):
        """Generate input files for self-consistent calculation"""
        os.makedirs(output_dir, exist_ok=True)

        # INCAR — set NBANDS up front so the WAVECAR is LOBSTER-ready.
        # The project POTCAR is built one level up before steps are generated.
        potcar_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), 'POTCAR')
        nbands = self._lobster_nbands(potcar_path, self._ncore_for('scf'))
        incar_content = self._generate_incar_scf(nbands=nbands)
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
        
        # KPOINTS — None triggers auto-detection from structure
        kpath = self.instructions.get('kpath')   # None if not specified
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

    def generate_lobster_input(self, output_dir: str, from_scf: str):
        """Generate the LOBSTER step (08_lobster).

        A symmetry-off (ISYM=0) NSCF that reads 02_scf's CHGCAR and writes a
        WAVECAR LOBSTER can consume, followed by a LOBSTER run. NBANDS is set
        >= the number of LOBSTER basis functions and LMAXMIX from the elements.
        """
        os.makedirs(output_dir, exist_ok=True)

        # NBANDS from the project-level POTCAR (built one level up before steps).
        potcar_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), 'POTCAR')
        nbands = self._lobster_nbands(potcar_path, self._ncore_for('scf'))

        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(self._generate_incar_lobster(nbands=nbands, lmax=self._lmaxmix()))

        # Same k-mesh density as the SCF.
        with open(f"{output_dir}/KPOINTS", 'w') as f:
            f.write(self._generate_kpoints_auto(density='high'))

        # Pull the converged charge density from the SCF at runtime.
        rel_scf = os.path.relpath(from_scf, output_dir)
        with open(f"{output_dir}/copy_from_scf.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write(f'SCF_DIR="$HERE/{rel_scf}"\n')
            f.write('cp "$SCF_DIR/CHGCAR" "$HERE/"\n')
            f.write('echo "  CHGCAR copied from 02_scf"\n')
            self._write_copy_if_newer(f, '$SCF_DIR', 'POSCAR', 'POSCAR', '02_scf')
        os.chmod(f"{output_dir}/copy_from_scf.sh", 0o755)

        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(self._generate_job_script_lobster(self._get_vasp_exec()))
        os.chmod(f"{output_dir}/run.sh", 0o755)

    def generate_wannier_input(self, output_dir: str, from_scf: str):
        """Generate input files for the VASP → Wannier90 interface (NSCF step)."""
        os.makedirs(output_dir, exist_ok=True)

        wannier_info = self.instructions.get('wannier', {})
        if not isinstance(wannier_info, dict):
            wannier_info = {}
        num_wann  = wannier_info.get('num_wann') or 8
        num_bands = max(num_wann + 8, num_wann * 2)   # sensible default; user should edit

        # k-mesh: use same density as SCF ('high' = 14 in-plane)
        is_2d  = self.instructions.get('is_2d', False)
        n_xy   = 14
        n_z    = 1 if is_2d else n_xy

        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(self._generate_incar_wannier(num_bands))

        with open(f"{output_dir}/KPOINTS", 'w') as f:
            f.write(self._generate_kpoints_auto(density='high'))

        with open(f"{output_dir}/wannier90.win", 'w') as f:
            f.write(self._generate_wannier90_win(num_wann, num_bands, n_xy, n_z, wannier_info))

        rel_scf = os.path.relpath(from_scf, output_dir)
        with open(f"{output_dir}/copy_from_scf.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f'HERE="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write(f'SCF_DIR="$HERE/{rel_scf}"\n')
            f.write('cp "$SCF_DIR/CHGCAR" "$HERE/"\n')
            f.write('echo "  CHGCAR copied from 02_scf"\n')
            self._write_copy_if_newer(f, '$SCF_DIR', 'POSCAR', 'POSCAR', '02_scf')
        os.chmod(f"{output_dir}/copy_from_scf.sh", 0o755)

        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(self._generate_job_script_wannier())
        os.chmod(f"{output_dir}/run.sh", 0o755)

    def _generate_incar_wannier(self, num_bands: int = 16) -> str:
        """Generate INCAR for VASP–Wannier90 interface (NSCF)."""
        encut = self._encut()
        lines = [
            "# VASP-Wannier90 interface (NSCF)",
            "SYSTEM = " + self.instructions.get('project_name', 'Wannier'),
            "",
            "# Electronic structure",
            "PREC = Accurate",
            f"ENCUT = {encut}",
            "EDIFF = 1E-8",
            "NELM = 100",
            f"NBANDS = {num_bands}   ! must match num_bands in wannier90.win",
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
            "# Wannier90 interface (VASP 6)",
            "LWANNIER90 = .TRUE.",
            "LWRITE_MMN_AMN = .TRUE.",
            "LWRITE_UNK = .FALSE.   ! set .TRUE. if you need real-space WFs",
            "",
        ]
        fl = self._functional_lines()
        if fl: lines += fl + [""]
        lines += self._mag_lines()
        lines += self._soc_lines()
        lines += self._u_lines()
        lines.extend(self._get_parallel_lines('bands'))  # KPAR=1 for Wannier
        lines.extend(["# Output", "LWAVE = .FALSE.", "LCHARG = .FALSE.", "LORBIT = 11"])
        lines = self._apply_incar_overrides(lines, 'wannier')
        return '\n'.join(lines) + '\n'

    def _parse_poscar_geometry(self) -> str:
        """Parse POSCAR and return unit_cell_cart + atoms_frac blocks for wannier90.win."""
        with open(self.poscar) as f:
            lines = f.readlines()
        scale = float(lines[1].strip())
        latt = [[float(x) * scale for x in lines[i].split()[:3]] for i in range(2, 5)]
        elems  = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        ctype  = lines[7].strip()[0].upper()   # D = Direct/fractional, C = Cartesian
        atoms, idx = [], 8
        for el, n in zip(elems, counts):
            for _ in range(n):
                xyz = [float(x) for x in lines[idx].split()[:3]]
                atoms.append((el, xyz))
                idx += 1

        out = ['\nbegin unit_cell_cart', 'Angstrom']
        for v in latt:
            out.append(f'  {v[0]:14.8f}  {v[1]:14.8f}  {v[2]:14.8f}')
        out.append('end unit_cell_cart')

        tag = 'atoms_frac' if ctype == 'D' else 'atoms_cart'
        out.append(f'\nbegin {tag}')
        if ctype == 'C':
            out.append('Angstrom')
        for el, xyz in atoms:
            out.append(f'  {el}  {xyz[0]:14.8f}  {xyz[1]:14.8f}  {xyz[2]:14.8f}')
        out.append(f'end {tag}')
        return '\n'.join(out) + '\n'

    def _kpoints_block_for_wannier(self, n_xy: int, n_z: int) -> str:
        """Generate begin kpoints ... end kpoints block for wannier90.win.

        Lists all k-points of the Gamma-centered Monkhorst-Pack mesh explicitly.
        This is required by wannier90.x -pp (VASP 5 file-based workflow).
        The ordering (k3 fastest) matches VASP's internal k-point ordering.
        """
        kpts = []
        for i in range(n_xy):
            for j in range(n_xy):
                for k in range(n_z):
                    kpts.append(f'  {i/n_xy:.8f}  {j/n_xy:.8f}  {k/n_z if n_z > 1 else 0.0:.8f}')
        lines = ['\nbegin kpoints'] + kpts + ['end kpoints']
        return '\n'.join(lines)

    def _generate_wannier90_win(self, num_wann: int, num_bands: int,
                                 n_xy: int, n_z: int,
                                 wannier_info: dict) -> str:
        """Generate wannier90.win for VASP 5 file-based interface.

        For VASP 5: wannier90.x -pp needs unit_cell_cart and atoms blocks
        present BEFORE running. VASP then reads wannier90.nnkp (written by -pp)
        and appends a kpoints block to wannier90.win after running.
        """
        projections = wannier_info.get('projections', [])
        dis_win     = wannier_info.get('dis_win', '')

        if projections:
            proj_block = '\n'.join(f'  {p}' for p in projections)
        else:
            el_list = self.elements or ['X']
            proj_block = '\n'.join(f'  {el} : sp3' for el in el_list)
            proj_block += '\n  ! edit: set correct angular-momentum projections'

        lines = [
            f'num_wann  = {num_wann}',
            f'num_bands = {num_bands}   ! must equal NBANDS in INCAR',
            '',
            '! K-point mesh — must match the KPOINTS file',
            f'mp_grid : {n_xy} {n_xy} {n_z}',
            '',
            '! Energy windows (eV, relative to Fermi level)',
        ]
        if dis_win:
            parts = dis_win.replace(':', ' ').split()
            if len(parts) >= 2:
                lines += [f'dis_win_min  = {parts[0]}',
                          f'dis_win_max  = {parts[1]}',
                          f'! dis_froz_min = {parts[0]}   ! uncomment to set inner (frozen) window',
                          f'! dis_froz_max = {parts[1]}']
        else:
            lines += [
                '! dis_win_min  = -5.0   ! edit: lower bound of outer disentanglement window',
                '! dis_win_max  = 10.0   ! edit: upper bound (must include all num_bands)',
                '! dis_froz_min = -5.0   ! edit: lower bound of inner (frozen) window',
                '! dis_froz_max =  6.0   ! edit: upper bound of frozen window',
            ]
        lines += [
            '',
            'begin projections',
            proj_block,
            'end projections',
            '',
            '! Plotting (uncomment to generate cube files for visualisation)',
            '! wannier_plot = .true.',
            '! wannier_plot_supercell = 3',
        ]

        # Geometry + kpoints — required by wannier90.x -pp (VASP 5 workflow)
        try:
            lines.append(self._parse_poscar_geometry())
        except Exception as e:
            lines.append(f'\n! WARNING: could not read geometry from POSCAR: {e}')
            lines.append('! Add unit_cell_cart and atoms_frac blocks manually before running -pp')

        lines.append(self._kpoints_block_for_wannier(n_xy, n_z))

        return '\n'.join(lines) + '\n'

    def _generate_job_script_wannier(self) -> str:
        """Generate run.sh for the Wannier90 interface step (VASP 5 file-based workflow)."""
        vasp_exec  = self._get_vasp_exec()
        launch_cmd = self._get_mpi_cmd(vasp_exec)
        w90        = self._profile_get('wannier90_x', _WANNIER90_X)
        preamble   = self._run_sh_preamble('vasp_wannier')
        return f"""#!/bin/bash
{preamble}# run.sh — VASP 5 Wannier90 interface (file-based workflow)
#
# Workflow:
#   1. wannier90.x -pp  — reads wannier90.win, writes wannier90.nnkp
#   2. VASP NSCF        — reads wannier90.nnkp, writes .mmn .amn .eig,
#                         appends geometry to wannier90.win
#   3. wannier90.x      — reads all files, computes MLWFs
#
# IMPORTANT: before running, edit wannier90.win:
#   - Set correct projections for your system
#   - Set dis_win_min/max to cover your bands of interest
#   - num_bands must equal NBANDS in INCAR

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$HERE/POTCAR" ]; then
    echo "ERROR: POTCAR not found in $HERE"; exit 1
fi

if [ -f "$HERE/copy_from_scf.sh" ]; then
    bash "$HERE/copy_from_scf.sh"
fi

W90={w90}
if ! command -v "$W90" &>/dev/null; then
    echo "ERROR: wannier90.x not found: $W90"
    echo "  Set WANNIER90_X in site.env or install: conda install -c conda-forge wannier90"
    exit 1
fi

# Step 1: wannier90 -pp — generates wannier90.nnkp (required by VASP)
echo "Step 1: wannier90.x -pp (generating .nnkp) ..."
cd "$HERE"
"$W90" -pp wannier90 2>&1 | tee wannier90_pp.out
if [ ! -f "$HERE/wannier90.nnkp" ]; then
    echo "ERROR: wannier90.nnkp not generated — check wannier90.win"
    exit 1
fi
echo "  OK: wannier90.nnkp written"

# Step 2: VASP NSCF — reads .nnkp, writes .mmn .amn .eig
echo "Step 2: VASP NSCF at $(date)"
{launch_cmd} > vasp.out 2>&1
echo "  VASP done — checking output ..."
grep -i "WANNIER\\|Routine DWANN" vasp.out | tail -3 || \
    echo "  (no WANNIER lines found in vasp.out — check LWANNIER90 in INCAR)"

# Step 3: wannier90.x — compute maximally-localised WFs
echo "Step 3: wannier90.x (Wannierization) ..."
"$W90" wannier90 2>&1 | tee wannier90.out
if grep -q "Final State" wannier90.out 2>/dev/null; then
    echo "  OK: Wannierization converged"
    grep "WF centre and spread" wannier90.out | tail -5
else
    echo "  WARNING: check wannier90.out for convergence"
fi
"""

    # ── DFPT ─────────────────────────────────────────────────────────────────

    def generate_dfpt_input(self, output_dir: str, from_scf: str):
        """Generate VASP DFPT input: Born effective charges + dielectric tensor."""
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(self._generate_incar_dfpt())

        with open(f"{output_dir}/KPOINTS", 'w') as f:
            f.write(self._generate_kpoints_auto(density='high'))

        rel_scf = os.path.relpath(from_scf, output_dir)
        with open(f"{output_dir}/copy_from_scf.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f'HERE="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write(f'SCF_DIR="$HERE/{rel_scf}"\n')
            for fname in ('WAVECAR', 'CHGCAR'):
                f.write(f'cp "$SCF_DIR/{fname}" "$HERE/" 2>/dev/null '
                        f'&& echo "  {fname} copied from 02_scf" '
                        f'|| echo "  WARNING: {fname} not found in 02_scf"\n')
            self._write_copy_if_newer(f, '$SCF_DIR', 'POSCAR', 'POSCAR', '02_scf')
        os.chmod(f"{output_dir}/copy_from_scf.sh", 0o755)

        with open(f"{output_dir}/extract_born.py", 'w') as f:
            f.write(self._extract_born_script())

        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(self._generate_job_script_dfpt())
        os.chmod(f"{output_dir}/run.sh", 0o755)

    def _generate_incar_dfpt(self) -> str:
        dfpt_info = self.instructions.get('dfpt', {})
        ediff = (dfpt_info or {}).get('ediff', '1E-8')
        encut = self._encut()
        lines = [
            "# VASP DFPT — Born effective charges + static dielectric tensor",
            "SYSTEM = " + self.instructions.get('project_name', 'DFPT'),
            "",
            "PREC   = Accurate",
            f"ENCUT  = {encut}",
            f"EDIFF  = {ediff}",
            "NELM   = 100",
            "",
            "# DFPT linear-response",
            "IBRION = 8      ! density-functional perturbation theory",
            "NSW    = 1",
            "POTIM  = 0",
            "",
            "LEPSILON = .TRUE.    ! Born charges + macroscopic dielectric tensor",
            "LRPA     = .FALSE.   ! include local-field effects (use .TRUE. for RPA)",
            "",
            "ICHARG = 1           ! read converged CHGCAR from SCF",
            "",
            "ISMEAR = 0",
            "SIGMA  = 0.01",
            "",
        ]
        fl = self._functional_lines()
        if fl: lines += fl + [""]
        lines += self._mag_lines()
        lines += self._soc_lines()
        lines += self._u_lines()
        lines.extend(self._get_parallel_lines('dfpt'))
        lines.extend(["LWAVE  = .FALSE.", "LCHARG = .FALSE."])
        lines = self._apply_incar_overrides(lines, 'dfpt')
        return '\n'.join(lines) + '\n'

    def _extract_born_script(self) -> str:
        """Python script to parse DFPT OUTCAR → phonopy BORN file + human-readable summary."""
        return r'''#!/usr/bin/env python3
"""extract_born.py — Extract Born charges and dielectric tensor from VASP DFPT OUTCAR.
Writes BORN (phonopy format) and born_charges.txt (human-readable).
Usage: python3 extract_born.py [OUTCAR]
"""
import sys, re

outcar = sys.argv[1] if len(sys.argv) > 1 else 'OUTCAR'
try:
    text = open(outcar).read()
except FileNotFoundError:
    print(f"ERROR: {outcar} not found"); sys.exit(1)

# Macroscopic static dielectric tensor
eps_m = re.search(
    r'MACROSCOPIC STATIC DIELECTRIC TENSOR.*?\n\s*[-]+\s*\n'
    r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
    r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
    r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
    text, re.DOTALL)
if not eps_m:
    print("ERROR: dielectric tensor not found — did LEPSILON=.TRUE. run?"); sys.exit(1)
eps = [[float(x) for x in eps_m.groups()[i*3:(i+1)*3]] for i in range(3)]

# Born effective charges — take last occurrence (cumulative output)
born_blocks = list(re.finditer(r'BORN EFFECTIVE CHARGES.*?(?=BORN EFFECTIVE CHARGES|\Z)', text, re.DOTALL))
if not born_blocks:
    print("ERROR: Born charges not found in OUTCAR"); sys.exit(1)
ions = re.findall(
    r'ion\s+\d+\s*\n'
    r'\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
    r'\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
    r'\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
    born_blocks[-1].group())
if not ions:
    print("ERROR: could not parse Born charge blocks"); sys.exit(1)
born = [[[float(x) for x in m[i*3:(i+1)*3]] for i in range(3)] for m in ions]

# Write BORN (phonopy format)
with open('BORN', 'w') as f:
    f.write("# Born effective charges and dielectric tensor from VASP DFPT OUTCAR\n")
    f.write("14.400\n")
    for row in eps:
        f.write("  " + "  ".join(f"{x:12.6f}" for x in row) + "\n")
    for z in born:
        for row in z:
            f.write("  " + "  ".join(f"{x:12.6f}" for x in row) + "\n")
print(f"Written BORN ({len(born)} atoms)")

# Human-readable summary
with open('born_charges.txt', 'w') as f:
    f.write("Born Effective Charges and Dielectric Tensor\n")
    f.write("=" * 50 + "\n\n")
    f.write("Macroscopic static dielectric tensor:\n")
    for row in eps:
        f.write("  " + "  ".join(f"{x:10.5f}" for x in row) + "\n")
    f.write("\nBorn effective charge tensors (Z*):\n")
    for i, z in enumerate(born):
        f.write(f"\n  Ion {i+1}:\n")
        for row in z:
            f.write("    " + "  ".join(f"{x:10.5f}" for x in row) + "\n")
print("Written born_charges.txt")
print(f"\nDielectric tensor (diagonal): {eps[0][0]:.4f}  {eps[1][1]:.4f}  {eps[2][2]:.4f}")
print("Born charges (diagonal elements):")
for i, z in enumerate(born):
    print(f"  Ion {i+1}: Zxx={z[0][0]:8.4f}  Zyy={z[1][1]:8.4f}  Zzz={z[2][2]:8.4f}")
'''

    def _generate_job_script_dfpt(self) -> str:
        vasp_exec  = self._get_vasp_exec()
        launch_cmd = self._get_mpi_cmd(vasp_exec)
        preamble   = self._run_sh_preamble('vasp_dfpt')
        return f"""#!/bin/bash
{preamble}# run.sh — VASP DFPT: Born effective charges + dielectric tensor
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [ ! -f "$HERE/POTCAR" ]; then echo "ERROR: POTCAR not found"; exit 1; fi
[ -f "$HERE/copy_from_scf.sh" ] && bash "$HERE/copy_from_scf.sh"

echo "Starting DFPT (LEPSILON) at $(date)"
{launch_cmd} > vasp.out 2>&1

grep -q "reached required accuracy" vasp.out 2>/dev/null \\
    && echo "  OK: converged" \\
    || echo "  WARNING: may not have converged — check vasp.out"

# Extract Born charges → BORN file for phonopy NAC correction
if [ -f "$HERE/OUTCAR" ]; then
    echo "Extracting Born charges ..."
    python3 "$HERE/extract_born.py" "$HERE/OUTCAR"
fi
echo "DFPT done. See born_charges.txt and BORN (phonopy NAC format)."
"""

    # ── Phonons (phonopy) ─────────────────────────────────────────────────────

    def generate_phonons_input(self, output_dir: str, from_scf: str, from_dfpt: str = None):
        """Generate phonopy + VASP phonon spectrum calculation."""
        os.makedirs(output_dir, exist_ok=True)

        phon_info = self.instructions.get('phonons', {})
        if not isinstance(phon_info, dict):
            phon_info = {}
        dim  = phon_info.get('dim',  '2 2 2')
        band = phon_info.get('band', '')
        mesh = phon_info.get('mesh', '20 20 20')
        disp = phon_info.get('disp', 0.01)
        nac  = phon_info.get('nac',  True)

        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(self._generate_incar_phonons())

        # Low k-mesh — supercells are large, Gamma-only is often sufficient
        with open(f"{output_dir}/KPOINTS", 'w') as f:
            f.write(self._generate_kpoints_auto(density='low'))

        # POSCAR copied from SCF (primitive cell for phonopy)
        rel_scf = os.path.relpath(from_scf, output_dir)
        with open(f"{output_dir}/copy_from_scf.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f'HERE="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write(f'SCF_DIR="$HERE/{rel_scf}"\n')
            self._write_copy_if_newer(f, '$SCF_DIR', 'POSCAR', 'POSCAR', '02_scf')
        os.chmod(f"{output_dir}/copy_from_scf.sh", 0o755)

        with open(f"{output_dir}/band.conf", 'w') as f:
            f.write(self._generate_phonopy_band_conf(dim, band, nac))

        with open(f"{output_dir}/mesh.conf", 'w') as f:
            f.write(self._generate_phonopy_mesh_conf(dim, mesh, nac))

        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(self._generate_job_script_phonons(dim, disp, from_dfpt, output_dir))
        os.chmod(f"{output_dir}/run.sh", 0o755)

    def _generate_incar_phonons(self) -> str:
        """INCAR for VASP single-point force calculations on displaced supercells."""
        encut = self._encut()
        lines = [
            "# VASP force calculation — phonopy displaced supercell",
            "SYSTEM = " + self.instructions.get('project_name', 'Phonons'),
            "",
            "PREC   = Accurate",
            f"ENCUT  = {encut}",
            "EDIFF  = 1E-8      ! tight — essential for accurate force constants",
            "NELM   = 100",
            "",
            "IBRION = -1        ! single-point; no ionic relaxation",
            "NSW    = 0",
            "",
            "ISMEAR = 0",
            "SIGMA  = 0.01",
            "",
            "LREAL  = .FALSE.   ! reciprocal-space projectors (more accurate for small cells)",
            "ADDGRID = .TRUE.",
            "",
        ]
        fl = self._functional_lines()
        if fl: lines += fl + [""]
        lines += self._mag_lines()
        lines += self._soc_lines()
        lines += self._u_lines()
        lines.extend(self._get_parallel_lines('scf'))
        lines.extend(["LWAVE  = .FALSE.", "LCHARG = .FALSE."])
        lines = self._apply_incar_overrides(lines, 'phonons')
        return '\n'.join(lines) + '\n'

    def _generate_phonopy_band_conf(self, dim: str, band: str, nac: bool) -> str:
        if not band:
            kpath = self.instructions.get('kpath', [])
            # Map common labels to fractional coordinates
            sym_pts = {
                'G': '0 0 0', 'X': '0.5 0 0', 'M': '0.5 0.5 0',
                'K': '0.333 0.333 0', 'L': '0.5 0.5 0.5',
                'A': '0 0 0.5',   'H': '0.333 0.333 0.5',
                'R': '0.5 0.5 0.5', 'W': '0.5 0.25 0.75',
            }
            pts = [sym_pts.get(k.upper(), '0 0 0') for k in (kpath or ['G', 'X', 'M', 'G'])]
            band = '  '.join(pts)
        lines = [
            "# band.conf — phonon band structure",
            "# Edit BAND path for your crystal structure (fractional reciprocal coords)",
            f"DIM = {dim}",
            f"BAND = {band}",
            "BAND_POINTS = 101",
            "BAND_LABELS = auto",
        ]
        if nac:
            lines.append("NAC = .TRUE.      ! non-analytic correction (LO-TO splitting)")
            lines.append("# NAC requires BORN file from 06_dfpt — run DFPT step first")
        return '\n'.join(lines) + '\n'

    def _generate_phonopy_mesh_conf(self, dim: str, mesh: str, nac: bool) -> str:
        lines = [
            "# mesh.conf — phonon DOS",
            "# Edit MP mesh for better DOS resolution",
            f"DIM = {dim}",
            f"MP = {mesh}",
            "PDOS = Auto",
            "GAMMA_CENTER = .TRUE.",
        ]
        if nac:
            lines.append("NAC = .TRUE.")
        return '\n'.join(lines) + '\n'

    def _generate_job_script_phonons(self, dim: str, disp: float,
                                      from_dfpt, output_dir: str) -> str:
        vasp_exec  = self._get_vasp_exec()
        launch_cmd = self._get_mpi_cmd(vasp_exec)
        preamble   = self._run_sh_preamble('vasp_phonons')
        if from_dfpt:
            rel_dfpt = os.path.relpath(from_dfpt, output_dir)
            born_block = f"""
# Copy BORN file from DFPT step for NAC correction
BORN_SRC="$HERE/{rel_dfpt}/BORN"
if [ -f "$BORN_SRC" ]; then
    cp "$BORN_SRC" "$HERE/BORN"
    echo "  BORN copied from 06_dfpt (NAC enabled)"
else
    echo "  NOTE: no BORN in 06_dfpt — LO-TO splitting disabled"
    echo "        Run 06_dfpt first, or set NAC = .FALSE. in band.conf/mesh.conf"
fi
"""
        else:
            born_block = ""

        return f"""#!/bin/bash
{preamble}# run.sh — Phonon spectrum: phonopy + VASP force calculations
#
# Steps:
#   1. Generate displaced supercells (phonopy -d)
#   2. Run VASP single-point on each displaced supercell
#   3. Collect forces  (phonopy -f)
#   4. Phonon band structure (band.conf)
#   5. Phonon DOS         (mesh.conf)
#
# Edit band.conf (BAND path) and mesh.conf (MP mesh) before running.
# For 2D: set DIM = 2 2 1 and KPOINTS to a dense 2D mesh.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

command -v phonopy &>/dev/null || {{ echo "ERROR: phonopy not found — conda install -c conda-forge phonopy"; exit 1; }}
[ -f "$HERE/POTCAR" ] || {{ echo "ERROR: POTCAR not found"; exit 1; }}

[ -f "$HERE/copy_from_scf.sh" ] && bash "$HERE/copy_from_scf.sh"
{born_block}
# ── Step 1: Generate displaced supercells ────────────────────────────────
echo "Step 1: phonopy displacement (DIM={dim}, amplitude={disp} Ang) ..."
phonopy -d --dim="{dim}" --vasp --amplitude={disp}
n_disp=$(ls POSCAR-* 2>/dev/null | wc -l | tr -d ' ')
[ "$n_disp" -eq 0 ] && {{ echo "ERROR: no displaced POSCARs generated"; exit 1; }}
echo "  $n_disp displaced supercell(s) generated"

# ── Step 2: VASP force calculations ─────────────────────────────────────
echo "Step 2: VASP force calculations ..."
for poscar in POSCAR-*; do
    n=${{poscar#POSCAR-}}
    echo "  DISP-$n ..."
    mkdir -p "DISP-$n"
    cp "$poscar" "DISP-$n/POSCAR"
    cp INCAR KPOINTS POTCAR "DISP-$n/"
    cd "DISP-$n"
    {launch_cmd} > vasp.out 2>&1
    grep -q "reached required accuracy" vasp.out 2>/dev/null \\
        && echo "    converged" \\
        || echo "    WARNING: check vasp.out"
    cd "$HERE"
done

# ── Step 3: Collect forces ────────────────────────────────────────────────
echo "Step 3: phonopy -f (collecting forces) ..."
phonopy -f DISP-*/vasprun.xml

# ── Step 4: Phonon band structure ─────────────────────────────────────────
echo "Step 4: Phonon band structure ..."
MPLBACKEND=Agg phonopy -p -s band.conf 2>&1 | tail -3
for ext in png svg pdf; do
    [ -f "band.$ext" ] && cp "band.$ext" "phonon_band.$ext"
done

# ── Step 5: Phonon DOS ────────────────────────────────────────────────────
echo "Step 5: Phonon DOS ..."
MPLBACKEND=Agg phonopy -p -s mesh.conf 2>&1 | tail -3
for ext in png svg pdf; do
    [ -f "mesh.$ext" ] && cp "mesh.$ext" "phonon_dos.$ext"
done

echo ""
echo "Done. Plots: phonon_band.png/svg  phonon_dos.png/svg"
echo "      Data:  band.yaml  FORCE_SETS"
"""

    # ─────────────────────────────────────────────────────────────────────────

    def _get_parallel_lines(self, calc_type: str = 'scf') -> list:
        """Return INCAR lines for MPI parallelization.

        Smart defaults scale with mpi_np (N = total MPI tasks):

          relax / scf / dos  →  KPAR = floor(sqrt(N)), NCORE = N // KPAR
          bands              →  KPAR = 1  (line-mode k-points), NCORE = N
          dfpt               →  KPAR = 1  (IBRION=8 requires KPAR=1), NCORE = N
          phonons            →  KPAR = 1, NCORE = floor(sqrt(N))

        Per-task overrides in instructions.txt (e.g. SCF_KPAR, DOS_NCORE)
        take priority over the global KPAR / NCORE keys, which in turn
        take priority over the computed defaults.
        """
        import math
        np = self.instructions.get('mpi_np', 1) or 1
        if np <= 1:
            return []

        soc = self.instructions.get('soc', False)
        kpar_sqrt = max(1, int(math.floor(math.sqrt(np))))

        # Per-type auto defaults
        defaults = {
            'relax':   (kpar_sqrt,  max(1, np // kpar_sqrt)),
            'scf':     (kpar_sqrt,  max(1, np // kpar_sqrt)),
            'bands':   (1,          np),
            'dos':     (kpar_sqrt,  max(1, np // kpar_sqrt)),
            'dfpt':    (1,          np),      # KPAR=1 required for IBRION=8
            'phonons': (1,          max(1, int(math.floor(math.sqrt(np))))),
        }
        if soc:
            # NCL doubles memory per band — cap NCORE to avoid OOM
            for k in defaults:
                kd, nd = defaults[k]
                defaults[k] = (kd, max(1, nd // 2))
            defaults['bands'] = (1, max(1, np // 2))

        kpar_def, ncore_def = defaults.get(calc_type, (kpar_sqrt, max(1, np // kpar_sqrt)))

        # Priority: per-task key > global key > auto default
        kpar  = (self.instructions.get(f'{calc_type}_kpar')
                 or self.instructions.get('kpar')
                 or kpar_def)
        ncore = (self.instructions.get(f'{calc_type}_ncore')
                 or self.instructions.get('ncore')
                 or ncore_def)

        # Safety: KPAR * NCORE must not exceed np
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

    @staticmethod
    def _incar_tag_name(line: str):
        """Return the upper-cased INCAR tag name from a line, or None.

        'EDIFF = 1E-6   ! comment'  ->  'EDIFF'
        '# heading'                  ->  None
        """
        s = line.strip()
        if not s or s.startswith('#') or s.startswith('!') or '=' not in s:
            return None
        return s.split('=', 1)[0].strip().upper()

    def _apply_incar_overrides(self, lines: list, step: str) -> list:
        """Merge user-supplied raw INCAR tags into a generated INCAR line list.

        Tags from the global ('all') block plus the per-step block are applied,
        with the per-step block winning on conflicts and both winning over the
        generated defaults. A tag that matches an existing generated line
        replaces it in place; any remaining tags are appended under a clearly
        labelled section so nothing is silently dropped.
        """
        raw = self.instructions.get('incar_raw', {}) or {}
        overrides = list(raw.get('all', [])) + list(raw.get(step, []))
        if not overrides:
            return lines

        # Later entries win (per-step block listed after the global one).
        ov_map, ordered = {}, []
        for o in overrides:
            tag = self._incar_tag_name(o)
            if tag is None:
                continue
            if tag not in ov_map:
                ordered.append(tag)
            ov_map[tag] = o.strip()

        used, new_lines = set(), []
        for line in lines:
            tag = self._incar_tag_name(line)
            if tag and tag in ov_map:
                new_lines.append(ov_map[tag])
                used.add(tag)
            else:
                new_lines.append(line)

        extra = [ov_map[t] for t in ordered if t not in used]
        if extra:
            new_lines += ["", "# User INCAR overrides (from instructions file)"] + extra
        return new_lines

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

    def _classify_bravais(self) -> str:
        """Classify the Bravais lattice type from POSCAR lattice vectors.
        Returns a key into _KPATH_LIBRARY: cF, cI, cP, hP, tP, oP, rP, mP.
        Falls back to 'oP' on any error.
        """
        try:
            lines = open(self.poscar).readlines()
            scale = abs(float(lines[1].strip()))
            lv    = np.array([[float(x) for x in lines[2+i].split()[:3]]
                               for i in range(3)]) * scale

            a, b, c = [float(np.linalg.norm(lv[i])) for i in range(3)]
            cos_a = np.dot(lv[1], lv[2]) / (b * c)
            cos_b = np.dot(lv[0], lv[2]) / (a * c)
            cos_g = np.dot(lv[0], lv[1]) / (a * b)
            alpha = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
            beta  = float(np.degrees(np.arccos(np.clip(cos_b, -1.0, 1.0))))
            gamma = float(np.degrees(np.arccos(np.clip(cos_g, -1.0, 1.0))))

            tl = max(a, b, c) * 0.03   # 3 % relative length tolerance
            ta = 2.0                     # 2 ° angle tolerance

            leq = lambda x, y: abs(x - y) < tl
            aeq = lambda x, y: abs(x - y) < ta

            abc   = leq(a, b) and leq(b, c)
            ab_eq = leq(a, b)
            right = aeq(alpha, 90) and aeq(beta, 90) and aeq(gamma, 90)

            # Primitive FCC: a=b=c, all angles ≈ 60°
            if abc and aeq(alpha, 60) and aeq(beta, 60) and aeq(gamma, 60):
                return 'cF'
            # Primitive BCC: a=b=c, all angles ≈ 109.47°
            if abc and aeq(alpha, 109.47) and aeq(beta, 109.47) and aeq(gamma, 109.47):
                return 'cI'
            # Rhombohedral: a=b=c, all angles equal but ≠ 90°
            if abc and aeq(alpha, beta) and aeq(beta, gamma) and not right:
                return 'rP'
            # Hexagonal: a≈b, α=β=90°, γ=120°
            if ab_eq and aeq(alpha, 90) and aeq(beta, 90) and aeq(gamma, 120):
                return 'hP'
            # Conventional cubic (a=b=c, all 90°): use SC path regardless of
            # FCC/BCC content — primitive cells are preferred for unfolded bands
            if abc and right:
                return 'cP'
            # Tetragonal: a≈b≠c, all 90°
            if ab_eq and right:
                return 'tP'
            # Orthorhombic: all 90°
            if right:
                return 'oP'
            # Monoclinic: α=γ=90°, β≠90°
            if aeq(alpha, 90) and aeq(gamma, 90):
                return 'mP'
            # Triclinic → safe fallback
            return 'oP'
        except Exception:
            return 'oP'

    def _read_atom_count(self) -> int:
        """Total number of atoms from POSCAR line 6 (element counts)."""
        try:
            with open(self.poscar) as f:
                lines = f.readlines()
            return sum(int(x) for x in lines[6].split())
        except Exception:
            return max(len(self.elements), 1)

    def _mag_lines(self) -> list:
        mag = self.instructions.get('magnetization', {})
        if not mag.get('enabled') or self.instructions.get('soc', False):
            return []
        n      = self._read_atom_count()
        moment = self.instructions.get('mag_moment', 2.0)
        return ["# Collinear magnetization", "ISPIN = 2",
                f"MAGMOM = {n}*{moment}", ""]

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
        """Generate INCAR for relaxation.

        If an external pressure is requested (PRESSURE in the instructions file)
        the step becomes a constant-pressure relaxation: ISIF=3 (relax cell
        shape + volume + ions) and IBRION=2, with the target pressure imposed
        via PSTRESS (kBar).
        """
        encut    = self._encut()
        nsw      = self.instructions.get('nsw')    or 100
        ediffg   = self.instructions.get('ediffg') or -0.01
        pressure = self.instructions.get('pressure', {}) or {}
        const_p  = pressure.get('enabled', False)

        # ISIF=3 (full cell relaxation) is the default and is required for a
        # meaningful constant-pressure run; an explicit user ISIF still wins.
        isif = self.instructions.get('isif') or 3

        header = "# Relaxation calculation"
        if const_p:
            header = (f"# Constant-pressure relaxation @ {pressure['value']} "
                      f"{pressure['unit']}  (PSTRESS = {pressure['pstress_kbar']:g} kBar)")

        lines = [
            header,
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
        ]
        if const_p:
            lines += [
                f"PSTRESS = {pressure['pstress_kbar']:g}   "
                f"! external pressure in kBar ({pressure['value']} {pressure['unit']})",
            ]
        lines += [
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
        lines = self._apply_incar_overrides(lines, 'relax')
        return '\n'.join(lines) + '\n'

    def _generate_incar_scf(self, nbands: int = None) -> str:
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
        ]
        if nbands:
            lines.append(f"NBANDS = {nbands}   # >= number of LOBSTER local basis functions")
        lines += [
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
        lines = self._apply_incar_overrides(lines, 'scf')
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
        lines = self._apply_incar_overrides(lines, 'bands')
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
        lines = self._apply_incar_overrides(lines, 'dos')
        return '\n'.join(lines) + '\n'

    def _generate_incar_lobster(self, nbands: int = None, lmax: int = 2) -> str:
        """INCAR for the LOBSTER NSCF: ISYM=0, fixed charge, WAVECAR for LOBSTER."""
        encut = self._encut()
        lines = [
            "# LOBSTER preparation: symmetry-off NSCF from the SCF charge density",
            "SYSTEM = " + self.instructions.get('project_name', 'LOBSTER'),
            "",
            "# Electronic structure",
            "PREC = Accurate",
            f"ENCUT = {encut}",
            "EDIFF = 1E-6",
            "NELM = 100",
            "ALGO = Normal",
        ]
        if nbands:
            lines.append(f"NBANDS = {nbands}   # >= number of LOBSTER local basis functions")
        lines += [
            "",
            "# Non-self-consistent from SCF CHGCAR; symmetry OFF is REQUIRED by LOBSTER",
            "ICHARG = 11",
            "ISTART = 1",
            "ISYM = 0",
            "IBRION = -1",
            "NSW = 0",
            f"LMAXMIX = {lmax}",
            "",
            "# Smearing (Gaussian, matches LOBSTER's gaussianSmearingWidth)",
            "ISMEAR = 0",
            "SIGMA = 0.05",
            "",
        ]
        fl = self._functional_lines()
        if fl:
            lines += fl + [""]
        lines += self._mag_lines()
        lines += self._soc_lines()
        lines += self._u_lines()
        lines.extend(self._get_parallel_lines('scf'))
        lines.extend([
            "# Output (WAVECAR is read by LOBSTER; dense DOSCAR for the energy window)",
            "LORBIT = 11",
            "NEDOS = 3000",
            "LWAVE = .TRUE.",
            "LCHARG = .FALSE.",
        ])
        lines = self._apply_incar_overrides(lines, 'lobster')
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

        An explicit KMESH in the instructions (parsed to a 3-int list) overrides
        the tiered density for every automatic mesh in the workflow.
        """
        # Explicit user override (KMESH in instructions) wins over the tier.
        kmesh = self.instructions.get('kmesh')
        if kmesh:
            nx, ny, nz = kmesh
            return (
                f"Automatic Gamma mesh (user KMESH)\n"
                f"0\n"
                f"Gamma\n"
                f"  {nx}  {ny}  {nz}\n"
                f"  0    0    0\n"
            )

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
    
    def _generate_kpoints_linemode(self, kpath=None, npoints: int = 40) -> str:
        """Generate KPOINTS file for band structure.

        If kpath is None, auto-detects the Bravais lattice type from POSCAR
        and uses the standard high-symmetry path from _KPATH_LIBRARY.
        If kpath is a list of labels, uses those with the generic coord table
        (backward-compatible behaviour).
        """
        npoints = self.instructions.get('nkpts_bands') or npoints
        is_2d   = self.instructions.get('is_2d', False)

        if kpath is None:
            # ── auto-detect ──────────────────────────────────────────────
            bravais = self._classify_bravais()
            entry   = _KPATH_LIBRARY.get(bravais, _KPATH_LIBRARY['oP'])
            kpath   = entry.get('path_2d', entry['path']) if is_2d else entry['path']
            kcoords = entry['coords']
            comment = f"Line-mode KPOINTS  [{entry['name']} — auto-detected]"
        else:
            # ── user-specified labels, generic coord table ────────────────
            is_hex = self.instructions.get('is_hex', False)
            kcoords = {
                'G': [0.000, 0.000, 0.000],
                'Z': [0.000, 0.000, 0.500],
                'M': [0.500, 0.000, 0.000] if is_hex else [0.500, 0.500, 0.000],
                'K': [1/3,   1/3,   0.000],
                'H': [1/3,   1/3,   0.500],
                'A': [0.000, 0.000, 0.500],
                'L': [0.500, 0.000, 0.500],
                'X': [0.500, 0.000, 0.000],
                'Y': [0.000, 0.500, 0.000],
                'R': [0.500, 0.500, 0.500],
                'S': [0.500, 0.500, 0.000],
                'T': [0.000, 0.500, 0.500],
                'U': [0.500, 0.000, 0.500],
                # FCC-specific (primitive cell)
                'W': [0.500, 0.250, 0.750],
                'P': [0.250, 0.250, 0.250],
                'N': [0.000, 0.000, 0.500],
            }
            comment = "Line-mode KPOINTS"

        out = [f"{comment}\n", f"{npoints}\n", "Line-mode\n", "rec\n"]
        for i in range(len(kpath) - 1):
            s_lbl, e_lbl = kpath[i], kpath[i + 1]
            s = kcoords.get(s_lbl, [0.0, 0.0, 0.0])
            e = kcoords.get(e_lbl, [0.0, 0.0, 0.0])
            out.append(f"  {s[0]:8.4f} {s[1]:8.4f} {s[2]:8.4f}  ! {s_lbl}\n")
            out.append(f"  {e[0]:8.4f} {e[1]:8.4f} {e[2]:8.4f}  ! {e_lbl}\n")
            out.append("\n")
        return ''.join(out)
    
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
        preamble   = self._run_sh_preamble(f'vasp_{calc_type}')

        script = f"""#!/bin/bash
{preamble}# run.sh for {calc_type}
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

    def _generate_job_script_lobster(self, vasp_exec: str) -> str:
        """run.sh for 08_lobster: symmetry-off NSCF, then a LOBSTER run.

        Built with token replacement (not an f-string) so the embedded shell
        '$' expansions, heredoc, and awk '{...}' survive verbatim.
        """
        launch_cmd = self._get_mpi_cmd(vasp_exec)
        lobster_bin = self._get_lobster_exec()
        preamble = self._run_sh_preamble('vasp_lobster')
        template = r'''#!/bin/bash
@PREAMBLE@# run.sh for lobster (symmetry-off NSCF + LOBSTER)
# VASP binary : @VEXEC@
# Launch      : @LAUNCH@
# LOBSTER bin : @LOBSTER@   (override at runtime with $LOBSTER_BIN)

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$HERE/POTCAR" ]; then
    echo "ERROR: POTCAR not found in $HERE"; exit 1
fi

# Pull the converged charge density from 02_scf
if [ -f "$HERE/copy_from_scf.sh" ]; then
    bash "$HERE/copy_from_scf.sh"
fi

# 1) Symmetry-off (ISYM=0) NSCF -> WAVECAR that LOBSTER can read
echo "Starting LOBSTER NSCF (@VEXEC@) at $(date)"
@LAUNCH@ > vasp.out 2>&1

# 2) Build lobsterin from the DOSCAR energy window (Fermi-referenced, E_F = 0)
LOBSTER_BIN="${LOBSTER_BIN:-@LOBSTER@}"
if [ ! -f DOSCAR ]; then echo "ERROR: no DOSCAR (NSCF failed?)"; exit 1; fi
read emax emin nedos efermi _ < <(sed -n '6p' DOSCAR)
starteng=$(awk -v a="$emin" -v f="$efermi" 'BEGIN{printf "%.4f", a-f}')
endeng=$(awk   -v a="$emax" -v f="$efermi" 'BEGIN{printf "%.4f", a-f}')
cat > lobsterin <<LOB
basisSet pbeVaspFit2015
cohpGenerator from 0.8 to 4
cobiGenerator from 0.8 to 4
COHPStartEnergy $starteng
COHPEndEnergy $endeng
gaussianSmearingWidth 0.05
LOB

# 3) Run LOBSTER
echo "Running LOBSTER ($LOBSTER_BIN) at $(date)"
"$LOBSTER_BIN" > lobster.out 2>&1 || { echo "  LOBSTER failed -- see lobster.out"; exit 1; }
echo "  charge spilling:"; grep -i spilling lobster.out | grep '%' || true
echo "  Done: COHPCAR/COBICAR/COOPCAR + ICO*LIST.lobster in $HERE"
'''
        return (template.replace('@PREAMBLE@', preamble)
                        .replace('@VEXEC@', vasp_exec)
                        .replace('@LAUNCH@', launch_cmd)
                        .replace('@LOBSTER@', lobster_bin))
