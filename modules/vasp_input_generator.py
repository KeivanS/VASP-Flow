#!/usr/bin/env python3
"""
VASP Input Generator Module
Generates INCAR, KPOINTS, and job scripts for VASP calculations
"""

import os, sys, shutil
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
_LOBSTER_X   = os.environ.get('LOBSTER_X',   'lobster')


# ── Default Hubbard U lookup ─────────────────────────────────────────────────
# U_eff values live ONLY in hubbard_u_defaults.csv (repo root).  Dudarev
# GGA+U: LDAUU = U_eff, LDAUJ = 0.  Used by default only when the compound
# contains a chalcogen or halogen (below); 'GGA_U: ON' applies it to every
# tabulated element, 'GGA_U: OFF' never; explicit 'GGA+U with U=... on El-orb'
# always wins.  See _u_lines().
_U_ANIONS = ('O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I')   # chalcogens + halogens


def load_u_defaults():
    """{element: {'orbital', 'U', 'J'}} from hubbard_u_defaults.csv (repo root).

    Whitespace/tab-separated table; '#' lines are comments.  Only the first
    three columns are read: element, orbital (3d, 4d, 5d, 4f, 5f or d/f) and
    U_eff (eV).  Dudarev's scheme uses U_eff only, so J is always 0 (any J
    column in the file is ignored).  Rows with U_eff = 0 get no correction.
    There are no built-in values: if the file cannot be read, a warning is
    printed and no default U is applied.
    """
    path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', 'hubbard_u_defaults.csv'))
    out = {}
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for ln in fh:
                if not ln.strip() or ln.lstrip().startswith('#'):
                    continue
                tok = ln.replace(',', ' ').split()
                if len(tok) < 3 or not tok[0][:1].isupper():
                    continue
                orb = tok[1].strip().lower()[-1:]
                try:
                    u = float(tok[2])
                except ValueError:
                    continue
                if orb in 'spdf' and u > 0:
                    out[tok[0]] = {'orbital': orb, 'U': u, 'J': 0.0}
    except OSError as e:
        print(f"WARNING: cannot read {path} ({e}); no default Hubbard U applied",
              file=sys.stderr)
    return out


# ── SLURM: continue a step that hits its time limit ─────────────────────────
# Default walltime per step (HH:MM:SS).  An explicit <STEP>_WALLTIME or a
# global WALLTIME in the instructions wins; steps not listed use the profile.
STEP_WALLTIME = {'relax': '08:00:00', 'scf': '04:00:00', 'bands': '02:00:00',
                 'dos': '02:00:00', 'lobster': '04:00:00'}

# #SBATCH lines that make SLURM warn the batch shell 15 min before the limit
# and allow the job to be requeued (same job id, so afterok chains survive).
RESUME_SBATCH = ("#SBATCH --signal=B:USR1@900\n"
                 "#SBATCH --requeue\n"
                 "#SBATCH --open-mode=append\n")

# Bash functions sourced into every SLURM step script.  Needs, set before:
#   VF_KIND   = relax | static | lobster      VF_SCRIPT = path of this script
#   VF_ARGS   = arguments to resubmit it with (optional)
# Flow:  vf_resume_prepare ; VF_PHASE=vasp ; vf_run <vasp launch> ;
#        vf_after vasp ; VF_PHASE=post ; vf_run <lobster> ; vf_after post
RESUME_LIB = r"""
# ---- auto-continue on time limit (VASP-Flow) --------------------------------
VF_MAX_RESTARTS="${VF_MAX_RESTARTS:-5}"
vf_timeout=0
vf_skip_vasp=0
vf_on_usr1() {            # SLURM: 15 min left
    vf_timeout=1
    echo "=== $(date): time limit near - stopping VASP cleanly ==="
    if [ "${VF_KIND:-static}" = relax ]; then echo "LSTOP = .TRUE." > STOPCAR   # after this ionic step
    else echo "LABORT = .TRUE." > STOPCAR; fi                        # after this electronic step
    # a non-VASP program (e.g. the LOBSTER binary) cannot stop cleanly: end it
    [ "${VF_PHASE:-vasp}" = post ] && [ -n "${vf_post_pid:-}" ] && kill "$vf_post_pid" 2>/dev/null
}
trap vf_on_usr1 USR1
vf_run() {                # run in the background so the USR1 trap can fire
    "$@" &
    local pid=$! rc=0
    vf_post_pid=$pid
    wait "$pid"; rc=$?
    while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done
    vf_post_pid=
    return $rc
}
vf_vasp_done() { [ -f OUTCAR ] && grep -q "General timing and accounting" OUTCAR; }
vf_poscar_from_xdatcar() { # last frame of XDATCAR -> POSCAR (cell included for ISIF=3)
    python3 - <<'PYX'
lines = open('XDATCAR').read().splitlines()
idx = [i for i, l in enumerate(lines) if l.strip().lower().startswith('direct configuration')]
if not idx:
    raise SystemExit(1)
last = idx[-1]
nat = sum(int(x) for x in lines[6].split())
# variable cell (ISIF=3): every frame repeats the 7-line header
head = lines[last - 7:last] if last >= 7 and lines[last - 7].strip() == lines[0].strip() \
    else lines[:7]
open('POSCAR', 'w').write('\n'.join(head + ['Direct'] + lines[last + 1:last + 1 + nat]) + '\n')
PYX
}
vf_resume_prepare() {     # called at the start of every (re)run
    [ -f .vf_resume ] || return 0
    local phase; phase=$(cat .vf_resume)
    rm -f .vf_resume STOPCAR
    echo "=== resuming after time limit (phase: $phase, restart $(cat .vf_restarts 2>/dev/null)) ==="
    if [ "$phase" = post ]; then vf_skip_vasp=1; return 0; fi   # VASP part done; redo the rest
    local n; n=$(ls OUTCAR.timeout* 2>/dev/null | wc -l)
    [ -f OUTCAR ] && mv OUTCAR "OUTCAR.timeout$n"
    [ -f OSZICAR ] && mv OSZICAR "OSZICAR.timeout$n"
    if [ "${VF_KIND:-static}" = relax ]; then
        local nat; nat=$(sed -n 7p POSCAR | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; print s}')
        if [ -s CONTCAR ] && [ "$(wc -l < CONTCAR)" -ge $((8 + nat)) ]; then
            cp CONTCAR POSCAR && echo "  continuing from CONTCAR"
        elif [ -s XDATCAR ] && vf_poscar_from_xdatcar; then
            echo "  continuing from the last XDATCAR frame"
        fi
        [ -f XDATCAR ] && mv XDATCAR "XDATCAR.timeout$n"
    fi
}
vf_after() {              # $1 = vasp | post : resubmit if the limit was hit
    [ "$vf_timeout" = 1 ] || return 0
    rm -f STOPCAR
    local n; n=$(( $(cat .vf_restarts 2>/dev/null || echo 0) + 1 ))
    echo "$n" > .vf_restarts
    if [ "$n" -gt "$VF_MAX_RESTARTS" ]; then
        echo "!!! time limit hit $n times - giving up (raise the walltime)"; exit 1
    fi
    echo "$1" > .vf_resume
    if [ -n "${SLURM_JOB_ID:-}" ] && scontrol requeue "$SLURM_JOB_ID" 2>/dev/null; then
        echo "=== requeued job $SLURM_JOB_ID (restart $n) ==="; sleep 300; exit 0
    fi
    # requeue not allowed: submit a copy and point dependent jobs at it
    local opts="--job-name=${SLURM_JOB_NAME:-vasp}" tl
    tl=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null | awk 'NR==1{print $1}')
    [[ "$tl" =~ ^[0-9]+(-[0-9]+)?(:[0-9]+)*$ ]] && opts="$opts --time=$tl"
    local new; new=$(cd "${SLURM_SUBMIT_DIR:-$PWD}" && sbatch --parsable $opts "$VF_SCRIPT" ${VF_ARGS:-}) \
        || { echo "!!! resubmit failed"; exit 1; }
    echo "=== resubmitted as job $new (restart $n) ==="
    squeue -h -u "${USER:-$(whoami)}" -o "%i %E" 2>/dev/null | awk -v j="${SLURM_JOB_ID:-none}" 'index($2, j) {print $1}' |
        while read -r dep; do scontrol update JobId="$dep" Dependency="afterok:$new"; done
    exit 0
}
# -----------------------------------------------------------------------------
"""


_INCAR_STEPS = ('all', 'relax', 'scf', 'bands', 'dos', 'wannier',
                'dfpt', 'phonons', 'lobster')


def merge_cli_incar(instructions, specs):
    """Merge --incar command-line overrides into instructions['incar_raw'].

    Each spec is  [STEP:]TAG=VAL[; TAG=VAL ...]  with STEP one of
    relax/scf/bands/dos/wannier/dfpt/phonons/lobster ('all' or no prefix =
    every step). Tags are appended AFTER any INCAR block from the
    instructions file, so the command line wins on conflicts. ';' separates
    tags (not ',' — values like MAGMOM may contain commas).
    """
    if not specs:
        return
    raw = instructions.setdefault('incar_raw', {})
    for spec in specs:
        step, sep, body = spec.partition(':')
        if sep and step.strip().lower() in _INCAR_STEPS:
            step = step.strip().lower()
        else:
            step, body = 'all', spec
        for tag in body.split(';'):
            tag = tag.strip()
            if '=' in tag:
                raw.setdefault(step, []).append(tag)


def write_shifted_poscar(src, dst, shift_cart=(0.01, 0.02, 0.03)):
    """Copy POSCAR src → dst with atom 1 displaced by shift_cart (Å, CARTESIAN).

    Used for the convergence-test POSCAR only: the displacement breaks the
    site symmetry so the force on atom 1 is non-zero and its convergence can
    be tracked. Production steps keep the unshifted POSCAR.

    Handles Direct and Cartesian coordinate modes (the Cartesian shift is
    converted through the lattice for Direct mode), an optional Selective
    dynamics block, VASP4/5 headers, and a negative scale factor (target
    volume). Everything after the shifted line is copied through verbatim.
    """
    with open(src) as f:
        lines = f.readlines()

    scale_raw = float(lines[1].split()[0])
    latt = np.array([[float(x) for x in lines[2 + i].split()[:3]]
                     for i in range(3)])
    if scale_raw < 0:                      # negative scale = target cell volume
        vol   = abs(np.linalg.det(latt))
        scale = (-scale_raw / vol) ** (1.0 / 3.0)
    else:
        scale = scale_raw
    A = latt * scale                       # rows = lattice vectors in Å

    # Header: line 5 is element symbols (VASP5) or counts (VASP4)
    idx = 5
    if not lines[5].split()[0].lstrip('+-').isdigit():
        idx = 6                            # skip the symbols line
    idx += 1                               # past the counts line
    if lines[idx].strip()[:1].upper() == 'S':
        idx += 1                           # past "Selective dynamics"
    mode_cartesian = lines[idx].strip()[:1].upper() in ('C', 'K')
    first_atom = idx + 1

    parts = lines[first_atom].split()
    pos = np.array([float(x) for x in parts[:3]])
    shift = np.asarray(shift_cart, dtype=float)
    if mode_cartesian:
        pos += shift / scale               # stored Cartesian coords are × scale
    else:
        pos += shift @ np.linalg.inv(A)    # Δfrac = Δcart · A⁻¹  (cart = frac·A)
    tail = parts[3:]                       # selective-dynamics flags etc.
    lines[first_atom] = ('  ' + '  '.join(f'{x:.16f}' for x in pos)
                         + ('   ' + ' '.join(tail) if tail else '') + '\n')
    lines[0] = (lines[0].rstrip('\n')
                + '  [atom 1 shifted by ({} {} {}) Ang for force convergence]\n'.format(*shift))

    with open(dst, 'w') as f:
        f.writelines(lines)

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

    def _second_shell_cutoff(self, default=4.0) -> float:
        """Bond-detection range (Å) for the lobsterin cohp/cobiGenerator:
        large enough to include every atom's 2nd-neighbour shell (+10%), so
        1st AND 2nd shell bonds appear in the COHP/COBI/COOP output. Floor at
        4.0 Å (the old fixed value), cap at 6.0 Å to keep the pair list sane.
        """
        try:
            lines = open(self.poscar).readlines()
            scale_raw = float(lines[1].split()[0])
            latt = np.array([[float(x) for x in lines[2 + i].split()[:3]]
                             for i in range(3)])
            if scale_raw < 0:
                scale = (-scale_raw / abs(np.linalg.det(latt))) ** (1.0 / 3.0)
            else:
                scale = scale_raw
            A = latt * scale
            idx = 5
            if not lines[5].split()[0].lstrip('+-').isdigit():
                idx = 6
            counts = [int(t) for t in lines[idx].split()]
            idx += 1
            if lines[idx].strip()[:1].upper() == 'S':
                idx += 1
            cartesian = lines[idx].strip()[:1].upper() in ('C', 'K')
            n = sum(counts)
            coords = np.array([[float(x) for x in lines[idx + 1 + i].split()[:3]]
                               for i in range(n)])
            cart = coords * scale if cartesian else coords @ A
            shifts = np.array([[i, j, k] for i in (-1, 0, 1)
                               for j in (-1, 0, 1) for k in (-1, 0, 1)],
                              dtype=float) @ A
            cut = 0.0
            for i in range(n):
                d = np.linalg.norm(cart[None, :, :] + shifts[:, None, :]
                                   - cart[i], axis=2).ravel()
                d = d[d > 1e-3]
                dmin = d.min()
                beyond = d[d > dmin * 1.2]
                cut = max(cut, beyond.min() if beyond.size else dmin)
            return round(min(max(cut * 1.10, default), 6.0), 2)
        except Exception:
            return default

    def _get_lobster_exec(self) -> str:
        """LOBSTER binary (profile 'lobster_x' overrides site.env LOBSTER_X).

        ~ is expanded here: the generated run.sh uses the path inside a quoted
        variable, where the shell would NOT expand a literal tilde.
        """
        return os.path.expanduser(self._profile_get('lobster_x', _LOBSTER_X))

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

        # Pure MPI: one thread per rank. vasp_std is not the OpenMP build, so a
        # threaded BLAS inside each rank buys nothing and, with a login shell
        # that exports OMP_NUM_THREADS=N, silently launches ranks x N threads --
        # e.g. 16 x 8 = 128 threads on 16 cores, which spends most of its time
        # in the scheduler. Set here (not in the user's profile) so other codes
        # on the same machine keep their own threading.
        lines.append('export OMP_NUM_THREADS=${VASP_OMP_NUM_THREADS:-1}')
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
        
        # KPOINTS — coarse mesh by default (RELAX_KMESH_DENSITY to change);
        # relaxation doesn't need a dense grid.  An explicit KPOINTS block wins.
        self._write_kpoints(output_dir, 'relax')

        # Copy POSCAR
        os.system(f"cp {self.poscar} {output_dir}/POSCAR")
        
        # Job script
        job_script = self._generate_job_script('relax', self._get_vasp_exec())
        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(job_script)
        os.chmod(f"{output_dir}/run.sh", 0o755)
    
    def _elf_mode(self) -> str:
        """'off', 'inline' (LELF in the SCF itself, KPAR = 1) or 'separate'
        (SCF at full KPAR, then a short KPAR = 1 restart in 02_scf/elf that
        only writes ELFCAR).  VASP computes no ELF for SOC/non-collinear runs."""
        if not self.instructions.get('elf', True) or self.instructions.get('soc', False):
            return 'off'
        return 'separate' if self.instructions.get('elf_mode') == 'separate' else 'inline'

    def _scf_force_kpar(self):
        """1 when the SCF itself runs LELF (ELF needs KPAR = 1), else None."""
        return 1 if self._elf_mode() == 'inline' else None

    def _ncore_for(self, step: str, force_kpar: int = None, incar_step: str = None) -> int:
        """Parse the NCORE value the parallel block would emit for `step`."""
        import re
        for ln in self._get_parallel_lines(step, force_kpar=force_kpar,
                                           incar_step=incar_step):
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
        nbands = self._lobster_nbands(
            potcar_path, self._ncore_for('scf', force_kpar=self._scf_force_kpar()))
        incar_content = self._generate_incar_scf(nbands=nbands)
        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(incar_content)
        
        # KPOINTS — user-selected density (default fine = 5000 kpra), KMESH
        # override, or an explicit `KPOINTS scf:` block (which wins).
        self._write_kpoints(output_dir, 'scf')

        # POSCAR: copy the input structure as a placeholder.
        # At runtime, run.sh will overwrite it with CONTCAR from 01_relax
        # if that directory exists and the relaxation completed.
        shutil.copy(self.poscar, f"{output_dir}/POSCAR")

        # Runtime copy script: relaxed geometry + electronic starting point
        # (CHGCAR, and WAVECAR when it is usable) from 01_relax.
        if from_relax:
            rel_relax = os.path.relpath(from_relax, output_dir)
            with open(f"{output_dir}/copy_from_relax.sh", 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f'HERE="$(cd "$(dirname "$0")" && pwd)"\n')
                f.write(f'RELAX_DIR="$HERE/{rel_relax}"\n')
                self._write_copy_if_newer(f, '$RELAX_DIR', 'CONTCAR', 'POSCAR', '01_relax')
                f.write(self._relax_restart_snippet())
            os.chmod(f"{output_dir}/copy_from_relax.sh", 0o755)
        
        # Separate ELF pass (see _elf_mode): elf/INCAR + run_elf.sh
        elf_dir = os.path.join(output_dir, 'elf')
        if self._elf_mode() == 'separate':
            os.makedirs(elf_dir, exist_ok=True)
            with open(os.path.join(elf_dir, 'INCAR'), 'w') as f:
                f.write(self._generate_incar_elf(incar_content))
            with open(os.path.join(output_dir, 'run_elf.sh'), 'w') as f:
                f.write(self._RUN_ELF_SH)
            os.chmod(os.path.join(output_dir, 'run_elf.sh'), 0o755)
        elif os.path.exists(os.path.join(output_dir, 'run_elf.sh')):
            os.remove(os.path.join(output_dir, 'run_elf.sh'))

        # Job script
        job_script = self._generate_job_script('scf', self._get_vasp_exec())
        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(job_script)
        os.chmod(f"{output_dir}/run.sh", 0o755)

    def _generate_incar_elf(self, scf_incar: str) -> str:
        """INCAR for the ELF pass: the SCF INCAR (same physics, same user
        overrides) restarted from its converged WAVECAR/CHGCAR, with KPAR = 1
        (LELF needs it) and only ELFCAR written.  NBANDS is set at run time
        to the value the SCF used (run_elf.sh), so the WAVECAR is read as is."""
        par = {l.split('=')[0].strip(): l.split('=')[1].strip()
               for l in self._get_parallel_lines('scf', force_kpar=1) if '=' in l}
        drop = {'KPAR', 'NCORE', 'NPAR', 'ISTART', 'ICHARG', 'LWAVE', 'LCHARG',
                'LORBIT', 'LELF', 'NBANDS'}
        keep = [l for l in scf_incar.splitlines()
                if self._incar_tag_name(l) not in drop
                and l.strip() not in ('# MPI parallelization', '# Output')]
        keep[0:1] = ["# ELF pass: restart from the converged 02_scf run (KPAR = 1 for LELF)"]
        keep += ["",
                 "# Restart from the converged SCF -> converges in a few steps",
                 "ISTART = 1",
                 "ICHARG = 1",
                 "LELF = .TRUE.   # electron localization function -> ELFCAR",
                 "LWAVE = .FALSE.",
                 "LCHARG = .FALSE.",
                 "",
                 "# MPI parallelization (LELF requires KPAR = 1)",
                 "KPAR  = 1"]
        if 'NCORE' in par:
            keep.append(f"NCORE = {par['NCORE']}")
        return '\n'.join(keep) + '\n'

    _RUN_ELF_SH = r'''#!/bin/bash
# ELF pass, run after the SCF in this folder:  bash run_elf.sh <launch command>
# The SCF ran with full k-point parallelism (no LELF).  This short restart
# reads its converged WAVECAR + CHGCAR with KPAR = 1 (required by LELF) and
# writes ELFCAR, which is copied back next to the SCF output.
HERE="$(cd "$(dirname "$0")" && pwd)"
E="$HERE/elf"
if [ ! -s "$HERE/WAVECAR" ] || [ ! -s "$HERE/CHGCAR" ]; then
    echo "ELF pass skipped: no WAVECAR/CHGCAR in $HERE"; exit 0
fi
cd "$E" || exit 1
for f in POSCAR KPOINTS WAVECAR CHGCAR; do cp "$HERE/$f" .; done
cp -L "$HERE/POTCAR" POTCAR
# Same NBANDS as the SCF, so its WAVECAR is read without band changes.
NB=$(awk '/NBANDS=/{print $NF; exit}' "$HERE/OUTCAR" 2>/dev/null)
sed -i.bak '/^ *NBANDS *=/d' INCAR && rm -f INCAR.bak
[ -n "$NB" ] && echo "NBANDS = $NB" >> INCAR
echo "Starting ELF pass (KPAR=1) at $(date)"
"$@" > vasp.out 2>&1
if [ -s ELFCAR ]; then
    cp ELFCAR "$HERE/ELFCAR"; echo "  OK: ELFCAR written"
else
    echo "  WARNING: no ELFCAR -- see $E/vasp.out"
fi
rm -f WAVECAR CHGCAR
'''

    # Bash appended to copy_from_relax.sh.  Decided at RUN time (not generation
    # time) so that k-meshes patched or edited after generation are honoured.
    _RELAX_RESTART_SH = r'''
# ── Electronic starting point from the relaxation ────────────────────────────
# CHGCAR is always usable as a starting density.  WAVECAR is only valid if the
# SCF is the same problem on the same k-mesh (k-points, spin, cutoff), so it is
# read only then; otherwise the SCF starts from the relaxed CHGCAR alone.
USER_SET_START=@USER_SET_START@   # 1: ISTART/ICHARG given explicitly in the instructions

norm_kp() {   # KPOINTS without the comment line: whitespace/case/number-format neutral
    tail -n +2 "$1" 2>/dev/null | sed 's/[[:space:]]\{1,\}/ /g; s/^ //; s/ $//' \
        | tr 'A-Z' 'a-z' | sed '/^$/d' | sed '2s/^\(.\).*/\1/' \
        | awk '{for(i=1;i<=NF;i++) if ($i ~ /^[-+]?[0-9.]+$/) $i=$i+0; print}'
}
incar_tag() { # file TAG -> value (upper-cased, blanks removed, comments dropped)
    sed 's/[!#].*//' "$1" 2>/dev/null | awk -F= -v t="$2" \
        'toupper($1) ~ "^[ \t]*" t "[ \t]*$" {gsub(/[ \t]/,"",$2); v=toupper($2)} END{print v}'
}
set_incar_tag() { # file TAG value  (replace any existing line, then append)
    sed -i.bak -E "/^[[:space:]]*$2[[:space:]]*=/d" "$1" && rm -f "$1.bak"
    printf '%s = %s\n' "$2" "$3" >> "$1"
}

if [ "$USER_SET_START" = "1" ]; then
    echo "  ISTART/ICHARG set explicitly in the instructions: leaving them, nothing read from 01_relax"
elif [ ! -s "$RELAX_DIR/CHGCAR" ]; then
    echo "  WARNING: 01_relax/CHGCAR not found; SCF starts from scratch"
else
    cp "$RELAX_DIR/CHGCAR" "$HERE/CHGCAR" && echo "  CHGCAR copied from 01_relax"
    same=1
    [ "$(norm_kp "$RELAX_DIR/KPOINTS")" = "$(norm_kp "$HERE/KPOINTS")" ] || same=0
    for t in ISPIN LSORBIT ENCUT; do
        [ "$(incar_tag "$RELAX_DIR/INCAR" $t)" = "$(incar_tag "$HERE/INCAR" $t)" ] || same=0
    done
    if [ "$same" = "1" ] && [ -s "$RELAX_DIR/WAVECAR" ]; then
        cp "$RELAX_DIR/WAVECAR" "$HERE/WAVECAR" && echo "  WAVECAR copied from 01_relax"
        set_incar_tag "$HERE/INCAR" ISTART 1
        set_incar_tag "$HERE/INCAR" ICHARG 1
        echo "  same k-mesh as 01_relax: SCF starts from WAVECAR + CHGCAR (ISTART=1, ICHARG=1)"
    else
        set_incar_tag "$HERE/INCAR" ISTART 0
        set_incar_tag "$HERE/INCAR" ICHARG 1
        echo "  k-mesh (or spin/cutoff) differs from 01_relax: SCF reads CHGCAR only (ISTART=0, ICHARG=1)"
    fi
fi
'''

    def _relax_restart_snippet(self) -> str:
        raw = self.instructions.get('incar_raw', {}) or {}
        tags = {self._incar_tag_name(l) for l in list(raw.get('all', [])) + list(raw.get('scf', []))}
        user = 1 if tags & {'ISTART', 'ICHARG'} else 0
        return self._RELAX_RESTART_SH.replace('@USER_SET_START@', str(user))

    def generate_bands_input(self, output_dir: str, from_scf: str):
        """Generate input files for band structure calculation"""
        os.makedirs(output_dir, exist_ok=True)
        
        # INCAR
        incar_content = self._generate_incar_bands()
        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(incar_content)
        
        # KPOINTS — line mode; kpath None triggers auto-detection from the
        # structure.  An explicit `KPOINTS bands:` block wins.
        self._write_kpoints(output_dir, 'bands')
        
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
            # Set NBANDS from NELECT in SCF OUTCAR, rounded to even.
            # Without SOC each band holds 2 electrons: occupied = NELECT/2, margin 10.
            # With SOC (noncollinear) each band holds 1: occupied = NELECT, margin 20.
            # A manually edited NBANDS is kept as long as it can hold all electrons.
            soc = self.instructions.get('soc', False)
            occ_expr = 'NELECT' if soc else 'NELECT / 2'
            margin = 20 if soc else 10
            f.write('if [ -f "$SCF_DIR/OUTCAR" ]; then\n')
            f.write('    NELECT=$(grep "^ *NELECT" "$SCF_DIR/OUTCAR" | head -1 | awk \'{print int($3)}\')\n')
            f.write('    if [ -n "$NELECT" ] && [ "$NELECT" -gt 0 ]; then\n')
            f.write(f'        OCC=$(( {occ_expr} ))\n')
            f.write(f'        NBANDS=$(( (OCC + {margin} + 1) / 2 * 2 ))\n')
            f.write('        CUR=$(grep "^ *NBANDS" "$HERE/INCAR" | head -1 | tr -cd "0-9")\n')
            f.write('        if [ -n "$CUR" ] && [ "$CUR" -gt "$OCC" ]; then\n')
            f.write('            echo "  NBANDS = $CUR  kept from INCAR (needs > $OCC occupied)"\n')
            f.write('        elif [ -n "$CUR" ]; then\n')
            f.write('            sed -i.bak "s/^ *NBANDS.*/NBANDS = $NBANDS/" "$HERE/INCAR"\n')
            f.write('            echo "  NBANDS = $NBANDS  (was $CUR: too small for NELECT = $NELECT, occupied = $OCC)"\n')
            f.write('        else\n')
            f.write('            echo "NBANDS = $NBANDS" >> "$HERE/INCAR"\n')
            f.write('            echo "  NBANDS = $NBANDS  (NELECT = $NELECT, occupied = $OCC)"\n')
            f.write('        fi\n')
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
        
        # KPOINTS — 2× the SCF mesh in every direction for better DOS resolution
        # (unless an explicit `KPOINTS dos:` block is given).
        self._write_kpoints(output_dir, 'dos')
        
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
        The k-mesh is 2× the SCF mesh in every direction (ratios preserved).
        """
        os.makedirs(output_dir, exist_ok=True)

        # NBANDS from the project-level POTCAR (built one level up before steps).
        potcar_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), 'POTCAR')
        nbands = self._lobster_nbands(potcar_path, self._ncore_for('lobster'))

        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(self._generate_incar_lobster(nbands=nbands, lmax=self._lmaxmix()))

        # KPOINTS — 2× the SCF mesh in every direction (same ratios). The NSCF
        # is cheap compared with the SCF, and COHP/COBI integrate over the BZ,
        # so a denser mesh smooths the bonding curves at little extra cost.
        self._write_kpoints(output_dir, 'lobster')

        # Editable lobsterin (run.sh uses it as-is if present). Energy window is
        # Fermi-referenced (E_F = 0); edit it or COHPStartEnergy for deep states.
        # The bond-generator range reaches the 2nd-neighbour shell so 2nd-shell
        # bonds (e.g. metal-metal in rocksalt) appear in COHP/COBI/COOP.
        sigma = self.instructions.get('lobster_sigma') or '0.10'
        bondmax = self._second_shell_cutoff()
        with open(f"{output_dir}/lobsterin", 'w') as f:
            f.write("basisSet pbeVaspFit2015\n"
                    f"cohpGenerator from 0.8 to {bondmax}\n"
                    f"cobiGenerator from 0.8 to {bondmax}\n"
                    "COHPStartEnergy -20\n"
                    "COHPEndEnergy 5\n"
                    f"gaussianSmearingWidth {sigma}\n")

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

        with open(f"{output_dir}/INCAR", 'w') as f:
            f.write(self._generate_incar_wannier(num_bands))

        kp_text = self._write_kpoints(output_dir, 'wannier')

        # Resolve the actual mesh integers so wannier90.win mp_grid matches KPOINTS.
        kind, val = self._parse_kpoints_text(kp_text)
        if kind != 'mesh':
            raise ValueError("Wannier90 needs an automatic Gamma mesh: give the "
                             "`KPOINTS wannier:` block in mesh form (0 / Gamma / n1 n2 n3).")
        nx, ny, nz = val

        with open(f"{output_dir}/wannier90.win", 'w') as f:
            f.write(self._generate_wannier90_win(num_wann, num_bands, nx, ny, nz, wannier_info))

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
        lines.extend(self._get_parallel_lines('wannier'))  # KPAR=1 for Wannier
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

    def _kpoints_block_for_wannier(self, nx: int, ny: int, nz: int) -> str:
        """Generate begin kpoints ... end kpoints block for wannier90.win.

        Lists all k-points of the Gamma-centred Monkhorst-Pack mesh explicitly.
        Required by wannier90.x -pp (VASP 5 file-based workflow).
        The ordering (k3 fastest) matches VASP's internal k-point ordering.
        """
        kpts = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    kpts.append(
                        f'  {i/nx:.8f}  {j/ny:.8f}  {k/nz if nz > 1 else 0.0:.8f}')
        return '\n'.join(['\nbegin kpoints'] + kpts + ['end kpoints'])

    def _generate_wannier90_win(self, num_wann: int, num_bands: int,
                                 nx: int, ny: int, nz: int,
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
            f'mp_grid : {nx} {ny} {nz}',
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

        lines.append(self._kpoints_block_for_wannier(nx, ny, nz))

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

        # Same k-mesh as SCF (user-selected density, or the explicit SCF block)
        # — DFPT Born/dielectric results converge with the same mesh used for
        # the charge density.  A `KPOINTS dfpt:` block wins.
        self._write_kpoints(output_dir, 'dfpt')

        rel_scf = os.path.relpath(from_scf, output_dir)
        with open(f"{output_dir}/copy_from_scf.sh", 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f'HERE="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write(f'SCF_DIR="$HERE/{rel_scf}"\n')
            f.write(
'''# Guard: 02_scf may have been converted in place for LOBSTER (ISYM=0/-1).
# Its CHGCAR is still the symmetric converged density (the conversion is a
# fixed-charge NSCF with LCHARG=.FALSE.), but its WAVECAR is a full-mesh
# symmetry-off one that this ISYM=2 step cannot use — skip it and let VASP
# regenerate the orbitals from CHGCAR.
ISYM_SCF=$(sed 's/[!#].*//' "$SCF_DIR/INCAR" 2>/dev/null \\
           | awk -F= 'toupper($1) ~ /^[ \\t]*ISYM[ \\t]*$/ {gsub(/[ \\t]/,"",$2); print $2}' \\
           | tail -1)
if [ "$ISYM_SCF" = "0" ] || [ "$ISYM_SCF" = "-1" ]; then
    echo "  NOTE: 02_scf/INCAR has ISYM=$ISYM_SCF (converted for LOBSTER) — skipping its"
    echo "        WAVECAR; DFPT (ISYM=2) will start from CHGCAR instead."
    if ! sed 's/[!#].*//' "$SCF_DIR/INCAR" | grep -qiE 'ICHARG[ \\t]*=[ \\t]*11|LCHARG[ \\t]*=[ \\t]*\\.FALSE\\.'; then
        echo "  WARNING: 02_scf is symmetry-off and NOT a fixed-charge NSCF — its CHGCAR"
        echo "           may have been generated with symmetry off. Consider re-running"
        echo "           a symmetric (ISYM=2) SCF before DFPT."
    fi
else
    cp "$SCF_DIR/WAVECAR" "$HERE/" 2>/dev/null && echo "  WAVECAR copied from 02_scf" \\
        || echo "  WARNING: WAVECAR not found in 02_scf"
fi
cp "$SCF_DIR/CHGCAR" "$HERE/" 2>/dev/null && echo "  CHGCAR copied from 02_scf" \\
    || echo "  WARNING: CHGCAR not found in 02_scf"
''')
            self._write_copy_if_newer(f, '$SCF_DIR', 'POSCAR', 'POSCAR', '02_scf')
        os.chmod(f"{output_dir}/copy_from_scf.sh", 0o755)

        with open(f"{output_dir}/extract_born.py", 'w') as f:
            f.write(self._extract_born_script())

        with open(f"{output_dir}/run.sh", 'w') as f:
            f.write(self._generate_job_script_dfpt())
        os.chmod(f"{output_dir}/run.sh", 0o755)

    def _dfpt_finite_field_reason(self):
        """Why the DFPT step must use the finite-field (LCALCEPS) route, or None.

        VASP's linear response (IBRION=8 + LEPSILON) is not implemented for
        noncollinear/SOC, hybrid functionals, or meta-GGAs — those need the
        finite-electric-field (PEAD) route instead.
        """
        if self.instructions.get('soc', False):
            return 'SOC (noncollinear)'
        f = self.instructions.get('functional', 'PBE')
        if f == 'HSE06':
            return 'the HSE06 hybrid functional'
        if f == 'R2SCAN':
            return 'the R2SCAN meta-GGA'
        return None

    def _generate_incar_dfpt(self) -> str:
        dfpt_info = self.instructions.get('dfpt', {})
        ediff = (dfpt_info or {}).get('ediff', '1E-8')
        encut = self._encut()
        ff_reason = self._dfpt_finite_field_reason()
        lines = [
            "# VASP DFPT — Born effective charges + static dielectric tensor",
            "SYSTEM = " + self.instructions.get('project_name', 'DFPT'),
            "",
            "PREC   = Accurate",
            f"ENCUT  = {encut}",
            f"EDIFF  = {ediff}",
            "NELM   = 100",
            "",
        ]
        if ff_reason:
            lines += [
                f"# Finite-field (PEAD) route: DFPT (IBRION=8 + LEPSILON) does not",
                f"# support {ff_reason}.  Requires an insulating system.",
                "IBRION = 6      ! finite differences -> ionic contributions",
                "NSW    = 1",
                "POTIM  = 0.015",
                "",
                "LCALCEPS = .TRUE.    ! response to a finite E-field: eps, Z*, piezo",
                "# EFIELD_PEAD = 0.01 0.01 0.01   ! field strength in eV/A (default 0.01)",
            ]
        else:
            lines += [
                "# DFPT linear-response",
                "IBRION = 8      ! density-functional perturbation theory",
                "NSW    = 1",
                "POTIM  = 0",
                "",
                "LEPSILON = .TRUE.    ! Born charges + macroscopic dielectric tensor",
                "LRPA     = .FALSE.   ! include local-field effects (use .TRUE. for RPA)",
            ]
        lines += [
            "",
            "ICHARG = 1           ! read converged CHGCAR from SCF",
            "ISYM   = 2           ! symmetry imposed — perturbation theory needs it;",
            "                     ! never inherit ISYM=0/-1 from NSCF/LOBSTER steps",
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
        if (self.instructions.get('mpi_np', 1) or 1) > 1:
            lines += [
                "# Linear response / PEAD supports neither k-point nor band",
                "# parallelisation: KPAR and NCORE must both be 1",
                "KPAR  = 1",
                "NCORE = 1",
                "",
            ]
        lines.extend(["LWAVE  = .FALSE.", "LCHARG = .FALSE."])
        lines = self._apply_incar_overrides(lines, 'dfpt')
        # ISYM = 2 is mandatory here: a symmetry-off tag (ISYM=0/-1, e.g. from
        # a global INCAR override block meant for NSCF/LOBSTER) must not leak
        # into the perturbation-theory step.
        lines = ["ISYM   = 2           ! enforced — symmetry must stay on for DFPT/PEAD"
                 if (self._incar_tag_name(ln) == 'ISYM'
                     and ln.split('=', 1)[1].split('!')[0].split('#')[0].strip() != '2')
                 else ln
                 for ln in lines]
        return '\n'.join(lines) + '\n'

    def _extract_born_script(self) -> str:
        """Python script to parse DFPT/finite-field OUTCAR → phonopy BORN + summary."""
        return r'''#!/usr/bin/env python3
"""extract_born.py — Extract Born charges, dielectric and piezoelectric tensors
from a VASP DFPT (IBRION=8 + LEPSILON) or finite-field (LCALCEPS) OUTCAR.
Writes BORN (phonopy format, uses electronic eps_inf) and born_charges.txt.
Usage: python3 extract_born.py [OUTCAR]
"""
import sys, os, re

outcar = sys.argv[1] if len(sys.argv) > 1 else 'OUTCAR'
try:
    text = open(outcar).read()
except FileNotFoundError:
    print(f"ERROR: {outcar} not found"); sys.exit(1)

def tensor3(segment):
    """First 3x3 float block after a dashed separator line."""
    m = re.search(
        r'\n\s*-+\s*\n'
        r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
        r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
        r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', segment)
    if not m:
        return None
    v = [float(x) for x in m.groups()]
    return [v[0:3], v[3:6], v[6:9]]

def find_eps(ionic):
    """Dielectric tensor blocks; ionic=False -> electronic eps_inf (prefer the
    variant including local-field effects), ionic=True -> ionic contribution."""
    best = None
    for m in re.finditer(r'MACROSCOPIC STATIC DIELECTRIC TENSOR[^\n]*', text):
        hdr = m.group()
        if ('IONIC' in hdr) != ionic:
            continue
        t = tensor3(text[m.end():m.end() + 400])
        if t and (best is None or 'local field' in hdr):
            best = t
    return best

def find_piezo(ionic):
    """Piezoelectric stress tensor e (C/m^2): 3 field rows x 6 strain columns."""
    best = None
    for m in re.finditer(r'PIEZOELECTRIC TENSOR[^\n]*', text):
        if ('IONIC' in m.group()) != ionic:
            continue
        rows = re.findall(
            r'\n\s*[xyz]\s+' + r'([-\d.]+)\s+' * 5 + r'([-\d.]+)',
            text[m.end():m.end() + 700])
        if len(rows) >= 3:
            best = [[float(x) for x in r] for r in rows[:3]]
    return best

def add3(a, b):
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]

eps_el = find_eps(ionic=False)
if not eps_el:
    print("ERROR: dielectric tensor not found — did LEPSILON/LCALCEPS run?"); sys.exit(1)
eps_ion = find_eps(ionic=True)
eps_tot = add3(eps_el, eps_ion) if eps_ion else None

# Born effective charges — take the last block (cumulative output)
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

# Element label per ion, from the POSCAR next to the OUTCAR
def ion_labels(n):
    try:
        lines = open(os.path.join(os.path.dirname(os.path.abspath(outcar)),
                                  'POSCAR')).read().splitlines()
        syms, counts = lines[5].split(), [int(x) for x in lines[6].split()]
        out = [s for s, c in zip(syms, counts) for _ in range(c)]
        if len(out) == n:
            return out
    except Exception:
        pass
    return [''] * n

labels = ion_labels(len(born))

# Acoustic sum rule: Born charges must sum to zero over all ions
asr = [[sum(z[i][j] for z in born) for j in range(3)] for i in range(3)]
asr_max = max(abs(x) for row in asr for x in row)

pz_el  = find_piezo(ionic=False)
pz_ion = find_piezo(ionic=True)
pz_tot = None
if pz_el:
    pz_tot = ([[pz_el[i][j] + pz_ion[i][j] for j in range(6)] for i in range(3)]
              if pz_ion else pz_el)

# Write BORN (phonopy format: electronic eps_inf + Z*)
with open('BORN', 'w') as f:
    f.write("# Born effective charges and dielectric tensor from VASP OUTCAR\n")
    f.write("14.400\n")
    for row in eps_el:
        f.write("  " + "  ".join(f"{x:12.6f}" for x in row) + "\n")
    for z in born:
        for row in z:
            f.write("  " + "  ".join(f"{x:12.6f}" for x in row) + "\n")
print(f"Written BORN ({len(born)} atoms)")

def w_tensor(f, t, indent='  ', ncol=3):
    for row in t:
        f.write(indent + "  ".join(f"{x:10.5f}" for x in row[:ncol]) + "\n")

with open('born_charges.txt', 'w') as f:
    f.write("Born Effective Charges, Dielectric and Piezoelectric Tensors\n")
    f.write("=" * 62 + "\n\n")
    f.write("Electronic (ion-clamped) dielectric tensor eps_inf:\n")
    w_tensor(f, eps_el)
    if eps_ion:
        f.write("\nIonic contribution to the dielectric tensor:\n")
        w_tensor(f, eps_ion)
        f.write("\nTotal static dielectric tensor eps_0 = eps_inf + ionic:\n")
        w_tensor(f, eps_tot)
        f.write(f"\n  eps_0 (diagonal): {eps_tot[0][0]:.4f}  "
                f"{eps_tot[1][1]:.4f}  {eps_tot[2][2]:.4f}\n")
    else:
        f.write("\n(no ionic contribution found in OUTCAR — eps_0 unavailable)\n")
    f.write("\nBorn effective charge tensors (Z*):\n")
    for i, z in enumerate(born):
        f.write(f"\n  Ion {i+1} {labels[i]}:\n")
        w_tensor(f, z, indent='    ')
    f.write(f"\nAcoustic sum rule: max |sum_ions Z*| component = {asr_max:.5f} e\n")
    f.write("  (should be ~0; large values indicate under-converged k-mesh/EDIFF)\n")
    if pz_el:
        f.write("\nPiezoelectric stress tensor e (C/m^2), rows Ex Ey Ez,\n"
                "columns XX YY ZZ XY YZ ZX:\n")
        f.write("\n  Electronic (clamped-ion):\n")
        w_tensor(f, pz_el, indent='    ', ncol=6)
        if pz_ion:
            f.write("\n  Ionic contribution:\n")
            w_tensor(f, pz_ion, indent='    ', ncol=6)
            f.write("\n  Total (relaxed-ion):\n")
            w_tensor(f, pz_tot, indent='    ', ncol=6)
        f.write("\n  (meaningful only for non-centrosymmetric crystals)\n")
print("Written born_charges.txt")

diag = eps_tot or eps_el
name = 'eps_0' if eps_tot else 'eps_inf'
print(f"\n{name} (diagonal): {diag[0][0]:.4f}  {diag[1][1]:.4f}  {diag[2][2]:.4f}")
print(f"Acoustic sum rule violation (max component): {asr_max:.5f} e")
print("Born charges (diagonal elements):")
for i, z in enumerate(born):
    print(f"  Ion {i+1} {labels[i]}: Zxx={z[0][0]:8.4f}  Zyy={z[1][1]:8.4f}  Zzz={z[2][2]:8.4f}")
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

echo "Starting Born-charge / dielectric response run at $(date)"
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

        # Coarse k-mesh — force-constant supercells are large; 1000 kpra is
        # generous for a supercell and Gamma-only is often sufficient.
        self._write_kpoints(output_dir, 'phonons')

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
        lines.extend(self._get_parallel_lines('phonons'))
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

    def _estimate_nkpts(self, calc_type: str = 'scf') -> int:
        """Irreducible k-point count this step will end up running.

        Uses spglib to reduce the Gamma mesh by the crystal symmetry, which is
        what VASP reports as NKPTS.  Falls back to half the full mesh (a rough
        stand-in for time-reversal alone) when spglib is unavailable, and to
        the full mesh for the LOBSTER NSCF, which runs ISYM=0 and therefore
        gets no reduction at all.  Band structure is line-mode: the count is
        the path length, not a mesh.
        """
        # An explicit KPOINTS block fixes the count outright.
        text = self._explicit_kpoints(calc_type)
        if text:
            kind, val = self._parse_kpoints_text(text)
            if kind == 'list':
                return max(1, int(val))
            if kind == 'line':
                return max(1, int(val) * 4)       # ~4 segments on a typical path
        if calc_type == 'bands':
            npts = self.instructions.get('nkpts_bands') or self.instructions.get('nkpts') or 40
            return max(1, int(npts) * 4)          # ~4 segments on a typical path
        if text:                                  # explicit automatic mesh
            mesh = self._parse_kpoints_text(text)[1]
        elif calc_type in ('dos', 'lobster'):
            mesh = self._mesh_x2()
        elif calc_type == 'relax':
            mesh = self._relax_mesh()
        elif calc_type == 'phonons':
            mesh = self._compute_mesh('coarse')
        else:
            mesh = self._scf_mesh()
        full = max(1, mesh[0] * mesh[1] * mesh[2])
        if calc_type == 'lobster':
            return full                            # ISYM=0: no reduction
        try:
            import spglib
            from pymatgen.core import Structure
            st = Structure.from_file(self.poscar)
            cell = (st.lattice.matrix, st.frac_coords,
                    [s.specie.Z for s in st])
            mapping, _ = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=[0, 0, 0])
            return max(1, len(set(mapping)))
        except Exception:
            return max(1, full // 2)

    def _estimate_nbands(self) -> int:
        """Approximate NBANDS VASP will choose, for parallel-layout purposes.

        VASP's default is roughly NELECT/2 + NIONS/2 (doubled for
        noncollinear/SOC).  NELECT comes from the POTCAR ZVALs when pymatgen
        and a POTCAR are available; otherwise a deliberately low estimate is
        used -- underestimating only makes the layout more conservative
        (larger NCORE, fewer band groups), never invalid.
        """
        n_atoms = max(1, self._read_atom_count())
        soc = self.instructions.get('soc', False)
        for cand in ('POTCAR',
                     os.path.join(os.path.dirname(os.path.abspath(self.poscar)), 'POTCAR')):
            try:
                if not os.path.isfile(cand):
                    continue
                from pymatgen.core import Structure
                from pymatgen.io.vasp.inputs import Potcar
                st = Structure.from_file(self.poscar)
                zval = {p.symbol.split('_')[0]: float(p.zval)
                        for p in Potcar.from_file(cand)}
                nelect = sum(zval[s.specie.symbol] for s in st
                             if s.specie.symbol in zval)
                if nelect > 0:
                    nb = nelect + n_atoms / 2 if soc else nelect / 2 + n_atoms / 2
                    return max(1, int(nb))
            except Exception:
                continue
        return max(1, 2 * n_atoms)                 # conservative fallback

    @staticmethod
    def _auto_kpar_ncore(np_ranks: int, n_k: int, n_bands: int = None,
                         kpar: int = None, ncore: int = None) -> tuple:
        """(KPAR, NCORE) for np_ranks MPI ranks and n_k irreducible k-points.

        The rule, in plain words:
          KPAR  = the largest divisor of np_ranks that is <= n_k
                  (k-point groups scale almost perfectly, so use as many as
                  there are k-points; a divisor so no rank is left idle)
          NCORE = the largest divisor of (np_ranks / KPAR) that is <= its
                  square root (VASP's own recommendation for the ranks
                  inside one k-group; 1 when a group has 1-3 ranks)

        A user-given KPAR or NCORE is kept (snapped down to a divisor) and only
        the other one is derived: with a fixed NCORE, KPAR is the largest
        divisor <= n_k that leaves a multiple of NCORE ranks per group.
        n_bands is accepted for backward compatibility and not used.
        """
        import math
        n_k = max(1, int(n_k))
        divs = lambda n: [d for d in range(1, n + 1) if n % d == 0]
        if kpar:
            kpar = max(d for d in divs(np_ranks) if d <= kpar)
        elif ncore:
            ncore = max(d for d in divs(np_ranks) if d <= ncore)
            kpar = max([d for d in divs(np_ranks)
                        if d <= n_k and (np_ranks // d) % ncore == 0] or [1])
        else:
            kpar = max(d for d in divs(np_ranks) if d <= n_k)
        rpg = np_ranks // kpar                       # ranks per k-group
        if ncore:
            ncore = max(d for d in divs(rpg) if d <= ncore)
        else:
            ncore = max(d for d in divs(rpg) if d <= math.sqrt(rpg))
        return kpar, ncore

    def _get_parallel_lines(self, calc_type: str = 'scf', force_kpar: int = None,
                            incar_step: str = None) -> list:
        """INCAR KPAR / NCORE lines for step *calc_type* (see _auto_kpar_ncore).

        n_k is the irreducible k-point count of that step's own KPOINTS
        (_estimate_nkpts).  Priority, highest first:
          force_kpar (ELF: LELF needs KPAR = 1)
          > KPAR/NCORE inside an INCAR block for the step (or unnamed block)
          > per-step keys (SCF_KPAR, DOS_NCORE, ...) > global KPAR / NCORE
          > the automatic rule.
        When only one of the pair is set, the other is derived for it.
        phonons and wannier default to KPAR = 1.  DFPT never calls this
        (linear response supports neither tag; it writes KPAR = NCORE = 1).
        """
        np = self.instructions.get('mpi_np', 1) or 1
        if np <= 1:
            return []

        blk = self._incar_block_ints(incar_step or calc_type, ('KPAR', 'NCORE'))
        u_ncore = (blk.get('NCORE')
                   or self.instructions.get(f'{calc_type}_ncore')
                   or self.instructions.get('ncore'))
        u_kpar  = (force_kpar
                   or blk.get('KPAR')
                   or self.instructions.get(f'{calc_type}_kpar')
                   or self.instructions.get('kpar'))
        # Phonon supercells and the Wannier90 interface run with one k-group.
        if calc_type in ('phonons', 'wannier') and not u_kpar:
            u_kpar = 1
        kpar, ncore = self._auto_kpar_ncore(np, self._estimate_nkpts(calc_type),
                                            kpar=u_kpar, ncore=u_ncore)
        return [
            "",
            "# MPI parallelization",
            f"KPAR  = {kpar}",
            f"NCORE = {ncore}",
        ]

    def _incar_block_ints(self, step: str, tags) -> dict:
        """Integer values of *tags* set in the INCAR blocks that apply to
        *step* (the step block wins over the unnamed one)."""
        raw = self.instructions.get('incar_raw', {}) or {}
        out = {}
        for line in list(raw.get('all', [])) + list(raw.get(step, [])):
            tag = self._incar_tag_name(line)
            if tag in tags:
                try:
                    out[tag] = int(float(line.split('=', 1)[1].split('!')[0]
                                         .split('#')[0].strip().split()[0]))
                except (ValueError, IndexError):
                    pass
        return out

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
        skip_cont = False          # inside the "\"-continued tail of a replaced tag
        for line in lines:
            if skip_cont:
                skip_cont = line.rstrip().endswith('\\')
                continue
            tag = self._incar_tag_name(line)
            if tag and tag in ov_map:
                new_lines.append(ov_map[tag])
                used.add(tag)
                skip_cont = line.rstrip().endswith('\\')
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

    def _species_counts(self) -> list:
        """[(element, count), ...] in POSCAR order ([] if not readable)."""
        try:
            with open(self.poscar) as f:
                lines = f.readlines()
            counts = [int(x) for x in lines[6].split()]
            if self.elements and len(self.elements) == len(counts):
                return list(zip(self.elements, counts))
        except Exception:
            pass
        return []

    def _magmom_values(self):
        """Resolve the MAGMOM instruction to numbers.

        Returns (values, is_vector): one float per atom (POSCAR order), or —
        for a non-collinear run given 3N numbers — the raw x,y,z components.
        Raises ValueError with a clear message if the spec does not fit the
        structure (never silently falls back to a default).
        """
        n    = self._read_atom_count()
        soc  = self.instructions.get('soc', False)
        spec = self.instructions.get('magmom')
        if not spec:                                   # keyword only: uniform default
            return [float(self.instructions.get('mag_moment', 2.0))] * n, False
        kind = spec['kind']
        if kind == 'uniform':
            return [float(spec['value'])] * n, False
        if kind == 'elements':
            sc = self._species_counts()
            names = {el for el, _ in sc}
            unknown = set(spec['values']) - names
            if not sc or unknown:
                raise ValueError(
                    f"MAGMOM names element(s) {sorted(unknown) or '?'} that are not "
                    f"in the POSCAR (elements: {sorted(names) or 'unreadable'}).")
            vals = []
            for el, cnt in sc:
                vals += [float(spec['values'].get(el, 0.0))] * cnt
            return vals, False
        vals = [float(v) for v in spec['values']]      # explicit list
        if len(vals) == n:
            return vals, False
        if soc and len(vals) == 3 * n:
            return vals, True
        raise ValueError(
            f"MAGMOM lists {len(vals)} value(s) but the POSCAR has {n} atom(s)"
            f"{f' (or {3*n} components for SOC)' if soc else ''}: {spec['raw']!r}")

    @staticmethod
    def _magmom_tag(values: list) -> list:
        """`MAGMOM = ...` INCAR line(s), consecutive equal values as n*val.
        Very long lists are split with VASP's trailing-backslash continuation."""
        toks, i = [], 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[j + 1] == values[i]:
                j += 1
            run, v = j - i + 1, f"{(values[i] or 0.0):g}"     # avoid "-0"
            if '.' not in v and 'e' not in v:
                v += '.0'
            toks.append(f"{run}*{v}" if run > 1 else v)
            i = j + 1
        line = "MAGMOM = " + ' '.join(toks)
        if len(line) <= 800:
            return [line]
        out, cur = [], "MAGMOM ="
        for t in toks:
            if len(cur) + len(t) + 1 > 700:
                out.append(cur + " \\")
                cur = "  "
            cur += " " + t
        return out + [cur]

    def _mag_lines(self) -> list:
        """ISPIN / MAGMOM lines honouring the `MAGMOM:` instruction.

        Collinear: ISPIN = 2 and one moment per atom (uniform value, per
        element, or an explicit list; negative values allowed).  SOC
        (non-collinear): the moments are 3N components, so a scalar per atom
        is rotated onto the requested magnetization direction (x, y or z) —
        without this VASP would start from its default (1,1,1) direction.
        """
        mag = self.instructions.get('magnetization', {})
        if not mag.get('enabled'):
            return []
        values, is_vec = self._magmom_values()
        if not self.instructions.get('soc', False):
            return (["# Collinear magnetization", "ISPIN = 2"]
                    + self._magmom_tag(values) + [""])
        if not is_vec:
            d = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}.get(
                mag.get('direction') or 'z', (0, 0, 1))
            values = [m * c for m in values for c in d]
        return (["# Non-collinear initial moments (x y z per atom)"]
                + self._magmom_tag(values) + [""])

    def _u_lines(self) -> list:
        """GGA+U INCAR lines (Dudarev, LDAUTYPE=2), identical in every step.

        1. Explicit 'GGA+U with U=... on El-orb' entries always win.
        2. Otherwise the tabulated U_eff (hubbard_u_defaults.csv) is used
           according to the GGA_U flag in the instructions:
             GGA_U: ON   -> every tabulated d/f element gets its U
             GGA_U: OFF  -> no U
             no flag     -> U only if the compound contains a chalcogen or
                            halogen (O, S, Se, Te, F, Cl, Br, I); otherwise
                            U = 0 (the table values were fitted for those).
           Never automatic under R2SCAN/HSE06 (already reduce the
           self-interaction error).
        """
        u_info = self.instructions.get('gga_u', {})
        els  = u_info.get('elements', {}) if u_info.get('enabled') else {}
        mode = self.instructions.get('gga_u_mode', 'auto')
        auto = False
        if not els and mode != 'off':
            func_ok = self.instructions.get('functional', 'PBE') not in ('R2SCAN', 'HSE06')
            chalc = any(a in (self.elements or []) for a in _U_ANIONS)
            if func_ok and (mode == 'on' or chalc):
                table = load_u_defaults()
                els = {el: table[el] for el in (self.elements or []) if el in table}
                auto = 'on' if mode == 'on' else 'auto'
        if not els:
            return []
        # Build LDAUL, LDAUU, LDAUJ arrays in species order
        orb_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        ldaul, ldauu, ldauj = [], [], []
        for el in (self.elements or list(els.keys())):
            if el in els:
                orb = els[el].get('orbital', 'd')
                ldaul.append(str(orb_map.get(orb, 2)))
                ldauu.append(str(els[el].get('U', 0.0)))
                ldauj.append('0.0')          # Dudarev: only U_eff matters
            else:
                ldaul.append('-1')
                ldauu.append('0.0')
                ldauj.append('0.0')
        lmax = 6 if any(els[e].get('orbital') == 'f' for e in els) else 4
        head = ["# GGA+U (Dudarev, LDAUTYPE=2)"]
        if auto:
            why = ("GGA_U: ON" if auto == 'on' else
                   "chalcogenide/halide (O/S/Se/Te/F/Cl/Br/I present), no GGA_U flag")
            head = ["# GGA+U (Dudarev, LDAUTYPE=2) — U_eff from hubbard_u_defaults.csv",
                    f"# because: {why}.",
                    "# 'GGA_U: OFF' disables it; 'GGA+U with U=<val> on <El>-<orb>' overrides."]
        return head + [
                "LDAU = .TRUE.", "LDAUTYPE = 2",
                f"LDAUL = {' '.join(ldaul)}",
                f"LDAUU = {' '.join(ldauu)}",
                f"LDAUJ = {' '.join(ldauj)}",
                f"LMAXMIX = {lmax}    ! d/f charge-density mixing (required with LDAU)",
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
            "# Start from scratch (ignore any stale WAVECAR/CHGCAR in this directory);",
            "# WAVECAR and CHGCAR are written for the SCF step to pick up.",
            "ISTART = 0",
            "ICHARG = 2",
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

    def _generate_incar_scf(self, nbands: int = None,
                            for_convergence: bool = False) -> str:
        """Generate INCAR for SCF calculation.

        for_convergence trims the run to what a convergence test actually
        consumes -- the total energy in OUTCAR.  The production output tags
        (LWAVE/LCHARG/LORBIT/LELF) are dropped, which besides the obvious I/O
        saving also releases KPAR: LELF forces KPAR=1, and a k-point sweep with
        no k-point parallelism is the slowest way to run one.
        """
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

        # Electron localization function (ELFCAR): on by default; VASP does not
        # compute ELF for SOC/non-collinear runs, and LELF requires KPAR=1.
        elf = self._elf_mode() == 'inline' and not for_convergence

        # LELF needs KPAR = 1: pin it and let NCORE be derived for the single
        # k-group of all ranks (not left at the value that suited KPAR > 1).
        par_lines = self._get_parallel_lines('scf', force_kpar=1 if elf else None)
        if elf:
            par_lines = [('KPAR  = 1   # forced to 1: ELF (LELF) needs KPAR=1'
                          if l.strip().startswith('KPAR') else l) for l in par_lines]
        lines.extend(par_lines)

        if for_convergence:
            out = ["# Output — a convergence test reads only the total energy",
                   "LWAVE = .FALSE.", "LCHARG = .FALSE."]
        else:
            out = ["# Output", "LWAVE = .TRUE.", "LCHARG = .TRUE.", "LORBIT = 11"]
            if elf:
                out.append("LELF = .TRUE.   # electron localization function -> ELFCAR")
        lines.extend(out)
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
            "ISTART = 1      ! read WAVECAR when present (VASP falls back to 0 if absent)",
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
        lines.extend(["# Output", "LWAVE = .FALSE.", "LCHARG = .FALSE.",
                      "LORBIT = 11     ! orbital projections -> PROCAR (fat bands)"])
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
        isym = str(self.instructions.get('lobster_isym') or '0')
        if isym not in ('0', '-1'):              # LOBSTER requires symmetry off
            isym = '0'
        if self.instructions.get('soc', False):
            isym = '-1'                          # SOC breaks time reversal: ISYM=0 is not enough
        lines += [
            "",
            "# Non-self-consistent from SCF CHGCAR; symmetry OFF is REQUIRED by LOBSTER",
            "ICHARG = 11",
            "ISTART = 1",
            f"ISYM = {isym}",
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
        u = self._u_lines()
        if u:   # the GGA+U block carries its own LMAXMIX -- keep one
            lines = [l for l in lines if not l.startswith('LMAXMIX')]
        lines += u
        lines.extend(self._get_parallel_lines('lobster'))
        lines.extend([
            "# Output (WAVECAR is read by LOBSTER; dense DOSCAR for the energy window)",
            "LORBIT = 11",
            "NEDOS = 3000",
            "LWAVE = .TRUE.",
            "LCHARG = .FALSE.",
        ])
        lines = self._apply_incar_overrides(lines, 'lobster')
        return '\n'.join(lines) + '\n'

    # ── k-point meshes ────────────────────────────────────────────────────────
    # Density vocabulary: 'coarse' = 1000 kpra, 'fine' = 5000 kpra, or an
    # integer kpra.  kpra = (N1*N2*N3) * N_atoms  (k-points per reciprocal atom).
    #
    # Precedence, highest first (applied per step by kpoints_text()):
    #   1. an explicit `KPOINTS <step>:` / `KPOINTS:` block (written verbatim)
    #   2. the explicit `KMESH: n1 n2 n3` override
    #   3. the density (KMESH_DENSITY / RELAX_KMESH_DENSITY, default fine / coarse)
    # Steps derived from the SCF (DOS, LOBSTER = 2x; DFPT, Wannier = 1x) follow
    # the SCF mesh actually in force, including an explicit SCF block.

    @staticmethod
    def _kpra_of(density) -> int:
        """k-points-per-reciprocal-atom target for a density value."""
        if isinstance(density, (int, float)) and not isinstance(density, bool):
            return max(1, int(density))
        s = str(density).strip().lower()
        if s.isdigit():
            return max(1, int(s))
        return {'coarse': 1000, 'fine': 5000}.get(s, 5000)

    def _density_label(self, density) -> str:
        s = str(density).strip().lower()
        kpra = self._kpra_of(density)
        return f"{s}, {kpra} kpra" if s in ('coarse', 'fine') else f"{kpra} kpra"

    def _relax_density(self):
        return self.instructions.get('relax_kmesh_density') or 'coarse'

    def _explicit_kpoints(self, step: str):
        """Verbatim KPOINTS text the user supplied for *step*, or None.
        A step block wins over the unnamed (all-steps) block, which never
        applies to the line-mode band-structure path."""
        raw = self.instructions.get('kpoints_raw') or {}
        if step in raw:
            return raw[step]
        if step != 'bands' and 'all' in raw:
            return raw['all']
        return None

    @staticmethod
    def _parse_kpoints_text(text: str):
        """Classify a KPOINTS file: ('mesh', (n1,n2,n3)) for an automatic
        Gamma/Monkhorst-Pack mesh, ('list', n) / ('line', n) for explicit
        points, or (None, None) if unreadable."""
        try:
            L = [l.strip() for l in text.splitlines()]
            nk = int(L[1].split()[0])
            mode = L[2][:1].upper()
            if nk == 0 and mode in ('G', 'M'):
                n = [int(float(x)) for x in L[3].split()[:3]]
                if len(n) == 1:
                    n = n * 3
                return 'mesh', tuple(n)
            if mode == 'L':
                return 'line', nk
            return 'list', nk
        except Exception:
            return None, None

    def _kpra_atoms(self) -> int:
        return max(1, self._read_atom_count())

    def _relax_mesh(self) -> tuple:
        text = self._explicit_kpoints('relax')
        if text:
            kind, val = self._parse_kpoints_text(text)
            if kind == 'mesh':
                return val
        return self._compute_mesh(self._relax_density())

    def _scf_mesh(self, density=None) -> tuple:
        """The mesh the SCF step runs (explicit SCF block > KMESH > density)."""
        text = self._explicit_kpoints('scf')
        if text:
            kind, val = self._parse_kpoints_text(text)
            if kind == 'mesh':
                return val
        return self._compute_mesh(density if density is not None
                                  else self.instructions.get('kmesh_density', 'fine'))

    def _is_hex_lattice(self) -> bool:
        """Geometric test only (ignores the GUI 'Hexagonal BZ' flag, which is
        ticked by default): |a1| = |a2| and the in-plane angle 120 (or 60) deg,
        a3 perpendicular to the plane."""
        try:
            with open(self.poscar) as f:
                ls = f.readlines()
            sc = float(ls[1].strip())
            a1, a2, a3 = [np.array([float(x) * sc for x in ls[i].split()[:3]]) for i in (2, 3, 4)]
            n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
            cos_g = abs(np.dot(a1, a2) / (n1 * n2))
            perp  = max(abs(np.dot(a3, a1)) / (np.linalg.norm(a3) * n1),
                        abs(np.dot(a3, a2)) / (np.linalg.norm(a3) * n2))
            return abs(n1 - n2) / n1 < 0.01 and abs(cos_g - 0.5) < 0.02 and perp < 0.02
        except Exception:
            return False

    def _is_hexagonal_cell(self) -> bool:
        """True when the lattice is hexagonal/trigonal: a=b and γ≈120°.
        Informational only (comments, k-path M point); the mesh itself follows
        the kpra target for every cell type."""
        if self.instructions.get('is_hex'):
            return True
        try:
            with open(self.poscar) as f:
                ls = f.readlines()
            sc = float(ls[1].strip())
            a1 = np.array([float(x) * sc for x in ls[2].split()[:3]])
            a2 = np.array([float(x) * sc for x in ls[3].split()[:3]])
            cos_g = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
            return (abs(np.linalg.norm(a1) - np.linalg.norm(a2)) / np.linalg.norm(a1) < 0.01
                    and abs(cos_g + 0.5) < 0.02)
        except Exception:
            return False

    def _compute_mesh(self, density) -> tuple:
        """Central mesh resolver: (Nx, Ny, Nz) for a density.

        1. Explicit `KMESH: n1 n2 n3` instruction (any system).
        2. Otherwise the kpra target (coarse=1000, fine=5000, or an integer):
           subdivisions ∝ |b_i*|, even integers, chosen so N1*N2*N3*N_atoms
           lands as close to the target as possible (see _kpoints_from_kpra).
        """
        kmesh = self.instructions.get('kmesh')
        if kmesh:
            return tuple(kmesh[:3])
        return self._kpoints_from_kpra(self._kpra_of(density))

    def _kpoints_from_kpra(self, kpra: int) -> tuple:
        """(Nx, Ny, Nz) Gamma mesh for a density of *kpra* k-points per
        reciprocal atom.  One formula, no search:

            s   = ( kpra / N_atoms / (|b1*| |b2*| |b3*|) )^(1/3)
            N_i = s * |b_i*|  rounded to the nearest multiple of 2 (minimum 2)

        so the spacing |b_i*|/N_i is the same along every axis (uniform mesh)
        up to that rounding, and N1*N2*N3*N_atoms ~ kpra.  Hexagonal cells round
        the two in-plane N to multiples of 6 (K and M on the mesh); 2-D slabs
        use the in-plane version of the formula and Nz = 1.
        """
        import math
        try:
            with open(self.poscar) as f:
                lines = f.readlines()
            scale = float(lines[1].strip())
            A = np.array([[float(x) * scale for x in lines[i].split()[:3]]
                          for i in range(2, 5)])
            b = np.linalg.norm(2 * np.pi * np.linalg.inv(A).T, axis=1)
        except Exception:
            return (6, 6, 6)
        n_k  = max(1.0, kpra / self._kpra_atoms())
        flat = bool(self.instructions.get('is_2d', False))
        bb   = b[:2] if flat else b[:3]
        s    = (n_k / float(np.prod(bb))) ** (1.0 / len(bb))
        hexa = self._is_hex_lattice()
        mesh = []
        for i, x in enumerate(bb):
            step = 6 if (hexa and i < 2) else 2
            mesh.append(max(step, step * int(math.floor(s * x / step + 0.5))))
        return tuple(mesh) + ((1,) if flat else ())

    def _spacing_note(self, mesh) -> str:
        """', spacing 0.100-0.104 1/A' — |b_i*|/N_i range (uniformity check)."""
        try:
            with open(self.poscar) as f:
                lines = f.readlines()
            sc = float(lines[1].strip())
            A = np.array([[float(x) * sc for x in lines[i].split()[:3]] for i in range(2, 5)])
            b = np.linalg.norm(2 * np.pi * np.linalg.inv(A).T, axis=1)
            sp = [b[i] / mesh[i] for i in range(3) if mesh[i] > 1]
            return f", spacing {min(sp):.3f}-{max(sp):.3f} 1/A"
        except Exception:
            return ""

    def _kpoints_auto_text(self, mesh: tuple, comment: str) -> str:
        nx, ny, nz = mesh
        actual = nx * ny * nz * self._kpra_atoms()
        return (f"{comment}, {nx}x{ny}x{nz} = {actual} kpra{self._spacing_note(mesh)}\n"
                f"0\nGamma\n  {nx}  {ny}  {nz}\n  0    0    0\n")

    def _generate_kpoints_auto(self, density='fine') -> str:
        """Automatic Gamma-centred KPOINTS text for a density (see _compute_mesh)."""
        mesh = self._compute_mesh(density)
        if self.instructions.get('kmesh'):
            comment = "Automatic Gamma mesh (explicit KMESH override)"
        else:
            comment = f"Automatic Gamma mesh ({self._density_label(density)} target)"
        return self._kpoints_auto_text(mesh, comment)

    def _mesh_x2(self, density=None) -> tuple:
        """SCF mesh doubled in every direction, keeping the SCF ratios.

        An SCF mesh of 3×4×5 becomes 6×8×10.  Used by the DOS and LOBSTER
        NSCF steps, which read the SCF charge density and so cost far less
        than the SCF itself.  A 2D slab keeps nz = 1 (Gamma-only out of plane).
        Follows an explicit SCF KPOINTS block when one is given.
        """
        nx, ny, nz = self._scf_mesh(density)
        nz2 = 1 if self.instructions.get('is_2d') else nz * 2
        return nx * 2, ny * 2, nz2

    def _generate_kpoints_x2(self, density=None, label: str = 'NSCF') -> str:
        """KPOINTS file at 2× the SCF mesh (see _mesh_x2)."""
        nx, ny, nz = self._mesh_x2(density)
        return self._kpoints_auto_text((nx, ny, nz),
                                       f"Automatic Gamma mesh ({label}, 2x SCF)")

    def kpoints_text(self, step: str, density=None) -> str:
        """KPOINTS file content for *step* — the single source of truth used
        by every agent and by the GUI when it regenerates k-meshes.

        step: relax | scf | bands | dos | dfpt | wannier | phonons | lobster.
        An explicit KPOINTS block for the step always wins.  *density*, if
        given, replaces the instruction's density for the mesh-based steps
        (used by the GUI's second phase).
        """
        text = self._explicit_kpoints(step)
        if text:
            return text
        if step == 'bands':
            return self._generate_kpoints_linemode(self.instructions.get('kpath'))
        d = density if density is not None else self.instructions.get('kmesh_density', 'fine')
        if step == 'relax':
            return self._generate_kpoints_auto(
                density if density is not None else self._relax_density())
        if step in ('dos', 'lobster'):
            return self._generate_kpoints_x2(density, label=step.upper())
        if step == 'phonons':
            return self._generate_kpoints_auto('coarse')
        # scf, dfpt, wannier: the SCF mesh (an explicit SCF block is followed too)
        return self._kpoints_auto_text(
            self._scf_mesh(d),
            "Automatic Gamma mesh (explicit KMESH override)"
            if self.instructions.get('kmesh') else
            f"Automatic Gamma mesh ({self._density_label(d)} target)")

    def _write_kpoints(self, output_dir: str, step: str, density=None) -> str:
        """Write <output_dir>/KPOINTS for *step*; returns the text.  A marker
        file `.explicit_kpoints` is left next to KPOINTS when the user's own
        block was used, so later k-mesh patching (convergence phase 2) skips it."""
        text   = self.kpoints_text(step, density)
        marker = os.path.join(output_dir, '.explicit_kpoints')
        with open(os.path.join(output_dir, 'KPOINTS'), 'w') as f:
            f.write(text)
        if self._explicit_kpoints(step):
            open(marker, 'w').write('KPOINTS taken verbatim from the instructions file\n')
        elif os.path.exists(marker):
            os.remove(marker)
        return text


    def _spglib_kpath(self):
        """Spglib/Setyawan-Curtarolo high-symmetry k-path for the POSCAR.

        Uses spglib (via pymatgen's KPathSetyawanCurtarolo) to find the real
        space group and standard k-path, then transforms every high-symmetry
        point through Cartesian reciprocal space into THIS POSCAR's reciprocal
        basis — so the path is correct for the actual cell used in the SCF
        (which the bands NSCF must share via CHGCAR), regardless of orientation.

        Returns (segments, coords, name) where `segments` is a list of label
        lists (path discontinuities preserved) and `coords` maps label ->
        fractional coords in the POSCAR reciprocal basis. Returns None on any
        failure so the caller can fall back to the geometric classifier.
        """
        try:
            import spglib
            # Make spglib's dataset attribute-accessible (spglib>=2 may return a
            # plain dict that older pymatgen expects to have .number/.international).
            if not getattr(spglib, '_vf_ds_patched', False):
                _orig = spglib.get_symmetry_dataset
                class _DS(dict):
                    __getattr__ = dict.get
                def _patched(*a, **k):
                    d = _orig(*a, **k)
                    return _DS(d) if isinstance(d, dict) else d
                spglib.get_symmetry_dataset = _patched
                spglib._vf_ds_patched = True

            import warnings
            from pymatgen.core import Structure
            from pymatgen.symmetry.kpath import KPathSetyawanCurtarolo

            st = Structure.from_file(self.poscar)
            with warnings.catch_warnings():
                # pymatgen warns the cell may not be its standard primitive; we
                # transform every k-point into THIS cell's basis below, so the
                # path is correct regardless — silence the (expected) warning.
                warnings.simplefilter('ignore')
                kp = KPathSetyawanCurtarolo(st)
            B_std = kp.prim.lattice.reciprocal_lattice.matrix   # rows = recip vecs
            B_inv = np.linalg.inv(st.lattice.reciprocal_lattice.matrix)

            def clean(lbl):
                return (lbl.replace('\\Gamma', 'G').replace('\\', '')
                           .replace('_', '').replace('{', '').replace('}', ''))

            coords = {}
            for lbl, fr in kp.kpath['kpoints'].items():
                cart = np.array(fr, dtype=float) @ B_std
                coords[clean(lbl)] = cart @ B_inv
            segments = [[clean(l) for l in seg] for seg in kp.kpath['path']]
            if not segments or not coords:
                return None
            name = getattr(kp, 'name', None) or 'spglib'
            return segments, coords, name
        except Exception:
            return None

    def _generate_kpoints_linemode(self, kpath=None, npoints: int = 40) -> str:
        """Generate KPOINTS file for band structure.

        If kpath is None, derives the high-symmetry path from spglib
        (Setyawan-Curtarolo) for the POSCAR, falling back to a geometric
        Bravais-lattice classifier if spglib/pymatgen are unavailable.
        If kpath is a list of labels, uses those with the generic coord table
        (backward-compatible behaviour).
        """
        npoints = self.instructions.get('nkpts_bands') or npoints
        is_2d   = self.instructions.get('is_2d', False)

        # ── auto-detect via spglib (3-D only; slabs keep the in-plane path) ──
        if kpath is None and not is_2d:
            sk = self._spglib_kpath()
            if sk is not None:
                segments, kcoords, name = sk
                out = [f"Line-mode KPOINTS  [spglib {name} — auto-detected]\n",
                       f"{npoints}\n", "Line-mode\n", "rec\n"]
                for seg in segments:
                    for i in range(len(seg) - 1):
                        for lbl in (seg[i], seg[i + 1]):
                            c = kcoords.get(lbl, [0.0, 0.0, 0.0])
                            out.append(f"  {c[0]:8.4f} {c[1]:8.4f} {c[2]:8.4f}  ! {lbl}\n")
                        out.append("\n")
                return ''.join(out)
            # else fall through to the geometric classifier below

        if kpath is None:
            # ── auto-detect (geometric fallback / 2-D) ───────────────────
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
        if calc_type == 'scf':
            script += f"""
# Separate ELF pass (KPAR = 1), when set up
if [ -f "$HERE/run_elf.sh" ]; then
    bash "$HERE/run_elf.sh" {launch_cmd}
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
        sigma = self.instructions.get('lobster_sigma') or '0.05'
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

# 2) lobsterin: use the editable file written by vasp-agent if present; else
#    build one from the DOSCAR energy window (Fermi-referenced, E_F = 0).
LOBSTER_BIN="${LOBSTER_BIN:-@LOBSTER@}"
LOBSTER_BIN="${LOBSTER_BIN/#\~/$HOME}"   # a quoted ~ never expands in bash
if [ ! -f lobsterin ]; then
    if [ ! -f DOSCAR ]; then echo "ERROR: no DOSCAR (NSCF failed?)"; exit 1; fi
    read emax emin nedos efermi _ < <(sed -n '6p' DOSCAR)
    starteng=$(awk -v a="$emin" -v f="$efermi" 'BEGIN{printf "%.4f", a-f}')
    endeng=$(awk   -v a="$emax" -v f="$efermi" 'BEGIN{printf "%.4f", a-f}')
    cat > lobsterin <<LOB
basisSet pbeVaspFit2015
cohpGenerator from 0.8 to @BONDMAX@
cobiGenerator from 0.8 to @BONDMAX@
COHPStartEnergy $starteng
COHPEndEnergy $endeng
gaussianSmearingWidth @SIGMA@
LOB
fi

# 3) Run LOBSTER
echo "Running LOBSTER ($LOBSTER_BIN) at $(date)"
"$LOBSTER_BIN" > lobster.out 2>&1 || { echo "  LOBSTER failed -- see lobster.out"; exit 1; }
echo "  charge spilling:"; grep -i spilling lobster.out | grep '%' || true
echo "  Done: COHPCAR/COBICAR/COOPCAR + ICO*LIST.lobster in $HERE"
'''
        return (template.replace('@PREAMBLE@', preamble)
                        .replace('@VEXEC@', vasp_exec)
                        .replace('@LAUNCH@', launch_cmd)
                        .replace('@LOBSTER@', lobster_bin)
                        .replace('@SIGMA@', str(sigma))
                        .replace('@BONDMAX@', str(self._second_shell_cutoff())))
