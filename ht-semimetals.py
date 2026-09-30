#!/usr/bin/env python3
"""
High-Throughput Semimetal Screen -> SCF + bands + LOBSTER (SLURM highthroughput directory)
==============================================================================
Builds a self-contained, transferable highthroughput directory that runs three steps per material
on a SLURM cluster:

    02_scf      self-consistent field  (Gamma-centred, commensurate k-mesh)
    03_bands    band structure         (spglib Setyawan-Curtarolo line mode)
    08_lobster  symmetry-off NSCF + LOBSTER COHP/COBI/COOP

Structure download and MP access are reused from ht-mp-scf.py; input-file
generation is delegated to vasp-agent-slurm.py, exactly as in the existing
high-throughput driver.  This script adds the screening-specific parts:

  * primitive-cell download with resume, and a hard <= --max-atoms filter;
  * a per-material Gamma k-mesh at fixed --kspacing, with N_i proportional to
    |b_i*| so that N_i * a_i is constant (commensurate with a, b, c) -- the
    right choice for anisotropic primitive cells and dense enough for the
    Fermi-surface features of a semimetal;
  * per-material ENCUT from the POTCAR ENMAX values (1.3 x max ENMAX);
  * automatic spin polarisation for cells containing 3d magnetic elements;
  * the LOBSTER NSCF capped at 1x the SCF mesh (it runs ISYM=0, so its cost is
    the *full* mesh -- 2x would make it dominate the whole screen);
  * per-material SLURM sizing (nodes / walltime) from the electron count and
    the k-point count, capped at --max-nodes x --cores-per-node;
  * a highthroughput directory with one job per material, a throttled submitter, a POTCAR
    builder, a status reporter and a result collector.

POTCARs are NOT written into the highthroughput directory (licensed, and ~0.9 GB for a full
screen).  ENCUT / NBANDS are computed here from the local POTCAR library and
baked into the INCARs; make_potcars.sh rebuilds the POTCAR files on the HPC
from $VASP_POTCAR_DIR and verifies them against potcar_manifest.json.
Use --include-potcars if the cluster has no POTCAR library.

Typical use
-----------
    export MP_API_KEY=...
    export VASP_POTCAR_DIR=...

    ./ht-semimetals.py --list lists/semimetals_smoke.txt --out highthroughput_smoke
    ./ht-semimetals.py --list lists/semimetals_test.txt  --out highthroughput_test
    ./ht-semimetals.py --list lists/semimetals_full.txt  --out highthroughput_full

Add  --kpra 8000 --single-node  for the semimetal production settings
(k-point density 8000, one 40-core node per material).
The three runs are identical in every setting; only the --list file (the
materials) differs.  To try a different subset, write a new list file rather
than passing ad-hoc overrides, so the run stays reproducible from its list.

Then transfer <out>/ to the cluster and follow <out>/README_HPC.md.
"""

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_DIR, 'modules'))
AGENT_SLURM = os.path.join(REPO_DIR, 'vasp-agent-slurm.py')

# ── defaults ────────────────────────────────────────────────────────────────
KSPACING        = 0.10      # A^-1; N_i = even(ceil(|b_i*| / KSPACING))
MAX_ATOMS       = 20        # primitive-cell atom limit
CORES_PER_NODE  = 40
MAX_NODES       = 4
PARTITION       = 'standard'
ACCOUNT         = 'elmgroup'
ENCUT_FLOOR     = 400       # eV; never go below the PBE default
ENCUT_FACTOR    = 1.3       # x max(ENMAX); hard cutoff for LOBSTER/bands consistency

# Work metric W = NELECT^2 * N_kpoints / 1e6.  NELECT captures both how many
# atoms there are and what kind they are (valence electrons per POTCAR), and
# NELECT^2 tracks the NBANDS^2 scaling of the diagonalisation.  N_kpoints is
# the FULL mesh because the LOBSTER NSCF runs ISYM=0 and dominates the cost.
# Tier table: (W_upper_bound, nodes, cores_per_node, walltime, partition).
# None as the bound means "everything above the previous tier".
#
# Sized for Rivanna as reported by sinfo:
#   standard  40-core nodes, 301 of them, 7-day limit
#   parallel  96-core nodes, 179 of them, 3-day limit
# Small and medium cells stay single-node on standard: they do not scale well
# past ~40 ranks, so extra nodes buy little.  Only the heaviest 10% go wide, on parallel's 96-core
# nodes -- requesting 40 tasks there would strand 56 cores per node.
TIERS = [
    (5.0,  1, 40, '24:00:00', 'standard'),
    (25.0, 1, 40, '72:00:00', 'standard'),
    (None, 2, 96, '72:00:00', 'parallel'),
]

# 3d elements that need a spin-polarised starting guess.
MAGNETIC_3D = {'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni'}
HIGH_MOMENT = 5.0
LOW_MOMENT  = 0.6

STAGE_DIR = '_ht_inputs'


def _load_module(name, filename):
    """Import a repo script whose filename contains dashes."""
    path = os.path.join(REPO_DIR, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── POSCAR / lattice ────────────────────────────────────────────────────────
def read_poscar(path):
    """Return {'A': 3x3 lattice (A), 'species': [...], 'counts': [...]}."""
    with open(path) as f:
        lines = f.readlines()
    scale = float(lines[1].split()[0])
    A = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:                      # negative scale = target volume
        scale = (-scale / abs(np.linalg.det(A))) ** (1 / 3)
    A = A * scale
    species = lines[5].split()
    counts  = [int(x) for x in lines[6].split()]
    if not species or species[0].isdigit():
        raise ValueError(f"{path}: not a VASP5 POSCAR (no element-symbol line)")
    return {'A': A, 'species': species, 'counts': counts,
            'natoms': sum(counts)}


def kmesh_from_lattice(A, kspacing=KSPACING):
    """Gamma-centred mesh with N_i proportional to |b_i*|.

    b_i* = 2*pi * (A^-1)^T rows.  Taking N_i = |b_i*| / kspacing makes the
    spacing between k-points equal along every reciprocal axis, which for an
    orthogonal cell is exactly N_1*a = N_2*b = N_3*c.  For monoclinic and
    triclinic cells the reciprocal-vector lengths are the correct
    generalisation.  Each N_i is rounded up to an even number so the BZ
    boundary (k = 1/2) is always sampled -- important for semimetals, whose
    band touchings sit at high-symmetry points.
    """
    B = 2 * np.pi * np.linalg.inv(A).T
    b = np.linalg.norm(B, axis=1)
    mesh = []
    for bi in b:
        n = int(math.ceil(bi / kspacing))
        if n % 2:
            n += 1
        mesh.append(max(2, n))
    return tuple(mesh)


# ── POTCAR library ──────────────────────────────────────────────────────────
def potcar_variant(element, potcar_dir, choices=None):
    """Resolve the POTCAR folder used for *element*.

    Mirrors build_potcar() in vasp-agent-slurm.py so that the ENMAX / ZVAL we
    read here belong to the very POTCAR the agent will concatenate.
    """
    choices = choices or {}
    variants = ([choices[element]] if element in choices
                else [element, f"{element}_sv", f"{element}_pv", f"{element}_d"])
    for v in variants:
        if os.path.isfile(os.path.join(potcar_dir, v, 'POTCAR')):
            return v
    return None


_ENMAX_RE = re.compile(r'ENMAX\s*=\s*([\d.]+)')
_ZVAL_RE  = re.compile(r'ZVAL\s*=\s*([\d.]+)')


def potcar_props(potcar_dir, variant):
    """Return (ENMAX, ZVAL) from the header of a single-element POTCAR."""
    path = os.path.join(potcar_dir, variant, 'POTCAR')
    with open(path, errors='ignore') as f:
        head = f.read(4000)
    enmax = _ENMAX_RE.search(head)
    zval  = _ZVAL_RE.search(head)
    if not enmax or not zval:
        raise ValueError(f"could not read ENMAX/ZVAL from {path}")
    return float(enmax.group(1)), float(zval.group(1))


def material_potcar_info(struct, potcar_dir):
    """Resolve POTCAR variants and derive ENCUT and NELECT for one material."""
    variants, enmaxes, nelect = [], [], 0.0
    for el, n in zip(struct['species'], struct['counts']):
        v = potcar_variant(el, potcar_dir)
        if v is None:
            raise ValueError(f"no POTCAR for element '{el}' in {potcar_dir}")
        enmax, zval = potcar_props(potcar_dir, v)
        variants.append(v)
        enmaxes.append(enmax)
        nelect += zval * n
    encut = max(ENCUT_FLOOR,
                int(math.ceil(ENCUT_FACTOR * max(enmaxes) / 10.0) * 10))
    return {'variants': variants, 'enmax': max(enmaxes),
            'encut': encut, 'nelect': nelect}


# ── per-material SLURM sizing ───────────────────────────────────────────────
def size_job(nelect, nk_full, tiers=None, max_nodes=MAX_NODES):
    """Return (nodes, cores_per_node, walltime, partition, W) for one material.

    W = NELECT^2 * N_kpoints / 1e6 -- see the TIERS comment.  The first tier
    whose upper bound exceeds W wins; the last tier is the catch-all.
    """
    tiers = tiers or TIERS
    W = (nelect ** 2) * nk_full / 1.0e6
    for bound, nodes, cores, walltime, partition in tiers:
        if bound is None or W < bound:
            return min(nodes, max_nodes), cores, walltime, partition, W
    bound, nodes, cores, walltime, partition = tiers[-1]
    return min(nodes, max_nodes), cores, walltime, partition, W


# ── instructions.txt ────────────────────────────────────────────────────────
def magmom_string(struct):
    """VASP MAGMOM for a spin-polarised start, or None if all species are
    non-magnetic.  High moment on the 3d magnetic species, small on the rest."""
    if not any(el in MAGNETIC_3D for el in struct['species']):
        return None
    parts = []
    for el, n in zip(struct['species'], struct['counts']):
        m = HIGH_MOMENT if el in MAGNETIC_3D else LOW_MOMENT
        parts.append(f"{n}*{m}")
    return ' '.join(parts)


def write_instructions(path, mp_id, struct, mesh, encut, nodes,
                       ntasks_per_node, walltime, partition, account,
                       functional='PBE', kpra=None, kpar=None, ncore=None):
    """Write the instructions.txt consumed by vasp-agent-slurm.py.

    Task keywords are matched as substrings over the whole file by
    InstructionParser, so this text deliberately avoids words that would
    switch on steps we do not want (relax, phonon, wannier, transport, and
    anything containing 'dos').
    """
    nx, ny, nz = mesh
    lines = [
        f"Project: {mp_id}",
        "# High-throughput semimetal screen (auto-generated by ht-semimetals.py)",
        f"Methods: {functional} functional",
        "",
        # No DFPT: for a zero-gap system LEPSILON/Born charges are ill-defined
        # and the linear-response Davidson loop tends not to converge.
        "Tasks: SCF calculation, band structure, LOBSTER COHP/COBI analysis",
        "",
        *(["# k-point density: even N_i proportional to |b_i*| (uniform k-space",
           f"# spacing), N1*N2*N3*N_atoms as close to {kpra} as possible.",
           f"KMESH_DENSITY: {kpra}"] if kpra else
          ["# Gamma-centred mesh with N_i proportional to |b_i*|, i.e. N_i * a_i is",
           f"# constant across the three axes (k-spacing {KSPACING} A^-1).",
           f"KMESH: {nx} {ny} {nz}"]),
        f"ENCUT: {encut}",
        "",
        "# ELF is on by default in this repo, and LELF forces KPAR = 1 -- which",
        "# would serialise the k-point parallelism the SCF depends on here.",
        "# Turn it back on only if ELFCAR is wanted, and expect a much slower SCF.",
        "ELF: off",
        "",
        *([f"KPAR: {kpar}"] if kpar else []),
        *([f"NCORE: {ncore}"] if ncore else []),
        f"NODES: {nodes}",
        f"NTASKS_PER_NODE: {ntasks_per_node}",
        f"WALLTIME: {walltime}",
        f"PARTITION: {partition}",
        f"ACCOUNT: {account}",
    ]
    magmom = magmom_string(struct)
    if magmom:
        lines += [
            "",
            "# Spin-polarised start: cell contains a 3d magnetic element.",
            "INCAR:",
            "   ISPIN = 2",
            f"   MAGMOM = {magmom}",
            "END_INCAR",
        ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ── project post-processing ─────────────────────────────────────────────────
_CD_RE = re.compile(r'^cd /.*$', re.MULTILINE)
_PORTABLE_CD = 'cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"'


def make_run_sh_portable(step_dir):
    """Rewrite the agent's absolute paths in run.sh so the script still works
    after the highthroughput directory is copied to a different machine.

    The agent bakes in the generation-time absolute path (`cd /Users/...`),
    which does not exist on the cluster.  Everything the script touches lives
    beside it, so an equivalent relative `cd` is enough.
    """
    path = os.path.join(step_dir, 'run.sh')
    if not os.path.isfile(path):
        return
    with open(path) as f:
        text = f.read()
    text = _CD_RE.sub(_PORTABLE_CD, text)
    # `bash /abs/copy_from_scf.sh` -> `bash ./copy_from_scf.sh` (we cd'd first)
    text = re.sub(r'bash /\S*/(copy_from_\w+\.sh)', r'bash ./\1', text)
    with open(path, 'w') as f:
        f.write(text)
    os.chmod(path, 0o755)


def cap_lobster_mesh(proj_dir):
    """Set the LOBSTER NSCF mesh to 1x the SCF mesh.

    generate_lobster_input() defaults to 2x the SCF mesh, which is sensible for
    a single material but not here: the step runs ISYM=0, so its cost scales
    with the *full* mesh and 2x in each direction is 8x the k-points.  At the
    SCF density the COHP/COBI integrals are already well converged.
    """
    scf_kp = os.path.join(proj_dir, '02_scf', 'KPOINTS')
    lob_kp = os.path.join(proj_dir, '08_lobster', 'KPOINTS')
    if not (os.path.isfile(scf_kp) and os.path.isfile(lob_kp)):
        return None
    with open(scf_kp) as f:
        body = f.read().split('\n', 1)[1]
    with open(lob_kp, 'w') as f:
        f.write("Automatic Gamma mesh (LOBSTER, 1x SCF - ISYM=0, full mesh)\n")
        f.write(body)
    return True


def localise_analyze_sh(proj_dir, tools_rel):
    """Repoint analyze.sh at the highthroughput directory's own copy of the repo helper modules.

    The agent writes the generation-time absolute path to
    modules/lobster_postprocess.py, which does not exist on the cluster.
    """
    path = os.path.join(proj_dir, 'analyze.sh')
    if not os.path.isfile(path):
        return
    with open(path) as f:
        text = f.read()
    text = re.sub(r'"/[^"]*/modules/(\w+\.py)"',
                  r'"$HERE/%s/\1"' % tools_rel, text)
    with open(path, 'w') as f:
        f.write(text)
    os.chmod(path, 0o755)


def strip_potcars(proj_dir):
    """Remove POTCAR (and the per-step symlinks) from a staged project."""
    for root, _dirs, files in os.walk(proj_dir):
        for name in files:
            if name == 'POTCAR':
                os.remove(os.path.join(root, name))
    # os.walk does not report dangling symlinks as files on every platform
    for step in ('02_scf', '03_bands', '08_lobster'):
        link = os.path.join(proj_dir, step, 'POTCAR')
        if os.path.lexists(link):
            os.remove(link)
    link = os.path.join(proj_dir, 'POTCAR')
    if os.path.lexists(link):
        os.remove(link)


# ── per-material job script ─────────────────────────────────────────────────
JOB_TEMPLATE = r'''#!/bin/bash
#SBATCH --job-name=@ID@
#SBATCH --partition=@PARTITION@
#SBATCH --nodes=@NODES@
#SBATCH --ntasks-per-node=@NTASKS@
#SBATCH --time=@TIME@
#SBATCH --account=@ACCOUNT@
#SBATCH --output=@ID@-%j.out
#SBATCH --error=@ID@-%j.err
#
# @ID@  --  @FORMULA@, @NATOMS@ atoms, NELECT=@NELECT@, mesh @MESH@ (@NK@ k-points)
# Sized from the electron count and k-point count: W=@W@ -> @NODES@ node(s).
# Runs 02_scf -> 03_bands -> 08_lobster in one allocation.
# Steps that already carry a finished OUTCAR are skipped, so re-submitting
# this script after a timeout resumes rather than restarting.

set -u

# Locate this material's directory.
#
# NOTE: under sbatch, SLURM copies the script to /var/spool/slurm/.../slurm_script
# before running it, so ${BASH_SOURCE[0]} points at the spool copy, NOT at the
# highthroughput directory.  SLURM_SUBMIT_DIR is the directory sbatch was invoked from, which is
# the reliable anchor.  Accept it whether the job was submitted from inside the
# material directory or from the highthroughput root, and fall back to the script's own
# location when the script is run directly with bash (no SLURM).
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/02_scf" ]; then
    PROJ="$SLURM_SUBMIT_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -d "$SLURM_SUBMIT_DIR/materials/@ID@/02_scf" ]; then
    PROJ="$SLURM_SUBMIT_DIR/materials/@ID@"
else
    PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [ ! -d "$PROJ/02_scf" ]; then
    echo "ERROR: cannot locate the @ID@ directory (tried '$PROJ')." >&2
    echo "       Submit with:  cd <highthroughput>/materials/@ID@ && sbatch job.sbatch" >&2
    exit 1
fi
cd "$PROJ"

ENV_SH="$PROJ/../../env.sh"
if [ ! -f "$ENV_SH" ]; then
    echo "ERROR: env.sh not found at $ENV_SH" >&2
    exit 1
fi
source "$ENV_SH"

echo "=== @ID@ (@FORMULA@) on $(hostname) ==="
echo "Started: $(date)"
FAILED=0

finished() {   # a VASP step is done when OUTCAR carries the final timing block
    [ -f "$1/OUTCAR" ] && grep -q "General timing and accounting" "$1/OUTCAR"
}

run_vasp_step() {
    local step=$1
    if finished "$step"; then
        echo "--- $step already complete, skipping"
        return 0
    fi
    if [ ! -d "$step" ]; then
        # Not an optional condition: every material is generated with all four
        # steps, so a missing directory means the job is looking in the wrong
        # place.  Fail loudly rather than "succeeding" without doing anything.
        echo "!!! $step directory missing under $PROJ - wrong working directory?"
        return 1
    fi
    echo "--- $step : $(date)"
    local rc=0
    cd "$PROJ/$step"
    [ -f copy_from_scf.sh ] && bash ./copy_from_scf.sh
    $VASP_LAUNCH "$VASP_STD"
    rc=$?
    cd "$PROJ"
    if ! finished "$step"; then
        echo "!!! $step did not reach the final timing block (exit $rc)"
        return 1
    fi
    echo "--- $step done"
    return 0
}

run_vasp_step 02_scf   || { echo "SCF failed - dependent steps skipped"; exit 1; }
run_vasp_step 03_bands || { echo "WARNING: bands step failed, continuing"; FAILED=1; }

# ── LOBSTER: symmetry-off NSCF, then the LOBSTER binary on the same node ────
if run_vasp_step 08_lobster; then
    cd "$PROJ/08_lobster"
    if [ -f lobster.out ] && grep -q "finished in" lobster.out; then
        echo "--- LOBSTER already complete, skipping"
    else
        echo "--- LOBSTER binary : $(date)"
        if "$LOBSTER_BIN" > lobster.out 2>&1; then
            grep -q "finished in" lobster.out || FAILED=1
        else
            echo "!!! LOBSTER failed - see lobster.out"
            FAILED=1
        fi
    fi
    # WAVECAR is large and is only needed by LOBSTER itself.
    [ "${KEEP_WAVECAR:-0}" = "1" ] || rm -f WAVECAR
    cd "$PROJ"
else
    echo "WARNING: LOBSTER NSCF failed"
    FAILED=1
fi

# CHGCAR/WAVECAR copies dominate the footprint of a 1865-material screen.
if [ "${KEEP_LARGE_FILES:-0}" != "1" ]; then
    rm -f "$PROJ"/03_bands/CHGCAR
    rm -f "$PROJ"/08_lobster/CHGCAR
fi

# ── Analysis: band / COHP / COBI / COOP / LOBSTER-DOS plots + bonding CSV ────
# Runs analyze.sh on the compute node once the VASP/LOBSTER steps are done.
# Never fails the job: a missing numpy/matplotlib only skips the plots, and
# ./analyze_all.sh can redo them later from the login node.
if [ "${RUN_ANALYSIS:-1}" = "1" ] && [ -f "$PROJ/analyze.sh" ]; then
    [ -n "${ANALYSIS_PYTHON:-}" ] && export PATH="$(dirname "$ANALYSIS_PYTHON"):$PATH"
    if python3 -c "import numpy, matplotlib" 2>/dev/null; then
        echo "--- analysis       : $(date)"
        mkdir -p "$PROJ/analysis"
        MPLBACKEND=Agg bash "$PROJ/analyze.sh" > "$PROJ/analysis/analyze.log" 2>&1 \
            || echo "WARNING: analyze.sh reported errors - see analysis/analyze.log"
    else
        echo "WARNING: python3 without numpy/matplotlib - analysis skipped."
        echo "         Set ANALYSIS_PYTHON in env.sh, then run ./analyze_all.sh."
    fi
fi

echo "Finished: $(date)"
# .done is what submit_all.sh uses to skip a material.  Only write it when
# every step succeeded, so a partially-failed material is retried on the next
# pass (finished steps are skipped, so the retry is cheap).
if [ "$FAILED" -eq 0 ]; then
    touch "$PROJ/.done"
else
    echo "One or more steps failed - not marking .done; re-submit to retry."
    exit 1
fi
'''


def write_job_script(proj_dir, mp_id, meta, partition, account):
    path = os.path.join(proj_dir, 'job.sbatch')
    text = (JOB_TEMPLATE
            .replace('@ID@',        mp_id)
            .replace('@PARTITION@', partition)
            .replace('@ACCOUNT@',   account)
            .replace('@NODES@',     str(meta['nodes']))
            .replace('@NTASKS@',    str(meta['ntasks_per_node']))
            .replace('@TIME@',      meta['walltime'])
            .replace('@FORMULA@',   meta['formula'])
            .replace('@NATOMS@',    str(meta['natoms']))
            .replace('@NELECT@',    f"{meta['nelect']:.0f}")
            .replace('@MESH@',      'x'.join(str(m) for m in meta['mesh']))
            .replace('@NK@',        str(meta['nk_full']))
            .replace('@W@',         f"{meta['W']:.1f}"))
    with open(path, 'w') as f:
        f.write(text)
    os.chmod(path, 0o755)


# ── highthroughput-level scripts ────────────────────────────────────────────────────
ENV_SH = r'''#!/bin/bash
# ============================================================================
# env.sh -- the ONE file to edit on the cluster.
# Sourced by every material's job.sbatch.  #SBATCH headers cannot read shell
# variables, so partition/account/walltime live in the job scripts themselves;
# retune.sh rewrites those in place if they need to change.
# ============================================================================

# --- modules -----------------------------------------------------------------
module purge 2>/dev/null
@MODULES@

# --- executables -------------------------------------------------------------
# How VASP is launched.  'srun' inherits the allocation from #SBATCH.
export VASP_LAUNCH="srun"
export VASP_STD="@VASP_STD@"

# LOBSTER binary (must be the LINUX build -- the macOS one will not run here).
export LOBSTER_BIN="@LOBSTER_BIN@"

# POTCAR library: directory holding element sub-folders (Bi/, Fe_pv/, ...).
# Used by make_potcars.sh only; the jobs themselves read the assembled POTCARs.
export VASP_POTCAR_DIR="${VASP_POTCAR_DIR:-@POTCAR_DIR@}"

# --- normalise and check -----------------------------------------------------
# A leading ~ is NOT expanded inside quotes, so "~/BIN/vasp_std" would stay a
# literal string and every srun would fail with "No such file or directory".
# Rewrite it to $HOME here so either spelling works.
VASP_STD="${VASP_STD/#\~/$HOME}"
LOBSTER_BIN="${LOBSTER_BIN/#\~/$HOME}"
VASP_POTCAR_DIR="${VASP_POTCAR_DIR/#\~/$HOME}"
export VASP_STD LOBSTER_BIN VASP_POTCAR_DIR

# Fail loudly and immediately rather than burning an allocation. Skipped when
# VASPFLOW_NO_CHECK=1 (e.g. if the binary only exists on the compute nodes).
if [ "${VASPFLOW_NO_CHECK:-0}" != "1" ]; then
    command -v "$VASP_STD" >/dev/null 2>&1 || [ -x "$VASP_STD" ] || {
        echo "ERROR: VASP_STD not found or not executable: $VASP_STD" >&2
        echo "       Edit env.sh. Use \$HOME/... , not ~/... , inside quotes." >&2
        exit 1
    }
    command -v "$LOBSTER_BIN" >/dev/null 2>&1 || [ -x "$LOBSTER_BIN" ] || {
        echo "WARNING: LOBSTER_BIN not found or not executable: $LOBSTER_BIN" >&2
        echo "         The three VASP steps will still run; 08_lobster will not." >&2
    }
fi

# --- analysis ------------------------------------------------------------------
# Each job runs analyze.sh (plots + LOBSTER CSV) after its last step.  It needs
# a python3 with numpy + matplotlib; module purge above usually leaves only the
# system python, so point ANALYSIS_PYTHON at a python that has them, e.g.
#   export ANALYSIS_PYTHON="$HOME/vaspenv/bin/python3"
export RUN_ANALYSIS=1
export ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-}"

# --- disk policy -------------------------------------------------------------
# CHGCAR/WAVECAR copies are deleted after each material by default.
# Set to 1 to keep them (needs far more scratch for a full screen).
export KEEP_LARGE_FILES=0
export KEEP_WAVECAR=0
'''

MAKE_POTCARS = r'''#!/bin/bash
# Build each material's POTCAR from $VASP_POTCAR_DIR and verify it against the
# manifest recorded when the highthroughput directory was generated.  ENCUT and NBANDS in the
# INCARs were derived from those ENMAX/ZVAL values, so a mismatch here means
# the INCARs do not match the cluster's POTCAR library and must be regenerated.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script only needs the POTCAR library, not VASP itself -- skip env.sh's
# executable check so a missing/unbuilt vasp_std cannot block POTCAR assembly.
VASPFLOW_NO_CHECK=1 source "$HERE/env.sh" 2>/dev/null

if [ -z "${VASP_POTCAR_DIR:-}" ] || [ ! -d "$VASP_POTCAR_DIR" ]; then
    echo "ERROR: VASP_POTCAR_DIR is not set or does not exist."
    echo "       Edit env.sh (or export it) to point at the PAW_PBE library."
    exit 1
fi
echo "POTCAR library: $VASP_POTCAR_DIR"

python3 - "$HERE" "$VASP_POTCAR_DIR" <<'PYEND'
import json, os, re, sys
here, lib = sys.argv[1], sys.argv[2]
manifest = json.load(open(os.path.join(here, 'potcar_manifest.json')))
enmax_re = re.compile(r'ENMAX\s*=\s*([\d.]+)')
zval_re  = re.compile(r'ZVAL\s*=\s*([\d.]+)')

cache, bad = {}, []
def props(variant):
    if variant not in cache:
        p = os.path.join(lib, variant, 'POTCAR')
        if not os.path.isfile(p):
            cache[variant] = None
        else:
            head = open(p, errors='ignore').read(4000)
            m1, m2 = enmax_re.search(head), zval_re.search(head)
            cache[variant] = (float(m1.group(1)), float(m2.group(1))) if m1 and m2 else None
    return cache[variant]

built = missing = 0
for mp_id, info in sorted(manifest['materials'].items()):
    d = os.path.join(here, 'materials', mp_id)
    if not os.path.isdir(d):
        continue
    chunks, ok = [], True
    for variant, ref_enmax, ref_zval in info['potcars']:
        pr = props(variant)
        if pr is None:
            bad.append(f"{mp_id}: no POTCAR folder '{variant}' in the library")
            ok = False
            break
        if abs(pr[0] - ref_enmax) > 0.01 or abs(pr[1] - ref_zval) > 0.01:
            bad.append(f"{mp_id}: {variant} ENMAX/ZVAL {pr} != recorded "
                       f"({ref_enmax}, {ref_zval})")
            ok = False
            break
        chunks.append(os.path.join(lib, variant, 'POTCAR'))
    if not ok:
        missing += 1
        continue
    out = os.path.join(d, 'POTCAR')
    with open(out, 'wb') as fh:
        for c in chunks:
            fh.write(open(c, 'rb').read())
    for step in ('02_scf', '03_bands', '08_lobster'):
        s = os.path.join(d, step)
        if os.path.isdir(s):
            link = os.path.join(s, 'POTCAR')
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(os.path.join('..', 'POTCAR'), link)
    built += 1

print(f"\nPOTCARs built : {built}")
print(f"failed        : {missing}")
for line in bad[:20]:
    print("  " + line)
if len(bad) > 20:
    print(f"  ... and {len(bad)-20} more")
sys.exit(1 if missing else 0)
PYEND
'''

SUBMIT_ALL = r'''#!/bin/bash
# Submit one job per material, keeping at most $MAX_QUEUED of our own jobs in
# the queue at a time.  Safe to re-run: materials that already carry .done, or
# that currently have a job in the queue, are skipped -- so this doubles as the
# resume command after a batch of timeouts.
#
#   ./submit_all.sh              submit everything (throttled)
#   ./submit_all.sh 20           submit at most 20 materials this pass
#   MAX_QUEUED=100 ./submit_all.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_QUEUED="${MAX_QUEUED:-50}"
LIMIT="${1:-0}"
USER_NAME="${USER:-$(whoami)}"

queued() { squeue -h -u "$USER_NAME" -o '%j' 2>/dev/null | wc -l | tr -d ' '; }
in_queue() { squeue -h -u "$USER_NAME" -o '%j' 2>/dev/null | grep -qx "$1"; }

# Build any missing POTCARs automatically (licensed files are not shipped).
need_potcar=0
while read -r id; do
    [ -z "$id" ] && continue
    [ -d "$HERE/materials/$id" ] && [ ! -f "$HERE/materials/$id/POTCAR" ] && \
        [ ! -f "$HERE/materials/$id/.done" ] && { need_potcar=1; break; }
done < "$HERE/material_list.txt"
if [ "$need_potcar" = 1 ]; then
    echo "POTCARs missing -- running make_potcars.sh ..."
    "$HERE/make_potcars.sh" || { echo "ERROR: make_potcars.sh failed; fix VASP_POTCAR_DIR in env.sh"; exit 1; }
fi

submitted=0
while read -r id; do
    [ -z "$id" ] && continue
    d="$HERE/materials/$id"
    [ -d "$d" ] || continue
    [ -f "$d/.done" ] && continue
    in_queue "$id" && continue
    if [ ! -f "$d/POTCAR" ]; then
        echo "SKIP $id -- no POTCAR (make_potcars.sh did not build one)"
        continue
    fi
    while [ "$(queued)" -ge "$MAX_QUEUED" ]; do sleep 60; done
    jid=$(cd "$d" && sbatch --parsable job.sbatch) || { echo "FAILED $id"; continue; }
    echo "submitted $id -> $jid"
    submitted=$((submitted + 1))
    [ "$LIMIT" -gt 0 ] && [ "$submitted" -ge "$LIMIT" ] && break
done < "$HERE/material_list.txt"

echo ""
echo "Submitted $submitted job(s).  Monitor:  squeue -u $USER_NAME"
'''

STATUS_SH = r'''#!/bin/bash
# Per-step completion summary across the whole screen.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 - "$HERE" <<'PYEND'
import os, sys
here = sys.argv[1]
ids = [l.strip() for l in open(os.path.join(here, 'material_list.txt')) if l.strip()]
steps = ['02_scf', '03_bands', '08_lobster']
done = {s: 0 for s in steps}
lob = full = 0
incomplete = []
for mp_id in ids:
    d = os.path.join(here, 'materials', mp_id)
    ok = True
    for s in steps:
        o = os.path.join(d, s, 'OUTCAR')
        if os.path.isfile(o):
            try:
                tail = open(o, errors='ignore').read()[-4000:]
            except OSError:
                tail = ''
            if 'General timing and accounting' in tail:
                done[s] += 1
                continue
        ok = False
    lf = os.path.join(d, '08_lobster', 'lobster.out')
    if os.path.isfile(lf) and 'finished in' in open(lf, errors='ignore').read():
        lob += 1
    else:
        ok = False
    if ok:
        full += 1
    else:
        incomplete.append(mp_id)
n = len(ids)
print(f"materials        : {n}")
for s in steps:
    print(f"  {s:<12} : {done[s]:5d} / {n}")
print(f"  LOBSTER binary : {lob:5d} / {n}")
print(f"fully complete   : {full} / {n}")
if incomplete:
    print(f"\nincomplete ({len(incomplete)}), first 20:")
    print('  ' + ' '.join(incomplete[:20]))
    with open(os.path.join(here, 'incomplete.txt'), 'w') as fh:
        fh.write('\n'.join(incomplete) + '\n')
    print("  full list -> incomplete.txt")
PYEND
'''

COLLECT_SH = r'''#!/bin/bash
# Aggregate the finished screen into one small tarball to bring back home:
# LOBSTER bonding CSVs, band eigenvalues,
# and every INCAR/KPOINTS for provenance.  No CHGCAR/WAVECAR/vasprun.xml.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/results"
mkdir -p "$OUT"

while read -r id; do
    [ -z "$id" ] && continue
    d="$HERE/materials/$id"
    [ -d "$d" ] || continue
    o="$OUT/$id"; mkdir -p "$o"
    for f in ICOHPLIST.lobster ICOBILIST.lobster ICOOPLIST.lobster \
             COHPCAR.lobster COBICAR.lobster COOPCAR.lobster \
             DOSCAR.lobster CHARGE.lobster lobsterin lobster.out; do
        [ -f "$d/08_lobster/$f" ] && cp "$d/08_lobster/$f" "$o/" 2>/dev/null
    done
    [ -f "$d/03_bands/EIGENVAL" ] && cp "$d/03_bands/EIGENVAL" "$o/" 2>/dev/null
    [ -f "$d/03_bands/KPOINTS" ]  && cp "$d/03_bands/KPOINTS" "$o/KPOINTS.bands" 2>/dev/null
    [ -f "$d/02_scf/OUTCAR" ] && \
        grep -E "free  energy|E-fermi" "$d/02_scf/OUTCAR" | tail -5 > "$o/SCF_summary.txt" 2>/dev/null
    cp "$d/POSCAR" "$o/" 2>/dev/null
    for s in 02_scf 03_bands 08_lobster; do
        [ -f "$d/$s/INCAR" ] && cp "$d/$s/INCAR" "$o/INCAR.$s" 2>/dev/null
    done
    [ -f "$d/02_scf/KPOINTS" ] && cp "$d/02_scf/KPOINTS" "$o/KPOINTS.scf" 2>/dev/null
done < "$HERE/material_list.txt"

cp "$HERE/materials_summary.csv" "$OUT/" 2>/dev/null
tar czf "$HERE/screen_results.tar.gz" -C "$HERE" results
echo "Wrote $HERE/screen_results.tar.gz"
du -sh "$HERE/screen_results.tar.gz"
'''

RETUNE_SH = r'''#!/bin/bash
# Rewrite the #SBATCH partition / account / walltime in every job.sbatch.
# Use this if the cluster's partition name or your allocation differs from
# what the highthroughput directory was generated with.
#
#   ./retune.sh --partition compute --account elmgroup
#   ./retune.sh --time 72:00:00
#
# Many clusters restrict their default/serial partition to a SINGLE node and
# require a different partition for multi-node jobs.  --single/--multi sets
# each job's partition according to how many nodes it actually asks for:
#
#   ./retune.sh --single standard --multi parallel
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART=""; ACCT=""; TIME=""; SINGLE=""; MULTI=""
while [ $# -gt 0 ]; do
    case "$1" in
        --partition) PART="$2"; shift 2 ;;
        --account)   ACCT="$2"; shift 2 ;;
        --time)      TIME="$2"; shift 2 ;;
        --single)    SINGLE="$2"; shift 2 ;;
        --multi)     MULTI="$2"; shift 2 ;;
        *) echo "unknown option: $1"; exit 1 ;;
    esac
done
n=0; n1=0; nm=0
for f in "$HERE"/materials/*/job.sbatch; do
    if [ -n "$SINGLE" ] || [ -n "$MULTI" ]; then
        nodes=$(awk -F= '/^#SBATCH --nodes=/{print $2; exit}' "$f")
        if [ "${nodes:-1}" -le 1 ]; then
            [ -n "$SINGLE" ] && sed -i "s|^#SBATCH --partition=.*|#SBATCH --partition=$SINGLE|" "$f" && n1=$((n1 + 1))
        else
            [ -n "$MULTI" ] && sed -i "s|^#SBATCH --partition=.*|#SBATCH --partition=$MULTI|" "$f" && nm=$((nm + 1))
        fi
    fi
    [ -n "$PART" ] && sed -i "s|^#SBATCH --partition=.*|#SBATCH --partition=$PART|" "$f"
    [ -n "$ACCT" ] && sed -i "s|^#SBATCH --account=.*|#SBATCH --account=$ACCT|" "$f"
    [ -n "$TIME" ] && sed -i "s|^#SBATCH --time=.*|#SBATCH --time=$TIME|" "$f"
    n=$((n + 1))
done
echo "Updated $n job script(s)."
[ -n "$SINGLE" ] && echo "  single-node -> $SINGLE : $n1"
[ -n "$MULTI" ]  && echo "  multi-node  -> $MULTI  : $nm"
exit 0
'''


README_HPC = r'''# Semimetal high-throughput screen — HPC instructions

Generated by `ht-semimetals.py`.  Every material runs four VASP steps in a
single SLURM job:

    02_scf      SCF, Gamma-centred mesh, N_i proportional to |b_i*|
    03_bands    band structure, spglib high-symmetry line mode
    08_lobster  ISYM=0 NSCF from the SCF CHGCAR, then the LOBSTER binary

## 1. What to transfer

Copy this whole directory to the cluster:

    rsync -av --progress @HTDIR@/ user@hpc:/scratch/$USER/@HTDIR@/

Nothing else from the VASP-Flow repo is needed — the jobs are plain bash and
sbatch.  POTCARs are deliberately absent; step 3 rebuilds them there.

## 2. Configure (edit ONE file)

    cd /scratch/$USER/@HTDIR@
    vi env.sh          # modules, VASP_STD, LOBSTER_BIN, VASP_POTCAR_DIR, ANALYSIS_PYTHON

Each job ends by running its `analyze.sh` (band / COHP / COBI / COOP /
LOBSTER-DOS plots and `lobster_summary.csv` in `analysis/`, log in
`analysis/analyze.log`).  That needs a python3 with numpy + matplotlib: set
`ANALYSIS_PYTHON` (e.g. `$HOME/vaspenv/bin/python3`) if the system python
lacks them.  `./analyze_all.sh` redoes the analysis for finished materials.

`LOBSTER_BIN` must be the **Linux** LOBSTER build — the macOS binary named in
the local config will not run on the cluster.

If the partition name, account or walltime need to change:

    ./retune.sh --partition <name> --account <alloc> --time 48:00:00

## 3. POTCARs (automatic)

`./submit_all.sh` runs `make_potcars.sh` itself when a material has no POTCAR
(and generation on the cluster builds them at the end).  Run
`./make_potcars.sh` by hand only to check the library early.  It concatenates
each material's POTCAR from `$VASP_POTCAR_DIR` and checks every ENMAX/ZVAL
against `potcar_manifest.json`.  ENCUT and the LOBSTER NBANDS in the INCARs
were derived from those numbers, so **a mismatch is a real error**, not a
warning to ignore -- tell me and I will regenerate the highthroughput directory
against your cluster's library.

## 4. Test first

    cd materials/@FIRST@
    sbatch job.sbatch
    squeue -u $USER

When it finishes, check:

    grep "General timing" 02_scf/OUTCAR 03_bands/OUTCAR 08_lobster/OUTCAR
    grep -i "finished in" 08_lobster/lobster.out
    head -30 08_lobster/ICOHPLIST.lobster

## 5. Submit the screen (SLURM)

Each material is one SLURM job (`materials/<id>/job.sbatch`); `submit_all.sh`
calls `sbatch` on them, keeping at most MAX_QUEUED in the queue.  Single material:
`cd materials/<id> && sbatch job.sbatch`.

    ./submit_all.sh              # throttled to MAX_QUEUED=50 concurrent jobs
    MAX_QUEUED=100 ./submit_all.sh
    ./submit_all.sh 20           # only 20 materials this pass

Re-running `submit_all.sh` is the resume command: materials with a `.done`
marker or a job already in the queue are skipped, and each `job.sbatch` skips
steps whose OUTCAR already carries the final timing block.

## 6. Monitor and collect

    ./status.sh                  # per-step completion counts; writes incomplete.txt
    ./analyze_all.sh             # redo per-material plots (each job already runs analyze.sh)
    ./postprocess_all.sh         # per-material lobster_summary.csv + screen_lobster_all.csv
    ./collect_results.sh         # -> screen_results.tar.gz (small, bring this home)

## Notes and caveats

* **Smearing.**  ISMEAR=0 with SIGMA=0.05 throughout — Gaussian smearing that
  matches LOBSTER's `gaussianSmearingWidth`, and safe for zero-gap systems.
* **No DFPT.**  Dielectric tensors and Born charges are not computed: with
  a zero gap they are ill-defined and the linear-response loop tends not to
  converge.  **After the test job, report the per-step timings:**

      grep "Elapsed time" materials/@FIRST@/0*/OUTCAR
* **k-mesh.**  @KMESH_NOTE@ Even subdivisions
  so the BZ boundary is always sampled.  The LOBSTER NSCF is capped at 1x the
  SCF mesh because ISYM=0 means it pays for the full mesh, not the irreducible
  wedge.
* **Spin.**  Cells containing V/Cr/Mn/Fe/Co/Ni start spin-polarised (ISPIN=2)
  with a high moment on the magnetic species.  Everything else is
  non-spin-polarised.
* **Disk.**  Each job deletes its CHGCAR/WAVECAR copies at the end.  Set
  `KEEP_LARGE_FILES=1` in env.sh to retain them.
'''


ANALYZE_ALL_SH = r'''#!/bin/bash
# Re-run every finished material's analyze.sh (plots + LOBSTER CSV) -- e.g.
# on the login node if the jobs skipped it.  Safe to re-run.
#   ./analyze_all.sh            all materials with a finished 02_scf
#   ./analyze_all.sh mp-8       one material
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VASPFLOW_NO_CHECK=1 source "$HERE/env.sh" >/dev/null 2>&1
[ -n "${ANALYSIS_PYTHON:-}" ] && export PATH="$(dirname "$ANALYSIS_PYTHON"):$PATH"
python3 -c "import numpy, matplotlib" 2>/dev/null || {
    echo "ERROR: python3 lacks numpy/matplotlib; set ANALYSIS_PYTHON in env.sh"; exit 1; }
ids="${1:-$(cat "$HERE/material_list.txt")}"
n=0
for id in $ids; do
    d="$HERE/materials/$id"
    [ -f "$d/02_scf/OUTCAR" ] && [ -f "$d/analyze.sh" ] || continue
    mkdir -p "$d/analysis"
    if MPLBACKEND=Agg bash "$d/analyze.sh" > "$d/analysis/analyze.log" 2>&1; then
        n=$((n + 1))
    else
        echo "  analysis errors: $id (see materials/$id/analysis/analyze.log)"
    fi
done
echo "Analysed $n material(s)."
'''


POSTPROCESS_SH = r'''#!/bin/bash
# Aggregate the finished screen into per-material LOBSTER bonding CSVs and one
# combined table.  Needs python3 with numpy on the login node; safe to re-run.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PP="$HERE/tools/lobster_postprocess.py"
[ -f "$PP" ] || { echo "ERROR: $PP missing"; exit 1; }

n=0
while read -r id; do
    [ -z "$id" ] && continue
    d="$HERE/materials/$id/08_lobster"
    [ -f "$d/COHPCAR.lobster" ] || continue
    python3 "$PP" --dir "$d" --out "$d/lobster_summary.csv" >/dev/null 2>&1 \
        && n=$((n + 1)) || echo "  postprocess failed: $id"
done < "$HERE/material_list.txt"
echo "Wrote $n per-material lobster_summary.csv"

# Concatenate into one table with an mp_id column in front.
python3 - "$HERE" <<'PYEND'
import csv, os, sys
here = sys.argv[1]
ids = [l.strip() for l in open(os.path.join(here, 'material_list.txt')) if l.strip()]
out = os.path.join(here, 'screen_lobster_all.csv')
header, rows = None, []
for mp_id in ids:
    f = os.path.join(here, 'materials', mp_id, '08_lobster', 'lobster_summary.csv')
    if not os.path.isfile(f):
        continue
    with open(f, newline='') as fh:
        r = list(csv.reader(fh))
    if len(r) < 2:
        continue
    if header is None:
        header = ['mp_id'] + r[0]
    for line in r[1:]:
        rows.append([mp_id] + line)
if header:
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"Combined {len(rows)} bond rows -> {out}")
else:
    print("No per-material summaries found.")
PYEND
'''


# ── main ────────────────────────────────────────────────────────────────────
def build_highthroughput_files(htdir, ids, first_id, args, manifest):
    """Write the cluster-side scripts and metadata into the highthroughput root."""
    modules = args.modules.split(',') if args.modules else []
    module_lines = '\n'.join(f"module load {m.strip()}" for m in modules if m.strip())

    files = {
        'env.sh': (ENV_SH
                   .replace('@MODULES@',     module_lines or '# (no modules)')
                   .replace('@VASP_STD@',    args.vasp_std)
                   .replace('@LOBSTER_BIN@', args.lobster_bin)
                   .replace('@POTCAR_DIR@',  args.hpc_potcar_dir)),
        'make_potcars.sh':    MAKE_POTCARS,
        'submit_all.sh':      SUBMIT_ALL,
        'status.sh':          STATUS_SH,
        'collect_results.sh': COLLECT_SH,
        'retune.sh':          RETUNE_SH,
        'postprocess_all.sh': POSTPROCESS_SH,
        'analyze_all.sh':     ANALYZE_ALL_SH,
        'README_HPC.md': (README_HPC
                          .replace('@HTDIR@',   os.path.basename(htdir))
                          .replace('@FIRST@',    first_id)
                          .replace('@KMESH_NOTE@',
                                   (f'k-point density {args.kpra} k-points per reciprocal atom '
                                    '(N_i proportional to |b_i*|, uniform spacing).') if args.kpra else
                                   f'Fixed spacing of {args.kspacing} A^-1 per material.')),
    }
    for name, text in files.items():
        p = os.path.join(htdir, name)
        with open(p, 'w') as f:
            f.write(text)
        if name.endswith('.sh'):
            os.chmod(p, 0o755)

    tools = os.path.join(htdir, 'tools')
    os.makedirs(tools, exist_ok=True)
    for mod in ('lobster_postprocess.py', 'cohp_plot.py', 'lobster_dos_plot.py',
                'outcar_parser.py'):
        src = os.path.join(REPO_DIR, 'modules', mod)
        if os.path.isfile(src):
            shutil.copy(src, tools)

    with open(os.path.join(htdir, 'material_list.txt'), 'w') as f:
        f.write('\n'.join(ids) + '\n')
    with open(os.path.join(htdir, 'potcar_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)


def write_summary_csv(path, metas):
    cols = ['mp_id', 'formula', 'natoms', 'nelect', 'encut', 'kmesh', 'nk_full',
            'W', 'nodes', 'ntasks_per_node', 'walltime', 'partition', 'ispin',
            'potcars']
    with open(path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for mp_id, m in metas.items():
            f.write(','.join([
                mp_id, m['formula'], str(m['natoms']), f"{m['nelect']:.0f}",
                str(m['encut']), 'x'.join(str(x) for x in m['mesh']),
                str(m['nk_full']), f"{m['W']:.2f}", str(m['nodes']),
                str(m['ntasks_per_node']), m['walltime'], m['partition'],
                '2' if m['ispin'] else '1', ' '.join(m['variants']),
            ]) + '\n')


def main():
    ap = argparse.ArgumentParser(
        description='High-throughput semimetal screen: SCF + bands + LOBSTER.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-l', '--list', required=True, help='file with one mp-id per line')
    ap.add_argument('-o', '--out', required=True, help='highthroughput directory to create')
    ap.add_argument('--kspacing', type=float, default=KSPACING,
                    help=f'k-point spacing in A^-1 (default {KSPACING})')
    ap.add_argument('--kpra', type=int, default=None,
                    help='k-point density in k-points per reciprocal atom (e.g. 8000); '
                         'replaces --kspacing.  Mesh rule: even N_i proportional to |b_i*|, '
                         'in-plane multiples of 6 for hexagonal lattices.')
    ap.add_argument('--single-node', action='store_true',
                    help='every material runs on ONE node with --cores-per-node cores on '
                         '--partition (walltime still tiered by size); no multi-node tier')
    ap.add_argument('--kpar', type=int, default=None,
                    help='force KPAR in the SCF/bands/LOBSTER INCARs (default: auto)')
    ap.add_argument('--ncore', type=int, default=None,
                    help='force NCORE (default: auto; the other value is derived)')
    ap.add_argument('--max-atoms', type=int, default=MAX_ATOMS,
                    help=f'skip primitive cells with more atoms (default {MAX_ATOMS})')
    ap.add_argument('--cores-per-node', type=int, default=CORES_PER_NODE)
    ap.add_argument('--max-nodes', type=int, default=MAX_NODES)
    ap.add_argument('--partition', default=PARTITION)
    ap.add_argument('--account', default=ACCOUNT)
    ap.add_argument('--functional', default='PBE')
    ap.add_argument('--profile', default='slurm', help='agent profile name')
    ap.add_argument('--modules', default='vasp/5.4.4',
                    help='comma-separated module loads written into env.sh. '
                         'The site vasp module resolves its own toolchain; '
                         'adding intel/impi by hand causes version swaps.')
    ap.add_argument('--vasp-std', default='vasp_std')
    ap.add_argument('--lobster-bin', default='$HOME/BIN/lobster-5.1.1')
    ap.add_argument('--hpc-potcar-dir', default='$HOME/vasp.5.4.4/PAW_PBE',
                    help='POTCAR library path on the CLUSTER (written into env.sh)')
    ap.add_argument('--include-potcars', action='store_true',
                    help='ship assembled POTCARs inside the highthroughput directory (large)')
    ap.add_argument('--build-potcars', dest='build_potcars', action='store_true', default=None,
                    help='run make_potcars.sh right after generation (default: automatic when '
                         'sbatch is available, i.e. on the cluster)')
    ap.add_argument('--no-build-potcars', dest='build_potcars', action='store_false',
                    help='do not build POTCARs after generation (e.g. generating on a Mac to rsync)')
    ap.add_argument('--api-key', default=None)
    args = ap.parse_args()

    ht = _load_module('ht_mp_scf', 'ht-mp-scf.py')
    api_key    = ht.get_api_key(args.api_key)
    potcar_dir = os.path.expanduser(os.environ.get('VASP_POTCAR_DIR', ''))
    if not os.path.isdir(potcar_dir):
        sys.exit("ERROR: VASP_POTCAR_DIR is not set or does not exist "
                 "(needed for ENCUT/NBANDS and the POTCAR manifest).")

    ids = ht.read_id_list(args.list)

    htdir      = os.path.abspath(args.out)
    materials  = os.path.join(htdir, 'materials')
    stage_root = os.path.join(htdir, STAGE_DIR)
    os.makedirs(materials,  exist_ok=True)
    os.makedirs(stage_root, exist_ok=True)

    print(f"\nSemimetal screen highthroughput directory: {htdir}")
    print(f"  materials  : {len(ids)}")
    print(f"  k-spacing  : {args.kspacing} A^-1 (even, commensurate with a,b,c)")
    print(f"  max atoms  : {args.max_atoms}")
    print(f"  resources  : up to {args.max_nodes} nodes x {args.cores_per_node} cores")
    print(f"  account    : {args.account}   partition: {args.partition}\n")

    metas, ok, skipped, failed = {}, [], [], []
    manifest = {'kspacing': args.kspacing, 'kpra': args.kpra, 'potcar_dir_local': potcar_dir,
                'materials': {}}

    for n, mp_id in enumerate(ids, 1):
        stage = os.path.join(stage_root, mp_id)
        os.makedirs(stage, exist_ok=True)
        poscar = os.path.join(stage, 'POSCAR')

        if not os.path.isfile(poscar):          # resume: keep what we have
            try:
                structure = ht.fetch_primitive_structure(mp_id, api_key)
                ht.write_poscar(structure, poscar)
            except Exception as e:
                print(f"[{n}/{len(ids)}] {mp_id}: DOWNLOAD FAILED ({e})")
                failed.append(mp_id)
                continue

        try:
            struct = read_poscar(poscar)
        except Exception as e:
            print(f"[{n}/{len(ids)}] {mp_id}: bad POSCAR ({e})")
            failed.append(mp_id)
            continue

        # Hard atom-count filter, applied to the PRIMITIVE cell that will
        # actually be run (not to the Materials Project nsites field).
        if struct['natoms'] > args.max_atoms:
            print(f"[{n}/{len(ids)}] {mp_id}: SKIP — {struct['natoms']} atoms "
                  f"> {args.max_atoms}")
            skipped.append((mp_id, struct['natoms']))
            continue

        try:
            pinfo = material_potcar_info(struct, potcar_dir)
        except Exception as e:
            print(f"[{n}/{len(ids)}] {mp_id}: {e}")
            failed.append(mp_id)
            continue

        if args.kpra:
            from vasp_input_generator import VASPInputGenerator
            mesh = tuple(VASPInputGenerator(poscar, {})._compute_mesh(args.kpra))
        else:
            mesh = kmesh_from_lattice(struct['A'], args.kspacing)
        nk_full = mesh[0] * mesh[1] * mesh[2]
        tiers = ([(b, 1, args.cores_per_node, wt, args.partition)
                  for b, _n, _c, wt, _p in TIERS] if args.single_node else None)
        nodes, ntpn, walltime, partition, W = size_job(
            pinfo['nelect'], nk_full, tiers=tiers, max_nodes=args.max_nodes)

        formula = ''.join(f"{el}{cnt if cnt > 1 else ''}"
                          for el, cnt in zip(struct['species'], struct['counts']))

        write_instructions(os.path.join(stage, 'instructions.txt'), mp_id,
                           struct, mesh, pinfo['encut'], nodes, ntpn, walltime,
                           partition, args.account, args.functional, kpra=args.kpra,
                           kpar=args.kpar, ncore=args.ncore)

        # Build INCAR/KPOINTS/POTCAR/run.sh via the existing SLURM agent.
        env = dict(os.environ, VASP_POTCAR_DIR=potcar_dir)
        proc = subprocess.run(
            [sys.executable, AGENT_SLURM,
             '-i', os.path.join(stage, 'instructions.txt'),
             '-s', poscar, '-p', args.profile],
            cwd=materials, env=env, capture_output=True, text=True)
        proj = os.path.join(materials, mp_id)
        if proc.returncode != 0 or not os.path.isdir(proj):
            print(f"[{n}/{len(ids)}] {mp_id}: AGENT FAILED")
            print('    ' + (proc.stderr or proc.stdout or '').strip()[-500:].replace('\n', '\n    '))
            failed.append(mp_id)
            continue

        expected = ['02_scf', '03_bands', '08_lobster']
        got = [d for d in expected if os.path.isdir(os.path.join(proj, d))]
        if got != expected:
            print(f"[{n}/{len(ids)}] {mp_id}: WRONG STEPS {got}")
            failed.append(mp_id)
            continue

        cap_lobster_mesh(proj)
        localise_analyze_sh(proj, '../../tools')
        for step in expected:
            make_run_sh_portable(os.path.join(proj, step))

        manifest['materials'][mp_id] = {
            'potcars': [[v] + list(potcar_props(potcar_dir, v))
                        for v in pinfo['variants']],
            'encut': pinfo['encut'], 'natoms': struct['natoms'],
        }
        if not args.include_potcars:
            strip_potcars(proj)

        # The agent also writes a per-material submit_all.sh chaining its own step
        # scripts; job.sbatch (below) is the single entry point here, so drop it.
        _sa = os.path.join(proj, 'submit_all.sh')
        if os.path.exists(_sa):
            os.remove(_sa)

        # Local copies so the cluster side is self-describing.
        shutil.copy(os.path.join(stage, 'instructions.txt'), proj)

        meta = {'formula': formula, 'natoms': struct['natoms'],
                'nelect': pinfo['nelect'], 'encut': pinfo['encut'],
                'mesh': mesh, 'nk_full': nk_full, 'W': W, 'nodes': nodes,
                'ntasks_per_node': ntpn, 'walltime': walltime,
                'ispin': magmom_string(struct) is not None,
                'partition': partition, 'variants': pinfo['variants']}
        write_job_script(proj, mp_id, meta, partition, args.account)
        metas[mp_id] = meta
        ok.append(mp_id)

        print(f"[{n}/{len(ids)}] {mp_id:<12} {formula:<12} "
              f"{struct['natoms']:>3} at  {'x'.join(str(x) for x in mesh):>10}  "
              f"ENCUT={pinfo['encut']:<4} W={W:7.1f}  {nodes}n x {ntpn}  {walltime}")

    if not ok:
        sys.exit("\nERROR: no materials staged.")

    build_highthroughput_files(htdir, ok, ok[0], args, manifest)
    write_summary_csv(os.path.join(htdir, 'materials_summary.csv'), metas)

    tier = {}
    for m in metas.values():
        key = (m['partition'], m['nodes'], m['ntasks_per_node'], m['walltime'])
        tier[key] = tier.get(key, 0) + 1

    print(f"\n{'='*66}")
    print(f"  Highthroughput: {htdir}")
    print(f"{'='*66}")
    print(f"  staged        : {len(ok)}")
    if skipped:
        print(f"  skipped (>{args.max_atoms} atoms): {len(skipped)}")
    if failed:
        print(f"  failed        : {len(failed)}  ({', '.join(failed[:10])}"
              f"{' ...' if len(failed) > 10 else ''})")
    print("  tiers         :")
    for (part, nodes, cores, wt), cnt in sorted(tier.items()):
        print(f"      {cnt:5d}  {part:<9} {nodes} node(s) x {cores}  {wt}")
    print(f"  summary       : materials_summary.csv")
    build = args.build_potcars
    if build is None:
        build = shutil.which('sbatch') is not None and not args.include_potcars
    if build and not args.include_potcars:
        print("  Building POTCARs (make_potcars.sh) ...")
        r = subprocess.run([os.path.join(htdir, 'make_potcars.sh')],
                           env=dict(os.environ, VASP_POTCAR_DIR=potcar_dir))
        if r.returncode:
            print("  WARNING: make_potcars.sh failed; fix env.sh and rerun it "
                  "(submit_all.sh also retries automatically).")
    if shutil.which('sbatch') and build:
        print(f"\n  Next: cd {htdir} && ./submit_all.sh\n")
    else:
        print(f"\n  Next: transfer the highthroughput directory, then follow README_HPC.md\n"
              f"  (POTCARs are built automatically by ./submit_all.sh on the cluster)\n")


if __name__ == '__main__':
    main()
