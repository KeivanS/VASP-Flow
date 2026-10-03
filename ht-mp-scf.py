#!/usr/bin/env python3
"""
High-Throughput MP -> SCF + ELF driver
======================================
Reads a list of Materials Project IDs (one per line) from an input file,
downloads the PRIMITIVE-cell POSCAR for each from the Materials Project,
and stages a per-material relaxation (01_relax: cell + ions, IBRION = 1,
ISIF = 3, so the structure matches this functional; --no-relax skips it)
followed by an SCF that also computes the electron localization function
(ELF -> ELFCAR).  The SCF runs with automatic
KPAR/NCORE; a short KPAR = 1 restart in 02_scf/elf then writes ELFCAR (LELF
needs KPAR = 1).  k-points: --kpra density (default coarse = 1000 kpra,
uniform mesh), or a fixed --kmesh; editable per material in
_ht_inputs/<id>/instructions.txt.

Folder/input construction and job execution are delegated to the existing
agents:

    vasp-agent.py        (workstation:  run.sh, sequential)
    vasp-agent-slurm.py  (SLURM:         submit_all.sh, sbatch)

This driver does NOT run VASP itself.  It:
  1. Fetches each primitive structure from MP                (needs network).
  2. Writes  _ht_inputs/<mp-id>/POSCAR  and  instructions.txt.
  3. Writes  runall.sh  which, for every mp-id, calls the chosen agent to
     build  <mp-id>/  (POTCAR + 02_scf/{INCAR,KPOINTS,run.sh}) and then
     runs (local) or submits (SLURM) the SCF job, one material after another.

Typical use
-----------
    export MP_API_KEY=...              # your Materials Project API key
    export VASP_POTCAR_DIR=...         # needed by the agent for POTCAR

    # list of IDs, one per line:
    printf 'mp-149\\nmp-2534\\n' > highthrouput_list

    ./ht-mp-scf.py --agent local   --mpi 16
    ./ht-mp-scf.py --agent slurm   --profile slurm           # chained (default)
    ./ht-mp-scf.py --agent slurm   --no-chain                # parallel submit

    ./runall.sh                        # build inputs + run/submit every job

Execution order
---------------
  local            run_all.sh blocks -> materials run strictly one-at-a-time.
  slurm (default)  each SCF is submitted with --dependency=afterok on the
                   previous material, so the cluster runs them sequentially.
  slurm --no-chain materials submitted independently (scheduler decides order).

Requirements
------------
    pip install pymatgen mp-api        # MP download + primitive-cell standardisation
"""

import os, shutil
import sys
import argparse
import textwrap

# Repo root = directory containing this script and the agents.
REPO_DIR        = os.path.dirname(os.path.abspath(__file__))
AGENT_LOCAL     = os.path.join(REPO_DIR, 'vasp-agent.py')
AGENT_SLURM     = os.path.join(REPO_DIR, 'vasp-agent-slurm.py')
DEFAULT_LIST    = 'highthrouput_list'      # spelling per the project convention
STAGE_DIR       = '_ht_inputs'

# Atomic numbers Z for elements 1..118 (symbol -> Z). Small static table,
# so no need to query Materials Project for it.
_PERIODIC = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
ELEMENT_Z = {sym: i + 1 for i, sym in enumerate(_PERIODIC)}


# ── input list ──────────────────────────────────────────────────────────────
def read_id_list(path):
    """Return the list of mp-IDs in *path*, skipping blanks and # comments."""
    if not os.path.isfile(path):
        # tolerate the correctly-spelled variant too
        alt = path.replace('highthrouput', 'highthroughput')
        if path == DEFAULT_LIST and os.path.isfile(alt):
            path = alt
        else:
            sys.exit(f"ERROR: ID list file not found: {path}")
    ids = []
    with open(path) as f:
        for raw in f:
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            ids.append(line.split()[0])    # first token on the line
    if not ids:
        sys.exit(f"ERROR: no mp-IDs found in {path}")
    # de-duplicate, preserve order
    seen, uniq = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i); uniq.append(i)
    return uniq


# ── Materials Project download ──────────────────────────────────────────────
def get_api_key(cli_key):
    key = cli_key or os.environ.get('MP_API_KEY') or os.environ.get('PMG_MAPI_KEY')
    if not key:
        sys.exit(textwrap.dedent("""\
            ERROR: no Materials Project API key.
              Set one with:  export MP_API_KEY=your_key
              (get a key at https://materialsproject.org/api)
              or pass --api-key on the command line."""))
    return key


def find_local_poscar(mp_id, dirs):
    """An already-downloaded POSCAR for *mp_id*, or None.  Each dir in *dirs*
    is searched as <dir>/<id>/POSCAR, <dir>/materials/<id>/POSCAR,
    <dir>/_ht_inputs/<id>/POSCAR and <dir>/<id>.vasp / <dir>/POSCAR_<id>."""
    for d in dirs or []:
        d = os.path.expanduser(d)
        for cand in (os.path.join(d, mp_id, 'POSCAR'),
                     os.path.join(d, 'materials', mp_id, 'POSCAR'),
                     os.path.join(d, '_ht_inputs', mp_id, 'POSCAR'),
                     os.path.join(d, f'{mp_id}.vasp'),
                     os.path.join(d, f'POSCAR_{mp_id}')):
            if os.path.isfile(cand) and os.path.getsize(cand) > 0:
                return cand
    return None


def fetch_primitive_structure(mp_id, api_key):
    """Download *mp_id* from MP and return its standardized primitive cell.

    Tries the modern mp-api client first, then the legacy pymatgen MPRester,
    so the script works with either generation of API key / install.
    """
    struct, errors = None, []

    try:
        from mp_api.client import MPRester
        with MPRester(api_key) as mpr:
            struct = mpr.get_structure_by_material_id(mp_id)
    except Exception as e:
        errors.append(f"mp_api: {e}")

    if struct is None:
        try:
            from pymatgen.ext.matproj import MPRester as LegacyMPRester
            with LegacyMPRester(api_key) as mpr:
                struct = mpr.get_structure_by_material_id(mp_id)
        except Exception as e:
            errors.append(f"legacy: {e}")

    if struct is None:
        raise RuntimeError("; ".join(errors) or "unknown MP error")

    return _to_primitive(struct)


def _to_primitive(struct):
    """Return the primitive cell, robust across spglib / pymatgen versions.

    Prefers the spglib-standardized primitive cell. Some installations raise
    here -- e.g. "'dict' object has no attribute 'number'" when an older
    spglib (< 2.5, dataset is a dict) is paired with a newer pymatgen (which
    expects attribute access). In that case fall back to pymatgen's own
    primitive-cell finder, and finally to the cell as downloaded, so the run
    is never blocked by an environment mismatch.

    Fix the root cause with:  pip install -U spglib
    """
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return SpacegroupAnalyzer(struct).get_primitive_standard_structure()
    except Exception:
        pass
    try:
        return struct.get_primitive_structure()
    except Exception:
        return struct


# ── per-material staging ────────────────────────────────────────────────────
def write_poscar(structure, path):
    """Write a VASP5 POSCAR (element-symbol line present) for the agent."""
    from pymatgen.io.vasp import Poscar
    Poscar(structure).write_file(path)


def write_instructions(path, mp_id, functional, mpi, encut, slurm_opts, kmesh=None,
                       lobster=True, kpra=None, kpar=None, ncore=None,
                       relax=True, relax_kpra=None, gga_u='auto'):
    """Write a minimal (relax +) SCF + ELF instructions.txt for one material."""
    tasks = (("structure relaxation, " if relax else "") + "SCF calculation"
             + (", LOBSTER COHP/COBI analysis" if lobster else ""))
    lines = [
        f"Project: {mp_id}",
        "# High-throughput SCF + ELF from a Materials Project primitive cell",
        f"Methods: {functional} functional",
        "",
        f"Tasks: {tasks}",
        "",
        "# ELF in two steps: the SCF runs with full k-point parallelism (auto",
        "# KPAR/NCORE), then a short restart in 02_scf/elf reads its converged",
        "# WAVECAR+CHGCAR with KPAR = 1 (required by LELF) and writes ELFCAR.",
        "# Use 'ELF: on' for the old single run (LELF in the SCF, KPAR = 1).",
        "ELF: separate",
        "",
        f"MPI: {mpi}",
    ]
    if kmesh:
        lines.append("# Fixed k-mesh (--kmesh); replace by KMESH_DENSITY for a uniform mesh")
        lines.append(f"KMESH: {kmesh}")
    elif kpra:
        lines.append("# k-point density (kpra): N_i proportional to |b_i*|, uniform spacing")
        lines.append(f"KMESH_DENSITY: {kpra}")
    if relax:
        lines += ["# 01_relax: full relaxation of cell shape, volume and ions",
                  "# (IBRION = 1, ISIF = 3) with this functional -- the MP cell was",
                  "# relaxed with a different setup.  The SCF starts from its CONTCAR",
                  "# and CHGCAR (+ WAVECAR when the k-mesh is the same).",
                  "INCAR relax:",
                  "   IBRION = 1   # quasi-Newton: IBRION=2 line search fails (ZBRENT) once a cell is converged",
                  "   ISIF = 3",
                  "END_INCAR"]
        if not kmesh:
            lines.append(f"RELAX_KMESH_DENSITY: {relax_kpra or kpra or 'coarse'}")
    if gga_u in ('on', 'off'):
        lines.append(f"GGA_U: {gga_u.upper()}   # default (no flag): U only for chalcogenides/halides")
    if kpar:
        lines.append(f"KPAR: {kpar}")
    if ncore:
        lines.append(f"NCORE: {ncore}")
    if encut:
        lines.append(f"ENCUT: {encut}")
    for key, val in (slurm_opts or {}).items():
        if val:
            lines.append(f"{key}: {val}")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def write_element_table(material_elements, path):
    """Tabulate the elements (and their Z) seen across the screening set.

    material_elements: dict  mp_id -> list of element symbols.
    Writes a two-column Z/Element table plus a per-material breakdown to
    *path* and returns the text so the caller can echo it too.
    """
    zkey = lambda s: ELEMENT_Z.get(s, 999)
    unique = sorted({e for els in material_elements.values() for e in els}, key=zkey)

    lines = [f"Elements across {len(material_elements)} material(s)",
             "",
             f"  {'Z':>3}  Element",
             f"  {'-'*3}  {'-'*7}"]
    for e in unique:
        z = ELEMENT_Z.get(e, '?')
        lines.append(f"  {z:>3}  {e}")
    lines += ["", "Per material:"]
    for mid, els in material_elements.items():
        tag = ", ".join(f"{e}({ELEMENT_Z.get(e, '?')})" for e in els)
        lines.append(f"  {mid:<14} {tag}")

    text = "\n".join(lines) + "\n"
    with open(path, 'w') as f:
        f.write(text)
    return text


# ── runall.sh ───────────────────────────────────────────────────────────────
def write_runall(path, ids, agent_kind, profile, chain=True):
    """Emit runall.sh: for each mp-id, call the agent then run/submit the SCF job.

    local:           run_all.sh blocks, so materials run strictly one-at-a-time.
    slurm + chain:   each material's SCF is submitted with
                     --dependency=afterok on the previous material's job, so the
                     cluster runs them sequentially even though all are queued
                     up front.  Within a material the steps (01_relax,
                     02_scf, 08_lobster) are chained the same way.
    slurm + no chain: each project's submit_all.sh is called; jobs are
                     independent and run whenever the scheduler allows.
    """
    agent    = AGENT_SLURM if agent_kind == 'slurm' else AGENT_LOCAL
    # Only pass a profile for the SLURM agent; the local agent uses site.env
    # defaults (passing the SLURM profile would force srun/SBATCH).
    prof     = f' -p "{profile}"' if agent_kind == 'slurm' else ''
    id_array = " ".join(ids)

    with open(path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated by ht-mp-scf.py\n")
        f.write("# Builds inputs and runs relax + SCF+ELF (+ LOBSTER) for each mp-id, in order.\n")
        mode = ("slurm, dependency-chained" if (agent_kind == 'slurm' and chain)
                else "slurm, independent" if agent_kind == 'slurm'
                else "local, sequential")
        f.write(f"# Agent: {os.path.basename(agent)}   mode: {mode}\n\n")
        f.write("set -e\n")
        f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n')
        f.write(f'AGENT="{agent}"\n')
        f.write(f'STAGE="$HERE/{STAGE_DIR}"\n\n')
        f.write('if [ -z "$VASP_POTCAR_DIR" ]; then\n')
        f.write('    echo "WARNING: VASP_POTCAR_DIR is not set; the agent cannot build POTCAR."\n')
        f.write('fi\n\n')
        f.write(f'IDS=({id_array})\n\n')

        if agent_kind == 'slurm' and chain:
            f.write('PREV=""\n\n')
            f.write('for id in "${IDS[@]}"; do\n')
            f.write('    echo ""\n')
            f.write('    echo ">>> $id"\n')
            f.write('    # 1. Build POTCAR/INCAR/KPOINTS/folders for this material\n')
            f.write(f'    "$AGENT" -i "$STAGE/$id/instructions.txt" -s "$STAGE/$id/POSCAR"{prof}\n')
            f.write('    # 2. Submit every step (01_relax, 02_scf, 08_lobster, ...) in order;\n')
            f.write('    #    the first one waits for the previous material\'s last job.\n')
            f.write('    n=0\n')
            f.write('    for SD in "$HERE/$id"/0[1-9]_*/; do\n')
            f.write('        [ -f "$SD/run.sh" ] || continue\n')
            f.write('        if [ -n "$PREV" ]; then\n')
            f.write('            JID=$(cd "$SD" && sbatch --parsable --dependency=afterok:$PREV run.sh)\n')
            f.write('        else\n')
            f.write('            JID=$(cd "$SD" && sbatch --parsable run.sh)\n')
            f.write('        fi\n')
            f.write('        echo "  Submitted $id $(basename "$SD") -> job $JID (after ${PREV:-none})"\n')
            f.write('        PREV=$JID; n=$((n + 1))\n')
            f.write('    done\n')
            f.write('    [ "$n" -gt 0 ] || echo "ERROR: no run.sh generated for $id; skipping."\n')
            f.write('done\n\n')
            f.write('echo ""\n')
            f.write('echo "All jobs submitted (dependency-chained). Monitor: squeue -u $USER"\n')
        elif agent_kind == 'slurm':
            f.write('for id in "${IDS[@]}"; do\n')
            f.write('    echo ""\n')
            f.write('    echo ">>> $id"\n')
            f.write(f'    "$AGENT" -i "$STAGE/$id/instructions.txt" -s "$STAGE/$id/POSCAR"{prof}\n')
            f.write('    if [ -x "$HERE/$id/submit_all.sh" ]; then\n')
            f.write('        ( cd "$HERE/$id" && ./submit_all.sh )\n')
            f.write('    else\n')
            f.write('        echo "ERROR: $id/submit_all.sh not generated; skipping run."\n')
            f.write('    fi\n')
            f.write('done\n\n')
            f.write('echo ""\n')
            f.write('echo "All jobs submitted (independent). Monitor: squeue -u $USER"\n')
        else:
            f.write('for id in "${IDS[@]}"; do\n')
            f.write('    echo ""\n')
            f.write('    echo "=================================================="\n')
            f.write('    echo ">>> $id"\n')
            f.write('    echo "=================================================="\n')
            f.write('    # 1. Build POTCAR/INCAR/KPOINTS/folders for this material\n')
            f.write(f'    "$AGENT" -i "$STAGE/$id/instructions.txt" -s "$STAGE/$id/POSCAR"{prof}\n')
            f.write('    # 2. Run the SCF job (blocking -> strictly sequential)\n')
            f.write('    if [ -x "$HERE/$id/run_all.sh" ]; then\n')
            f.write('        ( cd "$HERE/$id" && ./run_all.sh )\n')
            f.write('    else\n')
            f.write('        echo "ERROR: $id/run_all.sh not generated; skipping run."\n')
            f.write('    fi\n')
            f.write('done\n\n')
            f.write('echo ""\n')
            f.write('echo "All SCF+ELF calculations complete. ELFCAR is in each <mp-id>/02_scf/."\n')
    os.chmod(path, 0o755)


# ── main ────────────────────────────────────────────────────────────────────
def choose_agent(cli_choice):
    if cli_choice in ('local', 'slurm'):
        return cli_choice
    if not sys.stdin.isatty():
        return 'local'
    ans = input("Run on [l]ocal workstation or [s]lurm cluster? [l/s] ").strip().lower()
    return 'slurm' if ans.startswith('s') else 'local'


def main():
    ap = argparse.ArgumentParser(
        description="High-throughput Materials Project -> VASP SCF + ELF setup.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-l', '--list', default=DEFAULT_LIST,
                    help=f"file with one mp-id per line (default: {DEFAULT_LIST})")
    ap.add_argument('--agent', choices=['local', 'slurm'], default=None,
                    help="which agent to use; prompts if omitted")
    ap.add_argument('--functional', default='PBE',
                    help="DFT functional (PBE, PBEsol, R2SCAN, ...). Default PBE.")
    ap.add_argument('--mpi', type=int, default=1,
                    help="MPI tasks per job (sets MPI: in instructions). Default 1.")
    ap.add_argument('--encut', type=int, default=None,
                    help="optional ENCUT override (eV)")
    ap.add_argument('--lobster', dest='lobster', action='store_true', default=True,
                    help="add a LOBSTER (08_lobster) bonding-analysis step (default on)")
    ap.add_argument('--no-lobster', dest='lobster', action='store_false',
                    help="do not add the LOBSTER step")
    ap.add_argument('--relax', dest='relax', action='store_true', default=True,
                    help="first relax cell + ions (01_relax, IBRION=1, ISIF=3); default on")
    ap.add_argument('--no-relax', dest='relax', action='store_false',
                    help="skip the relaxation; SCF on the MP structure as downloaded")
    ap.add_argument('--relax-kpra', default=None,
                    help="k-point density of the relaxation (default: same as --kpra, "
                         "so the SCF can reuse the relaxed WAVECAR)")
    ap.add_argument('--gga_u', '--gga-u', '--gga+u', '--GGA_U', '--GGA-U', '--GGA+U',
                    dest='gga_u', type=str.lower, choices=['auto', 'on', 'off'], default='auto',
                    help="GGA+U with the tabulated U_eff: auto (default) = only if a "
                         "chalcogen/halogen (O,S,Se,Te,F,Cl,Br,I) is present; on = every "
                         "tabulated d/f element; off = never")
    ap.add_argument('--kmesh', default=None,
                    help="fixed SCF Gamma k-mesh for every material, e.g. '8 8 8'; "
                         "overrides --kpra (not recommended for mixed cell shapes)")
    ap.add_argument('--kpar', type=int, default=None,
                    help="force KPAR in the SCF (default: auto; the ELF pass always uses KPAR=1)")
    ap.add_argument('--kpra', default='coarse',
                    help="k-point density (default coarse): 'coarse' (1000), 'fine' "
                         "(5000) or an integer k-points per reciprocal atom, e.g. 8000")
    ap.add_argument('--ncore', type=int, default=None,
                    help="force NCORE (KPAR is then derived; the ELF pass always uses KPAR=1)")
    ap.add_argument('--profile', default='slurm',
                    help="agent profile name (SLURM); default 'slurm'")
    ap.add_argument('--chain', dest='chain', action='store_true', default=True,
                    help="SLURM: chain materials with --dependency=afterok "
                         "(sequential on the cluster). Default on.")
    ap.add_argument('--no-chain', dest='chain', action='store_false',
                    help="SLURM: submit each material independently (parallel).")
    ap.add_argument('--poscar-dir', action='append', default=None,
                    help="folder with already-downloaded POSCARs (<dir>/<id>/POSCAR, "
                         "<dir>/materials/<id>/POSCAR, <dir>/_ht_inputs/<id>/POSCAR, "
                         "<dir>/<id>.vasp); repeatable.  _ht_inputs/ here is always "
                         "checked first, so a rerun never re-downloads")
    ap.add_argument('--api-key', default=None,
                    help="Materials Project API key (else $MP_API_KEY / $PMG_MAPI_KEY)")
    # optional SLURM per-project overrides written into each instructions.txt
    ap.add_argument('--nodes', default=None)
    ap.add_argument('--ntasks-per-node', dest='ntasks_per_node', default=None)
    ap.add_argument('--partition', default=None)
    ap.add_argument('--walltime', default=None)
    ap.add_argument('--account', default=None)
    args = ap.parse_args()

    agent_kind = choose_agent(args.agent)
    api_key    = None            # asked for only if something must be downloaded
    ids        = read_id_list(args.list)

    slurm_opts = {}
    if agent_kind == 'slurm':
        slurm_opts = {
            'NODES':           args.nodes,
            'NTASKS_PER_NODE': args.ntasks_per_node,
            'PARTITION':       args.partition,
            'WALLTIME':        args.walltime,
            'ACCOUNT':         args.account,
        }

    print(f"\nHigh-throughput SCF + ELF setup")
    print(f"  agent      : {agent_kind}")
    print(f"  functional : {args.functional}")
    print(f"  k-points   : " + (f"fixed mesh {args.kmesh}" if args.kmesh
                                  else f"density {args.kpra} (kpra, uniform mesh)"))
    print(f"  relaxation : " + ("01_relax, IBRION=1 ISIF=3 (cell + ions), density "
                                 f"{args.relax_kpra or args.kpra}" if args.relax else "off"))
    print(f"  ELF        : two-step (SCF at auto KPAR, then KPAR=1 ELF pass in 02_scf/elf)")
    print(f"  IDs        : {len(ids)}  ({args.list})\n")

    stage_root = os.path.join(os.getcwd(), STAGE_DIR)
    os.makedirs(stage_root, exist_ok=True)

    ok, failed = [], []
    material_elements = {}
    for mp_id in ids:
        d = os.path.join(stage_root, mp_id)
        os.makedirs(d, exist_ok=True)
        local = find_local_poscar(mp_id, [stage_root] + (args.poscar_dir or []))
        try:
            if local:
                print(f"  {mp_id}: using {os.path.relpath(local)} (no download) ...", end=" ")
                from pymatgen.core import Structure
                structure = Structure.from_file(local)
                if os.path.abspath(local) != os.path.abspath(os.path.join(d, 'POSCAR')):
                    shutil.copy(local, os.path.join(d, 'POSCAR'))
            else:
                print(f"  fetching {mp_id} ...", end=" ", flush=True)
                api_key = api_key or get_api_key(args.api_key)
                structure = fetch_primitive_structure(mp_id, api_key)
                write_poscar(structure, os.path.join(d, 'POSCAR'))
        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(mp_id)
            continue
        write_instructions(os.path.join(d, 'instructions.txt'),
                           mp_id, args.functional, args.mpi, args.encut, slurm_opts,
                           kmesh=args.kmesh, lobster=args.lobster,
                           kpra=args.kpra, kpar=args.kpar, ncore=args.ncore,
                           relax=args.relax, relax_kpra=args.relax_kpra,
                           gga_u=args.gga_u)
        nat = len(structure)
        els = sorted({str(s) for s in structure.composition.elements},
                     key=lambda s: ELEMENT_Z.get(s, 999))
        material_elements[mp_id] = els
        print(f"OK  ({structure.composition.reduced_formula}, {nat} atoms, primitive)")
        ok.append(mp_id)

    if not ok:
        sys.exit("\nERROR: no structures downloaded; nothing to do.")

    runall = os.path.join(os.getcwd(), 'runall.sh')
    write_runall(runall, ok, agent_kind, args.profile, chain=args.chain)

    # Tabulate the elements and their atomic numbers across the screening set
    table = write_element_table(material_elements,
                                os.path.join(stage_root, 'elements_Z.txt'))
    print("\n" + table.rstrip())

    print(f"\nStaged {len(ok)} material(s) in {STAGE_DIR}/")
    print(f"Element/Z table written to {STAGE_DIR}/elements_Z.txt")
    if failed:
        print(f"Skipped {len(failed)} (download failed): {', '.join(failed)}")
    print(f"\nWrote runall.sh. Next:\n")
    print(f"    export VASP_POTCAR_DIR=...   # if not already set")
    print(f"    ./runall.sh\n")


if __name__ == '__main__':
    main()
