#!/usr/bin/env python3
"""
High-Throughput MP -> SCF + ELF driver
======================================
Reads a list of Materials Project IDs (one per line) from an input file,
downloads the PRIMITIVE-cell POSCAR for each from the Materials Project,
and stages a per-material SCF calculation that also computes the electron
localization function (ELF -> ELFCAR, via LELF = .TRUE.).

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

import os
import sys
import argparse
import textwrap

# Repo root = directory containing this script and the agents.
REPO_DIR        = os.path.dirname(os.path.abspath(__file__))
AGENT_LOCAL     = os.path.join(REPO_DIR, 'vasp-agent.py')
AGENT_SLURM     = os.path.join(REPO_DIR, 'vasp-agent-slurm.py')
DEFAULT_LIST    = 'highthrouput_list'      # spelling per the project convention
STAGE_DIR       = '_ht_inputs'


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

    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    return SpacegroupAnalyzer(struct).get_primitive_standard_structure()


# ── per-material staging ────────────────────────────────────────────────────
def write_poscar(structure, path):
    """Write a VASP5 POSCAR (element-symbol line present) for the agent."""
    from pymatgen.io.vasp import Poscar
    Poscar(structure).write_file(path)


def write_instructions(path, mp_id, functional, mpi, encut, slurm_opts):
    """Write a minimal SCF+ELF instructions.txt for one material."""
    lines = [
        f"Project: {mp_id}",
        "# High-throughput SCF + ELF from a Materials Project primitive cell",
        f"Methods: {functional} functional",
        "",
        "Tasks: SCF calculation",
        "",
        "# Electron localization function -> ELFCAR (raw INCAR passthrough)",
        "INCAR scf:",
        "   LELF = .TRUE.",
        "END_INCAR",
        "",
        f"MPI: {mpi}",
    ]
    if encut:
        lines.append(f"ENCUT: {encut}")
    for key, val in (slurm_opts or {}).items():
        if val:
            lines.append(f"{key}: {val}")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")


# ── runall.sh ───────────────────────────────────────────────────────────────
def write_runall(path, ids, agent_kind, profile, chain=True):
    """Emit runall.sh: for each mp-id, call the agent then run/submit the SCF job.

    local:           run_all.sh blocks, so materials run strictly one-at-a-time.
    slurm + chain:   each material's SCF is submitted with
                     --dependency=afterok on the previous material's job, so the
                     cluster runs them sequentially even though all are queued
                     up front. The SCF step (02_scf) is single-step here, so the
                     job id is captured directly with `sbatch --parsable`.
    slurm + no chain: each project's submit_all.sh is called; jobs are
                     independent and run whenever the scheduler allows.
    """
    agent    = AGENT_SLURM if agent_kind == 'slurm' else AGENT_LOCAL
    prof     = f' -p "{profile}"' if (agent_kind == 'slurm' or profile) else ''
    id_array = " ".join(ids)

    with open(path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated by ht-mp-scf.py\n")
        f.write("# Builds inputs and runs an SCF+ELF job for each mp-id, in order.\n")
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
            f.write('    # 2. Submit the SCF step, chained after the previous material\n')
            f.write('    SCFDIR="$HERE/$id/02_scf"\n')
            f.write('    if [ ! -f "$SCFDIR/run.sh" ]; then\n')
            f.write('        echo "ERROR: $SCFDIR/run.sh not generated; skipping $id."\n')
            f.write('        continue\n')
            f.write('    fi\n')
            f.write('    if [ -n "$PREV" ]; then\n')
            f.write('        JID=$(cd "$SCFDIR" && sbatch --parsable --dependency=afterok:$PREV run.sh)\n')
            f.write('    else\n')
            f.write('        JID=$(cd "$SCFDIR" && sbatch --parsable run.sh)\n')
            f.write('    fi\n')
            f.write('    echo "  Submitted $id SCF -> job $JID (after ${PREV:-none})"\n')
            f.write('    PREV=$JID\n')
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
    ap.add_argument('--profile', default='slurm',
                    help="agent profile name (SLURM); default 'slurm'")
    ap.add_argument('--chain', dest='chain', action='store_true', default=True,
                    help="SLURM: chain materials with --dependency=afterok "
                         "(sequential on the cluster). Default on.")
    ap.add_argument('--no-chain', dest='chain', action='store_false',
                    help="SLURM: submit each material independently (parallel).")
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
    api_key    = get_api_key(args.api_key)
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
    print(f"  IDs        : {len(ids)}  ({args.list})\n")

    stage_root = os.path.join(os.getcwd(), STAGE_DIR)
    os.makedirs(stage_root, exist_ok=True)

    ok, failed = [], []
    for mp_id in ids:
        print(f"  fetching {mp_id} ...", end=" ", flush=True)
        try:
            structure = fetch_primitive_structure(mp_id, api_key)
        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(mp_id)
            continue
        d = os.path.join(stage_root, mp_id)
        os.makedirs(d, exist_ok=True)
        write_poscar(structure, os.path.join(d, 'POSCAR'))
        write_instructions(os.path.join(d, 'instructions.txt'),
                           mp_id, args.functional, args.mpi, args.encut, slurm_opts)
        nat = len(structure)
        print(f"OK  ({structure.composition.reduced_formula}, {nat} atoms, primitive)")
        ok.append(mp_id)

    if not ok:
        sys.exit("\nERROR: no structures downloaded; nothing to do.")

    runall = os.path.join(os.getcwd(), 'runall.sh')
    write_runall(runall, ok, agent_kind, args.profile, chain=args.chain)

    print(f"\nStaged {len(ok)} material(s) in {STAGE_DIR}/")
    if failed:
        print(f"Skipped {len(failed)} (download failed): {', '.join(failed)}")
    print(f"\nWrote runall.sh. Next:\n")
    print(f"    export VASP_POTCAR_DIR=...   # if not already set")
    print(f"    ./runall.sh\n")


if __name__ == '__main__':
    main()
