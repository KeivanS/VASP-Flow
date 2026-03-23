#!/usr/bin/env python3
"""
VASP Workflow Agent
===================
Run from the directory that contains your instructions.txt and POSCAR:

    cd VASP_WORKFLOW
    ./vasp-agent.py                   # uses instructions.txt + POSCAR in cwd
    ./vasp-agent.py -i my_inst.txt    # custom instruction file name
    ./vasp-agent.py -i inst.txt -s struct.vasp

Output is written to  <ProjectName>/  in the same directory.
POTCAR is built ONCE at <ProjectName>/POTCAR using $VASP_POTCAR_DIR.
"""

import os, sys, re, shutil, argparse, json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules'))
from instruction_parser import InstructionParser
from vasp_input_generator import VASPInputGenerator


# ── helpers ──────────────────────────────────────────────────────────────
def slugify(name):
    return re.sub(r'[^\w\-]', '_', name.strip()).strip('_') or 'vasp_project'

def check_potcar_dir():
    d = os.environ.get('VASP_POTCAR_DIR', '')
    if not d or not os.path.isdir(d):
        print("\nERROR: VASP_POTCAR_DIR is not set or does not exist.")
        print("Add this line to your ~/.bash_profile and open a new terminal:\n")
        print('  export VASP_POTCAR_DIR="$HOME/path/to/potcar/PAW_PBE"\n')
        sys.exit(1)
    return d

def build_potcar(elements, potcar_dir, out_path, choices=None):
    choices = choices or {}
    with open(out_path, 'wb') as out:
        for el in elements:
            if el in choices:
                variants_to_try = [choices[el]]
            else:
                variants_to_try = [el, f"{el}_sv", f"{el}_pv", f"{el}_d"]
            for variant in variants_to_try:
                p = os.path.join(potcar_dir, variant, 'POTCAR')
                if os.path.isfile(p):
                    with open(p, 'rb') as src:
                        out.write(src.read())
                    print(f"    added {variant}")
                    break
            else:
                print(f"  WARNING: no POTCAR found for element '{el}' in {potcar_dir}")

def chmod_x(path):
    os.chmod(path, 0o755)

def link_potcar(calc_dir, potcar_path):
    link = os.path.join(calc_dir, 'POTCAR')
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.relpath(potcar_path, calc_dir), link)


# ── agent ─────────────────────────────────────────────────────────────────
class VASPWorkflowAgent:

    def __init__(self, instructions_file, poscar_file):
        self.cwd = os.getcwd()

        print("\n" + "="*58)
        print("  VASP Workflow Agent")
        print("="*58)
        print(f"\nReading {os.path.basename(instructions_file)} ...")

        self.parser = InstructionParser(instructions_file)
        inst = self.parser.instructions

        self.project_label = inst.get('project_name', 'VASP_Calculation')
        self.project_dir   = os.path.join(self.cwd, slugify(self.project_label))

        print(f"  Project   : {self.project_label}")
        print(f"  Output    : {os.path.relpath(self.project_dir, self.cwd)}/")
        print(f"  Functional: {inst.get('functional')}")
        print(f"  SOC       : {inst.get('soc')}")
        print(f"  MPI ranks : {inst.get('mpi_np', 1)}")
        print(f"  Tasks     : {', '.join(inst.get('tasks', []))}")

        self.poscar_file = os.path.abspath(poscar_file)
        self.inst_file   = os.path.abspath(instructions_file)
        self.generator   = VASPInputGenerator(self.poscar_file, inst)

        os.makedirs(self.project_dir, exist_ok=True)
        shutil.copy(self.inst_file,   self.project_dir)
        shutil.copy(self.poscar_file, os.path.join(self.project_dir, 'POSCAR'))

    def run(self):
        inst  = self.parser.instructions
        tasks = inst.get('tasks', [])
        pd    = self.project_dir

        # ── POTCAR built once ────────────────────────────────────────────
        potcar_path = os.path.join(pd, 'POTCAR')
        if self.generator.elements:
            potcar_dir = check_potcar_dir()
            # Read explicit POTCAR variant choices from GUI if present
            choices = {}
            choices_file = os.path.join(self.cwd, 'potcar_choices.json')
            if os.path.isfile(choices_file):
                try:
                    choices = json.loads(Path(choices_file).read_text())
                    print(f"  POTCAR choices: {choices}")
                except Exception:
                    pass
            print(f"\nBuilding POTCAR from $VASP_POTCAR_DIR ...")
            build_potcar(self.generator.elements, potcar_dir, potcar_path, choices)
        else:
            print("\nWARNING: could not read elements from POSCAR; POTCAR skipped.")

        # ── calculation steps ─────────────────────────────────────────────
        print("\nGenerating input files:")
        calc_dirs = {}

        if 'relax' in tasks:
            d = os.path.join(pd, '01_relax')
            self.generator.generate_relax_input(d)
            link_potcar(d, potcar_path)
            calc_dirs['relax'] = d
            print(f"  01_relax/    INCAR  KPOINTS  POSCAR  POTCAR  run.sh")

        if 'scf' in tasks:
            d = os.path.join(pd, '02_scf')
            self.generator.generate_scf_input(d, calc_dirs.get('relax'))
            link_potcar(d, potcar_path)
            calc_dirs['scf'] = d
            print(f"  02_scf/      INCAR  KPOINTS  POSCAR  POTCAR  run.sh")

        if 'bands' in tasks:
            d = os.path.join(pd, '03_bands')
            scf_d = calc_dirs.get('scf', os.path.join(pd, '02_scf'))
            self.generator.generate_bands_input(d, scf_d)
            link_potcar(d, potcar_path)
            calc_dirs['bands'] = d
            print(f"  03_bands/    INCAR  KPOINTS  POTCAR  run.sh")

        if 'dos' in tasks:
            d = os.path.join(pd, '04_dos')
            scf_d = calc_dirs.get('scf', os.path.join(pd, '02_scf'))
            self.generator.generate_dos_input(d, scf_d)
            link_potcar(d, potcar_path)
            calc_dirs['dos'] = d
            print(f"  04_dos/      INCAR  KPOINTS  POTCAR  run.sh")

        # ── convergence tests ─────────────────────────────────────────────
        conv = inst.get('convergence', {})
        has_convergence = (conv.get('kpoints', {}).get('enabled') or
                           conv.get('encut',   {}).get('enabled'))
        if has_convergence:
            self._gen_convergence(conv, potcar_path)
            print(f"  00_convergence/  kpoints/run.sh  encut/run.sh")

        # ── analysis scripts ──────────────────────────────────────────────
        self._gen_analysis(calc_dirs)
        print(f"  analysis/    plot_results.py")

        # ── runner scripts ────────────────────────────────────────────────
        if has_convergence:
            # Two separate scripts — user must intervene between them
            self._gen_run_convergence()
            self._gen_run_calculations(calc_dirs)
            print(f"  run_convergence.sh   (STEP 1 — run this first)")
            print(f"  run_calculations.sh  (STEP 2 — run after reviewing results)")
        else:
            self._gen_run_all(calc_dirs)
            print(f"  run_all.sh   (runs every step in order)")

        print(f"  analyze.sh   (extracts results after all calculations)")

        # ── terminal summary ──────────────────────────────────────────────
        proj_rel = os.path.relpath(self.project_dir, self.cwd)
        print(f"\n{'='*58}")
        print(f"  Done — {proj_rel}/")
        print(f"{'='*58}")

        if has_convergence:
            print(f"""
  Two-phase workflow (convergence tests requested):

  PHASE 1 — run convergence tests:

    cd {proj_rel}
    ./run_convergence.sh

  Then open  00_convergence/encut/encut_convergence.dat
         and 00_convergence/kpoints/kpoint_convergence.dat
  and choose your production ENCUT and k-mesh.

  Update INCAR and KPOINTS in every calculation directory:
    01_relax/  02_scf/  03_bands/  04_dos/

  PHASE 2 — run production calculations:

    ./run_calculations.sh

  After all steps complete:

    ./analyze.sh
""")
        else:
            print(f"""
  cd {proj_rel} && ./run_all.sh

  After completion:  ./analyze.sh
""")

    # ── runner scripts ──────────────────────────────────────────────────────
    def _gen_run_all(self, calc_dirs):
        """Single run_all.sh for the no-convergence case."""
        path = os.path.join(self.project_dir, 'run_all.sh')
        ordered_keys = [k for k in ['relax', 'scf', 'bands', 'dos'] if k in calc_dirs]
        with open(path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Run all calculations for: {self.project_label}\n\n")
            f.write("set -e\n")
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')
            for task in ordered_keys:
                dirname = os.path.basename(calc_dirs[task])
                f.write(f'echo ""\necho ">>> {dirname}"\n')
                f.write(f'cd "$HERE/{dirname}"\n./run.sh\ncd "$HERE"\n\n')
            f.write('echo ""\necho "All steps complete.  Run ./analyze.sh next."\n')
        chmod_x(path)

    def _gen_run_convergence(self):
        """PHASE 1 script — runs convergence tests only."""
        path = os.path.join(self.project_dir, 'run_convergence.sh')
        conv_dir = os.path.join(self.project_dir, '00_convergence')
        has_encut   = os.path.isdir(os.path.join(conv_dir, 'encut'))
        has_kpoints = os.path.isdir(os.path.join(conv_dir, 'kpoints'))
        with open(path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# PHASE 1 — Convergence tests for: {self.project_label}\n")
            f.write("#\n")
            f.write("# After this script finishes, review the output files:\n")
            if has_encut:
                f.write("#   00_convergence/encut/encut_convergence.dat\n")
            if has_kpoints:
                f.write("#   00_convergence/kpoints/kpoint_convergence.dat\n")
            f.write("#\n")
            f.write("# Choose your production ENCUT and k-mesh, then update\n")
            f.write("# INCAR (ENCUT line) and KPOINTS in:\n")
            f.write("#   01_relax/  02_scf/  03_bands/  04_dos/\n")
            f.write("#\n")
            f.write("# Then run:  ./run_calculations.sh\n\n")
            f.write("set -e\n")
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')
            if has_encut:
                f.write('echo ""\necho ">>> ENCUT convergence"\n')
                f.write('cd "$HERE/00_convergence/encut"\n./run.sh\ncd "$HERE"\n\n')
            if has_kpoints:
                f.write('echo ""\necho ">>> k-point convergence"\n')
                f.write('cd "$HERE/00_convergence/kpoints"\n./run.sh\ncd "$HERE"\n\n')
            f.write('echo ""\n')
            f.write('echo "Convergence tests complete."\n')
            if has_encut:
                f.write('echo "  ENCUT results : 00_convergence/encut/encut_convergence.dat"\n')
            if has_kpoints:
                f.write('echo "  k-point results: 00_convergence/kpoints/kpoint_convergence.dat"\n')
            f.write('echo ""\n')
            f.write('echo "Review the results, update INCAR/KPOINTS in each calculation"\n')
            f.write('echo "directory, then run:  ./run_calculations.sh"\n')
        chmod_x(path)

    def _gen_run_calculations(self, calc_dirs):
        """PHASE 2 script — patches ENCUT/KPOINTS from user input then runs calculations."""
        path = os.path.join(self.project_dir, 'run_calculations.sh')
        ordered_keys = [k for k in ['relax', 'scf', 'bands', 'dos'] if k in calc_dirs]
        # directories that use an automatic k-mesh (not line-mode)
        auto_kpoints_dirs = [os.path.basename(calc_dirs[k])
                             for k in ordered_keys if k != 'bands']

        with open(path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# PHASE 2 — Production calculations for: {self.project_label}\n")
            f.write("# Prompts for converged ENCUT and k-meshes, updates all input\n")
            f.write("# files, then runs every calculation in order.\n\n")
            f.write("set -e\n")
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')

            # ── prompt ───────────────────────────────────────────────────
            f.write('echo ""\n')
            f.write('echo "=== Convergence parameters ==="\n')
            f.write('echo "Check your convergence results:"\n')
            f.write('echo "  cat 00_convergence/encut/encut_convergence.dat"\n')
            f.write('echo "  cat 00_convergence/kpoints/kpoint_convergence.dat"\n')
            f.write('echo ""\n')
            f.write('read -p "Enter converged ENCUT (eV) [e.g. 520]: " ENCUT\n')
            f.write('if [ -z "$ENCUT" ]; then echo "ERROR: ENCUT cannot be empty."; exit 1; fi\n\n')
            f.write('read -p "Enter converged k-mesh for relax/SCF (e.g. 12x12x6): " KMESH\n')
            f.write('if [ -z "$KMESH" ]; then echo "ERROR: k-mesh cannot be empty."; exit 1; fi\n\n')
            f.write('read -p "Enter denser k-mesh for DOS (e.g. 24x24x12, or same as SCF): " KMESH_DOS\n')
            f.write('if [ -z "$KMESH_DOS" ]; then KMESH_DOS=$KMESH; fi\n\n')

            # parse both meshes
            for var, mesh_var in [('NX', 'KMESH'), ('NX_DOS', 'KMESH_DOS')]:
                prefix = '' if mesh_var == 'KMESH' else '_DOS'
                f.write(f'NX{prefix}=$(echo "${mesh_var}" | cut -dx -f1)\n')
                f.write(f'NY{prefix}=$(echo "${mesh_var}" | cut -dx -f2)\n')
                f.write(f'NZ{prefix}=$(echo "${mesh_var}" | cut -dx -f3)\n')
                f.write(f'if [ -z "$NZ{prefix}" ]; then NZ{prefix}=$NY{prefix}; fi\n')
            f.write('\n')

            # ── patch ENCUT in all INCAR files ───────────────────────────
            f.write('echo ""\n')
            f.write('echo "Updating input files:"\n')
            for key in ordered_keys:
                dirname = os.path.basename(calc_dirs[key])
                f.write(f'sed -i.bak "s/^ENCUT.*/ENCUT = $ENCUT/" "$HERE/{dirname}/INCAR"\n')
                f.write(f'echo "  {dirname}/INCAR  →  ENCUT = $ENCUT"\n')
            f.write('\n')

            # ── patch KPOINTS: relax and scf use KMESH ───────────────────
            for key in ordered_keys:
                if key in ('bands', 'dos'):
                    continue
                dirname = os.path.basename(calc_dirs[key])
                f.write(f'printf "Automatic Gamma mesh\\n0\\nGamma\\n  %d  %d  %d\\n  0  0  0\\n" \\\n')
                f.write(f'    $NX $NY $NZ > "$HERE/{dirname}/KPOINTS"\n')
                f.write(f'echo "  {dirname}/KPOINTS  →  $KMESH"\n')

            # ── patch KPOINTS: dos uses KMESH_DOS ────────────────────────
            if 'dos' in calc_dirs:
                dirname = os.path.basename(calc_dirs['dos'])
                f.write(f'printf "Automatic Gamma mesh\\n0\\nGamma\\n  %d  %d  %d\\n  0  0  0\\n" \\\n')
                f.write(f'    $NX_DOS $NY_DOS $NZ_DOS > "$HERE/{dirname}/KPOINTS"\n')
                f.write(f'echo "  {dirname}/KPOINTS  →  $KMESH_DOS  (denser for DOS)"\n')

            f.write('\necho ""\n')

            # ── run each calculation ──────────────────────────────────────
            for task in ordered_keys:
                dirname = os.path.basename(calc_dirs[task])
                f.write(f'echo ">>> {dirname}"\n')
                f.write(f'cd "$HERE/{dirname}"\n./run.sh\ncd "$HERE"\necho ""\n\n')

            f.write('echo "All calculations complete.  Run ./analyze.sh next."\n')
        chmod_x(path)

    # ── convergence ─────────────────────────────────────────────────────────
    def _gen_convergence(self, conv, potcar_path):
        pd  = self.project_dir
        mpi = self.generator._get_mpi_cmd(self.generator._get_vasp_exec())

        kp = conv.get('kpoints', {})
        if kp.get('enabled'):
            d = os.path.join(pd, '00_convergence', 'kpoints')
            os.makedirs(d, exist_ok=True)
            link_potcar(d, potcar_path)
            shutil.copy(self.poscar_file, os.path.join(d, 'POSCAR'))
            with open(os.path.join(d, 'INCAR'), 'w') as f:
                f.write(self.generator._generate_incar_scf())

            script = os.path.join(d, 'run.sh')
            with open(script, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("set -e\n")
                f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')

                explicit = kp.get('meshes', [])
                if explicit:
                    # ── explicit list mode ───────────────────────────────
                    f.write(f"# K-point convergence — explicit meshes\n\n")
                    for mesh in explicit:
                        nx, ny, nz = mesh
                        label = f"{nx}x{ny}x{nz}"
                        f.write(f'echo "  {label} ..."\n')
                        f.write(f'mkdir -p "{label}"\n')
                        f.write(f'cp INCAR POSCAR POTCAR "{label}/"\n')
                        f.write(f'printf "Automatic Gamma mesh\\n0\\nGamma\\n  {nx}  {ny}  {nz}\\n  0  0  0\\n" > "{label}/KPOINTS"\n')
                        f.write(f'cd "{label}"\n')
                        f.write(f'{mpi} > vasp.out 2>&1\n')
                        f.write(f'E=$(grep "energy  without" OUTCAR | tail -1 | awk \'{{print $7}}\')\n')
                        f.write(f'echo "{label}  $E" >> "$HERE/kpoint_convergence.dat"\n')
                        f.write(f'cd "$HERE"\n\n')
                else:
                    # ── range mode ───────────────────────────────────────
                    start, end = kp['range']
                    # Build the full mesh list respecting all three components
                    # step = smallest common increment across in-plane direction
                    k0x, k0y, k0z = start
                    k1x, k1y, k1z = end
                    f.write(f"# K-point convergence: {k0x}x{k0y}x{k0z} → {k1x}x{k1y}x{k1z}\n\n")
                    # Generate meshes by doubling / stepping the in-plane component
                    # and scaling kz proportionally if it differs from kx
                    step = (k1x - k0x) // 4 if (k1x - k0x) >= 4 else 2
                    f.write(f"for NX in $(seq {k0x} {step} {k1x}); do\n")
                    if k0z == k0x:   # isotropic: kz tracks kx
                        f.write(f'    NZ=$NX\n')
                    elif k0z > 0:    # proportional: kz = kx * (k0z/k0x)
                        ratio = k0z / k0x
                        f.write(f'    NZ=$(echo "$NX * {ratio}" | bc | xargs printf "%.0f")\n')
                    else:
                        f.write(f'    NZ=1\n')
                    f.write(f'    LABEL="${{NX}}x${{NX}}x${{NZ}}"\n')
                    f.write(f'    mkdir -p "$LABEL"\n')
                    f.write(f'    cp INCAR POSCAR POTCAR "$LABEL/"\n')
                    f.write(f'    printf "Automatic Gamma mesh\\n0\\nGamma\\n  %d  %d  %d\\n  0  0  0\\n" $NX $NX $NZ > "$LABEL/KPOINTS"\n')
                    f.write(f'    cd "$LABEL"\n')
                    f.write(f'    echo "  $LABEL ..."\n')
                    f.write(f'    {mpi} > vasp.out 2>&1\n')
                    f.write(f'    E=$(grep "energy  without" OUTCAR | tail -1 | awk \'{{print $7}}\')\n')
                    f.write(f'    echo "$LABEL  $E" >> "$HERE/kpoint_convergence.dat"\n')
                    f.write(f'    cd "$HERE"\n')
                    f.write(f"done\n\n")

                f.write('echo "Results → kpoint_convergence.dat"\n')
            chmod_x(script)

        ec = conv.get('encut', {})
        if ec.get('enabled') and ec.get('range'):
            e0, e1 = ec['range']
            d = os.path.join(pd, '00_convergence', 'encut')
            os.makedirs(d, exist_ok=True)
            link_potcar(d, potcar_path)
            shutil.copy(self.poscar_file, os.path.join(d, 'POSCAR'))
            with open(os.path.join(d, 'KPOINTS'), 'w') as f:
                f.write(self.generator._generate_kpoints_auto('medium'))
            with open(os.path.join(d, 'INCAR'), 'w') as f:
                f.write(self.generator._generate_incar_scf())
            script = os.path.join(d, 'run.sh')
            with open(script, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(f"# ENCUT convergence: {e0} to {e1} eV\n")
                f.write("set -e\n")
                f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')
                f.write(f"for EC in $(seq {e0} 50 {e1}); do\n")
                f.write('    DIR="encut_${EC}"\n')
                f.write('    mkdir -p "$DIR"\n')
                f.write('    cp POSCAR POTCAR KPOINTS "$DIR/"\n')
                f.write('    sed "s/ENCUT.*/ENCUT = $EC/" INCAR > "$DIR/INCAR"\n')
                f.write('    cd "$DIR"\n')
                f.write('    echo "  ENCUT=$EC ..."\n')
                f.write(f'    {mpi} > vasp.out 2>&1\n')
                f.write('    E=$(grep "energy  without" OUTCAR | tail -1 | awk \'{print $7}\')\n')
                f.write('    echo "$EC  $E" >> "$HERE/encut_convergence.dat"\n')
                f.write('    cd "$HERE"\n')
                f.write("done\n\n")
                f.write('echo "Results → encut_convergence.dat"\n')
            chmod_x(script)

    # ── analysis ────────────────────────────────────────────────────────────
    def _gen_analysis(self, calc_dirs):
        pd  = self.project_dir
        ana = os.path.join(pd, 'analysis')
        os.makedirs(ana, exist_ok=True)

        has_relax = 'relax' in calc_dirs
        has_scf   = 'scf'   in calc_dirs
        has_bands = 'bands' in calc_dirs
        has_dos   = 'dos'   in calc_dirs

        # Orbital projections from instructions (e.g. ['C:s', 'C:p'])
        dos_proj  = self.parser.instructions.get('dos_projections', [])
        # Build sumo --orbitals string, e.g. "C s p"
        # Group by element: C s p  Mo d  S p
        from collections import defaultdict
        orb_map = defaultdict(list)
        for dp in dos_proj:
            el, orb = dp.split(':') if ':' in dp else (dp, 's')
            orb_map[el].append(orb)
        # sumo format: "El orb1 orb2, El2 orb1"
        sumo_orb = ', '.join(f"{el} {' '.join(orbs)}" for el, orbs in orb_map.items())

        # ── analyze.sh ───────────────────────────────────────────────────
        sh = os.path.join(pd, 'analyze.sh')
        with open(sh, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Analysis script for: {self.project_label}\n")
            f.write("# Uses sumo for publication-quality plots (pip install sumo)\n\n")
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write('mkdir -p "$HERE/analysis"\n\n')

            # ── quick text extraction ────────────────────────────────────
            if has_relax:
                f.write('echo "=== Relaxation ==="\n')
                f.write('grep "energy  without" "$HERE/01_relax/OUTCAR" | tail -1\n')
                f.write('grep "reached required accuracy" "$HERE/01_relax/OUTCAR" '
                        '&& echo "  Converged" || echo "  WARNING: check NSW"\n\n')
            if has_scf:
                f.write('echo "=== SCF ==="\n')
                f.write('grep "E-fermi" "$HERE/02_scf/OUTCAR" | tail -1\n\n')

            # ── band structure with sumo ─────────────────────────────────
            if has_bands:
                f.write('echo "=== Band structure (sumo) ==="\n')
                f.write('if command -v sumo-bandplot &>/dev/null; then\n')
                # Extract Fermi level from DOS (dense k-mesh) with SCF fallback
                f.write('    # Use Fermi level from dense-mesh DOS OUTCAR (more accurate)\n')
                f.write('    EFERMI=""\n')
                f.write('    if [ -f "$HERE/04_dos/OUTCAR" ]; then\n')
                f.write('        EFERMI=$(grep "E-fermi" "$HERE/04_dos/OUTCAR" | tail -1 | awk \'{print $3}\')\n')
                f.write('    elif [ -f "$HERE/02_scf/OUTCAR" ]; then\n')
                f.write('        EFERMI=$(grep "E-fermi" "$HERE/02_scf/OUTCAR" | tail -1 | awk \'{print $3}\')\n')
                f.write('    fi\n')
                f.write('    if [ -n "$EFERMI" ] && [ -f "$HERE/03_bands/vasprun.xml" ]; then\n')
                f.write('        echo "  Patching vasprun.xml efermi to $EFERMI eV"\n')
                f.write('        sed -i.efermi_bak \'s|<i name="efermi">.*</i>|<i name="efermi">  \'$EFERMI\'  </i>|\' "$HERE/03_bands/vasprun.xml"\n')
                f.write('    fi\n')
                f.write('    cd "$HERE/03_bands"\n')
                f.write('    sumo-bandplot \\\n')
                f.write('        --prefix "$(basename $HERE)" \\\n')
                f.write('        --ymin -4 --ymax 4 \\\n')
                f.write('        2>&1\n')
                f.write('    mv *_band.* "$HERE/analysis/" 2>/dev/null || true\n')
                f.write('    cd "$HERE"\n')
                f.write('    echo "  Saved: analysis/*_band.*"\n')
                f.write('else\n')
                f.write('    echo "  sumo not found — falling back to EIGENVAL copy"\n')
                f.write('    cp "$HERE/03_bands/EIGENVAL" "$HERE/analysis/"\n')
                f.write('fi\n\n')

            # ── DOS with sumo ────────────────────────────────────────────
            if has_dos:
                f.write('echo "=== DOS (sumo) ==="\n')
                f.write('if command -v sumo-dosplot &>/dev/null; then\n')
                f.write('    cd "$HERE/04_dos"\n')
                f.write('    # Total DOS\n')
                f.write('    sumo-dosplot \\\n')
                f.write('        --prefix "$(basename $HERE)_total" \\\n')
                f.write('        --xmin -6 --xmax 6 \\\n')
                f.write('        2>&1\n')
                # Projected DOS — only if projections were specified
                if sumo_orb:
                    f.write('    # Orbital-projected DOS\n')
                    f.write(f'    sumo-dosplot \\\n')
                    f.write(f'        --prefix "$(basename $HERE)_proj" \\\n')
                    f.write(f'        --orbitals "{sumo_orb}" \\\n')
                    f.write(f'        --xmin -6 --xmax 6 \\\n')
                    f.write(f'        2>&1\n')
                f.write('    mv *_dos.* "$HERE/analysis/" 2>/dev/null || true\n')
                f.write('    cd "$HERE"\n')
                f.write('    echo "  Saved: analysis/*_dos.*"\n')
                f.write('else\n')
                f.write('    echo "  sumo not found — falling back to DOSCAR copy"\n')
                f.write('    cp "$HERE/04_dos/DOSCAR" "$HERE/analysis/"\n')
                f.write('fi\n\n')

            f.write('echo ""\n')
            f.write('echo "Done. Plots in analysis/"\n')
            f.write('echo "Tip: run individual sumo commands from each calc dir"\n')
            f.write('echo "  e.g.: cd 03_bands && sumo-bandplot --ymin -4 --ymax 4"\n')
        chmod_x(sh)

        # ── plot_results.py (matplotlib fallback) ────────────────────────
        py = os.path.join(ana, 'plot_results.py')
        with open(py, 'w') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write(f'"""Fallback matplotlib plots for {self.project_label}.\n')
            f.write('For better plots use sumo: sumo-bandplot / sumo-dosplot\n"""\n\n')
            f.write("import numpy as np, matplotlib.pyplot as plt, os\n\n")
            f.write("HERE = os.path.dirname(os.path.abspath(__file__))\n")
            f.write("ROOT = os.path.join(HERE, '..')\n\n")
            f.write("def efermi(outcar):\n")
            f.write("    for line in open(outcar):\n")
            f.write("        if 'E-fermi' in line: return float(line.split()[2])\n")
            f.write("    return 0.0\n\n")
            if has_scf:
                f.write("ef = efermi(os.path.join(ROOT, '02_scf/OUTCAR'))\n")
                f.write("print(f'E_F = {ef:.4f} eV')\n\n")
            else:
                f.write("ef = 0.0\n\n")
            if has_bands:
                f.write("# ── bands ─────────────────────────────────────\n")
                f.write("ev = os.path.join(ROOT, '03_bands/EIGENVAL')\n")
                f.write("lines = open(ev).readlines()\n")
                f.write("nkpts, nbands = int(lines[5].split()[1]), int(lines[5].split()[2])\n")
                f.write("E = [[float(lines[7+ik*(nbands+2)+ib].split()[1])\n")
                f.write("      for ib in range(nbands)] for ik in range(nkpts)]\n")
                f.write("E = np.array(E)\n")
                f.write("fig, ax = plt.subplots(figsize=(5,6))\n")
                f.write("for ib in range(nbands): ax.plot(E[:,ib]-ef,'b-',lw=0.6)\n")
                f.write("ax.axhline(0,color='k',lw=0.5,ls='--')\n")
                f.write("ax.set(ylim=(-4,4),ylabel='E−EF (eV)',xlabel='k-index',title='Bands')\n")
                f.write("plt.tight_layout()\n")
                f.write("fig.savefig(os.path.join(HERE,'bands.pdf'),dpi=300)\n")
                f.write("print('Saved bands.pdf')\n\n")
            if has_dos:
                f.write("# ── DOS ───────────────────────────────────────\n")
                f.write("dc = os.path.join(ROOT, '04_dos/DOSCAR')\n")
                f.write("with open(dc) as fh:\n")
                f.write("    [fh.readline() for _ in range(5)]\n")
                f.write("    nedos = int(fh.readline().split()[2])\n")
                f.write("# total DOS block: col0=energy, col1=total, col2=integrated\n")
                f.write("dos = np.loadtxt(dc, skiprows=6, max_rows=nedos)\n")
                # Projected DOS — read per-site blocks if LORBIT=11
                if dos_proj:
                    elements = list(orb_map.keys())
                    f.write("# projected DOS blocks follow the total block\n")
                    f.write("all_blocks = []\n")
                    f.write("with open(dc) as fh:\n")
                    f.write("    lines_dc = fh.readlines()\n")
                    f.write("# Each site block: nedos lines after a header line\n")
                    f.write("# Block 0 = total (already read); blocks 1..N = per site\n")
                    f.write("block_start = 6 + nedos + 1  # skip total block header+data\n")
                    f.write("try:\n")
                    f.write("    proj_raw = np.loadtxt(dc, skiprows=block_start, max_rows=nedos)\n")
                    f.write("    # LORBIT=11 cols: energy s py pz px dxy dyz dz2 dxz dx2\n")
                    f.write("    # sum s: col 1; p: cols 2+3+4; d: cols 5-9\n")
                    f.write("    s_dos = proj_raw[:, 1]\n")
                    f.write("    p_dos = proj_raw[:, 2] + proj_raw[:, 3] + proj_raw[:, 4]\n")
                    f.write("    has_proj = True\n")
                    f.write("except Exception:\n")
                    f.write("    has_proj = False\n")
                f.write("fig, ax = plt.subplots(figsize=(4,6))\n")
                f.write("ax.plot(dos[:,1], dos[:,0]-ef, 'k-', lw=1.2, label='Total')\n")
                if dos_proj:
                    f.write("if has_proj:\n")
                    f.write("    ax.plot(s_dos, dos[:,0]-ef, 'b--', lw=0.9, label='s')\n")
                    f.write("    ax.plot(p_dos, dos[:,0]-ef, 'r--', lw=0.9, label='p')\n")
                    f.write("ax.legend(fontsize=10)\n")
                f.write("ax.axvline(0,color='gray',lw=0.5,ls='--')\n")
                f.write("ax.set(xlabel='DOS (states/eV)',ylabel='E−EF (eV)',ylim=(-5,5),title='DOS')\n")
                f.write("plt.tight_layout()\n")
                f.write("fig.savefig(os.path.join(HERE,'dos.pdf'),dpi=300)\n")
                f.write("print('Saved dos.pdf')\n\n")
            f.write("plt.show()\n")
        chmod_x(py)


# ── entry point ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='VASP Workflow Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Place instructions.txt and POSCAR in VASP_WORKFLOW/, then run:

    ./vasp-agent.py

Output goes to  <ProjectName>/  in the same directory.
POTCAR is built once from $VASP_POTCAR_DIR (set in ~/.bash_profile).
        """)
    ap.add_argument('-i','--instructions', default='instructions.txt',
                    help='Instruction file (default: instructions.txt)')
    ap.add_argument('-s','--poscar',       default='POSCAR',
                    help='POSCAR file       (default: POSCAR)')
    args = ap.parse_args()

    for fpath, label in [(args.instructions,'instructions'), (args.poscar,'POSCAR')]:
        if not os.path.isfile(fpath):
            print(f"ERROR: {label} file not found: {os.path.abspath(fpath)}")
            sys.exit(1)

    try:
        VASPWorkflowAgent(args.instructions, args.poscar).run()
    except SystemExit:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
