#!/usr/bin/env python3
"""
VASP Workflow Agent — SLURM HPC edition
========================================
Generates the same INCAR / KPOINTS / POSCAR / POTCAR input files as
vasp-agent.py, but replaces every run.sh with a proper SLURM batch
script and generates submit_all.sh (or submit_convergence.sh +
submit_calculations.sh) that chain jobs with --dependency=afterok so
the full workflow runs unattended on the cluster.

Usage:
    ./vasp-agent-slurm.py                          # instructions.txt + POSCAR in cwd
    ./vasp-agent-slurm.py -i my_inst.txt -s struct.vasp
    ./vasp-agent-slurm.py --profile slurm          # load profiles/slurm.json

The profile JSON must contain a "slurm" block:
    {
      "vasp_std":         "~/BIN/vasp_std",
      "vasp_ncl":         "~/BIN/vasp_ncl",
      "vasp_gam":         "~/BIN/vasp_gam",
      "mpi_cmd":          "srun",
      "modules":          ["intel/2021", "impi/2021", "vasp/6.3.2"],
      "slurm": {
        "partition":       "standard",
        "nodes":           2,
        "ntasks_per_node": 64,
        "time":            "12:00:00",
        "account":         "",
        "output":          "slurm-%j.out",
        "error":           "slurm-%j.err"
      }
    }
"""

import os, sys, re, shutil, argparse, json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules'))
from instruction_parser import InstructionParser
from vasp_input_generator import VASPInputGenerator


# ── helpers (identical to vasp-agent.py) ─────────────────────────────────
def slugify(name):
    return re.sub(r'[^\w\-]', '_', name.strip()).strip('_') or 'vasp_project'

def check_potcar_dir():
    d = os.environ.get('VASP_POTCAR_DIR', '')
    if not d or not os.path.isdir(d):
        print("\nERROR: VASP_POTCAR_DIR is not set or does not exist.")
        print('  export VASP_POTCAR_DIR="$HOME/path/to/potcar/PAW_PBE"\n')
        sys.exit(1)
    return d

def build_potcar(elements, potcar_dir, out_path, choices=None):
    choices = choices or {}
    with open(out_path, 'wb') as out:
        for el in elements:
            variants = [choices[el]] if el in choices else [el, f"{el}_sv", f"{el}_pv", f"{el}_d"]
            for variant in variants:
                p = os.path.join(potcar_dir, variant, 'POTCAR')
                if os.path.isfile(p):
                    with open(p, 'rb') as src:
                        out.write(src.read())
                    print(f"    added {variant}")
                    break
            else:
                print(f"  WARNING: no POTCAR found for '{el}' in {potcar_dir}")

def chmod_x(path):
    os.chmod(path, 0o755)

def link_potcar(calc_dir, potcar_path):
    link = os.path.join(calc_dir, 'POTCAR')
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.relpath(potcar_path, calc_dir), link)

def load_profile(profile_name: str) -> dict:
    if not profile_name:
        return {}
    profiles_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles')
    path = os.path.join(profiles_dir, f'{profile_name}.json')
    if not os.path.isfile(path):
        print(f"WARNING: profile '{profile_name}' not found at {path}; using defaults.")
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith('_')}


# ── SLURM agent ───────────────────────────────────────────────────────────
class SLURMVASPAgent:

    def __init__(self, instructions_file, poscar_file, profile: dict = None):
        self.cwd = os.getcwd()

        print("\n" + "="*60)
        print("  VASP Workflow Agent — SLURM HPC edition")
        print("="*60)
        print(f"\nReading {os.path.basename(instructions_file)} ...")

        self.parser    = InstructionParser(instructions_file)
        inst           = self.parser.instructions
        self.profile   = profile or {}

        # ── SLURM settings: instructions override profile, profile overrides defaults ──
        slurm = self.profile.get('slurm', {})

        # Helper: instruction value → profile value → built-in default
        def _get(inst_key, prof_key, default):
            v = inst.get(inst_key)
            return v if v is not None else slurm.get(prof_key, default)

        self.partition       = _get('slurm_partition',      'partition',       'standard')
        self.nodes           = int(_get('slurm_nodes',      'nodes',           1))
        self.ntasks_per_node = int(_get('slurm_ntasks_per_node', 'ntasks_per_node', 16))
        self.slurm_time      = _get('slurm_walltime',       'time',            '12:00:00')
        self.account         = _get('slurm_account',        'account',         '')
        self.slurm_output    = slurm.get('output',          'slurm-%j.out')
        self.slurm_error     = slurm.get('error',           'slurm-%j.err')
        self.modules         = self.profile.get('modules',  [])
        self.mpi_cmd         = self.profile.get('mpi_cmd',  'srun')
        self.ntasks          = self.nodes * self.ntasks_per_node

        # ── Propagate total MPI ranks to instructions so the generator
        #    computes correct KPAR/NCORE defaults (unless MPI was explicit) ──
        if not inst.get('mpi_np') or inst.get('mpi_np') == 1:
            inst['mpi_np'] = self.ntasks

        # ── VASP binaries from profile ────────────────────────────────────
        soc = inst.get('soc', False)
        if soc:
            self.vasp_exec = self.profile.get('vasp_ncl', 'vasp_ncl')
        else:
            self.vasp_exec = self.profile.get('vasp_std', 'vasp_std')

        self.project_label = inst.get('project_name', 'VASP_Calculation')
        self.project_dir   = os.path.join(self.cwd, slugify(self.project_label))

        print(f"  Project   : {self.project_label}")
        print(f"  Output    : {os.path.relpath(self.project_dir, self.cwd)}/")
        print(f"  Functional: {inst.get('functional')}")
        print(f"  SOC       : {inst.get('soc')}")
        print(f"  Tasks     : {', '.join(inst.get('tasks', []))}")
        # ── KPAR / NCORE: instruction → auto-computed by generator later ──
        kpar  = inst.get('kpar')
        ncore = inst.get('ncore')
        kpar_note  = f"{kpar} (from instructions)" if kpar  else f"auto ({self.nodes}, one per node)"
        ncore_note = f"{ncore} (from instructions)" if ncore else f"auto (√{self.ntasks_per_node} ≈ {self._default_ncore()})"

        print(f"\n  SLURM settings (instructions override profile):")
        print(f"    partition        : {self.partition}")
        print(f"    nodes            : {self.nodes}")
        print(f"    ntasks-per-node  : {self.ntasks_per_node}  →  total MPI ranks: {self.ntasks}")
        print(f"    wall time        : {self.slurm_time}")
        if self.account:
            print(f"    account          : {self.account}")
        if self.modules:
            print(f"    modules          : {', '.join(self.modules)}")
        print(f"    KPAR             : {kpar_note}")
        print(f"    NCORE            : {ncore_note}")

        self.poscar_file = os.path.abspath(poscar_file)
        self.inst_file   = os.path.abspath(instructions_file)
        self.generator   = VASPInputGenerator(self.poscar_file, inst, profile=self.profile)

        os.makedirs(self.project_dir, exist_ok=True)
        shutil.copy(self.inst_file,   self.project_dir)
        shutil.copy(self.poscar_file, os.path.join(self.project_dir, 'POSCAR'))

    def _default_ncore(self) -> int:
        """Return the nearest power-of-2 ≤ sqrt(ntasks_per_node), minimum 1."""
        import math
        s = int(math.floor(math.sqrt(self.ntasks_per_node)))
        # round down to nearest power of 2
        p = 1
        while p * 2 <= s:
            p *= 2
        return max(1, p)

    # ── SLURM script builders ─────────────────────────────────────────────

    def _sbatch_header(self, job_name: str, time_override: str = None) -> str:
        """Return the #SBATCH preamble for a batch script."""
        t = time_override or self.slurm_time
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --nodes={self.nodes}",
            f"#SBATCH --ntasks-per-node={self.ntasks_per_node}",
            f"#SBATCH --time={t}",
            f"#SBATCH --output={self.slurm_output}",
            f"#SBATCH --error={self.slurm_error}",
        ]
        if self.account:
            lines.append(f"#SBATCH --account {self.account}")
        lines.append("")
        if self.modules:
            lines.append("module purge")
            for mod in self.modules:
                lines.append(f"module load {mod}")
            lines.append("")
        return "\n".join(lines)

    def _vasp_run_line(self) -> str:
        """Return the line that actually launches VASP."""
        if self.mpi_cmd.startswith('srun'):
            return f"srun {self.vasp_exec}"
        else:
            # mpirun / mpiexec style
            return f"{self.mpi_cmd} {self.ntasks} {self.vasp_exec}"

    def _write_step_script(self, step_dir: str, job_name: str,
                           time_override: str = None):
        """Replace the VASPInputGenerator-produced run.sh with a SLURM batch script."""
        script = os.path.join(step_dir, 'run.sh')
        abs_dir = os.path.abspath(step_dir)
        with open(script, 'w') as f:
            f.write(self._sbatch_header(job_name, time_override))
            f.write(f"cd {abs_dir}\n")
            f.write(f"echo \"Working directory: $(pwd)\"\n\n")
            # Call pre-run copy scripts if they exist (copy POSCAR/CHGCAR from previous step)
            for copy_script in ('copy_from_relax.sh', 'copy_from_scf.sh'):
                if os.path.isfile(os.path.join(step_dir, copy_script)):
                    f.write(f'echo "Copying input files..."\n')
                    f.write(f'bash {abs_dir}/{copy_script}\n\n')
            f.write(f"{self._vasp_run_line()}\n")
            f.write('\necho "Exit status: $?"\n')
        chmod_x(script)

    # ── main workflow ─────────────────────────────────────────────────────

    def run(self):
        inst  = self.parser.instructions
        tasks = inst.get('tasks', [])
        pd    = self.project_dir
        proj  = slugify(self.project_label)

        # ── POTCAR ────────────────────────────────────────────────────────
        potcar_path = os.path.join(pd, 'POTCAR')
        if self.generator.elements:
            potcar_dir = check_potcar_dir()
            choices = {}
            choices_file = os.path.join(self.cwd, 'potcar_choices.json')
            if os.path.isfile(choices_file):
                try:
                    choices = json.loads(Path(choices_file).read_text())
                except Exception:
                    pass
            print(f"\nBuilding POTCAR ...")
            build_potcar(self.generator.elements, potcar_dir, potcar_path, choices)
        else:
            print("\nWARNING: could not read elements; POTCAR skipped.")

        # ── calculation steps ─────────────────────────────────────────────
        print("\nGenerating input files + SLURM scripts:")
        calc_dirs = {}

        if 'relax' in tasks:
            d = os.path.join(pd, '01_relax')
            self.generator.generate_relax_input(d)
            link_potcar(d, potcar_path)
            self._write_step_script(d, f"{proj}_relax")
            calc_dirs['relax'] = d
            print(f"  01_relax/    INCAR  KPOINTS  POSCAR  POTCAR  run.sh (SLURM)")

        if 'scf' in tasks:
            d = os.path.join(pd, '02_scf')
            self.generator.generate_scf_input(d, calc_dirs.get('relax'))
            link_potcar(d, potcar_path)
            self._write_step_script(d, f"{proj}_scf")
            calc_dirs['scf'] = d
            print(f"  02_scf/      INCAR  KPOINTS  POSCAR  POTCAR  run.sh (SLURM)")

        if 'bands' in tasks:
            d = os.path.join(pd, '03_bands')
            scf_d = calc_dirs.get('scf', os.path.join(pd, '02_scf'))
            self.generator.generate_bands_input(d, scf_d)
            link_potcar(d, potcar_path)
            self._write_step_script(d, f"{proj}_bands")
            calc_dirs['bands'] = d
            print(f"  03_bands/    INCAR  KPOINTS  POTCAR  run.sh (SLURM)")

        if 'dos' in tasks:
            d = os.path.join(pd, '04_dos')
            scf_d = calc_dirs.get('scf', os.path.join(pd, '02_scf'))
            self.generator.generate_dos_input(d, scf_d)
            link_potcar(d, potcar_path)
            self._write_step_script(d, f"{proj}_dos")
            calc_dirs['dos'] = d
            print(f"  04_dos/      INCAR  KPOINTS  POTCAR  run.sh (SLURM)")

        if 'wannier' in tasks:
            d = os.path.join(pd, '05_wannier')
            scf_d = calc_dirs.get('scf', os.path.join(pd, '02_scf'))
            self.generator.generate_wannier_input(d, scf_d)
            link_potcar(d, potcar_path)
            self._write_step_script(d, f"{proj}_wannier")
            calc_dirs['wannier'] = d
            print(f"  05_wannier/  INCAR  KPOINTS  wannier90.win  POTCAR  run.sh (SLURM)")

        if 'dfpt' in tasks:
            d = os.path.join(pd, '06_dfpt')
            scf_d = calc_dirs.get('scf', os.path.join(pd, '02_scf'))
            self.generator.generate_dfpt_input(d, scf_d)
            link_potcar(d, potcar_path)
            self._write_step_script(d, f"{proj}_dfpt")
            calc_dirs['dfpt'] = d
            print(f"  06_dfpt/     INCAR  KPOINTS  POTCAR  run.sh (SLURM)")

        if 'phonons' in tasks:
            d = os.path.join(pd, '07_phonons')
            scf_d  = calc_dirs.get('scf',  os.path.join(pd, '02_scf'))
            dfpt_d = calc_dirs.get('dfpt', None)
            self.generator.generate_phonons_input(d, scf_d, dfpt_d)
            link_potcar(d, potcar_path)
            self._write_step_script(d, f"{proj}_phonons")
            calc_dirs['phonons'] = d
            print(f"  07_phonons/  INCAR  KPOINTS  POTCAR  run.sh (SLURM)")

        # ── convergence tests ─────────────────────────────────────────────
        conv = inst.get('convergence', {})
        has_convergence = (conv.get('kpoints', {}).get('enabled') or
                           conv.get('encut',   {}).get('enabled'))
        if has_convergence:
            self._gen_convergence(conv, potcar_path)
            self._gen_convergence_analysis()
            print(f"  00_convergence/  run.sh (SLURM, loops internally)")
            print(f"  analyze_convergence.sh  (plot energy/pressure/forces after jobs finish)")

        # ── analysis scripts (same as workstation version) ────────────────
        self._gen_analysis(calc_dirs)
        print(f"  analyze.sh   (run locally after jobs complete)")

        # ── submission scripts ────────────────────────────────────────────
        if has_convergence:
            self._gen_submit_convergence()
            self._gen_submit_calculations(calc_dirs)
            print(f"  submit_convergence.sh   (STEP 1 — submit convergence jobs)")
            print(f"  submit_calculations.sh  (STEP 2 — submit production jobs)")
        else:
            self._gen_submit_all(calc_dirs)
            print(f"  submit_all.sh   (submit all jobs with dependency chaining)")

        # ── summary ───────────────────────────────────────────────────────
        proj_rel = os.path.relpath(self.project_dir, self.cwd)
        print(f"\n{'='*60}")
        print(f"  Done — {proj_rel}/")
        print(f"{'='*60}")

        if has_convergence:
            print(f"""
  Two-phase SLURM workflow:

  PHASE 1 — submit convergence tests:

    cd {proj_rel}
    ./submit_convergence.sh

  Wait for the convergence jobs to finish:
    squeue -u $USER

  Review results:
    00_convergence/encut/encut_convergence.dat
    00_convergence/kpoints/kpoint_convergence.dat

  PHASE 2 — submit production calculations:

    ./submit_calculations.sh
    (prompts for ENCUT and k-mesh, then submits chained jobs)

  After all jobs complete:
    ./analyze.sh
""")
        else:
            print(f"""
  Submit the full workflow:

    cd {proj_rel}
    ./submit_all.sh

  Monitor jobs:
    squeue -u $USER

  After completion:
    ./analyze.sh
""")

    # ── convergence SLURM scripts ─────────────────────────────────────────

    def _gen_convergence(self, conv, potcar_path):
        """Generate convergence directories + a single SLURM script that
        loops through all convergence points within one job allocation."""
        pd   = self.project_dir
        proj = slugify(self.project_label)

        kp = conv.get('kpoints', {})
        if kp.get('enabled'):
            d = os.path.join(pd, '00_convergence', 'kpoints')
            os.makedirs(d, exist_ok=True)
            link_potcar(d, potcar_path)
            shutil.copy(self.poscar_file, os.path.join(d, 'POSCAR'))

            incar_text = self.generator._generate_incar_scf()
            if 'ISIF' not in incar_text:
                incar_text += '\nISIF = 2\n'
            with open(os.path.join(d, 'INCAR'), 'w') as f:
                f.write(incar_text)

            script = os.path.join(d, 'run.sh')
            vasp   = self._vasp_run_line()
            with open(script, 'w') as f:
                f.write(self._sbatch_header(f"{proj}_kconv",
                                            time_override=self.slurm_time))
                f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')

                explicit = kp.get('meshes', [])
                if explicit:
                    f.write("# K-point convergence — explicit meshes\n\n")
                    for mesh in explicit:
                        nx, ny, nz = mesh
                        label = f"{nx}x{ny}x{nz}"
                        f.write(f'echo "  {label} ..."\n')
                        f.write(f'mkdir -p "{label}"\n')
                        f.write(f'cp INCAR POSCAR POTCAR "{label}/"\n')
                        f.write(f'printf "Automatic Gamma mesh\\n0\\nGamma\\n  {nx}  {ny}  {nz}\\n  0  0  0\\n" > "{label}/KPOINTS"\n')
                        f.write(f'cd "{label}"\n')
                        f.write(f'{vasp} > vasp.out 2>&1\n')
                        f.write(f'E=$(grep "energy  without" OUTCAR | tail -1 | awk \'{{print $7}}\')\n')
                        f.write(f'echo "{label}  $E" >> "$HERE/kpoint_convergence.dat"\n')
                        f.write(f'cd "$HERE"\n\n')
                else:
                    start, end = kp['range']
                    k0x, k0y, k0z = start
                    k1x, k1y, k1z = end
                    step = (k1x - k0x) // 4 if (k1x - k0x) >= 4 else 2
                    f.write(f"# K-point convergence: {k0x}x{k0y}x{k0z} → {k1x}x{k1y}x{k1z}\n\n")
                    f.write(f"for NX in $(seq {k0x} {step} {k1x}); do\n")
                    if k0z == k0x:
                        f.write('    NZ=$NX\n')
                    elif k0z > 0:
                        ratio = k0z / k0x
                        f.write(f'    NZ=$(echo "$NX * {ratio}" | bc | xargs printf "%.0f")\n')
                    else:
                        f.write('    NZ=1\n')
                    f.write('    LABEL="${NX}x${NX}x${NZ}"\n')
                    f.write('    mkdir -p "$LABEL"\n')
                    f.write('    cp INCAR POSCAR POTCAR "$LABEL/"\n')
                    f.write('    printf "Automatic Gamma mesh\\n0\\nGamma\\n  %d  %d  %d\\n  0  0  0\\n" $NX $NX $NZ > "$LABEL/KPOINTS"\n')
                    f.write('    cd "$LABEL"\n')
                    f.write(f'    echo "  $LABEL ..."\n')
                    f.write(f'    {vasp} > vasp.out 2>&1\n')
                    f.write("    E=$(grep \"energy  without\" OUTCAR | tail -1 | awk '{print $7}')\n")
                    f.write('    echo "$LABEL  $E" >> "$HERE/kpoint_convergence.dat"\n')
                    f.write('    cd "$HERE"\n')
                    f.write("done\n\n")

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
            incar_text = self.generator._generate_incar_scf()
            if 'ISIF' not in incar_text:
                incar_text += '\nISIF = 2\n'
            with open(os.path.join(d, 'INCAR'), 'w') as f:
                f.write(incar_text)

            script = os.path.join(d, 'run.sh')
            vasp   = self._vasp_run_line()
            with open(script, 'w') as f:
                f.write(self._sbatch_header(f"{proj}_econv",
                                            time_override=self.slurm_time))
                f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')
                f.write(f"# ENCUT convergence: {e0} → {e1} eV\n")
                f.write(f"for EC in $(seq {e0} 50 {e1}); do\n")
                f.write('    DIR="encut_${EC}"\n')
                f.write('    mkdir -p "$DIR"\n')
                f.write('    cp POSCAR POTCAR KPOINTS "$DIR/"\n')
                f.write('    sed "s/ENCUT.*/ENCUT = $EC/" INCAR > "$DIR/INCAR"\n')
                f.write('    cd "$DIR"\n')
                f.write('    echo "  ENCUT=$EC ..."\n')
                f.write(f'    {vasp} > vasp.out 2>&1\n')
                f.write("    E=$(grep \"energy  without\" OUTCAR | tail -1 | awk '{print $7}')\n")
                f.write('    echo "$EC  $E" >> "$HERE/encut_convergence.dat"\n')
                f.write('    cd "$HERE"\n')
                f.write("done\n\n")
                f.write('echo "Results → encut_convergence.dat"\n')
            chmod_x(script)

    # ── submission scripts ────────────────────────────────────────────────

    def _gen_submit_all(self, calc_dirs):
        """submit_all.sh — submit every production step with dependency chaining."""
        path = os.path.join(self.project_dir, 'submit_all.sh')
        ordered = [k for k in ['relax', 'scf', 'bands', 'dos', 'wannier', 'dfpt', 'phonons']
                   if k in calc_dirs]
        with open(path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Submit all VASP workflow steps for: {self.project_label}\n")
            f.write("# Jobs run sequentially via --dependency=afterok\n\n")
            f.write('set -e\n')
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')
            f.write('PREV=""\n\n')
            f.write('submit_step() {\n')
            f.write('    local label=$1 script=$2\n')
            f.write('    if [ -n "$PREV" ]; then\n')
            f.write('        JID=$(sbatch --parsable --dependency=afterok:$PREV "$script")\n')
            f.write('    else\n')
            f.write('        JID=$(sbatch --parsable "$script")\n')
            f.write('    fi\n')
            f.write('    echo "  Submitted $label — job $JID"\n')
            f.write('    PREV=$JID\n')
            f.write('}\n\n')
            f.write('echo "Submitting workflow for: ' + self.project_label + '"\n')
            for task in ordered:
                dirname = os.path.basename(calc_dirs[task])
                f.write(f'submit_step "{dirname}" "$HERE/{dirname}/run.sh"\n')
            f.write('\necho ""\n')
            f.write('echo "All jobs submitted. Monitor with:  squeue -u $USER"\n')
            f.write('echo "After completion run:  ./analyze.sh"\n')
        chmod_x(path)

    def _gen_submit_convergence(self):
        """submit_convergence.sh — submit convergence jobs (no dependency needed)."""
        path     = os.path.join(self.project_dir, 'submit_convergence.sh')
        conv_dir = os.path.join(self.project_dir, '00_convergence')
        has_encut   = os.path.isdir(os.path.join(conv_dir, 'encut'))
        has_kpoints = os.path.isdir(os.path.join(conv_dir, 'kpoints'))
        with open(path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# PHASE 1 — Submit convergence tests for: {self.project_label}\n")
            f.write("# Each convergence test runs as a single SLURM job that\n")
            f.write("# loops through all parameter values internally.\n\n")
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')
            if has_encut:
                f.write('JID_EC=$(sbatch --parsable "$HERE/00_convergence/encut/run.sh")\n')
                f.write('echo "  Submitted ENCUT convergence — job $JID_EC"\n')
            if has_kpoints:
                f.write('JID_KP=$(sbatch --parsable "$HERE/00_convergence/kpoints/run.sh")\n')
                f.write('echo "  Submitted k-point convergence — job $JID_KP"\n')
            f.write('\necho ""\n')
            f.write('echo "Monitor with:  squeue -u $USER"\n')
            f.write('echo ""\n')
            f.write('echo "After jobs finish, review:"\n')
            if has_encut:
                f.write('echo "  00_convergence/encut/encut_convergence.dat"\n')
            if has_kpoints:
                f.write('echo "  00_convergence/kpoints/kpoint_convergence.dat"\n')
            f.write('echo ""\n')
            f.write('echo "Then run:  ./submit_calculations.sh"\n')
        chmod_x(path)

    def _gen_submit_calculations(self, calc_dirs):
        """submit_calculations.sh — PHASE 2: prompt for ENCUT/k-mesh, patch
        input files, then submit production jobs with dependency chaining."""
        path = os.path.join(self.project_dir, 'submit_calculations.sh')
        ordered = [k for k in ['relax', 'scf', 'bands', 'dos', 'wannier', 'dfpt', 'phonons']
                   if k in calc_dirs]
        with open(path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# PHASE 2 — Submit production calculations for: {self.project_label}\n")
            f.write("# Prompts for converged ENCUT/k-mesh, patches all input files,\n")
            f.write("# then submits jobs with dependency chaining.\n\n")
            f.write("set -e\n")
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')

            # prompt
            f.write('echo ""\necho "=== Convergence parameters ==="\n')
            f.write('echo "Review results first:"\n')
            f.write('echo "  cat 00_convergence/encut/encut_convergence.dat"\n')
            f.write('echo "  cat 00_convergence/kpoints/kpoint_convergence.dat"\n')
            f.write('echo ""\n')
            f.write('read -p "Enter converged ENCUT (eV) [e.g. 520]: " ENCUT\n')
            f.write('[ -z "$ENCUT" ] && echo "ERROR: ENCUT cannot be empty." && exit 1\n\n')
            f.write('read -p "Enter k-mesh for relax/SCF [e.g. 12x12x6]: " KMESH\n')
            f.write('[ -z "$KMESH" ] && echo "ERROR: k-mesh cannot be empty." && exit 1\n\n')
            f.write('read -p "Enter denser k-mesh for DOS [Enter = same as SCF]: " KMESH_DOS\n')
            f.write('[ -z "$KMESH_DOS" ] && KMESH_DOS=$KMESH\n\n')

            # parse meshes
            for suffix, var in [('', 'KMESH'), ('_DOS', 'KMESH_DOS')]:
                f.write(f'NX{suffix}=$(echo "${var}" | cut -dx -f1)\n')
                f.write(f'NY{suffix}=$(echo "${var}" | cut -dx -f2)\n')
                f.write(f'NZ{suffix}=$(echo "${var}" | cut -dx -f3)\n')
                f.write(f'[ -z "$NZ{suffix}" ] && NZ{suffix}=$NY{suffix}\n')
            f.write('\n')

            # patch INCAR
            f.write('echo "Patching input files:"\n')
            for task in ordered:
                dirname = os.path.basename(calc_dirs[task])
                f.write(f'sed -i.bak "s/^ENCUT.*/ENCUT = $ENCUT/" "$HERE/{dirname}/INCAR"\n')
                f.write(f'echo "  {dirname}/INCAR  → ENCUT=$ENCUT"\n')
            f.write('\n')

            # patch KPOINTS
            for task in ordered:
                if task in ('bands', 'dos'):
                    continue
                dirname = os.path.basename(calc_dirs[task])
                f.write(f'printf "Automatic Gamma mesh\\n0\\nGamma\\n  %d  %d  %d\\n  0  0  0\\n" ')
                f.write(f'$NX $NY $NZ > "$HERE/{dirname}/KPOINTS"\n')
                f.write(f'echo "  {dirname}/KPOINTS → $KMESH"\n')
            if 'dos' in calc_dirs:
                dirname = os.path.basename(calc_dirs['dos'])
                f.write(f'printf "Automatic Gamma mesh\\n0\\nGamma\\n  %d  %d  %d\\n  0  0  0\\n" ')
                f.write(f'$NX_DOS $NY_DOS $NZ_DOS > "$HERE/{dirname}/KPOINTS"\n')
                f.write(f'echo "  {dirname}/KPOINTS → $KMESH_DOS (DOS)"\n')
            f.write('\n')

            # submit with dependency chaining
            f.write('echo ""\necho "Submitting jobs:"\n')
            f.write('PREV=""\n')
            f.write('submit_step() {\n')
            f.write('    local label=$1 script=$2\n')
            f.write('    if [ -n "$PREV" ]; then\n')
            f.write('        JID=$(sbatch --parsable --dependency=afterok:$PREV "$script")\n')
            f.write('    else\n')
            f.write('        JID=$(sbatch --parsable "$script")\n')
            f.write('    fi\n')
            f.write('    echo "  $label — job $JID"\n')
            f.write('    PREV=$JID\n')
            f.write('}\n\n')
            for task in ordered:
                dirname = os.path.basename(calc_dirs[task])
                f.write(f'submit_step "{dirname}" "$HERE/{dirname}/run.sh"\n')
            f.write('\necho ""\n')
            f.write('echo "All jobs submitted. Monitor with:  squeue -u $USER"\n')
            f.write('echo "After completion run:  ./analyze.sh"\n')
        chmod_x(path)

    # ── convergence analysis ──────────────────────────────────────────────

    def _gen_convergence_analysis(self):
        """Generate 00_convergence/plot_convergence.py and analyze_convergence.sh."""
        pd       = self.project_dir
        conv_dir = os.path.join(pd, '00_convergence')
        modules_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules')

        py = os.path.join(conv_dir, 'plot_convergence.py')
        with open(py, 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write(f'"""\nConvergence plot generator for: {self.project_label}\n\n')
            f.write('Usage:\n')
            f.write('  python3 plot_convergence.py          # plot both encut and kpoints\n')
            f.write('  python3 plot_convergence.py encut    # ENCUT convergence only\n')
            f.write('  python3 plot_convergence.py kpoints  # k-point convergence only\n\n')
            f.write('Plots are saved as PNG files in the 00_convergence/ directory.\n"""\n\n')
            f.write('import sys, os, re\n')
            f.write('import matplotlib\n')
            f.write('matplotlib.use("Agg")\n')
            f.write('import matplotlib.pyplot as plt\n')
            f.write('from matplotlib.ticker import ScalarFormatter\n\n')
            f.write(f'sys.path.insert(0, {repr(modules_path)})\n')
            f.write('from outcar_parser import (parse_energy, parse_pressure_diagonal,\n')
            f.write('                            parse_forces_first_atom)\n\n')
            f.write('HERE = os.path.dirname(os.path.abspath(__file__))\n\n')
            f.write('nan = float("nan")\n\n')
            f.write('def _read(path):\n')
            f.write('    try:\n')
            f.write('        with open(path, errors="replace") as fh: return fh.read()\n')
            f.write('    except OSError: return ""\n\n')
            f.write('def _fmt_ax(ax):\n')
            f.write('    sc = ScalarFormatter(useOffset=False, useMathText=False)\n')
            f.write('    sc.set_scientific(False)\n')
            f.write('    ax.yaxis.set_major_formatter(sc)\n\n')
            f.write('def plot_convergence(dtype):\n')
            f.write('    conv_dir = os.path.join(HERE, dtype)\n')
            f.write('    if not os.path.isdir(conv_dir):\n')
            f.write('        print(f"  Skipping {dtype}: directory not found."); return\n\n')
            f.write('    entries = []\n')
            f.write('    for name in os.listdir(conv_dir):\n')
            f.write('        full = os.path.join(conv_dir, name)\n')
            f.write('        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "OUTCAR")):\n')
            f.write('            entries.append((name, full))\n')
            f.write('    if not entries:\n')
            f.write('        print(f"  No OUTCARs found in {conv_dir}/*/"); return\n\n')
            f.write('    if dtype == "encut":\n')
            f.write('        entries.sort(key=lambda x: int(re.search(r"\\d+", x[0]).group()))\n')
            f.write('        labels  = [re.search(r"\\d+", n).group() for n, _ in entries]\n')
            f.write('        xlabel  = "ENCUT (eV)"\n')
            f.write('    else:\n')
            f.write('        entries.sort(key=lambda x: [int(v) for v in re.findall(r"\\d+", x[0])])\n')
            f.write('        labels  = [n.replace("x", "\u00d7") for n, _ in entries]\n')
            f.write('        xlabel  = "K-mesh"\n\n')
            f.write('    xs = list(range(len(labels)))\n')
            f.write('    rot = 30 if len(labels) > 4 else 0\n\n')
            f.write('    energies, pxx, pyy, pzz, fx, fy, fz = [], [], [], [], [], [], []\n')
            f.write('    for _, path in entries:\n')
            f.write('        text = _read(os.path.join(path, "OUTCAR"))\n')
            f.write('        e = parse_energy(text)\n')
            f.write('        energies.append(e if e is not None else nan)\n')
            f.write('        p = parse_pressure_diagonal(text)\n')
            f.write('        pxx.append(p[0] if p else nan)\n')
            f.write('        pyy.append(p[1] if p else nan)\n')
            f.write('        pzz.append(p[2] if p else nan)\n')
            f.write('        fo = parse_forces_first_atom(text)\n')
            f.write('        fx.append(fo[0] if fo else nan)\n')
            f.write('        fy.append(fo[1] if fo else nan)\n')
            f.write('        fz.append(fo[2] if fo else nan)\n\n')
            f.write('    prefix = os.path.join(HERE, dtype)\n')
            f.write('    title_x = xlabel.split()[0]\n\n')
            f.write('    fig, ax = plt.subplots(figsize=(6, 4))\n')
            f.write('    ax.plot(xs, energies, "o-", color="#7c3aed", lw=1.5, ms=5)\n')
            f.write('    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=9)\n')
            f.write('    ax.set_xlabel(xlabel); ax.set_ylabel("Total energy (eV)")\n')
            f.write('    ax.set_title(f"Total energy vs {title_x}"); _fmt_ax(ax); ax.grid(True, alpha=0.3)\n')
            f.write('    plt.tight_layout(); fig.savefig(prefix + "_energy.png", dpi=150); plt.close(fig)\n')
            f.write('    print(f"  Saved: {dtype}_energy.png")\n\n')
            f.write('    fig, ax = plt.subplots(figsize=(6, 4))\n')
            f.write('    ax.plot(xs, pxx, "o-", color="#3b82f6", lw=1.5, ms=4, label="Pxx")\n')
            f.write('    ax.plot(xs, pyy, "s-", color="#10b981", lw=1.5, ms=4, label="Pyy")\n')
            f.write('    ax.plot(xs, pzz, "^-", color="#f59e0b", lw=1.5, ms=4, label="Pzz")\n')
            f.write('    ax.legend(); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=9)\n')
            f.write('    ax.set_xlabel(xlabel); ax.set_ylabel("Pressure (kBar)")\n')
            f.write('    ax.set_title(f"Pressure diagonal vs {title_x}"); _fmt_ax(ax); ax.grid(True, alpha=0.3)\n')
            f.write('    plt.tight_layout(); fig.savefig(prefix + "_pressure.png", dpi=150); plt.close(fig)\n')
            f.write('    print(f"  Saved: {dtype}_pressure.png")\n\n')
            f.write('    fig, ax = plt.subplots(figsize=(6, 4))\n')
            f.write('    ax.plot(xs, fx, "o-", color="#ef4444", lw=1.5, ms=4, label="Fx")\n')
            f.write('    ax.plot(xs, fy, "s-", color="#10b981", lw=1.5, ms=4, label="Fy")\n')
            f.write('    ax.plot(xs, fz, "^-", color="#3b82f6", lw=1.5, ms=4, label="Fz")\n')
            f.write('    ax.legend(); ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=9)\n')
            f.write('    ax.set_xlabel(xlabel); ax.set_ylabel("Force on atom 1 (eV/\u00c5)")\n')
            f.write('    ax.set_title(f"Forces (atom 1) vs {title_x}"); _fmt_ax(ax); ax.grid(True, alpha=0.3)\n')
            f.write('    plt.tight_layout(); fig.savefig(prefix + "_forces.png", dpi=150); plt.close(fig)\n')
            f.write('    print(f"  Saved: {dtype}_forces.png")\n\n')
            f.write('if __name__ == "__main__":\n')
            f.write('    dtype = sys.argv[1] if len(sys.argv) > 1 else None\n')
            f.write('    if dtype in ("encut", "kpoints"):\n')
            f.write('        print(f"\\nPlotting {dtype} convergence...")\n')
            f.write('        plot_convergence(dtype)\n')
            f.write('    else:\n')
            f.write('        for dt in ("encut", "kpoints"):\n')
            f.write('            print(f"\\nPlotting {dt} convergence...")\n')
            f.write('            plot_convergence(dt)\n')
            f.write('    print("\\nDone. PNG files saved in 00_convergence/")\n')
        chmod_x(py)

        sh = os.path.join(pd, 'analyze_convergence.sh')
        conv_dir_rel = os.path.relpath(conv_dir, pd)
        with open(sh, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'# Plot convergence results for: {self.project_label}\n')
            f.write('# Run this locally after SLURM convergence jobs complete.\n')
            f.write('# Generates PNG plots in 00_convergence/:\n')
            f.write('#   encut_energy.png    encut_pressure.png    encut_forces.png\n')
            f.write('#   kpoints_energy.png  kpoints_pressure.png  kpoints_forces.png\n\n')
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n\n')
            f.write('if ! command -v python3 &>/dev/null; then\n')
            f.write('    echo "ERROR: python3 not found."; exit 1\n')
            f.write('fi\n')
            f.write('if ! python3 -c "import matplotlib" 2>/dev/null; then\n')
            f.write('    echo "ERROR: matplotlib not installed.  Run: pip install matplotlib"; exit 1\n')
            f.write('fi\n\n')
            f.write(f'python3 "$HERE/{conv_dir_rel}/plot_convergence.py" "$@"\n')
        chmod_x(sh)

    # ── cumulative DOS script ────────────────────────────────────────────────
    def _gen_cumulative_dos_script(self, dos_dir):
        """Generate 04_dos/plot_cumulative_dos.py — stacked-area cumulative DOS."""
        slug  = slugify(self.project_label)
        label = self.project_label
        py = os.path.join(dos_dir, 'plot_cumulative_dos.py')
        lines = [
            '#!/usr/bin/env python3',
            f'"""Cumulative DOS plot for: {label}',
            '',
            'Each filled band = contribution of one element-orbital pair.',
            'Top of each band = running sum up to that contribution.',
            'Top of the topmost band = total DOS.',
            '',
            'Requires: pymatgen  (pip install pymatgen)',
            f'Output:   analysis/{slug}_cumulative_dos.png',
            '"""',
            'import sys, os',
            'import matplotlib',
            'matplotlib.use("Agg")',
            'import matplotlib.pyplot as plt',
            'import numpy as np',
            '',
            'HERE = os.path.dirname(os.path.abspath(__file__))',
            'ANA  = os.path.normpath(os.path.join(HERE, "..", "analysis"))',
            'os.makedirs(ANA, exist_ok=True)',
            '',
            'try:',
            '    from pymatgen.io.vasp import Vasprun',
            '    from pymatgen.electronic_structure.core import OrbitalType, Spin',
            'except ImportError:',
            '    print("ERROR: pymatgen not installed.  Run: pip install pymatgen")',
            '    sys.exit(1)',
            '',
            'vxml = os.path.join(HERE, "vasprun.xml")',
            'if not os.path.isfile(vxml):',
            '    print(f"ERROR: {vxml} not found"); sys.exit(1)',
            '',
            'print("Reading vasprun.xml ...")',
            'vr   = Vasprun(vxml, parse_projected_eigen=False)',
            'cdos = vr.complete_dos',
            'ef   = cdos.efermi',
            'energies = cdos.energies - ef',
            '',
            'EMIN, EMAX = -6.0, 6.0',
            'mask = (energies >= EMIN) & (energies <= EMAX)',
            'en   = energies[mask]',
            '',
            'has_both = Spin.down in cdos.densities',
            'elements = list(dict.fromkeys(s.specie.symbol for s in cdos.structure))',
            'ORB_TYPES = [OrbitalType.s, OrbitalType.p, OrbitalType.d, OrbitalType.f]',
            'ORB_NAMES = ["s", "p", "d", "f"]',
            '',
            'contributions = []',
            'for el_sym in elements:',
            '    try:',
            '        spd = cdos.get_element_spd_dos(el_sym)',
            '    except Exception:',
            '        continue',
            '    for orb, oname in zip(ORB_TYPES, ORB_NAMES):',
            '        if orb not in spd:',
            '            continue',
            '        d      = spd[orb]',
            '        dos_up   = d.densities.get(Spin.up,   np.zeros(len(energies)))[mask]',
            '        dos_down = d.densities.get(Spin.down,  np.zeros(len(energies)))[mask]',
            '        total  = dos_up + dos_down if has_both else dos_up',
            '        if total.max() < 0.01:',
            '            continue',
            '        contributions.append((f"{el_sym}-{oname}", total))',
            '',
            'if not contributions:',
            '    print("No significant orbital contributions found."); sys.exit(0)',
            '',
            'colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]',
            'fig, ax = plt.subplots(figsize=(7, 5))',
            'cumulative = np.zeros(len(en))',
            'for i, (label, dos) in enumerate(contributions):',
            '    prev       = cumulative.copy()',
            '    cumulative = cumulative + dos',
            '    color = colors[i % len(colors)]',
            '    ax.fill_between(en, prev, cumulative, alpha=0.35, color=color, label=label)',
            '    ax.plot(en, cumulative, color=color, lw=1.0)',
            '',
            'ax.axvline(0, color="k", ls="--", lw=0.8)',
            'ax.set_xlim(EMIN, EMAX)',
            'ax.set_ylim(bottom=0)',
            'ax.set_xlabel("Energy − $E_F$ (eV)", fontsize=12)',
            'ax.set_ylabel("Cumulative DOS (states/eV)", fontsize=12)',
            f'ax.set_title("Cumulative DOS — {label}")',
            'ax.legend(fontsize=8, ncol=2, loc="upper left")',
            'ax.grid(True, alpha=0.2)',
            'plt.tight_layout()',
            '',
            f'out = os.path.join(ANA, "{slug}_cumulative_dos.png")',
            'fig.savefig(out, dpi=150)',
            'plt.close(fig)',
            'print(f"  Saved: {out}")',
        ]
        with open(py, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        chmod_x(py)

    # ── analysis (identical to workstation version) ───────────────────────

    def _gen_analysis(self, calc_dirs):
        """Generate analyze.sh and analysis/plot_results.py.
        These run locally after the cluster jobs complete."""
        pd  = self.project_dir
        ana = os.path.join(pd, 'analysis')
        os.makedirs(ana, exist_ok=True)

        has_relax = 'relax' in calc_dirs
        has_scf   = 'scf'   in calc_dirs
        has_bands = 'bands' in calc_dirs
        has_dos   = 'dos'   in calc_dirs

        dos_proj = self.parser.instructions.get('dos_projections', [])
        from collections import defaultdict
        orb_map = defaultdict(list)
        for dp in dos_proj:
            el, orb = dp.split(':') if ':' in dp else (dp, 's')
            orb_map[el].append(orb)
        sumo_orb = ', '.join(f"{el} {' '.join(orbs)}" for el, orbs in orb_map.items())

        sh = os.path.join(pd, 'analyze.sh')
        with open(sh, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Post-processing for: {self.project_label}\n")
            f.write("# Run this locally after all SLURM jobs complete.\n\n")
            f.write('HERE="$(cd "$(dirname "$0")" && pwd)"\n')
            f.write('mkdir -p "$HERE/analysis"\n\n')

            if has_relax:
                f.write('echo "=== Relaxation ==="\n')
                f.write('grep "energy  without" "$HERE/01_relax/OUTCAR" | tail -1\n')
                f.write('grep "reached required accuracy" "$HERE/01_relax/OUTCAR" '
                        '&& echo "  Converged" || echo "  WARNING: check NSW"\n\n')
            if has_scf:
                f.write('echo "=== SCF ==="\n')
                f.write('grep "E-fermi" "$HERE/02_scf/OUTCAR" | tail -1\n\n')

            if has_bands:
                f.write('echo "=== Band structure (sumo) ==="\n')
                f.write('if command -v sumo-bandplot &>/dev/null; then\n')
                f.write('    EFERMI=""\n')
                f.write('    if [ -f "$HERE/04_dos/OUTCAR" ]; then\n')
                f.write('        EFERMI=$(grep "E-fermi" "$HERE/04_dos/OUTCAR" | tail -1 | awk \'{print $3}\')\n')
                f.write('    elif [ -f "$HERE/02_scf/OUTCAR" ]; then\n')
                f.write('        EFERMI=$(grep "E-fermi" "$HERE/02_scf/OUTCAR" | tail -1 | awk \'{print $3}\')\n')
                f.write('    fi\n')
                f.write('    if [ -n "$EFERMI" ] && [ -f "$HERE/03_bands/vasprun.xml" ]; then\n')
                f.write('        sed -i.efermi_bak \'s|<i name="efermi">.*</i>|<i name="efermi">  \'$EFERMI\'  </i>|\' "$HERE/03_bands/vasprun.xml"\n')
                f.write('    fi\n')
                f.write('    cd "$HERE/03_bands"\n')
                f.write('    sumo-bandplot --prefix "$(basename $HERE)" --ymin -4 --ymax 4 2>&1\n')
                f.write('    mv *_band.* "$HERE/analysis/" 2>/dev/null || true\n')
                f.write('    cd "$HERE"\n')
                f.write('    echo "  Saved: analysis/*_band.*"\n')
                f.write('else\n')
                f.write('    echo "  sumo not found — copying EIGENVAL"\n')
                f.write('    cp "$HERE/03_bands/EIGENVAL" "$HERE/analysis/"\n')
                f.write('fi\n\n')

            if has_dos:
                f.write('echo "=== DOS (sumo) ==="\n')
                f.write('if command -v sumo-dosplot &>/dev/null; then\n')
                f.write('    cd "$HERE/04_dos"\n')
                f.write('    sumo-dosplot --prefix "$(basename $HERE)_total" --xmin -6 --xmax 6 2>&1\n')
                if sumo_orb:
                    f.write(f'    sumo-dosplot --prefix "$(basename $HERE)_proj" --orbitals "{sumo_orb}" --xmin -6 --xmax 6 2>&1\n')
                f.write('    mv *_dos.* "$HERE/analysis/" 2>/dev/null || true\n')
                f.write('    cd "$HERE"\n')
                f.write('    echo "  Saved: analysis/*_dos.*"\n')
                f.write('else\n')
                f.write('    echo "  sumo not found — copying DOSCAR"\n')
                f.write('    cp "$HERE/04_dos/DOSCAR" "$HERE/analysis/"\n')
                f.write('fi\n\n')
                # Cumulative DOS
                f.write('echo "=== Cumulative DOS ==="\n')
                f.write('if python3 -c "import pymatgen" 2>/dev/null; then\n')
                f.write('    python3 "$HERE/04_dos/plot_cumulative_dos.py"\n')
                f.write('else\n')
                f.write('    echo "  pymatgen not found — skipping cumulative DOS (pip install pymatgen)"\n')
                f.write('fi\n\n')

            f.write('echo "Done. Results in analysis/"\n')
        chmod_x(sh)

        if has_dos:
            self._gen_cumulative_dos_script(calc_dirs['dos'])


# ── entry point ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description='VASP Workflow Agent — SLURM HPC edition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Place instructions.txt and POSCAR in the same directory, then run:

    ./vasp-agent-slurm.py --profile slurm

Output goes to  <ProjectName>/  containing SLURM batch scripts.
Submit with:  cd <ProjectName> && ./submit_all.sh

SLURM settings are read from profiles/slurm.json.
        """)
    ap.add_argument('-i', '--instructions', default='instructions.txt')
    ap.add_argument('-s', '--poscar',       default='POSCAR')
    ap.add_argument('-p', '--profile',      default='slurm',
                    help='Profile name (default: slurm → loads profiles/slurm.json)')
    args = ap.parse_args()

    for fpath, label in [(args.instructions, 'instructions'), (args.poscar, 'POSCAR')]:
        if not os.path.isfile(fpath):
            print(f"ERROR: {label} file not found: {os.path.abspath(fpath)}")
            sys.exit(1)

    profile = load_profile(args.profile)
    if not profile.get('slurm'):
        print("WARNING: no 'slurm' block found in profile — using built-in defaults.")
        print("         Edit profiles/slurm.json to set partition, nodes, time, etc.\n")

    try:
        SLURMVASPAgent(args.instructions, args.poscar, profile=profile).run()
    except SystemExit:
        raise
    except Exception:
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
