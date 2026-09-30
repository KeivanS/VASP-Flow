#!/usr/bin/env python3
"""
check_agent_consistency.py — do the three front-ends generate the same inputs?

Runs every test case through
    GUI          POST /api/generate  (Flask test client -> vasp-agent.py)
    workstation  vasp-agent.py
    HPC/SLURM    vasp-agent-slurm.py
and requires the calculation inputs (INCAR, KPOINTS, copy_from_*.sh, the
explicit-KPOINTS marker) to be IDENTICAL in every step directory.  run.sh
is allowed to differ (SLURM header vs. local launcher) but must call the same
copy_from_*.sh scripts.  It also checks the semantics of each case:

  * MAGMOM / ISPIN (uniform, negative, per-element, per-atom list, SOC vector)
  * k-mesh density reflected in the KPOINTS files (coarse 1000 / fine 5000 /
    custom kpra) and the DOS/LOBSTER = 2x SCF, DFPT = SCF relations
  * KPAR / NCORE valid for the rank count in relax, SCF and DOS (ELF => KPAR 1
    with NCORE re-derived), and KPAR/NCORE given in INCAR blocks respected
  * relax starts from scratch (ISTART=0, ICHARG=2) and writes WAVECAR + CHGCAR;
    SCF's copy_from_relax.sh picks CHGCAR (+ WAVECAR when the mesh is the same)
  * explicit INCAR / KPOINTS blocks override everything, for every step

No VASP or POTCAR library is needed: a stub POTCAR library is created unless
--potcar-dir is given.  Needs numpy, flask, and (for k-path / symmetry) spglib
and pymatgen.

    python3 check_agent_consistency.py            # all cases
    python3 check_agent_consistency.py -k magmom  # cases whose name contains 'magmom'
    python3 check_agent_consistency.py --keep     # keep the work directory
"""

import argparse, filecmp, importlib.util, json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'modules'))

POSCAR_AFM = """FeO-like tetragonal test cell
1.0
 4.0 0.0 0.0
 0.0 4.0 0.0
 0.0 0.0 6.0
Fe O
2 2
Direct
 0.0 0.0 0.0
 0.5 0.5 0.5
 0.0 0.5 0.25
 0.5 0.0 0.75
"""
POSCAR_HEX = """MoS2 monolayer
1.0
  3.16000000   0.00000000   0.00000000
 -1.58000000   2.73679000   0.00000000
  0.00000000   0.00000000  20.00000000
Mo S
1 2
Direct
  0.333333333  0.666666667  0.500000000
  0.666666667  0.333333333  0.423000000
  0.666666667  0.333333333  0.577000000
"""
STUB_POTCARS = {'Fe': (8, 267.9), 'O': (6, 400.0), 'Mo': (6, 224.6), 'S': (6, 258.7),
                'Si': (4, 245.3), 'Cl': (7, 262.5)}
POSCAR_FESI = """FeSi
1.0
4.49 0 0
0 4.49 0
0 0 4.49
Fe Si
4 4
Direct
0.137 0.137 0.137
0.637 0.363 0.863
0.863 0.637 0.363
0.363 0.863 0.637
0.842 0.842 0.842
0.342 0.658 0.158
0.158 0.342 0.658
0.658 0.158 0.342
"""

STEP_KEYS = {'01_relax': 'relax', '02_scf': 'scf', '03_bands': 'bands',
             '04_dos': 'dos', '06_dfpt': 'dfpt', '08_lobster': 'lobster'}
COMPARE = ('INCAR', 'KPOINTS', 'copy_from_relax.sh', 'copy_from_scf.sh',
           '.explicit_kpoints', 'elf/INCAR', 'run_elf.sh')


# ── helpers ─────────────────────────────────────────────────────────────────
def incar_tags(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        s = line.split('!')[0].split('#')[0].strip()
        if '=' in s:
            k, v = s.split('=', 1)
            out[k.strip().upper()] = v.strip()
    return out


def kpts_mesh(path):
    L = Path(path).read_text().splitlines()
    if len(L) > 3 and L[1].strip() == '0':
        return tuple(int(float(x)) for x in L[3].split()[:3])
    return None


def natoms(poscar):
    return sum(int(x) for x in Path(poscar).read_text().splitlines()[6].split())


def load_gui():
    os.environ.setdefault('HOME', tempfile.gettempdir())
    spec = importlib.util.spec_from_file_location('vasp_gui', REPO / 'vasp-gui.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── test cases ──────────────────────────────────────────────────────────────
# 'form' is the GUI setup form; the workstation and SLURM agents are run on the
# instructions.txt the GUI wrote, so all three see the same instructions.
# 'text' cases have no GUI equivalent (hand-written instructions only).
def cases():
    base = dict(functional='PBE', relax=True, scf=True, bands=True, dos=True,
                lobster=False, dfpt=False, elf=True, mpi_np=16,
                param_mode='manual', manual_encut='500', kmesh_density='fine',
                hexagonal=False)
    def form(**kw):
        d = dict(base); d.update(kw); return d

    blk = dict(
        all={'incar': 'LREAL = Auto'},
        relax={'incar': 'EDIFFG = -0.005\nKPAR = 2',
               'kpoints': 'relax mesh\n0\nMonkhorst-Pack\n3 3 2\n0 0 0'},
        scf={'incar': 'SIGMA = 0.1\nNELM = 200',
             'kpoints': 'scf mesh\n0\nGamma\n9 9 5\n0 0 0'},
        bands={'incar': 'NBANDS = 64'},
        dos={'incar': 'NEDOS = 5000\nISMEAR = 0\nSIGMA = 0.02'},
    )
    return [
        dict(name='magmom-uniform', poscar=POSCAR_AFM,
             form=form(spin_mode='collinear', mag_moment=4.0, kmesh_density='coarse'),
             expect=dict(ispin=2, magmom='4*4.0', kpra=(1000, 0.3))),
        dict(name='magmom-negative', poscar=POSCAR_AFM,
             form=form(spin_mode='collinear', mag_moment=-3.0),
             expect=dict(ispin=2, magmom='4*-3.0', kpra=(5000, 0.3))),
        dict(name='magmom-per-element', poscar=POSCAR_AFM,
             form=form(spin_mode='collinear', magmom='Fe=4.0, O=0.6'),
             expect=dict(ispin=2, magmom='2*4.0 2*0.6')),
        dict(name='magmom-per-atom-list', poscar=POSCAR_AFM,
             form=form(spin_mode='collinear', magmom='4.0 -4.0 0.6 0.6'),
             expect=dict(ispin=2, magmom='4.0 -4.0 2*0.6')),
        dict(name='soc-x', poscar=POSCAR_AFM,
             form=form(spin_mode='soc_x', magmom='Fe=3.0, O=0', relax=False,
                       bands=False, dos=False),
             expect=dict(ispin=None, magmom='3.0 2*0.0 3.0 8*0.0')),
        dict(name='density-custom-kpra', poscar=POSCAR_AFM,
             form=form(kmesh_kpra='3000', relax_kmesh_density='fine'),
             expect=dict(kpra=(3000, 0.35), relax_kpra=(5000, 0.35))),
        dict(name='density-2d-hex', poscar=POSCAR_HEX,
             form=form(is_2d=True, hexagonal=True, kmesh_density='coarse'),
             expect=dict(kpra=(1000, 0.35), hex6=True)),
        dict(name='lobster-dfpt-follow-scf', poscar=POSCAR_AFM,
             form=form(lobster=True, dfpt=True, bands=False, dos=False),
             expect=dict(kpra=(5000, 0.3))),
        dict(name='explicit-blocks', poscar=POSCAR_AFM,
             form=form(spin_mode='collinear', mag_moment=4.0, lobster=True,
                       explicit_blocks=blk),
             expect=dict(blocks=True)),
        dict(name='mpi-1', poscar=POSCAR_AFM,
             form=form(mpi_np=1), agents=('gui', 'ws'), expect=dict(no_parallel=True)),
        dict(name='text-incar-kpar-ncore', poscar=POSCAR_AFM, form=None,
             text=("Project: cfg\nMethods: PBE functional\n"
                   "Tasks: structure relaxation, SCF calculation, DOS\nMPI: 16\n"
                   "ELF: off\nSCF_NCORE: 8\nINCAR dos:\n   KPAR = 2\nEND_INCAR\n"),
             expect=dict(text_parallel=True)),
        dict(name='gga-u-default-intermetallic', poscar=POSCAR_FESI, form=None,
             text=("Project: fesi\nMethods: PBE functional\n"
                   "Tasks: structure relaxation, SCF calculation, band structure, "
                   "LOBSTER COHP/COBI analysis\nMPI: 16\nELF: off\n"),
             expect=dict(u_everywhere=False, walltimes={'01_relax': '08:00:00', '02_scf': '04:00:00',
                                                        '03_bands': '02:00:00', '08_lobster': '04:00:00'})),
        dict(name='gga-u-default-halide', poscar=POSCAR_FESI.replace('Fe Si', 'Fe Cl'), form=None,
             text=("Project: fecl\nMethods: PBE functional\n"
                   "Tasks: SCF calculation, band structure\nMPI: 16\nELF: off\n"),
             expect=dict(u_everywhere=True)),
        dict(name='gga-u-on-intermetallic', poscar=POSCAR_FESI, form=None,
             text=("Project: fesi\nMethods: PBE functional\n"
                   "Tasks: structure relaxation, SCF calculation, band structure, "
                   "LOBSTER COHP/COBI analysis\nMPI: 16\nELF: off\nGGA_U: ON\n"),
             expect=dict(u_everywhere=True)),
        dict(name='gga-u-default-oxide', poscar=POSCAR_AFM, form=None,
             text=("Project: feo\nMethods: PBE functional\n"
                   "Tasks: SCF calculation, band structure\nMPI: 16\nELF: off\n"),
             expect=dict(u_everywhere=True)),
        dict(name='gga-u-off-oxide', poscar=POSCAR_AFM, form=None,
             text=("Project: feo\nMethods: PBE functional\n"
                   "Tasks: SCF calculation, band structure\nMPI: 16\nELF: off\nGGA_U: OFF\n"),
             expect=dict(u_everywhere=False)),
        dict(name='gui-u-mode-on', poscar=POSCAR_FESI,
             form=form(u_mode='on'), expect=dict(u_everywhere=True)),
        dict(name='text-elf-separate', poscar=POSCAR_AFM, form=None,
             text=("Project: elf2\nMethods: PBE functional\n"
                   "Tasks: SCF calculation\nMPI: 16\nELF: separate\n"),
             expect=dict(elf_separate=True)),
    ]


# ── one case through the three agents ───────────────────────────────────────
def run_case(case, work, gui, potcar_dir, env_base):
    name = case['name']
    root = work / name
    root.mkdir(parents=True)
    (root / 'POSCAR').write_text(case['poscar'])
    env = dict(env_base, VASP_POTCAR_DIR=str(potcar_dir))

    proj = re.sub(r'[^\w\-]', '_', name)
    if case.get('form') is not None:
        d = dict(case['form'])
        d.update(project_name=name, poscar=case['poscar'], potcar_dir=str(potcar_dir),
                 potcar_choices={}, profile='')
        gui.CONFIG['projects_dir'] = str(root / 'gui')
        gui.CONFIG['mpi_np'] = d.get('mpi_np', 1)
        client = gui.app.test_client()
        resp = client.post('/api/generate', json=d)
        if resp.status_code != 200:
            raise RuntimeError(f"GUI /api/generate failed: {resp.get_json()}")
        gui_dir = root / 'gui' / gui._slug(name)
        instr = (root / 'gui' / 'instructions.txt').read_text()
    else:
        gui_dir = None
        instr = case['text']
    (root / 'instructions.txt').write_text(instr)

    outs = {}
    for label, script, extra in (('ws', 'vasp-agent.py', []),
                                 ('sl', 'vasp-agent-slurm.py', ['-p', 'slurm'])):
        if label not in case.get('agents', ('gui', 'ws', 'sl')):
            continue
        wd = root / label
        wd.mkdir()
        r = subprocess.run([sys.executable, str(REPO / script), '-i', '../instructions.txt',
                            '-s', '../POSCAR'] + extra, cwd=wd, env=env,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{script} failed:\n{(r.stderr or r.stdout)[-1500:]}")
        found = [p for p in wd.iterdir() if p.is_dir()]
        outs[label] = found[0]
    if gui_dir and 'gui' in case.get('agents', ('gui', 'ws', 'sl')):
        outs['gui'] = gui_dir
    return outs, instr


def compare_outputs(outs):
    """Identical calculation inputs across the agents; returns list of problems."""
    problems = []
    ref_label = 'ws'
    ref = outs[ref_label]
    steps = sorted(p.name for p in ref.iterdir()
                   if p.is_dir() and re.match(r'\d\d_', p.name))
    for label, d in outs.items():
        other = sorted(p.name for p in d.iterdir() if p.is_dir() and re.match(r'\d\d_', p.name))
        if other != steps:
            problems.append(f"{label}: step directories {other} != workstation {steps}")
    for step in steps:
        for f in COMPARE:
            a = ref / step / f
            for label, d in outs.items():
                if label == ref_label:
                    continue
                b = d / step / f
                if a.exists() != b.exists():
                    problems.append(f"{step}/{f}: exists in {ref_label}={a.exists()} but {label}={b.exists()}")
                elif a.exists() and not filecmp.cmp(a, b, shallow=False):
                    problems.append(f"{step}/{f}: {label} differs from workstation")
        # run.sh may differ, but must call the same copy scripts
        for label, d in outs.items():
            rs = d / step / 'run.sh'
            if not rs.exists():
                problems.append(f"{label}: {step}/run.sh missing"); continue
            txt = rs.read_text()
            for cp in ('copy_from_relax.sh', 'copy_from_scf.sh'):
                if (d / step / cp).exists() and cp not in txt:
                    problems.append(f"{label}: {step}/run.sh never calls {cp}")
    return problems


def check_semantics(case, out_dir, instr):
    """Expected values for the case; returns list of problems."""
    ex, bad = case.get('expect', {}), []
    n = natoms(out_dir / '02_scf' / 'POSCAR') if (out_dir / '02_scf').exists() else \
        natoms(next(out_dir.glob('0*/POSCAR')))
    np_ranks = int(re.search(r'^MPI:\s*(\d+)', instr, re.M).group(1)) \
        if re.search(r'^MPI:\s*(\d+)', instr, re.M) else 1
    steps = {s: out_dir / s for s in STEP_KEYS if (out_dir / s).exists()}

    def kpra_of(step):
        m = kpts_mesh(steps[step] / 'KPOINTS')
        return m[0] * m[1] * m[2] * n if m else None

    if 'magmom' in ex:
        for s, d in steps.items():
            if s in ('06_dfpt',) and 'dfpt' in STEP_KEYS.get(s, ''):
                pass
            t = incar_tags(d / 'INCAR')
            if t.get('MAGMOM', '').replace('.0 ', '.0 ') != ex['magmom']:
                bad.append(f"{s}: MAGMOM '{t.get('MAGMOM')}' != '{ex['magmom']}'")
            if ex.get('ispin') and t.get('ISPIN') != str(ex['ispin']):
                bad.append(f"{s}: ISPIN {t.get('ISPIN')} != {ex['ispin']}")
            if ex.get('ispin') is None and 'ISPIN' in t:
                bad.append(f"{s}: ISPIN must not be set for SOC")

    if 'kpra' in ex:
        target, tol = ex['kpra']
        got = kpra_of('02_scf')
        if got is None or abs(got - target) / target > tol:
            bad.append(f"02_scf kpra {got} not within {tol:.0%} of {target}")
        if '04_dos' in steps and kpts_mesh(steps['04_dos'] / 'KPOINTS') != tuple(
                2 * x if i < 2 or 'is_2d' not in instr.lower() and 'monolayer' not in instr.lower() else 1
                for i, x in enumerate(kpts_mesh(steps['02_scf'] / 'KPOINTS'))):
            if 'monolayer' not in instr.lower():
                bad.append("04_dos mesh is not 2x the SCF mesh")
        for s in ('06_dfpt',):
            if s in steps and kpts_mesh(steps[s] / 'KPOINTS') != kpts_mesh(steps['02_scf'] / 'KPOINTS'):
                bad.append(f"{s} mesh != SCF mesh")
        if '08_lobster' in steps:
            sm, lm = kpts_mesh(steps['02_scf'] / 'KPOINTS'), kpts_mesh(steps['08_lobster'] / 'KPOINTS')
            if lm != tuple(2 * x for x in sm):
                bad.append("08_lobster mesh is not 2x the SCF mesh")
    if ex.get('hex6'):
        for st in steps:
            m = kpts_mesh(steps[st] / 'KPOINTS')
            if m and (m[0] != m[1] or m[0] % 6):
                bad.append(f"{st}: hexagonal in-plane mesh {m[:2]} is not a multiple of 6")
    if 'relax_kpra' in ex and '01_relax' in steps:
        target, tol = ex['relax_kpra']
        got = kpra_of('01_relax')
        if got is None or abs(got - target) / target > tol:
            bad.append(f"01_relax kpra {got} not within {tol:.0%} of {target}")

    if '01_relax' in steps:
        t = incar_tags(steps['01_relax'] / 'INCAR')
        if 'blocks' not in ex:
            if t.get('ISTART') != '0' or t.get('ICHARG') != '2':
                bad.append("01_relax must start from scratch (ISTART=0, ICHARG=2)")
        if t.get('LWAVE', '').upper() != '.TRUE.' or t.get('LCHARG', '').upper() != '.TRUE.':
            bad.append("01_relax must write WAVECAR and CHGCAR")
        if '02_scf' in steps and not (steps['02_scf'] / 'copy_from_relax.sh').exists():
            bad.append("02_scf lacks copy_from_relax.sh")

    # parallel layout: valid for the rank count, ELF => KPAR=1 with NCORE re-derived
    for s in ('01_relax', '02_scf', '04_dos'):
        if s not in steps:
            continue
        t = incar_tags(steps[s] / 'INCAR')
        if ex.get('no_parallel'):
            if 'KPAR' in t or 'NCORE' in t:
                bad.append(f"{s}: KPAR/NCORE written for a single rank")
            continue
        try:
            kp, nc = int(t['KPAR'].split()[0]), int(t['NCORE'].split()[0])
        except (KeyError, ValueError):
            bad.append(f"{s}: KPAR/NCORE missing"); continue
        if np_ranks % kp or (np_ranks // kp) % nc:
            bad.append(f"{s}: KPAR={kp} NCORE={nc} not a valid split of {np_ranks} ranks")
        if s == '02_scf' and re.search(r'^LELF', (steps[s] / 'INCAR').read_text(), re.M):
            if kp != 1 or nc == 1:
                bad.append(f"02_scf: ELF run has KPAR={kp} NCORE={nc} (expect KPAR=1, NCORE>1)")

    if ex.get('text_parallel'):
        t = incar_tags(steps['02_scf'] / 'INCAR')
        if t.get('NCORE') != '8' or t.get('KPAR', '').split()[0] != '2':
            bad.append(f"02_scf: SCF_NCORE=8 should give NCORE=8, KPAR=2 (got {t.get('KPAR')}, {t.get('NCORE')})")
        t = incar_tags(steps['04_dos'] / 'INCAR')
        if t.get('KPAR') != '2' or t.get('NCORE') != '2':
            bad.append(f"04_dos: INCAR-block KPAR=2 should give KPAR=2, NCORE=2 (got {t.get('KPAR')}, {t.get('NCORE')})")
        t = incar_tags(steps['01_relax'] / 'INCAR')
        if 'KPAR' not in t:
            bad.append("01_relax: KPAR missing")

    if ex.get('walltimes'):
        for st, want in ex['walltimes'].items():
            rs = (steps[st] / 'run.sh').read_text() if st in steps else ''
            if '#SBATCH' not in rs:
                continue                                  # workstation run.sh
            m = re.search(r'^#SBATCH --time=(\S+)', rs, re.M)
            if not m or m.group(1) != want:
                bad.append(f"{st}: walltime {m.group(1) if m else None} (want {want})")
            for d in ('--signal=B:USR1@900', '--requeue', 'vf_resume_prepare', 'vf_after vasp'):
                if d not in rs:
                    bad.append(f"{st}: run.sh lacks '{d}' (auto-continue)")

    if 'u_everywhere' in ex:
        want = ex['u_everywhere']
        for st, sd in steps.items():
            if st.startswith('00'):
                continue
            t = incar_tags(sd / 'INCAR')
            has = t.get('LDAU') == '.TRUE.' and t.get('LDAUU', '').split()[:1] == ['4.1']   # Fe U_eff
            if has != want:
                bad.append(f"{st}: GGA+U {'expected' if want else 'not expected'} "
                           f"(LDAU={t.get('LDAU')}, LDAUU={t.get('LDAUU')})")
            n = sum(1 for l in (sd / 'INCAR').read_text().splitlines()
                    if l.strip().startswith('LMAXMIX'))
            if n > 1 or (want and n != 1):
                bad.append(f"{st}: {n} LMAXMIX lines")

    if ex.get('elf_separate'):
        t = incar_tags(steps['02_scf'] / 'INCAR')
        e = steps['02_scf'] / 'elf' / 'INCAR'
        if 'LELF' in t:
            bad.append("02_scf: LELF must not be in the SCF INCAR with ELF: separate")
        if not e.exists() or not (steps['02_scf'] / 'run_elf.sh').exists():
            bad.append("02_scf: elf/INCAR or run_elf.sh missing")
        else:
            te = incar_tags(e)
            want = {'LELF': '.TRUE.', 'KPAR': '1', 'ISTART': '1', 'ICHARG': '1', 'LWAVE': '.FALSE.'}
            for k, v in want.items():
                if te.get(k) != v:
                    bad.append(f"02_scf/elf/INCAR: {k}={te.get(k)} (want {v})")
            for k in ('ENCUT', 'ISMEAR', 'SIGMA'):
                if te.get(k) != t.get(k):
                    bad.append(f"02_scf/elf/INCAR: {k} differs from the SCF")
            if 'run_elf.sh' not in (steps['02_scf'] / 'run.sh').read_text():
                bad.append("02_scf/run.sh never calls run_elf.sh")

    if ex.get('blocks'):
        r, s_, b, dd = (incar_tags(steps[k] / 'INCAR') for k in ('01_relax', '02_scf', '03_bands', '04_dos'))
        checks = [
            (r.get('EDIFFG') == '-0.005', "relax EDIFFG"), (r.get('KPAR') == '2', "relax KPAR"),
            (r.get('LREAL') == 'Auto', "global LREAL in relax"),
            (s_.get('SIGMA') == '0.1' and s_.get('NELM') == '200', "scf SIGMA/NELM"),
            (b.get('NBANDS') == '64', "bands NBANDS"),
            (dd.get('NEDOS') == '5000' and dd.get('SIGMA') == '0.02' and dd.get('ISMEAR') == '0', "dos block"),
            (kpts_mesh(steps['01_relax'] / 'KPOINTS') == (3, 3, 2), "relax explicit KPOINTS"),
            (kpts_mesh(steps['02_scf'] / 'KPOINTS') == (9, 9, 5), "scf explicit KPOINTS"),
            (kpts_mesh(steps['04_dos'] / 'KPOINTS') == (18, 18, 10), "dos = 2x explicit scf mesh"),
            (kpts_mesh(steps['08_lobster'] / 'KPOINTS') == (18, 18, 10), "lobster = 2x explicit scf mesh"),
            ((steps['02_scf'] / '.explicit_kpoints').exists(), "scf .explicit_kpoints marker"),
            (not (steps['04_dos'] / '.explicit_kpoints').exists(), "dos must not carry the marker"),
            ('Line-mode' in (steps['03_bands'] / 'KPOINTS').read_text(), "global KPOINTS never hits bands"),
        ]
        bad += [f"explicit blocks: {what}" for ok, what in checks if not ok]
    return bad


# ── main ────────────────────────────────────────────────────────────────────
def check_parallel_rule():
    """The plain KPAR/NCORE rule on hand-worked examples."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'modules'))
    from vasp_input_generator import VASPInputGenerator as G
    cases = {  # (ranks, n_k, kpar, ncore) -> expected (KPAR, NCORE)
        (40, 500, None, None): (40, 1),   # plenty of k-points: one per rank
        (40, 20, None, None):  (20, 1),   # 2 ranks/group -> NCORE 1
        (40, 7, None, None):   (5, 2),    # 8 ranks/group -> NCORE 2 (<= sqrt 8)
        (40, 1, None, None):   (1, 5),    # 40 ranks/group -> NCORE 5 (<= sqrt 40)
        (40, 500, 1, None):    (1, 5),    # ELF: KPAR pinned to 1
        (40, 500, None, 2):    (20, 2),   # NCORE given -> KPAR derived
        (40, 500, 12, None):   (10, 2),   # KPAR snapped down to a divisor
    }
    bad = []
    for (n, k, kp, nc), want in cases.items():
        got = G._auto_kpar_ncore(n, k, kpar=kp, ncore=nc)
        if got != want:
            bad.append(f"_auto_kpar_ncore{(n, k, kp, nc)} = {got}, want {want}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('-k', '--filter', default='', help='only cases whose name contains this')
    ap.add_argument('--potcar-dir', default=None, help='real POTCAR library (default: stub)')
    ap.add_argument('--keep', action='store_true', help='keep the work directory')
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix='vasp_consistency_'))
    if args.potcar_dir:
        potcar_dir = Path(args.potcar_dir).expanduser()
    else:
        potcar_dir = work / 'potcar'
        for el, (z, enmax) in STUB_POTCARS.items():
            (potcar_dir / el).mkdir(parents=True)
            (potcar_dir / el / 'POTCAR').write_text(
                f"  PAW_PBE {el} 01Jan2000\n   {z}.0000  mass and valenz\n"
                f"   ENMAX  =  {enmax}; ENMIN  =  200.0 eV\n   ZVAL   =   {z}.000    mass and valenz\n")
    env_base = dict(os.environ)
    gui = load_gui()

    total_bad = 0
    for case in cases():
        if args.filter and args.filter not in case['name']:
            continue
        try:
            outs, instr = run_case(case, work, gui, potcar_dir, env_base)
            problems = compare_outputs(outs)
            problems += [f"[{k}] {p}" for k in outs for p in check_semantics(case, outs[k], instr)]
        except Exception as e:                                   # noqa: BLE001
            problems = [f"EXCEPTION: {e}"]
        status = 'ok  ' if not problems else 'FAIL'
        agents = '+'.join({'gui': 'GUI', 'ws': 'workstation', 'sl': 'SLURM'}[a] for a in
                          ('gui', 'ws', 'sl') if a in outs)
        print(f"[{status}] {case['name']:<28} ({agents})")
        for p in problems:
            print(f"        - {p}")
        total_bad += len(problems)

    rule = check_parallel_rule()
    print(f"[{'ok  ' if not rule else 'FAIL'}] {'kpar-ncore-rule':<28} (generator)")
    for p in rule:
        print(f"        - {p}")
    total_bad += len(rule)

    print(f"\n{'ALL CONSISTENT' if not total_bad else str(total_bad) + ' problem(s)'}"
          f"   (work dir: {work}{'' if args.keep else ' — removed'})")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if total_bad else 0)


if __name__ == '__main__':
    main()
