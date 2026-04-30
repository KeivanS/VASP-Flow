#!/usr/bin/env python3
"""
VASP Workflow GUI  — run with:  python3 vasp-gui.py
Open browser at:  http://localhost:5001
"""
try:
    from flask import Flask, request, jsonify, Response, send_file
except ImportError:
    print("Flask not installed.  Run:  pip install flask"); raise SystemExit(1)

import os, sys, re, subprocess, threading, time, shutil, json
import queue as Q
from pathlib import Path
from collections import defaultdict

APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.environ.get('VASP_PROJECTS_DIR', os.getcwd())
sys.path.insert(0, os.path.join(APP_DIR, 'modules'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

_jobs  = {}
_jlock = threading.Lock()

# ── defaults: env vars first, then hardcoded fallbacks ───────────────────────
CONFIG = {
    # execution mode
    'profile_mode':           'workstation',   # 'workstation' | 'slurm'
    # projects location
    'projects_dir':           os.environ.get('VASP_PROJECTS_DIR', PROJECTS_DIR),
    # system paths
    'vasp_std':               os.environ.get('VASP_STD',        ''),
    'vasp_ncl':               os.environ.get('VASP_NCL',        ''),
    'vasp_gam':               os.environ.get('VASP_GAM',        ''),
    'mpi_launch':             os.environ.get('MPI_LAUNCH',      'mpirun -np'),
    'mpi_np':                 int(os.environ.get('MPI_NP',      '1')),
    'wannier90_x':            os.environ.get('WANNIER90_X',     'wannier90.x'),
    'potcar_dir':             os.environ.get('VASP_POTCAR_DIR', ''),
    # SLURM settings
    'slurm_partition':        'standard',
    'slurm_nodes':            2,
    'slurm_ntasks_per_node':  64,
    'slurm_time':             '12:00:00',
    'slurm_account':          '',
    'slurm_mpi_cmd':          'srun',
    # physics defaults
    'kpath':                  'G-M-K-G',
    'nkpts_bands':            60,
    'nsw':                    100,
    'ediffg':                 '-0.01',
    'encut_manual':           520,
    'band_ymin':              '-4',
    'band_ymax':              '4',
    'dos_xmin':               '-6',
    'dos_xmax':               '6',
}

# ── load saved settings (settings.json in working directory) ──────────────────
_settings_path = os.path.join(PROJECTS_DIR, 'settings.json')
# Migrate legacy config.json → settings.json on first startup after upgrade
_legacy_config = os.path.join(PROJECTS_DIR, 'config.json')
if not os.path.isfile(_settings_path) and os.path.isfile(_legacy_config):
    try:
        shutil.copy2(_legacy_config, _settings_path)
    except Exception:
        pass
_first_run = not os.path.isfile(_settings_path)
if not _first_run:
    try:
        _saved = json.loads(Path(_settings_path).read_text())
        CONFIG.update({k: v for k, v in _saved.items() if k in CONFIG})
    except Exception:
        pass

def _slug(name):
    return re.sub(r'[^\w\-]', '_', name.strip()).strip('_') or 'vasp_project'

def _pd(slug):    return os.path.join(CONFIG['projects_dir'], slug)

def _steps(slug):
    pd = _pd(slug)
    if not os.path.isdir(pd): return []
    return sorted(d for d in os.listdir(pd)
                  if re.match(r'\d\d_', d) and os.path.isdir(os.path.join(pd, d)))

def _launch(script, slug, step):
    key = f"{slug}/{step}"
    q = Q.Queue()
    with _jlock: _jobs[key] = {'queue': q, 'status': 'running', 'rc': None}
    def _run():
        proc = subprocess.Popen(['bash', script], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=os.path.dirname(script), env=os.environ.copy())
        for line in iter(proc.stdout.readline, ''): q.put(line.rstrip())
        proc.wait()
        with _jlock:
            _jobs[key]['status'] = 'done' if proc.returncode == 0 else 'error'
            _jobs[key]['rc'] = proc.returncode
        q.put(None)
    threading.Thread(target=_run, daemon=True).start()
    return key

def _write_kpoints(path, nx, ny, nz):
    Path(path).write_text(f"Automatic Gamma mesh\n0\nGamma\n  {nx}  {ny}  {nz}\n  0  0  0\n")

def _parse_mesh(s):
    m = re.match(r'(\d+)x(\d+)(?:x(\d+))?', s.strip())
    if not m: return None
    return int(m[1]), int(m[2]), int(m[3]) if m[3] else int(m[2])

@app.route('/')
def index():
    cfg = dict(CONFIG, first_run=_first_run)
    return Response(HTML.replace('__CFG__', json.dumps(cfg)), mimetype='text/html')

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """GET: return full config. POST {key:val}: save all keys to settings.json."""
    if request.method == 'GET':
        return jsonify(dict(CONFIG))
    data = request.json or {}
    for k in CONFIG:
        if k in data:
            try:
                CONFIG[k] = type(CONFIG[k])(data[k])
            except (TypeError, ValueError):
                CONFIG[k] = data[k]
    # Resolve projects_dir to an absolute path
    if CONFIG['projects_dir']:
        CONFIG['projects_dir'] = os.path.abspath(
            os.path.expanduser(CONFIG['projects_dir']))
    Path(_settings_path).write_text(json.dumps(CONFIG, indent=2))
    # Keep profiles/slurm.json in sync so vasp-agent.py can still read it
    slurm_profile = {
        '_comment': 'Auto-generated from settings.json — edit settings.json instead.',
        'name':            'SLURM HPC',
        'vasp_std':        CONFIG['vasp_std'],
        'vasp_ncl':        CONFIG['vasp_ncl'],
        'vasp_gam':        CONFIG['vasp_gam'],
        'wannier90_x':     CONFIG['wannier90_x'],
        'mpi_cmd':         CONFIG['slurm_mpi_cmd'],
        'mpi_np':          CONFIG['slurm_nodes'] * CONFIG['slurm_ntasks_per_node'],
        'modules':         [],
        'slurm': {
            'partition':        CONFIG['slurm_partition'],
            'nodes':            CONFIG['slurm_nodes'],
            'ntasks_per_node':  CONFIG['slurm_ntasks_per_node'],
            'time':             CONFIG['slurm_time'],
            'account':          CONFIG['slurm_account'],
            'output':           'slurm-%j.out',
            'error':            'slurm-%j.err',
        },
    }
    profiles_dir = os.path.join(APP_DIR, 'profiles')
    os.makedirs(profiles_dir, exist_ok=True)
    Path(os.path.join(profiles_dir, 'slurm.json')).write_text(
        json.dumps(slurm_profile, indent=2))
    return jsonify(ok=True)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    d = request.json or {}
    name   = d.get('project_name','VASP Project').strip()
    poscar = d.get('poscar','').strip()
    if not poscar: return jsonify(error='POSCAR is required'), 400

    # ── build instruction text ──────────────────────────────────────────
    functional = d.get('functional','PBEsol')
    spin_mode  = d.get('spin_mode','none')      # none | collinear | soc_z | soc_x | soc_y
    use_u      = d.get('use_u', False)
    u_entries  = d.get('u_entries', [])         # [{element, orbital, U}]

    methods = [functional + ' functional']
    if spin_mode == 'collinear':       methods.append('collinear magnetization')
    elif spin_mode.startswith('soc'):  methods.append(f'SOC with magnetization in {spin_mode[-1]}-direction')
    if d.get('hexagonal'):  methods.append('hexagonal')
    if d.get('is_2d'):      methods.append('2D monolayer')
    for e in u_entries:
        methods.append(f"GGA+U with U={e['U']} on {e['element']}-{e['orbital']} orbitals")

    # Tasks
    tasks, conv_lines = [], []
    param_mode = d.get('param_mode', 'convergence')  # 'convergence' | 'manual'

    if param_mode == 'convergence':
        conv_kp    = d.get('conv_kp','').strip()
        conv_encut = d.get('conv_encut','').strip()
        if conv_kp or conv_encut:
            tasks.append('convergence study')
            if conv_kp:    conv_lines.append(f'Test k-points {conv_kp}')
            if conv_encut: conv_lines.append(f'Test ENCUT {conv_encut}')

    isif_map = {'ions_only':'2','fixed_vol':'4','full_relax':'3',
                'fixed_shape':'7','vol_only':'6'}
    isif     = isif_map.get(d.get('relax_type','full_relax'), '3')
    nsw      = d.get('nsw', CONFIG['nsw'])
    ediffg   = d.get('ediffg', CONFIG['ediffg'])
    nkpts    = d.get('nkpts_bands', CONFIG['nkpts_bands'])

    if d.get('relax'):  tasks.append('structure relaxation')
    if d.get('scf'):    tasks.append('SCF calculation')
    if d.get('bands'):
        tasks.append('band structure along ' + d.get('kpath', CONFIG['kpath']))
    if d.get('dos'):
        # Build projection string from dos_proj list [{element,orbitals:[s,p,d,f]}]
        proj_list = d.get('dos_proj', [])
        proj_str  = ' and '.join(
            ' '.join(f"{p['element']}-{o}" for o in p.get('orbitals',[]))
            for p in proj_list if p.get('orbitals')
        )
        tasks.append('DOS' + (f' (projected on {proj_str})' if proj_str else ''))
    if d.get('wannier'): tasks.append('Wannier90 wannierization')
    if d.get('dfpt'):    tasks.append('DFPT Born charges and dielectric')
    if d.get('phonons'): tasks.append('phonons lattice dynamics')

    mpi = max(1, int(d.get('mpi_np', CONFIG['mpi_np']) or 1))

    # POTCAR settings
    potcar_dir     = d.get('potcar_dir', '').strip() or CONFIG['potcar_dir']
    potcar_choices = d.get('potcar_choices', {})

    lines = [f'Project: {name}', '']
    lines += ['Methods: ' + '\n         '.join(methods), '']
    if tasks: lines += ['Tasks: ' + '\n       '.join(tasks), '']
    if conv_lines: lines += ['Convergence: ' + '\n             '.join(conv_lines), '']
    lines += [f'ISIF: {isif}', f'NSW: {nsw}', f'EDIFFG: {ediffg}', f'NKPTS: {nkpts}']
    if mpi > 1: lines.append(f'MPI: {mpi}')

    # DFPT parameters
    if d.get('dfpt'):
        ediff = d.get('dfpt_ediff', '1E-8').strip()
        if ediff: lines.append(f'DFPT_EDIFF: {ediff}')

    # Phonons parameters
    if d.get('phonons'):
        lines.append(f'PHONONS_DIM: {d.get("phonons_dim","2 2 2")}')
        lines.append(f'PHONONS_MESH: {d.get("phonons_mesh","20 20 20")}')
        lines.append(f'PHONONS_DISP: {d.get("phonons_disp",0.01)}')
        pb = d.get('phonons_band','').strip()
        if pb: lines.append(f'PHONONS_BAND: {pb}')
        if not d.get('phonons_nac', True): lines.append('PHONONS_NAC: FALSE')

    # Wannier90 parameters
    if d.get('wannier'):
        num_wann = d.get('wannier_num_wann', 8)
        proj     = d.get('wannier_proj', '').strip()
        ewin     = d.get('wannier_ewin', '').strip()
        if num_wann: lines.append(f'WANNIER_NUM_WANN: {num_wann}')
        if proj:     lines.append(f'WANNIER_PROJ: {proj}')
        if ewin:     lines.append(f'WANNIER_EWIN: {ewin}')

    # Manual params → add to instruction text so parser picks them up
    if param_mode == 'manual':
        encut_m = d.get('manual_encut','').strip()
        if encut_m: lines.append(f'ENCUT: {encut_m}')

    run_dir = CONFIG['projects_dir']
    os.makedirs(run_dir, exist_ok=True)
    Path(os.path.join(run_dir,'instructions.txt')).write_text('\n'.join(lines))
    Path(os.path.join(run_dir,'POSCAR')).write_text(poscar)

    # Write POTCAR choices for the agent; clear if none
    choices_file = os.path.join(run_dir, 'potcar_choices.json')
    if potcar_choices:
        Path(choices_file).write_text(json.dumps(potcar_choices))
    elif os.path.exists(choices_file):
        os.remove(choices_file)

    env = os.environ.copy()
    if potcar_dir:
        env['VASP_POTCAR_DIR'] = potcar_dir
    for key, evar in [('vasp_std','VASP_STD'), ('vasp_ncl','VASP_NCL'),
                      ('vasp_gam','VASP_GAM'), ('mpi_launch','MPI_LAUNCH'),
                      ('wannier90_x','WANNIER90_X')]:
        val = CONFIG.get(key, '')
        if val:
            env[evar] = val
    env['MPI_NP'] = str(CONFIG.get('mpi_np', 1))

    profile = d.get('profile', '').strip()
    agent_cmd = [sys.executable, os.path.join(APP_DIR,'vasp-agent.py'),
                 '-i','instructions.txt','-s','POSCAR']
    if profile and profile != 'default':
        agent_cmd += ['--profile', profile]
    result = subprocess.run(agent_cmd,
        capture_output=True, text=True, cwd=run_dir, env=env)

    if result.returncode != 0:
        return jsonify(error=(result.stderr or result.stdout or 'Error').strip()), 500

    slug = _slug(name)

    # ── if manual mode: patch KPOINTS immediately ───────────────────────
    if param_mode == 'manual':
        kmesh_scf = d.get('manual_kmesh_scf','').strip()
        kmesh_dos = d.get('manual_kmesh_dos','').strip() or kmesh_scf
        if kmesh_scf:
            m = _parse_mesh(kmesh_scf)
            if m:
                for step in ['01_relax','02_scf']:
                    kp = os.path.join(_pd(slug), step, 'KPOINTS')
                    if os.path.exists(kp): _write_kpoints(kp, *m)
        if kmesh_dos:
            m = _parse_mesh(kmesh_dos)
            if m:
                kp = os.path.join(_pd(slug), '04_dos', 'KPOINTS')
                if os.path.exists(kp): _write_kpoints(kp, *m)

    has_conv = bool(conv_lines)

    # Save DOS projection info for sumo to use later
    dos_proj = d.get('dos_proj', [])
    if dos_proj and os.path.isdir(_pd(slug)):
        Path(os.path.join(_pd(slug), 'dos_proj.json')).write_text(json.dumps(dos_proj))

    # Save project form data so the Setup page can be restored on resume
    if os.path.isdir(_pd(slug)):
        Path(os.path.join(_pd(slug), 'project.json')).write_text(json.dumps(d))

    return jsonify(ok=True, project=slug, steps=_steps(slug),
                   has_convergence=has_conv, output=result.stdout)

@app.route('/api/project_settings/<slug>')
def api_project_settings(slug):
    """Return saved project.json + POSCAR for a project so Setup can be pre-populated."""
    pd_ = _pd(slug)
    proj_file   = os.path.join(pd_, 'project.json')
    poscar_file = os.path.join(pd_, 'POSCAR')
    if not os.path.isdir(pd_):
        return jsonify(error='Project not found'), 404
    settings = {}
    if os.path.exists(proj_file):
        try: settings = json.loads(Path(proj_file).read_text())
        except Exception: pass
    poscar = ''
    if os.path.exists(poscar_file):
        try: poscar = Path(poscar_file).read_text()
        except Exception: pass
    return jsonify(settings=settings, poscar=poscar)


@app.route('/api/save_project', methods=['POST'])
def api_save_project():
    """Save project form data to project.json (and POSCAR) without running the agent.

    Creates the project directory if it does not exist, so the project
    appears in the resume dropdown immediately.
    """
    d    = request.json or {}
    name = d.get('project_name', '').strip()
    if not name:
        return jsonify(error='Project name is required'), 400
    slug = _slug(name)
    pd_  = _pd(slug)
    os.makedirs(pd_, exist_ok=True)
    poscar = d.get('poscar', '').strip()
    if poscar:
        Path(os.path.join(pd_, 'POSCAR')).write_text(poscar)
    Path(os.path.join(pd_, 'project.json')).write_text(json.dumps(d))
    return jsonify(ok=True, project=slug)


@app.route('/api/run', methods=['POST'])
def api_run():
    d    = request.json or {}
    slug = d.get('project', '')
    step = d.get('step', '')
    pd   = _pd(slug)
    special = {
        'convergence':  'run_convergence.sh',
        'all':          'run_all.sh',
        'calculations': 'run_calculations.sh',
        'analyze':      'analyze.sh',
    }
    script = (os.path.join(pd, special[step]) if step in special
              else os.path.join(pd, step, 'run.sh'))
    if not os.path.exists(script):
        return jsonify(error=f'Script not found: {script}'), 404
    return jsonify(job_key=_launch(script, slug, step))

@app.route('/api/run_phase2', methods=['POST'])
def api_run_phase2():
    d = request.json or {}
    slug = d['project']
    encut, kmesh, kmesh_dos = d.get('encut',''), d.get('kmesh',''), d.get('kmesh_dos','') or d.get('kmesh','')
    pd = _pd(slug)
    if not encut or not kmesh: return jsonify(error='ENCUT and k-mesh are required'), 400

    for step in _steps(slug):
        incar = os.path.join(pd, step, 'INCAR')
        if os.path.exists(incar):
            txt = Path(incar).read_text()
            Path(incar).write_text(re.sub(r'^ENCUT.*', f'ENCUT = {encut}', txt, flags=re.MULTILINE))

    for step in _steps(slug):
        if step == '03_bands': continue
        kp   = os.path.join(pd, step, 'KPOINTS')
        mesh = _parse_mesh(kmesh_dos if step == '04_dos' else kmesh)
        if mesh and os.path.exists(kp): _write_kpoints(kp, *mesh)

    prod = [s for s in _steps(slug) if not s.startswith('00')]
    tmp  = os.path.join(pd, '_phase2.sh')
    with open(tmp,'w') as f:
        f.write('#!/bin/bash\nset -e\n')
        for s in prod:
            f.write(f'echo ">>> {s}"\ncd "{os.path.join(pd,s)}" && bash run.sh\n')
    os.chmod(tmp,0o755)
    return jsonify(job_key=_launch(tmp, slug, 'phase2'))

@app.route('/api/stream/<path:job_key>')
def api_stream(job_key):
    def gen():
        for _ in range(40):
            with _jlock:
                if job_key in _jobs: break
            time.sleep(0.1)
        with _jlock: entry = _jobs.get(job_key)
        if not entry: yield 'data: [Job not found]\n\n'; return
        q = entry['queue']
        while True:
            try:
                line = q.get(timeout=2)
                if line is None: yield 'data: [DONE]\n\n'; break
                yield f'data: {line}\n\n'
            except Q.Empty:
                with _jlock: st = _jobs.get(job_key,{}).get('status','')
                if st in ('done','error'): yield 'data: [DONE]\n\n'; break
                yield ': keepalive\n\n'
    return Response(gen(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

@app.route('/api/check_overwrite', methods=['POST'])
def api_check_overwrite():
    """Return info about what would be overwritten.

    For 'generate': reports whether the project directory already exists and
    which step subdirectories contain files.
    For 'run': reports whether the target step already has an OUTCAR.
    """
    d    = request.json or {}
    mode = d.get('mode', 'generate')   # 'generate' | 'run'
    slug = _slug(d.get('project_name', d.get('project', '')))
    pd   = _pd(slug)

    if mode == 'generate':
        if not os.path.isdir(pd):
            return jsonify(exists=False)
        # Collect step subdirs that contain any files
        existing_steps = []
        for step in _steps(slug):
            step_dir = os.path.join(pd, step)
            if os.path.isdir(step_dir) and any(
                f for f in os.listdir(step_dir) if not f.startswith('.')
            ):
                existing_steps.append(step)
        return jsonify(exists=True, steps=existing_steps)

    elif mode == 'run':
        step = d.get('step', '')
        special_scripts = {'convergence', 'all', 'calculations', 'analyze'}
        if step in special_scripts:
            # For 'all'/'convergence'/'calculations': list steps with OUTCARs
            steps_with_outcar = []
            for s in _steps(slug):
                if os.path.exists(os.path.join(pd, s, 'OUTCAR')):
                    steps_with_outcar.append(s)
            return jsonify(has_outcar=bool(steps_with_outcar),
                           steps=steps_with_outcar)
        else:
            outcar = os.path.join(pd, step, 'OUTCAR')
            return jsonify(has_outcar=os.path.exists(outcar), steps=[step] if os.path.exists(outcar) else [])

    return jsonify(exists=False, has_outcar=False)


@app.route('/api/status/<slug>')
def api_status(slug):
    pd = _pd(slug)
    if not os.path.isdir(pd): return jsonify(error='not found'), 404
    out = {}
    for step in _steps(slug):
        outcar = os.path.join(pd, step, 'OUTCAR')
        out[step] = ('done' if os.path.exists(outcar) and
                     'reached required accuracy' in Path(outcar).read_text(errors='replace')
                     else 'ran' if os.path.exists(outcar) else 'ready')
    with _jlock:
        for key, info in _jobs.items():
            if key.startswith(slug+'/'):
                s = key.split('/',1)[1]
                if s in out or s in ('convergence','phase2','all'): out[s] = info['status']
    return jsonify(out)

@app.route('/api/convergence_data/<slug>/<dtype>')
def api_conv_data(slug, dtype):
    fname = 'encut_convergence.dat' if dtype=='encut' else 'kpoint_convergence.dat'
    path  = os.path.join(_pd(slug),'00_convergence',dtype,fname)
    if not os.path.exists(path): return jsonify(data=[])
    rows = []
    for line in Path(path).read_text().splitlines():
        p = line.strip().split()
        if len(p) >= 2:
            try: rows.append({'x':p[0],'y':float(p[1])})
            except ValueError: pass
    return jsonify(data=rows)

@app.route('/api/outcar/<slug>/<step>')
def api_outcar(slug, step):
    """Return last 300 lines of OUTCAR, plus key extracted values."""
    path = os.path.join(_pd(slug), step, 'OUTCAR')
    if not os.path.exists(path):
        return jsonify(error='OUTCAR not found — has this step run yet?'), 404
    txt   = Path(path).read_text(errors='replace')
    lines = txt.splitlines()
    tail  = '\n'.join(lines[-300:])

    # Extract key values for the summary header
    info = {}
    energies = re.findall(r'energy\s+without entropy=\s+([-\d.]+)', txt)
    if energies: info['energy'] = energies[-1] + ' eV'
    fermi = re.findall(r'E-fermi\s*:\s*([-\d.]+)', txt)
    if fermi: info['efermi'] = fermi[-1] + ' eV'
    mag = re.findall(r'number of electron\s+[\d.]+\s+magnetization\s+([-\d.]+)', txt)
    if mag: info['magmom'] = mag[-1] + ' μB'
    info['converged'] = 'reached required accuracy' in txt
    info['total_lines'] = len(lines)

    return jsonify(tail=tail, info=info)

def _cumulative_dos_plot(dos_dir, out_png, project_label):
    """Read DOSCAR + POSCAR and save a cumulative DOS PNG. No pymatgen needed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    import numpy as np
    from collections import defaultdict

    # Map ions → elements from POSCAR/CONTCAR
    ion_elements = []
    for fname in ('CONTCAR', 'POSCAR'):
        p = os.path.join(dos_dir, fname)
        if os.path.isfile(p):
            ls = open(p).readlines()
            elements = ls[5].split()
            counts   = [int(x) for x in ls[6].split()]
            for el, cnt in zip(elements, counts):
                ion_elements.extend([el] * cnt)
            break
    if not ion_elements:
        return False

    doscar = os.path.join(dos_dir, 'DOSCAR')
    if not os.path.isfile(doscar):
        return False

    raw    = open(doscar).readlines()
    nions  = int(raw[0].split()[0])
    parts  = raw[5].split()
    nedos  = int(parts[2])
    efermi = float(parts[3])

    tot      = np.array([[float(x) for x in l.split()] for l in raw[6:6+nedos]])
    energies = tot[:,0] - efermi
    spin_pol = tot.shape[1] == 5

    el_dos = defaultdict(lambda: np.zeros(nedos))
    offset = 6 + nedos
    for i in range(nions):
        start = offset + i * (nedos + 1) + 1
        d = np.array([[float(x) for x in l.split()] for l in raw[start:start+nedos]])
        ion_total = (d[:,1::2].sum(axis=1) + d[:,2::2].sum(axis=1)) if spin_pol else d[:,1:].sum(axis=1)
        el = ion_elements[i] if i < len(ion_elements) else f'ion{i}'
        el_dos[el] += ion_total

    xmin = float(CONFIG.get('dos_xmin', -6))
    xmax = float(CONFIG.get('dos_xmax',  6))
    mask = (energies >= xmin) & (energies <= xmax)
    en   = energies[mask]

    elements_ordered = list(dict.fromkeys(ion_elements))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(7, 5))
    cumulative = np.zeros(mask.sum())
    for i, el in enumerate(elements_ordered):
        prev       = cumulative.copy()
        cumulative = cumulative + el_dos[el][mask]
        color = colors[i % len(colors)]
        ax.fill_between(en, prev, cumulative, alpha=0.35, color=color, label=el)
        ax.plot(en, cumulative, color=color, lw=1.2)

    sc = ScalarFormatter(useOffset=False, useMathText=False)
    sc.set_scientific(False)
    ax.yaxis.set_major_formatter(sc)
    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Energy − $E_F$ (eV)', fontsize=12)
    ax.set_ylabel('Cumulative DOS (states/eV)', fontsize=12)
    ax.set_title(f'Cumulative DOS — {project_label}')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def _band_plot_opts(slug):
    """Return (ymin, ymax, labels_str|None) from project.json."""
    ymin, ymax, labels = CONFIG['band_ymin'], CONFIG['band_ymax'], None
    sf = os.path.join(_pd(slug), 'project.json')
    if os.path.exists(sf):
        try:
            d = json.loads(Path(sf).read_text())
            ewin = d.get('ewin', '').strip()
            if ewin:
                m = re.search(r'([-\d.]+)\s*(?:to|:)\s*([-\d.]+)', ewin)
                if m:
                    ymin, ymax = m.group(1), m.group(2)
            kpath = d.get('kpath', '').strip()
            if kpath:
                labels = ','.join(kpath.replace(' ', '').split('-'))
        except Exception:
            pass
    return ymin, ymax, labels


@app.route('/api/plot/<slug>/<ptype>')
def api_plot(slug, ptype):
    """Serve plot image. Normalise ptype, run sumo if needed, return png."""
    pd_  = _pd(slug)
    ana  = os.path.join(pd_, 'analysis')
    os.makedirs(ana, exist_ok=True)
    base = _slug_base(slug)

    # Normalise: bare 'dos' → prefer projected
    if ptype == 'dos': ptype = 'dos_proj'

    if ptype == 'bands':
        candidates = [
            os.path.join(ana,             f'{base}_band.png'),
            os.path.join(pd_,'03_bands',  f'{base}_band.png'),
            os.path.join(ana,             'bands.png'),
        ]
    elif ptype == 'dos_proj':
        candidates = [
            os.path.join(ana,          f'{base}_proj_dos.png'),
            os.path.join(pd_,'04_dos', f'{base}_proj_dos.png'),
            # no fallback to total DOS — show "not available" if proj doesn't exist
        ]
    else:  # dos_total → cumulative
        candidates = [
            os.path.join(ana,          f'{base}_cumulative_dos.png'),
            os.path.join(ana,          f'{base}_total_dos.png'),
            os.path.join(pd_,'04_dos', f'{base}_total_dos.png'),
            os.path.join(ana,          'dos.png'),
        ]

    for p in candidates:
        if os.path.exists(p): return send_file(p, mimetype='image/png')

    # ── run sumo ─────────────────────────────────────────────────────────
    if ptype == 'bands':
        src = os.path.join(pd_, '03_bands')
        if os.path.isdir(src):
            # Prefer Fermi level from 04_dos (dense k-mesh NSCF), fallback to 02_scf
            for oc in [os.path.join(pd_, '04_dos', 'OUTCAR'),
                       os.path.join(pd_, '02_scf', 'OUTCAR')]:
                if os.path.exists(oc):
                    m = re.findall(r'E-fermi\s*:\s*([-\d.]+)',
                                   Path(oc).read_text(errors='replace'))
                    if m:
                        # Patch efermi in vasprun.xml — sumo reads it from there
                        vr = os.path.join(src, 'vasprun.xml')
                        if os.path.exists(vr):
                            txt = Path(vr).read_text(errors='replace')
                            txt = re.sub(r'<i name="efermi">.*?</i>',
                                         f'<i name="efermi">  {m[-1]}  </i>', txt)
                            Path(vr).write_text(txt)
                        break
            ymin, ymax, labels = _band_plot_opts(slug)
            label_args = ['--labels', labels] if labels else []
            for fmt in ('png', 'pdf'):
                subprocess.run(
                    ['sumo-bandplot','--prefix',base,
                     '--ymin', ymin, '--ymax', ymax,
                     '--format', fmt] + label_args,
                    capture_output=True, cwd=src)
            for f in os.listdir(src):
                if '_band.' in f: shutil.move(os.path.join(src,f), os.path.join(ana,f))
    else:
        src = os.path.join(pd_, '04_dos')
        if os.path.isdir(src):
            orb_flag = []
            proj_json = os.path.join(pd_, 'dos_proj.json')
            if os.path.exists(proj_json):
                try:
                    proj_data = json.loads(Path(proj_json).read_text())
                    om = defaultdict(list)
                    for p in proj_data:
                        for orb in p.get('orbitals', []):
                            om[p['element']].append(orb)
                    sumo_orb = '; '.join(f"{el} {' '.join(orbs)}" for el, orbs in om.items())
                    if sumo_orb:
                        orb_flag = ['--orbitals', sumo_orb]
                except Exception:
                    pass
            # Cumulative DOS (replaces sumo total-DOS plot)
            _cumulative_dos_plot(src, os.path.join(ana, f'{base}_cumulative_dos.png'), slug)
            for fmt in ('png', 'pdf'):
                if orb_flag:
                    subprocess.run(
                        ['sumo-dosplot','--prefix',f'{base}_proj',
                         '--xmin', CONFIG['dos_xmin'], '--xmax', CONFIG['dos_xmax'],
                         '--format', fmt]+orb_flag,
                        capture_output=True, cwd=src)
            for f in os.listdir(src):
                if '_dos.' in f: shutil.move(os.path.join(src,f), os.path.join(ana,f))

    for p in candidates:
        if os.path.exists(p): return send_file(p, mimetype='image/png')

    # ── matplotlib fallback ───────────────────────────────────────────────
    py = os.path.join(ana,'plot_results.py')
    if os.path.exists(py):
        subprocess.run([sys.executable, py],
                       capture_output=True, cwd=ana,
                       env={**os.environ,'MPLBACKEND':'Agg'})
        stem = 'bands' if ptype=='bands' else 'dos'
        for ext in ['png','pdf']:
            p = os.path.join(ana, f'{stem}.{ext}')
            if os.path.exists(p):
                return send_file(p, mimetype='image/png' if ext=='png' else 'application/pdf')

    return jsonify(error='Not available yet — run the calculation first'), 404

def _slug_base(slug):
    return slug

@app.route('/api/projects')
def api_projects():
    """List existing project directories (have at least one run.sh step).

    Optional query param ?dir=<path> overrides CONFIG['projects_dir'] so the
    client can preview a new directory before saving settings.
    """
    projects = []
    try:
        raw = request.args.get('dir', '').strip()
        pd_root = os.path.abspath(os.path.expanduser(raw)) if raw else CONFIG['projects_dir']
        for name in sorted(os.listdir(pd_root)):
            full = os.path.join(pd_root, name)
            if not os.path.isdir(full):
                continue
            steps = _steps(name)
            has_poscar_or_settings = (
                os.path.isfile(os.path.join(full, 'POSCAR')) or
                os.path.isfile(os.path.join(full, 'project.json'))
            )
            if steps or has_poscar_or_settings:
                projects.append({'slug': name, 'steps': steps,
                                 'has_convergence': os.path.isdir(
                                     os.path.join(full, '00_convergence'))})
    except Exception:
        pass
    return jsonify(projects=projects)

@app.route('/api/clear_plots/<slug>')
def api_clear_plots(slug):
    """Delete cached sumo plot PNGs so they are regenerated on next request."""
    ana = os.path.join(_pd(slug), 'analysis')
    if os.path.isdir(ana):
        for f in os.listdir(ana):
            if f.endswith('.png') or f.endswith('.pdf'):
                os.remove(os.path.join(ana, f))
    return jsonify(ok=True)

@app.route('/api/plot_pdf/<slug>/<ptype>')
def api_plot_pdf(slug, ptype):
    """Serve a sumo plot PDF for download."""
    pd_  = _pd(slug)
    ana  = os.path.join(pd_, 'analysis')
    base = _slug_base(slug)
    if ptype == 'bands':
        candidates = [os.path.join(ana, f'{base}_band.pdf'),
                      os.path.join(pd_, '03_bands', f'{base}_band.pdf')]
    elif ptype == 'dos_proj':
        candidates = [os.path.join(ana, f'{base}_proj_dos.pdf'),
                      os.path.join(pd_, '04_dos', f'{base}_proj_dos.pdf')]
    else:
        candidates = [os.path.join(ana, f'{base}_total_dos.pdf'),
                      os.path.join(pd_, '04_dos', f'{base}_total_dos.pdf')]
    for p in candidates:
        if os.path.exists(p):
            return send_file(p, mimetype='application/pdf', as_attachment=True,
                             download_name=os.path.basename(p))
    return jsonify(error='PDF not found — run sumo first'), 404


@app.route('/api/born_charges/<slug>')
def api_born_charges(slug):
    """Return Born charges summary text, extracting from OUTCAR if needed."""
    step_dir = os.path.join(_pd(slug), '06_dfpt')
    txt_file = os.path.join(step_dir, 'born_charges.txt')
    if not os.path.exists(txt_file):
        outcar = os.path.join(step_dir, 'OUTCAR')
        if not os.path.exists(outcar):
            return jsonify(error='06_dfpt not run yet — run DFPT step first'), 404
        script = os.path.join(step_dir, 'extract_born.py')
        if os.path.exists(script):
            subprocess.run([sys.executable, script, outcar],
                           capture_output=True, cwd=step_dir)
    if os.path.exists(txt_file):
        return jsonify(text=Path(txt_file).read_text())
    return jsonify(error='Could not extract Born charges — check 06_dfpt/OUTCAR'), 404


@app.route('/api/phonon_plot/<slug>/<ptype>')
def api_phonon_plot(slug, ptype):
    """Serve phonon band structure or DOS image from 07_phonons."""
    step_dir = os.path.join(_pd(slug), '07_phonons')
    fname = 'phonon_band' if ptype == 'band' else 'phonon_dos'
    for ext, mime in [('png', 'image/png'), ('svg', 'image/svg+xml')]:
        p = os.path.join(step_dir, f'{fname}.{ext}')
        if os.path.exists(p):
            return send_file(p, mimetype=mime)
    return '', 404


@app.route('/api/phonon_plot_pdf/<slug>/<ptype>')
def api_phonon_plot_pdf(slug, ptype):
    """Serve phonon plot PDF for download."""
    step_dir = os.path.join(_pd(slug), '07_phonons')
    fname = 'phonon_band' if ptype == 'band' else 'phonon_dos'
    p = os.path.join(step_dir, f'{fname}.pdf')
    if os.path.exists(p):
        return send_file(p, mimetype='application/pdf', as_attachment=True,
                         download_name=f'{fname}.pdf')
    # Also check with phonopy default names
    alt = os.path.join(step_dir, 'band.pdf' if ptype == 'band' else 'mesh.pdf')
    if os.path.exists(alt):
        return send_file(alt, mimetype='application/pdf', as_attachment=True,
                         download_name=f'{fname}.pdf')
    return jsonify(error='PDF not found — run 07_phonons first'), 404


def _convergence_subdirs(base_dir, dtype):
    """Return sorted (label, path) pairs for subdirs that have a completed OUTCAR."""
    if not os.path.isdir(base_dir):
        return []
    entries = []
    for name in os.listdir(base_dir):
        path   = os.path.join(base_dir, name)
        outcar = os.path.join(path, 'OUTCAR')
        if os.path.isdir(path) and os.path.isfile(outcar):
            # Strip leading word prefix (e.g. "encut_300" → "300", "kpoints_4x4x4" → "4x4x4")
            label = re.sub(r'^[A-Za-z]+_', '', name)
            entries.append((label, path))

    def _sort_key(item):
        label = item[0]
        if dtype == 'encut':
            try: return int(label)
            except: return 0
        else:                           # kpoints  "NxNxN"
            try: return int(label.split('x')[0])
            except: return 0

    return sorted(entries, key=_sort_key)


def _make_convergence_plot(slug, dtype, ptype):
    """Build and return a matplotlib Figure for one convergence plot type.

    dtype : 'encut' | 'kpoints'
    ptype : 'energy' | 'pressure' | 'forces' | 'eigenvalues'
    Returns None if no data are available.
    """
    from outcar_parser import (parse_energy, parse_fermi_energy,
                                parse_pressure_diagonal,
                                parse_forces_first_atom,
                                parse_eigenvalues_near_fermi,
                                parse_eigenvalues_by_band)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base_dir = os.path.join(_pd(slug), '00_convergence', dtype)
    entries  = _convergence_subdirs(base_dir, dtype)
    if not entries:
        return None

    labels = [e[0] for e in entries]
    xs     = list(range(len(labels)))
    xlabel = 'ENCUT (eV)' if dtype == 'encut' else 'K-mesh'
    titles = {'energy':      'Total Energy',
              'pressure':    'Pressure Diagonal',
              'forces':      'Forces on Atom 1',
              'eigenvalues': 'Eigenvalues near Ef'}

    fig, ax = plt.subplots(figsize=(5, 3.5))

    nan = float('nan')

    def _read_outcar(path):
        try:
            return Path(os.path.join(path, 'OUTCAR')).read_text(errors='replace')
        except OSError:
            return ''

    if ptype == 'energy':
        ys = [parse_energy(_read_outcar(p)) for _, p in entries]
        ys = [y if y is not None else nan for y in ys]
        ax.plot(xs, ys, 'o-', color='#7c3aed', lw=1.5, ms=5)
        ax.set_ylabel('Total energy (eV)', fontsize=10)
        if all(y != y for y in ys):   # all NaN
            ax.text(0.5, 0.5, 'No data in OUTCAR', ha='center', va='center',
                    fontsize=9, color='#94a3b8', transform=ax.transAxes)

    elif ptype == 'pressure':
        pxx, pyy, pzz = [], [], []
        for _, path in entries:
            p = parse_pressure_diagonal(_read_outcar(path))
            pxx.append(p[0] if p else nan)
            pyy.append(p[1] if p else nan)
            pzz.append(p[2] if p else nan)
        ax.plot(xs, pxx, 'o-', color='#3b82f6', lw=1.5, ms=4, label='Pxx')
        ax.plot(xs, pyy, 's-', color='#10b981', lw=1.5, ms=4, label='Pyy')
        ax.plot(xs, pzz, '^-', color='#f59e0b', lw=1.5, ms=4, label='Pzz')
        ax.legend(fontsize=9)
        ax.set_ylabel('Pressure (kBar)', fontsize=10)
        if all(v != v for v in pxx):
            ax.text(0.5, 0.5, 'Stress not in OUTCAR\n(add ISIF=2 to INCAR)',
                    ha='center', va='center', fontsize=9, color='#94a3b8',
                    transform=ax.transAxes)

    elif ptype == 'forces':
        fx, fy, fz = [], [], []
        for _, path in entries:
            f = parse_forces_first_atom(_read_outcar(path))
            fx.append(f[0] if f else nan)
            fy.append(f[1] if f else nan)
            fz.append(f[2] if f else nan)
        ax.plot(xs, fx, 'o-', color='#ef4444', lw=1.5, ms=4, label='Fx')
        ax.plot(xs, fy, 's-', color='#10b981', lw=1.5, ms=4, label='Fy')
        ax.plot(xs, fz, '^-', color='#3b82f6', lw=1.5, ms=4, label='Fz')
        ax.legend(fontsize=9)
        ax.set_ylabel('Force on atom 1 (eV/Å)', fontsize=10)
        if all(v != v for v in fx):
            ax.text(0.5, 0.5, 'Forces not found in OUTCAR',
                    ha='center', va='center', fontsize=9, color='#94a3b8',
                    transform=ax.transAxes)

    elif ptype == 'eigenvalues':
        # Read all datasets keyed by band index
        all_by_band = []
        for _, path in entries:
            all_by_band.append(parse_eigenvalues_by_band(_read_outcar(path)))

        # Determine reference band set from the first dataset (within ±2 eV)
        ref_bands = sorted(
            idx for idx, e in all_by_band[0].items() if abs(e) <= 2.0
        ) if all_by_band else []

        # Connect each reference band across all x-points; extend beyond ±2 eV
        for band_idx in ref_bands:
            ys = [d.get(band_idx, float('nan')) for d in all_by_band]
            ax.plot(xs, ys, '-o', color='#6366f1', lw=0.9, ms=5, alpha=0.6)

        ax.axhline(0, color='#ef4444', lw=0.8, ls='--', alpha=0.8)
        # Y-axis spans actual data range (may exceed ±2 if bands drift)
        all_vals = [all_by_band[i].get(b, float('nan'))
                    for b in ref_bands for i in range(len(all_by_band))]
        finite = [v for v in all_vals if v == v]  # drop NaN
        if finite:
            margin = 0.3
            ax.set_ylim(min(finite) - margin, max(finite) + margin)
        if not ref_bands:
            ax.text(0.5, 0.5, 'No eigenvalues found\n(E-fermi missing in OUTCAR?)',
                    ha='center', va='center', fontsize=9, color='#94a3b8',
                    transform=ax.transAxes)
        ax.set_ylabel('E - Ef (eV)', fontsize=10)

    else:
        plt.close(fig)
        return None

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0,
                       ha='right', fontsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(titles.get(ptype, ptype), fontsize=11)
    ax.grid(True, alpha=0.25, lw=0.5)
    # Disable scientific notation on y-axis — avoids MathText/pyparsing errors
    # when matplotlib renders tick labels like "1.5×10³" for large energy values.
    from matplotlib.ticker import ScalarFormatter
    fmt = ScalarFormatter(useOffset=False, useMathText=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    plt.tight_layout()
    return fig


def _placeholder_png(msg):
    """Return a small PNG with a centred message (for missing / error states)."""
    import io, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=10,
            color='#94a3b8', transform=ax.transAxes, wrap=True)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Response(buf.read(), mimetype='image/png',
                    headers={'Cache-Control': 'no-store'})


@app.route('/api/convergence_plot/<slug>/<dtype>/<ptype>.png')
def api_convergence_plot_img(slug, dtype, ptype):
    """Return a PNG image for the requested convergence plot."""
    import io, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if dtype not in ('encut', 'kpoints') or \
       ptype not in ('energy', 'pressure', 'forces', 'eigenvalues'):
        return _placeholder_png('invalid type'), 400
    try:
        fig = _make_convergence_plot(slug, dtype, ptype)
    except Exception as exc:
        app.logger.exception('convergence_plot %s/%s/%s failed', slug, dtype, ptype)
        return _placeholder_png(f'Error: {exc}')
    if fig is None:
        return _placeholder_png('No data yet\nRun convergence first')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Response(buf.read(), mimetype='image/png',
                    headers={'Cache-Control': 'no-store'})


@app.route('/api/convergence_pdf/<slug>/<dtype>')
def api_convergence_pdf(slug, dtype):
    """Generate and serve convergence chart as a PDF using matplotlib."""
    fname = 'encut_convergence.dat' if dtype == 'encut' else 'kpoint_convergence.dat'
    dat   = os.path.join(_pd(slug), '00_convergence', dtype, fname)
    if not os.path.exists(dat):
        return jsonify(error='Data not found — run convergence first'), 404
    rows = []
    for line in Path(dat).read_text().splitlines():
        p = line.strip().split()
        if len(p) >= 2:
            try: rows.append((p[0], float(p[1])))
            except ValueError: pass
    if not rows:
        return jsonify(error='No data in file'), 404

    import tempfile, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    color  = '#7c3aed' if dtype == 'encut' else '#0ea5e9'
    xlabel = 'ENCUT (eV)' if dtype == 'encut' else 'K-mesh'
    title  = 'ENCUT Convergence' if dtype == 'encut' else 'K-point Convergence'

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(range(len(labels)), values, 'o-', color=color, lw=1.5, ms=5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30 if len(labels) > 5 else 0, ha='right')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Energy (eV)', fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    tmp = tempfile.mktemp(suffix='.pdf')
    fig.savefig(tmp, format='pdf', bbox_inches='tight')
    plt.close(fig)
    return send_file(tmp, mimetype='application/pdf', as_attachment=True,
                     download_name=f'{slug}_{dtype}_convergence.pdf')


@app.route('/api/profiles')
def api_profiles():
    """List available execution profiles from APP_DIR/profiles/*.json."""
    profiles_dir = os.path.join(APP_DIR, 'profiles')
    profiles = [{'id': 'default', 'name': 'Default (site.env)'}]
    if os.path.isdir(profiles_dir):
        for fname in sorted(os.listdir(profiles_dir)):
            if fname.endswith('.json'):
                pid = fname[:-5]
                try:
                    data = json.loads(Path(os.path.join(profiles_dir, fname)).read_text())
                    profiles.append({'id': pid, 'name': data.get('name', pid),
                                     'mpi_np': data.get('mpi_np', 0)})
                except Exception:
                    pass
    return jsonify(profiles=profiles)

@app.route('/api/potcar_variants', methods=['POST'])
def api_potcar_variants():
    """Return available POTCAR variant directories for a list of elements."""
    d = request.json or {}
    potcar_dir = d.get('potcar_dir', '').strip() or CONFIG['potcar_dir']
    elements   = d.get('elements', [])
    if not potcar_dir:
        return jsonify(error='No POTCAR directory configured'), 400
    if not os.path.isdir(potcar_dir):
        return jsonify(error=f'Directory not found: {potcar_dir}'), 404
    try:
        all_entries = sorted(os.listdir(potcar_dir))
    except Exception as e:
        return jsonify(error=str(e)), 500
    result = {}
    for el in elements:
        variants = [n for n in all_entries
                    if (n == el or n.startswith(el + '_'))
                    and os.path.isfile(os.path.join(potcar_dir, n, 'POTCAR'))]
        result[el] = variants if variants else [el]
    return jsonify(variants=result)

@app.route('/api/run_sumo/<slug>/<stype>')
def api_run_sumo(slug, stype):
    """Run sumo-bandplot or sumo-dosplot, streaming output."""
    if stype not in ('bands', 'dos'):
        return jsonify(error='type must be bands or dos'), 400
    pd_  = _pd(slug)
    ana  = os.path.join(pd_, 'analysis')
    os.makedirs(ana, exist_ok=True)
    base = _slug_base(slug)

    # Remove stale plots for this type
    if os.path.isdir(ana):
        for f in os.listdir(ana):
            if (stype == 'bands' and '_band.' in f) or \
               (stype == 'dos'   and '_dos.'  in f):
                os.remove(os.path.join(ana, f))

    tmp = os.path.join(pd_, f'_run_sumo_{stype}.sh')
    if os.path.exists(tmp): os.remove(tmp)   # never reuse a stale script
    with open(tmp, 'w') as f:
        f.write('#!/bin/bash\nset -e\n')
        if stype == 'bands':
            src = os.path.join(pd_, '03_bands')
            # Patch efermi in vasprun.xml from DOS (dense k-mesh) or SCF outcar
            for oc in [os.path.join(pd_, '04_dos', 'OUTCAR'),
                       os.path.join(pd_, '02_scf', 'OUTCAR')]:
                if os.path.exists(oc):
                    m = re.findall(r'E-fermi\s*:\s*([-\d.]+)',
                                   Path(oc).read_text(errors='replace'))
                    if m:
                        src_name = os.path.basename(os.path.dirname(oc))
                        f.write(f'echo "E_fermi = {m[-1]} eV  (from {src_name}/OUTCAR)"\n')
                        # sed-patch vasprun.xml so sumo picks up the correct EF
                        f.write(f'sed -i.efermi_bak '
                                f'"s|<i name=\\"efermi\\">.*</i>'
                                f'|<i name=\\"efermi\\">  {m[-1]}  </i>|" '
                                f'"{src}/vasprun.xml"\n')
                        break
            ymin, ymax, labels = _band_plot_opts(slug)
            labels_flag = f'--labels "{labels}"' if labels else ''
            f.write(f'cd "{src}"\n')
            f.write(f'sumo-bandplot --prefix "{base}" '
                    f'--ymin {ymin} --ymax {ymax} {labels_flag} '
                    f'--format png\n')
            f.write(f'sumo-bandplot --prefix "{base}" '
                    f'--ymin {ymin} --ymax {ymax} {labels_flag} '
                    f'--format pdf\n')
            f.write(f'mv *_band.* "{ana}/" 2>/dev/null || true\n')
            f.write('echo "Band plot (PNG + PDF) saved to analysis/"\n')
        else:
            src = os.path.join(pd_, '04_dos')
            orb_flag = ''
            proj_json = os.path.join(pd_, 'dos_proj.json')
            if os.path.exists(proj_json):
                try:
                    proj_data = json.loads(Path(proj_json).read_text())
                    om = defaultdict(list)
                    for p in proj_data:
                        for orb in p.get('orbitals', []):
                            om[p['element']].append(orb)
                    sumo_orb = '; '.join(f"{el} {' '.join(orbs)}"
                                       for el, orbs in om.items())
                    if sumo_orb:
                        orb_flag = f'--orbitals "{sumo_orb}"'
                except Exception:
                    pass
            f.write(f'cd "{src}"\n')
            f.write(f'sumo-dosplot --prefix "{base}_total" '
                    f'--xmin {CONFIG["dos_xmin"]} --xmax {CONFIG["dos_xmax"]} --format png\n')
            f.write(f'sumo-dosplot --prefix "{base}_total" '
                    f'--xmin {CONFIG["dos_xmin"]} --xmax {CONFIG["dos_xmax"]} --format pdf\n')
            if orb_flag:
                f.write(f'sumo-dosplot --prefix "{base}_proj" {orb_flag} '
                        f'--xmin {CONFIG["dos_xmin"]} --xmax {CONFIG["dos_xmax"]} --format png\n')
                f.write(f'sumo-dosplot --prefix "{base}_proj" {orb_flag} '
                        f'--xmin {CONFIG["dos_xmin"]} --xmax {CONFIG["dos_xmax"]} --format pdf\n')
            f.write(f'mv *_dos.* "{ana}/" 2>/dev/null || true\n')
            f.write('echo "DOS plots (PNG + PDF) saved to analysis/"\n')
    os.chmod(tmp, 0o755)
    return jsonify(job_key=_launch(tmp, slug, f'sumo_{stype}'))


@app.route('/api/summary/<slug>')
def api_summary(slug):
    pd, out = _pd(slug), {}
    for step in _steps(slug):
        if step.startswith('00'): continue
        outcar = os.path.join(pd, step, 'OUTCAR')
        if not os.path.exists(outcar): continue
        txt  = Path(outcar).read_text(errors='replace')
        info = {}
        energies = re.findall(r'energy\s+without entropy=\s+([-\d.]+)', txt)
        if energies: info['energy'] = energies[-1]
        fermi = re.findall(r'E-fermi\s*:\s*([-\d.]+)', txt)
        if fermi: info['efermi'] = fermi[-1]
        info['converged'] = 'reached required accuracy' in txt
        out[step] = info
    return jsonify(out)

# Files shown in the modal — INCAR/KPOINTS/POSCAR are editable; rest read-only
INPUT_FILES    = ['INCAR', 'KPOINTS', 'POSCAR']
READONLY_FILES = {'OUTCAR', 'OSZICAR', 'vasp.out'}

def _step_dir(slug, step):
    """Resolve the filesystem directory for a step; '_root' maps to project root."""
    if step == '_root':
        return _pd(slug)
    return os.path.join(_pd(slug), step)

ROOT_INPUT_FILES = ['POSCAR', 'instructions.txt', 'project.json']

@app.route('/api/files/<slug>/<step>')
def api_files(slug, step):
    """Return list of files available in a step directory."""
    step_dir = _step_dir(slug, step)
    if not os.path.isdir(step_dir):
        return jsonify(error='Step not found'), 404
    files = []
    if step == '_root':
        for name in ROOT_INPUT_FILES:
            p = os.path.join(step_dir, name)
            if os.path.isfile(p):
                files.append({'name': name, 'readonly': False})
    else:
        # Editable input files first
        for name in INPUT_FILES:
            p = os.path.join(step_dir, name)
            if os.path.isfile(p) and not os.path.islink(p):
                files.append({'name': name, 'readonly': False})
        # Output / log files (read-only)
        for name in ['OUTCAR', 'OSZICAR', 'vasp.out']:
            p = os.path.join(step_dir, name)
            if os.path.isfile(p):
                files.append({'name': name, 'readonly': True})
        # Shell scripts (read-only)
        for name in sorted(os.listdir(step_dir)):
            if name.endswith('.sh'):
                files.append({'name': name, 'readonly': True})
    return jsonify(files=files)

@app.route('/api/file/<slug>/<step>/<filename>', methods=['GET', 'POST'])
def api_file(slug, step, filename):
    """GET: read a file (last 500 lines for large files). POST {content}: write it back."""
    if not re.match(r'^[\w\.\-]+$', filename):
        return jsonify(error='Invalid filename'), 400
    path = os.path.join(_step_dir(slug, step), filename)
    if not os.path.isfile(path):
        return jsonify(error='File not found'), 404

    is_readonly = filename in READONLY_FILES or filename.endswith('.sh')

    if request.method == 'GET':
        txt   = Path(path).read_text(errors='replace')
        lines = txt.splitlines()
        truncated = len(lines) > 500
        content   = '\n'.join(lines[-500:]) if truncated else txt
        header    = f'[... {len(lines)} lines total — showing last 500 ...]\n\n' if truncated else ''
        return jsonify(content=header+content, readonly=is_readonly,
                       total_lines=len(lines), truncated=truncated)

    # POST — save (only allowed for editable files)
    if is_readonly:
        return jsonify(error='This file is read-only'), 403
    content = (request.json or {}).get('content', '')
    bak = path + '.bak'
    if not os.path.exists(bak):
        shutil.copy(path, bak)
    Path(path).write_text(content)
    return jsonify(ok=True)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VASP Workflow GUI</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#f0f4f8;--card:#fff;--sidebar:#1a2233;
  --accent:#4f8ef7;--accent2:#7c3aed;
  --ok:#22c55e;--warn:#f59e0b;--err:#ef4444;--run:#818cf8;
  --text:#1e293b;--sub:#64748b;--border:#dde3ec;
  --term:#0d1117;--term-fg:#c9d1d9;--r:8px;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
     background:var(--bg);color:var(--text);font-size:14px;display:flex;flex-direction:column;min-height:100vh;}
header{background:var(--sidebar);color:#e2e8f0;padding:0 24px;height:52px;
       display:flex;align-items:center;gap:12px;flex-shrink:0;}
header h1{font-size:16px;font-weight:600;letter-spacing:.02em;}
header .sub{font-size:11px;color:#94a3b8;margin-left:auto;}
nav{background:white;border-bottom:1px solid var(--border);padding:0 24px;display:flex;gap:2px;}
.tab{padding:12px 18px;cursor:pointer;font-size:13px;font-weight:500;
     color:var(--sub);border-bottom:2px solid transparent;transition:all .15s;}
.tab:hover{color:var(--text);}
.tab.active{color:var(--accent);border-bottom-color:var(--accent);}
main{flex:1;padding:20px 24px;max-width:1120px;width:100%;}
.panel{display:none;}
.panel.active{display:block;}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
      padding:16px 20px;margin-bottom:12px;}
.card-title{font-size:11px;font-weight:700;margin-bottom:12px;color:var(--sub);
            text-transform:uppercase;letter-spacing:.07em;}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
.f{display:flex;flex-direction:column;gap:4px;}
.f label{font-size:11px;font-weight:600;color:var(--sub);text-transform:uppercase;letter-spacing:.04em;}
.f input,.f select,.f textarea{padding:7px 10px;border:1px solid var(--border);border-radius:6px;
  font-size:13px;font-family:inherit;background:white;color:var(--text);transition:border-color .15s;}
.f input:focus,.f select:focus,.f textarea:focus{outline:none;border-color:var(--accent);}
.f textarea{resize:vertical;font-family:"SF Mono",Monaco,monospace;font-size:12px;}
.ck{display:inline-flex;align-items:center;gap:5px;font-size:13px;cursor:pointer;user-select:none;}
.ck input{width:14px;height:14px;cursor:pointer;accent-color:var(--accent);}
.checks{display:flex;flex-wrap:wrap;gap:12px;}
.btn{padding:7px 16px;border-radius:6px;font-size:13px;font-weight:500;cursor:pointer;
     border:none;display:inline-flex;align-items:center;gap:5px;transition:all .15s;}
.btn-primary{background:var(--accent);color:white;}
.btn-primary:hover{filter:brightness(1.1);}
.btn-green{background:#16a34a;color:white;}
.btn-green:hover{background:#15803d;}
.btn-ghost{background:var(--bg);border:1px solid var(--border);color:var(--text);}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent);}
.btn-red{background:#fee2e2;border:1px solid #fca5a5;color:#991b1b;}
.btn-del{background:#f3f0ff;border:1px solid #c4b5fd;color:#5b21b6;font-size:11px;padding:2px 6px;}
.btn-sm{padding:4px 10px;font-size:12px;}
.btn:disabled{opacity:.45;cursor:not-allowed;pointer-events:none;}
.badge{display:inline-flex;align-items:center;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600;}
.b-ready{background:#f1f5f9;color:#94a3b8;}
.b-running{background:#ede9fe;color:#7c3aed;}
.b-done{background:#dcfce7;color:#16a34a;}
.b-error{background:#fee2e2;color:#dc2626;}
.b-ran{background:#fef9c3;color:#a16207;}
.step-row{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
          padding:10px 14px;display:flex;align-items:center;gap:9px;margin-bottom:5px;}
.step-icon{width:26px;height:26px;border-radius:5px;background:var(--bg);
           display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:var(--sub);}
.step-name{flex:1;font-weight:500;font-size:13px;}
.term{background:var(--term);color:var(--term-fg);border-radius:var(--r);
      padding:10px 12px;font-family:"SF Mono",Monaco,monospace;font-size:12px;
      line-height:1.6;height:240px;overflow-y:auto;white-space:pre-wrap;word-break:break-all;margin-top:6px;}
.term .ok{color:#4ade80;}.term .warn{color:#fbbf24;}.term .err{color:#f87171;}.term .info{color:#60a5fa;}
.phase{border-left:3px solid var(--accent);padding-left:14px;margin-bottom:18px;}
.phase-title{font-size:11px;font-weight:700;color:var(--accent);text-transform:uppercase;
             letter-spacing:.07em;margin-bottom:10px;}
.plot-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;}
.plot-card{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:12px;text-align:center;}
.plot-card h4{font-size:12px;font-weight:600;margin-bottom:8px;color:var(--sub);}
.plot-card img{max-width:100%;border-radius:4px;cursor:pointer;}
.no-plot{color:var(--sub);font-size:12px;padding:28px 0;}
.tbl{width:100%;border-collapse:collapse;font-size:13px;}
.tbl th{text-align:left;padding:6px 10px;background:var(--bg);font-size:11px;
        text-transform:uppercase;color:var(--sub);letter-spacing:.04em;}
.tbl td{padding:7px 10px;border-top:1px solid var(--border);}
.tbl .ok{color:#16a34a;font-weight:600;}.tbl .fail{color:#dc2626;}
.alert{padding:9px 13px;border-radius:6px;font-size:13px;margin-bottom:10px;}
.alert-info{background:#dbeafe;color:#1d4ed8;}
.alert-error{background:#fee2e2;color:#991b1b;border:1px solid #fca5a5;}
.sep{height:1px;background:var(--border);margin:14px 0;}

/* mode toggle */
.mode-row{display:flex;gap:8px;margin-bottom:14px;}
.mode-btn{padding:7px 14px;border-radius:6px;border:2px solid var(--border);
          background:var(--bg);color:var(--sub);font-size:12px;font-weight:600;cursor:pointer;}
.mode-btn.active{border-color:var(--accent);background:#eff6ff;color:var(--accent);}

/* task section */
.task-block{border:1px solid var(--border);border-radius:var(--r);padding:12px 14px;margin-bottom:8px;}
.task-header{display:flex;align-items:center;gap:10px;margin-bottom:0;}
.task-header.expanded{margin-bottom:12px;}
.task-body{margin-top:12px;padding-top:10px;border-top:1px solid var(--border);}

/* U rows, proj rows */
.u-row,.proj-row{display:grid;gap:8px;align-items:end;margin-bottom:8px;}
.u-row{grid-template-columns:1fr 1fr 100px auto;}
.proj-row{grid-template-columns:80px auto auto;}
.orb-checks{display:flex;gap:8px;flex-wrap:wrap;}

.hidden{display:none!important;}
/* potcar variants */
.potcar-row{display:flex;align-items:center;gap:8px;margin-bottom:6px;}
.potcar-el{font-size:12px;font-weight:700;min-width:28px;color:var(--text);}
/* sumo log in results */
.sumo-log{height:110px;margin-bottom:8px;}

/* ── file editor modal ── */
.modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:100;
                display:flex;align-items:center;justify-content:center;}
.modal{background:var(--card);border-radius:10px;width:min(820px,95vw);max-height:90vh;
       display:flex;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.3);}
.modal-header{display:flex;align-items:center;gap:10px;padding:14px 18px;
              border-bottom:1px solid var(--border);}
.modal-header h3{font-size:14px;font-weight:600;flex:1;}
.file-tabs{display:flex;gap:2px;padding:8px 18px 0;border-bottom:1px solid var(--border);}
.file-tab{padding:6px 12px;font-size:12px;font-weight:500;cursor:pointer;
          border-radius:4px 4px 0 0;color:var(--sub);border:1px solid transparent;
          border-bottom:none;transition:all .12s;}
.file-tab:hover{color:var(--text);}
.file-tab.active{background:var(--card);border-color:var(--border);
                 border-bottom-color:var(--card);color:var(--accent);margin-bottom:-1px;}
.modal-body{flex:1;overflow:hidden;display:flex;flex-direction:column;padding:14px 18px;}
.file-editor{flex:1;width:100%;min-height:340px;max-height:50vh;
             font-family:"SF Mono",Monaco,Consolas,monospace;font-size:12px;
             line-height:1.6;padding:10px;border:1px solid var(--border);
             border-radius:6px;resize:vertical;color:var(--text);background:var(--bg);}
.file-editor:focus{outline:none;border-color:var(--accent);}
.modal-footer{display:flex;align-items:center;gap:8px;padding:12px 18px;
              border-top:1px solid var(--border);}
.save-msg{font-size:12px;color:var(--ok);flex:1;}</style>
</style>
</head>
<body>

<header>
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7dd3fc" stroke-width="2">
    <polygon points="12 2 22 8.5 22 15.5 12 22 2 15.5 2 8.5"/>
    <line x1="12" y1="22" x2="12" y2="15.5"/>
    <polyline points="22 8.5 12 15.5 2 8.5"/>
  </svg>
  <h1>VASP Workflow GUI</h1>
  <span id="proj-label" class="sub"></span>
</header>

<nav>
  <div class="tab active" id="t-setup"    onclick="tab('setup',   this)">⚙ Setup</div>
  <div class="tab"        id="t-workflow" onclick="tab('workflow',this)">▶ Workflow</div>
  <div class="tab"        id="t-results"  onclick="tab('results', this)">📊 Results</div>
</nav>

<div id="resume-bar" style="background:#f8fafc;border-bottom:1px solid var(--border);
     padding:7px 24px;display:flex;align-items:center;gap:10px;font-size:13px;flex-wrap:wrap;">
  <span style="color:var(--sub);font-weight:600;">Resume project:</span>
  <select id="resume-sel" style="padding:4px 8px;border:1px solid var(--border);border-radius:6px;
          font-size:13px;background:white;color:var(--text);min-width:220px;">
    <option value="">— select existing project —</option>
  </select>
  <button class="btn btn-primary btn-sm" onclick="loadProjectSettings()"
          title="Populate Setup page with this project's settings for editing">✏ Edit &amp; Regenerate</button>
  <button class="btn btn-ghost btn-sm" onclick="resumeProject()"
          title="Jump directly to Workflow page without regenerating">→ Workflow</button>
  <span id="resume-msg" style="font-size:12px;color:var(--sub);"></span>
</div>

<main>
<!-- ════════ SETUP ════════ -->
<div id="p-setup" class="panel active">
<div id="alert-top"></div>

<!-- System Setup -->
<div class="card" id="system-setup-card">
  <div style="display:flex;align-items:center;justify-content:space-between;cursor:pointer;"
       onclick="toggleSystemSetup()">
    <div class="card-title" style="margin:0;">⚙ System Setup
      <span id="system-setup-badge" style="font-size:11px;font-weight:400;margin-left:8px;"></span>
    </div>
    <span id="system-setup-chevron" style="font-size:16px;">▼</span>
  </div>
  <div id="system-setup-body" style="margin-top:14px;">
    <div style="font-size:12px;color:var(--sub);margin-bottom:12px;">
      All defaults are stored in <code>settings.json</code> in your working directory. Fill in once and click <strong>Save Settings</strong>.
    </div>

    <!-- Projects directory -->
    <div style="font-size:13px;font-weight:600;color:var(--fg);margin-bottom:8px;">Projects Directory</div>
    <div class="f" style="margin-bottom:14px;">
      <label>Folder where calculation projects are stored</label>
      <input id="cfg_projects_dir" placeholder="e.g. /home/user/vasp_runs  or  ./  (relative to launch dir)"
             onblur="populateResumeList()">
      <div style="font-size:11px;color:var(--sub);margin-top:3px;">
        All project sub-folders live here. Change this to point to a different disk or HPC scratch area.
      </div>
    </div>

    <!-- VASP paths -->
    <div style="font-size:13px;font-weight:600;color:var(--fg);margin-bottom:8px;">VASP Paths</div>
    <div class="g2" style="margin-bottom:14px;">
      <div class="f"><label>VASP standard binary <span style="font-weight:400;color:var(--sub);">(collinear)</span></label>
        <input id="cfg_vasp_std" placeholder="e.g. ~/bin/vasp_std or /opt/vasp/bin/vasp_std"></div>
      <div class="f"><label>VASP non-collinear binary <span style="font-weight:400;color:var(--sub);">(SOC)</span></label>
        <input id="cfg_vasp_ncl" placeholder="e.g. ~/bin/vasp_ncl"></div>
      <div class="f"><label>VASP gamma-only binary <span style="font-weight:400;color:var(--sub);">(optional)</span></label>
        <input id="cfg_vasp_gam" placeholder="e.g. ~/bin/vasp_gam"></div>
      <div class="f"><label>Wannier90 binary</label>
        <input id="cfg_wannier90_x" placeholder="wannier90.x"></div>
      <div class="f" style="grid-column:1/-1;"><label>POTCAR library directory</label>
        <input id="cfg_potcar_dir" placeholder="e.g. /opt/vasp/potpaw_PBE.54"
               oninput="document.getElementById('potcar_dir').value=this.value; onPotcarDir()">
        <div style="font-size:11px;color:var(--sub);margin-top:3px;">
          Folder that contains element sub-folders (e.g. <code>Ga/</code>, <code>As/</code> …)
        </div></div>
    </div>

    <!-- Workstation MPI -->
    <div style="font-size:13px;font-weight:600;color:var(--fg);margin-bottom:8px;">Workstation MPI</div>
    <div class="g2" style="margin-bottom:14px;">
      <div class="f"><label>MPI launch command</label>
        <input id="cfg_mpi_launch" placeholder="e.g. mpirun -np">
        <div style="font-size:11px;color:var(--sub);margin-top:3px;">Command used before the VASP binary (workstation).</div></div>
    </div>

    <!-- SLURM settings -->
    <div>
      <div style="font-size:13px;font-weight:600;color:var(--fg);margin-bottom:8px;">SLURM Settings</div>
      <div class="g2" style="margin-bottom:14px;">
        <div class="f"><label>Partition</label>
          <input id="cfg_slurm_partition" placeholder="e.g. standard"></div>
        <div class="f"><label>Account <span style="font-weight:400;color:var(--sub);">(optional)</span></label>
          <input id="cfg_slurm_account" placeholder="your-project-account"></div>
        <div class="f"><label>Nodes</label>
          <input id="cfg_slurm_nodes" type="number" min="1" placeholder="2"></div>
        <div class="f"><label>Tasks per node</label>
          <input id="cfg_slurm_ntasks_per_node" type="number" min="1" placeholder="64"></div>
        <div class="f"><label>Wall time</label>
          <input id="cfg_slurm_time" placeholder="12:00:00"></div>
        <div class="f"><label>MPI command <span style="font-weight:400;color:var(--sub);">(usually srun)</span></label>
          <input id="cfg_slurm_mpi_cmd" placeholder="srun"></div>
      </div>
    </div>

    <!-- Physics defaults -->
    <div style="font-size:13px;font-weight:600;color:var(--fg);margin-bottom:8px;">Calculation Defaults</div>
    <div class="g2" style="margin-bottom:14px;">
      <div class="f"><label>K-path for bands</label>
        <input id="cfg_kpath" placeholder="e.g. G-M-K-G"></div>
      <div class="f"><label>K-points along path</label>
        <input id="cfg_nkpts_bands" type="number" min="10" placeholder="60"></div>
      <div class="f"><label>NSW (ionic steps)</label>
        <input id="cfg_nsw" type="number" min="1" placeholder="100"></div>
      <div class="f"><label>EDIFFG (eV/Å)</label>
        <input id="cfg_ediffg" placeholder="-0.01"></div>
      <div class="f"><label>Default ENCUT (eV)</label>
        <input id="cfg_encut_manual" type="number" min="100" placeholder="520"></div>
      <div class="f"><label>Band plot energy window (eV)</label>
        <div style="display:flex;gap:6px;">
          <input id="cfg_band_ymin" type="number" placeholder="-4" style="flex:1;">
          <span style="align-self:center;color:var(--sub);">to</span>
          <input id="cfg_band_ymax" type="number" placeholder="4" style="flex:1;">
        </div></div>
      <div class="f"><label>DOS plot energy window (eV)</label>
        <div style="display:flex;gap:6px;">
          <input id="cfg_dos_xmin" type="number" placeholder="-6" style="flex:1;">
          <span style="align-self:center;color:var(--sub);">to</span>
          <input id="cfg_dos_xmax" type="number" placeholder="6" style="flex:1;">
        </div></div>
    </div>

    <div style="display:flex;align-items:center;gap:10px;margin-top:4px;">
      <button class="btn btn-primary btn-sm" onclick="saveSystemConfig()">💾 Save Settings</button>
      <span id="cfg-save-msg" style="font-size:12px;color:var(--ok);"></span>
    </div>
  </div>
</div>

<!-- Project -->
<div class="card">
  <div class="card-title">Project</div>
  <div class="g3">
    <div class="f"><label>Project Name</label>
      <input id="project_name" value="graphene test">
      <div style="font-size:11px;color:var(--sub);margin-top:3px;">
        Each unique name creates a separate project folder. Change the name to start a new project without overwriting the previous one.
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-top:6px;">
        <button class="btn btn-ghost btn-sm" id="btn-save-proj" onclick="saveProjectSettings()"
                title="Save settings now — project appears in the resume dropdown without generating">
          💾 Save Project Settings
        </button>
        <span id="save-proj-msg" style="font-size:12px;color:var(--sub);"></span>
      </div>
    </div>
    <div class="f"><label>Execution Profile</label>
      <select id="profile" onchange="onProfileChange()">
        <option value="default">Workstation</option>
      </select>
      <div style="font-size:11px;color:var(--sub);margin-top:3px;">
        Workstation runs VASP locally; SLURM submits a batch job.
        MPI cores below are set automatically when you switch profiles.
      </div></div>
    <div class="f"><label>MPI Cores</label>
      <input id="mpi_np" type="number" value="16" min="1">
      <div style="font-size:11px;color:var(--sub);margin-top:3px;">
        Auto-filled from System Setup when you switch profiles.
      </div></div>
    <div class="f"><label>Description (optional)</label>
      <input id="description" placeholder="brief note for records"></div>
  </div>
</div>

<!-- POSCAR -->
<div class="card">
  <div class="card-title">POSCAR</div>
  <div class="f">
    <label>Paste POSCAR — or load from file</label>
    <textarea id="poscar" rows="7" oninput="onPoscar()"
      placeholder="graphene&#10;1.0&#10;  2.46  0.00  0.00&#10; -1.23  2.13  0.00&#10;  0.00  0.00 20.00&#10;C&#10;2&#10;Direct&#10;..."></textarea>
  </div>
  <div style="display:flex;align-items:center;gap:10px;margin-top:8px;">
    <label class="btn btn-ghost btn-sm" style="cursor:pointer;">
      📁 Load file <input type="file" id="poscar_file" style="display:none" onchange="loadFile(this)">
    </label>
    <span id="poscar-elements" style="font-size:12px;color:var(--sub);"></span>
  </div>
</div>

<!-- POTCAR -->
<div class="card">
  <div class="card-title">POTCAR</div>
  <div class="f" style="margin-bottom:10px;">
    <label>POTCAR Directory</label>
    <input id="potcar_dir" placeholder="uses $VASP_POTCAR_DIR if blank" oninput="onPotcarDir()">
  </div>
  <div id="potcar-status" style="font-size:12px;color:var(--sub);margin-bottom:6px;"></div>
  <div id="potcar-variants"></div>
</div>

<!-- DFT Method -->
<div class="card">
  <div class="card-title">DFT Method</div>
  <div class="g2" style="margin-bottom:12px;">
    <div class="f"><label>Functional</label>
      <select id="functional">
        <option>PBE</option>
        <option selected>PBEsol</option>
        <option>LDA</option>
        <option>AM05</option>
        <option>R2SCAN</option>
        <option>HSE06</option>
        <option>RVV10</option>
      </select>
    </div>
    <div class="f"><label>Spin / SOC</label>
      <select id="spin_mode">
        <option value="none">None (non-magnetic)</option>
        <option value="collinear">Collinear spin (ISPIN=2)</option>
        <option value="soc_z">SOC, magnetization ∥ z  (vasp_ncl)</option>
        <option value="soc_x">SOC, magnetization ∥ x  (vasp_ncl)</option>
        <option value="soc_y">SOC, magnetization ∥ y  (vasp_ncl)</option>
      </select>
    </div>
  </div>
  <div class="checks" style="margin-bottom:12px;">
    <label class="ck"><input type="checkbox" id="hexagonal" checked> Hexagonal BZ</label>
    <label class="ck"><input type="checkbox" id="is_2d"> 2D / slab (k<sub>z</sub>=1)</label>
    <label class="ck"><input type="checkbox" id="use_u" onchange="toggleU(this)"> GGA+U</label>
  </div>
  <!-- GGA+U rows -->
  <div id="u-section" class="hidden">
    <div style="font-size:11px;font-weight:600;color:var(--sub);margin-bottom:6px;">
      Element / orbital / U value (eV)
    </div>
    <div id="u-rows"></div>
    <button class="btn btn-ghost btn-sm" onclick="addURow()">+ Add element</button>
  </div>
</div>

<!-- Parameters -->
<div class="card">
  <div class="card-title">Calculation Parameters</div>
  <div class="mode-row">
    <button class="mode-btn active" id="mode-conv"   onclick="setMode('convergence')">📊 Run convergence tests first</button>
    <button class="mode-btn"        id="mode-manual" onclick="setMode('manual')">✏️ Set parameters manually</button>
  </div>

  <!-- Convergence mode -->
  <div id="conv-section">
    <div class="g2">
      <div class="f"><label>K-point meshes (comma-separated)</label>
        <input id="conv_kp" placeholder="6x6x3, 12x12x6, 18x18x9"></div>
      <div class="f"><label>ENCUT range (eV)</label>
        <input id="conv_encut" placeholder="300-900"></div>
    </div>
    <div style="font-size:11px;color:var(--sub);margin-top:6px;">
      After convergence tests complete, you will be prompted for your chosen values before running production calculations.
    </div>
  </div>

  <!-- Manual mode -->
  <div id="manual-section" class="hidden">
    <div class="g3">
      <div class="f"><label>ENCUT (eV)</label>
        <input id="manual_encut" value="520" placeholder="520"></div>
      <div class="f"><label>K-mesh (relax / SCF)</label>
        <input id="manual_kmesh_scf" placeholder="12x12x6"></div>
      <div class="f"><label>K-mesh for DOS (denser)</label>
        <input id="manual_kmesh_dos" placeholder="24x24x12"></div>
    </div>
  </div>
</div>

<!-- Tasks -->
<div class="card">
  <div class="card-title">Tasks</div>

  <!-- Relaxation -->
  <div class="task-block" id="tb-relax">
    <div class="task-header" id="th-relax">
      <label class="ck"><input type="checkbox" id="t_relax" checked onchange="toggleTask('relax',this)"> Structure Relaxation</label>
    </div>
    <div class="task-body" id="tbody-relax">
      <div class="g3">
        <div class="f"><label>Type (ISIF)</label>
          <select id="relax_type">
            <option value="full_relax">Full relax — ions + cell (ISIF=3)</option>
            <option value="ions_only">Ions only, fixed cell (ISIF=2)</option>
            <option value="fixed_vol">Fixed volume — shape+ions (ISIF=4)</option>
            <option value="fixed_shape">Fixed shape — volume only (ISIF=7)</option>
            <option value="vol_only">Volume only (ISIF=6)</option>
          </select>
        </div>
        <div class="f"><label>Max ionic steps (NSW)</label>
          <input id="nsw" type="number" value="100" min="1"></div>
        <div class="f"><label>Force threshold (EDIFFG, eV/Å)</label>
          <input id="ediffg" value="-0.01"></div>
      </div>
    </div>
  </div>

  <!-- SCF -->
  <div class="task-block" id="tb-scf">
    <div class="task-header">
      <label class="ck"><input type="checkbox" id="t_scf" checked> Self-consistent field (SCF)</label>
    </div>
  </div>

  <!-- Band structure -->
  <div class="task-block" id="tb-bands">
    <div class="task-header" id="th-bands">
      <label class="ck"><input type="checkbox" id="t_bands" checked onchange="toggleTask('bands',this)"> Band Structure</label>
    </div>
    <div class="task-body" id="tbody-bands">
      <div class="g3">
        <div class="f"><label>K-path</label>
          <input id="kpath" value="G-M-K-G" placeholder="G-M-K-G"></div>
        <div class="f"><label>K-points per segment</label>
          <input id="nkpts_bands" type="number" value="60" min="10"></div>
        <div class="f"><label>Energy window (eV, relative to EF)</label>
          <input id="ewin" value="-4 to 4" placeholder="-4 to 4"></div>
      </div>
      <div style="font-size:11px;color:var(--sub);margin-top:6px;">
        Use standard labels: Γ→G, Σ→S, Λ→L, Δ→D. Segment separator: −
      </div>
    </div>
  </div>

  <!-- DOS -->
  <div class="task-block" id="tb-dos">
    <div class="task-header" id="th-dos">
      <label class="ck"><input type="checkbox" id="t_dos" checked onchange="toggleTask('dos',this)"> Density of States (DOS)</label>
    </div>
    <div class="task-body" id="tbody-dos">
      <div style="font-size:11px;font-weight:600;color:var(--sub);margin-bottom:8px;">
        Orbital projections — add one row per element you want projected
      </div>
      <div id="proj-rows"></div>
      <button class="btn btn-ghost btn-sm" onclick="addProjRow()">+ Add projection</button>
      <div style="font-size:11px;color:var(--sub);margin-top:8px;">
        Elements are auto-detected from your POSCAR. Selecting no projections plots total DOS only.
      </div>
    </div>
  </div>

  <!-- DFPT -->
  <div class="task-block" id="tb-dfpt">
    <div class="task-header" id="th-dfpt">
      <label class="ck"><input type="checkbox" id="t_dfpt" onchange="toggleTask('dfpt',this)"> Born Charges &amp; Dielectric Constant (DFPT)</label>
    </div>
    <div class="task-body" id="tbody-dfpt" style="display:none">
      <div class="g3">
        <div class="f"><label>EDIFF (SCF convergence)</label>
          <input id="dfpt_ediff" value="1E-8" placeholder="1E-8"></div>
      </div>
      <div style="font-size:11px;color:var(--sub);margin-top:6px;">
        Runs VASP DFPT (IBRION=8, LEPSILON=.TRUE.) to compute Born effective charges and the macroscopic static dielectric tensor. Results are written to OUTCAR and extracted to a <code>BORN</code> file for phonon LO-TO splitting.
      </div>
    </div>
  </div>

  <!-- Phonons -->
  <div class="task-block" id="tb-phonons">
    <div class="task-header" id="th-phonons">
      <label class="ck"><input type="checkbox" id="t_phonons" onchange="toggleTask('phonons',this)"> Phonon Spectrum (phonopy)</label>
    </div>
    <div class="task-body" id="tbody-phonons" style="display:none">
      <div class="g3">
        <div class="f"><label>Supercell dimensions</label>
          <input id="phonons_dim" value="2 2 2" placeholder="2 2 2"></div>
        <div class="f"><label>DOS q-mesh</label>
          <input id="phonons_mesh" value="20 20 20" placeholder="20 20 20"></div>
        <div class="f"><label>Displacement (Å)</label>
          <input id="phonons_disp" type="number" value="0.01" step="0.001" min="0.001"></div>
      </div>
      <div class="g3" style="margin-top:8px;">
        <div class="f" style="grid-column:1/-1">
          <label>Q-path for band structure (phonopy format, leave blank to use k-path above)</label>
          <input id="phonons_band" placeholder="e.g.  0 0 0  0.5 0 0  0.5 0.5 0  0 0 0"></div>
      </div>
      <div style="margin-top:10px;">
        <label class="ck"><input type="checkbox" id="phonons_nac" checked>
          NAC correction (LO-TO splitting) — requires Born charges from DFPT step above</label>
      </div>
      <div style="font-size:11px;color:var(--sub);margin-top:6px;">
        Requires <code>phonopy</code> (conda install -c conda-forge phonopy). Generates displaced supercells, runs VASP forces on each, then computes phonon band structure and DOS. For 2D: use dim <code>2 2 1</code>.
      </div>
    </div>
  </div>

  <!-- Wannierization -->
  <div class="task-block" id="tb-wannier">
    <div class="task-header" id="th-wannier">
      <label class="ck"><input type="checkbox" id="t_wannier" onchange="toggleTask('wannier',this)"> Wannierization (Wannier90)</label>
    </div>
    <div class="task-body" id="tbody-wannier" style="display:none">
      <div class="g3">
        <div class="f"><label>Num. Wannier functions</label>
          <input id="wannier_num_wann" type="number" value="8" min="1"></div>
        <div class="f"><label>Projections (e.g. Mo:d, S:p)</label>
          <input id="wannier_proj" placeholder="Mo:d, S:p"></div>
        <div class="f"><label>Energy window (eV, e.g. -4:12)</label>
          <input id="wannier_ewin" placeholder="-4:12"></div>
      </div>
      <div style="font-size:11px;color:var(--sub);margin-top:6px;">
        Requires Wannier90 installed. VASP writes .mmn/.amn/.eig via LWANNIER90=.TRUE.;
        wannier90.win is auto-generated. Edit it before running 05_wannier.
      </div>
    </div>
  </div>

</div><!-- end tasks card -->

<button class="btn btn-primary" id="btn-gen" onclick="generate()" style="margin-top:4px;">
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
    <polygon points="5 3 19 12 5 21 5 3"/>
  </svg>
  Generate Workflow
</button>
</div>

<!-- ════════ WORKFLOW ════════ -->
<div id="p-workflow" class="panel">
  <div id="wf"></div>
</div>

<!-- ════════ RESULTS ════════ -->
<div id="p-results" class="panel">
  <div id="res"></div>
</div>

</main>

<script>
// ── config injected from Python ─────────────────────────────────────────────
const CFG = __CFG__;

// ── per-step file buttons (add/remove entries here to change the UI) ───────
const STEP_FILES = [
  { name: 'INCAR',    icon: '✏'  },
  { name: 'KPOINTS',  icon: '⊞'  },
  { name: 'POSCAR',   icon: '📐' },
  { name: 'OUTCAR',   icon: '📋' },
  { name: 'OSZICAR',  icon: '📊' },
  { name: 'vasp.out', icon: '📄' },
  { name: 'run.sh',   icon: '⚙'  },
];

// ── state ──────────────────────────────────────────────────────────────────
let PROJECT=null, HAS_CONV=false, ACTIVE_ES=null, PARAM_MODE='convergence';
let _charts={}, ELEMENTS=[];
let POTCAR_VARIANTS={}, POTCAR_CHOICES={}, _potcarTimer=null, _sumoES=null;

window.addEventListener('DOMContentLoaded', () => {
  const s = (id,v) => { const el=document.getElementById(id); if(el!=null&&v!=null&&v!=='') el.value=v; };
  // System Setup — projects directory
  s('cfg_projects_dir',         CFG.projects_dir);
  // System Setup — paths
  s('cfg_vasp_std',             CFG.vasp_std);
  s('cfg_vasp_ncl',             CFG.vasp_ncl);
  s('cfg_vasp_gam',             CFG.vasp_gam);
  s('cfg_wannier90_x',          CFG.wannier90_x);
  s('cfg_potcar_dir',           CFG.potcar_dir);
  // System Setup — workstation MPI
  s('cfg_mpi_launch',           CFG.mpi_launch);
  // System Setup — SLURM
  s('cfg_slurm_partition',      CFG.slurm_partition);
  s('cfg_slurm_account',        CFG.slurm_account);
  s('cfg_slurm_nodes',          CFG.slurm_nodes);
  s('cfg_slurm_ntasks_per_node',CFG.slurm_ntasks_per_node);
  s('cfg_slurm_time',           CFG.slurm_time);
  s('cfg_slurm_mpi_cmd',        CFG.slurm_mpi_cmd);
  // System Setup — physics defaults
  s('cfg_kpath',       CFG.kpath);
  s('cfg_nkpts_bands', CFG.nkpts_bands);
  s('cfg_nsw',         CFG.nsw);
  s('cfg_ediffg',      CFG.ediffg);
  s('cfg_encut_manual',CFG.encut_manual);
  s('cfg_band_ymin',   CFG.band_ymin);
  s('cfg_band_ymax',   CFG.band_ymax);
  s('cfg_dos_xmin',    CFG.dos_xmin);
  s('cfg_dos_xmax',    CFG.dos_xmax);
  // Calculation defaults (project form)
  s('potcar_dir',      CFG.potcar_dir);
  s('kpath',           CFG.kpath);
  s('nkpts_bands',     CFG.nkpts_bands);
  s('nsw',             CFG.nsw);
  s('ediffg',          CFG.ediffg);
  s('mpi_np',          CFG.mpi_np);
  s('manual_encut',    CFG.encut_manual);
  if(CFG.potcar_dir) fetchPotcarVariants();
  // First-run: expand System Setup and show banner
  if(CFG.first_run){
    document.getElementById('system-setup-body').style.display='block';
    document.getElementById('system-setup-chevron').textContent='▲';
    alertTop('👋 First time? Fill in System Setup above and click Save Settings before generating a workflow.','info');
  } else {
    document.getElementById('system-setup-body').style.display='none';
    document.getElementById('system-setup-chevron').textContent='▶';
    const badge=document.getElementById('system-setup-badge');
    if(badge) badge.textContent='(saved)';
  }
  populateResumeList();
  populateProfileList();
});


function toggleSystemSetup(){
  const body=document.getElementById('system-setup-body');
  const chev=document.getElementById('system-setup-chevron');
  const hidden=body.style.display==='none';
  body.style.display=hidden?'block':'none';
  chev.textContent=hidden?'▲':'▶';
}

async function saveSystemConfig(){
  const v  = id => document.getElementById(id)?.value || '';
  const vi = id => parseInt(document.getElementById(id)?.value) || 0;
  // Persist profile_mode and mpi_np from the Project section (authoritative)
  const profileSel = document.getElementById('profile');
  const curProfile = profileSel?.value || 'workstation';
  const curMpiNp   = parseInt(document.getElementById('mpi_np')?.value) || 1;
  const body={
    profile_mode:           curProfile === 'slurm' ? 'slurm' : 'workstation',
    mpi_np:                 curMpiNp,
    projects_dir:           v('cfg_projects_dir') || CFG.projects_dir,
    // paths
    vasp_std:               v('cfg_vasp_std'),
    vasp_ncl:               v('cfg_vasp_ncl'),
    vasp_gam:               v('cfg_vasp_gam'),
    wannier90_x:            v('cfg_wannier90_x') || 'wannier90.x',
    potcar_dir:             v('cfg_potcar_dir'),
    // workstation MPI (launch command only; core count is set per-project)
    mpi_launch:             v('cfg_mpi_launch') || 'mpirun -np',
    // SLURM
    slurm_partition:        v('cfg_slurm_partition') || 'standard',
    slurm_account:          v('cfg_slurm_account'),
    slurm_nodes:            vi('cfg_slurm_nodes') || 2,
    slurm_ntasks_per_node:  vi('cfg_slurm_ntasks_per_node') || 64,
    slurm_time:             v('cfg_slurm_time') || '12:00:00',
    slurm_mpi_cmd:          v('cfg_slurm_mpi_cmd') || 'srun',
    // physics defaults
    kpath:        v('cfg_kpath') || 'G-M-K-G',
    nkpts_bands:  vi('cfg_nkpts_bands') || 60,
    nsw:          vi('cfg_nsw') || 100,
    ediffg:       v('cfg_ediffg') || '-0.01',
    encut_manual: vi('cfg_encut_manual') || 520,
    band_ymin:    v('cfg_band_ymin') || '-4',
    band_ymax:    v('cfg_band_ymax') || '4',
    dos_xmin:     v('cfg_dos_xmin') || '-6',
    dos_xmax:     v('cfg_dos_xmax') || '6',
  };
  // Sync calculation fields to new defaults
  document.getElementById('mpi_np').value    = body.mpi_np;
  document.getElementById('potcar_dir').value = body.potcar_dir;
  const r=await post('/api/config', body);
  const msg=document.getElementById('cfg-save-msg');
  if(r.ok){
    msg.textContent='✓ Saved';
    document.getElementById('system-setup-badge').textContent='(saved)';
    setTimeout(()=>{msg.textContent='';},3000);
    if(CFG.potcar_dir !== body.potcar_dir){ CFG.potcar_dir=body.potcar_dir; fetchPotcarVariants(); }
    CFG.projects_dir = body.projects_dir;
    // Refresh project list with (possibly new) projects directory
    populateResumeList();
  } else {
    msg.style.color='var(--err)'; msg.textContent='✗ Save failed';
  }
}

async function populateProfileList(){
  try{
    const{profiles}=await(await fetch('/api/profiles')).json();
    const sel=document.getElementById('profile');
    sel.innerHTML='';
    profiles.forEach(p=>{
      const opt=document.createElement('option');
      opt.value=p.id; opt.textContent=p.name;
      sel.appendChild(opt);
    });
    // Auto-select based on saved profile_mode
    if(CFG.profile_mode==='slurm'){
      const slurmOpt=[...sel.options].find(o=>o.value==='slurm');
      if(slurmOpt){ slurmOpt.selected=true; onProfileChange(); }
    }
  }catch{}
}

function onProfileChange(){
  const pid = document.getElementById('profile')?.value || '';
  const mpiEl = document.getElementById('mpi_np');
  if(!mpiEl) return;
  if(pid === 'slurm'){
    // Auto-fill MPI cores = nodes × tasks-per-node from system config
    const np = (CFG.slurm_nodes||2) * (CFG.slurm_ntasks_per_node||64);
    mpiEl.value = np;
  } else {
    // Workstation: use saved workstation MPI core count
    if(CFG.mpi_np) mpiEl.value = CFG.mpi_np;
  }
}

async function populateResumeList(){
  try{
    // If the Projects Directory field has a value different from the saved
    // config, preview that directory without requiring a Save first.
    const dirEl=document.getElementById('cfg_projects_dir');
    const dir=(dirEl&&dirEl.value.trim()!==CFG.projects_dir)?dirEl.value.trim():'';
    const url='/api/projects'+(dir?'?dir='+encodeURIComponent(dir):'');
    const{projects}=await(await fetch(url)).json();
    const sel=document.getElementById('resume-sel');
    if(!sel) return;
    const prev=sel.value;
    sel.innerHTML='<option value="">— select existing project —</option>';
    projects.forEach(p=>{
      const opt=document.createElement('option');
      opt.value=p.slug;
      opt.textContent=`${p.slug}  (${p.steps.length} steps)`;
      opt.dataset.hasConv=p.has_convergence;
      opt.dataset.steps=JSON.stringify(p.steps);
      sel.appendChild(opt);
    });
    if(prev) sel.value=prev;  // restore selection if still present
  }catch{}
}

async function resumeProject(){
  // Jump directly to Workflow tab for an existing project (no regeneration)
  const sel=document.getElementById('resume-sel');
  const msg=document.getElementById('resume-msg');
  const slug=sel.value;
  if(!slug){if(msg) msg.textContent='Select a project first.'; return;}
  if(msg) msg.textContent='Loading…';
  try{
    const{projects}=await(await fetch('/api/projects')).json();
    const proj=projects.find(p=>p.slug===slug);
    if(!proj){if(msg) msg.textContent='Project not found on disk.'; return;}
    PROJECT=slug; HAS_CONV=proj.has_convergence;
    document.getElementById('proj-label').textContent='Project: '+PROJECT+'/';
    if(msg) msg.textContent='✓ loaded';
    buildWorkflow(proj.steps, proj.has_convergence);
    goTab('workflow');
    refreshStatus();
  }catch(e){if(msg) msg.textContent='Error: '+e.message;}
}

async function loadProjectSettings(){
  // Populate Setup page fields from saved project.json, stay on Setup tab
  const sel=document.getElementById('resume-sel');
  const msg=document.getElementById('resume-msg');
  const slug=sel.value;
  if(!slug){if(msg) msg.textContent='Select a project first.'; return;}
  if(msg) msg.textContent='Loading…';
  try{
    const r=await fetch(`/api/project_settings/${slug}`);
    if(!r.ok){if(msg) msg.textContent='Error loading settings.'; return;}
    const{settings,poscar}=await r.json();

    const s=(id,val)=>{const el=document.getElementById(id);if(el&&val!==undefined&&val!==null)el.value=val;};
    const c=(id,val)=>{const el=document.getElementById(id);if(el)el.checked=!!val;};

    // Basic fields
    if(poscar) s('poscar', poscar);
    s('project_name', settings.project_name||'');
    s('potcar_dir',   settings.potcar_dir);
    s('functional',   settings.functional);
    s('spin_mode',    settings.spin_mode);
    c('hexagonal',    settings.hexagonal);
    c('is_2d',        settings.is_2d);
    s('mpi_np',       settings.mpi_np);

    // GGA+U
    c('use_u', settings.use_u);
    toggleU(document.getElementById('use_u'));
    if(settings.use_u && settings.u_entries?.length){
      document.getElementById('u-rows').innerHTML='';
      settings.u_entries.forEach(e=>addURow(e.element||'',e.orbital||'d',e.U||'3.0'));
    }

    // Param mode
    if(settings.param_mode) setMode(settings.param_mode);
    s('conv_kp',         settings.conv_kp);
    s('conv_encut',      settings.conv_encut);
    s('manual_encut',    settings.manual_encut);
    s('manual_kmesh_scf',settings.manual_kmesh_scf);
    s('manual_kmesh_dos',settings.manual_kmesh_dos);

    // Task checkboxes + body visibility
    const tasks=['relax','scf','bands','dos','dfpt','phonons','wannier'];
    tasks.forEach(t=>{
      const cb=document.getElementById('t_'+t);
      if(cb){
        const on=!!settings[t];
        cb.checked=on;
        const body=document.getElementById('tbody-'+t);
        if(body) body.style.display=on?'block':'none';
      }
    });

    // Relaxation params
    s('relax_type', settings.relax_type);
    s('nsw',        settings.nsw);
    s('ediffg',     settings.ediffg);

    // Bands params
    s('kpath',       settings.kpath);
    s('nkpts_bands', settings.nkpts_bands);
    s('ewin',        settings.ewin || '-4 to 4');

    // DOS projections
    document.getElementById('proj-rows').innerHTML='';
    if(settings.dos_proj?.length){
      settings.dos_proj.forEach(p=>{
        addProjRow();
        const rows=document.querySelectorAll('#proj-rows .proj-row');
        const row=rows[rows.length-1];
        const sel2=row.querySelector('.proj-el-sel');
        if(sel2){
          // Add element option if not already present
          if(![...sel2.options].some(o=>o.value===p.element)){
            sel2.innerHTML+=`<option>${p.element}</option>`;
          }
          sel2.value=p.element;
        }
        row.querySelectorAll('[data-orb]').forEach(cb2=>{
          cb2.checked=p.orbitals?.includes(cb2.dataset.orb)||false;
        });
      });
    }

    // DFPT params
    s('dfpt_ediff', settings.dfpt_ediff);

    // Phonons params
    s('phonons_dim',  settings.phonons_dim);
    s('phonons_mesh', settings.phonons_mesh);
    s('phonons_disp', settings.phonons_disp);
    s('phonons_band', settings.phonons_band);
    c('phonons_nac',  settings.phonons_nac !== false);

    // Wannier params
    s('wannier_num_wann', settings.wannier_num_wann);
    s('wannier_proj',     settings.wannier_proj);
    s('wannier_ewin',     settings.wannier_ewin);

    if(poscar) onPoscar();
    if(settings.potcar_dir) fetchPotcarVariants();

    if(msg) msg.textContent='✓ settings restored — edit then click Generate Workflow';
    goTab('setup');
  }catch(e){if(msg) msg.textContent='Error: '+e.message;}
}

// ── tabs ───────────────────────────────────────────────────────────────────
function tab(name,el){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('p-'+name).classList.add('active');
  el.classList.add('active');
  if(name==='results') buildResults();
  if(name==='workflow'&&PROJECT) refreshStatus();
}
function goTab(name){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('p-'+name).classList.add('active');
  document.getElementById('t-'+name).classList.add('active');
  if(name==='results') buildResults();
}

// ── mode toggle ────────────────────────────────────────────────────────────
function setMode(m){
  PARAM_MODE=m;
  document.getElementById('mode-conv').classList.toggle('active',m==='convergence');
  document.getElementById('mode-manual').classList.toggle('active',m==='manual');
  document.getElementById('conv-section').classList.toggle('hidden',m!=='convergence');
  document.getElementById('manual-section').classList.toggle('hidden',m!=='manual');
}

// ── task toggling ──────────────────────────────────────────────────────────
function toggleTask(name,cb){
  const body=document.getElementById('tbody-'+name);
  if(body) body.style.display=cb.checked?'block':'none';
}

// ── POSCAR parsing ─────────────────────────────────────────────────────────
function parseElements(poscar){
  const lines=poscar.trim().split('\n');
  if(lines.length<6) return [];
  const p=lines[5].trim().split(/\s+/);
  if(p.every(x=>/^\d+$/.test(x))) return [];
  return p.filter(x=>/^[A-Z][a-z]?$/.test(x));
}
function onPoscar(){
  const els=parseElements(v('poscar'));
  ELEMENTS=els;
  const span=document.getElementById('poscar-elements');
  if(span) span.textContent=els.length?'Elements: '+els.join(', '):'';
  refreshProjRows();
  fetchPotcarVariants();
}
function loadFile(inp){
  const r=new FileReader();
  r.onload=e=>{document.getElementById('poscar').value=e.target.result; onPoscar();};
  r.readAsText(inp.files[0]);
}

// ── POTCAR directory + variant selection ───────────────────────────────────
function onPotcarDir(){
  clearTimeout(_potcarTimer);
  _potcarTimer = setTimeout(fetchPotcarVariants, 700);
}
async function fetchPotcarVariants(){
  const dir = v('potcar_dir');
  const status = document.getElementById('potcar-status');
  const varEl  = document.getElementById('potcar-variants');
  if(!dir){ if(status) status.textContent=''; if(varEl) varEl.innerHTML=''; return; }
  if(!ELEMENTS.length){ if(status) status.textContent='Load a POSCAR first.'; return; }
  if(status) status.textContent='Scanning…';
  try{
    const r = await post('/api/potcar_variants',{potcar_dir:dir, elements:ELEMENTS});
    const d = await r.json();
    if(!r.ok){ if(status) status.textContent='✗ '+d.error; return; }
    POTCAR_VARIANTS = d.variants||{};
    // Auto-select recommended variant if not already chosen
    for(const [el,vars] of Object.entries(POTCAR_VARIANTS)){
      if(!POTCAR_CHOICES[el] && vars.length) POTCAR_CHOICES[el] = vars[0];
    }
    if(status) status.textContent = '✓ POTCARs found';
    buildPotcarUI();
  }catch(e){ if(status) status.textContent='✗ '+e.message; }
}
function buildPotcarUI(){
  const el = document.getElementById('potcar-variants');
  if(!el) return;
  const entries = Object.entries(POTCAR_VARIANTS);
  if(!entries.length){ el.innerHTML=''; return; }
  let h='<div style="display:flex;flex-direction:column;gap:6px;">';
  for(const [elem,vars] of entries){
    const chosen = POTCAR_CHOICES[elem]||vars[0];
    h+=`<div class="potcar-row">
      <span class="potcar-el">${elem}</span>
      <div style="display:flex;gap:4px;flex-wrap:wrap;">`;
    vars.forEach(variant=>{
      const active = variant===chosen;
      h+=`<button class="btn btn-sm ${active?'btn-primary':'btn-ghost'}"
            onclick="choosePotcar('${elem}','${variant}')">${variant}</button>`;
    });
    h+=`</div></div>`;
  }
  h+='</div>';
  el.innerHTML=h;
}
function choosePotcar(elem,variant){
  POTCAR_CHOICES[elem]=variant;
  buildPotcarUI();
}

// ── GGA+U rows ─────────────────────────────────────────────────────────────
function toggleU(cb){
  document.getElementById('u-section').classList.toggle('hidden',!cb.checked);
  if(cb.checked&&document.getElementById('u-rows').children.length===0) addURow();
}
function addURow(el='',orb='d',U='3.0'){
  const c=document.getElementById('u-rows');
  const div=document.createElement('div');
  div.className='u-row';
  div.innerHTML=`
    <div class="f"><label>Element</label>
      <input value="${el}" placeholder="e.g. Mo"></div>
    <div class="f"><label>Orbital</label>
      <select>
        <option ${orb==='s'?'selected':''}>s</option>
        <option ${orb==='p'?'selected':''}>p</option>
        <option ${orb==='d'?'selected':''} ${orb!=='s'&&orb!=='p'&&orb!=='f'?'selected':''}>d</option>
        <option ${orb==='f'?'selected':''}>f</option>
      </select></div>
    <div class="f"><label>U (eV)</label>
      <input type="number" value="${U}" step="0.5" min="0"></div>
    <button class="btn btn-del" onclick="this.parentElement.remove()" style="margin-bottom:2px;">✕</button>`;
  c.appendChild(div);
}
function getUEntries(){
  return Array.from(document.querySelectorAll('#u-rows .u-row')).map(r=>{
    const inps=r.querySelectorAll('input,select');
    return{element:inps[0].value.trim(),orbital:inps[1].value,U:inps[2].value};
  }).filter(e=>e.element);
}

// ── DOS projection rows ────────────────────────────────────────────────────
function refreshProjRows(){
  const c=document.getElementById('proj-rows');
  if(!c) return;
  // update element selects
  c.querySelectorAll('.proj-el-sel').forEach(sel=>{
    const cur=sel.value;
    sel.innerHTML=ELEMENTS.map(e=>`<option${e===cur?' selected':''}>${e}</option>`).join('');
  });
}
function addProjRow(){
  const c=document.getElementById('proj-rows');
  const el=ELEMENTS[0]||'';
  const div=document.createElement('div');
  div.className='proj-row';
  div.innerHTML=`
    <div class="f"><label>Element</label>
      <select class="proj-el-sel" style="min-width:70px;">
        ${ELEMENTS.map(e=>`<option>${e}</option>`).join('')}
      </select></div>
    <div class="f"><label>Orbitals</label>
      <div class="orb-checks">
        <label class="ck"><input type="checkbox" data-orb="s" checked> s</label>
        <label class="ck"><input type="checkbox" data-orb="p" checked> p</label>
        <label class="ck"><input type="checkbox" data-orb="d"> d</label>
        <label class="ck"><input type="checkbox" data-orb="f"> f</label>
      </div></div>
    <button class="btn btn-del" onclick="this.parentElement.remove()" style="align-self:end;">✕</button>`;
  c.appendChild(div);
}
function getProjEntries(){
  return Array.from(document.querySelectorAll('#proj-rows .proj-row')).map(r=>{
    const el=r.querySelector('.proj-el-sel')?.value||'';
    const orbs=Array.from(r.querySelectorAll('[data-orb]'))
      .filter(cb=>cb.checked).map(cb=>cb.dataset.orb);
    return{element:el,orbitals:orbs};
  }).filter(e=>e.element&&e.orbitals.length);
}

// ── alert ──────────────────────────────────────────────────────────────────
function alertTop(msg,type){
  document.getElementById('alert-top').innerHTML=
    msg?`<div class="alert alert-${type}">${msg}</div>`:'';
}

// ── save project settings (without generating) ───────────────────────────
async function saveProjectSettings(){
  const btn=document.getElementById('btn-save-proj');
  const msg=document.getElementById('save-proj-msg');
  btn.disabled=true; msg.textContent='Saving…';
  const body={
    project_name:   v('project_name'),
    poscar:         v('poscar'),
    potcar_dir:     v('potcar_dir'),
    potcar_choices: POTCAR_CHOICES,
    functional:     v('functional'),
    spin_mode:      v('spin_mode'),
    hexagonal:      chk('hexagonal'),
    is_2d:          chk('is_2d'),
    use_u:          chk('use_u'),
    u_entries:      getUEntries(),
    param_mode:     PARAM_MODE,
    conv_kp:        v('conv_kp'),
    conv_encut:     v('conv_encut'),
    manual_encut:   v('manual_encut'),
    manual_kmesh_scf:v('manual_kmesh_scf'),
    manual_kmesh_dos:v('manual_kmesh_dos'),
    relax:  chk('t_relax'), scf:  chk('t_scf'),
    bands:  chk('t_bands'), dos:  chk('t_dos'),
    dfpt:   chk('t_dfpt'),  phonons:chk('t_phonons'),
    wannier:chk('t_wannier'),
    relax_type:   v('relax_type'),
    nsw:          parseInt(v('nsw'))||100,
    ediffg:       parseFloat(v('ediffg'))||-0.01,
    kpath:        v('kpath'),
    nkpts_bands:  parseInt(v('nkpts_bands'))||60,
    mpi_np:       parseInt(v('mpi_np'))||1,
    profile:      v('profile'),
  };
  if(!body.project_name){msg.textContent='Enter a project name first.';btn.disabled=false;return;}
  try{
    const r=await post('/api/save_project',body);
    if(!r.ok){const e=await r.json();msg.style.color='var(--err)';msg.textContent='Error: '+(e.error||r.status);return;}
    msg.style.color='var(--ok)'; msg.textContent='✓ Saved';
    populateResumeList();
    setTimeout(()=>{msg.textContent='';},3000);
  }catch(e){msg.style.color='var(--err)';msg.textContent='Error: '+e.message;}
  finally{btn.disabled=false;}
}

// ── overwrite confirmation dialog ─────────────────────────────────────────
function confirmOverwrite(title, lines, onConfirm){
  const existing=document.getElementById('overwrite-modal');
  if(existing) existing.remove();
  const modal=document.createElement('div');
  modal.id='overwrite-modal';
  modal.style.cssText='position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.55)';
  modal.innerHTML=`<div style="background:var(--card);border:1px solid var(--border);border-radius:10px;padding:28px 32px;max-width:480px;width:90%;box-shadow:0 8px 32px rgba(0,0,0,0.4)">
    <div style="font-size:15px;font-weight:700;margin-bottom:10px;color:var(--warn,#f59e0b)">\u26a0\ufe0f ${title}</div>
    <div style="font-size:13px;color:var(--sub);margin-bottom:16px;line-height:1.6">${lines.join('<br>')}</div>
    <div style="display:flex;gap:10px;justify-content:flex-end">
      <button id="owCancel" class="btn btn-ghost btn-sm">Cancel</button>
      <button id="owConfirm" class="btn" style="background:#ef4444;color:#fff;font-size:12px;padding:6px 16px">Overwrite</button>
    </div>
  </div>`;
  document.body.appendChild(modal);
  modal.querySelector('#owCancel').onclick=()=>modal.remove();
  modal.querySelector('#owConfirm').onclick=()=>{modal.remove();onConfirm();};
}

// ── generate ───────────────────────────────────────────────────────────────
async function generate(){
  const btn=document.getElementById('btn-gen');
  btn.disabled=true; btn.textContent='⏳ Generating…';
  alertTop('','');

  const body={
    project_name:   v('project_name'),
    poscar:         v('poscar'),
    potcar_dir:     v('potcar_dir'),
    potcar_choices: POTCAR_CHOICES,
    functional:     v('functional'),
    spin_mode:    v('spin_mode'),
    hexagonal:    chk('hexagonal'),
    is_2d:        chk('is_2d'),
    use_u:        chk('use_u'),
    u_entries:    getUEntries(),
    param_mode:   PARAM_MODE,
    // convergence
    conv_kp:      v('conv_kp'),
    conv_encut:   v('conv_encut'),
    // manual
    manual_encut:    v('manual_encut'),
    manual_kmesh_scf:v('manual_kmesh_scf'),
    manual_kmesh_dos:v('manual_kmesh_dos'),
    // tasks
    relax:        chk('t_relax'),
    scf:          chk('t_scf'),
    bands:        chk('t_bands'),
    dos:          chk('t_dos'),
    // relaxation
    relax_type:   v('relax_type'),
    nsw:          parseInt(v('nsw'))||100,
    ediffg:       parseFloat(v('ediffg'))||-0.01,
    // bands
    kpath:        v('kpath'),
    nkpts_bands:  parseInt(v('nkpts_bands'))||60,
    ewin:         v('ewin'),
    // dos
    dos_proj:     getProjEntries(),
    // dfpt
    dfpt:       chk('t_dfpt'),
    dfpt_ediff: v('dfpt_ediff')||'1E-8',
    // phonons
    phonons:       chk('t_phonons'),
    phonons_dim:   v('phonons_dim')||'2 2 2',
    phonons_mesh:  v('phonons_mesh')||'20 20 20',
    phonons_disp:  parseFloat(v('phonons_disp'))||0.01,
    phonons_band:  v('phonons_band'),
    phonons_nac:   chk('phonons_nac'),
    // wannier
    wannier:          chk('t_wannier'),
    wannier_num_wann: parseInt(v('wannier_num_wann'))||8,
    wannier_proj:     v('wannier_proj'),
    wannier_ewin:     v('wannier_ewin'),
    mpi_np:       parseInt(v('mpi_np'))||1,
    profile:      v('profile'),
  };

  // Check if project directory already has files that would be overwritten
  let doGenerate=false;
  try{
    const chk=await post('/api/check_overwrite',{mode:'generate',project_name:body.project_name});
    const cd=await chk.json();
    if(cd.exists && cd.steps && cd.steps.length>0){
      btn.disabled=false; btn.innerHTML='<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> Generate Workflow';
      confirmOverwrite(
        'Existing files will be overwritten',
        [`Project <strong>${body.project_name}</strong> already exists with the following steps:`,
         cd.steps.map(s=>`&nbsp;&nbsp;• ${s}`).join('<br>'),
         'Input files (INCAR, KPOINTS, run.sh, etc.) in these folders will be replaced.<br>Existing <strong>OUTCAR / WAVECAR / CHGCAR</strong> files are not touched.'],
        ()=>_doGenerate(body)
      );
      return;
    }
  }catch(e){/* network error — proceed anyway */}

  await _doGenerate(body);
}

async function _doGenerate(body){
  const btn=document.getElementById('btn-gen');
  btn.disabled=true; btn.textContent='⏳ Generating…';
  try{
    const r=await post('/api/generate',body);
    const d=await r.json();
    if(!r.ok){alertTop(d.error||'Generation failed','error');return;}
    PROJECT=d.project; HAS_CONV=d.has_convergence;
    document.getElementById('proj-label').textContent='Project: '+PROJECT+'/';
    buildWorkflow(d.steps,d.has_convergence);
    populateResumeList();
    goTab('workflow');
  }catch(e){alertTop('Error: '+e.message,'error');}
  finally{
    btn.disabled=false;
    btn.innerHTML='<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> Generate Workflow';
  }
}

// ── build workflow panel ───────────────────────────────────────────────────
function buildWorkflow(steps,hasConv){
  if(!steps.length){
    document.getElementById('wf').innerHTML=`
      <div class="alert alert-info" style="margin-bottom:14px;">
        Folder: <strong>${PROJECT}/</strong> &nbsp;·&nbsp; No workflow steps yet
      </div>
      <div>
        <button class="btn btn-ghost btn-sm" onclick="openFiles('_root')">📂 Browse / Edit Files</button>
      </div>`;
    return;
  }
  const prod=steps.filter(s=>!s.startsWith('00'));
  let h=`<div class="alert alert-info" style="margin-bottom:14px;">
    Output: <strong>${PROJECT}/</strong> &nbsp;·&nbsp; ${steps.length} step(s) generated
  </div>`;

  if(hasConv){
    h+=`<div class="phase">
      <div class="phase-title">Phase 1 — Convergence Tests</div>
      ${stepRow('convergence','00','Convergence tests')}
      <div id="log-convergence" class="term hidden"></div>
      <div class="g2" style="margin-top:12px;" id="conv-plots-wf"></div>
      <div class="sep"></div>
      <div style="font-size:11px;font-weight:600;color:var(--sub);margin-bottom:10px;">
        Review results then set production parameters:
      </div>
      <div class="g3">
        <div class="f"><label>ENCUT (eV)</label><input id="encut_p2" placeholder="e.g. 520"></div>
        <div class="f"><label>K-mesh for relax / SCF</label><input id="kmesh_p2" placeholder="e.g. 12x12x6"></div>
        <div class="f"><label>K-mesh for DOS (denser)</label><input id="kmesh_dos_p2" placeholder="e.g. 24x24x12"></div>
      </div>
    </div>
    <div class="phase"><div class="phase-title">Phase 2 — Production Calculations</div>`;
  }

  h+=`<div id="step-list">`;
  prod.forEach((s,i)=>h+=stepRow(s,String(i+1).padStart(2,'0'),s));
  h+=`</div>`;

  if(hasConv){
    h+=`<div style="margin-top:12px;">
      <button class="btn btn-green" onclick="runPhase2()">▶ Run Phase 2 with these parameters</button>
    </div></div>`;
  }else{
    h+=`<div style="margin-top:12px;">
      <button class="btn btn-green" onclick="runAll()">▶ Run All Steps</button>
    </div>`;
  }

  h+=`<div id="log-phase2" class="term hidden" style="margin-top:10px;"></div>`;
  h+=`<div id="log-all"    class="term hidden" style="margin-top:10px;"></div>`;
  document.getElementById('wf').innerHTML=h;
}

function stepRow(step,num,label){
  const showFiles = !step.startsWith('00') && step !== 'convergence';
  const extraFiles =
    step === '05_wannier'
      ? `<button class="btn btn-ghost btn-sm" onclick="openFileModal('${step}','wannier90.win')">🔧 wannier90.win</button>`
    : step === '06_dfpt'
      ? `<button class="btn btn-ghost btn-sm" onclick="showBornCharges()">⚛ Born charges</button>`
    : step === '07_phonons'
      ? `<button class="btn btn-ghost btn-sm" onclick="openFileModal('${step}','band.conf')">🎵 band.conf</button>` +
        `<button class="btn btn-ghost btn-sm" onclick="openFileModal('${step}','mesh.conf')">📊 mesh.conf</button>`
    : '';
  const fileBtns  = showFiles
    ? STEP_FILES.map(f =>
        `<button class="btn btn-ghost btn-sm" onclick="openFileModal('${step}','${f.name}')">${f.icon} ${f.name}</button>`
      ).join('') + extraFiles
    : '';
  return `<div class="step-row" id="row-${step}">
    <div class="step-icon">${num}</div>
    <div class="step-name">${label}</div>
    <span class="badge b-ready" id="badge-${step}">ready</span>
    <div style="display:flex;gap:5px;flex-wrap:wrap;">
      <button class="btn btn-primary btn-sm" onclick="runStep('${step}')">▶ Run</button>
      <button class="btn btn-ghost btn-sm"   onclick="toggleLog('${step}')">Log</button>
      ${fileBtns}
    </div>
  </div>
  <div id="log-${step}" class="term hidden"></div>`;
}

// ── run controls ───────────────────────────────────────────────────────────
async function runStep(step){
  if(!PROJECT) return;
  try{
    const chk=await post('/api/check_overwrite',{mode:'run',project:PROJECT,step});
    const cd=await chk.json();
    if(cd.has_outcar){
      confirmOverwrite(
        'OUTCAR already exists — re-run this step?',
        [`Step <strong>${step}</strong> already has an OUTCAR from a previous run.`,
         'Running again will overwrite OUTCAR and any other output files in this folder.',
         'Existing WAVECAR / CHGCAR are also overwritten by VASP.'],
        ()=>_doRunStep(step)
      );
      return;
    }
  }catch(e){/* proceed */}
  _doRunStep(step);
}
async function _doRunStep(step){
  setBadge(step,'running');
  const r=await post('/api/run',{project:PROJECT,step});
  if(!r.ok){const e=await r.json().catch(()=>({}));alert(e.error||'Run failed ('+r.status+')');setBadge(step,'error');return;}
  const d=await r.json();
  showLog(step); stream(d.job_key,step);
}
async function runAll(){
  if(!PROJECT) return;
  try{
    const chk=await post('/api/check_overwrite',{mode:'run',project:PROJECT,step:'all'});
    const cd=await chk.json();
    if(cd.has_outcar){
      confirmOverwrite(
        'Existing output will be overwritten',
        [`The following steps already have OUTCAR files:`,
         cd.steps.map(s=>`&nbsp;&nbsp;• ${s}`).join('<br>'),
         'Running all steps will overwrite these results.'],
        ()=>_doRunAll()
      );
      return;
    }
  }catch(e){/* proceed */}
  _doRunAll();
}
async function _doRunAll(){
  const r=await post('/api/run',{project:PROJECT,step:'all'});
  if(!r.ok){const e=await r.json().catch(()=>({}));alert(e.error||'Run failed ('+r.status+')');return;}
  const d=await r.json();
  showLog('all'); stream(d.job_key,'all');
}
async function runPhase2(){
  const encut=v('encut_p2'),kmesh=v('kmesh_p2'),kmesh_dos=v('kmesh_dos_p2');
  if(!encut||!kmesh){alert('Enter ENCUT and k-mesh first.');return;}
  try{
    const chk=await post('/api/check_overwrite',{mode:'run',project:PROJECT,step:'calculations'});
    const cd=await chk.json();
    if(cd.has_outcar){
      confirmOverwrite(
        'Existing output will be overwritten',
        [`The following steps already have OUTCAR files:`,
         cd.steps.map(s=>`&nbsp;&nbsp;• ${s}`).join('<br>'),
         'Running Phase 2 will overwrite these results.'],
        ()=>_doRunPhase2(encut,kmesh,kmesh_dos)
      );
      return;
    }
  }catch(e){/* proceed */}
  _doRunPhase2(encut,kmesh,kmesh_dos);
}
async function _doRunPhase2(encut,kmesh,kmesh_dos){
  const r=await post('/api/run_phase2',{project:PROJECT,encut,kmesh,kmesh_dos});
  if(!r.ok){const e=await r.json().catch(()=>({}));alert(e.error||'Run failed ('+r.status+')');return;}
  const d=await r.json();
  showLog('phase2'); stream(d.job_key,'phase2');
}
function toggleLog(step){document.getElementById('log-'+step)?.classList.toggle('hidden');}
function showLog(step){document.getElementById('log-'+step)?.classList.remove('hidden');}

// ── SSE streaming ──────────────────────────────────────────────────────────
function stream(key,step){
  if(ACTIVE_ES){ACTIVE_ES.close();ACTIVE_ES=null;}
  const logEl=document.getElementById('log-'+step);
  if(logEl) logEl.textContent='';
  ACTIVE_ES=new EventSource('/api/stream/'+key);
  ACTIVE_ES.onmessage=e=>{
    if(e.data==='[DONE]'){
      ACTIVE_ES.close();ACTIVE_ES=null;
      refreshStatus();
      if(HAS_CONV&&step==='convergence') loadConvPlots('wf');
      return;
    }
    if(!logEl||!e.data.trim()) return;
    const div=document.createElement('div');
    const t=e.data.toLowerCase();
    div.className=t.includes('error')?'err':t.includes('warning')?'warn':
      (t.includes('ok')||t.includes('converged')||t.includes('done'))?'ok':
      (t.includes('>>>')||t.includes('starting')||t.includes('encut')||t.includes('kpoints')||t.includes('updating'))?'info':'';
    div.textContent=e.data;
    logEl.appendChild(div);
    logEl.scrollTop=logEl.scrollHeight;
  };
  ACTIVE_ES.onerror=()=>{ACTIVE_ES.close();ACTIVE_ES=null;};
}

// ── status ─────────────────────────────────────────────────────────────────
const BL={'ready':'ready','running':'running…','done':'✓ done','error':'✗ error','ran':'ran'};
const BC={'ready':'b-ready','running':'b-running','done':'b-done','error':'b-error','ran':'b-ran'};
function setBadge(step,status){
  const el=document.getElementById('badge-'+step);
  if(!el) return;
  el.className='badge '+(BC[status]||'b-ready');
  el.textContent=BL[status]||status;
}
async function refreshStatus(){
  if(!PROJECT) return;
  try{const d=await(await fetch('/api/status/'+PROJECT)).json();
    Object.entries(d).forEach(([s,st])=>setBadge(s,st));}catch{}
}
setInterval(()=>{
  if(document.getElementById('p-workflow').classList.contains('active')&&PROJECT)
    refreshStatus();
},4000);

// ── convergence charts ─────────────────────────────────────────────────────
const CONV_PTYPES = [
  {key:'energy',      label:'Total Energy'},
  {key:'pressure',    label:'Pressure (kBar)'},
  {key:'forces',      label:'Forces atom 1 (eV/Å)'},
  {key:'eigenvalues', label:'Eigenvalues near Ef'},
];

function _convImgGrid(dtype){
  const t = Date.now();
  return CONV_PTYPES.map(pt => `
    <div>
      <div style="font-size:11px;font-weight:600;color:var(--sub);margin-bottom:3px;">${pt.label}</div>
      <img src="/api/convergence_plot/${PROJECT}/${dtype}/${pt.key}.png?t=${t}"
           style="width:100%;border-radius:4px;display:block;">
    </div>`).join('');
}

async function loadConvPlots(suffix){
  const el=document.getElementById(`conv-plots-${suffix}`);
  if(!el||!PROJECT) return;
  el.innerHTML=`
    <div class="card">
      <div class="card-title">ENCUT Convergence</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px;">
        ${_convImgGrid('encut')}
      </div>
      <a href="/api/convergence_pdf/${PROJECT}/encut" class="btn btn-ghost btn-sm"
         download style="display:inline-block;margin-top:8px;">⬇ PDF (energy)</a>
    </div>
    <div class="card">
      <div class="card-title">K-point Convergence</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px;">
        ${_convImgGrid('kpoints')}
      </div>
      <a href="/api/convergence_pdf/${PROJECT}/kpoints" class="btn btn-ghost btn-sm"
         download style="display:inline-block;margin-top:8px;">⬇ PDF (energy)</a>
    </div>`;
}

// ── results ────────────────────────────────────────────────────────────────
async function buildResults(){
  const el=document.getElementById('res');
  if(!PROJECT){el.innerHTML='<div class="alert alert-info">Generate and run a workflow first.</div>';return;}
  el.innerHTML='<div style="color:var(--sub);padding:16px;">Loading…</div>';

  const ts=Date.now();

  // ── calculation summary table ──────────────────────────────────────────
  let h=`<div class="card">
    <div class="card-title">Calculation Summary</div>
    <table class="tbl">
      <thead><tr><th>Step</th><th>Final energy (eV)</th><th>E-Fermi (eV)</th><th>Mag. moment</th><th>Converged</th><th></th></tr></thead>
      <tbody id="sumtb"></tbody>
    </table>
  </div>`;

  // ── band structure ─────────────────────────────────────────────────────
  h+=`<div class="card">
    <div class="card-title">Band Structure</div>
    <div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
      <button class="btn btn-ghost btn-sm" onclick="runSumo('bands')">▶ Run sumo-bandplot</button>
      <a href="/api/plot_pdf/${PROJECT}/bands" class="btn btn-ghost btn-sm" download title="Download PDF">⬇ PDF</a>
    </div>
    <div id="log-sumo-bands" class="term hidden sumo-log"></div>
    <div style="text-align:center;">
      <img id="img-bands" src="/api/plot/${PROJECT}/bands?t=${ts}" style="max-width:100%;max-height:500px;cursor:pointer;"
           onclick="window.open(this.src)"
           onerror="this.parentElement.innerHTML='<div class=no-plot>Not available — run 03_bands then click Run sumo-bandplot</div>'">
    </div>
  </div>`;

  // ── DOS (total + projected side by side) ──────────────────────────────
  h+=`<div class="card">
    <div class="card-title">Density of States</div>
    <div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
      <button class="btn btn-ghost btn-sm" onclick="runSumo('dos')">▶ Run sumo-dosplot</button>
      <a href="/api/plot_pdf/${PROJECT}/dos_total" class="btn btn-ghost btn-sm" download title="Download total DOS PDF">⬇ Total PDF</a>
      <a href="/api/plot_pdf/${PROJECT}/dos_proj"  class="btn btn-ghost btn-sm" download title="Download projected DOS PDF">⬇ Proj PDF</a>
    </div>
    <div id="log-sumo-dos" class="term hidden sumo-log"></div>
    <div class="plot-grid">
      <div class="plot-card">
        <h4>Total DOS</h4>
        <img class="dos-img" data-ptype="dos_total"
             src="/api/plot/${PROJECT}/dos_total?t=${ts}" style="max-width:100%;cursor:pointer;"
             onclick="window.open(this.src)"
             onerror="this.parentElement.innerHTML='<h4>Total DOS</h4><div class=no-plot>Not available yet</div>'">
      </div>
      <div class="plot-card">
        <h4>Projected DOS</h4>
        <img class="dos-img" data-ptype="dos_proj"
             src="/api/plot/${PROJECT}/dos_proj?t=${ts}" style="max-width:100%;cursor:pointer;"
             onclick="window.open(this.src)"
             onerror="this.parentElement.innerHTML='<h4>Projected DOS</h4><div class=no-plot>Not available yet</div>'">
      </div>
    </div>
  </div>`;

  // ── Born charges + dielectric ──────────────────────────────────────────
  h+=`<div class="card" id="card-born">
    <div class="card-title">Born Effective Charges &amp; Dielectric Tensor</div>
    <div style="display:flex;gap:8px;margin-bottom:8px;">
      <button class="btn btn-ghost btn-sm" onclick="loadBornCharges()">▶ Extract from OUTCAR</button>
    </div>
    <div id="born-output" style="font-family:monospace;font-size:12px;white-space:pre;
         background:var(--code-bg,#f5f5f5);padding:10px;border-radius:6px;
         max-height:300px;overflow:auto;display:none;"></div>
    <div id="born-placeholder" style="color:var(--sub);font-size:12px;">
      Run 06_dfpt then click Extract to view Born charges and dielectric tensor.</div>
  </div>`;

  // ── Phonon spectrum ─────────────────────────────────────────────────────
  h+=`<div class="card">
    <div class="card-title">Phonon Spectrum</div>
    <div style="display:flex;gap:8px;margin-bottom:8px;flex-wrap:wrap;">
      <a href="/api/phonon_plot_pdf/${PROJECT}/band" class="btn btn-ghost btn-sm" download>⬇ Band PDF</a>
      <a href="/api/phonon_plot_pdf/${PROJECT}/dos"  class="btn btn-ghost btn-sm" download>⬇ DOS PDF</a>
    </div>
    <div class="plot-grid">
      <div class="plot-card">
        <h4>Phonon Band Structure</h4>
        <img src="/api/phonon_plot/${PROJECT}/band?t=${ts}" style="max-width:100%;cursor:pointer;"
             onclick="window.open(this.src)"
             onerror="this.parentElement.innerHTML='<h4>Phonon Band Structure</h4><div class=no-plot>Not available — run 07_phonons</div>'">
      </div>
      <div class="plot-card">
        <h4>Phonon DOS</h4>
        <img src="/api/phonon_plot/${PROJECT}/dos?t=${ts}" style="max-width:100%;cursor:pointer;"
             onclick="window.open(this.src)"
             onerror="this.parentElement.innerHTML='<h4>Phonon DOS</h4><div class=no-plot>Not available — run 07_phonons</div>'">
      </div>
    </div>
  </div>`;

  // ── convergence ────────────────────────────────────────────────────────
  if(HAS_CONV){
    h+=`<div class="card"><div class="card-title">Convergence</div>
      <div id="conv-plots-rc"></div></div>`;
  }

  h+=`<div style="margin-top:8px;display:flex;gap:8px;">
    <button class="btn btn-ghost btn-sm" onclick="buildResults()">↺ Refresh plots</button>
    <button class="btn btn-ghost btn-sm" onclick="rerunSumo()">▶ Re-run sumo plots</button>
    <button class="btn btn-ghost btn-sm" onclick="runAnalyze()">▶ Run analyze.sh (re-generate plots)</button>
  </div>
  <div id="log-analyze" class="term hidden" style="margin-top:8px;"></div>`;

  el.innerHTML=h;

  // Fill summary table
  try{
    const d=await(await fetch('/api/summary/'+PROJECT)).json();
    const tb=document.getElementById('sumtb');
    if(tb) tb.innerHTML=Object.entries(d).map(([step,info])=>`
      <tr>
        <td>${step}</td>
        <td style="font-family:monospace">${info.energy||'—'}</td>
        <td style="font-family:monospace">${info.efermi?info.efermi+' eV':'—'}</td>
        <td style="font-family:monospace">${info.magmom||'—'}</td>
        <td class="${info.converged?'ok':'fail'}">${info.converged?'✓ yes':'✗ no'}</td>
        <td><button class="btn btn-ghost btn-sm" onclick="openOutcar('${step}')">📋 OUTCAR</button></td>
      </tr>`).join('');
  }catch{}

  if(HAS_CONV){ loadConvPlots('rc'); }
}

async function rerunSumo(){
  if(!PROJECT) return;
  await fetch('/api/clear_plots/'+PROJECT);
  buildResults();
}

async function loadBornCharges(){
  if(!PROJECT) return;
  const out=document.getElementById('born-output');
  const ph =document.getElementById('born-placeholder');
  if(out) out.textContent='Loading…';
  if(out) out.style.display='block';
  if(ph)  ph.style.display='none';
  try{
    const d=await(await fetch(`/api/born_charges/${PROJECT}`)).json();
    if(d.error){if(out) out.textContent='ERROR: '+d.error; return;}
    if(out) out.textContent=d.text;
  }catch(e){if(out) out.textContent='Error: '+e.message;}
}

function showBornCharges(){
  goTab('results');
  buildResults().then(()=>loadBornCharges());
}

async function runSumo(stype){
  if(!PROJECT) return;
  const logId='log-sumo-'+stype;
  const logEl=document.getElementById(logId);
  if(logEl){logEl.classList.remove('hidden'); logEl.textContent='';}
  const r=await fetch(`/api/run_sumo/${PROJECT}/${stype}`);
  if(!r.ok){const e=await r.json().catch(()=>({}));if(logEl) logEl.textContent='ERROR: '+(e.error||r.status); return;}
  const d=await r.json();
  if(_sumoES){_sumoES.close(); _sumoES=null;}
  _sumoES=new EventSource('/api/stream/'+d.job_key);
  _sumoES.onmessage=e=>{
    if(e.data==='[DONE]'){
      _sumoES.close(); _sumoES=null;
      const ts=Date.now();
      if(stype==='bands'){
        const img=document.getElementById('img-bands');
        if(img) img.src=`/api/plot/${PROJECT}/bands?t=${ts}`;
      } else {
        document.querySelectorAll('.dos-img').forEach(img=>{
          img.src=`/api/plot/${PROJECT}/${img.dataset.ptype}?t=${ts}`;
        });
      }
      return;
    }
    if(!logEl||!e.data.trim()) return;
    const div=document.createElement('div');
    const t=e.data.toLowerCase();
    div.className=t.includes('error')?'err':t.includes('warning')?'warn':
      (t.includes('saved')||t.includes('e_fermi')||t.includes('done'))?'ok':'';
    div.textContent=e.data;
    logEl.appendChild(div);
    logEl.scrollTop=logEl.scrollHeight;
  };
  _sumoES.onerror=()=>{_sumoES.close(); _sumoES=null;};
}

async function runAnalyze(){
  if(!PROJECT) return;
  const logEl=document.getElementById('log-analyze');
  if(logEl){logEl.classList.remove('hidden'); logEl.textContent='';}
  const r=await post('/api/run',{project:PROJECT,step:'analyze'});
  if(!r.ok){const e=await r.json().catch(()=>({}));if(logEl) logEl.textContent='ERROR: '+(e.error||r.status); return;}
  const d=await r.json();
  stream(d.job_key,'analyze');
}

// ── file editor ────────────────────────────────────────────────────────────
let _editorStep=null, _editorFile=null, _editorFiles=[], _origContent='';

async function openFiles(step){
  if(!PROJECT) return;
  _editorStep=step;
  const r=await fetch(`/api/files/${PROJECT}/${step}`);
  if(!r.ok){const e=await r.json().catch(()=>({}));alert(e.error||'Could not list files ('+r.status+')');return;}
  const d=await r.json();
  _editorFiles=d.files;   // [{name, readonly}]
  if(!_editorFiles.length){alert('No files found in '+step);return;}
  document.getElementById('modal-title').textContent=step+' — files';
  buildFileTabs(_editorFiles);
  // Default to INCAR if present, otherwise first file
  const first=_editorFiles.find(f=>f.name==='INCAR')||_editorFiles[0];
  await loadFileTab(first.name, first.readonly);
  document.getElementById('file-modal').classList.remove('hidden');
  document.getElementById('save-msg').textContent='';
}

async function openFileModal(step, filename){
  if(!PROJECT) return;
  _editorStep=step;
  const r=await fetch(`/api/files/${PROJECT}/${step}`);
  if(!r.ok){const e=await r.json().catch(()=>({}));alert('Could not list files: '+(e.error||r.status));return;}
  const d=await r.json();
  _editorFiles=d.files;
  const target=_editorFiles.find(f=>f.name===filename);
  if(!target){
    alert(`${filename} not found in ${step} — has this step run yet?`);
    return;
  }
  document.getElementById('modal-title').textContent=`${step} — ${filename}`;
  buildFileTabs(_editorFiles);
  await loadFileTab(target.name, target.readonly);
  document.getElementById('file-modal').classList.remove('hidden');
  document.getElementById('save-msg').textContent='';
}
// Keep openOutcar as alias
async function openOutcar(step){ return openFileModal(step,'OUTCAR'); }

function buildFileTabs(files){
  const el=document.getElementById('file-tabs');
  el.innerHTML=files.map(f=>{
    const icon=f.readonly
      ? (f.name==='OUTCAR'?'📋 ': f.name==='OSZICAR'?'📊 ': f.name.endsWith('.sh')?'⚙ ':'🔒 ')
      : '✏️ ';
    return `<div class="file-tab" id="ftab-${f.name}" onclick="loadFileTab('${f.name}',${f.readonly})">${icon}${f.name}</div>`;
  }).join('');
}

async function loadFileTab(filename, readonly){
  _editorFile=filename;
  document.querySelectorAll('.file-tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('ftab-'+filename)?.classList.add('active');

  const r=await fetch(`/api/file/${PROJECT}/${_editorStep}/${filename}`);
  if(!r.ok){const e=await r.json().catch(()=>({}));document.getElementById('file-editor').value='Error: '+(e.error||r.status);return;}
  const d=await r.json();

  _origContent=d.content;
  const editor=document.getElementById('file-editor');
  editor.value=d.content;

  // OUTCAR: scroll to bottom so most recent lines are visible
  if(filename==='OUTCAR'||filename==='vasp.out'||filename==='OSZICAR'){
    setTimeout(()=>{editor.scrollTop=editor.scrollHeight;},50);
  }

  const isRO=readonly||d.readonly;
  editor.readOnly=isRO;
  document.getElementById('save-btn').disabled=isRO;
  document.getElementById('revert-btn').disabled=isRO;
  const roEl=document.getElementById('modal-readonly');
  roEl.style.display=isRO?'':'none';
  if(isRO&&d.truncated){
    roEl.style.display='';
    roEl.textContent=`⚠ Read-only · ${d.total_lines} lines total · showing last 500`;
  } else if(isRO){
    roEl.textContent='⚠ Read-only';
  }
  document.getElementById('save-msg').textContent='';
}

async function saveFile(){
  const content=document.getElementById('file-editor').value;
  const r=await post(`/api/file/${PROJECT}/${_editorStep}/${_editorFile}`,{content});
  const msg=document.getElementById('save-msg');
  if(r.ok){
    _origContent=content;
    msg.style.color='var(--ok)';
    msg.textContent='✓ Saved  (original backed up as '+_editorFile+'.bak)';
  }else{
    const e=await r.json().catch(()=>({}));
    msg.style.color='var(--err)';
    msg.textContent='✗ '+(e.error||r.status);
  }
}

function revertFile(){
  document.getElementById('file-editor').value=_origContent;
  document.getElementById('save-msg').textContent='';
}

function closeModal(event){
  if(event&&event.target!==document.getElementById('file-modal')) return;
  document.getElementById('file-modal').classList.add('hidden');
  _editorStep=null; _editorFile=null;
}

// ── helpers ────────────────────────────────────────────────────────────────
const v  =id=>document.getElementById(id)?.value||'';
const chk=id=>document.getElementById(id)?.checked||false;
const post=(url,body)=>fetch(url,{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
</script>
<!-- ════════ FILE EDITOR MODAL ════════ -->
<div id="file-modal" class="modal-backdrop hidden" onclick="closeModal(event)">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="modal-header">
      <h3 id="modal-title">Input files</h3>
      <span id="modal-readonly" style="font-size:11px;color:var(--warn);display:none;">
        ⚠ Read-only (.sh)
      </span>
      <button class="btn btn-ghost btn-sm" onclick="closeModal()">✕ Close</button>
    </div>
    <div class="file-tabs" id="file-tabs"></div>
    <div class="modal-body">
      <textarea class="file-editor" id="file-editor" spellcheck="false"></textarea>
    </div>
    <div class="modal-footer">
      <span class="save-msg" id="save-msg"></span>
      <button class="btn btn-ghost btn-sm" id="revert-btn" onclick="revertFile()">↩ Revert</button>
      <button class="btn btn-primary btn-sm" id="save-btn"   onclick="saveFile()">💾 Save</button>
    </div>
  </div>
</div>

</body>
</html>"""

if __name__=='__main__':
    import webbrowser, threading
    url = 'http://localhost:5001'
    print("\nVASP Workflow GUI")
    print("="*40)
    print(f"Opening {url} in your browser...")
    print("Stop with:  Ctrl-C\n")
    # Open browser 1.5 s after Flask starts (gives the server time to bind)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=5001, debug=False, threaded=True)
