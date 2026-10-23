#!/usr/bin/env python3
"""Check that this computer is ready for VASP-Flow, before the first job.

    python3 check_setup.py                 # the uses that fit this computer
    python3 check_setup.py gui local       # or name them: gui local slurm ht
    make check                             # the same, with default_paths loaded

Uses:  gui    the browser GUI (make run)
       local  vasp-agent.py on this computer
       slurm  vasp-agent-slurm.py on a SLURM cluster
       ht     the high-throughput drivers (ht_runs.py, ht-mp-scf.py)
Without names: slurm + ht where sbatch exists, else gui + local.

Read-only.  Prints ok / WARN / FAIL per item with the command that fixes it; exit
code 1 if anything needed for the chosen uses FAILs.  Written for Python >= 3.6
so that it can still report a too-old Python.
"""
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
USES = ('gui', 'local', 'slurm', 'ht')
counts = {'FAIL': 0, 'WARN': 0}


def say(state, text, fix=''):
    if state in counts:
        counts[state] += 1
    print('  %-5s %s' % (state, text) + ('\n        -> %s' % fix if fix else ''))


def default_paths():
    """KEY = value lines of default_paths (what make run / make check export)."""
    vals = {}
    try:
        for line in open(os.path.join(HERE, 'default_paths')):
            m = re.match(r'\s*([A-Z_0-9]+)\s*[:?]?=\s*(.*?)\s*(#.*)?$', line)
            if m and not line.lstrip().startswith('#'):
                vals[m.group(1)] = m.group(2)
    except IOError:
        pass
    return vals


DP = default_paths()


def setting(key):
    """-> (value, where it came from): environment first, then default_paths."""
    if os.environ.get(key):
        return os.environ[key], 'environment'
    if DP.get(key):
        return DP[key], 'default_paths (read by make run; export it for vasp-agent.py)'
    return '', ''


def profile(name):
    try:
        return json.load(open(os.path.join(HERE, 'profiles', name + '.json')))
    except (IOError, ValueError) as e:
        return {'_error': str(e)}


def module_ok(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def exe(cmd):
    """First word of cmd (with ~ and $VARS expanded) -> full path or None."""
    if not cmd:
        return None
    word = os.path.expandvars(os.path.expanduser(cmd.split()[0]))
    if os.sep in word:
        return word if os.path.isfile(word) and os.access(word, os.X_OK) else None
    return shutil.which(word)


def lda_sibling(pbe_dir):           # same rule as modules/vasp_input_generator.lda_sibling
    p = os.path.expanduser(pbe_dir or '').rstrip('/')
    head, tail = os.path.split(p)
    new = re.sub('PBE', 'LDA', re.sub('pbe', 'lda', tail))
    return os.path.join(head, new) if p and new != tail else ''


def check_potcars(need_lda=False):
    pbe, where = setting('VASP_POTCAR_DIR')
    if not pbe:
        say('FAIL', 'VASP_POTCAR_DIR is not set (folder with Si/, Ga_d/, ... sub-folders)',
            'export VASP_POTCAR_DIR=/path/to/PAW_PBE   (in ~/.bashrc, or in default_paths)')
        return
    d = os.path.expanduser(pbe)
    if not os.path.isdir(d):
        say('FAIL', 'POTCAR library %s does not exist (from %s)' % (d, where))
        return
    probe = [e for e in ('Si', 'O', 'C', 'Fe') if os.path.isfile(os.path.join(d, e, 'POTCAR'))]
    if probe:
        say('ok', 'POTCAR library %s (from %s): %s/POTCAR found' % (d, where, probe[0]))
    else:
        say('FAIL', 'POTCAR library %s has no Si/POTCAR, O/POTCAR, ... - wrong folder?' % d)
    lda, lwhere = setting('VASP_POTCAR_DIR_LDA')
    lda = os.path.expanduser(lda) if lda else lda_sibling(d)
    lwhere = lwhere or 'the PBE folder with PBE -> LDA'
    if lda and os.path.isdir(lda):
        say('ok', 'LDA POTCAR library %s (%s)' % (lda, lwhere))
    else:
        say('FAIL' if need_lda else 'WARN', 'no LDA POTCAR library (%s): only LDA runs need it'
            % (lda or 'none'), 'export VASP_POTCAR_DIR_LDA=/path/to/PAW_LDA')


def check_binary(label, value, where, fix, required=True, modules=None):
    path = exe(value)
    if path:
        say('ok', '%s: %s (%s)' % (label, path, where))
    elif modules:
        say('WARN', '%s: %s is not on PATH here; it should come from the modules %s'
            % (label, value or '(not set)', ' '.join(modules)),
            'check: module load %s && which %s' % (' '.join(modules), (value or 'vasp_std').split()[0]))
    else:
        say('FAIL' if required else 'WARN', '%s: %s not found (%s)'
            % (label, value or '(not set)', where or 'not set'), fix)


def main(argv):
    names = [a.lower() for a in argv if not a.startswith('-')]
    if any(a in ('-h', '--help') for a in argv) or any(n not in USES for n in names):
        print(__doc__)
        return 0 if not names or all(n in USES for n in names) else 2
    on_cluster = shutil.which('sbatch') is not None
    uses = names or (['slurm', 'ht'] if on_cluster else ['gui', 'local'])
    print('VASP-Flow setup check  -  uses: %s%s' % (', '.join(uses), '' if names else
          ('   (sbatch found: a SLURM cluster)' if on_cluster else '   (no sbatch: a workstation)')))

    print('\nPython and the code')
    v = sys.version_info
    if v >= (3, 8):
        say('ok', 'Python %d.%d.%d (%s)' % (v[0], v[1], v[2], sys.executable))
    else:
        say('FAIL', 'Python %d.%d at %s is too old (3.8 or newer needed)' % (v[0], v[1], sys.executable),
            'load a newer Python (cluster: module load ...), then make and activate the '
            'environment: python3 -m venv ~/dftphonon-env && source ~/dftphonon-env/bin/activate')
    for mod in ('numpy', 'matplotlib'):
        if module_ok(mod):
            say('ok', '%s installed' % mod)
        else:
            say('FAIL', '%s missing' % mod, 'pip install numpy matplotlib')
    try:
        out = subprocess.check_output(['git', '-C', HERE, 'status', '--porcelain', '-uno'],
                                      stderr=subprocess.STDOUT, universal_newlines=True)
        rev = subprocess.check_output(['git', '-C', HERE, 'log', '-1', '--format=%h %cd', '--date=short'],
                                      universal_newlines=True).strip()
        if out.strip():
            say('WARN', 'code %s, with local changes in %d file(s): git pull will refuse'
                % (rev, len(out.splitlines())), 'git -C %s status   (unwanted: git stash)' % HERE)
        else:
            say('ok', 'code %s, no local changes (update: git pull)' % rev)
    except Exception:
        say('WARN', '%s is not a git clone: updates with git pull will not work' % HERE)

    print('\nPOTCARs')
    check_potcars()

    if 'gui' in uses:
        print('\nGUI (make run)')
        for mod, pkg in (('flask', 'flask'), ('sumo', 'sumo')):
            if module_ok(mod):
                say('ok', '%s installed' % pkg)
            else:
                say('FAIL', '%s missing' % pkg, 'pip install flask sumo')
        if os.path.isfile(os.path.join(HERE, 'default_paths')):
            say('ok', 'default_paths present')
        else:
            say('WARN', 'no default_paths: the GUI starts with empty paths (set them on System Setup)',
                'make setup && nano default_paths')

    if 'local' in uses or 'gui' in uses:
        print('\nRunning VASP on this computer (GUI, vasp-agent.py)')
        prof = profile('workstation')
        for key, label, req in (('VASP_STD', 'vasp_std', True), ('VASP_NCL', 'vasp_ncl (SOC)', False),
                                ('LOBSTER_X', 'lobster (LOBSTER step)', False)):
            val, where = (prof.get(key.lower()), 'profiles/workstation.json') \
                if prof.get(key.lower()) else setting(key)
            check_binary(label, val, where, 'set %s in default_paths (or export it)' % key, req)
        mpi, where = setting('MPI_LAUNCH')
        if mpi:
            check_binary('MPI launcher', mpi, where, 'install MPI or set MPI_LAUNCH in default_paths')
        else:
            say('WARN', 'MPI_LAUNCH not set: VASP runs serially', 'MPI_LAUNCH = mpirun -np  in default_paths')

    if 'slurm' in uses or 'ht' in uses:
        print('\nSLURM cluster')
        for tool in ('sbatch', 'squeue', 'sacct'):
            if shutil.which(tool):
                say('ok', '%s on PATH' % tool)
            else:
                say('FAIL' if tool == 'sbatch' else 'WARN', '%s not found: not a SLURM login node?' % tool)
    if 'slurm' in uses:
        prof = profile('slurm')
        if '_error' in prof:
            say('FAIL', 'profiles/slurm.json unreadable: %s' % prof['_error'])
        else:
            s = prof.get('slurm') or {}
            for key in ('partition', 'account'):
                val = str(s.get(key, ''))
                if not val or 'your' in val.lower() or 'mygroup' in val.lower():
                    say('FAIL', 'profiles/slurm.json: slurm.%s = "%s"' % (key, val),
                        'nano %s/profiles/slurm.json' % HERE)
                else:
                    say('ok', 'profiles/slurm.json: %s = %s' % (key, val))
            mods = prof.get('modules') or []
            check_binary('vasp_std (slurm.json)', prof.get('vasp_std', ''), 'profiles/slurm.json',
                         'set vasp_std or modules in profiles/slurm.json', True, mods)

    if 'ht' in uses:
        print('\nHigh-throughput drivers (ht_runs.py, ht-mp-scf.py)')
        missing = [p for m, p in (('pymatgen', 'pymatgen'), ('mp_api', 'mp-api'), ('spglib', 'spglib'))
                   if not module_ok(m)]
        if missing:
            say('FAIL', 'missing: %s' % ' '.join(missing), 'pip install pymatgen mp-api spglib   '
                '(in the environment: source ~/dftphonon-env/bin/activate)')
        else:
            say('ok', 'pymatgen, mp-api, spglib installed')
        if os.environ.get('MP_API_KEY'):
            say('ok', 'MP_API_KEY set')
        else:
            say('WARN', 'MP_API_KEY not set: only structures already downloaded (--poscar-dir) can be used',
                'export MP_API_KEY=your_key   (free: https://materialsproject.org/api)')
        lob = os.environ.get('LOBSTER_BIN') or os.path.expandvars('$HOME/BIN/lobster-5.1.1')
        check_binary('LOBSTER binary for the jobs', lob,
                     'LOBSTER_BIN' if os.environ.get('LOBSTER_BIN') else 'ht_runs.py default',
                     'ht_runs.py --lobster-bin /path/to/lobster   (or LOBSTER_BIN in the run\'s env.sh)', False)

    print('\n%s' % ('Ready.' if not counts['FAIL'] and not counts['WARN'] else
                    '%d problem(s), %d warning(s).' % (counts['FAIL'], counts['WARN'])))
    return 1 if counts['FAIL'] else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
