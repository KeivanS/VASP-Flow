#!/usr/bin/env python3
"""vasp_diagnose.py -- diagnose VASP-Flow runs and write fix-and-resubmit scripts.

Give it a list file, one entry per line ('#' starts a comment; only the first word
counts).  Typically this is the *_list.txt of material IDs the runs were made from.
An entry can be
  * a material name - any folder name: an MP ID (mp-149), a formula (GaAs), GaAs_PBE, ...:
    looked up under the list file's folder (and --root): <root>/<name>, <root>/*/<name>,
    <root>/*/materials/<name>;
  * one material / project folder: it has some of 00_convergence 01_relax 02_scf
    03_bands 04_dos 06_dfpt 08_lobster;
  * a highthroughput folder (has materials/): every materials/<id> is checked;
  * any folder whose sub-folders are material folders (e.g. an ht-mp-scf run).
Relative paths are relative to the list file's folder.  A run folder holding a
MOVED_TO_SCRATCH.txt note ("continues in <dir>") is followed to <dir>.

Diagnosis is read-only.  For every step it decides ok / problem from OUTCAR, OSZICAR,
lobster.out and the SLURM logs (+ sacct when available).  00_convergence: every point of
the ENCUT and k-point tests must have finished with a converged SCF.

  quota         disk quota / "error during write"        resubmit unchanged (check space first)
  oom           insufficient virtual memory, UCX "Cannot allocate memory", OUT_OF_MEMORY
                                                          KPAR capped (<= --kpar-max, divisor of
                                                          the ranks), whole-node memory
  mpi_init      MPI start-up failure (node problem)      resubmit unchanged
  timeout       time limit                               walltime x2; a relax continues from CONTCAR
  wavecar       WAVECAR does not fit (NBANDS/ENCUT/cell) WAVECAR moved aside, start from CHGCAR
  chgcar        CHGCAR unreadable / truncated            SCF from scratch (ICHARG = 2), or the SCF
                                                          is redone for bands/DOS/LOBSTER/DFPT
  zbrent        ZBRENT fatal error in a relax            IBRION 2 -> 1, continue from CONTCAR
  symmetry      symmetry-operation errors                SYMPREC = 1E-4 (ISYM = 0 if already set)
  edddav        ZHEGV / EDDDAV errors                    SCF: ALGO = All; relax: ALGO = Normal, then All;
                                                          NSCF: as nscf_diverged
  nscf_diverged NSCF energies |E| > 1e6                  NBANDS >= 1.3 x SCF NBANDS; ALGO Davidson
                                                          (Normal/default) -> VeryFast, Fast -> Normal
  relax_unconv  relax finished without convergence       continue from CONTCAR
  scf_nelm      SCF hit NELM                             ALGO = All (NELM >= 150), later steps redone
  nscf_unconv   bands/DOS hit NELM                       ALGO as nscf_diverged, NELM >= 150
  lobster_fail  NSCF finished, LOBSTER binary did not    rerun only the LOBSTER binary
  not_run       never ran (e.g. upstream failed)         submitted
  unknown       anything else                            script written, but only submitted with
                                                          --retry-unknown (error lines in its header)
  spilling_high LOBSTER finished, charge spilling > --spill-max %   reported only
  missing       a listed material ID was not found         reported only
  phase2_pending  convergence done, production steps never started: run
                submit_calculations.sh / run_calculations.sh yourself     reported only

00_convergence: broken points are moved aside, the test INCAR gets the fix of the class
(ALGO/NELM, KPAR, SYMPREC, walltime) and the test's run.sh is resubmitted - it skips
the points that are already done.  The production steps are not touched for it.

When the relax or the SCF is redone, every later step is redone too; otherwise only the
broken later steps.  Old outputs of every redone step are moved to <folder>/stale_<date>/
(nothing is deleted).

Writes into --out (default ./vasp_diagnose_<date>/):
  diagnosis.tsv            one line per folder: layout, per-step status, planned fix
  fix/<label>.sh           per folder: guard (no queued jobs, outputs unchanged since the
                           diagnosis), the fixes, then the resubmission chain
  submit_all_fixes.sh      runs every fix script (at most MAX_QUEUED jobs in the queue);
                           safe to re-run - finished ones are skipped (marker in .submitted/)

Layouts: ht (materials/<id>/job.sbatch <step>, ht-semimetals), slurm (vasp-agent-slurm:
<step>/run.sh with #SBATCH), local (vasp-agent: <step>/run.sh run in the foreground).

Usage:
  ./vasp_diagnose.py folders.txt                       diagnose, write the scripts
  less vasp_diagnose_*/diagnosis.tsv vasp_diagnose_*/fix/<label>.sh     review
  nohup bash vasp_diagnose_*/submit_all_fixes.sh > fixes.log 2>&1 &
Options: --out DIR  --root DIR (more places to look up IDs)  --since YYYY-MM-DD (sacct
         window, default 14 days)  --kpar-max 8  --spill-max 5  --retry-unknown  --max-queued 200
"""
import argparse
import datetime
import glob
import math
import os
import re
import shlex
import subprocess
import sys
from collections import Counter

STEPS = [('01_relax', 'relax'), ('02_scf', 'scf'), ('03_bands', 'bands'),
         ('04_dos', 'dos'), ('06_dfpt', 'dfpt'), ('08_lobster', 'lobster')]
STEP_DIR = dict((s, d) for d, s in STEPS)
ORDER = [s for _, s in STEPS]
UNHANDLED = ('05_wannier', '07_phonons')
CONV_DIR = '00_convergence'
CONV_TESTS = ('encut', 'kpoints')
NSCF = ('bands', 'dos', 'lobster')
DEFAULT_TIME = {'conv': '08:00:00', 'relax': '08:00:00', 'scf': '04:00:00', 'bands': '02:00:00',
                'dos': '02:00:00', 'dfpt': '08:00:00', 'lobster': '04:00:00'}
HT_TIME_KEY = {'relax': 'TIME_RELAX', 'scf': 'TIME_SCF', 'bands': 'TIME_BANDS',
               'lobster': 'TIME_LOBSTER'}
HT_STEPS = ('relax', 'scf', 'bands', 'lobster')
INCAR_FIX = ('scf_nelm', 'nscf_diverged', 'nscf_unconv', 'edddav', 'zbrent', 'symmetry')     # what job.sbatch understands

PAT = [   # (class, regex) -- first match wins; searched in the job logs (+ OUTCAR tail)
    ('quota',    r'Disk quota exceeded|No space left on device|error during write'),
    ('oom',      r'insufficient virtual memory|cannot allocate memory|out of memory|'
                 r'oom[-_ ]kill|ud_ep\.c.*unhandled timeout'),
    ('mpi_init', r'PMPI_Init|MPI_Init_thread|MPIDI_OFI_mpi_init'),
    ('wavecar',  r'ERROR.*while reading WAVECAR|plane wave coefficients changed'),
    ('chgcar',   r'end of record during read.*CHGCAR|charge density could not be read'),
    ('zbrent',   r'ZBRENT: fatal error'),
    ('symmetry', r'inverse of rotation matrix was not found|POSMAP internal error|'
                 r'internal error in subroutine SGRCON|PRICELV|RHOSYG'),
    ('edddav',   r'Error EDDDAV|ZHEGV failed'),
]
# VASP prints these and carries on - they are not the cause of a failure:
#   'Sub-Space-Matrix is not hermitian' (warning), 'The number of bands has changed ...
#   orbitals not found in the WAVECAR will be initialized ...' (info)
STOPPED = r'(hard|soft) stop (encountered|was set)'        # STOPCAR of the walltime guard (log / OUTCAR)
ERRLINE = re.compile(r'error|bad news|severe|abort|killed|memory|quota|segmentation|'
                     r'cannot|not found|exit code|fatal|warning', re.I)
ERRSKIP = re.compile(r'IBZKPT|Often results|\(SIGTERM\)|^Image|Unknown  |Stack trace|'
                     r'srun: error: .*tasks? [0-9]', re.I)


# ── small file helpers ───────────────────────────────────────────────────────
def read(path, tail=None, head=None):
    try:
        with open(path, 'rb') as f:
            if head:
                return f.read(head).decode('utf-8', 'replace')
            if tail:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - tail))
            return f.read().decode('utf-8', 'replace')
    except (OSError, IOError):
        return ''


def contains(path, text):
    """Streaming substring search (OUTCARs can be large)."""
    b = text.encode()
    prev = b''
    try:
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    return False
                if b in prev + chunk:
                    return True
                prev = chunk[-len(b):]
    except (OSError, IOError):
        return False


def incar(path):
    tags = {}
    for line in read(path).splitlines():
        line = re.split(r'[#!]', line)[0]
        for part in line.split(';'):
            if '=' in part:
                k, v = part.split('=', 1)
                tags[k.strip().upper()] = v.strip()
    return tags


def to_int(v, default=None):
    try:
        return int(float(str(v).split()[0]))
    except (ValueError, IndexError, TypeError):
        return default


def to_sec(t):
    """[D-]HH:MM:SS | MM:SS | minutes -> seconds"""
    if not t:
        return None
    days = 0
    if '-' in t:
        d, t = t.split('-', 1)
        days = int(d)
    parts = [int(x) for x in t.split(':')]
    if len(parts) == 1:
        sec = parts[0] * 60
    elif len(parts) == 2:
        sec = parts[0] * 60 + parts[1]
    else:
        sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
    return days * 86400 + sec


def from_sec(s):
    s = int(s)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return ('%d-' % d if d else '') + '%02d:%02d:%02d' % (h, m, s)


def chg_complete(path):
    """CHGCAR/CHG has at least ceil(NGX*NGY*NGZ/5) data lines after the grid line."""
    try:
        with open(path, 'rb') as f:
            head = [f.readline() for _ in range(9)]
            nat = sum(int(x) for x in head[6].split())
            sel = 1 if head[7].strip()[:1] in (b's', b'S') else 0
            for _ in range(9, 10 + sel + nat - 1):
                f.readline()
            grid = [int(x) for x in f.readline().split()[:3]]
            need = (grid[0] * grid[1] * grid[2] + 4) // 5
            n = 0
            for n, _ in enumerate(f, 1):
                if n >= need:
                    return True
            return False
    except (OSError, IOError, ValueError, IndexError):
        return False


def ere(text):
    """Escape text for grep -E (re.escape adds backslashes grep warns about)."""
    return re.sub(r'([.\[\]()*+?{}|^$\\])', r'\\\1', text)


def sbatch_jobname(path):
    m = re.search(r'#SBATCH\s+(?:--job-name=|-J\s*)(\S+)', read(path))
    return m.group(1) if m else None


def run(cmd):
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             universal_newlines=True)
        return p.communicate()[0]
    except OSError:
        return ''


# ── one material / project folder ────────────────────────────────────────────
class Material(object):
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.name = os.path.basename(self.path)
        self.steps = [s for d, s in STEPS if os.path.isdir(os.path.join(self.path, d))]
        self.unhandled = [d for d in UNHANDLED if os.path.isdir(os.path.join(self.path, d))]
        self.conv = [t for t in CONV_TESTS if os.path.isfile(os.path.join(self.path, CONV_DIR, t, 'run.sh'))]
        parent = os.path.dirname(self.path)
        if os.path.isfile(os.path.join(self.path, 'job.sbatch')):
            self.layout = 'ht'
            self.root = os.path.dirname(parent) if os.path.basename(parent) == 'materials' else parent
            self.label = os.path.basename(self.root) + '__' + self.name
        else:
            runs = [read(self.f(s, 'run.sh')) for s in self.steps] + \
                   [read(self.conv_f(t, 'run.sh')) for t in self.conv]
            self.layout = 'slurm' if any('#SBATCH' in r for r in runs) else 'local'
            self.root = parent
            self.label = os.path.basename(parent) + '__' + self.name
        self.env = self._ht_env() if self.layout == 'ht' else {}

    def f(self, step, name):
        return os.path.join(self.path, STEP_DIR[step], name)

    def conv_f(self, test, name=''):
        return os.path.join(self.path, CONV_DIR, test, name)

    def jobname(self, step):
        """Job name of a step (or of a convergence test 'conv:<test>')."""
        if step.startswith('conv:'):
            return sbatch_jobname(self.conv_f(step[5:], 'run.sh')) or '%s_%sconv' % (self.name, step[5])
        if self.layout != 'ht':
            jn = sbatch_jobname(self.f(step, 'run.sh'))
            if jn:
                return jn
        return '%s_%s' % (self.name, step)

    def jobnames(self):
        names = [self.jobname(s) for s in self.steps] + [self.jobname('conv:' + t) for t in self.conv]
        if self.layout == 'ht':                  # job.sbatch names its jobs <name>_<step>
            names += ['%s_%s' % (self.name, s) for s in ORDER]
        return set(names)

    def queued(self, queue):
        return bool(self.jobnames() & queue)

    def _ht_env(self):
        env = {}
        for line in read(os.path.join(self.root, 'env.sh')).splitlines():
            m = re.match(r'\s*(?:export\s+)?(TIME_[A-Z]+)=["\']?([0-9:-]+)', line)
            if m:
                env[m.group(1)] = m.group(2)
        return env

    def walltime(self, step):
        if step.startswith('conv:'):
            m = re.search(r'#SBATCH\s+(?:--time=|-t\s*)(\S+)', read(self.conv_f(step[5:], 'run.sh')))
            return m.group(1) if m else DEFAULT_TIME['conv']
        if self.layout == 'ht':
            return self.env.get(HT_TIME_KEY.get(step, ''), DEFAULT_TIME[step])
        m = re.search(r'#SBATCH\s+(?:--time=|-t\s*)(\S+)', read(self.f(step, 'run.sh')))
        return m.group(1) if m else DEFAULT_TIME[step]

    def ranks(self, step, outcar=None, script=None):
        m = re.search(r'running on\s+(\d+)\s+total cores', read(outcar or self.f(step, 'OUTCAR'), head=20000))
        if m:
            return int(m.group(1))
        if script:
            script = read(script)
        else:
            script = read(os.path.join(self.path, 'job.sbatch')) if self.layout == 'ht' \
                else read(self.f(step, 'run.sh'))
        nodes = re.search(r'#SBATCH\s+--nodes=(\d+)', script)
        tpn = re.search(r'#SBATCH\s+--ntasks-per-node=(\d+)', script)
        nt = re.search(r'#SBATCH\s+(?:--ntasks=|-n\s*)(\d+)', script)
        if tpn:
            return int(tpn.group(1)) * (int(nodes.group(1)) if nodes else 1)
        return int(nt.group(1)) if nt else None

    def logs(self, step, rec):
        """Text of the newest job logs of this step (tails)."""
        files = []
        if self.layout == 'ht' and step in STEP_DIR:
            if rec:
                files = [os.path.join(self.path, '%s-%s-%s.%s' % (self.name, self.jobname(step), rec['id'], e))
                         for e in ('err', 'out')]
            if not any(os.path.isfile(x) for x in files):
                cand = sorted(glob.glob(os.path.join(self.path, '%s-%s-*.out' % (self.name, self.jobname(step)))),
                              key=os.path.getmtime)
                if cand:
                    files = [cand[-1][:-4] + '.err', cand[-1]]
        elif rec:
            sd = self.conv_f(step[5:]) if step.startswith('conv:') else os.path.join(self.path, STEP_DIR[step])
            for d in (self.path, sd, self.root):
                for e in ('err', 'out'):
                    p = os.path.join(d, 'slurm-%s.%s' % (rec['id'], e))
                    if os.path.isfile(p):
                        files.append(p)
        if step in STEP_DIR and os.path.isfile(self.f(step, 'vasp.out')):
            files.append(self.f(step, 'vasp.out'))
        return '\n'.join(read(x, tail=400000) for x in files if os.path.isfile(x))


# ── per-step state ───────────────────────────────────────────────────────────
def finished(m, step):
    return contains(m.f(step, 'OUTCAR'), 'General timing and accounting')


def converged(m, step):
    return contains(m.f(step, 'OUTCAR'), 'aborting loop because EDIFF is reached')


def lobster_done(m):
    return 'finished in' in read(m.f('lobster', 'lobster.out'))


def spilling(m):
    vals = re.findall(r'abs\. charge spilling:\s*([0-9.]+)', read(m.f('lobster', 'lobster.out')), re.I)
    return max(float(v) for v in vals) if vals else None


def nscf_diverged(m, step):
    for line in read(m.f(step, 'OSZICAR')).splitlines():
        p = line.split()
        if len(p) > 2 and p[0] in ('DAV:', 'RMM:', 'CG:'):
            try:
                if abs(float(p[2])) > 1e6:
                    return True
            except ValueError:
                pass
    return False


def step_ok(m, step):
    """'ok' or a provisional problem class decided from the outputs alone."""
    if step == 'relax':
        if contains(m.f(step, 'OUTCAR'), 'reached required accuracy'):
            return 'ok'
        return 'relax_unconv' if finished(m, step) else None
    if step == 'lobster':
        if lobster_done(m):
            return 'ok'
        if finished(m, step):
            return 'lobster_fail'
        return None
    if not finished(m, step):
        return None
    if step == 'scf':
        return 'ok' if converged(m, step) else 'scf_nelm'
    if step == 'dfpt':
        out = m.f(step, 'OUTCAR')
        return 'ok' if (contains(out, 'BORN EFFECTIVE CHARGES') or
                        contains(out, 'MACROSCOPIC STATIC DIELECTRIC TENSOR')) else 'unknown'
    if nscf_diverged(m, step):
        return 'nscf_diverged'
    if step == 'dos':
        doscar = m.f(step, 'DOSCAR')
        if not (os.path.isfile(doscar) and os.path.getsize(doscar) > 0):
            return 'unknown'
    return 'ok' if converged(m, step) else 'nscf_unconv'


def classify(m, step, rec, provisional, point=None):
    """Problem class + up to 6 error lines for a step that is not ok (point: one
    convergence-test folder, whose vasp.out/OUTCAR are read instead of the step's)."""
    log = m.logs(step, rec)
    outcar = os.path.join(point, 'OUTCAR') if point else m.f(step, 'OUTCAR')
    if point:
        log += '\n' + read(os.path.join(point, 'vasp.out'), tail=200000)
    text = log + '\n' + read(outcar, tail=20000)
    errs = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if ERRLINE.search(line) and not ERRSKIP.search(line) and line not in seen:
            seen.add(line)
            errs.append(line[:160])
    state = rec['state'] if rec else ''
    # a diverged NSCF never finishes, however long it runs: this beats a time limit
    if step in NSCF and nscf_diverged(m, step):
        return 'nscf_diverged', errs[:6]
    # the job record of the last attempt is the most reliable source
    if state.startswith('TIMEOUT'):
        return 'timeout', errs[:6]
    if state.startswith('OUT_OF_ME'):
        return 'oom', errs[:6]
    # VASP stopped by the STOPCAR the walltime guard writes before the limit
    stopped = re.search(STOPPED, text + read(outcar, tail=200000), re.I)
    if provisional in ('relax_unconv', 'scf_nelm', 'nscf_unconv', 'nscf_diverged', 'lobster_fail') \
            and not stopped:
        return provisional, errs[:6]
    for cls, rx in PAT:
        if re.search(rx, text, re.I):
            if cls == 'edddav' and step in NSCF:
                return 'nscf_diverged', errs[:6]
            return cls, errs[:6]
    if stopped or re.search(r'DUE TO TIME LIMIT|time limit hit', text):
        return 'timeout', errs[:6]
    if step in NSCF and nscf_diverged(m, step):
        return 'nscf_diverged', errs[:6]
    if provisional:
        return provisional, errs[:6]
    if not os.path.isfile(outcar) and not log.strip():
        return 'not_run', errs[:6]
    return 'unknown', errs[:6]


# ── 00_convergence ───────────────────────────────────────────────────────────
def conv_points(m, test):
    """Point folders the test's run.sh will make (encut_<E>, <N1>x<N2>x<N3>)."""
    tdir = m.conv_f(test)
    script = read(os.path.join(tdir, 'run.sh'))
    pts = []
    if test == 'encut':
        r = re.search(r'for EC in \$\(seq (\d+) (\d+) (\d+)\)', script)
        if r:
            a, st, b = (int(x) for x in r.groups())
            pts = ['encut_%d' % v for v in range(a, b + 1, max(st, 1))]
    else:
        pts = re.findall(r'vf_done "(\d+x\d+x\d+)"', script)
        r = re.search(r'for NX in \$\(seq (\d+) (\d+) (\d+)\)', script)
        if not pts and r:
            a, st, b = (int(x) for x in r.groups())
            for nx in range(a, b + 1, max(st, 1)):
                found = sorted(glob.glob(os.path.join(tdir, '%dx%dx*' % (nx, nx))))
                pts.append(os.path.basename(found[0]) if found else '%dx%dx?' % (nx, nx))
    if not pts:
        pts = sorted(d for d in os.listdir(tdir) if os.path.isfile(os.path.join(tdir, d, 'INCAR')))
    return pts


def conv_fp_files(m):
    return [os.path.join(CONV_DIR, t, p, 'OUTCAR') for t in m.conv for p in conv_points(m, t) if '?' not in p]


def conv_check(m, recs):
    """-> {test: (status, class, errs, bad points)} for the tests that are not ok."""
    bad = {}
    for t in m.conv:
        key = 'conv:' + t
        rec = recs.get(m.jobname(key))
        broken, nelm = [], []
        for p in conv_points(m, t):
            out = os.path.join(m.conv_f(t), p, 'OUTCAR')
            if contains(out, 'General timing and accounting'):
                if not contains(out, 'aborting loop because EDIFF is reached'):
                    nelm.append(p)
            else:
                broken.append(p)
        if not broken and not nelm:
            continue
        if broken:
            first = os.path.join(m.conv_f(t), broken[0].replace('?', '*'))
            first = (glob.glob(first) or [first])[0]
            cls, errs = classify(m, key, rec, 'scf_nelm' if nelm else None, point=first)
        else:
            cls, errs = 'scf_nelm', []
        bad[t] = ('%s (%d of %d points)' % (cls, len(broken) + len(nelm), len(conv_points(m, t))),
                  cls, errs, broken + nelm, nelm)
    return bad


def conv_not_converged(m):
    """Parameters choose_params.py could not converge in the tested range."""
    txt = read(os.path.join(m.path, CONV_DIR, 'converged_params.txt'))
    return re.findall(r'^(\w+)_CONVERGED\s*=\s*no', txt, re.M)


def conv_plan(m, bad, recs, args):
    """-> (bash fix lines, [(test, jobname, walltime, extra sbatch opts)])"""
    lines, subs = [], []
    for t, (_, cls, _, pts, nelm) in sorted(bad.items()):
        key, d = 'conv:' + t, os.path.join(CONV_DIR, t)
        tags = incar(m.conv_f(t, 'INCAR'))
        wall, extra = m.walltime(key), []

        def say(x):
            lines.append('echo "  %s: %s"' % (d, x.replace('"', "'")))
        if cls in ('scf_nelm', 'edddav') or nelm:
            lines.append('set_tag %s/INCAR ALGO All' % d)
            if to_int(tags.get('NELM'), 60) < 150:
                lines.append('set_tag %s/INCAR NELM 150' % d)
            say('ALGO = All, NELM >= 150 (%s)' % ('scf_nelm' if nelm else cls))
        if cls == 'symmetry':
            lines.append('set_tag %s/INCAR %s' % (d, 'ISYM 0' if 'SYMPREC' in tags else 'SYMPREC 1E-4'))
            say('symmetry error -> %s' % ('ISYM = 0' if 'SYMPREC' in tags else 'SYMPREC = 1E-4'))
        elif cls == 'oom':
            pt = [os.path.join(m.conv_f(t), p, 'OUTCAR') for p in pts]
            n = m.ranks(key, outcar=(pt or [None])[0], script=m.conv_f(t, 'run.sh'))
            kp = to_int(tags.get('KPAR'))
            if n and kp:
                k = largest_divisor(n, min(args.kpar_max, kp))
                if k < kp:
                    lines.append('set_tag %s/INCAR KPAR %d' % (d, k))
                    say('KPAR %d -> %d (%d ranks)' % (kp, k, n))
            extra.append('--mem=0')
        elif cls == 'timeout':
            rec = recs.get(m.jobname(key))
            try:
                sec = to_sec(rec['limit'])
            except (TypeError, KeyError, ValueError):
                sec = None
            wall = from_sec(2 * (sec or to_sec(wall)))
            say('walltime %s' % wall)
        for p in pts:
            if '?' not in p:
                lines.append('stash_file %s %s' % (d, p))
        say('%d point(s) moved aside, run.sh redoes them and skips the finished ones' % len(pts))
        if m.layout != 'local' and '#SBATCH' in read(m.conv_f(t, 'run.sh')):
            # run.sh may locate itself with $0, which is SLURM's spool copy: pin it to its folder
            lines.append('sed -i.bak -E "s|^HERE=.*|HERE=$D/%s; cd \\"\\$HERE\\"|" %s/run.sh && rm -f %s/run.sh.bak' % (d, d, d))
        subs.append((t, m.jobname(key), wall, extra))
    return lines, subs


# ── the fix plan ─────────────────────────────────────────────────────────────
def largest_divisor(n, cap):
    for k in range(min(n, cap), 0, -1):
        if n % k == 0:
            return k
    return 1


def scf_nbands(m):
    v = re.findall(r'NBANDS=\s*(\d+)', read(m.f('scf', 'OUTCAR'), head=200000))
    if v:
        return int(v[0])
    return to_int(incar(m.f('scf', 'INCAR')).get('NBANDS'))


def plan(m, status, cls_of, recs, args):
    """-> (redo steps, bash fix lines, sbatch extras per step, walltime per step, binary-only?)"""
    first_bad = [s for s in ORDER if s in m.steps and not status[s].startswith('ok')]
    redo = []
    if 'relax' in first_bad or 'scf' in first_bad:
        start = 'relax' if 'relax' in first_bad else 'scf'
        redo = [s for s in ORDER if s in m.steps and ORDER.index(s) >= ORDER.index(start)]
    else:
        redo = [s for s in first_bad]
        wav = m.f('lobster', 'WAVECAR')
        lob_bin = (cls_of.get('lobster') == 'lobster_fail' and os.path.isfile(wav) and os.path.getsize(wav) > 0)
        needs_chg = [s for s in redo if not (s == 'lobster' and lob_bin)]
        chg = m.f('scf', 'CHGCAR')
        if needs_chg and 'scf' in m.steps and not (os.path.isfile(chg) and chg_complete(chg)):
            redo = ['scf'] + redo
            cls_of['scf'] = 'chgcar_missing'
            status['scf'] = 'ok but 02_scf/CHGCAR missing or incomplete'
    lines, extra, wall = [], {}, {}
    for s in redo:
        wall[s] = m.walltime(s)
        extra[s] = []
    wav = m.f('lobster', 'WAVECAR')
    binary_only = ('lobster' in redo and 'scf' not in redo and m.layout != 'local'
                   and cls_of.get('lobster') == 'lobster_fail'
                   and os.path.isfile(wav) and os.path.getsize(wav) > 0)
    reasons = set(cls_of[s] for s in first_bad if s in cls_of) | set([cls_of.get('scf')])

    def say(t):
        lines.append('echo "  %s"' % t.replace('"', "'"))

    changed = set()      # step dirs whose INCAR really changes

    def tag(d, key, val):
        cur = incar(os.path.join(m.path, d, 'INCAR')).get(key)
        if cur is None or cur.strip().upper() != str(val).upper():
            lines.append('set_tag %s/INCAR %s %s' % (d, key, val))
            changed.add(d)

    # memory: cap KPAR in every step that will run
    if 'oom' in reasons:
        for s in redo:
            n = m.ranks(s)
            kp = to_int(incar(m.f(s, 'INCAR')).get('KPAR'))
            if n and kp:
                k = largest_divisor(n, min(args.kpar_max, kp))
                if k < kp:
                    lines.append('set_tag %s/INCAR KPAR %d' % (STEP_DIR[s], k))
                    say('%s: KPAR %d -> %d (%d ranks)' % (s, kp, k, n))
            if m.layout != 'local':
                extra[s].append('--mem=0')
    # walltime x2 for the step(s) that timed out
    for s in first_bad:
        if cls_of.get(s) == 'timeout' and s in wall:
            rec = recs.get(s)
            try:
                t = to_sec(rec['limit'])
            except (TypeError, KeyError, ValueError):
                t = None
            t = t or to_sec(wall[s])
            wall[s] = from_sec(2 * t)
            say('%s: walltime %s' % (s, wall[s]))
    c_relax, c_scf = cls_of.get('relax'), cls_of.get('scf')
    if 'relax' in redo:
        if c_relax == 'zbrent' and incar(m.f('relax', 'INCAR')).get('IBRION', '').strip() == '2':
            tag('01_relax', 'IBRION', 1)
            say('relax: IBRION 2 -> 1 (ZBRENT)')
        if c_relax == 'symmetry':
            tags = incar(m.f('relax', 'INCAR'))
            tag('01_relax', *(('ISYM', 0) if 'SYMPREC' in tags else ('SYMPREC', '1E-4')))
            say('relax: symmetry error -> %s' % ('ISYM = 0' if 'SYMPREC' in tags else 'SYMPREC = 1E-4'))
        if c_relax == 'edddav':            # Davidson -> Normal; if that already failed -> All (CG)
            algo = 'All' if incar(m.f('relax', 'INCAR')).get('ALGO', '').strip().upper().startswith('N') else 'Normal'
            tag('01_relax', 'ALGO', algo)
            say('relax: ALGO = %s (EDDDAV)' % algo)
        lines.append('continue_relax')
        if c_relax == 'wavecar':
            lines.append('rm -f 01_relax/WAVECAR')
    if 'scf' in redo:
        tags = incar(m.f('scf', 'INCAR'))
        if c_scf in ('scf_nelm', 'edddav'):
            tag('02_scf', 'ALGO', 'All')
            if to_int(tags.get('NELM'), 60) < 150:
                tag('02_scf', 'NELM', 150)
            say('scf: ALGO = All, NELM >= 150 (%s)' % c_scf)
        if c_scf == 'symmetry':
            tag('02_scf', *(('ISYM', 0) if 'SYMPREC' in tags else ('SYMPREC', '1E-4')))
            say('scf: symmetry error -> %s' % ('ISYM = 0' if 'SYMPREC' in tags else 'SYMPREC = 1E-4'))
        if 'relax' in m.steps and 'relax' not in redo:
            rchg = m.f('relax', 'CHGCAR')
            if c_scf in ('wavecar',):
                lines.append('stash_file 01_relax WAVECAR')
                say('scf: relax WAVECAR moved aside, start from the relaxed CHGCAR')
            if not (os.path.isfile(rchg) and chg_complete(rchg)):
                lines.append('stash_file 01_relax CHGCAR; stash_file 01_relax WAVECAR')
                lines.append('set_tag 02_scf/INCAR ISTART 0; set_tag 02_scf/INCAR ICHARG 2')
                say('scf: relax CHGCAR missing/incomplete -> from scratch (ICHARG = 2)')
        elif 'relax' not in m.steps and c_scf in ('wavecar', 'chgcar'):
            lines.append('set_tag 02_scf/INCAR ISTART 0; set_tag 02_scf/INCAR ICHARG 2')
            say('scf: start from scratch (ICHARG = 2)')
    for s in ('bands', 'dos', 'lobster'):
        if s in redo and cls_of.get(s) in ('nscf_diverged', 'nscf_unconv'):
            more = ''
            if cls_of[s] == 'nscf_diverged' and s != 'lobster':     # LOBSTER sets its own NBANDS
                nb = scf_nbands(m)
                if nb:
                    new = int(math.ceil(1.3 * nb / 2.0)) * 2
                    cur = to_int(incar(m.f(s, 'INCAR')).get('NBANDS'), 0)
                    if new > cur:
                        tag(STEP_DIR[s], 'NBANDS', new)
                        more = ', NBANDS %d' % new
            # Davidson (Normal, VASP's default) -> RMM-DIIS (VeryFast): with a fixed density
            # Davidson can run away (Ba2YReO6, Sr2YReO6: |E| -> 1e10) where RMM-DIIS converges
            cur = incar(m.f(s, 'INCAR')).get('ALGO', 'Normal').strip().upper()
            algo = 'Normal' if cur.startswith('F') else 'VeryFast'
            tag(STEP_DIR[s], 'ALGO', algo)
            if to_int(incar(m.f(s, 'INCAR')).get('NELM'), 60) < 150:
                tag(STEP_DIR[s], 'NELM', 150)
            if STEP_DIR[s] in changed:
                say('%s: %s -> ALGO = %s%s' % (s, cls_of[s], algo, more))
    # stash old outputs of every redone step
    for s in redo:
        if s == 'lobster' and binary_only:
            lines.append('stash 08_lobster lobster-only; echo post > 08_lobster/.vf_resume')
            say('lobster: NSCF kept, only the LOBSTER binary reruns')
        else:
            lines.append('stash %s' % STEP_DIR[s])
    return redo, lines, extra, wall, changed


# ── bash pieces ──────────────────────────────────────────────────────────────
HELPERS = r'''STAMP=$(date +%Y%m%d_%H%M)
set_tag() {   # file TAG value: replace every TAG line, then append
    sed -i.bak -E "/^[[:space:]]*$2[[:space:]]*=/d" "$1" && rm -f "$1.bak"
    printf '%s = %s\n' "$2" "$3" >> "$1"
}
OUTS="OUTCAR OSZICAR vasprun.xml CONTCAR XDATCAR CHG CHGCAR WAVECAR EIGENVAL DOSCAR PROCAR
      IBZKPT PCDAT REPORT vaspout.h5 ELFCAR LOCPOT STOPCAR vasp.out .vf_resume .vf_restarts lobsterout lobster.out"
stash() {     # step-dir [lobster-only]: move outputs to stale_<date>/<step>/
    local d=$1 s=stale_$STAMP/$1 f list=$OUTS
    [ "${2:-}" = lobster-only ] && list="STOPCAR .vf_resume .vf_restarts lobsterout lobster.out"
    for f in $list "$d"/*.lobster; do
        f=${f#"$d/"}
        [ -e "$d/$f" ] || continue
        mkdir -p "$s" && mv -f "$d/$f" "$s/"
    done
}
stash_file() { [ -e "$1/$2" ] || return 0; mkdir -p "stale_$STAMP/$1"; mv -f "$1/$2" "stale_$STAMP/$1/"; }
continue_relax() {   # newest complete CONTCAR (or last XDATCAR frame) -> 01_relax/POSCAR
    local r=01_relax nat
    nat=$(sed -n 7p $r/POSCAR | awk '{s=0; for (i=1;i<=NF;i++) s+=$i; print s}')
    [ -f $r/POSCAR.orig ] || cp -p $r/POSCAR $r/POSCAR.orig
    if [ -s $r/CONTCAR ] && [ "$(wc -l < $r/CONTCAR)" -ge $((8 + nat)) ]; then
        cp $r/CONTCAR $r/POSCAR; echo "  relax: continues from CONTCAR"
    elif [ -s $r/XDATCAR ] && ( cd $r && python3 - <<'PYX'
lines = open('XDATCAR').read().splitlines()
idx = [i for i, l in enumerate(lines) if l.strip().lower().startswith('direct configuration')]
nat = sum(int(x) for x in lines[6].split())
if not idx or len(lines) < idx[-1] + 1 + nat:
    raise SystemExit(1)
last = idx[-1]
head = lines[last - 7:last] if last >= 7 and lines[last - 7].strip() == lines[0].strip() else lines[:7]
open('POSCAR', 'w').write('\n'.join(head + ['Direct'] + lines[last + 1:last + 1 + nat]) + '\n')
PYX
    ); then echo "  relax: continues from the last XDATCAR frame"
    else echo "  relax: starts from its current POSCAR"; fi
}
'''


def fingerprint_files(m):
    files = conv_fp_files(m) + [os.path.join(STEP_DIR[s], 'OUTCAR') for s in m.steps]
    if 'lobster' in m.steps:
        files.append('08_lobster/lobster.out')
    return files


def fingerprint(m):
    out = []
    for f in fingerprint_files(m):
        p = os.path.join(m.path, f)
        out.append('%s:%s' % (f, int(os.path.getmtime(p)) if os.path.exists(p) else 'none'))
    return ' '.join(out)


def conv_submit_lines(m, subs):
    q = shlex.quote
    out = []
    for t, jn, wall, extra in subs:
        d = os.path.join(CONV_DIR, t)
        if m.layout == 'local' or '#SBATCH' not in read(m.conv_f(t, 'run.sh')):
            out.append('( cd %s && bash run.sh ) || { echo "!!! %s failed"; exit 1; }' % (d, d))
            continue
        v = 'J_conv_' + t
        opts = ' '.join(['--parsable', '--job-name=%s' % q(jn), '--time=%s' % wall] + extra)
        out.append('%s=$(cd %s && sbatch %s run.sh) || { echo "!!! sbatch %s failed"; exit 1; }' % (v, d, opts, d))
        out.append('echo "  submitted %s = $%s (%s)"' % (d, v, wall))
    return out


def submit_lines(m, redo, extra, wall):
    q = shlex.quote
    out = []
    if m.layout == 'local':
        for s in redo:
            out.append('( cd %s && bash run.sh ) || { echo "!!! %s failed"; exit 1; }' % (STEP_DIR[s], s))
        return out
    jid = {}
    for i, s in enumerate(redo):
        dep = ''
        if m.layout == 'ht':
            prev = redo[i - 1] if i else None
            if prev:
                dep = '--dependency=%s:$J_%s' % ('afterany' if (s == 'lobster' and prev == 'bands') else 'afterok', prev)
            target = 'job.sbatch %s' % s
        else:
            up = 'relax' if s == 'scf' else ('scf' if s != 'relax' else None)
            if up in jid:
                dep = '--dependency=afterok:$J_%s' % up
            target = '%s/run.sh' % STEP_DIR[s]
        opts = ' '.join(['--parsable', '--job-name=%s' % q(m.jobname(s)), '--time=%s' % wall[s]] + extra[s]
                        + ([dep] if dep else []))
        out.append('J_%s=$(sbatch %s %s) || { echo "!!! sbatch %s failed"; exit 1; }' % (s, opts, target, s))
        out.append('echo "  submitted %s = $J_%s (%s)"' % (s, s, wall[s]))
        jid[s] = True
    return out


def write_script(path, m, status, errs, redo_txt, fix, submit, out_dir):
    q = shlex.quote
    steps_txt = '  '.join('%s: %s' % (s, status[s]) for s in status)
    body = ['#!/bin/bash',
            '# vasp_diagnose fix for %s' % m.label,
            '# folder : %s   (layout: %s)' % (m.path, m.layout),
            '# diagnosed %s:  %s' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), steps_txt)]
    for s in status:
        for e in errs.get(s, []):
            body.append('#   %s | %s' % (s, e))
    body += ['# resubmits: %s' % redo_txt, 'set -u',
             'D=%s; OUT=%s; LABEL=%s' % (q(m.path), q(out_dir), q(m.label)),
             'cd "$D" || exit 1',
             '# guard: nothing of this folder queued, outputs unchanged since the diagnosis']
    if m.layout != 'local':
        names = ['^%s$' % ere(j) for j in sorted(m.jobnames())]
        body.append('if squeue -h -u "${USER:-$(whoami)}" -o "%%j" 2>/dev/null | grep -qE %s; then '
                    'echo "-- $LABEL: jobs queued, skipped"; exit 0; fi' % q('|'.join(names)))
    files = fingerprint_files(m)
    body += ['FP=""; for f in %s; do FP="$FP $f:$(stat -c %%Y "$f" 2>/dev/null || echo none)"; done' % ' '.join(files),
             'if [ "${FP# }" != %s ]; then echo "-- $LABEL: outputs changed since the diagnosis - '
             'run vasp_diagnose.py again"; exit 0; fi' % q(fingerprint(m)),
             'echo "== $LABEL"', HELPERS.rstrip('\n')]
    body += fix
    body += submit
    body += ['mkdir -p "$OUT/.submitted" && touch "$OUT/.submitted/$LABEL"', '']
    with open(path, 'w') as fh:
        fh.write('\n'.join(body))
    os.chmod(path, 0o755)


# ── collect folders ──────────────────────────────────────────────────────────
def is_material(p):
    return any(os.path.isdir(os.path.join(p, d)) for d, _ in STEPS) or \
        any(os.path.isfile(os.path.join(p, CONV_DIR, t, 'run.sh')) for t in CONV_TESTS)


def moved(p):
    """Follow a MOVED_TO_SCRATCH.txt note ('This run continues in <dir> ...')."""
    r = re.search(r'continues in (\S+)', read(os.path.join(p, 'MOVED_TO_SCRATCH.txt')))
    if r and os.path.isdir(r.group(1)):
        print('-> %s was moved: using %s' % (p, r.group(1)))
        return r.group(1)
    return p


def follow(p):
    """Material folder p, redirected when its run (p or p/../..) was moved."""
    parent = os.path.dirname(p)
    if os.path.basename(parent) == 'materials':
        root = moved(os.path.dirname(parent))
        return os.path.join(root, 'materials', os.path.basename(p))
    return moved(p)


def expand(entry, base):
    p = os.path.abspath(os.path.join(base, os.path.expanduser(entry)))
    if os.path.isdir(p):
        p = moved(p)
        if is_material(p):
            return [follow(p)]
        mat = os.path.join(p, 'materials')
        top = mat if os.path.isdir(mat) else p
        found = sorted(os.path.join(top, x) for x in os.listdir(top)
                       if os.path.isdir(os.path.join(top, x)) and is_material(os.path.join(top, x)))
        lst = os.path.join(p, 'material_list.txt')
        if top == mat and os.path.isfile(lst):    # the run's own list says which materials belong to it
            names = set(l.split('#')[0].split()[0] for l in open(lst) if l.split('#')[0].split())
            extra = [os.path.basename(f) for f in found if os.path.basename(f) not in names]
            if extra:
                print('-- %s: %d folder(s) in materials/ not in its material_list.txt, skipped: %s'
                      % (entry, len(extra), ' '.join(extra)))
            found = [f for f in found if os.path.basename(f) in names]
        return found or None                     # no runs in it: try entry as a material name
    return None                                  # not a folder: a material ID


_ROOT_CACHE = {}


def find_id(mid, roots):
    """All material folders called mid under the roots (one level, or */materials/)."""
    hits = []
    for r in roots:
        if r not in _ROOT_CACHE:
            subs = [os.path.join(r, x) for x in sorted(os.listdir(r)) if os.path.isdir(os.path.join(r, x))]
            _ROOT_CACHE[r] = [moved(x) for x in subs]
        for d in [r] + _ROOT_CACHE[r]:
            for c in (os.path.join(d, mid), os.path.join(d, 'materials', mid)):
                if os.path.isdir(c) and is_material(c) and os.path.realpath(c) not in \
                        [os.path.realpath(h) for h in hits]:
                    hits.append(c)
    return hits


def read_list(path, roots):
    """-> (material folders, missing IDs)"""
    base = os.path.dirname(os.path.abspath(path))
    folders, missing = [], []
    for line in open(path):
        tok = line.split('#')[0].split()
        if not tok:
            continue
        got = expand(tok[0], base)
        if got is None:
            got = find_id(tok[0], roots)
            if not got:
                missing.append(tok[0])
                if os.path.isdir(os.path.join(base, os.path.expanduser(tok[0]))):
                    print('?? %s: a folder without step folders (00_convergence, 01_relax, ...), '
                          'and no material of that name' % tok[0])
            elif len(got) > 1:
                print('!! %s found in %d runs, all checked: %s' % (tok[0], len(got), ' '.join(got)))
        for g in got:
            if g not in folders:
                folders.append(g)
    return folders, missing


def sacct_records(since):
    recs = {}
    txt = run(['sacct', '-u', os.environ.get('USER', ''), '-S', since, '-X', '-n', '-P',
               '-o', 'JobID,JobName,State,Elapsed,Timelimit'])
    for line in txt.splitlines():
        p = line.split('|')
        if len(p) < 5 or not p[0].split('_')[0].isdigit():
            continue
        jid = int(p[0].split('_')[0].split('.')[0])
        if p[1] not in recs or jid >= recs[p[1]]['num']:
            recs[p[1]] = {'num': jid, 'id': str(jid), 'state': p[2], 'elapsed': p[3], 'limit': p[4]}
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('list', help='list file: material IDs and/or folders, one per line')
    ap.add_argument('--root', action='append', default=[],
                    help='also look up material IDs under this folder (repeatable)')
    ap.add_argument('--out', default='vasp_diagnose_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    ap.add_argument('--since', default=(datetime.date.today() - datetime.timedelta(days=14)).isoformat())
    ap.add_argument('--kpar-max', type=int, default=8)
    ap.add_argument('--spill-max', type=float, default=5.0)
    ap.add_argument('--retry-unknown', action='store_true')
    ap.add_argument('--max-queued', type=int, default=200)
    args = ap.parse_args()

    roots = [os.path.dirname(os.path.abspath(args.list))] + \
            [os.path.abspath(os.path.expanduser(r)) for r in args.root]
    folders, missing = read_list(args.list, [r for r in roots if os.path.isdir(r)])
    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, 'fix'), exist_ok=True)
    recs = sacct_records(args.since)
    queue = set(run(['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%j']).split())
    rows, scripts, review = [], [], []
    count = Counter()
    for mid in missing:
        count['missing'] += 1
        rows.append([mid, '-', '-', 'not found under ' + ' '.join(roots), 'missing', ''])
    for p in folders:
        m = Material(p)
        status, cls_of, errs, mrecs = {}, {}, {}, {}
        if m.queued(queue):
            count['queued'] += 1
            rows.append([m.label, m.layout, m.path, 'queued', 'queued', ''])
            continue
        cbad = conv_check(m, recs)
        for t in m.conv:
            key = 'conv:' + t
            if t in cbad:
                status[key], cls_of[key], errs[key] = cbad[t][0], cbad[t][1], cbad[t][2]
            else:
                status[key] = 'ok'
        for s in m.steps:
            mrecs[s] = recs.get(m.jobname(s))
            st = step_ok(m, s)
            if st == 'ok':
                status[s] = 'ok'
                if s == 'lobster':
                    sp = spilling(m)
                    if sp is not None and sp > args.spill_max:
                        status[s] = 'ok (spilling %.1f%%)' % sp
                continue
            cls, e = classify(m, s, mrecs[s], st)
            status[s], cls_of[s], errs[s] = cls, cls, e
        bad = [s for s in m.steps if not status[s].startswith('ok')]
        notes = ['%s:unchecked' % d for d in m.unhandled]
        nc = conv_not_converged(m)
        if nc:
            notes.append('convergence not reached in the tested range: ' + ','.join(nc))
        # two-phase project: production steps wait for the convergence results
        phase2 = (m.conv and m.steps and all(status[s] == 'not_run' for s in m.steps) and
                  any(os.path.isfile(os.path.join(m.path, f))
                      for f in ('submit_calculations.sh', 'run_calculations.sh')))
        if phase2:
            bad = []
            notes.append('production steps not started yet (phase 2)')
        steps_txt = ' '.join(['%s:%s' % kv for kv in status.items()] + notes)
        if not bad and not cbad:
            spill = [s for s in m.steps if 'spilling' in status[s]]
            act = 'phase2_pending' if phase2 else ('spilling_high' if spill else 'ok')
            count[act] += 1
            rows.append([m.label, m.layout, m.path, steps_txt, act, ''])
            continue
        fix, submit, redo_txt, repeat = [], [], [], []
        if cbad:
            cfix, csubs = conv_plan(m, cbad, recs, args)
            fix += cfix
            submit += conv_submit_lines(m, csubs)
            redo_txt += ['%s/%s' % (CONV_DIR, t) for t in sorted(cbad)]
            if phase2:
                submit.append('echo "  when the tests are done: check them and start the production steps '
                              '(submit_calculations.sh / run_calculations.sh)"')
        if bad:
            if m.layout == 'ht' and any(s not in HT_STEPS for s in bad):
                notes.append('job.sbatch cannot run: ' + ','.join(s for s in bad if s not in HT_STEPS))
                steps_txt = ' '.join(['%s:%s' % kv for kv in status.items()] + notes)
            redo, pfix, extra, wall, changed = plan(m, status, cls_of, mrecs, args)
            for s in bad:          # the INCAR fix of this cause is already there: it did not help
                if cls_of[s] in INCAR_FIX and STEP_DIR[s] not in changed:
                    status[s] += ' (fix already in INCAR)'
                    repeat.append(s)
            if m.layout == 'ht':
                redo = [s for s in redo if s in HT_STEPS]
            fix += pfix
            submit += submit_lines(m, redo, extra, wall)
            redo_txt.append(' -> '.join(redo))
            steps_txt = ' '.join(['%s:%s' % kv for kv in status.items()] + notes)
        first = bad[0] if bad else 'conv:' + sorted(cbad)[0]
        main_cls = cls_of[first]
        count[main_cls] += 1
        script = os.path.join(out_dir, 'fix', m.label + '.sh')
        write_script(script, m, status, errs, ' ; '.join(x for x in redo_txt if x), fix, submit, out_dir)
        unknown = any(c == 'unknown' for c in cls_of.values()) or bool(repeat)
        hold = unknown and not args.retry_unknown
        (review if hold else scripts).append(script)
        rows.append([m.label, m.layout, m.path, steps_txt,
                     ('REVIEW ' if hold else '') + main_cls + ' -> ' + ' ; '.join(
                         x.replace(' -> ', '>') for x in redo_txt if x), script])

    with open(os.path.join(out_dir, 'diagnosis.tsv'), 'w') as fh:
        fh.write('label\tlayout\tpath\tsteps\taction\tscript\n')
        for r in rows:
            fh.write('\t'.join(r) + '\n')
    master = os.path.join(out_dir, 'submit_all_fixes.sh')
    with open(master, 'w') as fh:
        fh.write('#!/bin/bash\n# Runs every fix script written by vasp_diagnose.py; safe to re-run\n'
                 '# (done ones are skipped).  Start it under nohup.  MAX_QUEUED=%d by default.\n'
                 'OUT=%s\nMAX_QUEUED=${MAX_QUEUED:-%d}\n'
                 'queued() { squeue -h -u "${USER:-$(whoami)}" 2>/dev/null | wc -l | tr -d " "; }\n'
                 'run_fix() {\n'
                 '    [ -f "$OUT/.submitted/$(basename "$1" .sh)" ] && return 0\n'
                 '    while [ "$(queued)" -ge "$MAX_QUEUED" ]; do sleep 60; done\n'
                 '    bash "$1" || echo "!!! $(basename "$1") failed"\n'
                 '}\n' % (args.max_queued, shlex.quote(out_dir), args.max_queued))
        for s in scripts:
            fh.write('run_fix %s\n' % shlex.quote(s))
        if review:
            fh.write('\n# unknown cause, or its fix is already in the INCAR - read the script header first;'
                     ' uncomment to submit:\n')
            for s in review:
                fh.write('# run_fix %s\n' % shlex.quote(s))
        fh.write('echo "All fix scripts done.  Monitor: squeue -u ${USER:-$(whoami)}"\n')
    os.chmod(master, 0o755)

    print('Folders checked: %d   (layouts: %s)' % (len(rows), ', '.join(
        '%s %d' % kv for kv in sorted(Counter(r[1] for r in rows).items()))))
    for k, v in count.most_common():
        print('  %-14s %4d' % (k, v))
    print('Fix scripts: %d to submit, %d to review (unknown cause, or fix already tried)' % (len(scripts), len(review)))
    for s in review:
        print('  review: %s' % s)
    print('Report : %s\nSubmit : nohup bash %s > %s.log 2>&1 &' % (
        os.path.join(out_dir, 'diagnosis.tsv'), master, master[:-3]))


if __name__ == '__main__':
    main()
