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
                                                          the ranks), whole-node memory; LOBSTER
                                                          NSCF: its k-mesh halved per axis
  mpi_init      MPI start-up failure (node problem)      resubmit unchanged
  timeout       time limit                               walltime x2; a relax continues from CONTCAR
  hang          a static step converged, then stalled       rerun on 1 node (the other redone static
                until the time limit (multi-node MPI)       steps too); on 1 node already: walltime x2
  stopcar       stopped early by a STOPCAR an earlier      STOPCAR removed, the step (and the ones
                run left behind (looks finished)            that used its output) redone
  wavecar       WAVECAR does not fit (NBANDS/ENCUT/cell) WAVECAR moved aside, start from CHGCAR
  chgcar        CHGCAR unreadable / truncated            SCF from scratch (ICHARG = 2), or the SCF
                                                          is redone for bands/DOS/LOBSTER/DFPT
  zbrent        ZBRENT fatal error in a relax            IBRION 2 -> 1, continue from CONTCAR
  symmetry      symmetry-operation errors                SYMPREC = 1E-4 (ISYM = 0 if already set)
  edddav        ZHEGV / EDDDAV errors                    SCF: ALGO = All; relax: ALGO = Normal, then All;
                                                          NSCF: as nscf_diverged
  nscf_diverged NSCF energies |E| > 1e6                  NBANDS >= 1.3 x SCF NBANDS; ALGO Normal
                                                          (Davidson, default) -> Fast -> VeryFast
  relax_unconv  relax finished without convergence       continue from CONTCAR
  scf_nelm      SCF hit NELM                             ALGO = All (NELM >= 150), later steps redone
  nscf_unconv   bands/DOS hit NELM                       ALGO as nscf_diverged, NELM >= 150
  lobster_fail  NSCF finished, LOBSTER binary did not    rerun only the LOBSTER binary
  not_run       never ran (e.g. upstream failed)         submitted
  unknown       anything else                            script written, but only submitted with
                                                          --retry-unknown (error lines in its header)
  spilling_high LOBSTER finished, charge spilling > --spill-max %   reported; repair with
                ht_tools/fix_spilling.sh (more bands + LOBSTER basis scan)
  missing       a listed material ID was not found         reported only
  user_params   you changed its settings (set_params.sh)  the step and the later ones redone with them
  phase2_pending  convergence done, production steps never started: run
                submit_calculations.sh / run_calculations.sh yourself     reported only

00_convergence: broken points are moved aside, the test INCAR gets the fix of the class
(ALGO/NELM, KPAR, SYMPREC, walltime) and the test's run.sh is resubmitted - it skips
the points that are already done.  The production steps are not touched for it.

When the relax or the SCF is redone, every later step is redone too; otherwise only the
broken later steps.  Repairs are made in place.  The failed run of every redone step goes to
<folder>/round_<i>_<date>/<step>/ (i = the material's repair round): INCAR, KPOINTS, POSCAR
copied; OUTCAR, OSZICAR, EIGENVAL, CONTCAR, XDATCAR, lobster.out, ICO*LIST.lobster and the job
logs moved; WAVECAR, CHGCAR, CHG, vasprun.xml, DOSCAR, PROCAR and the other *.lobster deleted.  WHY.txt in it holds
the report of that repair; <folder>/repair_history.tsv gets one line per repaired step.

A material is held for review (its script is written, not submitted) when the cause is
unknown, the fix for its cause is already in the INCAR, the previous repair of the step had
the same cause, or it was repaired --max-rounds (5) times already.

Your own settings: ht_tools/set_params.sh writes <folder>/user_params.txt
("<step> TAG = value", plus KMESH = n1 n2 n3, TIME = HH:MM:SS, NODES = n).  They are applied
after the automatic fixes (they win), and a step whose settings changed after its last run is
redone.

Certainty: each cause gets a number from its evidence (SLURM record / OUTCAR 0.95, a known
error message 0.7-0.95, indirect 0.6, unknown 0.2); each fix a rule value (FIX_CERT), blended
with how earlier repairs of the same cause went in the runs checked (repair_history.tsv).

Writes into --out (default ./diagnosis_round<N>/, N = 1 + the last round in this folder):
  report.txt               per material: problem, cause, evidence, certainty, changes, backup
  diagnosis.tsv            one line per folder: layout, per-step status, planned fix, round,
                           cause and fix certainty
  failed_list.txt          the materials that get a fix script; review_list.txt the held ones
  fix/<label>.sh           per folder: guard (no queued jobs, outputs unchanged since the
                           diagnosis), backup, the fixes, then the resubmission chain
  submit_all_fixes.sh      runs every fix script (at most MAX_QUEUED jobs in the queue);
                           safe to re-run - finished ones are skipped (marker in .submitted/)

Layouts: ht (materials/<id>/job.sbatch <step>, ht_runs.py), slurm (vasp-agent-slurm:
<step>/run.sh with #SBATCH), local (vasp-agent: <step>/run.sh run in the foreground).

Usage (a highthroughput run: ht_tools/next_round.sh does all three):
  ./vasp_diagnose.py material_list.txt                 diagnose, write the scripts
  less diagnosis_round1/report.txt                     review
  nohup bash diagnosis_round1/submit_all_fixes.sh > diagnosis_round1/submit_all_fixes.log 2>&1 &
Options: --out DIR  --root DIR (more places to look up IDs)  --since YYYY-MM-DD (sacct
         window, default 14 days)  --kpar-max 8  --spill-max 5  --max-rounds 5  --retry-unknown
         --max-queued 200
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
import time
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
MAX_ROUNDS = 5             # repairs per material before it is held for review
HISTORY = 'repair_history.tsv'      # per material: one line per repaired step (written by the fix scripts)
USER_PARAMS = 'user_params.txt'     # per material: your own settings (ht_tools/set_params.sh)
# a one-node resubmission cannot go to a partition that needs >= 2 nodes (Rivanna: parallel)
ONE_NODE_PARTITION = {'parallel': 'standard'}

# What each cause means, in words, for report.txt.
CAUSE_TEXT = {
    'quota':         'the disk quota was full while VASP wrote its files',
    'oom':           'the job ran out of memory',
    'mpi_init':      'MPI could not start (usually a bad node)',
    'timeout':       'the step hit its time limit before it finished',
    'hang':          'VASP converged, then stopped making progress until the time limit '
                     '(MPI stall after the last electronic step, seen on multi-node runs)',
    'stopcar':       'VASP was stopped early by a STOPCAR left behind by an earlier run '
                     '(the step looks finished but is not converged)',
    'wavecar':       'the WAVECAR it started from does not fit (NBANDS, ENCUT or cell changed)',
    'chgcar':        'the CHGCAR it read is unreadable or truncated',
    'chgcar_missing': 'the SCF finished, but its CHGCAR (needed by the later steps) is missing or truncated',
    'zbrent':        'the relaxation line search failed (ZBRENT)',
    'symmetry':      'VASP could not handle the symmetry of the structure',
    'edddav':        'the Davidson eigensolver failed (EDDDAV / ZHEGV)',
    'nscf_diverged': 'the fixed-density run diverged (band energies above 1e6 eV)',
    'relax_unconv':  'the relaxation finished without reaching the force/stress criterion',
    'scf_nelm':      'the SCF used all NELM electronic steps without converging',
    'nscf_unconv':   'the fixed-density run used all NELM steps without converging',
    'lobster_fail':  'the VASP part finished, but the LOBSTER program did not',
    'not_run':       'the step never ran (an earlier step failed, or it was never submitted)',
    'user_params':   'you changed its settings with set_params.sh',
    'unknown':       'no known cause was found in the logs - read the error lines',
}
# How sure the diagnosis is of the cause, by the evidence it is based on (0..1).
CAUSE_CERT = {
    'sacct': 0.95,        # SLURM's own record (TIMEOUT, OUT_OF_MEMORY)
    'outputs': 0.95,      # read directly from OUTCAR / OSZICAR (NELM reached, not relaxed, |E| > 1e6)
    'pattern': 0.85,      # a known error message in the logs
    'weak': 0.6,          # only indirect evidence
    'unknown': 0.2,
}
PATTERN_CERT = {'quota': 0.9, 'oom': 0.9, 'mpi_init': 0.7, 'wavecar': 0.85, 'chgcar': 0.85,
                'zbrent': 0.95, 'symmetry': 0.85, 'edddav': 0.8}
# How likely the planned fix is to work, before this run has any repair history (0..1).
# report.txt blends it with what happened to earlier repairs of the same step and cause
# (FIX_PRIOR_WEIGHT counts as that many earlier repairs).
FIX_CERT = {'quota': 0.5, 'oom': 0.5, 'oom_lobster_mesh': 0.85, 'mpi_init': 0.7, 'timeout': 0.85,
            'hang': 0.75, 'hang_wall': 0.4, 'stopcar': 0.9, 'wavecar': 0.9, 'chgcar': 0.85,
            'chgcar_missing': 0.9, 'zbrent': 0.8, 'symmetry': 0.7, 'edddav': 0.6,
            'nscf_diverged': 0.6, 'relax_unconv': 0.8, 'scf_nelm': 0.7, 'nscf_unconv': 0.7,
            'lobster_fail': 0.5, 'not_run': 0.9, 'user_params': 0.8, 'unknown': 0.1}
FIX_PRIOR_WEIGHT = 4
# repeating these is progress, not a failed fix (a long relax continues from its CONTCAR)
REPEAT_OK = (('relax', 'timeout'), ('relax', 'relax_unconv'), ('relax', 'not_run'))

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

    def batch(self, step='scf'):
        """(partition, nodes, tasks per node) from the #SBATCH lines of the step's job script."""
        txt = read(os.path.join(self.path, 'job.sbatch')) if self.layout == 'ht' else read(self.f(step, 'run.sh'))
        g = lambda rx: (re.search(rx, txt) or [None, None])[1]
        return (g(r'#SBATCH\s+(?:--partition=|-p\s*)(\S+)'), to_int(g(r'#SBATCH\s+(?:--nodes=|-N\s*)(\d+)')),
                to_int(g(r'#SBATCH\s+--ntasks-per-node=(\d+)')))

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
    ico = m.f('lobster', 'ICOHPLIST.lobster')
    return 'finished in' in read(m.f('lobster', 'lobster.out')) and \
        os.path.isfile(ico) and os.path.getsize(ico) > 0


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


def ionic_done(m, step):
    """OSZICAR of a static step has its 'F=' line: the electronic loop had converged."""
    return re.search(r'^\s*1 F=', read(m.f(step, 'OSZICAR')), re.M) is not None


def classify(m, step, rec, provisional, point=None):
    """-> (problem class, up to 6 error lines, certainty 0..1, evidence) for a step that is
    not ok (point: one convergence-test folder, whose vasp.out/OUTCAR are read instead)."""
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
    errs = errs[:6]
    state = rec['state'] if rec else ''
    job = ('sacct: job %s %s after %s of %s' % (rec['id'], state, rec['elapsed'], rec['limit'])) if rec else ''
    static = step in ('scf', 'bands', 'dos', 'lobster') and not point

    def out(cls, cert, ev):
        return cls, errs, cert, '; '.join(x for x in (ev, job) if x)

    def time_limit(cert, ev):
        # a static step that had converged but never finished: it hung, more time will not help
        if static and ionic_done(m, step) and not finished(m, step):
            return out('hang', 0.8, ev + '; the electronic loop had converged (F= line in OSZICAR) '
                       'but VASP never reached the final timing block')
        return out('timeout', cert, ev)
    # a diverged NSCF never finishes, however long it runs: this beats a time limit
    if step in NSCF and nscf_diverged(m, step):
        return out('nscf_diverged', CAUSE_CERT['outputs'], '|E| > 1e6 eV in OSZICAR')
    # the job record of the last attempt is the most reliable source
    if state.startswith('TIMEOUT'):
        return time_limit(CAUSE_CERT['sacct'], 'SLURM state TIMEOUT')
    if state.startswith('OUT_OF_ME'):
        return out('oom', CAUSE_CERT['sacct'], 'SLURM state OUT_OF_MEMORY')
    # VASP stopped by the STOPCAR the walltime guard writes before the limit
    stopped = re.search(STOPPED, text + read(outcar, tail=200000), re.I)
    if stopped and provisional in ('relax_unconv', 'scf_nelm', 'nscf_unconv') and not point:
        # stopped, but not near the time limit: a STOPCAR left behind by an earlier run
        d = os.path.dirname(outcar)
        early = False
        if rec and not state.startswith('TIMEOUT'):
            el, lim = to_sec(rec['elapsed']), to_sec(rec['limit'])
            early = bool(el is not None and lim and el < lim - 1200)
        if early:
            return out('stopcar', 0.85, 'OUTCAR: "%s" long before the time limit' % stopped.group(0))
        if not rec and os.path.isfile(os.path.join(d, 'STOPCAR')) and not os.path.isfile(os.path.join(d, '.vf_resume')):
            return out('stopcar', 0.7, 'OUTCAR: "%s", a STOPCAR is still in the step folder' % stopped.group(0))
    if provisional in ('relax_unconv', 'scf_nelm', 'nscf_unconv', 'nscf_diverged', 'lobster_fail') \
            and not stopped:
        ev = {'relax_unconv': 'OUTCAR finished without "reached required accuracy"',
              'scf_nelm': 'OUTCAR finished without "EDIFF is reached"',
              'nscf_unconv': 'OUTCAR finished without "EDIFF is reached"',
              'nscf_diverged': '|E| > 1e6 eV in OSZICAR',
              'lobster_fail': 'VASP finished; lobster.out has no "finished in" (or ICOHPLIST is empty)'}[provisional]
        return out(provisional, 0.8 if provisional == 'lobster_fail' else CAUSE_CERT['outputs'], ev)
    for cls, rx in PAT:
        hit = re.search(rx, text, re.I)
        if hit:
            if cls == 'edddav' and step in NSCF:
                return out('nscf_diverged', PATTERN_CERT[cls], 'log: "%s"' % hit.group(0))
            return out(cls, PATTERN_CERT[cls], 'log: "%s"' % hit.group(0))
    if stopped:
        return time_limit(0.75, 'OUTCAR: "%s" (the walltime guard stops VASP 15 min before the limit)'
                          % stopped.group(0))
    lim = re.search(r'DUE TO TIME LIMIT|time limit hit', text)
    if lim:
        return time_limit(0.9, 'log: "%s"' % lim.group(0))
    if provisional:
        return out(provisional, CAUSE_CERT['weak'], 'outputs only')
    if not os.path.isfile(outcar) and not log.strip():
        return out('not_run', 0.9, 'no OUTCAR and no job log')
    return out('unknown', CAUSE_CERT['unknown'], 'no known error message')


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
            cls, errs, cert, ev = classify(m, key, rec, 'scf_nelm' if nelm else None, point=first)
        else:
            cls, errs, cert, ev = 'scf_nelm', [], CAUSE_CERT['outputs'], 'OUTCAR finished without "EDIFF is reached"'
        bad[t] = ('%s (%d of %d points)' % (cls, len(broken) + len(nelm), len(conv_points(m, t))),
                  cls, errs, broken + nelm, nelm, cert, ev)
    return bad


def conv_not_converged(m):
    """Parameters choose_params.py could not converge in the tested range."""
    txt = read(os.path.join(m.path, CONV_DIR, 'converged_params.txt'))
    return re.findall(r'^(\w+)_CONVERGED\s*=\s*no', txt, re.M)


def conv_plan(m, bad, recs, args):
    """-> (bash fix lines, [(test, jobname, walltime, extra sbatch opts)])"""
    lines, subs = [], []
    for t, (_, cls, _, pts, nelm, _, _) in sorted(bad.items()):
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


def plan(m, status, cls_of, recs, args, user=None):
    """-> (redo steps, bash lines (backup first, then the fixes), sbatch extras per step,
    walltime per step, step dirs whose INCAR changes, fix-certainty key per step)
    user: {step: [(KEY, value)]} from user_params.txt - applied last, so it wins."""
    user = user or {}
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
        # LOBSTER NSCF (ISYM=0): memory grows with the full k-mesh -> halve it
        if 'lobster' in redo and cls_of.get('lobster') == 'oom':
            kp = read(m.f('lobster', 'KPOINTS')).split('\n')
            try:
                mesh = [int(x) for x in kp[3].split()[:3]]
                auto = kp[2].strip()[:1].upper() in ('G', 'M') and kp[1].strip() == '0'
            except (IndexError, ValueError):
                mesh, auto = [], False
            if auto and len(mesh) == 3 and max(mesh) > 2:
                new = [1 if n == 1 else max(2, (n // 2 + 1) // 2 * 2) for n in mesh]
                d = STEP_DIR['lobster']
                lines.append("sed -i.bak -e '4s/.*/  %d  %d  %d/' -e '1s/$/ [vasp_diagnose: %dx%dx%d halved for memory]/' %s/KPOINTS"
                             " && rm -f %s/KPOINTS.bak || exit 1" % tuple(new + mesh + [d, d]))
                say('lobster: k-mesh %dx%dx%d -> %dx%dx%d (ISYM=0 memory)' % tuple(mesh + new))
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
    # nodes: a step that hung after convergence on several nodes reruns on one node, and so
    # do the other redone static steps; a step whose last run used fewer nodes than the job
    # script asks for (a deliberate one-node rerun) keeps that node count
    nodes_of = {}
    part, nn, tpn = m.batch()

    def last_nodes(s):     # nodes of the step's last run: OUTCAR "running on N total cores", else sacct
        r = re.search(r'running on\s+(\d+)\s+total cores', read(m.f(s, 'OUTCAR'), head=20000))
        if r and tpn:
            return max(1, -(-int(r.group(1)) // tpn))
        try:
            return int(recs[s].get('nodes'))
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
    hung, hung_wall = [s for s in first_bad if cls_of.get(s) == 'hang'], False
    for s in hung:
        if (last_nodes(s) or nn or 1) > 1:
            for t in redo:
                if t != 'relax':
                    nodes_of[t] = 1
            say('%s: hung after convergence on %d nodes -> %s on 1 node' % (
                s, last_nodes(s) or nn, ', '.join(t for t in redo if t != 'relax')))
        elif s in wall:
            wall[s] = from_sec(2 * to_sec(wall[s]))
            hung_wall = True
            say('%s: hung after convergence on 1 node -> walltime %s' % (s, wall[s]))
    for s in redo:
        last = last_nodes(s)
        if s not in nodes_of and last and nn and last < nn and cls_of.get(s) != 'oom':
            nodes_of[s] = last
            say('%s: %d node(s), as in its last run' % (s, last))
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
        if c_scf == 'stopcar':
            say('scf: the STOPCAR left by an earlier run is removed; scf and the steps after it are redone')
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
                lines.append('drop_file 01_relax WAVECAR')
                say('scf: relax WAVECAR moved aside, start from the relaxed CHGCAR')
            if not (os.path.isfile(rchg) and chg_complete(rchg)):
                lines.append('drop_file 01_relax CHGCAR; drop_file 01_relax WAVECAR')
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
            # fixed-density eigensolver ladder: Davidson (Normal, VASP's default) -> Fast
            # (Davidson start, then RMM-DIIS) -> VeryFast (RMM-DIIS); then held for review.
            # With a fixed density pure Davidson can run away (Ba2YReO6, Sr2YReO6: |E| -> 1e10);
            # ALGO = All (CG) crashes with ICHARG = 11.
            cur = incar(m.f(s, 'INCAR')).get('ALGO', 'Normal').strip().upper()
            algo = 'VeryFast' if cur.startswith(('F', 'V')) else 'Fast'
            tag(STEP_DIR[s], 'ALGO', algo)
            if to_int(incar(m.f(s, 'INCAR')).get('NELM'), 60) < 150:
                tag(STEP_DIR[s], 'NELM', 150)
            if STEP_DIR[s] in changed:
                say('%s: %s -> ALGO = %s%s' % (s, cls_of[s], algo, more))
    for s in ('relax', 'bands', 'dos', 'lobster'):
        if s in redo and cls_of.get(s) == 'stopcar':
            say('%s: the STOPCAR left by an earlier run is removed' % s)
    # your own settings (set_params.sh): applied last, so they win over the fixes above
    for s in redo:
        d = STEP_DIR[s]
        for key, val in user.get('all', []) + user.get(s, []):
            if key == 'TIME':
                wall[s] = val
            elif key == 'NODES':
                nodes_of[s] = to_int(val, 1)
            elif key == 'KMESH':
                mesh = val.split()
                lines.append("printf 'Gamma mesh set with set_params.sh\\n0\\nGamma\\n  %s  %s  %s\\n  0  0  0\\n' > %s/KPOINTS"
                             % (mesh[0], mesh[1], mesh[2], d))
            elif val == '':
                lines.append('del_tag %s/INCAR %s' % (d, key))
                changed.add(d)
            else:
                lines.append('set_tag %s/INCAR %s %s' % (d, key, shlex.quote(val)))
                changed.add(d)
            say('%s: your setting %s%s' % (s, key, (' = ' + val) if val else ' removed'))
    # node count -> sbatch options; KPAR must divide the new number of ranks
    for s, n in sorted(nodes_of.items()):
        if s not in extra or not tpn:
            continue
        extra[s] = [x for x in extra[s] if not x.startswith(('--nodes', '--ntasks', '--partition'))]
        extra[s] += ['--nodes=%d' % n, '--ntasks-per-node=%d' % tpn]
        if n == 1 and part in ONE_NODE_PARTITION:
            extra[s].append('--partition=%s' % ONE_NODE_PARTITION[part])
        d = STEP_DIR[s]
        kp = None
        for ln in lines:
            r = re.match(r"set_tag %s/INCAR KPAR '?(\d+)" % d, ln)
            if r:
                kp = int(r.group(1))
        kp = kp or to_int(incar(os.path.join(m.path, d, 'INCAR')).get('KPAR'))
        if kp and (n * tpn) % kp:
            k = largest_divisor(n * tpn, kp)
            lines.append('set_tag %s/INCAR KPAR %d' % (d, k))
            say('%s: KPAR %d -> %d (divisor of %d ranks)' % (s, kp, k, n * tpn))
    # the failed run of every redone step -> the round backup, before anything is changed
    first = []
    for s in redo:
        if s == 'lobster' and binary_only:
            first.append('stash 08_lobster lobster lobster-only; echo post > 08_lobster/.vf_resume')
            say('lobster: NSCF kept, only the LOBSTER binary reruns')
        else:
            first.append('stash %s %s' % (STEP_DIR[s], s))
    # which prior of FIX_CERT describes the fix of each broken step
    fkey = {}
    for s in first_bad:
        c = cls_of.get(s, 'unknown')
        if c == 'oom' and s == 'lobster' and any('halved for memory' in x for x in lines):
            c = 'oom_lobster_mesh'
        if c == 'hang' and hung_wall:
            c = 'hang_wall'
        if c == 'not_run' and fkey:
            continue            # follows from the broken step before it
        fkey[s] = c
    return redo, first + lines, extra, wall, changed, fkey


# ── bash pieces ──────────────────────────────────────────────────────────────
HELPERS = r'''STAMP=$(date +%Y%m%d_%H%M)
RB=round_${ROUND}_$STAMP    # backup of the failed run(s) of this round
set_tag() {   # file TAG value: replace every TAG line, then append
    sed -i.bak -E "/^[[:space:]]*$2[[:space:]]*=/d" "$1" && rm -f "$1.bak"
    printf '%s = %s\n' "$2" "$3" >> "$1"
}
del_tag() { sed -i.bak -E "/^[[:space:]]*$2[[:space:]]*=/d" "$1" && rm -f "$1.bak"; }
KEEP="OUTCAR OSZICAR EIGENVAL CONTCAR XDATCAR vasp.out lobsterout lobster.out
      ICOHPLIST.lobster ICOBILIST.lobster ICOOPLIST.lobster CHARGE.lobster"
DROP="CHG CHGCAR WAVECAR WAVEDER vasprun.xml DOSCAR PROCAR IBZKPT PCDAT REPORT vaspout.h5 ELFCAR LOCPOT
      STOPCAR .vf_resume .vf_restarts"
stash() {     # step-dir step [lobster-only]: the failed run -> $RB/<step-dir>/
    # inputs copied (INCAR KPOINTS POSCAR), small outputs (with ICO*LIST) and the job logs
    # moved, large regenerable files (WAVECAR, CHGCAR, vasprun.xml, *CAR.lobster) deleted
    local d=$1 s=$RB/$1 f
    mkdir -p "$s"
    for f in INCAR KPOINTS POSCAR lobsterin; do [ -f "$d/$f" ] && cp -p "$d/$f" "$s/"; done
    if [ "${3:-}" = lobster-only ]; then
        for f in lobsterout lobster.out ICOHPLIST.lobster ICOBILIST.lobster ICOOPLIST.lobster CHARGE.lobster; do
            [ -e "$d/$f" ] && mv -f "$d/$f" "$s/"
        done
        rm -f "$d"/*.lobster "$d/STOPCAR" "$d/.vf_resume" "$d/.vf_restarts"
    else
        for f in $KEEP "$d"/OUTCAR.* "$d"/OSZICAR.* "$d"/XDATCAR.*; do
            f=${f#"$d/"}; [ -e "$d/$f" ] && mv -f "$d/$f" "$s/"
        done
        for f in $DROP; do rm -f "$d/$f"; done
        rm -f "$d"/*.lobster
    fi
    for f in "$d"/slurm-*.out "$d"/slurm-*.err "$NAME-${NAME}_$2"-*.out "$NAME-${NAME}_$2"-*.err; do
        [ -e "$f" ] && mv -f "$f" "$s/"
    done
    return 0
}
stash_file() { [ -e "$1/$2" ] || return 0; mkdir -p "$RB/$1"; mv -f "$1/$2" "$RB/$1/"; }
drop_file() { rm -f "$1/$2"; }
continue_relax() {   # newest complete CONTCAR (or last XDATCAR frame) of the failed run -> 01_relax/POSCAR
    local r=01_relax b=$RB/01_relax nat
    nat=$(sed -n 7p $r/POSCAR | awk '{s=0; for (i=1;i<=NF;i++) s+=$i; print s}')
    [ -f $r/POSCAR.orig ] || cp -p $r/POSCAR $r/POSCAR.orig
    if [ -s $b/CONTCAR ] && [ "$(wc -l < $b/CONTCAR)" -ge $((8 + nat)) ]; then
        cp $b/CONTCAR $r/POSCAR; echo "  relax: continues from CONTCAR"
    elif [ -s $b/XDATCAR ] && ( cd $b && python3 - <<'PYX'
lines = open('XDATCAR').read().splitlines()
idx = [i for i, l in enumerate(lines) if l.strip().lower().startswith('direct configuration')]
nat = sum(int(x) for x in lines[6].split())
if not idx or len(lines) < idx[-1] + 1 + nat:
    raise SystemExit(1)
last = idx[-1]
head = lines[last - 7:last] if last >= 7 and lines[last - 7].strip() == lines[0].strip() else lines[:7]
open('POSCAR.next', 'w').write('\n'.join(head + ['Direct'] + lines[last + 1:last + 1 + nat]) + '\n')
PYX
    ); then mv $b/POSCAR.next $r/POSCAR; echo "  relax: continues from the last XDATCAR frame"
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


def write_script(path, m, status, errs, redo_txt, fix, submit, out_dir, rnd=1, why='', hist=()):
    """hist: (step, cause, cause certainty, fix text) per broken step -> repair_history.tsv"""
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
             'D=%s; OUT=%s; LABEL=%s; NAME=%s; ROUND=%d' % (q(m.path), q(out_dir), q(m.label), q(m.name), rnd),
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
    body += ['mkdir -p "$RB"', "cat > \"$RB/WHY.txt\" <<'VF_WHY'", why.rstrip('\n'), 'VF_WHY']
    body += fix
    body += submit
    for h in hist:
        body.append("printf '%%s\\t%%s\\t%%s\\t%%s\\t%%s\\t%%s\\t%%s\\n' %s \"$STAMP\" %s %s %s %s %s >> %s"
                    % (rnd, q(h[0]), q(h[1]), q('%.2f' % h[2]), q(h[3]), q(os.path.basename(out_dir)), HISTORY))
    body += ['echo "  backup of the failed run: $D/$RB/"',
             'mkdir -p "$OUT/.submitted" && touch "$OUT/.submitted/$LABEL"', '']
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
               '-o', 'JobID,JobName,State,Elapsed,Timelimit,NNodes'])
    for line in txt.splitlines():
        p = line.split('|')
        if len(p) < 5 or not p[0].split('_')[0].isdigit():
            continue
        jid = int(p[0].split('_')[0].split('.')[0])
        if p[1] not in recs or jid >= recs[p[1]]['num']:
            recs[p[1]] = {'num': jid, 'id': str(jid), 'state': p[2], 'elapsed': p[3], 'limit': p[4],
                          'nodes': p[5] if len(p) > 5 else ''}
    return recs


# ── repair history, your settings, certainty ─────────────────────────────────
def rounds_done(m):
    """Earlier repairs of this material = its round_<i>_<date>/ (and older stale_<date>/) backups."""
    return len([d for d in os.listdir(m.path) if re.match(r'(round_\d+|stale)_', d)
                and os.path.isdir(os.path.join(m.path, d))])


def read_history(m):
    out = []
    for line in read(os.path.join(m.path, HISTORY)).splitlines():
        p = line.split('\t')
        if len(p) >= 5 and p[0].isdigit():
            out.append({'round': int(p[0]), 'stamp': p[1], 'step': p[2], 'cause': p[3],
                        'fix': p[5] if len(p) > 5 else ''})
    return out


def read_user(m):
    """user_params.txt -> {step: [(KEY, value, 'YYYY-MM-DD HH:MM')]}, the last line of a KEY wins."""
    out = {}
    for line in read(os.path.join(m.path, USER_PARAMS)).splitlines():
        body, _, when = line.partition('# set ')
        r = re.match(r'\s*(\w+)\s+([A-Za-z_0-9]+)\s*=\s*(.*?)\s*$', body)
        if not r:
            continue
        step, key, val = r.group(1).lower(), r.group(2).upper(), r.group(3)
        lst = [x for x in out.get(step, []) if x[0] != key]
        out[step] = lst + [(key, val, when.strip())]
    return out


def user_forced(m, user):
    """Steps whose settings you changed after their last run (they are redone)."""
    forced = {}
    for s in m.steps:
        items = user.get(s, []) + user.get('all', [])
        newest = max([x[2] for x in items] or [''])
        if not newest:
            continue
        try:
            t = time_of(newest)
        except ValueError:
            continue
        out = m.f(s, 'OUTCAR')
        if not os.path.isfile(out) or os.path.getmtime(out) < t:
            forced[s] = newest
    return forced


def time_of(stamp):
    fmt = '%Y-%m-%d %H:%M:%S' if len(stamp) >= 19 else '%Y-%m-%d %H:%M'
    return time.mktime(datetime.datetime.strptime(stamp[:19], fmt).timetuple())


def repair_record(mats):
    """Did earlier repairs work?  {cause: [worked, judged]} over all materials.
    A repair worked if its step is ok now, or failed later for a different cause."""
    tally = {}
    for hist, status in mats:
        by_step = {}
        for e in hist:
            by_step.setdefault(e['step'], []).append(e)
        for step, es in by_step.items():
            for i, e in enumerate(es):
                if i + 1 < len(es):
                    ok = es[i + 1]['cause'] != e['cause']
                else:
                    cur = (status or {}).get(step)
                    if cur is None or cur.startswith(('not_run', 'user_params')):
                        continue        # queued, or not run yet: not judged
                    ok = cur.startswith('ok') or cur.split()[0] != e['cause']
                t = tally.setdefault(e['cause'], [0, 0])
                t[0] += int(ok)
                t[1] += 1
    return tally


def fix_certainty(key, cause, tally):
    prior = FIX_CERT.get(key, FIX_CERT.get(cause, 0.5))
    won, n = tally.get(cause, [0, 0])
    return (prior * FIX_PRIOR_WEIGHT + won) / float(FIX_PRIOR_WEIGHT + n), prior, won, n


def report_block(r, tally, max_rounds):
    """Text block of one material for report.txt (also the WHY.txt of its backup)."""
    L = ['=' * 100,
         '%s   repair round %d of at most %d' % (r['label'], r['round'], max_rounds),
         '  folder : %s' % r['path']]
    for s, cls, cc, ev, errs in r['steps']:
        L.append('  %-8s problem : %s   (certainty %.2f)' % (s, cls, cc))
        L.append('           cause   : %s' % CAUSE_TEXT.get(cls, cls))
        if ev:
            L.append('           seen    : %s' % ev)
        for e in errs[:2]:
            L.append('           log     | %s' % e)
    for s, key, cls in r['fkeys']:
        c, prior, won, n = fix_certainty(key, cls, tally)
        hist = ('; %d of %d earlier %s repairs in these runs worked' % (won, n, cls)) if n else '; no earlier repairs of this cause yet'
        L.append('  %-8s fix certainty %.2f  (rule %.2f%s)' % (s, c, prior, hist))
    if r['says']:
        L.append('  changes : ' + r['says'][0])
        L += ['            ' + x for x in r['says'][1:]]
    L.append('  redo    : %s' % (r['redo'] or '-'))
    if r['user']:
        L.append('  your settings (user_params.txt): ' + '; '.join(r['user']))
    if r['script']:
        L.append('  backup  : <folder>/round_%d_<date>/ (INCAR KPOINTS POSCAR OUTCAR OSZICAR EIGENVAL ICO*LIST logs)' % r['round'])
        L.append('  script  : %s   %s' % (r['script'], ('HELD - ' + r['hold']) if r['hold'] else '-> submitted by submit_all_fixes.sh'))
    return '\n'.join(L) + '\n'


def next_round_dir(base):
    n = [int(x[len('diagnosis_round'):]) for x in os.listdir(base)
         if re.match(r'diagnosis_round\d+$', x)]
    return os.path.join(base, 'diagnosis_round%d' % (max(n or [0]) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('list', help='list file: material IDs and/or folders, one per line')
    ap.add_argument('--root', action='append', default=[],
                    help='also look up material IDs under this folder (repeatable)')
    ap.add_argument('--out', default=None, help='output folder (default: ./diagnosis_round<N>, N = next round)')
    ap.add_argument('--since', default=(datetime.date.today() - datetime.timedelta(days=14)).isoformat())
    ap.add_argument('--kpar-max', type=int, default=8)
    ap.add_argument('--spill-max', type=float, default=5.0)
    ap.add_argument('--max-rounds', type=int, default=MAX_ROUNDS,
                    help='repairs per material before it is held for review (default %d)' % MAX_ROUNDS)
    ap.add_argument('--retry-unknown', action='store_true')
    ap.add_argument('--max-queued', type=int, default=200)
    args = ap.parse_args()

    roots = [os.path.dirname(os.path.abspath(args.list))] + \
            [os.path.abspath(os.path.expanduser(r)) for r in args.root]
    folders, missing = read_list(args.list, [r for r in roots if os.path.isdir(r)])
    out_dir = os.path.abspath(args.out) if args.out else next_round_dir(os.getcwd())
    os.makedirs(os.path.join(out_dir, 'fix'), exist_ok=True)
    recs = sacct_records(args.since)
    queue = set(run(['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%j']).split())
    rows, todo, mats = [], [], []
    count = Counter()
    for mid in missing:
        count['missing'] += 1
        rows.append([mid, '-', '-', 'not found under ' + ' '.join(roots), 'missing', '', '', '', ''])
    for p in folders:
        m = Material(p)
        status, cls_of, errs, mrecs, cert, ev = {}, {}, {}, {}, {}, {}
        hist = read_history(m)
        if m.queued(queue):
            count['queued'] += 1
            rows.append([m.label, m.layout, m.path, 'queued', 'queued', '', '', '', ''])
            mats.append((hist, None))
            continue
        cbad = conv_check(m, recs)
        for t in m.conv:
            key = 'conv:' + t
            if t in cbad:
                status[key], cls_of[key], errs[key] = cbad[t][0], cbad[t][1], cbad[t][2]
                cert[key], ev[key] = cbad[t][5], cbad[t][6]
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
            cls, e, c, v = classify(m, s, mrecs[s], st)
            status[s], cls_of[s], errs[s], cert[s], ev[s] = cls, cls, e, c, v
        # your settings changed after a step's last run: the step is redone
        user = read_user(m)
        for s, when in user_forced(m, user).items():
            if status[s].startswith('ok'):
                status[s], cls_of[s], errs[s], cert[s] = 'user_params', 'user_params', [], 1.0
                ev[s] = 'user_params.txt changed %s, after the last run of this step' % when
        mats.append((hist, dict(status)))
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
            rows.append([m.label, m.layout, m.path, steps_txt, act, '', '', '', ''])
            continue
        fix, submit, redo_txt, holds, fkeys = [], [], [], [], []
        if cbad:
            cfix, csubs = conv_plan(m, cbad, recs, args)
            fix += cfix
            submit += conv_submit_lines(m, csubs)
            redo_txt += ['%s/%s' % (CONV_DIR, t) for t in sorted(cbad)]
            fkeys += [('conv:' + t, cbad[t][1], cbad[t][1]) for t in sorted(cbad)]
            if phase2:
                submit.append('echo "  when the tests are done: check them and start the production steps '
                              '(submit_calculations.sh / run_calculations.sh)"')
        if bad:
            if m.layout == 'ht' and any(s not in HT_STEPS for s in bad):
                notes.append('job.sbatch cannot run: ' + ','.join(s for s in bad if s not in HT_STEPS))
            redo, pfix, extra, wall, changed, fk = plan(m, status, cls_of, mrecs, args,
                                                        dict((k, [(a, b) for a, b, _ in v]) for k, v in user.items()))
            last = {}
            for e in hist:
                last[e['step']] = e['cause']
            for i, s in enumerate(bad):
                c = cls_of[s]
                if c == 'not_run' and i:
                    continue            # follows from the broken step before it
                if c in INCAR_FIX and STEP_DIR[s] not in changed:      # its INCAR fix is already there
                    status[s] += ' (fix already in INCAR)'
                    holds.append('%s: the fix for %s is already in its INCAR and did not help' % (s, c))
                elif last.get(s) == c and (s, c) not in REPEAT_OK and c != 'user_params':
                    holds.append('%s: same cause (%s) as after the previous repair' % (s, c))
            if m.layout == 'ht':
                redo = [s for s in redo if s in HT_STEPS]
            fix += pfix
            submit += submit_lines(m, redo, extra, wall)
            redo_txt.append(' -> '.join(redo))
            fkeys += [(s, fk.get(s, cls_of[s]), cls_of[s]) for s in bad if s in fk]
            steps_txt = ' '.join(['%s:%s' % kv for kv in status.items()] + notes)
        rnd = rounds_done(m) + 1
        if rnd > args.max_rounds:
            holds.append('already repaired %d times (--max-rounds %d)' % (rnd - 1, args.max_rounds))
        if any(c == 'unknown' for c in cls_of.values()) and not args.retry_unknown:
            holds.append('unknown cause (read the log lines; --retry-unknown submits it anyway)')
        first = bad[0] if bad else 'conv:' + sorted(cbad)[0]
        main_cls = cls_of[first]
        count[main_cls] += 1
        script = os.path.join(out_dir, 'fix', m.label + '.sh')
        says = [re.sub(r'^echo "  (.*)"$', r'\1', x) for x in fix if x.startswith('echo "  ')]
        r = {'label': m.label, 'path': m.path, 'round': rnd, 'script': script,
             'hold': '; '.join(holds), 'redo': ' ; '.join(x for x in redo_txt if x), 'says': says,
             'steps': [(s, cls_of[s], cert.get(s, 0.5), ev.get(s, ''), errs.get(s, []))
                       for s in list(status) if s in cls_of],
             'fkeys': fkeys,
             'user': ['%s %s = %s' % (st_, k, v) for st_, lst in sorted(user.items()) for k, v, _ in lst]}
        todo.append((r, m, status, errs, fix, submit))
        rows.append([m.label, m.layout, m.path, steps_txt,
                     ('REVIEW ' if holds else '') + main_cls + ' -> ' + ' ; '.join(
                         x.replace(' -> ', '>') for x in redo_txt if x), script, str(rnd)])

    # certainty of each fix: its rule, blended with how earlier repairs of that cause went
    tally = repair_record(mats)
    scripts, review, blocks = [], [], []
    for r, m, status, errs, fix, submit in todo:
        block = report_block(r, tally, args.max_rounds)
        blocks.append(block)
        roots = set(k[0] for k in r['fkeys'])
        hist = [(s, cls, cc, '; '.join(x for x in r['says'] if x.startswith(s + ':')) or r['redo'])
                for s, cls, cc, _, _ in r['steps'] if s in roots and not s.startswith('conv:')]
        write_script(r['script'], m, status, errs, r['redo'], fix, submit, out_dir,
                     rnd=r['round'], why=block, hist=hist)
        (review if r['hold'] else scripts).append(r)
    cc = dict((r['label'], (r['steps'][0][2] if r['steps'] else '',
                            min([fix_certainty(k, c, tally)[0] for _, k, c in r['fkeys']] or [0])))
              for r, _, _, _, _, _ in todo)
    for row in rows:
        if row[0] in cc:
            row[7:9] = ['%.2f' % cc[row[0]][0], '%.2f' % cc[row[0]][1]]

    with open(os.path.join(out_dir, 'diagnosis.tsv'), 'w') as fh:
        fh.write('label\tlayout\tpath\tsteps\taction\tscript\tround\tcause_certainty\tfix_certainty\n')
        for row in rows:
            fh.write('\t'.join((row + [''] * 9)[:9]) + '\n')
    with open(os.path.join(out_dir, 'report.txt'), 'w') as fh:
        fh.write('vasp_diagnose report  %s  (%s)\n' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), args.list))
        fh.write('%d materials checked: %s\n' % (len(rows), ', '.join('%s %d' % kv for kv in count.most_common())))
        fh.write('%d fix scripts will be submitted, %d are held for you to review.\n' % (len(scripts), len(review)))
        fh.write('Certainty: of the cause, from the kind of evidence (SLURM record or OUTCAR 0.95, a known\n'
                 'error message 0.7-0.95, indirect 0.6, unknown 0.2); of the fix, its rule value blended with\n'
                 'how earlier repairs of the same cause went in these runs.\n')
        if count['spilling_high']:
            fh.write('%d material(s) have a LOBSTER charge spilling above %g %% (spilling_high in diagnosis.tsv);\n'
                     'redo them with more bands and a basis scan: ht_tools/fix_spilling.sh --above %g --submit\n'
                     % (count['spilling_high'], args.spill_max, args.spill_max))
        fh.write('Change a setting yourself:  bash ~/VASP-Flow/ht_tools/set_params.sh ID STEP "TAG=VALUE; ..."'
                 '  then run the round again.\n\n')
        held = [b for b, (r, _, _, _, _, _) in zip(blocks, todo) if r['hold']]
        fh.write(''.join(held + [b for b in blocks if b not in held]))
        for mid in missing:
            fh.write('=' * 100 + '\n%s   not found (looked under %s)\n' % (mid, ' '.join(roots)))
    with open(os.path.join(out_dir, 'failed_list.txt'), 'w') as fh:
        for r in scripts + review:
            fh.write('%s   # %s\n' % (r['path'], r['steps'][0][1] if r['steps'] else ''))
    with open(os.path.join(out_dir, 'review_list.txt'), 'w') as fh:
        for r in review:
            fh.write('%s   # %s\n' % (r['path'], r['hold']))
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
        for r in scripts:
            fh.write('run_fix %s\n' % shlex.quote(r['script']))
        if review:
            fh.write('\n# held for review (see report.txt) - uncomment to submit anyway:\n')
            for r in review:
                fh.write('# run_fix %s\n' % shlex.quote(r['script']))
        fh.write('echo "All fix scripts done.  Monitor: squeue -u ${USER:-$(whoami)}"\n')
    os.chmod(master, 0o755)

    print('Folders checked: %d   (layouts: %s)' % (len(rows), ', '.join(
        '%s %d' % kv for kv in sorted(Counter(r[1] for r in rows).items()))))
    for k, v in count.most_common():
        print('  %-14s %4d' % (k, v))
    print('Fix scripts: %d to submit, %d held for review' % (len(scripts), len(review)))
    if count['spilling_high']:
        print('%d material(s) with LOBSTER charge spilling > %g %%: bash ~/VASP-Flow/ht_tools/fix_spilling.sh '
              '--above %g  (dry run; --submit to redo them)' % (count['spilling_high'], args.spill_max, args.spill_max))
    for r in review:
        print('  held: %s  (%s)' % (r['label'], r['hold']))
    print('Report : %s\nTable  : %s\nSubmit : nohup bash %s > %s.log 2>&1 &' % (
        os.path.join(out_dir, 'report.txt'), os.path.join(out_dir, 'diagnosis.tsv'), master, master[:-3]))


if __name__ == '__main__':
    main()
