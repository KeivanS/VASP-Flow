#!/usr/bin/env python3
"""vasp-doctor.py -- diagnose VASP-Flow runs and write fix-and-resubmit scripts.

Give it a file that lists folders, one per line ('#' starts a comment).  A line can be
  * one material / project folder: it has some of 01_relax 02_scf 03_bands 04_dos
    06_dfpt 08_lobster;
  * a highthroughput folder (has materials/): every materials/<id> is checked;
  * any folder whose sub-folders are material folders (e.g. an ht-mp-scf run).

Diagnosis is read-only.  For every step it decides ok / problem from OUTCAR, OSZICAR,
lobster.out and the SLURM logs (+ sacct when available):

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
  edddav        ZHEGV / EDDDAV / sub-space errors        SCF: ALGO = All; relax: ALGO = Normal;
                                                          NSCF: as nscf_diverged
  nscf_diverged NSCF energies |E| > 1e6                  NBANDS >= 1.3 x SCF NBANDS, ALGO = Normal
  relax_unconv  relax finished without convergence       continue from CONTCAR
  scf_nelm      SCF hit NELM                             ALGO = All (NELM >= 150), later steps redone
  nscf_unconv   bands/DOS hit NELM                       ALGO = Normal, NELM >= 150
  lobster_fail  NSCF finished, LOBSTER binary did not    rerun only the LOBSTER binary
  not_run       never ran (e.g. upstream failed)         submitted
  unknown       anything else                            script written, but only submitted with
                                                          --retry-unknown (error lines in its header)
  spilling_high LOBSTER finished, charge spilling > --spill-max %   reported only

When the relax or the SCF is redone, every later step is redone too; otherwise only the
broken later steps.  Old outputs of every redone step are moved to <folder>/stale_<date>/
(nothing is deleted).

Writes into --out (default ./vasp_doctor_<date>/):
  diagnosis.tsv            one line per folder: layout, per-step status, planned fix
  fix/<label>.sh           per folder: guard (no queued jobs, outputs unchanged since the
                           diagnosis), the fixes, then the resubmission chain
  submit_all_fixes.sh      runs every fix script (at most MAX_QUEUED jobs in the queue);
                           safe to re-run - finished ones are skipped (marker in .submitted/)

Layouts: ht (materials/<id>/job.sbatch <step>, ht-semimetals), slurm (vasp-agent-slurm:
<step>/run.sh with #SBATCH), local (vasp-agent: <step>/run.sh run in the foreground).

Usage:
  ./vasp-doctor.py folders.txt                       diagnose, write the scripts
  less vasp_doctor_*/diagnosis.tsv vasp_doctor_*/fix/<label>.sh     review
  nohup bash vasp_doctor_*/submit_all_fixes.sh > fixes.log 2>&1 &
Options: --out DIR  --since YYYY-MM-DD (sacct window, default 14 days)  --kpar-max 8
         --spill-max 5  --retry-unknown  --max-queued 200
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
UNHANDLED = ('00_convergence', '05_wannier', '07_phonons')
NSCF = ('bands', 'dos', 'lobster')
DEFAULT_TIME = {'relax': '08:00:00', 'scf': '04:00:00', 'bands': '02:00:00',
                'dos': '02:00:00', 'dfpt': '08:00:00', 'lobster': '04:00:00'}
HT_TIME_KEY = {'relax': 'TIME_RELAX', 'scf': 'TIME_SCF', 'bands': 'TIME_BANDS',
               'lobster': 'TIME_LOBSTER'}
HT_STEPS = ('relax', 'scf', 'bands', 'lobster')     # what job.sbatch understands

PAT = [   # (class, regex) -- first match wins; searched in the job logs (+ OUTCAR tail)
    ('quota',    r'Disk quota exceeded|No space left on device|error during write'),
    ('oom',      r'insufficient virtual memory|cannot allocate memory|out of memory|'
                 r'oom[-_ ]kill|ud_ep\.c.*unhandled timeout'),
    ('mpi_init', r'PMPI_Init|MPI_Init_thread|MPIDI_OFI_mpi_init'),
    ('wavecar',  r'number of bands has changed|different cutoff or change in lattice|'
                 r'while reading WAVECAR'),
    ('chgcar',   r'end of record during read.*CHGCAR|charge density could not be read'),
    ('zbrent',   r'ZBRENT: fatal error'),
    ('symmetry', r'inverse of rotation matrix was not found|POSMAP internal error|'
                 r'internal error in subroutine SGRCON|PRICELV|RHOSYG'),
    ('edddav',   r'Error EDDDAV|ZHEGV failed|Sub-Space-Matrix is not hermitian'),
]
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
        parent = os.path.dirname(self.path)
        if os.path.isfile(os.path.join(self.path, 'job.sbatch')):
            self.layout = 'ht'
            self.root = os.path.dirname(parent) if os.path.basename(parent) == 'materials' else parent
            self.label = os.path.basename(self.root) + '__' + self.name
        else:
            runs = [read(self.f(s, 'run.sh')) for s in self.steps]
            self.layout = 'slurm' if any('#SBATCH' in r for r in runs) else 'local'
            self.root = parent
            self.label = os.path.basename(parent) + '__' + self.name
        self.env = self._ht_env() if self.layout == 'ht' else {}

    def f(self, step, name):
        return os.path.join(self.path, STEP_DIR[step], name)

    def jobname(self, step):
        return '%s_%s' % (self.name, step)

    def _ht_env(self):
        env = {}
        for line in read(os.path.join(self.root, 'env.sh')).splitlines():
            m = re.match(r'\s*(?:export\s+)?(TIME_[A-Z]+)=["\']?([0-9:-]+)', line)
            if m:
                env[m.group(1)] = m.group(2)
        return env

    def walltime(self, step):
        if self.layout == 'ht':
            return self.env.get(HT_TIME_KEY.get(step, ''), DEFAULT_TIME[step])
        m = re.search(r'#SBATCH\s+(?:--time=|-t\s*)(\S+)', read(self.f(step, 'run.sh')))
        return m.group(1) if m else DEFAULT_TIME[step]

    def ranks(self, step):
        m = re.search(r'running on\s+(\d+)\s+total cores', read(self.f(step, 'OUTCAR'), head=20000))
        if m:
            return int(m.group(1))
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
        if self.layout == 'ht':
            if rec:
                files = [os.path.join(self.path, '%s-%s-%s.%s' % (self.name, self.jobname(step), rec['id'], e))
                         for e in ('err', 'out')]
            if not any(os.path.isfile(x) for x in files):
                cand = sorted(glob.glob(os.path.join(self.path, '%s-%s-*.out' % (self.name, self.jobname(step)))),
                              key=os.path.getmtime)
                if cand:
                    files = [cand[-1][:-4] + '.err', cand[-1]]
        elif rec:
            for d in (self.path, os.path.join(self.path, STEP_DIR[step]), self.root):
                for e in ('err', 'out'):
                    p = os.path.join(d, 'slurm-%s.%s' % (rec['id'], e))
                    if os.path.isfile(p):
                        files.append(p)
        for name in ('vasp.out',):
            p = self.f(step, name)
            if os.path.isfile(p):
                files.append(p)
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


def classify(m, step, rec, provisional):
    """Problem class + up to 6 error lines for a step that is not ok."""
    log = m.logs(step, rec)
    text = log + '\n' + read(m.f(step, 'OUTCAR'), tail=20000)
    errs = []
    seen = set()
    for line in text.splitlines():
        line = line.strip()
        if ERRLINE.search(line) and not ERRSKIP.search(line) and line not in seen:
            seen.add(line)
            errs.append(line[:160])
    state = rec['state'] if rec else ''
    if provisional in ('relax_unconv', 'scf_nelm', 'nscf_unconv', 'nscf_diverged', 'lobster_fail') \
            and not state.startswith(('TIMEOUT', 'OUT_OF_ME')):
        return provisional, errs[:6]
    for cls, rx in PAT:
        if re.search(rx, text, re.I):
            if cls == 'edddav' and step in NSCF:
                return 'nscf_diverged', errs[:6]
            return cls, errs[:6]
    if state.startswith('OUT_OF_ME'):
        return 'oom', errs[:6]
    if state.startswith('TIMEOUT') or re.search(r'DUE TO TIME LIMIT|time limit hit', text):
        return 'timeout', errs[:6]
    if step in NSCF and nscf_diverged(m, step):
        return 'nscf_diverged', errs[:6]
    if provisional:
        return provisional, errs[:6]
    if not os.path.isfile(m.f(step, 'OUTCAR')) and not log.strip():
        return 'not_run', errs[:6]
    return 'unknown', errs[:6]


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
            lines.append('set_tag 01_relax/INCAR IBRION 1')
            say('relax: IBRION 2 -> 1 (ZBRENT)')
        if c_relax == 'symmetry':
            tags = incar(m.f('relax', 'INCAR'))
            lines.append('set_tag 01_relax/INCAR ISYM 0' if 'SYMPREC' in tags else 'set_tag 01_relax/INCAR SYMPREC 1E-4')
            say('relax: symmetry error -> %s' % ('ISYM = 0' if 'SYMPREC' in tags else 'SYMPREC = 1E-4'))
        if c_relax == 'edddav':
            lines.append('set_tag 01_relax/INCAR ALGO Normal')
            say('relax: ALGO = Normal (EDDDAV)')
        lines.append('continue_relax')
        if c_relax == 'wavecar':
            lines.append('rm -f 01_relax/WAVECAR')
    if 'scf' in redo:
        tags = incar(m.f('scf', 'INCAR'))
        if c_scf in ('scf_nelm', 'edddav'):
            lines.append('set_tag 02_scf/INCAR ALGO All')
            if to_int(tags.get('NELM'), 60) < 150:
                lines.append('set_tag 02_scf/INCAR NELM 150')
            say('scf: ALGO = All, NELM >= 150 (%s)' % c_scf)
        if c_scf == 'symmetry':
            lines.append('set_tag 02_scf/INCAR ISYM 0' if 'SYMPREC' in tags else 'set_tag 02_scf/INCAR SYMPREC 1E-4')
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
    for s in ('bands', 'dos'):
        if s in redo and cls_of.get(s) in ('nscf_diverged', 'nscf_unconv'):
            if cls_of[s] == 'nscf_diverged':
                nb = scf_nbands(m)
                if nb:
                    new = int(math.ceil(1.3 * nb / 2.0)) * 2
                    cur = to_int(incar(m.f(s, 'INCAR')).get('NBANDS'), 0)
                    if new > cur:
                        lines.append('set_tag %s/INCAR NBANDS %d' % (STEP_DIR[s], new))
            lines.append('set_tag %s/INCAR ALGO Normal' % STEP_DIR[s])
            if to_int(incar(m.f(s, 'INCAR')).get('NELM'), 60) < 150:
                lines.append('set_tag %s/INCAR NELM 150' % STEP_DIR[s])
            say('%s: %s -> ALGO = Normal%s' % (s, cls_of[s], ', more bands' if cls_of[s] == 'nscf_diverged' else ''))
    # stash old outputs of every redone step
    for s in redo:
        if s == 'lobster' and binary_only:
            lines.append('stash 08_lobster lobster-only; echo post > 08_lobster/.vf_resume')
            say('lobster: NSCF kept, only the LOBSTER binary reruns')
        else:
            lines.append('stash %s' % STEP_DIR[s])
    return redo, lines, extra, wall, binary_only


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
    files = [os.path.join(STEP_DIR[s], 'OUTCAR') for s in m.steps]
    if 'lobster' in m.steps:
        files.append('08_lobster/lobster.out')
    return files


def fingerprint(m):
    out = []
    for f in fingerprint_files(m):
        p = os.path.join(m.path, f)
        out.append('%s:%s' % (f, int(os.path.getmtime(p)) if os.path.exists(p) else 'none'))
    return ' '.join(out)


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


def write_script(path, m, status, errs, redo, fix, extra, wall, out_dir):
    q = shlex.quote
    steps_txt = '  '.join('%s: %s' % (s, status[s]) for s in m.steps)
    body = ['#!/bin/bash',
            '# vasp-doctor fix for %s' % m.label,
            '# folder : %s   (layout: %s)' % (m.path, m.layout),
            '# diagnosed %s:  %s' % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), steps_txt)]
    for s in m.steps:
        for e in errs.get(s, []):
            body.append('#   %s | %s' % (s, e))
    body += ['# resubmits: %s' % ' -> '.join(redo), 'set -u',
             'D=%s; OUT=%s; LABEL=%s' % (q(m.path), q(out_dir), q(m.label)),
             'cd "$D" || exit 1',
             '# guard: nothing of this folder queued, outputs unchanged since the diagnosis']
    if m.layout != 'local':
        body.append('if squeue -h -u "${USER:-$(whoami)}" -o "%%j" 2>/dev/null | grep -q "^%s_"; then '
                    'echo "-- $LABEL: jobs queued, skipped"; exit 0; fi' % re.escape(m.name))
    files = fingerprint_files(m)
    body += ['FP=""; for f in %s; do FP="$FP $f:$(stat -c %%Y "$f" 2>/dev/null || echo none)"; done' % ' '.join(files),
             'if [ "${FP# }" != %s ]; then echo "-- $LABEL: outputs changed since the diagnosis - '
             'run vasp-doctor.py again"; exit 0; fi' % q(fingerprint(m)),
             'echo "== $LABEL"', HELPERS.rstrip('\n')]
    body += fix
    body += submit_lines(m, redo, extra, wall)
    body += ['mkdir -p "$OUT/.submitted" && touch "$OUT/.submitted/$LABEL"', '']
    with open(path, 'w') as fh:
        fh.write('\n'.join(body))
    os.chmod(path, 0o755)


# ── collect folders ──────────────────────────────────────────────────────────
def is_material(p):
    return any(os.path.isdir(os.path.join(p, d)) for d, _ in STEPS)


def expand(line):
    p = os.path.abspath(os.path.expanduser(line))
    if not os.path.isdir(p):
        print('?? not a folder: %s' % line)
        return []
    if is_material(p):
        return [p]
    base = os.path.join(p, 'materials') if os.path.isdir(os.path.join(p, 'materials')) else p
    found = sorted(os.path.join(base, x) for x in os.listdir(base)
                   if os.path.isdir(os.path.join(base, x)) and is_material(os.path.join(base, x)))
    if not found:
        print('?? no step folders (01_relax, 02_scf, ...) in %s' % line)
    return found


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
    ap.add_argument('folders', help='file with one folder per line')
    ap.add_argument('--out', default='vasp_doctor_' + datetime.datetime.now().strftime('%Y%m%d_%H%M'))
    ap.add_argument('--since', default=(datetime.date.today() - datetime.timedelta(days=14)).isoformat())
    ap.add_argument('--kpar-max', type=int, default=8)
    ap.add_argument('--spill-max', type=float, default=5.0)
    ap.add_argument('--retry-unknown', action='store_true')
    ap.add_argument('--max-queued', type=int, default=200)
    args = ap.parse_args()

    folders = []
    for line in open(args.folders):
        line = line.split('#')[0].strip()
        if line:
            folders += expand(line)
    out_dir = os.path.abspath(args.out)
    os.makedirs(os.path.join(out_dir, 'fix'), exist_ok=True)
    recs = sacct_records(args.since)
    queue = set(run(['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%j']).split())
    rows, scripts, review = [], [], []
    count = Counter()
    for p in folders:
        m = Material(p)
        status, cls_of, errs, mrecs = {}, {}, {}, {}
        if any(j.startswith(m.name + '_') for j in queue):
            count['queued'] += 1
            rows.append([m.label, m.layout, m.path, 'queued', '', ''])
            continue
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
        notes = ' '.join('%s:unchecked' % d for d in m.unhandled)
        if not bad:
            spill = [s for s in m.steps if 'spilling' in status[s]]
            count['spilling_high' if spill else 'ok'] += 1
            rows.append([m.label, m.layout, m.path, ' '.join('%s:%s' % (s, status[s]) for s in m.steps) + notes,
                         'spilling_high' if spill else 'ok', ''])
            continue
        if m.layout == 'ht' and any(s not in HT_STEPS for s in bad):
            notes += ' job.sbatch cannot run: ' + ','.join(s for s in bad if s not in HT_STEPS)
        redo, fix, extra, wall, _ = plan(m, status, cls_of, mrecs, args)
        if m.layout == 'ht':
            redo = [s for s in redo if s in HT_STEPS]
        main_cls = cls_of[bad[0]]
        count[main_cls] += 1
        script = os.path.join(out_dir, 'fix', m.label + '.sh')
        write_script(script, m, status, errs, redo, fix, extra, wall, out_dir)
        unknown = any(cls_of.get(s) == 'unknown' for s in bad)
        (review if unknown and not args.retry_unknown else scripts).append(script)
        rows.append([m.label, m.layout, m.path, ' '.join('%s:%s' % (s, status[s]) for s in m.steps) + notes,
                     ('REVIEW ' if unknown and not args.retry_unknown else '') + main_cls + ' -> ' + '>'.join(redo),
                     script])

    with open(os.path.join(out_dir, 'diagnosis.tsv'), 'w') as fh:
        fh.write('label\tlayout\tpath\tsteps\taction\tscript\n')
        for r in rows:
            fh.write('\t'.join(r) + '\n')
    master = os.path.join(out_dir, 'submit_all_fixes.sh')
    with open(master, 'w') as fh:
        fh.write('#!/bin/bash\n# Runs every fix script written by vasp-doctor.py; safe to re-run\n'
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
            fh.write('\n# unknown cause - read the script header first; uncomment to submit:\n')
            for s in review:
                fh.write('# run_fix %s\n' % shlex.quote(s))
        fh.write('echo "All fix scripts done.  Monitor: squeue -u ${USER:-$(whoami)}"\n')
    os.chmod(master, 0o755)

    print('Folders checked: %d   (layouts: %s)' % (len(rows), ', '.join(
        '%s %d' % kv for kv in sorted(Counter(r[1] for r in rows).items()))))
    for k, v in count.most_common():
        print('  %-14s %4d' % (k, v))
    print('Fix scripts: %d to submit, %d to review (unknown cause)' % (len(scripts), len(review)))
    for s in review:
        print('  review: %s' % s)
    print('Report : %s\nSubmit : nohup bash %s > %s.log 2>&1 &' % (
        os.path.join(out_dir, 'diagnosis.tsv'), master, master[:-3]))


if __name__ == '__main__':
    main()
