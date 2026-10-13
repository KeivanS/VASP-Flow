#!/usr/bin/env python3
"""LOBSTER basis choice and charge-spilling scan (plain python3, no pymatgen needed).

The charge spilling measures how much of the occupied PAW wavefunctions the local
LOBSTER basis cannot represent; above a few percent the COHP/COBI/ICOHP values are
less reliable.  RWIGS has no effect (it only changes VASP's own projections).  What
lowers it:
  * a larger basis: pbeVaspFit2015 offers optional orbitals for 43 POTCARs (4p for
    3d metals, 5p/6p for 4d/5d metals, p for Li/Na/Mg/Be, 5d for Eu/Yb/Er, 4f/5f for
    La/Ac); the minimal and maximal sets are in lobster_basis_table.json (from
    pymatgen's BASIS_PBE_54_{min,standard,max}.yaml);
  * enough bands in the symmetry-off NSCF: NBANDS must be at least the number of
    basis functions; with NBANDS equal to it, many k-points can fail LOBSTER's
    orthonormalisation ("k-points could not be orthonormalized").

Commands (DIR = a finished 08_lobster folder: WAVECAR, POSCAR, POTCAR, INCAR, KPOINTS):
  lobster_basis.py info DIR                 candidate bases, basis-function counts, NBANDS
  lobster_basis.py nbands DIR [--factor F]  NBANDS for the largest candidate basis (x F, even)
  lobster_basis.py lobsterin DIR KIND       print basisfunctions lines (KIND = min|standard|max)
  lobster_basis.py scan DIR [--lobster BIN] [--max N] [--clean]
        run LOBSTER once per candidate basis in DIR/basis_scan/<k>/, write
        DIR/basis_scan.tsv, and copy the outputs with the lowest charge spilling into
        DIR (only if lower than the run already in DIR).  --clean deletes WAVECAR after.
"""
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, 'lobster_basis_table.json')
MULT = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
KINDS = ('min', 'standard', 'max')
OUTPUTS_KEEP = ('lobsterin', 'lobsterout', 'lobster.out')


def table():
    with open(TABLE) as f:
        return json.load(f)['basis']


def potcar_symbols(potcar):
    """POTCAR symbols in file order, from the TITEL lines (PAW_PBE Ga_d 06Sep2000 -> Ga_d)."""
    titel, first = [], []
    for line in open(potcar, errors='replace'):
        m = re.match(r'\s*TITEL\s*=\s*\S+\s+(\S+)', line)
        if m:
            titel.append(m.group(1))
        m = re.match(r'\s*PAW\S*\s+(\S+)\s+\d', line)     # first line of each block
        if m:
            first.append(m.group(1))
    return titel or first


def poscar_counts(poscar):
    """[(element, count)] from a VASP5 POSCAR (element line 6, counts line 7)."""
    lines = open(poscar).read().split('\n')
    els, cnt = lines[5].split(), lines[6].split()
    return list(zip(els, [int(c) for c in cnt]))


def nfunc(orbitals):
    return sum(MULT[o[-1]] for o in orbitals.split())


def candidates(dirname, max_n=12):
    """List of {element: orbitals} bases: every combination of the distinct
    min/standard/max options per element (the smallest first), at most max_n."""
    t = table()
    syms = potcar_symbols(os.path.join(dirname, 'POTCAR'))
    counts = poscar_counts(os.path.join(dirname, 'POSCAR'))
    if len(syms) != len(counts):
        raise SystemExit('POTCAR has %d entries, POSCAR %d species' % (len(syms), len(counts)))
    opts = []
    for (el, _), sym in zip(counts, syms):
        if sym not in t:
            raise SystemExit('no LOBSTER basis known for POTCAR %s' % sym)
        o = []
        for k in KINDS:
            if t[sym][k] not in o:
                o.append(t[sym][k])
        opts.append([(el, x) for x in sorted(o, key=nfunc)])
    combos = [dict(c) for c in itertools.product(*opts)]
    combos.sort(key=lambda b: n_basis(b, counts))
    return combos[:max_n], counts, syms


def n_basis(basis, counts):
    return sum(nfunc(basis[el]) * n for el, n in counts)


def basis_for(poscar, potcar, kind='max'):
    """{element: orbitals} of one pbeVaspFit2015 set (min | standard | max) for a structure."""
    t = table()
    syms = potcar_symbols(potcar)
    counts = poscar_counts(poscar)
    return dict((el, t[s][kind]) for (el, _), s in zip(counts, syms)), counts


def nbasis_for(poscar, potcar, kind='max'):
    """Number of LOBSTER basis functions of the structure for one set."""
    b, counts = basis_for(poscar, potcar, kind)
    return n_basis(b, counts)


def nbands_in(dirname):
    for f in ('OUTCAR', 'INCAR'):
        p = os.path.join(dirname, f)
        if os.path.isfile(p):
            m = re.search(r'NBANDS\s*=\s*(\d+)', open(p, errors='replace').read())
            if m:
                return int(m.group(1))
    return None


def spilling(lobsterout):
    """(abs. charge spilling, abs. total spilling) in %, max over spins, or (None, None)."""
    if not os.path.isfile(lobsterout):
        return None, None
    txt = open(lobsterout, errors='replace').read()
    c = [float(x) for x in re.findall(r'abs\. charge spilling:\s*([0-9.]+)', txt, re.I)]
    t = [float(x) for x in re.findall(r'abs\. total spilling:\s*([0-9.]+)', txt, re.I)]
    return (max(c) if c else None), (max(t) if t else None)


def basis_lines(basis):
    return ['basisfunctions %s %s' % (el, orb) for el, orb in basis.items()]


def lobsterin_with(dirname, basis):
    base = [l for l in open(os.path.join(dirname, 'lobsterin')).read().split('\n')
            if l.strip() and not re.match(r'\s*(basisfunctions|userecommendedbasisfunctions)', l, re.I)]
    return '\n'.join(base + basis_lines(basis)) + '\n'


def scan(dirname, lobster, max_n=12, clean=False):
    dirname = os.path.abspath(dirname)
    combos, counts, syms = candidates(dirname, max_n)
    nb = nbands_in(dirname)
    ref_c, ref_t = spilling(os.path.join(dirname, 'lobsterout'))
    rows = [('current', ref_c, ref_t, '', 'run already in the folder')]
    root = os.path.join(dirname, 'basis_scan')
    os.makedirs(root, exist_ok=True)
    skip = set(OUTPUTS_KEEP) | {'basis_scan.tsv', 'basis_scan_best.txt'}
    best = None
    for k, b in enumerate(combos, 1):
        nbas = n_basis(b, counts)
        desc = '; '.join('%s %s' % kv for kv in b.items())
        if nb and nbas > nb:
            rows.append((str(k), None, None, desc, 'skipped: %d basis functions > NBANDS %d' % (nbas, nb)))
            continue
        sub = os.path.join(root, str(k))
        shutil.rmtree(sub, ignore_errors=True)
        os.makedirs(sub)
        for f in os.listdir(dirname):
            p = os.path.join(dirname, f)
            if os.path.isfile(p) and f not in skip and not f.endswith('.lobster'):
                os.symlink(p, os.path.join(sub, f))
        with open(os.path.join(sub, 'lobsterin'), 'w') as fh:
            fh.write(lobsterin_with(dirname, b))
        with open(os.path.join(sub, 'lobster.out'), 'w') as fh:
            rc = subprocess.call([lobster], cwd=sub, stdout=fh, stderr=subprocess.STDOUT)
        c, t = spilling(os.path.join(sub, 'lobsterout'))
        rows.append((str(k), c, t, desc, 'ok (%d basis functions)' % nbas if c is not None else 'LOBSTER failed (rc %d)' % rc))
        print('  basis %d: %s -> charge spilling %s%%' % (k, desc, c))
        if c is not None and (best is None or c < best[1] - 1e-9):
            best = (k, c, sub, desc)
    with open(os.path.join(dirname, 'basis_scan.tsv'), 'w') as fh:
        fh.write('candidate\tcharge_spilling_pct\ttotal_spilling_pct\tbasis\tnote\n')
        for r in rows:
            fh.write('\t'.join('' if x is None else str(x) for x in r) + '\n')
    msg = 'kept the run already in the folder (charge spilling %s%%)' % ref_c
    if best and (ref_c is None or best[1] < ref_c - 1e-9):
        for f in os.listdir(best[2]):
            p = os.path.join(best[2], f)
            if os.path.isfile(p) and not os.path.islink(p):
                shutil.copy2(p, os.path.join(dirname, f))
        msg = 'basis %d (%s): charge spilling %s%% (was %s%%)' % (best[0], best[3], best[1], ref_c)
    with open(os.path.join(dirname, 'basis_scan_best.txt'), 'w') as fh:
        fh.write(msg + '\n')
    for k in os.listdir(root):     # keep only the small text outputs of each try
        for f in os.listdir(os.path.join(root, k)):
            p = os.path.join(root, k, f)
            if os.path.islink(p) or f not in OUTPUTS_KEEP:
                os.remove(p)
    if clean:
        w = os.path.join(dirname, 'WAVECAR')
        if os.path.isfile(w):
            os.remove(w)
    print(msg)
    return msg


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['info', 'nbands', 'lobsterin', 'scan'])
    ap.add_argument('dir')
    ap.add_argument('kind', nargs='?', default='max', choices=KINDS)
    ap.add_argument('--factor', type=float, default=1.0)
    ap.add_argument('--lobster', default=None)
    ap.add_argument('--max', type=int, default=12)
    ap.add_argument('--clean', action='store_true')
    a = ap.parse_args(argv[1:])
    d = a.dir
    if a.cmd == 'info':
        combos, counts, syms = candidates(d, 99)
        print('POTCARs: %s   atoms: %s   NBANDS now: %s' % (' '.join(syms), counts, nbands_in(d)))
        for k, b in enumerate(combos, 1):
            print('  %2d  %3d functions  %s' % (k, n_basis(b, counts), '; '.join('%s %s' % kv for kv in b.items())))
    elif a.cmd == 'nbands':
        combos, counts, _ = candidates(d, 99)
        n = max(n_basis(b, counts) for b in combos)
        print(int(2 * math.ceil(a.factor * n / 2.0)))
    elif a.cmd == 'lobsterin':
        t = table()
        syms = potcar_symbols(os.path.join(d, 'POTCAR'))
        counts = poscar_counts(os.path.join(d, 'POSCAR'))
        print('\n'.join(basis_lines(dict((el, t[s][a.kind]) for (el, _), s in zip(counts, syms)))))
    else:
        lob = a.lobster or os.environ.get('LOBSTER_BIN') or 'lobster'
        scan(d, os.path.expanduser(lob), a.max, clean=a.clean)


if __name__ == '__main__':
    main(sys.argv)
