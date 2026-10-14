"""Niggli reduction of a POSCAR (numpy only, no pymatgen).

LOBSTER projects the plane waves onto atom-centred orbitals in real space; in a long, thin,
skewed cell (e.g. the Materials Project primitive cell of 15R-SiC, mp-11713, three 12.7 A
vectors a few degrees apart) that projection breaks down: charge spilling 18-58 % whatever
the basis, 1.5 % in the Niggli-reduced cell of the same crystal.  The agents therefore reduce
the cell before any step when LOBSTER is requested (REDUCE_CELL: auto|on|off).

    python3 cell_reduce.py IN [OUT]     # OUT default: IN (original kept as IN.input)

Krivy & Gruber (1976) with the epsilon tests of Grosse-Kunstleve et al. (2004), applied to the
lattice vectors themselves (G-parameters recomputed every step), so the integer
transformation is exact.  Atom order, selective-dynamics flags and the POSCAR title are kept.
"""
import itertools
import os
import shutil
import sys

import numpy as np


def _gparams(L):
    a, b, c = L
    return a @ a, b @ b, c @ c, 2 * b @ c, 2 * a @ c, 2 * a @ b


def _sgn(x, e):
    return 1 if x > e else (-1 if x < -e else 0)


def niggli(A, tol=1e-5):
    """Niggli-reduced lattice of the rows of A (Angstrom), right-handed. Returns (B, M), B = M @ A."""
    L = np.array(A, dtype=float)
    e = tol * abs(np.linalg.det(L)) ** (2.0 / 3.0)
    for _ in range(1000):
        A_, B_, C_, xi, eta, zeta = _gparams(L)
        if A_ > B_ + e or (abs(A_ - B_) <= e and abs(xi) > abs(eta) + e):          # A1
            L = np.array([-L[1], -L[0], -L[2]])
            continue
        if B_ > C_ + e or (abs(B_ - C_) <= e and abs(eta) > abs(zeta) + e):        # A2
            L = np.array([-L[0], -L[2], -L[1]])
            continue
        s = (_sgn(xi, e), _sgn(eta, e), _sgn(zeta, e))
        want = 1 if s[0] * s[1] * s[2] == 1 else -1                               # A3 / A4
        for i, j, k in itertools.product((1, -1), repeat=3):
            v = (j * k * xi, i * k * eta, i * j * zeta)
            if (want == 1 and all(x > e for x in v)) or (want == -1 and all(x <= e for x in v)):
                L = np.array([i * L[0], j * L[1], k * L[2]])
                break
        A_, B_, C_, xi, eta, zeta = _gparams(L)
        if abs(xi) > B_ + e or (abs(xi - B_) <= e and 2 * eta < zeta - e) or \
                (abs(xi + B_) <= e and zeta < -e):                                 # A5
            L[2] = L[2] - np.sign(xi) * L[1]
            continue
        if abs(eta) > A_ + e or (abs(eta - A_) <= e and 2 * xi < zeta - e) or \
                (abs(eta + A_) <= e and zeta < -e):                                # A6
            L[2] = L[2] - np.sign(eta) * L[0]
            continue
        if abs(zeta) > A_ + e or (abs(zeta - A_) <= e and 2 * xi < eta - e) or \
                (abs(zeta + A_) <= e and eta < -e):                                # A7
            L[1] = L[1] - np.sign(zeta) * L[0]
            continue
        t = xi + eta + zeta + A_ + B_
        if t < -e or (abs(t) <= e and 2 * (A_ + eta) + zeta > e):                  # A8
            L[2] = L[2] + L[0] + L[1]
            continue
        break
    else:
        raise RuntimeError('Niggli reduction did not converge')
    if np.linalg.det(L) < 0:
        L = -L
    M = np.rint(L @ np.linalg.inv(A)).astype(int)
    if round(abs(np.linalg.det(M))) != 1:
        raise RuntimeError('Niggli reduction gave a non-unimodular transformation')
    return M @ np.array(A, dtype=float), M


def describe(L):
    """'a b c (alpha beta gamma)' of the rows of L."""
    n = np.linalg.norm(L, axis=1)
    ang = lambda i, j: np.degrees(np.arccos(np.clip(L[i] @ L[j] / n[i] / n[j], -1, 1)))
    return '%.2f %.2f %.2f A (%.1f %.1f %.1f deg)' % (n[0], n[1], n[2], ang(1, 2), ang(0, 2), ang(0, 1))


def read_poscar(path):
    L = open(path).read().splitlines()
    s = float(L[1].split()[0])
    A = np.array([[float(x) for x in L[i].split()[:3]] for i in range(2, 5)])
    if s < 0:
        s = (-s / abs(np.linalg.det(A))) ** (1.0 / 3.0)
    A = A * s
    if not L[5].split() or L[5].split()[0][0].isdigit():
        raise ValueError('%s: no element-symbol line (VASP5 POSCAR needed)' % path)
    cnt = [int(x) for x in L[6].split()]
    sel = L[7].strip()[:1] in 'Ss'
    i0 = 8 if sel else 7
    cart = L[i0].strip()[:1] in 'CcKk'
    rows = [L[i0 + 1 + k].split() for k in range(sum(cnt))]
    X = np.array([[float(x) for x in r[:3]] for r in rows])
    F = X * s @ np.linalg.inv(A) if cart else X
    flags = [' '.join(r[3:6]) for r in rows] if sel else None
    return {'title': L[0], 'A': A, 'species': L[5], 'counts': L[6], 'F': F, 'flags': flags}


def is_reduced(A, B, rel=1e-4):
    """True when B has the same vector lengths as A (A is already a reduced cell)."""
    la, lb = sorted(np.linalg.norm(A, axis=1)), sorted(np.linalg.norm(B, axis=1))
    return all(abs(x - y) <= rel * max(lb) for x, y in zip(la, lb))


def reduced_poscar(path):
    """(text of the Niggli-reduced POSCAR, 'before -> after') or (None, cell) if already reduced."""
    p = read_poscar(path)
    B, M = niggli(p['A'])
    if is_reduced(p['A'], B):
        return None, describe(p['A'])
    F = (p['F'] @ np.linalg.inv(M)) % 1.0
    F[np.isclose(F, 1.0, atol=1e-10)] = 0.0
    out = [p['title'].rstrip() + ' (Niggli-reduced)', '1.0']
    out += ['  %.10f %.10f %.10f' % tuple(v) for v in B]
    out += [p['species'], p['counts']]
    if p['flags']:
        out.append('Selective dynamics')
    out.append('Direct')
    for k, f in enumerate(F):
        out.append('  %.10f %.10f %.10f' % tuple(f) + ('  ' + p['flags'][k] if p['flags'] else ''))
    return '\n'.join(out) + '\n', '%s -> %s' % (describe(p['A']), describe(B))


def reduce_mode(inst):
    """'on' | 'off' | 'auto' from the parsed instructions (key REDUCE_CELL / NIGGLI)."""
    return (inst.get('reduce_cell') or 'auto').lower()


def should_reduce(inst):
    """(reduce?, reason) for the parsed instructions; the cell itself is tested later."""
    mode = reduce_mode(inst)
    if mode == 'off':
        return False, 'REDUCE_CELL: off'
    if inst.get('is_2d'):
        return False, '2-D slab (the vacuum axis must stay c)'
    if mode == 'on':
        return True, 'REDUCE_CELL: on'
    if 'lobster' not in (inst.get('tasks') or []):
        return False, 'no LOBSTER step'
    raw_k = inst.get('kpoints_raw') or {}
    if inst.get('kmesh') or raw_k:
        return False, 'KMESH/KPOINTS given for this cell'
    return True, 'LOBSTER step'


def prepare_poscar(src, dst, inst, log=print):
    """Copy src to dst, Niggli-reduced when the instructions call for it and the cell is
    skewed (original kept as dst + '.input').  Returns True if the cell was reduced."""
    want, why = should_reduce(inst)
    text, cell = reduced_poscar(src)
    if (text is None or not want) and os.path.abspath(src) != os.path.abspath(dst) \
            and os.path.abspath(src) != os.path.abspath(dst + '.input') and os.path.isfile(dst + '.input'):
        os.remove(dst + '.input')          # left from an earlier, reduced generation
    if text is None:                       # already a reduced cell: nothing to decide
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)
        return False
    if not want:
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy(src, dst)
        if 'lobster' in (inst.get('tasks') or []):
            log('  WARNING: the cell is not reduced (%s) and is kept (%s); LOBSTER charge\n'
                '           spilling can be large in a skewed cell. REDUCE_CELL: on reduces it.'
                % (cell.split(' -> ')[0], why))
        return False
    if os.path.abspath(src) != os.path.abspath(dst + '.input'):
        shutil.copy(src, dst + '.input')
    with open(dst, 'w') as fh:
        fh.write(text)
    log('  Cell      : Niggli-reduced (%s): %s\n'
        '              same crystal; original kept as POSCAR.input (REDUCE_CELL: off keeps it)'
        % (why, cell))
    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    text, cell = reduced_poscar(src)
    if text is None:
        print('already reduced: %s' % cell)
        if dst != src:
            shutil.copy(src, dst)
    else:
        if dst == src:
            shutil.copy(src, src + '.input')
        open(dst, 'w').write(text)
        print('reduced: %s' % cell)
