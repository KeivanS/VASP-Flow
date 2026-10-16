"""Niggli reduction of a POSCAR (numpy only, no pymatgen).

LOBSTER projects the plane waves onto atom-centred orbitals in real space; in a long, thin,
skewed cell (e.g. the Materials Project primitive cell of 15R-SiC, mp-11713, three 12.7 A
vectors a few degrees apart) that projection breaks down: charge spilling 18-58 % whatever
the basis, 1.5 % in the Niggli-reduced cell of the same crystal.  The agents therefore reduce
the cell before any step when LOBSTER is requested (REDUCE_CELL: auto|on|off).

    python3 cell_reduce.py IN [OUT]     # OUT default: IN (original kept as IN.input)
    python3 cell_reduce.py --after-relax SCF_DIR [--no-reduce]
        run by 02_scf/copy_from_relax.sh right after CONTCAR was copied to SCF_DIR/POSCAR:
        drops the relaxation's selective-dynamics flags and Niggli-reduces the relaxed cell;
        if that changes the lattice basis, the KPOINTS of 02_scf and the later steps are
        rewritten for the new basis (original kept as KPOINTS.prereduce) and SCF_DIR gets a
        .cell_reduced marker (the relaxation's CHGCAR/WAVECAR do not fit the new cell)

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


def orient(B, F, types):
    """Rotate the lattice rows B (Cartesian frame only; fractional coordinates unchanged) so
    that the axes of the conventional cell lie along x, y, z: spglib's standardized
    orientation, e.g. a body-centred tetragonal cell becomes (a/2, a/2, c/2)-type vectors and a
    hexagonal one has a along x and c along z.  Without spglib (or no symmetry found):
    a along x, b in the x-y plane.  Returns (rotated B, how)."""
    B = np.array(B, dtype=float)
    try:
        import spglib
        ds = None
        for prec in (1e-3, 1e-2, 1e-1):
            ds = spglib.get_symmetry_dataset((B, np.asarray(F, dtype=float), list(types)), symprec=prec)
            if ds is not None:
                break
        if ds is not None:
            R = np.array(ds['std_rotation_matrix'] if isinstance(ds, dict) else ds.std_rotation_matrix)
            sg = ds['international'] if isinstance(ds, dict) else ds.international
            if abs(np.linalg.det(R) - 1) < 1e-6:
                Bs = B @ R.T
                # of the 180-degree turns about x, y, z (axes stay aligned), the one with the
                # most positive components; -0.0 printed as 0.0
                Bs = max((Bs * d for d in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))),
                         key=lambda X: (np.sum(X > 1e-8), X.sum()))
                Bs[np.abs(Bs) < 1e-10] = 0.0
                return Bs, 'axes along the conventional cell of %s (spglib)' % sg
    except Exception:
        pass
    Q, Rt = np.linalg.qr(B.T)                     # B.T = Q Rt -> rows of Rt.T: a along x, b in xy
    D = np.diag(np.sign(np.diag(Rt)) + (np.diag(Rt) == 0))
    Bt = (D @ Rt).T
    Bt[np.abs(Bt) < 1e-10] = 0.0
    return Bt, 'a along x, b in the x-y plane (no spglib)'


def _types(p):
    return [k for k, n in enumerate(int(x) for x in p['counts'].split()) for _ in range(n)]


def describe(L):
    """'a b c (alpha beta gamma)' of the rows of L."""
    n = np.linalg.norm(L, axis=1)
    ang = lambda i, j: np.degrees(np.arccos(np.clip(L[i] @ L[j] / n[i] / n[j], -1, 1)))
    return '%.2f %.2f %.2f A (%.1f %.1f %.1f deg)' % (n[0], n[1], n[2], ang(1, 2), ang(0, 2), ang(0, 1))


def read_poscar(path):
    return parse_poscar(open(path).read())


def parse_poscar(text):
    L = text.splitlines()
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
    B, how = orient(B, F, _types(p))
    out = [p['title'].rstrip() + ' (Niggli-reduced)', '1.0']
    out += ['  %.10f %.10f %.10f' % tuple(v) for v in B]
    out += [p['species'], p['counts']]
    if p['flags']:
        out.append('Selective dynamics')
    out.append('Direct')
    for k, f in enumerate(F):
        out.append('  %.10f %.10f %.10f' % tuple(f) + ('  ' + p['flags'][k] if p['flags'] else ''))
    return '\n'.join(out) + '\n', '%s -> %s; %s' % (describe(p['A']), describe(B), how)


def selective_first_fixed(text):
    """POSCAR text for a relaxation: 'Selective dynamics', atom 1 fixed (F F F), every other
    atom free (T T T).  Fixing one atom removes the rigid drift of the whole crystal, which
    carries no information, without constraining the cell (ISIF=3 still relaxes it).
    A POSCAR that already has selective dynamics is returned unchanged (your flags win)."""
    L = text.splitlines()
    sp = L[5].split()
    i_cnt = 5 if sp and sp[0].isdigit() else 6          # VASP4 POSCAR: no element line
    n = sum(int(x) for x in L[i_cnt].split())
    i_mode = i_cnt + 1
    if L[i_mode].strip()[:1] in 'Ss':
        return text if text.endswith('\n') else text + '\n'
    out = L[:i_mode] + ['Selective dynamics', L[i_mode]]
    for k in range(n):
        t = L[i_mode + 1 + k].split()
        out.append('  %s %s %s  %s' % (t[0], t[1], t[2], 'F F F' if k == 0 else 'T T T')
                   + ('  ' + ' '.join(t[3:]) if len(t) > 3 else ''))
    out += L[i_mode + 1 + n:]
    return '\n'.join(out).rstrip('\n') + '\n'


def _recip(A):
    return 2 * np.pi * np.linalg.inv(A).T


def transform_kpoints(text, A, B, M):
    """KPOINTS text written for lattice A, rewritten for B = M @ A (same crystal).
    Reciprocal coordinates (line mode, explicit lists) are transformed exactly
    (k' = k M^T); an automatic mesh is rescaled to the same density along the new
    reciprocal vectors (even divisions stay even); Cartesian and fully automatic
    KPOINTS are left as they are.  Returns (text, note)."""
    L = text.splitlines()
    if len(L) < 3:
        return text, 'unchanged (too short)'
    try:
        nk = int(L[1].split()[0])
    except (ValueError, IndexError):
        return text, 'unchanged (unreadable)'
    mode = L[2].strip()[:1].lower()
    if nk == 0 and mode in 'gm' and len(L) > 3:
        old = [int(x) for x in L[3].split()[:3]]
        lo, ln = np.linalg.norm(_recip(A), axis=1), np.linalg.norm(_recip(B), axis=1)
        dens = np.mean([n / l for n, l in zip(old, lo)])
        even = all(n % 2 == 0 for n in old)
        new = []
        for l in ln:
            x = dens * l
            new.append(max(2, 2 * int(round(x / 2))) if even else max(1, int(round(x))))
        L[3] = '  %d  %d  %d' % tuple(new)
        L[0] = L[0] + ' [mesh rescaled for the Niggli-reduced cell]'
        return '\n'.join(L) + '\n', 'mesh %dx%dx%d -> %dx%dx%d' % tuple(old + new)
    if nk == 0:
        return text, 'unchanged (automatic length mode)'
    i_coord = 3 if mode == 'l' else 2
    if L[i_coord].strip()[:1].lower() in 'ck':
        return text, 'unchanged (Cartesian)'
    MT = np.array(M, dtype=float).T
    out = L[:i_coord + 1]
    for line in L[i_coord + 1:]:
        t = line.split()
        try:
            k = np.array([float(x) for x in t[:3]])
        except ValueError:
            k = None
        if k is None or len(k) < 3:              # blank line between path segments
            out.append(line)
            continue
        kn = k @ MT
        rest = line.split(None, 3)[3] if len(t) > 3 else ''
        out.append('  %.10f %.10f %.10f   %s' % (kn[0], kn[1], kn[2], rest))
    out[0] = out[0] + ' [coordinates for the Niggli-reduced cell]'
    return '\n'.join(out).rstrip() + '\n', 'k-point coordinates transformed'


# steps after the SCF whose KPOINTS follow the cell; wannier90.win and the phonon
# supercells were written for the cell at generation time and are only reported
AFTER_SCF = ('02_scf', '03_bands', '04_dos', '06_dfpt', '08_lobster')
NOT_ADAPTED = ('05_wannier', '07_phonons')


def after_relax(scf_dir, reduce=True, log=print):
    """See the module doc (--after-relax).  reduce=False (--no-reduce: REDUCE_CELL off, 2-D
    slab, your own KMESH/KPOINTS) only drops the flags.  Returns True when the basis changed."""
    scf_dir = os.path.abspath(scf_dir)
    proj = os.path.dirname(scf_dir)
    pos = os.path.join(scf_dir, 'POSCAR')
    p = parse_poscar(open(pos).read())
    B, M = niggli(p['A'])
    changed = reduce and not is_reduced(p['A'], B)
    if not changed:
        B, M = p['A'], np.eye(3, dtype=int)
    F = (p['F'] @ np.linalg.inv(M)) % 1.0
    F[np.isclose(F, 1.0, atol=1e-10)] = 0.0
    how = ''
    if changed:
        B, how = orient(B, F, _types(p))
    title = p['title'].rstrip() + (' (Niggli-reduced)' if changed and 'Niggli' not in p['title'] else '')
    out = [title, '1.0'] + ['  %.10f %.10f %.10f' % tuple(v) for v in B]
    out += [p['species'], p['counts'], 'Direct'] + ['  %.10f %.10f %.10f' % tuple(f) for f in F]
    with open(pos, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    mark = os.path.join(scf_dir, '.cell_reduced')
    if p['flags']:
        log('  relaxed POSCAR: selective-dynamics flags removed')
    if not reduce:
        return False
    for d in AFTER_SCF:
        kp = os.path.join(proj, d, 'KPOINTS')
        keep = kp + '.prereduce'
        if not os.path.isfile(kp):
            continue
        if os.path.isfile(os.path.join(proj, d, '.explicit_kpoints')):
            if changed:
                log('  WARNING: %s/KPOINTS is your own (KPOINTS block): not adapted to the new cell basis' % d)
            continue
        if not changed:
            if os.path.isfile(keep):                 # an earlier relaxation needed the new basis
                shutil.move(keep, kp)
                log('  %s/KPOINTS restored (the relaxed cell keeps the original basis)' % d)
            continue
        if not os.path.isfile(keep):
            shutil.copy(kp, keep)
        text, note = transform_kpoints(open(keep).read(), p['A'], B, M)
        with open(kp, 'w') as fh:
            fh.write(text)
        log('  %s/KPOINTS: %s' % (d, note))
    if changed:
        with open(mark, 'w') as fh:
            fh.write('%s\nM = %s\n' % ('%s -> %s' % (describe(p['A']), describe(B)), M.tolist()))
        log('  Cell      : relaxed cell Niggli-reduced: %s -> %s; %s' % (describe(p['A']), describe(B), how))
        for d in NOT_ADAPTED:
            if os.path.isdir(os.path.join(proj, d)):
                log('  WARNING: %s was generated for the old cell basis; regenerate the project '
                    '(or set REDUCE_CELL: on) before running it' % d)
    else:
        if os.path.isfile(mark):
            os.remove(mark)
        log('  Cell      : relaxed cell is Niggli-reduced (%s)' % describe(B))
    return changed


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
    tasks = inst.get('tasks') or []
    if 'lobster' not in tasks and 'relax' not in tasks:
        return False, 'no relaxation or LOBSTER step'
    raw_k = inst.get('kpoints_raw') or {}
    if inst.get('kmesh') or raw_k:
        return False, 'KMESH/KPOINTS given for this cell'
    return True, 'LOBSTER step' if 'lobster' in tasks else 'relaxation'


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
        elif 'relax' in (inst.get('tasks') or []):
            log('  NOTE: the cell is not reduced (%s) and is kept as given (%s).'
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
    if sys.argv[1] == '--after-relax':
        args = [a for a in sys.argv[2:] if a != '--no-reduce']
        after_relax(args[0] if args else '.', reduce='--no-reduce' not in sys.argv)
        sys.exit(0)
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
