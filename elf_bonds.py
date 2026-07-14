#!/usr/bin/env python3
"""Plot the Electron Localization Function (ELF) along nearest-neighbour bonds.

Reads a VASP ELFCAR (CHGCAR-format grid, values in [0,1]) and, for each
symmetry-distinct nearest-neighbour bond type, samples ELF on a straight line
from one atom centre to the other and plots ELF vs. distance along the bond.

"Non-equivalent" bonds are grouped by (element pair, bond length rounded to
0.01 A) — a dependency-light proxy for symmetry-inequivalent bonds that needs
no spglib/pymatgen.  Only numpy + matplotlib are required, so it runs in the
same post-processing environment as the rest of analyze.sh.

Usage:
    ./elf_bonds.py [ELFCAR] [--out PREFIX] [--nn-scale 1.2] [--npoints 200]

If ELFCAR is omitted it looks for ./ELFCAR, then ./02_scf/ELFCAR.
Output: <PREFIX>_elf_bonds.png and .pdf
(PREFIX defaults to the reduced chemical formula, e.g. MoSe2_elf_bonds.pdf)
"""
import os
import sys
import argparse
from math import gcd
from functools import reduce
import numpy as np


def reduced_formula(elements):
    """Reduced chemical formula from a per-atom element list, e.g.
    ['Mo','Se','Se'] -> 'MoSe2'.  Order follows first appearance (POSCAR)."""
    order = list(dict.fromkeys(elements))
    counts = [elements.count(e) for e in order]
    g = reduce(gcd, counts) if counts else 1
    g = g or 1
    return ''.join(e + ('' if c // g == 1 else str(c // g))
                   for e, c in zip(order, counts))


# ── ELFCAR parsing ────────────────────────────────────────────────────────────
def read_elfcar(path):
    """Parse a CHGCAR-format ELFCAR.

    Returns (lattice 3x3 in A [rows = a1,a2,a3], elements list per atom,
    frac_coords Nx3, elf grid ndarray[nx,ny,nz]).  Reads the first volumetric
    block (spin-up / total ELF); warns if a second (spin-down) block exists.
    """
    with open(path, errors='replace') as fh:
        lines = fh.readlines()

    scale = float(lines[1].split()[0])
    raw = np.array([[float(x) for x in lines[i].split()[:3]] for i in (2, 3, 4)])
    if scale < 0:                       # negative = target volume
        vol = abs(np.dot(raw[0], np.cross(raw[1], raw[2])))
        factor = (abs(scale) / vol) ** (1.0 / 3.0)
    else:
        factor = scale
    lattice = raw * factor

    # Line 5 is either element symbols (VASP5) or the counts (VASP4)
    tok5 = lines[5].split()
    if tok5 and all(t.lstrip('-').isdigit() for t in tok5):
        symbols = [f'A{i+1}' for i in range(len(tok5))]   # VASP4: no names
        counts = [int(t) for t in tok5]
        idx = 6
    else:
        symbols = tok5
        counts = [int(t) for t in lines[6].split()]
        idx = 7

    elements = []
    for sym, n in zip(symbols, counts):
        elements += [sym] * n
    natoms = sum(counts)

    mode = lines[idx].strip()[:1].lower()   # 'd' direct, 'c'/'k' cartesian
    idx += 1
    coords = np.array([[float(x) for x in lines[idx + i].split()[:3]]
                       for i in range(natoms)])
    idx += natoms

    if mode in ('c', 'k'):
        frac = (coords * factor) @ np.linalg.inv(lattice)
    else:
        frac = coords
    frac = frac % 1.0

    # Find the grid-dimension line: first subsequent line with exactly 3 ints
    grid_line = None
    while idx < len(lines):
        t = lines[idx].split()
        idx += 1
        if len(t) == 3 and all(x.lstrip('-').isdigit() for x in t):
            grid_line = t
            break
    if grid_line is None:
        raise ValueError(f'{path}: could not locate grid dimensions')
    nx, ny, nz = (int(v) for v in grid_line)
    ntot = nx * ny * nz

    vals = []
    while idx < len(lines) and len(vals) < ntot:
        for tok in lines[idx].split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass                     # skip augmentation headers, etc.
        idx += 1
    if len(vals) < ntot:
        raise ValueError(f'{path}: expected {ntot} grid values, got {len(vals)}')

    # Extra volumetric blocks appear for spin-polarised (2 blocks) or
    # non-collinear (4: total + mx,my,mz) ELFCARs. The first block is the
    # total ELF, which is what we use.
    extra = sum(len(line.split()) for line in lines[idx:])
    if extra >= ntot:
        sys.stderr.write('  note: multi-component ELFCAR (spin-polarised or '
                         'non-collinear) — using the total-ELF (first) block\n')

    # VASP writes x fastest, then y, then z  → Fortran order
    grid = np.array(vals[:ntot], dtype=float).reshape((nx, ny, nz), order='F')
    return lattice, elements, frac, grid


# ── PAW augmentation radii ────────────────────────────────────────────────────
AU_TO_ANG = 0.529177

def read_rcore(potcar_path):
    """{element: augmentation radius (Å)} from the POTCAR RCORE lines.

    Inside these spheres VASP's ELFCAR holds the PSEUDO-ELF (no PAW
    reconstruction) — the values there are not physical. Returns {} if the
    POTCAR is missing/unreadable.
    """
    radii = {}
    try:
        el = None
        with open(potcar_path, errors='replace') as fh:
            for ln in fh:
                if 'TITEL' in ln:
                    # e.g. "TITEL  = PAW_PBE Ga_d 06Sep2000" -> Ga
                    parts = ln.split()
                    for tok in parts:
                        if tok[:1].isupper() and tok not in ('TITEL', 'PAW',
                                                             'PAW_PBE', 'PAW_GGA'):
                            el = tok.split('_')[0]
                            break
                elif 'RCORE' in ln and el:
                    try:
                        radii[el] = float(ln.split('=')[1].split()[0]) * AU_TO_ANG
                    except (IndexError, ValueError):
                        pass
                    el = None
    except OSError:
        pass
    return radii


DEFAULT_RAUG = 1.1        # Å — conservative fallback when no POTCAR is found


# ── nearest-neighbour bonds ───────────────────────────────────────────────────
def nn_bonds(lattice, elements, frac, nn_scale=1.2, shells=2):
    """Return non-equivalent 1st- (and optionally 2nd-) shell bonds.

    For every atom, the 1st shell is everything within nn_scale x (its own
    nearest distance); the 2nd shell is the next distinct distance group
    (within 8% of the shortest post-1st-shell distance). One representative
    is kept per (element pair, rounded length). Each bond is a dict:
    iA, elemA, rA_cart, elemB, rB_cart, dist, shell (1 or 2).
    """
    cart = frac @ lattice
    natoms = len(frac)
    shifts = np.array([[i, j, k]
                       for i in (-1, 0, 1)
                       for j in (-1, 0, 1)
                       for k in (-1, 0, 1)], dtype=float)
    shift_cart = shifts @ lattice

    seen = {}
    bonds = []
    for i in range(natoms):
        # all periodic images of every atom, relative to atom i
        images = cart[:, None, :] + shift_cart[None, :, :]   # (natoms, 27, 3)
        diff = images - cart[i]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist < 1e-3] = np.inf                           # drop self
        dmin = dist.min()
        cutoff1 = dmin * nn_scale
        # 2nd shell: next distinct distance beyond the 1st-shell cutoff
        beyond = dist[dist > cutoff1]
        cutoff2 = beyond.min() * 1.08 if (shells >= 2 and beyond.size
                                          and np.isfinite(beyond.min())) else -1.0
        for j in range(natoms):
            for s in range(len(shifts)):
                d = dist[j, s]
                if d <= cutoff1:
                    shell = 1
                elif 0 < d <= cutoff2:
                    shell = 2
                else:
                    continue
                key = (frozenset((elements[i], elements[j])), round(d, 2))
                if key in seen:
                    continue
                seen[key] = True
                bonds.append(dict(
                    iA=i, elemA=elements[i], rA_cart=cart[i].copy(),
                    elemB=elements[j], rB_cart=images[j, s].copy(),
                    dist=float(d), shell=shell))
    bonds.sort(key=lambda b: (b['shell'], b['elemA'], b['elemB'], b['dist']))
    return bonds


# ── ELF sampling along a line ─────────────────────────────────────────────────
def _trilerp(grid, frac_pts):
    """Periodic trilinear interpolation of `grid` at fractional points (M,3)."""
    nx, ny, nz = grid.shape
    n = np.array([nx, ny, nz])
    g = (frac_pts % 1.0) * n
    i0 = np.floor(g).astype(int)
    d = g - i0
    i0 %= n
    i1 = (i0 + 1) % n
    x0, y0, z0 = i0.T
    x1, y1, z1 = i1.T
    dx, dy, dz = d.T
    c = (grid[x0, y0, z0] * (1 - dx) * (1 - dy) * (1 - dz) +
         grid[x1, y0, z0] * dx * (1 - dy) * (1 - dz) +
         grid[x0, y1, z0] * (1 - dx) * dy * (1 - dz) +
         grid[x0, y0, z1] * (1 - dx) * (1 - dy) * dz +
         grid[x1, y1, z0] * dx * dy * (1 - dz) +
         grid[x1, y0, z1] * dx * (1 - dy) * dz +
         grid[x0, y1, z1] * (1 - dx) * dy * dz +
         grid[x1, y1, z1] * dx * dy * dz)
    return c


def sample_bond(grid, lattice, rA_cart, rB_cart, npoints=200):
    """Sample ELF from atom A (t=0) to atom B (t=1). Returns (dist_A, elf)."""
    t = np.linspace(0.0, 1.0, npoints)
    pts_cart = rA_cart[None, :] + t[:, None] * (rB_cart - rA_cart)[None, :]
    frac_pts = pts_cart @ np.linalg.inv(lattice)
    elf = _trilerp(grid, frac_pts)
    dist = t * np.linalg.norm(rB_cart - rA_cart)
    return dist, elf


# ── ELF on the plane through an atom + its 1st and 2nd neighbours ──────────────
def _neighbour_images(lattice, elements, frac, iA, nmax=3.5):
    """All periodic images of every atom within nmax*(nearest dist) of atom iA.
    Returns a list of (dist, elem, cart) sorted by distance (self excluded)."""
    cart = frac @ lattice
    shifts = np.array([[i, j, k] for i in range(-2, 3) for j in range(-2, 3)
                       for k in range(-2, 3)], dtype=float) @ lattice
    out = []
    for j in range(len(frac)):
        for sc in shifts:
            r = cart[j] + sc
            d = float(np.linalg.norm(r - cart[iA]))
            if d > 1e-3:
                out.append((d, elements[j], r))
    out.sort(key=lambda t: t[0])
    dmin = out[0][0]
    return [o for o in out if o[0] <= dmin * nmax], cart[iA]


def plane_basis(A, B, C):
    """Orthonormal in-plane basis (u,v) and normal n for the plane through A,B,C."""
    u = B - A
    u /= np.linalg.norm(u)
    w = C - A
    v = w - np.dot(w, u) * u
    v /= np.linalg.norm(v)
    n = np.cross(u, v)
    return u, v, n


def sample_plane(grid, lattice, A, u, v, umin, umax, vmin, vmax, ngrid=220):
    """Sample ELF on a rectangular patch of the (u,v) plane centred on A.
    Returns (U, V, ELF) meshgrids for contourf."""
    us = np.linspace(umin, umax, ngrid)
    vs = np.linspace(vmin, vmax, ngrid)
    U, V = np.meshgrid(us, vs)
    pts = A[None, :] + U.ravel()[:, None] * u[None, :] + V.ravel()[:, None] * v[None, :]
    frac = pts @ np.linalg.inv(lattice)
    elf = _trilerp(grid, frac).reshape(U.shape)
    return U, V, elf


def plot_elf_plane(grid, lattice, elements, frac, prefix, iA=0, ngrid=220,
                   radii=None):
    """Filled ELF contour on the plane through atom iA and its 1st+2nd neighbours.

    Atoms whose (imaged) position lies in that plane and inside the patch are
    marked. Saves <prefix>_elf_plane.{png,pdf}. Returns the plane description.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    neigh, A = _neighbour_images(lattice, elements, frac, iA)
    if not neigh:
        return None
    elemA = elements[iA]
    d1 = neigh[0][0]                                  # 1st-shell distance
    B = neigh[0][2]                                   # 1st neighbour
    uAB = (B - A) / np.linalg.norm(B - A)
    # 2nd point: a genuine 2nd-SHELL neighbour (distance > 1st shell) that is
    # non-collinear with A->B, so the plane spans the 1st and 2nd neighbours.
    C = None
    for d, el, r in neigh[1:]:
        perp = (r - A) - np.dot(r - A, uAB) * uAB
        if d > d1 * 1.05 and np.linalg.norm(perp) > 0.3:
            C = r; elemC = el; distC = d
            break
    if C is None:                                     # fall back to any non-collinear
        for d, el, r in neigh[1:]:
            perp = (r - A) - np.dot(r - A, uAB) * uAB
            if np.linalg.norm(perp) > 0.3:
                C = r; elemC = el; distC = d
                break
    if C is None:
        return None
    u, v, n = plane_basis(A, B, C)

    def uv(r):
        wv = r - A
        return np.dot(wv, u), np.dot(wv, v)

    # Box sized from just the 3 defining atoms (central + 1st + 2nd NN) + pad,
    # so the plot zooms on the local bonding region rather than the whole cell.
    key_uv = [uv(A), uv(B), uv(C)]
    pad = 1.8
    umin = min(p[0] for p in key_uv) - pad; umax = max(p[0] for p in key_uv) + pad
    vmin = min(p[1] for p in key_uv) - pad; vmax = max(p[1] for p in key_uv) + pad

    # Mark every atom image that lies in the plane AND inside the box.
    marks = []
    for d, el, r in [(0.0, elemA, A)] + neigh:
        if abs(np.dot(r - A, n)) < 0.35:             # lies (nearly) in the plane
            x, y = uv(r)
            if umin <= x <= umax and vmin <= y <= vmax:
                marks.append((x, y, el))

    U, V, ELF = sample_plane(grid, lattice, A, u, v, umin, umax, vmin, vmax, ngrid)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cf = ax.contourf(U, V, ELF, levels=np.linspace(0, 1, 21), cmap='jet', vmin=0, vmax=1)
    ax.contour(U, V, ELF, levels=[0.5], colors='k', linewidths=0.6, alpha=0.5)
    cbar = fig.colorbar(cf, ax=ax, ticks=np.linspace(0, 1, 11))
    cbar.set_label('ELF')
    # element -> marker colour
    els = list(dict.fromkeys([m[2] for m in marks]))
    cmap = {e: c for e, c in zip(els, plt.rcParams['axes.prop_cycle'].by_key()['color'])}
    # PAW augmentation spheres: semi-transparent gray discs — the ELF shown
    # underneath them is the pseudo-ELF and not physical.
    radii = radii or {}
    for x, y, el in marks:
        r = radii.get(el, DEFAULT_RAUG)
        ax.add_patch(plt.Circle((x, y), r, facecolor='0.55', edgecolor='0.25',
                                linewidth=0.6, linestyle='--', alpha=0.45,
                                zorder=4))
    for x, y, el in marks:
        ax.scatter([x], [y], s=90, facecolor=cmap[el], edgecolor='white',
                   linewidth=1.2, zorder=5)
        ax.annotate(el, (x, y), color='white', fontsize=8, ha='center', va='center',
                    zorder=6, fontweight='bold')
    ax.text(0.02, 0.02, 'gray discs = PAW augmentation spheres (pseudo-ELF)',
            transform=ax.transAxes, fontsize=7, color='0.15',
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.8),
            zorder=7)
    ax.set_aspect('equal')
    ax.set_xlabel('in-plane x (Å)', fontsize=11)
    ax.set_ylabel('in-plane y (Å)', fontsize=11)
    ax.set_title(f'ELF plane through {elemA} + 1st ({elemA}–{neigh[0][1]}) '
                 f'& 2nd ({elemA}–{elemC}) neighbours', fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        out = f'{prefix}_elf_plane.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None)
        print(f'  Saved: {out}')
    plt.close(fig)
    return dict(central=elemA, first=neigh[0][1], second=elemC)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('elfcar', nargs='?', default=None,
                    help='ELFCAR path (default: ./ELFCAR or ./02_scf/ELFCAR)')
    ap.add_argument('--out', default=None,
                    help='output prefix (default: reduced formula, e.g. MoSe2)')
    ap.add_argument('--nn-scale', type=float, default=1.2,
                    help='neighbour cutoff = nn_scale x nearest distance (1.2)')
    ap.add_argument('--shells', type=int, default=2, choices=(1, 2),
                    help='neighbour shells to profile (default 2: 1st + 2nd)')
    ap.add_argument('--npoints', type=int, default=200,
                    help='samples per bond (default: 200)')
    ap.add_argument('--plane-atom', type=int, default=0,
                    help='central atom index for the 2D ELF plane plot (default 0)')
    ap.add_argument('--no-plane', action='store_true',
                    help='skip the 2D ELF plane plot')
    args = ap.parse_args()

    path = args.elfcar
    if path is None:
        for cand in ('ELFCAR', os.path.join('02_scf', 'ELFCAR')):
            if os.path.exists(cand):
                path = cand
                break
    if not path or not os.path.exists(path):
        sys.stderr.write('ERROR: ELFCAR not found (pass its path explicitly)\n')
        sys.exit(1)

    lattice, elements, frac, grid = read_elfcar(path)

    # An all-zero grid means VASP wrote no ELF. The usual cause is a
    # non-collinear / spin-orbit SCF: VASP prints "ELF not implemented for
    # non collinear case" and leaves the ELFCAR filled with zeros.
    if float(np.abs(grid).max()) < 1e-6:
        sys.stderr.write(
            f'ERROR: {path} contains no ELF data (all zeros) — nothing to plot.\n'
            '       VASP does not compute ELF for non-collinear / spin-orbit\n'
            '       runs ("WARNING: ELF not implemented for non collinear case").\n'
            '       Re-run the SCF without LSORBIT / LNONCOLLINEAR to obtain ELF.\n')
        sys.exit(2)

    prefix = args.out or reduced_formula(elements)

    bonds = nn_bonds(lattice, elements, frac, args.nn_scale, shells=args.shells)
    if not bonds:
        sys.stderr.write('ERROR: no nearest-neighbour bonds found\n')
        sys.exit(1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # PAW augmentation radii: ELF inside these spheres is the pseudo-ELF and
    # not physical — drawn DOTTED there; stats use only the region outside.
    radii = read_rcore(os.path.join(os.path.dirname(os.path.abspath(path)),
                                    'POTCAR'))
    if radii:
        print('  PAW augmentation radii (Å): '
              + '  '.join(f'{e}={r:.2f}' for e, r in radii.items()))
    else:
        print(f'  note: no POTCAR next to ELFCAR — using {DEFAULT_RAUG} Å '
              'augmentation radius for all elements')

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    stats = []
    print(f'{"bond":<12}{"shell":>6}{"length (A)":>12}{"ELF max":>10}{"ELF min":>10}')
    for k, b in enumerate(bonds):
        dist, elf = sample_bond(grid, lattice, b['rA_cart'], b['rB_cart'],
                                args.npoints)
        shell = b.get('shell', 1)
        c     = colors[k % len(colors)]
        L     = b['dist']
        rA = radii.get(b['elemA'], DEFAULT_RAUG)
        rB = radii.get(b['elemB'], DEFAULT_RAUG)
        inside = (dist < rA) | (L - dist < rB)
        label = (f"{b['elemA']}–{b['elemB']}  {L:.2f} Å"
                 + ('  (2nd)' if shell == 2 else ''))
        # dotted inside the augmentation spheres (pseudo-ELF), solid outside
        ax.plot(dist, np.where(inside, elf, np.nan), color=c, lw=1.1, ls=':',
                alpha=0.65)
        ax.plot(dist, np.where(inside, np.nan, elf), color=c, lw=1.8,
                ls='--' if shell == 2 else '-', label=label)
        # outside-only stats: <ELF>, <|dELF/dx|>, <d2ELF/dx2>
        if (~inside).sum() >= 5:
            d1 = np.gradient(elf, dist)
            d2 = np.gradient(d1, dist)
            o  = ~inside
            stats.append((label,
                          float(elf[o].mean()),
                          float(np.abs(d1[o]).mean()),
                          float(d2[o].mean())))
        print(f"{b['elemA']+'-'+b['elemB']:<12}{shell:>6d}{L:>12.3f}"
              f"{elf.max():>10.3f}{elf.min():>10.3f}")

    ax.set_xlabel('Distance along bond (Å)', fontsize=12)
    ax.set_ylabel('ELF', fontsize=12)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', lw=0.7, ls=':')   # uniform-electron-gas ref
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, title='atom → atom  (dotted = inside PAW sphere)',
              title_fontsize=8, loc='upper right')
    if stats:
        box = 'outside augmentation spheres:\n' + '\n'.join(
            f'{lbl}:  ⟨ELF⟩={m:.2f}  ⟨|ELF′|⟩={a:.2f} Å$^{{-1}}$  '
            f'⟨ELF″⟩={s:+.1f} Å$^{{-2}}$' for lbl, m, a, s in stats)
        ax.text(0.02, 0.02, box, transform=ax.transAxes, fontsize=6.8,
                va='bottom', ha='left',
                bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.85))
    fig.tight_layout()

    for ext in ('png', 'pdf'):
        out = f'{prefix}_elf_bonds.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None)
        print(f'  Saved: {out}')
    plt.close(fig)

    # 2D ELF density on the plane through an atom + its 1st & 2nd neighbours
    if not args.no_plane:
        try:
            plot_elf_plane(grid, lattice, elements, frac, prefix,
                           iA=args.plane_atom, radii=radii)
        except Exception as e:
            sys.stderr.write(f'  note: ELF plane plot skipped ({e})\n')


if __name__ == '__main__':
    main()
