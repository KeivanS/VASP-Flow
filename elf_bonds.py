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
Output: <PREFIX>_elf_bonds.png and .pdf   (default PREFIX = ELF)
"""
import os
import sys
import argparse
import numpy as np


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


# ── nearest-neighbour bonds ───────────────────────────────────────────────────
def nn_bonds(lattice, elements, frac, nn_scale=1.2):
    """Return non-equivalent nearest-neighbour bonds.

    For every atom, gather neighbours within nn_scale x (its own nearest
    distance), then keep one representative per (element pair, rounded length).
    Each bond is a dict: iA, elemA, rA_cart, elemB, rB_cart, dist.
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
        cutoff = dmin * nn_scale
        for j in range(natoms):
            for s in range(len(shifts)):
                d = dist[j, s]
                if d <= cutoff:
                    key = (frozenset((elements[i], elements[j])), round(d, 2))
                    if key in seen:
                        continue
                    seen[key] = True
                    bonds.append(dict(
                        iA=i, elemA=elements[i], rA_cart=cart[i].copy(),
                        elemB=elements[j], rB_cart=images[j, s].copy(),
                        dist=float(d)))
    bonds.sort(key=lambda b: (b['elemA'], b['elemB'], b['dist']))
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


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('elfcar', nargs='?', default=None,
                    help='ELFCAR path (default: ./ELFCAR or ./02_scf/ELFCAR)')
    ap.add_argument('--out', default='ELF', help='output prefix (default: ELF)')
    ap.add_argument('--nn-scale', type=float, default=1.2,
                    help='neighbour cutoff = nn_scale x nearest distance (1.2)')
    ap.add_argument('--npoints', type=int, default=200,
                    help='samples per bond (default: 200)')
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

    bonds = nn_bonds(lattice, elements, frac, args.nn_scale)
    if not bonds:
        sys.stderr.write('ERROR: no nearest-neighbour bonds found\n')
        sys.exit(1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    print(f'{"bond":<12}{"length (A)":>12}{"ELF max":>10}{"ELF min":>10}')
    for k, b in enumerate(bonds):
        dist, elf = sample_bond(grid, lattice, b['rA_cart'], b['rB_cart'],
                                args.npoints)
        label = f"{b['elemA']}–{b['elemB']}  {b['dist']:.2f} Å"
        ax.plot(dist, elf, color=colors[k % len(colors)], lw=1.6, label=label)
        print(f"{b['elemA']+'-'+b['elemB']:<12}{b['dist']:>12.3f}"
              f"{elf.max():>10.3f}{elf.min():>10.3f}")

    ax.set_xlabel('Distance along bond (Å)', fontsize=12)
    ax.set_ylabel('ELF', fontsize=12)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', lw=0.7, ls=':')   # uniform-electron-gas ref
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=9, title='atom → atom')
    fig.tight_layout()

    for ext in ('png', 'pdf'):
        out = f'{args.out}_elf_bonds.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None)
        print(f'  Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
