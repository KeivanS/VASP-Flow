#!/usr/bin/env python3
"""Plot a band structure directly from VASP EIGENVAL (no sumo / pymatgen).

EIGENVAL is written incrementally during the run and survives even when
vasprun.xml is left truncated (e.g. an out-of-memory finish), so this is the
default band plotter in VASP-Flow.

Reads:  03_bands/EIGENVAL         eigenvalues + k-points
        03_bands/KPOINTS          line-mode path -> high-symmetry tick labels
        POSCAR / CONTCAR          reciprocal lattice for k-distances
        OUTCAR (04_dos > 02_scf > 03_bands)   Fermi energy
Writes: <prefix>_band.png and .pdf

Usage:  band_plot.py [BANDS_DIR] [--out PREFIX] [--ymin -4] [--ymax 4]
                     [--efermi EF] [--labels G,X,W,K,...]
If BANDS_DIR is omitted it uses ./ (expects EIGENVAL here) or ./03_bands.
"""
import argparse
import os
import re
import sys
import numpy as np


def _fmt(lbl):
    return {'G': 'Γ', 'GAMMA': 'Γ', 'GshiftedGamma': 'Γ'}.get(lbl.upper(), lbl) if lbl else lbl


def read_eigenval(path):
    """Return (kpts_frac (nk,3), eig (nspin,nk,nb))."""
    with open(path) as fh:
        lines = fh.readlines()
    nspin = int(lines[0].split()[3])
    nelect, nk, nb = (int(x) for x in lines[5].split())
    kpts = np.zeros((nk, 3))
    eig = np.zeros((nspin, nk, nb))
    idx = 6
    for k in range(nk):
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        kpts[k] = [float(x) for x in lines[idx].split()[:3]]
        idx += 1
        for b in range(nb):
            parts = lines[idx].split()
            # parts: band_index E[up] (E_dn) occ...  -> energies are cols 1..nspin
            for s in range(nspin):
                eig[s, k, b] = float(parts[1 + s])
            idx += 1
    return kpts, eig


def read_lattice(bands_dir):
    for name in ('CONTCAR', 'POSCAR'):
        p = os.path.join(bands_dir, name)
        if os.path.isfile(p):
            ls = open(p).readlines()
            scale = abs(float(ls[1].split()[0]))
            A = np.array([[float(x) for x in ls[2 + i].split()[:3]] for i in range(3)]) * scale
            return A
    return np.eye(3)


def read_kpoints_labels(bands_dir):
    """Line-mode KPOINTS -> (nkdiv, [labels]) where labels come in start/end pairs."""
    p = os.path.join(bands_dir, 'KPOINTS')
    if not os.path.isfile(p):
        return None, []
    ls = open(p).readlines()
    try:
        nkdiv = int(ls[1].split()[0])
    except Exception:
        return None, []
    labels = []
    for ln in ls[4:]:
        s = ln.strip()
        if not s:
            continue
        m = re.search(r'!\s*(\S+)', s)
        labels.append(m.group(1) if m else '')
    return nkdiv, labels


def read_efermi(bands_dir):
    root = os.path.dirname(os.path.abspath(bands_dir))
    for rel in ('04_dos/OUTCAR', '02_scf/OUTCAR', os.path.join(bands_dir, 'OUTCAR')):
        p = rel if os.path.isabs(rel) else os.path.join(root, rel)
        if os.path.isfile(p):
            m = re.findall(r'E-fermi\s*:\s*([-\d.]+)', open(p, errors='replace').read())
            if m:
                return float(m[-1])
    return 0.0


def plot(bands_dir, prefix, ymin, ymax, efermi=None, labels_override=None):
    kpts, eig = read_eigenval(os.path.join(bands_dir, 'EIGENVAL'))
    nspin, nk, nb = eig.shape
    A = read_lattice(bands_dir)
    B = 2 * np.pi * np.linalg.inv(A).T          # reciprocal lattice (rows = b1,b2,b3)
    kcart = kpts @ B
    if efermi is None:
        efermi = read_efermi(bands_dir)

    nkdiv, klabels = read_kpoints_labels(bands_dir)
    if labels_override:
        klabels = labels_override
    if not nkdiv or nk % nkdiv != 0:
        nkdiv = nk                              # single segment fallback
    nseg = nk // nkdiv

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    up_c, dn_c = '#1f4fd8', '#d81f2a'
    xoff = 0.0
    ticks, ticklabels = [], []
    for sgm in range(nseg):
        sl = slice(sgm * nkdiv, (sgm + 1) * nkdiv)
        kc = kcart[sl]
        dloc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
        x = xoff + dloc
        for s, c, ls in ((0, up_c, '-'), (1, dn_c, '--'))[:nspin]:
            for b in range(nb):
                ax.plot(x, eig[s, sl, b] - efermi, color=c, lw=0.9, ls=ls, alpha=0.85)
        # start / end tick labels for this segment
        s_lbl = klabels[2 * sgm]     if 2 * sgm     < len(klabels) else ''
        e_lbl = klabels[2 * sgm + 1] if 2 * sgm + 1 < len(klabels) else ''
        if sgm == 0:
            ticks.append(x[0]); ticklabels.append(_fmt(s_lbl))
        else:                                    # merge with previous end if same point
            prev = ticklabels[-1]
            cur = _fmt(s_lbl)
            ticklabels[-1] = cur if cur == prev else f'{prev}|{cur}'
        ticks.append(x[-1]); ticklabels.append(_fmt(e_lbl))
        xoff = x[-1]

    for t in ticks:
        ax.axvline(t, color='k', lw=0.7, zorder=0)
    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, fontsize=11)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Energy − $E_F$ (eV)', fontsize=12)
    ax.set_title(f'{prefix} — band structure (from EIGENVAL)', fontsize=10)
    if nspin == 2:
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([0], [0], color=up_c, lw=1.4, label='spin ↑'),
                           Line2D([0], [0], color=dn_c, lw=1.4, ls='--', label='spin ↓')],
                  fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.15)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        out = f'{prefix}_band.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None)
        print(f'  Saved: {out}')
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bands_dir', nargs='?', default=None,
                    help='directory with EIGENVAL (default ./ or ./03_bands)')
    ap.add_argument('--out', default='bands', help='output prefix (default: bands)')
    ap.add_argument('--ymin', type=float, default=-4.0)
    ap.add_argument('--ymax', type=float, default=4.0)
    ap.add_argument('--efermi', type=float, default=None)
    ap.add_argument('--labels', default=None,
                    help='comma-separated high-symmetry labels (override KPOINTS)')
    args = ap.parse_args()

    bd = args.bands_dir
    if bd is None:
        bd = '.' if os.path.isfile('EIGENVAL') else '03_bands'
    if not os.path.isfile(os.path.join(bd, 'EIGENVAL')):
        sys.stderr.write(f'ERROR: EIGENVAL not found in {bd}\n')
        sys.exit(1)
    labels = args.labels.split(',') if args.labels else None
    plot(bd, args.out, args.ymin, args.ymax, args.efermi, labels)


if __name__ == '__main__':
    main()
