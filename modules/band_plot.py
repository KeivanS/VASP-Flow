#!/usr/bin/env python3
"""Plot a band structure directly from VASP EIGENVAL (no sumo / pymatgen).

EIGENVAL is written incrementally during the run and survives even when
vasprun.xml is left truncated (e.g. an out-of-memory finish), so this is the
default band plotter in VASP-Flow.

Reads:  03_bands/EIGENVAL         eigenvalues + k-points
        03_bands/KPOINTS          line-mode path -> high-symmetry tick labels
        POSCAR / CONTCAR          reciprocal lattice for k-distances
        OUTCAR (04_dos > 02_scf > 03_bands)   Fermi energy
        03_bands/PROCAR           (optional, LORBIT=11) orbital projections
Writes: <prefix>_band.png/.pdf and — when PROCAR exists — the orbital-
        weighted FAT BANDS <prefix>_fatbands.png/.pdf: thin gray bands with
        colored markers per (element, l-channel), marker size ∝ projection
        weight.

Usage:  band_plot.py [BANDS_DIR] [--out PREFIX] [--ymin -4] [--ymax 4]
                     [--efermi EF] [--labels G,X,W,K,...] [--no-fatbands]
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


def read_elements(bands_dir):
    """Per-ion element list from POSCAR/CONTCAR (VASP5), or [] if unnamed."""
    for name in ('CONTCAR', 'POSCAR'):
        p = os.path.join(bands_dir, name)
        if os.path.isfile(p):
            ls = open(p).readlines()
            syms = ls[5].split()
            if syms and syms[0].lstrip('+-').isdigit():
                return []
            counts = [int(t) for t in ls[6].split()]
            out = []
            for s, n in zip(syms, counts):
                out += [s] * n
            return out
    return []


def read_procar(path):
    """Parse a LORBIT=11 PROCAR -> (w[nspin, nk, nb, nion, nl], lchans).

    Weights are l-summed (p = px+py+pz, etc.). Handles collinear spin (two
    stacked spin sections) and noncollinear/SOC files (the extra mx/my/mz
    blocks after each band's total block are skipped).
    """
    lmap = {'s': 's', 'p': 'p', 'd': 'd', 'f': 'f', 'x': 'd'}   # x2-y2 -> d
    w = None
    lchans, chan_of_col = [], []
    ik = ib = 0
    spin = -1
    nk = nb = nion = 0
    with open(path, errors='replace') as fh:
        for ln in fh:
            s = ln.strip()
            if s.startswith('# of k-points'):
                m = re.findall(r'(\d+)', s)
                nk, nb, nion = int(m[0]), int(m[1]), int(m[2])
                spin += 1
                if spin > 1:            # phase/extra section — stop
                    break
                if w is not None and spin == 1:
                    w = np.concatenate([w, np.zeros_like(w[:1])], axis=0)
                continue
            if s.startswith('k-point'):
                m = re.match(r'k-point\s+(\d+)', s)
                if m:
                    ik = int(m.group(1)) - 1
                continue
            if s.startswith('band'):
                m = re.match(r'band\s+(\d+)', s)
                if m:
                    ib = int(m.group(1)) - 1
                continue
            if s.startswith('ion '):    # column header defines the orbitals
                if not lchans:
                    cols = s.split()[1:-1]              # between 'ion' and 'tot'
                    chan_of_col = [lmap.get(c[0], None) for c in cols]
                    lchans = [l for l in ('s', 'p', 'd', 'f')
                              if l in chan_of_col]
                    w = np.zeros((1, nk, nb, nion, len(lchans)))
                continue
            if w is not None and s and s[0].isdigit():
                parts = s.split()
                ion = int(parts[0]) - 1
                if ion >= nion:
                    continue
                vals = parts[1:1 + len(chan_of_col)]
                # SOC files repeat the ion table (mx,my,mz) after the 'tot'
                # row; only the FIRST occurrence per (k, band, ion) is the
                # total weight — later ones are skipped by the check below.
                if np.any(w[spin, ik, ib, ion]):
                    continue
                for c, v in zip(chan_of_col, vals):
                    if c is not None:
                        w[spin, ik, ib, ion, lchans.index(c)] += float(v)
    return w, lchans


def kpath_geometry(bands_dir, kpts, labels_override=None):
    """x-coordinate along the concatenated path + high-symmetry ticks."""
    A = read_lattice(bands_dir)
    B = 2 * np.pi * np.linalg.inv(A).T
    kcart = kpts @ B
    nkdiv, klabels = read_kpoints_labels(bands_dir)
    if labels_override:
        klabels = labels_override
    nk = len(kpts)
    if not nkdiv or nk % nkdiv != 0:
        nkdiv = nk
    nseg = nk // nkdiv
    x = np.zeros(nk)
    ticks, ticklabels = [], []
    xoff = 0.0
    for sgm in range(nseg):
        sl = slice(sgm * nkdiv, (sgm + 1) * nkdiv)
        kc = kcart[sl]
        dloc = np.concatenate([[0.0],
                               np.cumsum(np.linalg.norm(np.diff(kc, axis=0), axis=1))])
        x[sl] = xoff + dloc
        s_lbl = klabels[2 * sgm]     if 2 * sgm     < len(klabels) else ''
        e_lbl = klabels[2 * sgm + 1] if 2 * sgm + 1 < len(klabels) else ''
        if sgm == 0:
            ticks.append(x[sl][0]); ticklabels.append(_fmt(s_lbl))
        else:
            prev, cur = ticklabels[-1], _fmt(s_lbl)
            ticklabels[-1] = cur if cur == prev else f'{prev}|{cur}'
        ticks.append(x[sl][-1]); ticklabels.append(_fmt(e_lbl))
        xoff = x[sl][-1]
    return x, ticks, ticklabels, nkdiv


def plot_fatbands(bands_dir, prefix, ymin, ymax, efermi=None,
                  labels_override=None, min_weight=0.05):
    """Orbital-weighted (fat) bands from PROCAR: thin gray bands + colored
    markers per (element, l-channel), marker area ∝ projection weight.
    Channels whose weight never reaches `min_weight` are dropped."""
    procar = os.path.join(bands_dir, 'PROCAR')
    kpts, eig = read_eigenval(os.path.join(bands_dir, 'EIGENVAL'))
    nspin, nk, nb = eig.shape
    if efermi is None:
        efermi = read_efermi(bands_dir)
    x, ticks, ticklabels, nkdiv = kpath_geometry(bands_dir, kpts, labels_override)

    w, lchans = read_procar(procar)
    if w is None or w.shape[1] != nk or w.shape[2] != nb:
        raise ValueError('PROCAR does not match EIGENVAL (k-points/bands)')
    elements = read_elements(bands_dir) or [f'ion{i+1}' for i in range(w.shape[3])]

    # (element, l) projections, weakest channels dropped
    projs = []
    for el in dict.fromkeys(elements):
        ions = [i for i, e in enumerate(elements) if e == el]
        for li, l in enumerate(lchans):
            pw = w[:, :, :, ions, li].sum(axis=3)        # (nspin, nk, nb)
            if float(pw.max()) >= min_weight:
                projs.append((f'{el} {l}', pw))
    if not projs:
        raise ValueError('no projection reaches the weight threshold')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    nseg = nk // nkdiv
    for sgm in range(nseg):                               # thin gray backbone
        sl = slice(sgm * nkdiv, (sgm + 1) * nkdiv)
        for s in range(nspin):
            for b in range(nb):
                ax.plot(x[sl], eig[s, sl, b] - efermi, color='0.55', lw=0.5,
                        zorder=1)
    colors = plt.get_cmap('tab10').colors
    E = eig - efermi                                      # (nspin, nk, nb)
    vis = (E >= ymin - 0.5) & (E <= ymax + 0.5)
    X = np.broadcast_to(x[None, :, None], E.shape)
    for p, (lbl, pw) in enumerate(projs):
        m = vis & (pw > 0.01)
        ax.scatter(X[m], E[m], s=28.0 * np.minimum(pw[m], 1.0),
                   color=colors[p % len(colors)], alpha=0.55, lw=0,
                   zorder=2 + p, label=lbl)
    for t in ticks:
        ax.axvline(t, color='k', lw=0.7, zorder=0)
    ax.axhline(0, color='gray', lw=0.7, ls=':')
    ax.set_xticks(ticks); ax.set_xticklabels(ticklabels, fontsize=11)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Energy − $E_F$ (eV)', fontsize=12)
    title = prefix[:-5] if prefix.endswith('_zoom') else prefix
    ax.set_title(f'{title} — fat bands (marker ∝ orbital weight, from PROCAR)',
                 fontsize=10)
    leg = ax.legend(fontsize=9, loc='upper right', framealpha=0.9,
                    scatterpoints=1)
    for h in leg.legend_handles:                          # uniform legend dots
        h.set_sizes([40]); h.set_alpha(1.0)
    ax.grid(True, axis='y', alpha=0.15)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        out = f'{prefix}_fatbands.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None)
        print(f'  Saved: {out}')
    plt.close(fig)
    return True


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
    title = prefix[:-5] if prefix.endswith('_zoom') else prefix
    ax.set_title(f'{title} — band structure (from EIGENVAL)', fontsize=10)
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
    ap.add_argument('--no-fatbands', action='store_true',
                    help='skip the PROCAR fat-bands plot')
    args = ap.parse_args()

    bd = args.bands_dir
    if bd is None:
        bd = '.' if os.path.isfile('EIGENVAL') else '03_bands'
    if not os.path.isfile(os.path.join(bd, 'EIGENVAL')):
        sys.stderr.write(f'ERROR: EIGENVAL not found in {bd}\n')
        sys.exit(1)
    labels = args.labels.split(',') if args.labels else None
    plot(bd, args.out, args.ymin, args.ymax, args.efermi, labels)

    # Fat bands whenever a PROCAR is available (LORBIT=11 in 03_bands).
    # Never let a PROCAR problem break the plain band plot above.
    if not args.no_fatbands and os.path.isfile(os.path.join(bd, 'PROCAR')):
        try:
            plot_fatbands(bd, args.out, args.ymin, args.ymax, args.efermi, labels)
        except Exception as e:
            sys.stderr.write(f'  note: fat bands skipped ({e})\n')


if __name__ == '__main__':
    main()
