#!/usr/bin/env python3
"""Cumulative LOBSTER projected DOS from DOSCAR.lobster, with the VASP total
DOS overlaid for comparison.

Same layout as the 04_dos projected-DOS plot (modules/dos_plot.py): each
(element, orbital) channel is summed over all atoms of that element and
stacked, so the areas don't overlap.  Two reference curves are drawn on top:

  solid black   VASP total DOS from 04_dos (tetrahedron, ISMEAR=-5) -- the
                projection-free reference
  dashed grey   LOBSTER total (identically the sum of the stacked pDOS)

The gap between them is the projection error: LOBSTER re-expands the PAW
states in a local atomic-orbital basis, so its total is reconstructed rather
than counted, and falls short by roughly the charge spilling.  Note the two
curves also carry different broadening (VASP tetrahedron vs LOBSTER Gaussian
gaussianSmearingWidth), which is stated in the legend so the comparison
isn't read as pure projection error.

Usage:
    lobster_dos_plot.py <lobster_dir> <out_stem> <project_label>
                        [--dos-dir 04_dos] [--emin -10] [--emax 5]

    <out_stem>  full path WITHOUT extension, e.g. analysis/MoSe2_lobster_dos
    Output:     <out_stem>.png  and  <out_stem>.pdf
"""
import argparse
import os
import re
import sys
import numpy as np
from collections import defaultdict


def _ion_elements(d):
    """Per-ion element symbols from POSCAR/CONTCAR in directory `d`."""
    for fname in ('CONTCAR', 'POSCAR'):
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            ls = open(p).readlines()
            try:
                out = []
                for el, cnt in zip(ls[5].split(), [int(x) for x in ls[6].split()]):
                    out.extend([el] * cnt)
                return out
            except (ValueError, IndexError):
                return []
    return []


def read_lobster_doscar(lobster_dir):
    """Parse DOSCAR.lobster -> dict, or None when the file is absent.

    Energies are already Fermi-referenced by LOBSTER (E_F = 0).  Each per-ion
    header carries its own basis labels, e.g. '; Z= 42; 5s 4d_xy 4d_yz ...',
    so the orbital of every column is read from the file rather than assumed
    from an lm ordering (the minimal basis differs per element).
    """
    path = os.path.join(lobster_dir, 'DOSCAR.lobster')
    if not os.path.isfile(path):
        return None
    L = open(path).read().splitlines()
    try:
        nions = int(L[0].split()[0])
        nedos = int(L[5].split()[2])
    except (ValueError, IndexError):
        return None

    def block(start):
        return np.array([[float(x) for x in L[i].split()]
                         for i in range(start, start + nedos)])

    tot = block(6)
    spin_pol = tot.shape[1] >= 5          # E, up, dn, int_up, int_dn
    energies = tot[:, 0]

    ions = _ion_elements(lobster_dir)
    # channel key 'El l' -> summed pDOS (up, and dn when spin-polarised)
    up = defaultdict(lambda: np.zeros(nedos))
    dn = defaultdict(lambda: np.zeros(nedos))
    order = []
    i = 6 + nedos
    for n in range(nions):
        if i >= len(L):
            break
        head, i = L[i], i + 1
        labels = head.split(';')[-1].split() if ';' in head else []
        d = block(i); i += nedos
        el = ions[n] if n < len(ions) else f'ion{n}'
        for j, lab in enumerate(labels):
            m = re.match(r'\d*([spdf])', lab)      # '4d_xy' -> d, '5s' -> s
            if not m:
                continue
            key = f'{el} {m.group(1)}'
            if spin_pol:
                cu, cd = 1 + 2 * j, 2 + 2 * j
                if cd < d.shape[1]:
                    up[key] += d[:, cu]; dn[key] += d[:, cd]
            elif 1 + j < d.shape[1]:
                up[key] += d[:, 1 + j]
            if key not in order:
                order.append(key)

    return dict(nedos=nedos, energies=energies, spin_pol=spin_pol,
                tot_up=tot[:, 1], tot_dn=tot[:, 2] if spin_pol else None,
                up=up, dn=dn, order=order)


def read_vasp_total(dos_dir):
    """VASP total DOS from `dos_dir`/DOSCAR, Fermi-shifted.

    Returns (energies, up, dn|None, method) or None.  `method` reports the
    integration scheme read from the matching INCAR so the legend can't claim
    'tetrahedron' for a run that actually used smearing.
    """
    doscar = os.path.join(dos_dir, 'DOSCAR')
    if not os.path.isfile(doscar):
        return None
    raw = open(doscar).readlines()
    try:
        nedos, efermi = int(raw[5].split()[2]), float(raw[5].split()[3])
        tot = np.array([[float(x) for x in l.split()]
                        for l in raw[6:6 + nedos]])
    except (ValueError, IndexError):
        return None
    spin_pol = tot.shape[1] >= 5

    method = 'VASP total'
    incar = os.path.join(dos_dir, 'INCAR')
    if os.path.isfile(incar):
        m = re.search(r'^\s*ISMEAR\s*=\s*(-?\d+)', open(incar).read(),
                      re.MULTILINE)
        if m:
            ism = int(m.group(1))
            sig = re.search(r'^\s*SIGMA\s*=\s*([\d.]+)', open(incar).read(),
                            re.MULTILINE)
            method = ('VASP total (tetrahedron)' if ism == -5 else
                      f'VASP total (Gaussian σ={sig.group(1)} eV)' if sig
                      else 'VASP total (smearing)')
    return (tot[:, 0] - efermi, tot[:, 1],
            tot[:, 2] if spin_pol else None, method)


def _lobster_sigma(lobster_dir):
    """gaussianSmearingWidth from lobsterin, as a display string or None."""
    p = os.path.join(lobster_dir, 'lobsterin')
    if not os.path.isfile(p):
        return None
    m = re.search(r'^\s*gaussianSmearingWidth\s+([\d.]+)', open(p).read(),
                  re.MULTILINE | re.IGNORECASE)
    return m.group(1) if m else None


def _window(energies, dos):
    """Energy range where the DOS is non-negligible, +0.5 eV margin."""
    d = np.abs(np.asarray(dos))
    nz = np.where(d > 1e-3 * d.max())[0] if d.max() > 0 else []
    if len(nz) == 0:
        return float(np.min(energies)), float(np.max(energies))
    return float(energies[nz[0]]) - 0.5, float(energies[nz[-1]]) + 0.5


def plot_lobster_dos(lobster_dir, out_stem, project_label, dos_dir=None,
                     emin=None, emax=None):
    """Draw the plot and save <out_stem>.png + .pdf.  False if no input."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    D = read_lobster_doscar(lobster_dir)
    if D is None or not D['order']:
        return False

    # Overlay reference: prefer the real 04_dos tetrahedron run; fall back to
    # the DOSCAR of the LOBSTER NSCF itself so the plot still works alone.
    V = read_vasp_total(dos_dir) if dos_dir else None
    if V is None:
        V = read_vasp_total(lobster_dir)
        if V is not None:
            V = (V[0], V[1], V[2], V[3] + ' — 08_lobster NSCF, no 04_dos')

    en, spin_pol = D['energies'], D['spin_pol']
    tot_ref = D['tot_up'] + (D['tot_dn'] if spin_pol else 0.0)
    xmin, xmax = _window(en, tot_ref)
    if emin is not None: xmin = float(emin)
    if emax is not None: xmax = float(emax)
    m = (en >= xmin) & (en <= xmax)
    x = en[m]

    sigma = _lobster_sigma(lobster_dir)
    lob_lbl = 'LOBSTER total (Σ pDOS'
    lob_lbl += f', Gaussian σ={sigma} eV)' if sigma else ')'

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(7, 5))

    if spin_pol:
        cu = np.zeros(m.sum()); cd = np.zeros(m.sum())
        for i, k in enumerate(D['order']):
            c = colors[i % len(colors)]
            pu = cu.copy(); cu = cu + D['up'][k][m]
            pd = cd.copy(); cd = cd + D['dn'][k][m]
            ax.fill_between(x, pu, cu, alpha=0.35, color=c, label=k)
            ax.plot(x, cu, color=c, lw=1.0)
            ax.fill_between(x, -pd, -cd, alpha=0.35, color=c)
            ax.plot(x, -cd, color=c, lw=1.0, ls='--')
        ax.plot(x,  D['tot_up'][m], color='0.45', lw=1.2, ls='--', zorder=4,
                label=lob_lbl)
        ax.plot(x, -D['tot_dn'][m], color='0.45', lw=1.2, ls='--', zorder=4)
        ax.axhline(0, color='k', lw=0.9)
        ax.set_ylabel('DOS (states/eV)   ↑ up  /  ↓ down', fontsize=10)
    else:
        cu = np.zeros(m.sum())
        for i, k in enumerate(D['order']):
            c = colors[i % len(colors)]
            pu = cu.copy(); cu = cu + D['up'][k][m]
            ax.fill_between(x, pu, cu, alpha=0.35, color=c, label=k)
            ax.plot(x, cu, color=c, lw=1.0)
        ax.plot(x, D['tot_up'][m], color='0.45', lw=1.2, ls='--', zorder=4,
                label=lob_lbl)
        ax.set_ylim(bottom=0)
        ax.set_ylabel('DOS (states/eV)', fontsize=12)

    # The comparison curve: projection-free VASP total, solid black.
    if V is not None:
        ve, vu, vd, vlabel = V
        vm = (ve >= xmin) & (ve <= xmax)
        ax.plot(ve[vm], vu[vm], color='k', lw=1.7, zorder=6, label=vlabel)
        if spin_pol and vd is not None:
            ax.plot(ve[vm], -vd[vm], color='k', lw=1.7, zorder=6, ls='--')

    sc = ScalarFormatter(useOffset=False, useMathText=False)
    sc.set_scientific(False)
    ax.yaxis.set_major_formatter(sc)
    ax.axvline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('Energy − $E_F$ (eV)', fontsize=12)
    ax.set_title(f'LOBSTER projected DOS (cumulative) — {project_label}')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_stem}.{ext}', dpi=150 if ext == 'png' else None)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('lobster_dir', help='directory holding DOSCAR.lobster')
    ap.add_argument('out_stem', help='output path without extension')
    ap.add_argument('project_label', help='project name shown in the title')
    ap.add_argument('--dos-dir', default=None,
                    help='04_dos directory whose total DOS is overlaid')
    ap.add_argument('--emin', type=float, default=None)
    ap.add_argument('--emax', type=float, default=None)
    args = ap.parse_args()
    ok = plot_lobster_dos(args.lobster_dir, args.out_stem, args.project_label,
                          dos_dir=args.dos_dir, emin=args.emin, emax=args.emax)
    if not ok:
        sys.stderr.write(f'ERROR: DOSCAR.lobster not found in {args.lobster_dir}\n')
        sys.exit(1)
    print(f'  Saved: {args.out_stem}.png / .pdf')


if __name__ == '__main__':
    main()
