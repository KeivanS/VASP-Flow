#!/usr/bin/env python3
"""Plot COHP, COBI, or COOP vs. energy from a LOBSTER *CAR.lobster file.

Portrait layout: energy on the vertical axis (Fermi-referenced, E_F = 0);
the descriptor (bonding positive) on the horizontal axis.  Thick black =
total (mean per bond); thin overlays = per-bond nearest-neighbour shells.
Legend and stats box are placed outside the plot area so they never
obscure the curves.

Usage:
    cohp_plot.py <lobster_dir> <out_stem> <project_label>
                 [--which cohp|cobi|coop] [--emin -10] [--emax 5]

    <out_stem>  full path WITHOUT extension, e.g. analysis/MoSe2_cohp
    Output:     <out_stem>.png  and  <out_stem>.pdf
"""
import os
import sys
import argparse
import re
import numpy as np
from collections import defaultdict


def plot_cohp_cobi(lobster_dir, out_stem, project_label,
                   which='cohp', emin=-10.0, emax=5.0):
    """Draw COHP/COBI/COOP and save <out_stem>.png + <out_stem>.pdf.

    Returns True on success, False when the *CAR.lobster file is absent.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    files = {'cohp': 'COHPCAR.lobster', 'cobi': 'COBICAR.lobster',
             'coop': 'COOPCAR.lobster'}
    path = os.path.join(lobster_dir, files.get(which, 'COHPCAR.lobster'))
    if not os.path.isfile(path):
        return False
    lines = open(path).readlines()
    ctrl = lines[1].split()
    ncol, nspin = int(ctrl[0]), int(ctrl[1])
    labels = [lines[2 + k].strip() for k in range(ncol)]
    data = np.array([ln.split() for ln in lines[2 + ncol:] if ln.strip()], dtype=float)
    if data.size == 0:
        return False
    E = data[:, 0]
    sign = -1.0 if which == 'cohp' else 1.0   # plot bonding-positive for all three

    def curve(idx):
        y = np.zeros(len(E))
        for s in range(nspin):
            y += data[:, 1 + s * 2 * ncol + 2 * idx]
        return sign * y

    def integ(idx):
        y = np.zeros(len(E))
        for s in range(nspin):
            y += data[:, 1 + s * 2 * ncol + 2 * idx + 1]
        return y

    # TOTAL per cell. Same-atom pairs (e.g. La1->La1 ± T) appear twice → weight 1/2.
    label_re = re.compile(r"No\.\d+:([A-Za-z]+)(\d+)->([A-Za-z]+)(\d+)\(([0-9.]+)\)")
    groups = defaultdict(list)
    total  = np.zeros(len(E))
    itotal = np.zeros(len(E))
    nbonds = 0.0
    for k in range(1, ncol):
        mlab = label_re.match(labels[k])
        same = bool(mlab) and mlab.group(1, 2) == mlab.group(3, 4)
        w = 0.5 if same else 1.0
        total  += w * curve(k)
        itotal += w * integ(k)
        nbonds += w
        if mlab:
            a, b, dist = mlab.group(1), mlab.group(3), float(mlab.group(5))
            groups["-".join(sorted((a, b)))].append((dist, k, same))
    # Normalise to MEAN PER BOND (LOBSTER's "Average" convention).
    if nbonds > 0:
        total  = total  / nbonds
        itotal = itotal / nbonds

    # Integrated value and antibonding metrics at E_F.
    icoxp = float(np.interp(0.0, E, itotal))
    below = E <= 0.0
    a_int = sign * float(np.trapz(np.minimum(total, 0.0)[below], E[below]))
    b_int = icoxp - a_int
    fab   = abs(a_int) / (abs(a_int) + abs(b_int)) if abs(a_int) + abs(b_int) > 0 else 0.0

    # Bonding→antibonding threshold: + → - crossing of total nearest to E_F.
    ecross = None
    s_arr = np.sign(total)
    for i in np.where(np.diff(s_arr) != 0)[0]:
        d = total[i + 1] - total[i]
        if d >= 0:
            continue
        e0 = E[i] - total[i] * (E[i + 1] - E[i]) / d
        if ecross is None or abs(e0) < abs(ecross):
            ecross = e0

    # Per-bond shell overlays: 1st shell solid, 2nd dashed.
    def _shell(pair, sel, ls_, tag=''):
        n    = len(sel)
        mult = int(round(sum(0.5 if s else 1.0 for _, _, s in sel)))
        pb   = sum(curve(k) for _, k, _ in sel) / n
        ipb  = float(np.interp(0.0, E, sum(integ(k) for _, k, _ in sel) / n))
        return (f"{pair} ({sel[0][0]:.2f} Å{tag}, ×{mult}): {ipb:+.2f}/bond",
                pb, ls_)

    nn = []
    for pair, items in sorted(groups.items()):
        dmin   = min(d for d, _, _ in items)
        shell1 = [it for it in items if it[0] <= dmin * 1.05]
        rest   = [it for it in items if it[0] > dmin * 1.05]
        nn.append(_shell(pair, shell1, '-'))
        if rest:
            d2 = min(d for d, _, _ in rest)
            nn.append(_shell(pair, [it for it in rest if it[0] <= d2 * 1.05],
                             '--', ', 2nd'))

    emin, emax = float(emin), float(emax)
    m  = (E >= emin) & (E <= emax)
    Em = E[m]
    xlabel = {'cohp': '−COHP', 'cobi': 'COBI', 'coop': 'COOP'}[which] + ' (bonding +)'

    fig, ax = plt.subplots(figsize=(3.6, 6.4))   # portrait: energy vertical
    t = total[m]
    ax.fill_betweenx(Em, 0, t, where=(t >= 0), color='#2a9d4a', alpha=0.30)
    ax.fill_betweenx(Em, 0, t, where=(t < 0),  color='#c0392b', alpha=0.30)
    ax.plot(t, Em, color='k', lw=1.8, label='total (mean per bond)')
    for lbl, yy, ls in nn:
        ax.plot(yy[m], Em, lw=0.9, alpha=0.9, ls=ls, label=lbl)
    ax.axvline(0, color='k', lw=0.7)
    ax.axhline(0, color='gray', lw=0.9, ls='--')
    ax.set_ylim(emin, emax)
    ax.set_ylabel('Energy − $E_F$ (eV)', fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(f'{which.upper()} — {project_label}', fontsize=10)
    # Legend and stats box placed outside the axes so they never overlap curves.
    ax.legend(fontsize=6.5, loc='upper left',
              bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.2)

    iname = {'cohp': 'ICOHP', 'cobi': 'ICOBI', 'coop': 'ICOOP'}[which]
    unit  = ' eV' if which == 'cohp' else ''
    box = (f"{iname}(E$_F$) = {icoxp:.3f}{unit}/bond\n"
           f"  (mean of {nbonds:.0f} bonds;\n"
           f"  cell {icoxp*nbonds:.2f}{unit})\n"
           f"A{iname[1:]}(E$_F$) = {a_int:.3f}{unit}/bond\n"
           f"f$_{{AB}}$ = {100*fab:.1f}%\n"
           + (f"B→AB @ {ecross:+.2f} eV" if ecross is not None else "B→AB: no crossing"))
    ax.text(1.02, 0.0, box, transform=ax.transAxes, fontsize=7.5,
            va='bottom', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='0.6', alpha=0.85))
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out_stem}.{ext}',
                    dpi=150 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('lobster_dir',
                    help='directory containing *CAR.lobster files')
    ap.add_argument('out_stem',
                    help='output path without extension, e.g. analysis/MoSe2_cohp')
    ap.add_argument('project_label',
                    help='project name shown in the plot title')
    ap.add_argument('--which', choices=('cohp', 'cobi', 'coop'), default='cohp',
                    help='which descriptor to plot (default: cohp)')
    ap.add_argument('--emin', type=float, default=-10.0,
                    help='lower energy limit in eV (default -10)')
    ap.add_argument('--emax', type=float, default=5.0,
                    help='upper energy limit in eV (default 5)')
    args = ap.parse_args()
    ok = plot_cohp_cobi(args.lobster_dir, args.out_stem, args.project_label,
                        which=args.which, emin=args.emin, emax=args.emax)
    if not ok:
        sys.stderr.write(
            f'ERROR: {args.which.upper()}CAR.lobster not found in {args.lobster_dir}\n')
        sys.exit(1)
    print(f'  Saved: {args.out_stem}.png / .pdf')


if __name__ == '__main__':
    main()
