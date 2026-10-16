#!/usr/bin/env python3
"""Relaxation history plots from 01_relax (matplotlib + plain Python, no pymatgen).

    python3 plot_relax.py RELAX_DIR OUT_PREFIX [LABEL]

  OUT_PREFIX_relax_energy.{png,pdf}    total energy E0 (sigma->0) per ionic step
  OUT_PREFIX_relax_pressure.{png,pdf}  external pressure per ionic step, only when the cell
                                       volume relaxes (ISIF = 3, 6, 7, 8, or the volume changes),
                                       with the target (PSTRESS, else 0) dashed
  OUT_PREFIX_relax_forces.{png,pdf}    Fx, Fy, Fz on atom 1 per ionic step (OUTCAR TOTAL-FORCE),
                                       with the EDIFFG force criterion dashed when EDIFFG < 0.
                                       Atom 1 is the one the relaxation keeps fixed (F F F): the
                                       force on it goes to zero as the rest of the cell relaxes

Every OUTCAR of the relaxation is read in the order it was written: OUTCAR.timeout<n>
(auto-continue at the walltime), OUTCAR.pass<n> (ht_runs.py restart), then OUTCAR
(OUTCAR.trim if there is no OUTCAR); a dotted line marks each restart.
"""
import glob
import os
import re
import sys

VOLUME_ISIF = (3, 6, 7, 8)


def outcars(d):
    """The relaxation's OUTCARs, oldest first."""
    def order(p):
        b = os.path.basename(p)
        m = re.match(r'OUTCAR\.(timeout|pass)(\d*)$', b)
        return (0, os.path.getmtime(p), int(m.group(2) or 0)) if m else (1, 0, 0)
    old = [p for p in glob.glob(os.path.join(d, 'OUTCAR.*'))
           if re.match(r'OUTCAR\.(timeout|pass)\d*$', os.path.basename(p))]
    cur = [p for p in (os.path.join(d, 'OUTCAR'), os.path.join(d, 'OUTCAR.trim')) if os.path.isfile(p)]
    return sorted(old, key=order) + cur[:1]


def parse_outcar(path):
    """[(E0, external pressure kB or None, volume A^3 or None, (Fx, Fy, Fz) on atom 1 or None)]
    per ionic step."""
    steps, p, vol, f1, in_free, in_force = [], None, None, None, False, 0
    with open(path, errors='replace') as fh:
        for line in fh:
            if in_force:                       # 1: dashes line next, 2: atom 1
                if in_force == 2:
                    t = line.split()
                    try:
                        f1 = tuple(float(x) for x in t[3:6])
                    except ValueError:
                        f1 = None
                    in_force = 0
                else:
                    in_force = 2
                continue
            if 'TOTAL-FORCE' in line and 'POSITION' in line:
                in_force = 1
            elif 'external pressure' in line:
                m = re.search(r'external pressure\s*=\s*([-\d.Ee+]+)', line)
                if m:
                    p = float(m.group(1))
            elif 'volume of cell' in line:
                m = re.search(r'volume of cell\s*:\s*([-\d.Ee+]+)', line)
                if m:
                    vol = float(m.group(1))
            elif 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
                in_free = True
            elif in_free and 'energy  without entropy' in line:
                m = re.search(r'energy\(sigma->0\)\s*=\s*([-\d.Ee+]+)', line)
                if m:
                    steps.append((float(m.group(1)), p, vol, f1))
                in_free, f1 = False, None
    return steps


def incar_tags(d):
    tags = {}
    try:
        for line in open(os.path.join(d, 'INCAR'), errors='replace'):
            line = line.split('#')[0].split('!')[0]
            for part in line.split(';'):
                if '=' in part:
                    k, v = part.split('=', 1)
                    tags[k.strip().upper()] = v.strip()
    except OSError:
        pass
    return tags


def read_relax(d):
    """{'energy', 'pressure', 'volume': lists per ionic step, 'restarts': step indices,
        'isif', 'pstress' (kB), 'volume_relax', 'converged'} or None if nothing was found."""
    E, P, V, F1, restarts, conv = [], [], [], [], [], False
    for k, f in enumerate(outcars(d)):
        st = parse_outcar(f)
        if k and st and E:
            restarts.append(len(E) + 0.5)
        for e, p, v, f1 in st:
            E.append(e); P.append(p); V.append(v); F1.append(f1)
        if os.path.basename(f) in ('OUTCAR', 'OUTCAR.trim'):
            with open(f, errors='replace') as fh:
                conv = 'reached required accuracy' in fh.read()
    if not E:
        return None
    t = incar_tags(d)
    try:
        isif = int(float(t.get('ISIF', '2')))
    except ValueError:
        isif = 2
    try:
        pstress = float(t.get('PSTRESS', '0'))
    except ValueError:
        pstress = 0.0
    try:
        ediffg = float(t.get('EDIFFG', '0'))
    except ValueError:
        ediffg = 0.0
    vs = [v for v in V if v is not None]
    vol_changes = len(vs) > 1 and (max(vs) - min(vs)) > 1e-4 * max(vs)
    return {'energy': E, 'pressure': P, 'volume': V, 'force1': F1, 'restarts': restarts,
            'isif': isif, 'pstress': pstress, 'ediffg': ediffg,
            'volume_relax': isif in VOLUME_ISIF or vol_changes, 'converged': conv}


def _fig(r, label, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    for k, x in enumerate(r['restarts']):
        ax.axvline(x, color='0.6', ls=':', lw=1, label='restart' if k == 0 else None)
    ax.set_xlabel('Ionic step')
    ax.set_title('%s: %s' % (label, title) if label else title)
    ax.grid(True, alpha=0.3)
    return plt, fig, ax


def plot_energy(r, out_png, label=''):
    plt, fig, ax = _fig(r, label, 'total energy during the relaxation')
    xs = list(range(1, len(r['energy']) + 1))
    ax.plot(xs, r['energy'], 'o-', color='#7c3aed', lw=1.5, ms=4)
    ax.set_ylabel('Total energy E0 (eV)')
    if r['restarts']:
        ax.legend(fontsize=8, loc='center right')
    dE = r['energy'][-1] - r['energy'][0]
    ax.text(0.98, 0.95, 'final %.5f eV\n$\\Delta E$ = %.4f eV, %d steps%s'
            % (r['energy'][-1], dE, len(xs), '' if r['converged'] else '\nnot converged'),
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            color='k' if r['converged'] else '#b91c1c')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(os.path.splitext(out_png)[0] + '.pdf')
    plt.close(fig)
    return out_png


def plot_pressure(r, out_png, label=''):
    pts = [(i + 1, p) for i, p in enumerate(r['pressure']) if p is not None]
    if not pts:
        return None
    plt, fig, ax = _fig(r, label, 'pressure during the relaxation')
    ax.plot([x for x, _ in pts], [p for _, p in pts], 's-', color='#0891b2', lw=1.5, ms=4,
            label='external pressure')
    ax.axhline(r['pstress'], color='k', ls='--', lw=1,
               label='target (PSTRESS %g kB)' % r['pstress'] if r['pstress'] else 'target (0)')
    ax.set_ylabel('Pressure (kB)')
    vs = [v for v in r['volume'] if v is not None]
    if vs:
        ax.text(0.98, 0.05, 'final %.2f kB, V = %.3f Å³' % (pts[-1][1], vs[-1]),
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(os.path.splitext(out_png)[0] + '.pdf')
    plt.close(fig)
    return out_png


def plot_forces(r, out_png, label=''):
    pts = [(i + 1, f) for i, f in enumerate(r.get('force1') or []) if f is not None]
    if not pts:
        return None
    plt, fig, ax = _fig(r, label, 'force on atom 1 during the relaxation')
    xs = [x for x, _ in pts]
    for k, (name, col, mk) in enumerate((('$F_x$', '#dc2626', 'o'), ('$F_y$', '#16a34a', 's'),
                                         ('$F_z$', '#2563eb', '^'))):
        ax.plot(xs, [f[k] for _, f in pts], mk + '-', color=col, lw=1.4, ms=4, label=name)
    ax.axhline(0, color='k', lw=0.8)
    if r.get('ediffg', 0) < 0:
        g = abs(r['ediffg'])
        ax.axhline(g, color='0.4', ls='--', lw=1, label='EDIFFG = %g eV/Å' % r['ediffg'])
        ax.axhline(-g, color='0.4', ls='--', lw=1)
    ax.set_ylabel('Force on atom 1 (eV/Å)')
    fx, fy, fz = pts[-1][1]
    ax.text(0.98, 0.05, 'final |F| = %.4f eV/Å' % (fx * fx + fy * fy + fz * fz) ** 0.5,
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9)
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    fig.savefig(os.path.splitext(out_png)[0] + '.pdf')
    plt.close(fig)
    return out_png


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    d, prefix = argv[1], argv[2]
    label = argv[3] if len(argv) > 3 else ''
    r = read_relax(d)
    if r is None:
        print('  no ionic steps in %s/OUTCAR - relaxation plots skipped' % d)
        return 0
    print('  Saved: %s' % os.path.basename(plot_energy(r, prefix + '_relax_energy.png', label)))
    if r['volume_relax']:
        out = plot_pressure(r, prefix + '_relax_pressure.png', label)
        if out:
            print('  Saved: %s' % os.path.basename(out))
    else:
        print('  fixed volume (ISIF=%d): no pressure plot' % r['isif'])
    out = plot_forces(r, prefix + '_relax_forces.png', label)
    print('  Saved: %s' % os.path.basename(out) if out else '  no forces in OUTCAR: force plot skipped')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
