#!/usr/bin/env python3
"""Cumulative total + projected DOS from a VASP DOSCAR (no sumo needed).

Produces two plots that share ONE energy window (the full eigenvalue range,
i.e. where the total DOS is non-zero, +0.5 eV margin), matching the GUI:
  <prefix>_cumulative_dos.{png,pdf}  total DOS (black) + element-stacked areas
  <prefix>_proj_dos.{png,pdf}        (element,orbital)-stacked areas + total

Usage: dos_plot.py [DOS_DIR] --out PREFIX [--proj dos_proj.json]
DOS_DIR (default ./ or ./04_dos) must hold DOSCAR + POSCAR/CONTCAR.
--proj points to a JSON list [{"element":X,"orbitals":[s,p,d,f]}, ...];
omit it (or leave empty) to make only the total plot.
"""
import argparse
import json
import os
import sys
import numpy as np


def _read_doscar(dos_dir):
    ion_elements = []
    for fname in ('CONTCAR', 'POSCAR'):
        p = os.path.join(dos_dir, fname)
        if os.path.isfile(p):
            ls = open(p).readlines()
            for el, cnt in zip(ls[5].split(), [int(x) for x in ls[6].split()]):
                ion_elements.extend([el] * cnt)
            break
    doscar = os.path.join(dos_dir, 'DOSCAR')
    if not ion_elements or not os.path.isfile(doscar):
        return None
    raw = open(doscar).readlines()
    nions = int(raw[0].split()[0])
    parts = raw[5].split()
    nedos, efermi = int(parts[2]), float(parts[3])
    tot = np.array([[float(x) for x in l.split()] for l in raw[6:6 + nedos]])
    energies = tot[:, 0] - efermi
    spin_pol = tot.shape[1] == 5
    return dict(raw=raw, nions=nions, nedos=nedos, tot=tot, energies=energies,
                spin_pol=spin_pol, ion_elements=ion_elements)


def _window(energies, dos_total):
    """smallest→largest eigenvalue: range where total DOS > 0, +0.5 eV margin."""
    d = np.abs(np.asarray(dos_total))
    nz = np.where(d > 1e-3 * d.max())[0] if d.max() > 0 else []
    if len(nz) == 0:
        return float(np.min(energies)), float(np.max(energies))
    return float(energies[nz[0]]) - 0.5, float(energies[nz[-1]]) + 0.5


def _finish(fig, ax, energies, xmin, xmax, label):
    from matplotlib.ticker import ScalarFormatter
    sc = ScalarFormatter(useOffset=False, useMathText=False); sc.set_scientific(False)
    ax.yaxis.set_major_formatter(sc)
    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel('Energy − $E_F$ (eV)', fontsize=12)
    ax.set_title(label)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()


def _save(fig, prefix, stem):
    for ext in ('png', 'pdf'):
        out = f'{prefix}_{stem}.{ext}'
        fig.savefig(out, dpi=150 if ext == 'png' else None)
        print(f'  Saved: {out}')


def total_dos(D, prefix):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from collections import defaultdict
    raw, nedos, tot = D['raw'], D['nedos'], D['tot']
    energies, spin_pol, ions = D['energies'], D['spin_pol'], D['ion_elements']
    el_up = defaultdict(lambda: np.zeros(nedos))
    el_dn = defaultdict(lambda: np.zeros(nedos))
    offset = 6 + nedos
    for i in range(D['nions']):
        start = offset + i * (nedos + 1) + 1
        d = np.array([[float(x) for x in l.split()] for l in raw[start:start + nedos]])
        el = ions[i] if i < len(ions) else f'ion{i}'
        if spin_pol:
            el_up[el] += d[:, 1::2].sum(axis=1); el_dn[el] += d[:, 2::2].sum(axis=1)
        else:
            el_up[el] += d[:, 1:].sum(axis=1)
    xmin, xmax = _window(energies, tot[:, 1] + (tot[:, 2] if spin_pol else 0.0))
    m = (energies >= xmin) & (energies <= xmax); en = energies[m]
    order = list(dict.fromkeys(ions))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(7, 5))
    tu, td = tot[:, 1], (tot[:, 2] if spin_pol else None)
    if spin_pol:
        cu = np.zeros(m.sum()); cd = np.zeros(m.sum())
        for i, el in enumerate(order):
            c = colors[i % len(colors)]
            pu = cu.copy(); cu = cu + el_up[el][m]
            pd = cd.copy(); cd = cd + el_dn[el][m]
            ax.fill_between(en, pu, cu, alpha=0.30, color=c); ax.plot(en, cu, color=c, lw=1.2)
            ax.fill_between(en, -pd, -cd, alpha=0.30, color=c); ax.plot(en, -cd, color=c, lw=1.2, ls='--')
        ax.plot(en, tu[m], color='k', lw=1.6, zorder=5); ax.plot(en, -td[m], color='k', lw=1.6, zorder=5, ls='--')
        ax.axhline(0, color='k', lw=0.9)
        yv = max(tu[m].max(), td[m].max()) * 1.1 or 1; ax.set_ylim(-yv, yv)
        ax.set_ylabel('DOS (states/eV)   ↑ up / ↓ down', fontsize=10)
        h = [Line2D([0], [0], color=colors[i % len(colors)], lw=1.5, label=el) for i, el in enumerate(order)]
        h += [Line2D([0], [0], color='k', lw=1.6, label='Total ↑'),
              Line2D([0], [0], color='k', lw=1.6, ls='--', label='Total ↓')]
        ax.legend(handles=h, fontsize=8, loc='upper left')
    else:
        cum = np.zeros(m.sum())
        for i, el in enumerate(order):
            c = colors[i % len(colors)]; prev = cum.copy(); cum = cum + el_up[el][m]
            ax.fill_between(en, prev, cum, alpha=0.35, color=c, label=el); ax.plot(en, cum, color=c, lw=1.2)
        ax.plot(en, tu[m], color='k', lw=1.6, zorder=5, label='Total')
        ax.set_ylim(bottom=0); ax.set_ylabel('DOS (states/eV)', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
    _finish(fig, ax, energies, xmin, xmax, f'Cumulative DOS — {prefix}')
    _save(fig, prefix, 'cumulative_dos'); plt.close(fig)


def proj_dos(D, prefix, proj_list):
    if not proj_list:
        return
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict
    raw, nedos, tot = D['raw'], D['nedos'], D['tot']
    energies, spin_pol, ions = D['energies'], D['spin_pol'], D['ion_elements']
    LM = {'s': (0, 1), 'p': (1, 4), 'd': (4, 9), 'f': (9, 16)}

    def cols(orb):
        lo, hi = LM[orb]
        if spin_pol:
            return [1 + 2 * j for j in range(lo, hi)], [2 + 2 * j for j in range(lo, hi)]
        return [1 + j for j in range(lo, hi)], []

    req = [(p['element'], o) for p in proj_list for o in p.get('orbitals', [])]
    up = defaultdict(lambda: np.zeros(nedos)); dn = defaultdict(lambda: np.zeros(nedos))
    offset = 6 + nedos
    for i in range(D['nions']):
        el = ions[i] if i < len(ions) else f'ion{i}'
        start = offset + i * (nedos + 1) + 1
        d = np.array([[float(x) for x in l.split()] for l in raw[start:start + nedos]])
        for rel, orb in req:
            if rel != el:
                continue
            cu, cd = cols(orb)
            cu = [c for c in cu if c < d.shape[1]]; cd = [c for c in cd if c < d.shape[1]]
            k = f'{el} {orb}'
            if cu: up[k] += d[:, cu].sum(axis=1)
            if cd: dn[k] += d[:, cd].sum(axis=1)
    keys, seen = [], set()
    for el, orb in req:
        k = f'{el} {orb}'
        if k in up and k not in seen:
            seen.add(k); keys.append(k)
    if not keys:
        return
    xmin, xmax = _window(energies, tot[:, 1] + (tot[:, 2] if spin_pol else 0.0))
    m = (energies >= xmin) & (energies <= xmax); en = energies[m]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(7, 5))
    tu = tot[:, 1]; td = tot[:, 2] if spin_pol else None
    if spin_pol:
        cu = np.zeros(m.sum()); cd = np.zeros(m.sum())
        for i, k in enumerate(keys):
            c = colors[i % len(colors)]
            pu = cu.copy(); cu = cu + up[k][m]; pd = cd.copy(); cd = cd + dn[k][m]
            ax.fill_between(en, pu, cu, alpha=0.35, color=c, label=k); ax.plot(en, cu, color=c, lw=1.0)
            ax.fill_between(en, -pd, -cd, alpha=0.35, color=c); ax.plot(en, -cd, color=c, lw=1.0, ls='--')
        ax.plot(en, tu[m], color='k', lw=1.6, zorder=5, label='Total'); ax.plot(en, -td[m], color='k', lw=1.6, zorder=5, ls='--')
        ax.axhline(0, color='k', lw=0.9); ax.set_ylabel('DOS (states/eV)   ↑ up / ↓ down', fontsize=10)
    else:
        cu = np.zeros(m.sum())
        for i, k in enumerate(keys):
            c = colors[i % len(colors)]; prev = cu.copy(); cu = cu + up[k][m]
            ax.fill_between(en, prev, cu, alpha=0.35, color=c, label=k); ax.plot(en, cu, color=c, lw=1.0)
        ax.plot(en, tu[m], color='k', lw=1.6, zorder=5, label='Total')
        ax.set_ylim(bottom=0); ax.set_ylabel('DOS (states/eV)', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    _finish(fig, ax, energies, xmin, xmax, f'Projected DOS (cumulative) — {prefix}')
    _save(fig, prefix, 'proj_dos'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dos_dir', nargs='?', default=None)
    ap.add_argument('--out', default='dos')
    ap.add_argument('--proj', default=None, help='dos_proj.json with orbital projections')
    args = ap.parse_args()
    dd = args.dos_dir or ('.' if os.path.isfile('DOSCAR') else '04_dos')
    D = _read_doscar(dd)
    if D is None:
        sys.stderr.write(f'ERROR: DOSCAR/POSCAR not found in {dd}\n'); sys.exit(1)
    total_dos(D, args.out)
    proj_list = []
    if args.proj and os.path.isfile(args.proj):
        try: proj_list = json.loads(open(args.proj).read())
        except Exception: pass
    proj_dos(D, args.out, proj_list)


if __name__ == '__main__':
    main()
