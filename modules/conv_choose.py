#!/usr/bin/env python3
"""choose_params.py — auto-select converged ENCUT and k-mesh.

Scans the convergence-test runs
    encut/encut_<E>/OUTCAR
    kpoints/<nx>x<ny>x<nz>/OUTCAR
(relative to this script's directory), computes the change in the force on
atom 1 and in the stress diagonal between successive parameter values, and
selects the smallest parameter from which EVERY subsequent change stays
below both thresholds:

    max component change of the force on atom 1  <  5 meV/Å
    max change of a stress-diagonal component    <  0.5 kbar

The convergence-test POSCAR has atom 1 displaced by (0.01, 0.02, 0.03) Å,
so the force on it is non-zero and its convergence is meaningful.

Writes converged_params.txt next to this script; production phase-2 scripts
read it as the default ENCUT / k-mesh.

Usage: python3 choose_params.py [--ftol MEV_PER_A] [--ptol KBAR]

Self-contained: python3 standard library only.
"""
import os, re, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))


def _read(path):
    try:
        with open(path, errors='replace') as f:
            return f.read()
    except OSError:
        return ''


def parse_forces_first_atom(text):
    """(Fx, Fy, Fz) on atom 1 in eV/Å, last ionic step, or None."""
    blocks = list(re.finditer(r'TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\s*\n', text))
    if not blocks:
        return None
    m = re.search(r'^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)'
                  r'\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
                  text[blocks[-1].end():], re.MULTILINE)
    return (float(m.group(4)), float(m.group(5)), float(m.group(6))) if m else None


def parse_pressure_diagonal(text):
    """(Pxx, Pyy, Pzz) in kBar, last ionic step, or None."""
    m = re.findall(r'^\s+in kB\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
                   text, re.MULTILINE)
    return tuple(float(x) for x in m[-1]) if m else None


def parse_energy(text):
    m = re.findall(r'energy\s+without\s+entropy\s*=\s*([-\d.]+)', text)
    return float(m[-1]) if m else None


def collect(dtype):
    """Sorted [(label, forces, pressures, energy)] for one dataset, or []."""
    base = os.path.join(HERE, dtype)
    if not os.path.isdir(base):
        return []
    entries = []
    for name in os.listdir(base):
        out = os.path.join(base, name, 'OUTCAR')
        if not os.path.isfile(out):
            continue
        nums = [int(v) for v in re.findall(r'\d+', name)]
        if not nums:
            continue
        entries.append((nums, name))
    entries.sort()
    rows = []
    for _, name in entries:
        text = _read(os.path.join(base, name, 'OUTCAR'))
        F, P, E = (parse_forces_first_atom(text),
                   parse_pressure_diagonal(text), parse_energy(text))
        if F is None or P is None:
            print(f'  WARNING: {dtype}/{name}/OUTCAR incomplete '
                  f'(no {"forces" if F is None else "stress"}) — skipped')
            continue
        rows.append((name, F, P, E))
    return rows


def choose(rows, ftol_ev, ptol_kbar):
    """Index of the smallest parameter from which all later successive
    changes are below both tolerances, plus the per-step diff table."""
    diffs = []
    for (n0, F0, P0, _), (n1, F1, P1, _) in zip(rows, rows[1:]):
        df = max(abs(a - b) for a, b in zip(F0, F1))
        dp = max(abs(a - b) for a, b in zip(P0, P1))
        diffs.append((df, dp))
    chosen = None
    for i in range(len(diffs)):
        if all(df < ftol_ev and dp < ptol_kbar for df, dp in diffs[i:]):
            chosen = i
            break
    return chosen, diffs


def report(dtype, rows, chosen, diffs, ftol_mev, ptol_kbar):
    unit = 'eV' if dtype == 'encut' else ''
    print(f'\n── {dtype} convergence '
          f'(ΔF < {ftol_mev} meV/Å on atom 1, ΔP < {ptol_kbar} kbar) ──')
    print(f'  {"value":>12s}  {"E (eV)":>14s}  {"ΔF (meV/Å)":>11s}  {"ΔP (kbar)":>10s}')
    for i, (name, F, P, E) in enumerate(rows):
        df = f'{diffs[i - 1][0] * 1000:11.2f}' if i > 0 else ' ' * 11
        dp = f'{diffs[i - 1][1]:10.3f}'        if i > 0 else ' ' * 10
        mark = '  ← selected' if chosen is not None and i == chosen else ''
        e = f'{E:14.6f}' if E is not None else ' ' * 14
        print(f'  {name:>12s}  {e}  {df}  {dp}{mark}')
    if chosen is None:
        print(f'  NOT CONVERGED within the tested range — extend the range;'
              f' falling back to the largest value tested ({rows[-1][0]} {unit}).')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--ftol', type=float, default=5.0,
                    help='max force change on atom 1, meV/Å (default 5)')
    ap.add_argument('--ptol', type=float, default=0.5,
                    help='max stress-diagonal change, kbar (default 0.5)')
    args = ap.parse_args()
    ftol_ev = args.ftol / 1000.0

    out_lines = [
        '# Auto-selected convergence parameters (choose_params.py)',
        f'# criteria: successive change of force on atom 1 < {args.ftol} meV/A',
        f'#           and of stress diagonal < {args.ptol} kbar',
    ]
    found_any = False
    for dtype, key in (('encut', 'ENCUT'), ('kpoints', 'KMESH')):
        rows = collect(dtype)
        if len(rows) < 2:
            if rows:
                print(f'\n── {dtype}: only one completed run — cannot assess convergence')
            continue
        found_any = True
        chosen, diffs = choose(rows, ftol_ev, args.ptol)
        report(dtype, rows, chosen, diffs, args.ftol, args.ptol)
        idx = chosen if chosen is not None else len(rows) - 1
        name = rows[idx][0]
        value = re.search(r'\d+', name).group() if dtype == 'encut' else name
        out_lines.append(f'{key} = {value}')
        out_lines.append(f'{key}_CONVERGED = {"yes" if chosen is not None else "no"}')

    if not found_any:
        print('No completed convergence runs found — nothing to select.')
        sys.exit(1)

    path = os.path.join(HERE, 'converged_params.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f'\nWrote {os.path.relpath(path)}')


if __name__ == '__main__':
    main()
