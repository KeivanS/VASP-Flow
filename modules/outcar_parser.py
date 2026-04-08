"""
outcar_parser.py — modular VASP OUTCAR parsing utilities.

Each function accepts the full OUTCAR text as a string and returns
the relevant quantity from the LAST ionic step (or the only step for
single-point / convergence calculations).

All functions return None (or an empty list) if the quantity is not
found, so callers can gracefully skip failed / incomplete runs.
"""

import re
from typing import Optional, Tuple, List


# ── helpers ───────────────────────────────────────────────────────────────────

def _last(matches):
    """Return the last element of a list, or None if empty."""
    return matches[-1] if matches else None


# ── public API ────────────────────────────────────────────────────────────────

def parse_energy(text: str) -> Optional[float]:
    """Total energy without entropy from the last ionic step (eV)."""
    matches = re.findall(r'energy\s+without\s+entropy\s*=\s*([-\d.]+)', text)
    m = _last(matches)
    return float(m) if m is not None else None


def parse_fermi_energy(text: str) -> Optional[float]:
    """Fermi energy from the last occurrence (eV)."""
    matches = re.findall(r'E-fermi\s*:\s*([-\d.]+)', text)
    m = _last(matches)
    return float(m) if m is not None else None


def parse_pressure_diagonal(text: str) -> Optional[Tuple[float, float, float]]:
    """Return (Pxx, Pyy, Pzz) in kBar from the last ionic step.

    VASP prints the stress tensor as:
        in kB   -0.01   -0.01   -0.01    0.00    0.00    0.00
    where the six values are XX YY ZZ XY YZ ZX.
    """
    matches = re.findall(
        r'^\s+in kB\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
        text, re.MULTILINE
    )
    m = _last(matches)
    return (float(m[0]), float(m[1]), float(m[2])) if m else None


def parse_forces_first_atom(text: str) -> Optional[Tuple[float, float, float]]:
    """Return (Fx, Fy, Fz) in eV/Å on atom 1 from the last ionic step.

    The TOTAL-FORCE block looks like:
        TOTAL-FORCE (eV/Angst)
        ---...---
          pos_x  pos_y  pos_z    force_x  force_y  force_z   ← atom 1
          ...
    """
    # Find every TOTAL-FORCE header; take the last one
    blocks = list(re.finditer(
        r'TOTAL-FORCE \(eV/Angst\)\s*\n\s*-+\s*\n', text
    ))
    if not blocks:
        return None
    tail = text[blocks[-1].end():]
    m = re.search(
        r'^\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
        tail, re.MULTILINE
    )
    if not m:
        return None
    return (float(m.group(4)), float(m.group(5)), float(m.group(6)))


def parse_eigenvalues_by_band(text: str) -> dict:
    """Return {band_index: mean_energy_relative_to_Fermi} averaged over all k-points.

    Band indices are 1-based (as printed by VASP).
    Returns an empty dict if eigenvalue blocks or Fermi energy are not found.
    """
    efermi = parse_fermi_energy(text)
    if efermi is None:
        return {}

    # Accumulate (sum, count) per band index across all k-point blocks
    sums:   dict = {}
    counts: dict = {}
    for header in re.finditer(r'band No\.\s+band energies\s+occupation\s*\n', text):
        block_start = header.end()
        for line in text[block_start:].splitlines():
            m = re.match(r'^\s+(\d+)\s+([-\d.]+)\s+[-\d.]+\s*$', line)
            if not m:
                break
            idx = int(m.group(1))
            e_rel = float(m.group(2)) - efermi
            sums[idx]   = sums.get(idx, 0.0)   + e_rel
            counts[idx] = counts.get(idx, 0)   + 1

    return {idx: round(sums[idx] / counts[idx], 4) for idx in sums}


def parse_eigenvalues_near_fermi(
    text: str, window: float = 2.0
) -> List[float]:
    """Return eigenvalues (eV, relative to E_fermi) within ±window eV.

    Collects eigenvalues across all k-points and spin channels.
    Returns a flat sorted list of unique values (rounded to 4 dp).
    """
    efermi = parse_fermi_energy(text)
    if efermi is None:
        return []

    # Each k-point eigenvalue block is preceded by:
    #   band No.  band energies     occupation
    # Extract all numbers that appear in these blocks.
    eigs: set = set()
    for header in re.finditer(r'band No\.\s+band energies\s+occupation\s*\n', text):
        block_start = header.end()
        # Read lines until a non-data line
        for line in text[block_start:].splitlines():
            m = re.match(r'^\s+\d+\s+([-\d.]+)\s+[-\d.]+\s*$', line)
            if not m:
                break
            e_rel = float(m.group(1)) - efermi
            if abs(e_rel) <= window:
                eigs.add(round(e_rel, 4))

    return sorted(eigs)
