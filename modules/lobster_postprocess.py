#!/usr/bin/env python3
"""
Post-process LOBSTER COHPCAR / COBICAR / COOPCAR files for a list of
Materials Project IDs.

For every material in `highthrouput_list` this reads, from <id>/02_scf/ :
    COHPCAR.lobster , COBICAR.lobster , COOPCAR.lobster
and produces one CSV row per *inequivalent bond* -- where "inequivalent bond"
means a unique (element-pair, bond-length) combination, e.g. "Ge-Se @ 2.55 A".

LOBSTER lists every atom-atom contact twice (A->B and B->A); contacts in the
same group are summed and divided by 2 so each physical contact is counted once.

Per bond group and per measure (COHP / COBI / COOP) we report:
  * I<measure>   : integrated value at E_Fermi (total integral up to E_F)
  * B<measure>   : bonding contribution up to E_F      (algebraic / signed)
  * A<measure>   : antibonding contribution up to E_F  (algebraic / signed)
  * fAB_<measure>: antibonding fraction  |A| / (|B| + |A|)   (0..1)
  * Esign_<measure> : all bonding<->antibonding sign-change energies below E_F
                      (semicolon-separated; energies are Fermi-referenced)

B and A are reported with their natural algebraic signs (NOT absolute values);
B + A equals I up to the small difference between a trapezoid of the curve and
LOBSTER's own running integral.

Sign conventions as stored in the *CAR.lobster files:
  * COHP : negative = bonding,  positive = antibonding  -> antibonding A = ( > 0 )
  * COBI : positive = bonding,  negative = antibonding  -> antibonding A = ( < 0 )
  * COOP : positive = bonding,  negative = antibonding  -> antibonding A = ( < 0 )

Output: lobster_acohp_acobi.csv   (a test table for now)
"""

import os
import re
import sys
import csv
import numpy as np

LIST_FILE = "highthrouput_list"
SCF_SUBDIR = "02_scf"
OUT_CSV = "lobster_acohp_acobi.csv"
DIST_DECIMALS = 2          # rounding used to group bonds by length

# Sign-change ("bonding -> antibonding") noise filter.  A zero crossing is kept
# only if the curve reaches at least SIGN_REL_THRESH of its peak magnitude
# within +-SIGN_WIN_EV of the crossing; this discards tiny numerical wiggles in
# deep, near-zero-weight energy regions.  Set SIGN_REL_THRESH=0 to keep all.
SIGN_REL_THRESH = 0.05
SIGN_WIN_EV = 0.3

# measure -> (filename, antibonding sign)  +1 means antibonding is positive
MEASURES = {
    "COHP": ("COHPCAR.lobster", +1),
    "COBI": ("COBICAR.lobster", -1),
    "COOP": ("COOPCAR.lobster", -1),
}

LABEL_RE = re.compile(r"No\.\d+:([A-Za-z]+)(\d+)->([A-Za-z]+)(\d+)\(([0-9.]+)\)")


def read_lobster_car(path):
    """Parse a *CAR.lobster file.

    Returns (E, bonds) where E is the Fermi-referenced energy grid (E_F = 0) and
    `bonds` is a list of dicts: {pair, dist, y, iy} with per-bond curve `y`
    and integrated curve `iy` (summed over spin channels).
    """
    with open(path) as fh:
        lines = fh.readlines()

    ctrl = lines[1].split()
    ncol = int(ctrl[0])          # 1 average + Nbonds
    nspin = int(ctrl[1])
    nbonds = ncol - 1

    labels = [lines[2 + k].strip() for k in range(ncol)]   # [0]="Average", ...
    data = np.array([ln.split() for ln in lines[2 + ncol:] if ln.strip()],
                    dtype=float)
    E = data[:, 0]

    bonds = []
    for k in range(1, nbonds + 1):           # skip k=0 (the Average)
        m = LABEL_RE.match(labels[k])
        if not m:
            continue
        a, b, dist = m.group(1), m.group(3), float(m.group(5))
        pair = "-".join(sorted((a, b)))
        # A bond between two DIFFERENT atoms is listed once (A->B); a
        # same-atom pair (e.g. La1->La1 + translation) is listed twice (±T).
        same = m.group(1, 2) == m.group(3, 4)
        y = np.zeros(len(E))
        iy = np.zeros(len(E))
        for s in range(nspin):
            base = 1 + s * 2 * ncol
            y += data[:, base + 2 * k]
            iy += data[:, base + 2 * k + 1]
        bonds.append({"pair": pair, "dist": dist, "y": y, "iy": iy,
                      "same": same})
    return E, bonds


def bonding_antibonding(E, y, anti_sign):
    """Algebraic (signed) bonding/antibonding integrals of `y` up to E_Fermi.

    Returns (B, AB, frac):
      B    = signed integral of the bonding part      (E <= 0)
      AB   = signed integral of the antibonding part  (E <= 0)
      frac = antibonding fraction  |AB| / (|B| + |AB|)

    anti_sign = +1 (COHP): antibonding = y > 0, bonding = y < 0.
    anti_sign = -1 (COBI/COOP): antibonding = y < 0, bonding = y > 0.
    Note B + AB equals the full integrated value (ICOxP) at E_F.
    """
    mask = E <= 0.0
    e, v = E[mask], y[mask]
    Ipos = float(np.trapz(np.clip(v, 0.0, None), e))   # integral of positive part
    Ineg = float(np.trapz(np.clip(v, None, 0.0), e))   # integral of negative part
    if anti_sign > 0:                 # COHP: antibonding positive, bonding negative
        AB, B = Ipos, Ineg
    else:                             # COBI/COOP: antibonding negative, bonding positive
        AB, B = Ineg, Ipos
    denom = abs(B) + abs(AB)
    frac = abs(AB) / denom if denom > 0 else 0.0
    return B, AB, frac


def sign_changes_below_fermi(E, y):
    """Zero-crossing energies of `y` at/below E_Fermi (E <= 0), noise-filtered.

    A crossing is discarded unless |y| reaches SIGN_REL_THRESH of its peak
    within +-SIGN_WIN_EV of the crossing (see module constants).
    """
    peak = float(np.max(np.abs(y)))
    if peak == 0.0:
        return []
    s = np.sign(y)
    out = []
    for i in np.where(np.diff(s) != 0)[0]:
        denom = y[i + 1] - y[i]
        if denom == 0:
            continue
        e0 = E[i] - y[i] * (E[i + 1] - E[i]) / denom
        if e0 > 0.0:
            continue
        win = (E >= e0 - SIGN_WIN_EV) & (E <= e0 + SIGN_WIN_EV)
        if win.any() and float(np.max(np.abs(y[win]))) >= SIGN_REL_THRESH * peak:
            out.append(e0)
    return sorted(out)


def integral_at_fermi(E, iy):
    """Integrated value (ICOxP) at E_Fermi = 0."""
    return float(np.interp(0.0, E, iy))


def measure_groups(path, anti_sign):
    """Return {(pair, dist_rounded): {I, B, A, fAB, Esign, n}} for one *CAR file.

    I   = total integrated value at E_F (ICOxP)
    B   = bonding contribution (signed/algebraic)
    A   = antibonding contribution (signed/algebraic)
    fAB = antibonding fraction |A| / (|B| + |A|)
    """
    E, bonds = read_lobster_car(path)
    acc = {}
    for bd in bonds:
        key = (bd["pair"], round(bd["dist"], DIST_DECIMALS))
        g = acc.setdefault(key, {"y": np.zeros(len(E)), "iy": np.zeros(len(E)),
                                 "n": 0, "mult": 0.0})
        g["y"] += bd["y"]
        g["iy"] += bd["iy"]
        g["n"] += 1
        # unique bonds per cell: same-atom pairs are listed ±T (count 1/2),
        # different-atom pairs once (count 1)
        g["mult"] += 0.5 if bd["same"] else 1.0
    out = {}
    for key, g in acc.items():
        # PER BOND: mean over the equivalent contacts. The curve of one bond
        # is direction-independent, so the mean is exact whether the group
        # was listed once or in both ±T directions — this is directly
        # comparable to the per-bond entries of ICO*LIST.lobster.
        y = g["y"] / g["n"]
        iy = g["iy"] / g["n"]
        B, A, fAB = bonding_antibonding(E, y, anti_sign)
        out[key] = {
            "I": integral_at_fermi(E, iy),
            "B": B,
            "A": A,
            "fAB": fAB,
            "Esign": ";".join(f"{e:.4f}" for e in sign_changes_below_fermi(E, y)),
            "n": int(round(g["mult"])),
        }
    return out


# Per descriptor: I=total integrated, B=bonding, A=antibonding (all algebraic),
# fAB=antibonding fraction |A|/(|B|+|A|), Esign=sign-change energies.
FIELDS = ["Material_ID", "Bond", "Distance_Ang", "N_contacts",
          "ICOHP", "BCOHP", "ACOHP", "fAB_COHP", "Esign_COHP",
          "ICOBI", "BCOBI", "ACOBI", "fAB_COBI", "Esign_COBI",
          "ICOOP", "BCOOP", "ACOOP", "fAB_COOP", "Esign_COOP"]


def rows_for_dir(lob_dir, material_id, suffix=""):
    """Build per-inequivalent-bond rows for one LOBSTER output directory."""
    per_measure = {}
    for name, (fname, anti_sign) in MEASURES.items():
        path = os.path.join(lob_dir, fname + suffix)
        if os.path.exists(path):
            per_measure[name] = measure_groups(path, anti_sign)
        else:
            per_measure[name] = {}
            print(f"[warn] {path} missing", file=sys.stderr)
    keys = sorted(set().union(*[m.keys() for m in per_measure.values()]),
                  key=lambda k: (k[0], k[1]))
    rows = []
    for pair, dist in keys:
        row = {"Material_ID": material_id, "Bond": pair, "Distance_Ang": dist,
               "N_contacts": ""}
        for name in MEASURES:
            g = per_measure[name].get((pair, dist))
            if g:
                row[f"I{name}"]     = f"{g['I']:.6f}"
                row[f"B{name}"]     = f"{g['B']:.6f}"
                row[f"A{name}"]     = f"{g['A']:.6f}"
                row[f"fAB_{name}"]  = f"{g['fAB']:.4f}"
                row[f"Esign_{name}"] = g["Esign"]
                row["N_contacts"]   = g["n"]
            else:
                row[f"I{name}"] = row[f"B{name}"] = row[f"A{name}"] = ""
                row[f"fAB_{name}"] = row[f"Esign_{name}"] = ""
        rows.append(row)
        print(f"{material_id}  {pair} @ {dist} A  "
              f"COHP[I={row['ICOHP']} B={row['BCOHP']} AB={row['ACOHP']} fAB={row['fAB_COHP']}]  "
              f"COBI[fAB={row['fAB_COBI']}]")
    return rows


def write_csv(rows, out_csv):
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symmetric", action="store_true",
                    help="process the *.lobster.symmetric backup files (the "
                         "ISYM-on run kept by run_scf_for_lobster.sh) and write "
                         "to a separate CSV, for comparison with the ISYM=0 run.")
    ap.add_argument("--dir", metavar="LOBSTER_DIR",
                    help="process a single LOBSTER output directory (e.g. a "
                         "project's 08_lobster/) instead of the highthrouput_list.")
    ap.add_argument("--out", metavar="CSV", help="output CSV path (single-dir mode).")
    args = ap.parse_args()
    suffix = ".symmetric" if args.symmetric else ""

    # ── single-directory mode (per-project analyze.sh) ──────────────────────
    if args.dir:
        lob_dir = os.path.abspath(args.dir)
        # material id = the project folder name (parent of 08_lobster)
        material_id = os.path.basename(os.path.dirname(lob_dir)) or os.path.basename(lob_dir)
        rows = rows_for_dir(lob_dir, material_id, suffix)
        out_csv = args.out or os.path.join(lob_dir, "lobster_summary.csv")
        write_csv(rows, out_csv)
        print(f"\nWrote {out_csv} ({len(rows)} bond rows)")
        return

    # ── high-throughput list mode ───────────────────────────────────────────
    out_csv = (OUT_CSV[:-4] + "_symmetric.csv") if args.symmetric else OUT_CSV
    if not os.path.exists(LIST_FILE):
        sys.exit(f"List file not found: {LIST_FILE}")
    with open(LIST_FILE) as fh:
        mats = [ln.strip() for ln in fh if ln.strip()]

    rows = []
    for mat in mats:
        # LOBSTER output lives in 08_lobster/ (current workflow); fall back to
        # 02_scf/ for materials prepared the old way. For --symmetric backups,
        # only 02_scf/ is relevant.
        if suffix:
            lob = os.path.join(mat, "02_scf")
        elif os.path.exists(os.path.join(mat, "08_lobster", "COHPCAR.lobster")):
            lob = os.path.join(mat, "08_lobster")
        else:
            lob = os.path.join(mat, SCF_SUBDIR)
        mrows = rows_for_dir(lob, mat, suffix)
        if not mrows:
            print(f"[warn] {mat}: no LOBSTER bond data found", file=sys.stderr)
        rows.extend(mrows)

    write_csv(rows, out_csv)
    label = " (symmetric backup)" if args.symmetric else ""
    print(f"\nWrote {out_csv} ({len(rows)} bond rows){label}")


if __name__ == "__main__":
    main()
