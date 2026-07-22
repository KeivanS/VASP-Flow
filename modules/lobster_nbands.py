#!/usr/bin/env python3
"""
Compute (and optionally write) an NBANDS value large enough for a subsequent
LOBSTER analysis, from a VASP run's POSCAR + POTCAR.

LOBSTER's hard requirement is NBANDS >= the number of *local basis functions*
of the projection, i.e. the sum over all atoms of the orbital multiplicities
(s=1, p=3, d=5, f=7) of that element's pbeVaspFit2015 basis.  This is usually
LARGER than the number of occupied bands (NELECT/2), which is why simply adding
a fixed buffer to NELECT/2 is unreliable -- we compute the basis count exactly.

We set
    NBANDS = max(n_basis_functions, ceil(NELECT/2) + extra)
rounded up to a multiple of NCORE (so VASP does not silently change it).

Usage:
    lobster_nbands.py POSCAR POTCAR [--ncore 4] [--extra 0] [--write INCAR]

With --write, the NBANDS tag in the given INCAR is added or replaced in place.
"""

import argparse
import math
import re
import sys

ORB_MULT = {"s": 1, "p": 3, "d": 5, "f": 7}


def compute(poscar, potcar):
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Potcar
    from pymatgen.io.lobster import Lobsterin

    st = Structure.from_file(poscar)
    pot = Potcar.from_file(potcar)
    symbols = [p.symbol for p in pot]
    zval = {p.symbol.split("_")[0]: float(p.zval) for p in pot}

    # total valence electrons
    counts = {}
    for site in st:
        el = site.specie.symbol
        counts[el] = counts.get(el, 0) + 1
    nelect = sum(zval[el] * n for el, n in counts.items())

    # number of LOBSTER basis functions = sum of orbital multiplicities
    basis = Lobsterin.get_basis(st, potcar_symbols=symbols)   # e.g. "Ge 4p 4s"
    per_el = {}
    for entry in basis:
        toks = entry.split()
        el = toks[0].split("_")[0]
        per_el[el] = sum(ORB_MULT[o[-1]] for o in toks[1:])
    n_basis = sum(per_el[el] * counts[el] for el in counts)

    occupied = math.ceil(nelect / 2)
    return {"nelect": nelect, "occupied": occupied,
            "n_basis": n_basis, "basis": basis}


def recommended_nbands(info, ncore=4, extra=0):
    nb = max(info["n_basis"], info["occupied"] + extra)
    if ncore > 1:
        nb = int(math.ceil(nb / ncore) * ncore)
    return nb


def write_incar(incar_path, nbands):
    with open(incar_path) as fh:
        text = fh.read()
    line = f"NBANDS = {nbands}"
    if re.search(r"(?im)^\s*NBANDS\s*=", text):
        text = re.sub(r"(?im)^\s*NBANDS\s*=.*$", line, text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += f"\n# NBANDS set for LOBSTER (>= number of local basis functions)\n{line}\n"
    with open(incar_path, "w") as fh:
        fh.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("poscar")
    ap.add_argument("potcar")
    ap.add_argument("--ncore", type=int, default=4,
                    help="round NBANDS up to a multiple of this (default 4)")
    ap.add_argument("--extra", type=int, default=0,
                    help="extra empty bands above occupied (default 0)")
    ap.add_argument("--write", metavar="INCAR",
                    help="add/replace NBANDS in this INCAR")
    args = ap.parse_args()

    info = compute(args.poscar, args.potcar)
    nb = recommended_nbands(info, ncore=args.ncore, extra=args.extra)

    print(f"NELECT          = {info['nelect']:.0f}")
    print(f"occupied bands  = {info['occupied']}  (NELECT/2)")
    print(f"basis functions = {info['n_basis']}   {info['basis']}")
    print(f"recommended NBANDS = {nb}")

    if args.write:
        write_incar(args.write, nb)
        print(f"wrote NBANDS = {nb} into {args.write}")
    else:
        # also emit just the number on the last line for easy shell capture
        print(nb)


if __name__ == "__main__":
    main()
