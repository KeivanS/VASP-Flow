#!/bin/bash
# Generate ELF-along-bond plots for every mp-* project directory.
# Output is named after the reduced formula, e.g. MoSe2_elf_bonds.pdf
for x in mp-*; do
    [ -f "$x/02_scf/ELFCAR" ] || continue
    ( cd "$x/02_scf/" && python3 ../../elf_bonds.py )   # prefix = formula
    pdf=$(ls "$x"/02_scf/*_elf_bonds.pdf 2>/dev/null | head -1)
    [ -n "$pdf" ] && open "$pdf"    # skips SOC/non-collinear (all-zero ELFCAR)
done
