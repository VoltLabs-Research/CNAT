"""Converts an Al2SiO5 CIF to a LAMMPS 'charge' data file with types 1 Al, 2 Si, 3 O (Pedone charges),
and reports composition and Al coordination (O within 2.2 A) as a sanity check."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list

CHARGES = {"Al": 1.8, "Si": 2.4, "O": -1.2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cif", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    atoms = read(args.cif)
    counts = Counter(atoms.get_chemical_symbols())
    i, j = neighbor_list("ij", atoms, 2.2)
    sym = np.array(atoms.get_chemical_symbols())
    al = np.nonzero(sym == "Al")[0]
    cn = Counter(int(((i == a) & (sym[j] == "O")).sum()) for a in al)
    print(f"{args.cif.name}: {dict(counts)}, cell {atoms.cell.cellpar().round(3).tolist()}, Al-O coordination {dict(sorted(cn.items()))}")
    atoms.set_initial_charges([CHARGES[s] for s in atoms.get_chemical_symbols()])
    write(args.out, atoms, format="lammps-data", atom_style="charge", specorder=["Al", "Si", "O"], masses=True)


if __name__ == "__main__":
    main()
