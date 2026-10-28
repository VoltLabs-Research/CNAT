"""The crystal of each crystal class of a CNAT model, which the plugin aligns that class's atoms onto to
write OpenDXA's structure contract. A class takes a lattice topology of the registry (fcc, hcp, bcc,
cubic_diamond, ...) or the reference cell its training crystal was built from (a LAMMPS data file).

    --crystal FCC=fcc --crystal OLIVINE=lammps/Fo.lmp
"""

from pathlib import Path

import numpy as np


def read_lammps_data(path):
    """Cell (rows a, b, c), types and Cartesian positions (origin at the cell's lower corner) of a LAMMPS
    data file, orthogonal or triclinic, 'atomic' or 'charge' style."""
    lines = Path(path).read_text().splitlines()
    lo, hi, tilt, natoms, start, charge = [0.0] * 3, [0.0] * 3, [0.0] * 3, 0, None, False
    for i, line in enumerate(lines):
        tok = line.split()
        if len(tok) >= 2 and tok[1] == "atoms":
            natoms = int(tok[0])
        for d, axis in enumerate("xyz"):
            if len(tok) >= 4 and tok[2] == axis + "lo":
                lo[d], hi[d] = float(tok[0]), float(tok[1])
        if len(tok) >= 6 and tok[3:6] == ["xy", "xz", "yz"]:
            tilt = [float(t) for t in tok[:3]]
        if tok and tok[0] == "Atoms":
            start, charge = i + 1, "charge" in line
            break
    if start is None or natoms == 0:
        raise SystemExit(f"{path}: no atoms or no Atoms section")
    types, positions = [], []
    for line in lines[start:]:
        values = line.split("#")[0].split()
        if len(values) < 5:
            if positions and len(positions) == natoms:
                break
            continue
        x = 3 if charge or len(values) in (6, 9) else 2
        types.append(int(values[1]))
        positions.append([float(v) for v in values[x:x + 3]])
        if len(positions) == natoms:
            break
    cell = [[hi[0] - lo[0], 0.0, 0.0], [tilt[0], hi[1] - lo[1], 0.0], [tilt[1], tilt[2], hi[2] - lo[2]]]
    return cell, types, (np.array(positions) - np.array(lo)).tolist()


def crystal_entry(spec):
    """CLASS=SPEC -> (CLASS, entry): a registry topology name, or a LAMMPS data file whose cell and atoms
    are stored."""
    name, sep, value = spec.partition("=")
    if not sep or not name or not value:
        raise SystemExit(f"--crystal takes CLASS=topology or CLASS=cell.lmp, got '{spec}'")
    if not Path(value).is_file():
        return name, {"topology": value.lower()}
    cell, types, positions = read_lammps_data(value)
    return name, {"cell": cell, "atoms": [[t, *p] for t, p in zip(types, positions)], "source": Path(value).name}
