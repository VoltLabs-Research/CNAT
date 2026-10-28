"""Al2SiO5 datasets: DISORDERED (liquid, glass), KYANITE, ANDALUSITE and SILLIMANITE, with the neighbours
taken from one sublattice (--neighbor_types: 1 2 for the Al+Si cations, 3 for the oxygens).

The oxygen framework of kyanite is cubic close packed, as is that of the non-kyanite phase in the user's
Kyanite_010 sample, so an oxygen model cannot tell them apart; the cation positions can. Boxes may be
triclinic (kyanite is P-1), so neighbours come from the 27 periodic images of the cell. Query points are
random positions and atom positions of every species, because the plugin labels every atom from the
sublattice around it. A crystal snapshot is used when at least --min_on_site of its atoms sit within 1 A of
their lattice site (the unit cell replicated like LAMMPS does), a disordered one when at most
--max_on_site do.

train   : 1000 random + 1000 atom points per main snapshot, 3000 + 2000 per extra one
heldout : 1000 random points per held-out snapshot;  atoms: 1000 atom points
"""

import argparse
import re
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from cnat import K_5NN

CLASSES = ["DISORDERED", "KYANITE", "ANDALUSITE", "SILLIMANITE"]
NAME = re.compile(r"data\.al2sio5\.(kyanite|andalusite|sillimanite|liquid|glass)\.(\d+)K(?:\.(heldout|extra))?$")


def read_cell(path):
    """LAMMPS 'atomic' or 'charge' data file, orthogonal or triclinic, sorted by id.
    Returns (ids, types, positions, H with the cell vectors as columns, origin)."""
    lines = Path(path).read_text().splitlines()
    lo, hi, tilt, natoms = np.zeros(3), np.zeros(3), np.zeros(3), 0
    for i, line in enumerate(lines):
        tok = line.split()
        if len(tok) >= 2 and tok[1] == "atoms":
            natoms = int(tok[0])
        for d, name in enumerate(("xlo", "ylo", "zlo")):
            if len(tok) >= 4 and tok[2] == name:
                lo[d], hi[d] = float(tok[0]), float(tok[1])
        if len(tok) >= 6 and tok[3:6] == ["xy", "xz", "yz"]:
            tilt = np.array([float(t) for t in tok[:3]])
        if line.startswith("Atoms"):
            start, first = i + 2, (3 if "charge" in line else 2)
            break
    rows = np.array([l.split()[:first + 3] for l in lines[start:start + natoms]], dtype=np.float64)
    rows = rows[np.argsort(rows[:, 0])]
    length = hi - lo
    H = np.array([[length[0], tilt[0], tilt[1]], [0.0, length[1], tilt[2]], [0.0, 0.0, length[2]]])
    return rows[:, 0].astype(np.int64), rows[:, 1].astype(np.int64), rows[:, first:first + 3].copy(), H, lo


def reduced(pos, H, origin):
    return np.linalg.solve(H, (pos - origin).T).T


class PeriodicNeighbors:
    """k nearest neighbours under periodic boundaries of a possibly triclinic cell."""

    def __init__(self, pos, H, origin):
        self.H, self.origin = H, origin
        frac = np.mod(reduced(pos, H, origin), 1.0)
        shifts = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.images = np.concatenate([(frac + s) @ H.T for s in shifts])
        self.tree = cKDTree(self.images)

    def query(self, points, k):
        q = np.mod(reduced(points, self.H, self.origin), 1.0) @ self.H.T
        _, idx = self.tree.query(q, k=k, workers=-1)
        return (self.images[idx] - q[:, None, :]).astype(np.float32)


def reference_sites(cell_file, reps):
    _, _, pos, H, origin = read_cell(cell_file)
    frac = reduced(pos, H, origin)
    cells = np.stack(np.meshgrid(*[np.arange(r) for r in reps], indexing="ij"), -1).reshape(-1, 3)
    order = np.lexsort((cells[:, 0], cells[:, 1], cells[:, 2]))  # LAMMPS replicate: x fastest
    return np.concatenate([frac + c for c in cells[order]]) / np.array(reps)


def on_site_fraction(path, ref):
    _, _, pos, H, origin = read_cell(path)
    d = reduced(pos, H, origin) - ref
    d -= np.round(d)
    return float((np.linalg.norm(d @ H.T, axis=1) < 1.0).mean())


def sample(path, neighbor_types, n_random, n_atoms, rng):
    _, types, pos, H, origin = read_cell(path)
    cloud = pos[np.isin(types, neighbor_types)]
    nb = PeriodicNeighbors(cloud, H, origin)
    parts = []
    if n_random:
        parts.append(nb.query(origin + rng.random((n_random, 3)) @ H.T, K_5NN))
    if n_atoms:
        parts.append(nb.query(pos[rng.choice(len(pos), n_atoms, replace=False)], K_5NN))
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cells", type=Path, required=True, help="directory with kyanite.lmp, andalusite.lmp, sillimanite.lmp")
    ap.add_argument("--neighbor_types", type=int, nargs="+", required=True)
    ap.add_argument("--reps", type=int, nargs=3, default=[4, 4, 5])
    ap.add_argument("--min_on_site", type=float, default=0.95)
    ap.add_argument("--max_on_site", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    refs = {p: reference_sites(args.cells / f"{p}.lmp", args.reps) for p in ("kyanite", "andalusite", "sillimanite")}

    sets = {name: {"atoms": [], "label": [], "temperature": [], "kind": []} for name in ("train", "heldout", "atoms")}

    def add(name, x, label, temp, kind):
        sets[name]["atoms"].append(x)
        sets[name]["label"].append(np.full(len(x), label, np.int64))
        sets[name]["temperature"].append(np.full(len(x), temp, np.float32))
        sets[name]["kind"].append(np.full(len(x), kind))

    for path in sorted(args.runs.iterdir()):
        m = NAME.match(path.name)
        if not m:
            continue
        phase, temp, part = m.group(1), float(m.group(2)), m.group(3)
        label = CLASSES.index(phase.upper()) if phase in refs else 0
        # Disordered runs start from the kyanite cell, so their distance from it measures the melting.
        frac = on_site_fraction(path, refs.get(phase, refs["kyanite"]))
        ok = frac >= args.min_on_site if label else frac <= args.max_on_site
        print(f"{path.name}: {frac:.3f} of atoms on their lattice site -> {'used' if ok else 'REJECTED'}", flush=True)
        if not ok:
            continue
        rng = np.random.default_rng([args.seed, zlib.crc32(path.name.encode())])
        if part is None:
            add("train", sample(path, args.neighbor_types, 1000, 1000, rng), label, temp, phase)
        elif part == "extra":
            add("train", sample(path, args.neighbor_types, 3000, 2000, rng), label, temp, phase)
        else:
            add("heldout", sample(path, args.neighbor_types, 1000, 0, rng), label, temp, phase)
            add("atoms", sample(path, args.neighbor_types, 0, 1000, rng), label, temp, phase)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, s in sets.items():
        arrays = {k: np.concatenate(v) for k, v in s.items()}
        np.savez(args.out / f"{name}.npz", **arrays)
        print(f"{name}: {len(arrays['label'])} samples", {c: int((arrays["label"] == i).sum()) for i, c in enumerate(CLASSES)})


if __name__ == "__main__":
    main()
