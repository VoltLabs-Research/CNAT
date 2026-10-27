"""Forsterite datasets for an O-sublattice CNAT model: DISORDERED (liquid and glass) vs OLIVINE.

The neighbours are oxygens only (type 3). Query points are random positions, oxygen positions (the
oxygen itself is the first neighbour) and Mg/Si positions, because the plugin labels every atom from
the oxygens around it. A crystal snapshot is used only when at least --min_on_site of its atoms sit
within 1 A of their lattice site (Fo.lmp replicated, rescaled to the snapshot's box); a disordered one
only when at most --max_on_site do.

train   : 1000 random + 500 O + 500 cation points per main snapshot, 3000 + 1000 + 1000 per extra one
heldout : 1000 random points per held-out snapshot;  atoms: 500 O + 500 cation points
"""

import argparse
import re
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from cnat import K_5NN, neighbor_sets, read_lammps_data

CLASSES = ["DISORDERED", "OLIVINE"]
NAME = re.compile(r"data\.fo\.(crystal|liquid|glass)\.(\d+)K(?:\.(heldout|extra))?$")
OXYGEN = 3


def reference_sites(fo_lmp, reps):
    _, pos, box, types = read_lammps_data(fo_lmp, with_types=True)
    length = box[:, 1] - box[:, 0]
    cells = np.stack(np.meshgrid(*[np.arange(r) for r in reps], indexing="ij"), -1).reshape(-1, 3)
    # LAMMPS replicate: image (i, j, k) shifts by the cell, ids grow per image in i-fastest order.
    order = np.lexsort((cells[:, 0], cells[:, 1], cells[:, 2]))
    reduced = [(pos - box[:, 0]) / length + c for c in cells[order]]
    return np.concatenate(reduced) / np.array(reps), np.tile(types, len(cells))


def on_site_fraction(path, ref_reduced):
    _, pos, box = read_lammps_data(path)
    length = box[:, 1] - box[:, 0]
    d = (pos - box[:, 0]) / length - ref_reduced
    d -= np.round(d)
    return float((np.linalg.norm(d * length, axis=1) < 1.0).mean())


def sample(path, n_random, n_oxygen, n_cation, rng):
    _, pos, box, types = read_lammps_data(path, with_types=True)
    lo, length = box[:, 0], box[:, 1] - box[:, 0]
    oxy = pos[types == OXYGEN]
    tree = cKDTree(np.mod(oxy - lo, length), boxsize=length)
    parts = []
    if n_random:
        parts.append(neighbor_sets(oxy, box, lo + rng.random((n_random, 3)) * length, K_5NN, tree))
    if n_oxygen:
        parts.append(neighbor_sets(oxy, box, oxy[rng.choice(len(oxy), n_oxygen, replace=False)], K_5NN, tree))
    if n_cation:
        cations = pos[types != OXYGEN]
        parts.append(neighbor_sets(oxy, box, cations[rng.choice(len(cations), n_cation, replace=False)], K_5NN, tree))
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fo", type=Path, default=Path("lammps/Fo.lmp"))
    ap.add_argument("--reps", type=int, nargs=3, default=[6, 3, 5])
    ap.add_argument("--min_on_site", type=float, default=0.95)
    ap.add_argument("--max_on_site", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ref, _ = reference_sites(args.fo, args.reps)

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
        label = int(phase == "crystal")
        frac = on_site_fraction(path, ref)
        ok = frac >= args.min_on_site if label else frac <= args.max_on_site
        print(f"{path.name}: {frac:.3f} of atoms on their lattice site -> {'used' if ok else 'REJECTED'}", flush=True)
        if not ok:
            continue
        rng = np.random.default_rng([args.seed, zlib.crc32(path.name.encode())])
        if part is None:
            add("train", sample(path, 1000, 500, 500, rng), label, temp, phase)
        elif part == "extra":
            add("train", sample(path, 3000, 1000, 1000, rng), label, temp, phase)
        else:
            add("heldout", sample(path, 1000, 0, 0, rng), label, temp, phase)
            add("atoms", sample(path, 0, 500, 500, rng), label, temp, phase)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, s in sets.items():
        arrays = {k: np.concatenate(v) for k, v in s.items()}
        np.savez(args.out / f"{name}.npz", **arrays)
        print(f"{name}: {len(arrays['label'])} samples", {CLASSES[c]: int((arrays["label"] == c).sum()) for c in (0, 1)})


if __name__ == "__main__":
    main()
