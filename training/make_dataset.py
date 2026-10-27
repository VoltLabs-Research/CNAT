"""Builds the CNAT datasets from the bulk LAMMPS snapshots (paper Sec. 2.2).

paper     : 1000 random points per main snapshot (12 FCC + 12 liquid), random 80/20 split
heldout   : 1000 random points per snapshot taken 5 ps later (independent thermal noise)
atoms     : 2000 atoms per held-out snapshot, queried at the atom position (the atom itself
            is the first neighbour), which is how the plugin labels atoms
extra     : 4000 random points + 1000 atom positions per snapshot 10 ps later (plugin training)
"""

import argparse
import re
import zlib
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from cnat import K_5NN, PHASES, neighbor_sets, read_lammps_data

PATTERN = re.compile(r"data\.(fcc|liquid)\.(\d+)K(?:\.(heldout|extra))?$")


def sample(path, n_points, n_atoms, rng):
    _, pos, box = read_lammps_data(path)
    lo, length = box[:, 0], box[:, 1] - box[:, 0]
    tree = cKDTree(np.mod(pos - lo, length), boxsize=length)
    parts = []
    if n_points:
        pts = lo + rng.random((n_points, 3)) * length
        parts.append(neighbor_sets(pos, box, pts, K_5NN, tree))
    if n_atoms:
        sel = rng.choice(len(pos), n_atoms, replace=False)
        parts.append(neighbor_sets(pos, box, pos[sel], K_5NN, tree))
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sets = {name: {"atoms": [], "label": [], "temperature": []} for name in ("paper", "heldout", "atoms", "extra")}

    def add(name, x, phase, temp):
        sets[name]["atoms"].append(x)
        sets[name]["label"].append(np.full(len(x), PHASES[phase], np.int64))
        sets[name]["temperature"].append(np.full(len(x), temp, np.float32))

    files = sorted(p for p in args.runs.iterdir() if PATTERN.match(p.name))
    for path in files:
        phase, temp, kind = PATTERN.match(path.name).groups()
        temp = float(temp)
        # One stream per file, so a rerun with more snapshots reproduces the same samples.
        rng = np.random.default_rng([args.seed, zlib.crc32(path.name.encode())])
        if kind is None:
            add("paper", sample(path, 1000, 0, rng), phase, temp)
        elif kind == "heldout":
            add("heldout", sample(path, 1000, 0, rng), phase, temp)
            add("atoms", sample(path, 0, 2000, rng), phase, temp)
        else:
            add("extra", sample(path, 4000, 1000, rng), phase, temp)
        print(f"{path.name}: done", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, s in sets.items():
        if not s["atoms"]:
            continue
        arrays = {k: np.concatenate(v) for k, v in s.items()}
        if name == "paper":
            perm = np.random.default_rng([args.seed, 1]).permutation(len(arrays["label"]))
            n_train = int(0.8 * len(perm))
            for split, idx in (("train", perm[:n_train]), ("test", perm[n_train:])):
                np.savez(args.out / f"{split}.npz", **{k: v[idx] for k, v in arrays.items()})
                print(f"{split}: {len(idx)} samples")
        else:
            np.savez(args.out / f"{name}.npz", **arrays)
            print(f"{name}: {len(arrays['label'])} samples")


if __name__ == "__main__":
    main()
