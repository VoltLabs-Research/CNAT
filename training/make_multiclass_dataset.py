"""Multi-structure CNAT datasets (LIQUID, FCC, HCP, BCC) from bulk snapshots of several metals.

Snapshots are named data.<metal>.<structure>.<T>K[.heldout|.extra] (in.metal), and the Ni runs of the
paper reproduction data.<fcc|liquid>.<T>K[...] are read as metal "ni". A snapshot is used only when PTM
(ptm_check.json, from ptm_check.py) agrees with its label: at least --min_crystal of its atoms in the
labelled crystal structure at rmsd 0.2, or for a liquid at most --max_liquid_crystal of them crystalline
at rmsd 0.12 (at 0.2, PTM matches a few percent of liquid atoms to some template).

train   : 1000 random points + 500 atom positions per main snapshot, plus 3000 + 1000 per extra snapshot
heldout : 1000 random points per held-out snapshot;  atoms: 1000 atom positions per held-out snapshot
"""

import argparse
import json
import re
import zlib
from pathlib import Path

import numpy as np

from cnat import K_5NN
from make_dataset import sample

CLASSES = ["LIQUID", "FCC", "HCP", "BCC"]
METAL = re.compile(r"data\.([a-z]+)\.(fcc|hcp|bcc|liquid)\.(\d+)K(?:\.(heldout|extra))?$")
NICKEL = re.compile(r"data\.(fcc|liquid)\.(\d+)K(?:\.(heldout|extra))?$")


def parse(name):
    if m := METAL.match(name):
        return m.group(1), m.group(2), float(m.group(3)), m.group(4)
    if m := NICKEL.match(name):
        return "ni", m.group(1), float(m.group(2)), m.group(3)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--ptm", type=Path, required=True)
    ap.add_argument("--min_crystal", type=float, default=0.6)
    ap.add_argument("--max_liquid_crystal", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude_train", nargs="*", default=[], help="metals left out of the training set only")
    args = ap.parse_args()
    ptm = json.loads(args.ptm.read_text())

    sets = {name: {"atoms": [], "label": [], "temperature": [], "metal": []} for name in ("train", "heldout", "atoms")}
    rejected = []

    def add(name, x, label, temp, metal):
        sets[name]["atoms"].append(x)
        sets[name]["label"].append(np.full(len(x), label, np.int64))
        sets[name]["temperature"].append(np.full(len(x), temp, np.float32))
        sets[name]["metal"].append(np.full(len(x), metal))

    for d in args.dirs:
        for path in sorted(d.iterdir()):
            parsed = parse(path.name)
            if parsed is None:
                continue
            metal, structure, temp, kind = parsed
            label = CLASSES.index(structure.upper())
            frac = ptm.get(path.name)
            if frac is None:
                rejected.append((path.name, "no PTM check"))
                continue
            if label == 0:
                crystalline = 1.0 - frac["strict"]["OTHER"] - frac["strict"]["ICO"]
                if crystalline > args.max_liquid_crystal:
                    rejected.append((path.name, f"liquid but {crystalline:.3f} crystalline"))
                    continue
            elif frac["loose"][structure.upper()] < args.min_crystal:
                rejected.append((path.name, f"{structure} fraction {frac['loose'][structure.upper()]:.3f}"))
                continue
            rng = np.random.default_rng([args.seed, zlib.crc32(path.name.encode())])
            if kind is None:
                if metal not in args.exclude_train:
                    add("train", sample(path, 1000, 500, rng), label, temp, metal)
            elif kind == "extra":
                if metal not in args.exclude_train:
                    add("train", sample(path, 3000, 1000, rng), label, temp, metal)
            else:
                add("heldout", sample(path, 1000, 0, rng), label, temp, metal)
                add("atoms", sample(path, 0, 1000, rng), label, temp, metal)
            print(f"{path.name}: {CLASSES[label]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, s in sets.items():
        arrays = {k: np.concatenate(v) for k, v in s.items()}
        assert arrays["atoms"].shape[1] == K_5NN
        np.savez(args.out / f"{name}.npz", **arrays)
        counts = {CLASSES[c]: int((arrays["label"] == c).sum()) for c in range(len(CLASSES))}
        print(f"{name}: {len(arrays['label'])} samples {counts}")
    for name, why in rejected:
        print(f"rejected {name}: {why}")


if __name__ == "__main__":
    main()
