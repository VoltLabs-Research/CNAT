"""PTM structure fractions, strict (rmsd 0.12) and loose (0.2), of every snapshot in the given directories -> JSON."""

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

NAMES = ["OTHER", "FCC", "HCP", "BCC", "ICO", "SC", "DCUB", "DHEX", "GRAPHENE"]


def check(lmp, script, path):
    with tempfile.NamedTemporaryFile(suffix=".dump") as out:
        subprocess.run([lmp, "-in", script, "-var", "dfile", str(path), "-var", "out", out.name,
                        "-log", "none", "-screen", "none"], check=True, env=os.environ | {"OMP_NUM_THREADS": "1"})
        types = np.loadtxt(out.name, skiprows=9)[:, 1:3].astype(int)
    fractions = {}
    for col, key in ((0, "strict"), (1, "loose")):
        counts = np.bincount(types[:, col], minlength=len(NAMES)) / len(types)
        fractions[key] = {n: float(c) for n, c in zip(NAMES, counts)}
    return path.name, fractions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--lmp", default="lmp")
    ap.add_argument("--script", default="lammps/in.ptmcheck")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    paths = [p for d in args.dirs for p in sorted(d.glob("data.*")) if p.name not in result and "interface" not in p.name]
    with ThreadPoolExecutor(args.jobs) as pool:
        for name, frac in pool.map(lambda p: check(args.lmp, args.script, p), paths):
            result[name] = frac
            print(name, " | ".join(
                key + " " + " ".join(f"{k} {v:.3f}" for k, v in sorted(f.items(), key=lambda kv: -kv[1])[:2])
                for key, f in frac.items()), flush=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
