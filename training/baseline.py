"""CNA and PTM accuracy of every Ni snapshot (paper Figs. 4a and 5a): LAMMPS cna/atom with cutoff 3.00451 A
and ptm/atom default 0.16 (lammps/in.baseline, as in na_dataset), scored like na_dataset's eval_cna.py and
eval_ptm.py: the fraction of FCC atoms in FCC snapshots and of CNA "unknown" / PTM "other" atoms in liquid
ones. Writes results/baseline.json for eval.py."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

NAME = re.compile(r"data\.(fcc|liquid)\.(\d+)K(?:\.(heldout|extra))?$")


def score(lmp, script, pfile, path):
    phase, temp, kind = NAME.match(path.name).groups()
    with tempfile.NamedTemporaryFile(suffix=".dump") as out:
        subprocess.run([lmp, "-in", script, "-var", "dfile", str(path), "-var", "pfile", pfile,
                        "-var", "out", out.name, "-log", "none", "-screen", "none"],
                       check=True, env=os.environ | {"OMP_NUM_THREADS": "1"})
        a = np.loadtxt(out.name, skiprows=9)
    cna, ptm = a[:, 4].astype(int), a[:, 5].astype(int)
    fcc = phase == "fcc"
    return {"phase": phase, "T": int(temp), "kind": kind or "main",
            "cna": float((cna == 1).mean() if fcc else (cna == 5).mean()),
            "ptm": float((ptm == 1).mean() if fcc else (ptm == 0).mean()),
            "ptm_fcc": float((ptm == 1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--pfile", default="Mishin-Ni-Al-2009.eam.alloy")
    ap.add_argument("--lmp", default="lmp")
    ap.add_argument("--script", default="lammps/in.baseline")
    ap.add_argument("--out", type=Path, default=Path("results/baseline.json"))
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    paths = sorted(p for p in args.runs.iterdir() if NAME.match(p.name))
    with ThreadPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(lambda p: score(args.lmp, args.script, args.pfile, p), paths))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1))
    for r in rows:
        if r["kind"] == "main":
            print(f"{r['phase']:7s} {r['T']:5d}  CNA {r['cna']:.3f}  PTM {r['ptm']:.3f}")


if __name__ == "__main__":
    main()
