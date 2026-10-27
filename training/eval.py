"""Per-temperature accuracy of trained CNAT models next to CNA and PTM (paper Figs. 4a and 5a)."""

import argparse
import json
from pathlib import Path

import numpy as np

from cnat import load_model, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--baseline", type=Path, default=Path("results/baseline.json"))
    ap.add_argument("--out", type=Path, default=Path("results/accuracy.json"))
    args = ap.parse_args()

    sets = {name: np.load(args.data / f"{name}.npz") for name in ("test", "heldout", "atoms")}
    table = {}
    for path in args.models:
        model, ckpt = load_model(path, "cuda")
        k = ckpt["k"]
        for name, d in sets.items():
            pred = (predict(model, d["atoms"], k) >= 0.5).astype(np.int64)
            for label, phase in ((1, "fcc"), (0, "liquid")):
                for t in np.unique(d["temperature"][d["label"] == label]):
                    m = (d["label"] == label) & (d["temperature"] == t)
                    table.setdefault(f"{phase}.{int(t)}", {})[f"{path.stem}/{name}"] = float((pred[m] == label).mean())
    for row in json.loads(args.baseline.read_text()):
        if row["kind"] == "main":
            key = f"{row['phase']}.{row['T']}"
            table.setdefault(key, {})["CNA"] = row["cna"]
            table[key]["PTM"] = row["ptm"]

    cols = sorted({c for v in table.values() for c in v}, key=lambda c: (c in ("CNA", "PTM"), c))
    print(f"{'phase.T':12s} " + " ".join(f"{c[-22:]:>22s}" for c in cols))
    for key in sorted(table, key=lambda s: (s.split(".")[0], int(s.split(".")[1]))):
        print(f"{key:12s} " + " ".join(f"{table[key].get(c, float('nan')):22.4f}" for c in cols))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
