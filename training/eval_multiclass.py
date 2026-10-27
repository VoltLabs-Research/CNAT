"""Accuracy of a multi-structure CNAT model by metal, structure and temperature, with the confusion
matrix, next to PTM (loose rmsd 0.2 for crystals and strict 0.12 for liquids, its best case for each)."""

import argparse
import json
from pathlib import Path

import numpy as np

from cnat import labels_from, load_model, probabilities

CLASSES = ["LIQUID", "FCC", "HCP", "BCC"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data_multi"))
    ap.add_argument("--ptm", type=Path, default=Path("results/ptm_check.json"))
    ap.add_argument("--out", type=Path, default=Path("results/accuracy_multi.json"))
    args = ap.parse_args()
    model, ckpt = load_model(args.model, "cuda")
    ptm = json.loads(args.ptm.read_text())

    report = {}
    for name in ("heldout", "atoms"):
        d = np.load(args.data / f"{name}.npz")
        pred = labels_from(probabilities(model, d["atoms"], ckpt["k"]))
        conf = np.zeros((len(CLASSES), len(CLASSES)), int)
        np.add.at(conf, (d["label"], pred), 1)
        rows = []
        for metal in np.unique(d["metal"]):
            for label in range(len(CLASSES)):
                for t in np.unique(d["temperature"][(d["metal"] == metal) & (d["label"] == label)]):
                    m = (d["metal"] == metal) & (d["label"] == label) & (d["temperature"] == t)
                    wrong = np.bincount(pred[m], minlength=len(CLASSES))
                    rows.append({"metal": str(metal), "structure": CLASSES[label], "temperature": int(t),
                                 "accuracy": float((pred[m] == label).mean()),
                                 "predicted": {CLASSES[c]: int(wrong[c]) for c in range(len(CLASSES)) if wrong[c]}})
        report[name] = {"overall": float((pred == d["label"]).mean()), "confusion": conf.tolist(), "rows": rows}
        print(f"\n{name}: overall {report[name]['overall']:.4f}")
        print("confusion (rows true, columns predicted):", " ".join(f"{c:>7s}" for c in CLASSES))
        for c, row in zip(CLASSES, conf):
            print(f"  {c:7s}", " ".join(f"{v:7d}" for v in row))

    print(f"\n{'metal':5s} {'structure':9s} {'T':>5s} {'heldout':>8s} {'atoms':>8s} {'PTM':>8s}  errors (atoms)")
    for h, a in zip(report["heldout"]["rows"], report["atoms"]["rows"]):
        key = (f"data.{h['metal']}.{h['structure'].lower()}.{h['temperature']}K.heldout" if h["metal"] != "ni"
               else f"data.{h['structure'].lower()}.{h['temperature']}K.heldout")
        f = ptm.get(key)
        p = (f["strict"]["OTHER"] + f["strict"]["ICO"] if h["structure"] == "LIQUID" else f["loose"][h["structure"]]) if f else float("nan")
        errs = {k: v for k, v in a["predicted"].items() if k != a["structure"]}
        print(f"{h['metal']:5s} {h['structure']:9s} {h['temperature']:5d} {h['accuracy']:8.4f} {a['accuracy']:8.4f} {p:8.4f}  {errs or ''}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
