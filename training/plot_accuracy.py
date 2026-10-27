"""Accuracy versus temperature for FCC and liquid (paper Figs. 4a and 5a) from results/accuracy.json."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = [  # (column in accuracy.json, label, colour, marker)
    ("b15_k19_lr03/heldout", "≤2NN (paper model, lr 0.03)", "#4c6ef5", "o"),
    ("b15_k79_lr03/heldout", "≤5NN (paper model, lr 0.03)", "#f03e3e", "s"),
    ("b15_k79/heldout", "≤5NN (paper recipe, lr 0.01)", "#f03e3e", "."),
    ("s64_k79/atoms", "≤5NN compact (plugin), at atoms", "#2b8a3e", "D"),
    ("CNA", "CNA", "#f59f00", "x"),
    ("PTM", "PTM", "#862e9c", "o"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accuracy", type=Path, default=Path("results/accuracy.json"))
    ap.add_argument("--out", type=Path, default=Path("results/accuracy.png"))
    args = ap.parse_args()
    table = json.loads(args.accuracy.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, phase, title in zip(axes, ("fcc", "liquid"), ("FCC", "Liquid")):
        keys = sorted((k for k in table if k.startswith(phase + ".")), key=lambda k: int(k.split(".")[1]))
        temps = [int(k.split(".")[1]) for k in keys]
        for col, label, color, marker in SERIES:
            ys = [table[k].get(col) for k in keys]
            if all(y is None for y in ys):
                continue
            ls = ":" if col == "b15_k79/heldout" else "-"
            ax.plot(temps, ys, ls, color=color, marker=marker, ms=4, lw=1.4, label=label)
        if phase == "liquid":
            ax.axvline(1700, color="#e03131", ls="--", lw=1)
            ax.text(1710, 0.32, "Ni melting point", color="#e03131", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Accuracy [-]")
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
