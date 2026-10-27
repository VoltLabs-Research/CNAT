"""Solid-liquid interface comparison of CNAT and PTM (paper Sec. 3.2, Fig. 6).

Both methods label the same 1000 x 1000 grid of the x-y plane at mid-height in z: CNAT from the k nearest
atoms of each grid point, PTM through the label of the nearest atom (an atom-centred field). The interface
line is the upper edge of the connected FCC slab in each x column; FCC regions not connected to the slab
are counted as islands in the liquid.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from cnat import load_model, neighbor_sets, predict, read_lammps_data


def periodic_labels(mask):
    """Connected components of a 2D mask with periodic boundaries."""
    lab, n = ndimage.label(mask)
    parent = np.arange(n + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in list(zip(lab[0], lab[-1])) + list(zip(lab[:, 0], lab[:, -1])):
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    roots = np.array([find(i) for i in range(n + 1)])
    return roots[lab]


def analyse(fcc, ys):
    """fcc: (ny, nx) bool. Returns the interface line (y per column), and island statistics."""
    lab = periodic_labels(fcc)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    slab = lab == np.argmax(sizes)
    ny = fcc.shape[0]
    # Shift rows so the liquid is centred and the upper slab edge is a single crossing per column.
    liquid_rows = (~slab).mean(axis=1)
    centre = int(np.argmax(np.convolve(np.r_[liquid_rows, liquid_rows], np.ones(ny // 4), "same")[:ny]))
    shift = ny // 2 - centre
    slab_s = np.roll(slab, shift, axis=0)
    dy = ys[1] - ys[0]
    line = np.empty(fcc.shape[1])
    for x in range(fcc.shape[1]):
        rows = np.nonzero(slab_s[: ny // 2, x])[0]
        line[x] = (rows.max() if rows.size else 0) * dy
    islands = [s for i, s in enumerate(sizes) if i and i != np.argmax(sizes) and s > 0]
    lab_l = periodic_labels(~fcc)
    sizes_l = np.bincount(lab_l.ravel())
    sizes_l[0] = 0
    holes = [s for i, s in enumerate(sizes_l) if i and i != np.argmax(sizes_l) and s > 0]
    return line, islands, holes, np.roll(fcc, shift, axis=0), shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("data", type=Path)
    ap.add_argument("ptm_dump", type=Path)
    ap.add_argument("--grid", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results/interface"))
    args = ap.parse_args()

    ids, pos, box = read_lammps_data(args.data)
    dump = np.loadtxt(args.ptm_dump, skiprows=9)
    order = np.argsort(dump[:, 0])
    assert np.array_equal(dump[order, 0].astype(np.int64), ids)
    ptm_fcc = dump[order, 5].astype(int) == 1

    lo, length = box[:, 0], box[:, 1] - box[:, 0]
    xs = lo[0] + (np.arange(args.grid) + 0.5) / args.grid * length[0]
    ys = lo[1] + (np.arange(args.grid) + 0.5) / args.grid * length[1]
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, lo[2] + 0.5 * length[2])], 1)

    tree = cKDTree(np.mod(pos - lo, length), boxsize=length)
    _, nearest = tree.query(np.mod(pts - lo, length), k=1, workers=-1)
    ptm_img = ptm_fcc[nearest].reshape(args.grid, args.grid)

    model, ckpt = load_model(args.model, "cuda")
    p = np.concatenate([predict(model, neighbor_sets(pos, box, pts[i:i + 100000], ckpt["k"], tree), ckpt["k"])
                        for i in range(0, len(pts), 100000)])
    cnat_img = (p >= 0.5).reshape(args.grid, args.grid)

    summary = {}
    images = {}
    for name, img in (("PTM", ptm_img), ("CNAT", cnat_img)):
        line, islands, holes, shifted, shift = analyse(img, ys - lo[1])
        dl = np.diff(np.r_[line, line[0]])
        pixel = (length[0] / args.grid) * (length[1] / args.grid)
        big = lambda regions: [r * pixel for r in regions if r * pixel >= 1.0]  # noqa: E731
        summary[name] = {
            "fcc_fraction": float(img.mean()),
            "interface_mean_y": float(line.mean()),
            "interface_rms": float(line.std()),
            "interface_length_ratio": float(np.sum(np.hypot(dl, length[0] / args.grid)) / length[0]),
            "fcc_islands_in_liquid_ge_1A2": len(big(islands)),
            "liquid_holes_in_solid_ge_1A2": len(big(holes)),
            "mean_region_size_A2": float(np.mean(big(islands) + big(holes))) if big(islands) + big(holes) else 0.0,
            "largest_region_A2": float(max(big(islands) + big(holes), default=0.0)),
        }
        images[name] = (shifted, line)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#f2f2f2", "#55c855"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1.3]})
    extent = [0, length[0], 0, length[1]]
    for ax, name in zip(axes[0], ("PTM", "CNAT")):
        ax.imshow(images[name][0], origin="lower", extent=extent, cmap=cmap, interpolation="nearest")
        ax.set_title(f"{name}: FCC (green) / liquid (white)")
        ax.set_xlabel("x [Å]")
        ax.set_ylabel("y [Å] (shifted)")
    ax = axes[1][0]
    xline = (np.arange(args.grid) + 0.5) / args.grid * length[0]
    for name, color in (("PTM", "#3b5bdb"), ("CNAT", "#e03131")):
        ax.plot(xline, images[name][1] - images[name][1].mean(), color=color, lw=0.9, label=name)
    ax.set_xlabel("x [Å]")
    ax.set_ylabel("interface y − mean [Å]")
    ax.legend()
    axes[1][1].axis("off")
    axes[1][1].text(0, 0.95, "\n".join(
        f"{n}: rms {s['interface_rms']:.2f} Å, length ratio {s['interface_length_ratio']:.2f}, "
        f"FCC islands {s['fcc_islands_in_liquid_ge_1A2']}, liquid holes {s['liquid_holes_in_solid_ge_1A2']} (≥1 Å²)"
        for n, s in summary.items()), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out.with_suffix(".png"), dpi=130)
    print(f"wrote {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
