"""Exports a trained CNAT checkpoint (training/train.py) to the plugin's .cnat format, and optionally
a parity fixture: neighbour sets with PyTorch probabilities for tests/parity_check.cpp.

    python export_model.py model.pt model.cnat --name ni-fcc-liquid --crystal FCC=fcc --fixture parity.bin --data heldout.npz
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnat import load_model, probabilities  # noqa: E402
from crystal import crystal_entry  # noqa: E402


def tensors(model):
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    yield sd["embed.weight"]
    yield sd["embed.bias"]
    yield sd["cls"].reshape(-1)
    for i in range(model.config["layers"]):
        for name in ("ln1.weight", "ln1.bias", "qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
                     "ln2.weight", "ln2.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
            yield sd[f"blocks.{i}.{name}"]
    yield sd["norm.weight"]
    yield sd["norm.bias"]
    yield sd["head.weight"]
    yield sd["head.bias"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--classes", nargs="+", default=["LIQUID", "FCC"], help="class names; the first is the disordered phase")
    ap.add_argument("--crystal", action="append", default=[],
                    help="CLASS=topology (fcc, hcp, bcc, ...) or CLASS=cell.lmp, the crystal the plugin aligns that "
                         "class's atoms onto for OpenDXA's structure contract; repeat per crystal class")
    ap.add_argument("--neighbor_types", help="comma-separated atom types the model was trained with as neighbours "
                                             "(the plugin's default --types); all atoms when omitted")
    ap.add_argument("--fixture", type=Path)
    ap.add_argument("--data", type=Path, help="npz with neighbour sets for the parity fixture")
    ap.add_argument("--fixture_size", type=int, default=2000)
    args = ap.parse_args()

    model, ckpt = load_model(args.checkpoint, "cpu")
    cfg, k = model.config, ckpt["k"]
    results = args.checkpoint.with_suffix(".json")
    if len(args.classes) != cfg["classes"]:
        raise SystemExit(f"--classes names {len(args.classes)} classes, the model has {cfg['classes']}")
    meta = {"name": args.name, "description": args.description, "classes": args.classes,
            "normalization": "divide by the largest neighbour distance", "neighbors": k}
    crystals = dict(crystal_entry(spec) for spec in args.crystal)
    unknown = [name for name in crystals if name not in args.classes[1:]]
    if unknown:
        raise SystemExit(f"--crystal names {', '.join(unknown)}, not crystal classes of {', '.join(args.classes[1:])}")
    if crystals:
        meta["crystals"] = crystals
    if args.neighbor_types:
        meta["neighbor_types"] = [int(t) for t in args.neighbor_types.split(",")]
    if results.exists():
        meta["accuracy"] = {n: e["overall"] for n, e in json.loads(results.read_text())["eval"].items()}
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode()

    with open(args.out, "wb") as f:
        f.write(b"CNAT")
        f.write(struct.pack("<8I", 1, cfg["d"], cfg["layers"], cfg["heads"], cfg["mlp"], cfg["classes"], k, len(meta_bytes)))
        f.write(meta_bytes)
        for t in tensors(model):
            f.write(np.ascontiguousarray(t, dtype="<f4").tobytes())
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")

    if args.fixture:
        d = np.load(args.data)
        rng = np.random.default_rng(0)
        sel = rng.choice(len(d["label"]), min(args.fixture_size, len(d["label"])), replace=False)
        atoms = np.ascontiguousarray(d["atoms"][sel, :k], dtype="<f4")
        p = 1.0 - probabilities(model, atoms, k, device="cpu")[:, 0]
        with open(args.fixture, "wb") as f:
            f.write(struct.pack("<2I", len(sel), k))
            f.write(atoms.tobytes())
            f.write(np.ascontiguousarray(p, dtype="<f4").tobytes())
        print(f"wrote {args.fixture} ({len(sel)} neighbour sets)")


if __name__ == "__main__":
    main()
