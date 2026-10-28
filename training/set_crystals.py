"""Stores the crystal of each crystal class (see crystal.py) and the neighbour types in an exported .cnat
model, in place, without the training checkpoint: only the metadata changes, not the weights.

    python set_crystals.py ../models/forsterite-o.cnat --crystal OLIVINE=lammps/Fo.lmp --neighbor_types 3
"""

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crystal import crystal_entry  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("--crystal", action="append", default=[],
                    help="CLASS=topology (fcc, hcp, bcc, ...) or CLASS=cell.lmp; repeat per class")
    ap.add_argument("--neighbor_types", help="comma-separated atom types the model was trained with as neighbours")
    args = ap.parse_args()

    data = args.model.read_bytes()
    if data[:4] != b"CNAT":
        raise SystemExit(f"{args.model} is not a .cnat model")
    header = list(struct.unpack_from("<8I", data, 4))
    meta_start = 4 + 32
    meta_end = meta_start + header[7]
    meta = json.loads(data[meta_start:meta_end])
    crystals = meta.get("crystals", {})
    for spec in args.crystal:
        name, entry = crystal_entry(spec)
        if name not in meta["classes"][1:]:
            raise SystemExit(f"{name} is not a crystal class of {args.model} ({', '.join(meta['classes'][1:])})")
        crystals[name] = entry
    if crystals:
        meta["crystals"] = crystals
    if args.neighbor_types:
        meta["neighbor_types"] = [int(t) for t in args.neighbor_types.split(",")]
    meta_bytes = json.dumps(meta, separators=(",", ":")).encode()
    header[7] = len(meta_bytes)
    args.model.write_bytes(data[:4] + struct.pack("<8I", *header) + meta_bytes + data[meta_end:])
    print(f"{args.model}: crystals of {', '.join(crystals) or 'no class'}, neighbour types {meta.get('neighbor_types', 'all')}")


if __name__ == "__main__":
    main()
