# CNAT: Comprehensive Neighbor Analysis Transformer

CNAT is the name the authors gave their reference implementation; the plugin keeps it.

Identifies the phase at arbitrary points of an atomistic configuration with a Transformer over the
k nearest atoms, following Watanabe, Sato & Shibuta, *Transformer-based structural identification in
molecular dynamics simulations*, Comput. Mater. Sci. 276 (2027) 115172 (reference code:
[cnat-jax](https://github.com/MaterialsModellingLab/cnat-jax)). Atoms are labelled from the
neighbourhood of their own position, and a regular grid over the cell gives the phase field, whose
iso-surface is exported as the interface mesh.

## Install

```bash
vpm install @voltlabs/cnat
```

## CLI

```bash
cnat <input_dump> [output_base] [options]
```

| Option | Default | Description |
|---|---|---|
| `--model` | `metals` | Built-in model name (below) or a `.cnat` file. |
| `--threshold` | `0.5` | A point is crystalline when 1 − P(liquid) reaches it, and then takes its most probable crystal class. |
| `--grid_spacing` | `2.0` | Spacing of the phase-field grid in Å; `0` labels atoms only. |
| `--export_interface_mesh` | `true` | Export the iso-surface 1 − P(liquid) = threshold. |
| `--interface_smoothing` | `2.0` | Gaussian σ in Å applied to the field before meshing; `0` disables. |
| `--export_phase_field` | `false` | Export the grid points with their class probabilities. |
| `--types` | all | Comma-separated atom types that form the neighbour sublattice (e.g. the oxygens of a silicate). |

## Built-in models

All use the architecture of the paper scaled down (hidden 64, 4 layers, 4 heads, MLP 256, 200k
parameters, 79 neighbours: the FCC shells up to the 5th neighbour plus one). Positions are divided by
the largest neighbour distance, so a model does not depend on the lattice parameter.

| Name | Classes | Training data | Held-out accuracy |
|---|---|---|---|
| `metals` (default) | LIQUID, FCC, HCP, BCC | Ni, Cu (FCC), Fe (BCC), Mg, Zr (HCP) EAM, crystals 0.3–1.0 Tm, liquids 0.85–1.5 Tm | 0.9987 (at atoms 0.9995); at Tm Cu 0.993, Fe 0.979, Mg 0.982, against PTM 0.948, 0.874, 0.964 |
| `ni-fcc-liquid` | LIQUID, FCC | Ni, Purja Pun–Mishin 2009 EAM: FCC 600–1700 K, liquid 1500–2600 K (the paper's set) | 1.000 at every temperature |
| `forsterite-o` | DISORDERED, OLIVINE | Mg2SiO4, Pedone 2006: crystal 300–2100 K, liquid 2500–3500 K, glass 300–1500 K; oxygen neighbours only, run with `--types <O type>` | 0.982 (crystal to 1500 K and liquids 1.000, glass 0.95–0.99) |

Structure ids follow PTM: LIQUID/OTHER 0, FCC 1, HCP 2, BCC 3; for `forsterite-o`, DISORDERED 0 and
OLIVINE 1. A `metals` model trained without Cu labels Cu as well as the full one, so the models should
carry over to other metals of the same structures. Details: `training/README.md`.

On the Ni solid–liquid coexistence of the paper (Sec. 3.2; 153,600 atoms at 1700 K, 1000 × 1000 grid
of the x–y plane), PTM leaves 136 liquid holes in the solid and 18 crystalline islands in the liquid
(≥ 1 Å², the largest 43 Å²), while `ni-fcc-liquid` leaves 3 and 5 (the largest 4 Å²), with a smoother
interface line (rms 2.26 Å against 2.63 Å).

Cost: about 2.6 ms per point and thread on a laptop CPU (AVX2 clone of the dense layers on Linux
x86-64). The coexistence frame (153,600 atoms and a 232,804-point grid) takes 131 s on 16 threads.

The phase field is nearly binary, so its raw iso-surface is a staircase of grid cells: on the
coexistence frame it measures 1.65 times the flat interface. `--interface_smoothing 2` (the default)
brings that to 1.06 without moving the solid fraction (55.3% before and after).

## Exports

| Output file | Exposure | Exporter → artifact |
|---|---|---|
| `{output_base}_atoms.parquet` | Phase Identification | AtomisticExporter → glb |
| `{output_base}_interface_mesh.parquet` (+ `.vertices`, `.facets`) | Solid-Liquid Interface | MeshExporter → glb |
| `{output_base}_phase_field.parquet` | Phase Field | AtomisticExporter → glb |
| `{output_base}_cnat.parquet` | Phase Summary | — (listing: atom counts, volume fractions, interface area) |

Per-atom and per-point tables carry `structure_id`, `structure_name`, `probability_crystalline`
(1 − P(class 0), the field of the interface mesh) and one `probability_<class>` column per class.

## Models and training

`training/` holds the pipeline that produced the built-in models: LAMMPS inputs for the bulk
snapshots and the coexistence run, dataset sampling (random points and atom positions, periodic
k-nearest neighbours), the PyTorch model (`cnat.py`, equivalent to cnat-jax's `CnaTransformer`),
training, evaluation and `export_model.py`, which writes a `.cnat` file. `.cnat` files in `models/`
listed in `CNAT_MODELS` (CMake) are compiled into the binary; any other `.cnat` file can
be passed with `--model`.

Deviations from the paper, found while reproducing it:

- Liquids are melted at 4000 K, as in na_dataset's Makefile; 5 ps at the paper's 2500 K leaves a
  perfect Ni lattice crystalline.
- SGD with learning rate 0.03 instead of 0.01. With the paper's 0.01 and 20,000 steps the ≤5NN and
  ≤2NN models both stop at 96.8% (FCC at 1700 K: 0.82); with 0.03 the ≤5NN model reaches 0.999 and
  the ≤2NN model 0.98, which is the difference the paper reports.

## Tests

```bash
cmake -DCNAT_BUILD_TESTS=ON ...
cnat-parity-check model.cnat fixture.bin   # C++ against PyTorch (export_model.py --fixture)
cnat-mesh-check                            # sphere and periodic slab iso-surfaces
```

---

Full input contract and examples: https://docs.voltcloud.dev/docs/plugins
