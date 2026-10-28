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
| `--types` | the model's | Comma-separated atom types that form the neighbour sublattice (e.g. the oxygens of a silicate); by default those the model was trained with (`forsterite-o`: 3), else all. |
| `--topologies` | `auto` | Lattice topologies of the structure contract (below), comma-separated names of the lattice registry; `auto` takes the crystal the model stores for each class, FCC and HCP each with the other; `none` writes no contract. |
| `--rmsd` | `0.2` | Largest RMSD, in units of the mean neighbour distance, at which a crystalline atom joins the contract. |
| `--lattice_dir` | bundled | Directory of the lattice topology YAMLs (`lattices/`, the same files as OpenDXA's). |

## Built-in models

All use the architecture of the paper scaled down (hidden 64, 4 layers, 4 heads, MLP 256, 200k
parameters, 79 neighbours: the FCC shells up to the 5th neighbour plus one). Positions are divided by
the largest neighbour distance, so a model does not depend on the lattice parameter.

| Name | Classes | Training data | Held-out accuracy |
|---|---|---|---|
| `metals` (default) | LIQUID, FCC, HCP, BCC | Ni, Cu (FCC), Fe (BCC), Mg, Zr (HCP) EAM, crystals 0.3–1.0 Tm, liquids 0.85–1.5 Tm | 0.9987 (at atoms 0.9995); at Tm Cu 0.993, Fe 0.979, Mg 0.982, against PTM 0.948, 0.874, 0.964 |
| `ni-fcc-liquid` | LIQUID, FCC | Ni, Purja Pun–Mishin 2009 EAM: FCC 600–1700 K, liquid 1500–2600 K (the paper's set) | 1.000 at every temperature |
| `forsterite-o` | DISORDERED, OLIVINE | Mg2SiO4, Pedone 2006: crystal 300–2100 K, liquid 2500–3500 K, glass 300–1500 K; oxygen neighbours only (type 3, stored with the model; pass `--types` if the oxygens have another type) | 0.982 (crystal to 1500 K and liquids 1.000, glass 0.95–0.99) |
| `al2sio5` | DISORDERED, KYANITE, ANDALUSITE, SILLIMANITE | Al2SiO5, Pedone 2006: the three polymorphs 300–1500 K, liquid 3000–3500 K, glass 300–1500 K; Al and Si neighbours (types 1, 2, stored with the model) | 0.9994; labels unseen Al2SiO5 phases with high confidence (training README) |

Structure ids are those of the other VOLT structure identifiers: LIQUID/DISORDERED 0, FCC 2, HCP 3,
BCC 4; the other crystal classes of the built-in models keep one id each, OLIVINE 1001, KYANITE 1002,
ANDALUSITE 1003, SILLIMANITE 1004, and those of any other model count up from 1100. A `metals` model trained without Cu labels Cu as well as the full one, so the models should
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
| `{output_base}_annotated.dump`, `_clusters.table`, `_cluster_transitions.table`, `_neighbor_lattice.parquet` | Clusters Table, Clusters Transitions, Neighbor Lattice | — (`volt/structure-identification@1`, read by OpenDXA) |

Per-atom and per-point tables carry `structure_id`, `structure_name`, `probability_crystalline`
(1 − P(class 0), the field of the interface mesh) and one `probability_<class>` column per class. With
the contract, the atoms table adds `lattice_structure` (the topology the atom joined, or OTHER) and
`lattice_rmsd`.

## Structure contract (OpenDXA)

CNAT writes the same contract as CNA, PTM and PSM, so OpenDXA, ElasticStrain and LineReconstructionDXA
can run on its output. CNAT decides where the crystal is and of which class; the neighbour order that
DXA's Burgers circuits need comes from the lattice topologies, the YAML files of the lattice registry
under `lattices/` (fcc, hcp, bcc, sc, cubic_diamond, hex_diamond, bct, fct, st, a7, a7_bi, or any file
added to `--lattice_dir`), not from PTM's or CNA's built-in templates.

Each crystalline atom of the neighbour sublattice is aligned onto every topology: of the proper
rotations that carry two shell vectors onto two of its nearest neighbours, the one whose one-to-one
assignment leaves the smallest residual, refined by a least-squares rotation and isotropic scale. It
keeps the topology its class names unless another fits at least twice as well and two of its shell
neighbours took that one too: a hot FCC atom stays FCC, a stacking fault becomes HCP even with
`ni-fcc-liquid`, and an atom that fits HCP alone (thermal noise at the melting point) is OTHER. A class
with no topology (OLIVINE, or an explicit `--topologies`) takes the best fit. Atoms within `--rmsd`
become lattice sites in that order, and the shared cluster builder of structure-identification grows
the crystal clusters, with the rotations taken over all the lattice vectors of a topology as CNA does,
so the two sublattices of hcp and of the diamonds join one cluster; sc, st and a7, whose bonds lack
common neighbours, join by orientation as PTM's SC does. With `--types`, the contract holds only the
sublattice atoms. Disordered atoms are never lattice sites.

### Crystals stored with the model

Every crystal class of a model carries the crystal it was trained on, so the contract needs no option
for any phase: a lattice topology of the registry (`metals`: FCC, HCP and BCC; `ni-fcc-liquid`: FCC),
or the reference cell its training crystal was built from (`forsterite-o`: OLIVINE, the 28-atom
Pedone cell of `training/lammps/Fo.lmp`; `al2sio5`: the kyanite, andalusite and sillimanite cells it was
trained on). The model also stores the neighbour types it was trained with, so `--model forsterite-o`
alone writes the forsterite contract.

A cell is not one registry topology: every site of its sparsest sublattice (the species with the
fewest atoms in the cell, Si in forsterite) becomes a topology of its own, whose shell is the site's
neighbours of that species in the cell, and each atom takes the sites of its type only. The sparsest
sublattice has sites farthest apart and least alike under the pseudo-symmetries of the denser ones
(forsterite's oxygens are nearly hexagonal close packed, and the full crystal has sites that nearly
repeat at c/2), and one Wyckoff orbit has every translation of the crystal, so its Burgers vectors are
the crystal's. Sites that a rotation or translation of the space group or a pseudo-symmetry relates fit
an atom alike, so an atom's own best fit does not decide its site: each grain grows from its
best-aligned atom, in order of RMSD, as a registration does. A neighbour is the site the bond to it
lands on in the cell and takes the orientation of the atom it is reached from, if its refined fit
stays within 10° and `--rmsd`, and agrees on every bond with the neighbours already in the grain. Two
grains of one cell that meet within 10° are one crystal whose registrations disagree; the smaller one,
and a seed no neighbour joined, are no lattice sites. Grains join into clusters by orientation.

On Fo_Disloc (135,408 atoms; a forsterite slab with a ~4° twist boundary), `--model forsterite-o` and
OpenDXA give the two a[100] screws (b = 4.84 Å, 341 Å of the 342 Å that two lines across the cell
make) and the two c[001] screws (b = 6.00 Å, 274 Å of 280 Å), with 20 Å of short segments where they
cross; 15,592 of the 15,769 crystalline Si in one cluster and 12 in a second. With `--model al2sio5` alone
(contract on Si) the six kyanite dislocation fixtures at σ 0.1 Å and twelve 900–1500 K MD frames of the
[100] and [001] screws give the exact b and no other segment; the hot [001] lines are wavy (up to 49 Å
in a 28 Å cell), the core resolved only to the Si spacing. Perfect cells and 300–1500 K bulk MD of the
three polymorphs give one cluster and no segment, except andalusite at 1500 K, where a layer of 64 Si
takes a setting turned by 120°: the 14-neighbour Si shell of andalusite is nearly invariant under that
turn, so its grain grows apart (no segment in the perfect crystal, but a circuit through both settings
around a dislocation would not close on a lattice vector).

To give a model its crystals, pass `--crystal CLASS=topology` or `--crystal CLASS=cell.lmp` and
`--neighbor_types` to `export_model.py`, or to `set_crystals.py` for an exported model (only the
metadata changes). A model without crystals keeps the old behaviour: a class names its topology.

OpenDXA's `--reference_topology` takes the same names (e.g. `fcc`); for a cell, the clusters carry its
name (`olivine`), which OpenDXA uses whatever the reference.

Checked against PTM (`--rmsd 0.1`) through OpenDXA on a Ni vacancy Frank loop (51,323 atoms, Mishin
2009 EAM) after 8 ps of NPT, which dissociates into 1/6<112>, 1/6<110> and 1/3<111> segments; the
energy-minimised frame of each temperature is the reference:

| T | PTM: lattice sites, line length | CNAT: lattice sites, line length | Minimised (PTM) |
|---|---|---|---|
| 300 K | 99.2%, 409 Å | 99.3%, 386 Å | — |
| 1000 K | 97.9%, 365 Å | 98.6%, 352 Å | 437 Å |
| 1400 K | 77.3%, 296 Å | 93.6%, 343 Å | 423 Å |
| 1600 K | 52.2%, 74 Å | 86.8%, 322 Å | 397 Å |

Near the melting point CNAT keeps the loop that PTM loses. Most of its shortfall at 1400–1600 K is
partial-dislocation detail: where the hot stacking fault thins, a Shockley and a stair-rod
(1/6<112> + 1/6<110> = 1/3<111>) are traced as one Frank segment (121 Å of 1/3<111> at 1600 K against
32 Å in the minimised frame).

On the solid-liquid coexistence (153,600 atoms at 1700 K, no dislocations), PTM leaves 29% of the atoms
crystalline and a defect mesh of 55,628 facets; CNAT's contract gives the interface alone (6,688
facets) and no dislocation segment. Perfect fcc, hcp, bcc, cubic diamond, sc and st lattices each give one cluster,
and a screw dipole in st (b = c, a lattice PTM has no template for) gives two lines with b = ±[0 0 1.2].

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
