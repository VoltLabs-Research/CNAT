# Training the built-in models

Reproduction of Watanabe, Sato & Shibuta, *Transformer-based structural identification in molecular
dynamics simulations*, Comput. Mater. Sci. 276 (2027) 115172, and the two extensions built on it. The
scripts need Python 3.12 with `torch numpy scipy pyarrow matplotlib`, and LAMMPS with MANYBODY, PTM,
OPENMP (and KSPACE + EXTRA-PAIR for forsterite). Run them from this directory.

`cnat.py` is the model (pre-norm encoder, CLS token, no positional embedding, flax-equivalent
initialisation, so it matches cnat-jax's `CnaTransformer`) plus the periodic k-nearest-neighbour
sampling and the Shoemake rotations of Appendix B. `train.py` applies a fresh random rotation to every
sample at every step, checkpoints to `<out>.resume.pt` and resumes from it. `export_model.py` writes
the `.cnat` file the plugin reads and a parity fixture for `tests/parity_check.cpp`.

## 1. Ni FCC / liquid (the paper): `ni-fcc-liquid`

```bash
lmp -in lammps/in.bulk -var phase fcc    -var tinit 300  -var temp 1200 -var ncell 20 -var seed 1 -var pfile Mishin-Ni-Al-2009.eam.alloy
lmp -in lammps/in.bulk -var phase liquid -var tinit 4000 -var temp 1500 -var ncell 20 -var seed 2 -var pfile Mishin-Ni-Al-2009.eam.alloy
# FCC 600-1700 K and liquid 1500-2600 K in 100 K steps, 32,000 atoms each
python make_dataset.py runs data
python train.py --train data/train.npz data/extra.npz --eval data/test.npz data/heldout.npz data/atoms.npz \
    --k 79 --d 64 --layers 4 --heads 4 --mlp 256 --lr 0.03 --cls_std 1 --embed_bias_std 0.1 \
    --steps 40000 --warmup 1000 --precision bf16 --out models/s64_k79.pt
python export_model.py models/s64_k79.pt ni-fcc-liquid.cnat --name ni-fcc-liquid
```

The potential is Purja Pun & Mishin 2009 (NIST IPR, `2009--Purja-Pun-G-P-Mishin-Y--Ni-Al/2`).
`baseline.py runs` scores CNA and PTM on the same snapshots (`results/baseline.json`), and `eval.py`
and `plot_accuracy.py` make the per-temperature comparison. The coexistence run of Sec. 3.2 is
`lammps/in.interface` (80 x 80 x 6 cells, 153,600 atoms, 100 ps of anisotropic NPT at 1700 K), and
`interface_analysis.py` compares CNAT with PTM on a 1000 x 1000 grid of its x-y plane.

Results (`results/accuracy.png`, `results/interface_s64_k79.png`):

| Model | Recipe | FCC 1700 K | liquid 1500 K | all |
|---|---|---|---|---|
| paper size, ≤2NN (k = 19) | SGD lr 0.01, 20k steps (paper) | 0.82 | 0.86 | 0.965 |
| paper size, ≤5NN (k = 79) | SGD lr 0.01, 20k steps (paper) | 0.82 | 0.88 | 0.968 |
| paper size, ≤2NN | SGD lr 0.03, 20k steps | 0.89 | 0.93 | 0.981 |
| paper size, ≤5NN | SGD lr 0.03, 20k steps | 0.99 | 0.99 | 0.999 |
| compact ≤5NN (built in) | SGD lr 0.03, 40k steps, + extra snapshot | 1.000 | 1.000 | 1.000 |
| CNA (cutoff 3.00451 Å) | | 0.33 | 1.00* | |
| PTM (rmsd 0.16) | | 0.97 | 0.95 | |

\* CNA labels nearly every liquid atom "unknown", which counts as liquid (as the paper notes).
Held-out accuracy is on a snapshot 5 ps after the training one; "at atoms" (the plugin's per-atom
labels) is the same or better. With the paper's own recipe, training is the bottleneck and the wider
context brings nothing; with lr 0.03 the ≤5NN model beats the ≤2NN one as the paper reports.

Deviations from the paper: liquids melted at 4000 K (as in na_dataset's Makefile; 5 ps at the paper's
2500 K leaves the perfect lattice crystalline), 32,000-atom boxes instead of 500,000, and the
coexistence box built in one cell (half melted, solid held fixed) rather than joined from two slabs.

## 2. Metals, LIQUID / FCC / HCP / BCC: `metals`

`lammps/in.metal` with `lammps/metal_jobs.txt` (11,000-14,000 atoms each): Cu (Mishin 2001, FCC), Fe
(Mendelev 2003, BCC), Mg (Sun-Mendelev 2006, HCP), Zr (Mendelev-Ackland 2007, HCP below its 1233 K
transition) at T/Tm = 0.35-1.0, their liquids at 0.85-1.5 Tm, plus the Ni runs above. `ptm_check.py`
labels every snapshot with PTM, and `make_multiclass_dataset.py` keeps a crystal only if at least 60% of
its atoms are in the labelled structure (rmsd 0.2) and a liquid only if at most 3% are crystalline
(rmsd 0.12; at 0.2, PTM matches ~10% of liquid atoms to some template). The β-Zr runs were rejected
(they melted or transformed), so BCC comes from Fe. Class ids follow PTM.

```bash
python ptm_check.py results/ptm_check.json metals runs
python make_multiclass_dataset.py data_multi metals runs --ptm results/ptm_check.json
python train.py --train data_multi/train.npz --eval data_multi/heldout.npz data_multi/atoms.npz --classes 4 \
    --k 79 --d 64 --layers 4 --heads 4 --mlp 256 --lr 0.03 --cls_std 1 --embed_bias_std 0.1 \
    --steps 40000 --warmup 1000 --precision bf16 --out models/metals.pt
python eval_multiclass.py models/metals.pt
python export_model.py models/metals.pt metals.cnat --name metals --classes LIQUID FCC HCP BCC
```

Held-out 0.9987, at atoms 0.9995 (`results/accuracy_metals.json`). Near the melting point:

| | CNAT held-out | CNAT at atoms | PTM rmsd 0.2 |
|---|---|---|---|
| Cu FCC 1327 K (Tm) | 0.993 | 0.999 | 0.948 |
| Fe BCC 1772 K (Tm) | 0.979 | 0.994 | 0.874 |
| Mg HCP 914 K (Tm) | 0.982 | 0.988 | 0.964 |

Errors are almost all liquid against crystal at Tm; crystal structures are practically never confused.
A model trained without Cu (`--exclude_train cu`) labels Cu as well (FCC at Tm 0.985 / 0.998, liquid
≥ 0.998; `results/accuracy_metals_nocu.json`): the normalisation by the farthest neighbour makes the
model independent of the lattice parameter, so it transfers to metals it was not trained on.

## 3. Forsterite on its oxygen sublattice: `forsterite-o`

Pedone et al. 2006 (`lammps/pedone_mgsio.in`; `lammps/in.fo_relax` on the 28-atom `lammps/Fo.lmp`
cell gives a 4.8454, b 10.1888, c 6.0044 Å, against 4.8435, 10.1864, 6.0014 Å in the file). `lammps/in.fo_md` with `lammps/fo_jobs.txt` (2,520 atoms, 8/10 Å cutoffs):
crystal 300-2100 K, liquid 2500-3500 K after a 5000 K melt, and glasses quenched to 300-1500 K.
`make_forsterite_dataset.py` uses only oxygens as neighbours and queries random points, oxygen and
cation positions; it keeps crystals with ≥ 95% of atoms within 1 Å of their site and disordered
snapshots with ≤ 30%.

Held-out 0.982: crystal 300-1500 K and liquids 1.000, crystal 1800 K 0.91, glass 300-900 K 0.95-0.97.
Exported with `--crystal OLIVINE=lammps/Fo.lmp --neighbor_types 3`, so `--model forsterite-o` alone
runs on the oxygens and writes the structure contract on the Si of that cell. On Fo_Disloc (a 135,408-atom slab with a twist
boundary of four screw dislocations) the interior is 99.8% olivine except along the two a[100] screw
lines at x ≈ 50 and 119 Å, which come out disordered over the whole cell; the two c[001] screws are
only faintly marked, and the outer ~4 Å of the free surfaces read as disordered because the model has
not seen vacuum (`results/fo_disloc_cnat.png`).

## 4. Al2SiO5 on its cation sublattice: `al2sio5`

Kyanite, andalusite and sillimanite cells from COD 9000720, 9000715 and 9000710 (Winter & Ghose 1979),
converted by `cif_to_lammps.py` to `lammps/{kyanite,andalusite,sillimanite}.lmp` (types 1 Al, 2 Si,
3 O). Pedone et al. 2006 with Al-O D 0.361581, a 1.900442, r0 2.164818, C 0.9 (`lammps/pedone_alsio_md.in`).
`lammps/in.alsio_md` (triclinic NPT, 2,560 atoms): the three polymorphs at 300, 900 and 1500 K, liquids
at 3000 and 3500 K after a 5000 K melt, and glasses quenched to 300, 900 and 1500 K.
`make_al2sio5_dataset.py --neighbor_types 1 2` takes the Al and Si as neighbours, with the periodic
images of the triclinic cell; same on-site rules as forsterite.

Held-out 0.9994. Exported with `--crystal KYANITE=lammps/kyanite.lmp --crystal ANDALUSITE=lammps/andalusite.lmp
--crystal SILLIMANITE=lammps/sillimanite.lmp --neighbor_types 1,2`.

Known limitation: the model has seen only these phases, their melt and glass. On a kyanite sample
whose non-kyanite part is a phase it never saw (Kyanite_010), it labels 39% of what PSM leaves OTHER
as KYANITE with a median confidence of 0.998, although only 40% of those Al are six-fold coordinated
(100% in kyanite): confident false positives on the unseen phase. Check the Al coordination where
it matters, or train with that phase as a negative.
