"""Shared helpers for the CNAT reproduction (Watanabe, Sato & Shibuta, Comput. Mater. Sci. 276 (2027) 115172)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

# Class indices: the plugin reports FCC as 1 and everything else as 0, like the VOLT classifiers.
LIQUID, FCC = 0, 1
PHASES = {"liquid": LIQUID, "fcc": FCC}

# 19 and 79 neighbours: shells up to the 2nd and 5th FCC neighbour (18 and 78 atoms) plus one,
# as in na_dataset_util.const and the paper.
K_2NN, K_5NN = 19, 79


def read_lammps_data(path: str | Path, with_types: bool = False):
    """Reads an orthogonal 'atomic' or 'charge' LAMMPS data file, sorted by id.
    Returns (ids, positions, box[3,2]), plus the atom types when with_types is set."""
    lines = Path(path).read_text().splitlines()
    box = np.zeros((3, 2))
    natoms = 0
    for i, line in enumerate(lines):
        tok = line.split()
        if len(tok) >= 2 and tok[1] == "atoms":
            natoms = int(tok[0])
        for d, name in enumerate(("xlo", "ylo", "zlo")):
            if len(tok) >= 4 and tok[2] == name:
                box[d] = float(tok[0]), float(tok[1])
        if len(tok) >= 6 and tok[3] == "xy" and any(float(t) != 0.0 for t in tok[:3]):
            raise ValueError(f"{path}: triclinic boxes are not supported here")
        if line.startswith("Atoms"):
            start = i + 2
            first = 3 if "charge" in line else 2  # id type [q] x y z
            break
    rows = np.array([l.split()[:first + 3] for l in lines[start:start + natoms]], dtype=np.float64)
    rows = rows[np.argsort(rows[:, 0])]
    ids, pos = rows[:, 0].astype(np.int64), rows[:, first:first + 3].copy()
    return (ids, pos, box, rows[:, 1].astype(np.int64)) if with_types else (ids, pos, box)


def neighbor_sets(positions: np.ndarray, box: np.ndarray, points: np.ndarray, k: int,
                  tree: cKDTree | None = None) -> np.ndarray:
    """k nearest atoms of each point under periodic boundaries, as minimum-image vectors
    relative to the point, sorted by distance. Returns float32 (n_points, k, 3)."""
    lo, length = box[:, 0], box[:, 1] - box[:, 0]
    wrapped = np.mod(positions - lo, length)
    if tree is None:
        tree = cKDTree(wrapped, boxsize=length)
    q = np.mod(points - lo, length)
    _, idx = tree.query(q, k=k, workers=-1)
    delta = wrapped[idx] - q[:, None, :]
    delta -= length * np.round(delta / length)
    return delta.astype(np.float32)


def random_rotations(n: int, device=None, generator=None) -> torch.Tensor:
    """Uniform SO(3) rotations (Shoemake 1992; Appendix B of the paper). Returns (n, 3, 3)."""
    x = torch.rand(n, 3, device=device, generator=generator)
    r1, r2 = torch.sqrt(1 - x[:, 0]), torch.sqrt(x[:, 0])
    t1, t2 = 2 * math.pi * x[:, 1], 2 * math.pi * x[:, 2]
    # Paper's quaternion q = (q0, q1, q2, q3) with q0 the scalar part.
    q0, q1, q2, q3 = r1 * torch.sin(t1), r1 * torch.cos(t1), r2 * torch.sin(t2), r2 * torch.cos(t2)
    return torch.stack([
        torch.stack([q0*q0 + q1*q1 - q2*q2 - q3*q3, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)], -1),
        torch.stack([2*(q1*q2 + q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2*(q2*q3 - q0*q1)], -1),
        torch.stack([2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3], -1),
    ], -2)


def normalize(atoms: torch.Tensor) -> torch.Tensor:
    """Spatial normalisation of cnat-jax: divide by the largest neighbour distance."""
    return atoms / atoms.norm(dim=-1).amax(dim=-1, keepdim=True)[..., None]


class EncoderBlock(nn.Module):
    """Pre-norm block: x + MHA(LN(x)), then x + MLP(LN(x)) (Fig. 2; cnat-jax Encoder1DBlock)."""

    def __init__(self, d: int, heads: int, mlp: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=1e-6)
        self.fc1 = nn.Linear(d, mlp)
        self.fc2 = nn.Linear(mlp, d)
        # flax DenseGeneral flattens each of q, k, v and out to a d x d kernel before xavier_uniform.
        for i in range(3):
            nn.init.xavier_uniform_(self.qkv.weight[i * d:(i + 1) * d])
        for lin in (self.proj, self.fc1, self.fc2):
            nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.proj.bias)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class CNAT(nn.Module):
    """Linear embedding of normalised neighbour vectors, CLS token, encoder, final LN, linear head.
    No positional embedding, so the output does not depend on the neighbour order."""

    def __init__(self, d: int = 256, layers: int = 4, heads: int = 4, mlp: int = 1024, classes: int = 2,
                 embed_bias_std: float = 0.0, cls_std: float = 0.0):
        super().__init__()
        self.config = dict(d=d, layers=layers, heads=heads, mlp=mlp, classes=classes)
        self.embed = nn.Linear(3, d)
        std = 1 / math.sqrt(3) / 0.87962566103423978  # flax lecun_normal: truncated at 2 sigma
        nn.init.trunc_normal_(self.embed.weight, std=std, a=-2 * std, b=2 * std)
        # Zero by default, as in flax; then the first LayerNorm sees W x, which is scale invariant.
        nn.init.normal_(self.embed.bias, std=embed_bias_std) if embed_bias_std > 0 else nn.init.zeros_(self.embed.bias)
        # Zero by default, as in flax; then the CLS query is zero and its first attention a uniform mean.
        self.cls = nn.Parameter(torch.randn(1, 1, d) * cls_std)
        self.blocks = nn.ModuleList(EncoderBlock(d, heads, mlp) for _ in range(layers))
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.head = nn.Linear(d, classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, atoms: torch.Tensor) -> torch.Tensor:
        x = self.embed(atoms)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, 0])


def load_model(path: str | Path, device="cpu") -> tuple[CNAT, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CNAT(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def probabilities(model: CNAT, atoms: np.ndarray, k: int, device="cuda", batch: int = 1024) -> np.ndarray:
    """Class probabilities (n, classes) for neighbour sets of shape (n, >=k, 3)."""
    out = []
    for i in range(0, len(atoms), batch):
        x = normalize(torch.from_numpy(np.ascontiguousarray(atoms[i:i + batch, :k])).to(device))
        out.append(torch.softmax(model(x).float(), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config["classes"]), np.float32)


def predict(model: CNAT, atoms: np.ndarray, k: int, device="cuda", batch: int = 1024) -> np.ndarray:
    """P(FCC) of a binary LIQUID/FCC model."""
    return probabilities(model, atoms, k, device, batch)[:, FCC]


def labels_from(p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """The plugin's rule: class 0 (disordered) unless 1 - P(0) reaches the threshold, else the most
    probable other class. For a binary model this is P(1) >= threshold."""
    return np.where(1.0 - p[:, 0] >= threshold, 1 + np.argmax(p[:, 1:], axis=1), 0)
