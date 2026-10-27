"""Trains CNAT with the recipe of the paper (Sec. 2.3) and cnat-jax: cross-entropy, SGD with
momentum 0.9, lr 0.01 with 500 warm-up steps and cosine decay, batch 512, 20,000 steps,
global-norm clipping at 1.0, a fresh uniform random rotation of every sample at every step."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cnat import CNAT, labels_from, normalize, probabilities, random_rotations


def lr_at(step, total, base, warmup):
    progress = min(max((step - warmup) / float(total - warmup), 0.0), 1.0)
    return base * 0.5 * (1.0 + math.cos(math.pi * progress)) * min(1.0, step / warmup)


def per_temperature(model, path, k, device):
    d = np.load(path)
    pred = labels_from(probabilities(model, d["atoms"], k, device))
    names = {0: "liquid", 1: "fcc", 2: "hcp", 3: "bcc"}
    rows = []
    for label in sorted(set(d["label"].tolist()), reverse=True):
        for t in np.unique(d["temperature"][d["label"] == label]):
            m = (d["label"] == label) & (d["temperature"] == t)
            rows.append({"phase": names.get(label, str(label)), "temperature": float(t),
                         "n": int(m.sum()), "accuracy": float((pred[m] == label).mean())})
    return {"overall": float((pred == d["label"]).mean()), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", type=Path, required=True)
    ap.add_argument("--eval", nargs="+", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=79)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    ap.add_argument("--embed_bias_std", type=float, default=0.0)
    ap.add_argument("--cls_std", type=float, default=0.0)
    ap.add_argument("--classes", type=int, default=2)
    ap.add_argument("--precision", choices=("tf32", "bf16"), default="tf32")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    data = [np.load(p) for p in args.train]
    atoms = torch.from_numpy(np.concatenate([d["atoms"][:, :args.k] for d in data])).to(device)
    labels = torch.from_numpy(np.concatenate([d["label"] for d in data])).to(device)
    n = len(labels)
    print(f"train samples {n}, k {args.k}", flush=True)

    model = CNAT(args.d, args.layers, args.heads, args.mlp, args.classes, embed_bias_std=args.embed_bias_std, cls_std=args.cls_std).to(device)
    print(f"parameters {sum(p.numel() for p in model.parameters()):,}", flush=True)
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    perm, cursor = torch.randperm(n, device=device, generator=gen), 0
    history, t0, start = [], time.time(), 1
    resume = args.out.with_suffix(".resume.pt")
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        gen.set_state(state["gen"].cpu())  # generator states are CPU byte tensors, even for CUDA generators
        perm, cursor, history, start = state["perm"].to(device), state["cursor"], state["history"], state["step"] + 1
        print(f"resumed from {resume} at step {start}", flush=True)
    for step in range(start, args.steps + 1):
        if cursor + args.batch > n:
            perm, cursor = torch.randperm(n, device=device, generator=gen), 0
        idx = perm[cursor:cursor + args.batch]
        cursor += args.batch

        x = atoms[idx] @ random_rotations(len(idx), device, gen).transpose(1, 2)
        x = normalize(x)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            loss = F.cross_entropy(model(x).float(), labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            accs = {p.stem: per_temperature(model, p, args.k, device)["overall"] for p in args.eval}
            model.train()
            history.append({"step": step, "loss": loss.item(), **accs})
            el = time.time() - t0
            print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                  + " ".join(f"{k} {v:.4f}" for k, v in accs.items())
                  + f" | {el:.0f}s, eta {el / (step - start + 1) * (args.steps - step):.0f}s", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                        "perm": perm, "cursor": cursor, "history": history, "step": step}, resume)

    model.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = model.config | {"k": args.k}
    torch.save({"config": model.config, "k": args.k, "state_dict": model.state_dict(),
                "args": {k: str(v) for k, v in vars(args).items()}}, args.out)
    results = {"config": config, "history": history,
               "eval": {p.stem: per_temperature(model, p, args.k, device) for p in args.eval},
               "train_seconds": time.time() - t0}
    args.out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    resume.unlink(missing_ok=True)
    print(json.dumps({k: v["overall"] for k, v in results["eval"].items()}), flush=True)


if __name__ == "__main__":
    main()
