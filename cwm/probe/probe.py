"""Linear / MLP probes for board reconstruction.

Freeze a trained model, read hidden states at each ply, and train a probe to predict the
64-square board (13 classes each) from those hidden states. Reports mean per-square
accuracy and the linear-vs-MLP gap (a small gap means the board is stored *linearly*).

This is the harness validator for Phase 1: on a real, well-trained model the linear probe
should hit ~99%.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cwm.data.dataset import GameDataset
from cwm.model.ar import ARModel
from cwm.model.gpt import GPTConfig
from cwm.probe import boards
from cwm.utils.config import pick_device


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = GPTConfig(**ckpt["model_cfg"])
    if ckpt["arm"] == "ar":
        model = ARModel(cfg)
    else:
        from cwm.model.jepa import JEPAModel

        model = JEPAModel(cfg, ckpt.get("jepa_cfg", {}))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def collect_features(model, dataset, device, layer=-1, relative=True,
                     max_positions=60000, batch_size=64):
    """Gather (hidden_state, board_label) pairs across games."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats, labs = [], []
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].tolist()
        hidden = model.hidden_states(input_ids, layer=layer)  # (B, T, C)
        for b, length in enumerate(lengths):
            tokens = input_ids[b, :length].cpu().numpy()
            positions, labels = boards.game_labels(tokens, relative=relative, skip_start=True)
            if len(positions) == 0:
                continue
            feats.append(hidden[b, positions].float().cpu())
            labs.append(torch.from_numpy(labels))
            total += len(positions)
        if total >= max_positions:
            break
    X = torch.cat(feats)[:max_positions]
    Y = torch.cat(labs)[:max_positions]
    return X, Y


@torch.no_grad()
def collect_all_layer_features(model, dataset, device, relative=True,
                              max_positions=60000, batch_size=64):
    """Like collect_features but gathers *every* layer's hidden states in one forward pass.

    Returns (Xs, Y) where Xs[li] is the feature matrix at layer li (0..n_layer, last = final
    ln_f output, i.e. what --layer -1 selects), and Y is the shared board labels.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats_per_layer, labs, total = None, [], 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].tolist()
        _, hiddens = model.backbone(input_ids, is_causal=True, return_hidden=True)
        if feats_per_layer is None:
            feats_per_layer = [[] for _ in range(len(hiddens))]
        for b, length in enumerate(lengths):
            tokens = input_ids[b, :length].cpu().numpy()
            positions, labels = boards.game_labels(tokens, relative=relative, skip_start=True)
            if len(positions) == 0:
                continue
            for li, h in enumerate(hiddens):
                feats_per_layer[li].append(h[b, positions].float().cpu())
            labs.append(torch.from_numpy(labels))
            total += len(positions)
        if total >= max_positions:
            break
    Xs = [torch.cat(f)[:max_positions] for f in feats_per_layer]
    Y = torch.cat(labs)[:max_positions]
    return Xs, Y


class LinearProbe(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.head = nn.Linear(dim, boards.NUM_SQUARES * boards.NUM_CLASSES)

    def forward(self, x):
        return self.head(x).view(-1, boards.NUM_SQUARES, boards.NUM_CLASSES)


class MLPProbe(nn.Module):
    def __init__(self, dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, boards.NUM_SQUARES * boards.NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x).view(-1, boards.NUM_SQUARES, boards.NUM_CLASSES)


def majority_baseline(Y: torch.Tensor) -> float:
    """No-information floor: always predict each square's most common class.

    A probe is only meaningful to the extent it beats this — many squares are simply
    empty most of the time, so a high raw accuracy can be trivial.
    """
    n = len(Y)
    acc = 0.0
    for sq in range(boards.NUM_SQUARES):
        counts = torch.bincount(Y[:, sq], minlength=boards.NUM_CLASSES)
        acc += counts.max().item() / n
    return acc / boards.NUM_SQUARES


def train_probe(X, Y, kind, device, epochs=30, batch_size=4096, lr=1e-3, val_frac=0.1):
    n = len(X)
    perm = torch.randperm(n)
    X, Y = X[perm], Y[perm]
    n_val = int(n * val_frac)
    Xtr, Ytr, Xva, Yva = X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]

    dim = X.shape[1]
    probe = (LinearProbe(dim) if kind == "linear" else MLPProbe(dim)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xva, Yva = Xva.to(device), Yva.to(device)
    for _ in range(epochs):
        idx = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), batch_size):
            b = idx[i:i + batch_size]
            logits = probe(Xtr[b])
            loss = F.cross_entropy(logits.reshape(-1, boards.NUM_CLASSES), Ytr[b].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(Xva).argmax(-1)  # (N, 64)
        correct = (preds == Yva).float()
        mean_acc = correct.mean().item()
        per_square = correct.mean(0).cpu().numpy()  # (64,)
    return mean_acc, per_square


def run(args) -> None:
    device = pick_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    dataset = GameDataset(args.data_dir, args.split, ctx=ckpt["model_cfg"]["ctx"])
    rel = not args.absolute

    if args.layer_sweep:
        print(f"layer sweep: {args.checkpoint}  relative={rel}  device={device}")
        Xs, Y = collect_all_layer_features(model, dataset, device, relative=rel,
                                           max_positions=args.max_positions)
        base = majority_baseline(Y)
        print(f"collected {len(Y)} positions  layers={len(Xs)}  baseline {base*100:.2f}%")
        best_li, best_acc = 0, -1.0
        for li, X in enumerate(Xs):
            acc, _ = train_probe(X, Y, "linear", device, epochs=args.epochs)
            name = f"layer {li}" + ("  (final = -1)" if li == len(Xs) - 1 else "")
            print(f"  {name:24s} linear {acc*100:.2f}%")
            if acc > best_acc:
                best_li, best_acc = li, acc
        mlp_acc, _ = train_probe(Xs[best_li], Y, "mlp", device, epochs=args.epochs)
        print(f"BEST layer {best_li}: linear {best_acc*100:.2f}%  "
              f"(+{(best_acc-base)*100:.2f} over baseline);  "
              f"MLP {mlp_acc*100:.2f}%  gap {(mlp_acc-best_acc)*100:.2f} pts")
        return

    print(f"probing {args.checkpoint}  layer={args.layer}  relative={rel}  device={device}")
    X, Y = collect_features(
        model, dataset, device, layer=args.layer, relative=rel,
        max_positions=args.max_positions,
    )
    print(f"collected {len(X)} positions  dim={X.shape[1]}")

    base = majority_baseline(Y)
    lin_acc, lin_sq = train_probe(X, Y, "linear", device, epochs=args.epochs)
    mlp_acc, _ = train_probe(X, Y, "mlp", device, epochs=args.epochs)
    print(f"majority-class baseline (no info):     {base*100:.2f}%")
    print(f"linear probe mean per-square accuracy: {lin_acc*100:.2f}%  "
          f"(+{(lin_acc-base)*100:.2f} over baseline)")
    print(f"MLP    probe mean per-square accuracy: {mlp_acc*100:.2f}%")
    print(f"linearity gap (mlp - linear): {(mlp_acc-lin_acc)*100:.2f} pts")
    print(f"worst squares (acc): {np.sort(lin_sq)[:5].round(3).tolist()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--layer-sweep", action="store_true",
                    help="probe every layer in one pass and report the best")
    ap.add_argument("--absolute", action="store_true", help="use absolute (white/black) labels")
    ap.add_argument("--max-positions", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="auto")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
