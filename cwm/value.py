"""Value head for latent planning (Phase 4b).

A small head maps a JEPA latent to a scalar position value, so the JEPA world model can be
*planned* with: roll the dynamics `g` forward over candidate moves and pick the best-valued
resulting latent (see `cwm.play.JEPAPlanner`).

Value target = **material balance** from the side-to-move's perspective, read off the
reconstructed board (ground-truth-as-target, like the board probe — never fed to the model).
This needs no game results, so it runs on the existing shards. Game-outcome value is a clean
upgrade once the data pipeline stores results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cwm import moves
from cwm.data.dataset import GameDataset
from cwm.probe.probe import load_model
from cwm.utils.config import pick_device

_PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def board_value(board: chess.Board) -> float:
    """Material balance in pawns, from the side-to-move's perspective."""
    mover = board.turn
    total = 0
    for piece in board.piece_map().values():
        v = _PIECE_VALUE[piece.piece_type]
        total += v if piece.color == mover else -v
    return float(total)


class ValueHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@torch.no_grad()
def collect_value_data(model, dataset, device, max_positions=60000, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats, vals, total = [], [], 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].tolist()
        hidden = model.hidden_states(input_ids, layer=-1)  # (B,T,C) final-layer latent
        for b, length in enumerate(lengths):
            tokens = input_ids[b, :length].cpu().numpy()
            board = chess.Board()
            for i, tok in enumerate(tokens[1:], start=1):
                board.push(chess.Move.from_uci(moves.decode_id(int(tok))))
                feats.append(hidden[b, i].float().cpu())
                vals.append(board_value(board))
            total += length - 1
        if total >= max_positions:
            break
    X = torch.stack(feats)[:max_positions]
    y = torch.tensor(vals, dtype=torch.float32)[:max_positions]
    return X, y


def fit_value_head(X, y, device, epochs=30, batch_size=4096, lr=1e-3) -> ValueHead:
    head = ValueHead(X.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    X, y = X.to(device), y.to(device)
    head.train()
    for _ in range(epochs):
        idx = torch.randperm(len(X), device=device)
        for i in range(0, len(X), batch_size):
            b = idx[i:i + batch_size]
            loss = F.mse_loss(head(X[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    head.eval()
    return head


def load_value_head(path, device) -> ValueHead:
    blob = torch.load(path, map_location=device, weights_only=False)
    head = ValueHead(blob["dim"], blob["hidden"])
    head.load_state_dict(blob["state_dict"])
    return head.to(device).eval()


def run(args):
    device = pick_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    ctx = ckpt["model_cfg"]["ctx"]
    Xtr, ytr = collect_value_data(model, GameDataset(args.data_dir, "train", ctx=ctx),
                                  device, max_positions=args.max_positions)
    head = fit_value_head(Xtr, ytr, device, epochs=args.epochs)

    Xva, yva = collect_value_data(model, GameDataset(args.data_dir, "val", ctx=ctx),
                                  device, max_positions=20000)
    with torch.no_grad():
        pred = head(Xva.to(device)).cpu()
    mse = F.mse_loss(pred, yva).item()
    corr = torch.corrcoef(torch.stack([pred, yva]))[0, 1].item()
    print(f"value head: val MSE {mse:.3f} (pawns^2), corr {corr:.3f}  [{len(Xtr)} train positions]")

    out = Path(args.checkpoint).parent / "value_head.pt"
    torch.save({"state_dict": head.state_dict(), "dim": Xtr.shape[1], "hidden": 256}, out)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="JEPA checkpoint to attach a value head to")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-positions", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="auto")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
