"""Unified trainer.

``arm=ar`` trains the autoregressive baseline (Phase 1). ``arm=jepa`` is wired for
Phase 2 and dispatches to the JEPA training step once that arm lands. Checkpoints store
the model state, the model config, and the data meta so probing/eval can rebuild the
model without guessing hyperparameters.

Usage:
    python -m cwm.train --config configs/dev_mps.yaml --arm ar \\
        --data-dir data/dev --out checkpoints/dev_ar
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cwm.data.dataset import GameDataset
from cwm.model.ar import ARModel
from cwm.model.gpt import GPTConfig
from cwm.utils.config import load_config, pick_device


def build_model(arm: str, model_cfg: GPTConfig):
    if arm == "ar":
        return ARModel(model_cfg)
    if arm == "jepa":
        from cwm.model.jepa import JEPAModel  # Phase 2

        return JEPAModel(model_cfg)
    raise ValueError(f"unknown arm: {arm}")


def lr_at(step: int, base_lr: float, warmup: int, total: int, min_lr_frac: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return base_lr * min_lr_frac
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)


def infinite_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        loss = model.compute_loss(batch["input_ids"].to(device))
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def train(args) -> None:
    cfg = load_config(args.config)
    device = pick_device(args.device)
    torch.manual_seed(cfg.get("seed", 0))

    train_ds = GameDataset(args.data_dir, "train", ctx=cfg["model"]["ctx"])
    val_ds = GameDataset(args.data_dir, "val", ctx=cfg["model"]["ctx"])
    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model_cfg = GPTConfig(vocab_size=train_ds.meta["vocab_size"], **cfg["model"])
    model = build_model(args.arm, model_cfg).to(device)
    print(f"device={device}  arm={args.arm}  params={model.backbone.num_params()/1e6:.1f}M  "
          f"train_games={len(train_ds)}  val_games={len(val_ds)}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )

    total_steps = tcfg["max_steps"]
    warmup = tcfg.get("warmup_steps", max(1, total_steps // 20))
    batches = infinite_batches(train_loader)
    t0 = time.time()
    for step in range(total_steps):
        lr = lr_at(step, tcfg["lr"], warmup, total_steps)
        for g in optim.param_groups:
            g["lr"] = lr

        batch = next(batches)
        loss = model.compute_loss(batch["input_ids"].to(device))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
        optim.step()
        if hasattr(model, "update_teacher"):
            model.update_teacher()

        if step % tcfg.get("log_every", 50) == 0 or step == total_steps - 1:
            dt = time.time() - t0
            print(f"step {step:5d}/{total_steps}  loss {loss.item():.4f}  lr {lr:.2e}  "
                  f"{dt:.1f}s", flush=True)

    val_loss = evaluate(model, val_loader, device)
    print(f"final val loss: {val_loss:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_cfg": vars(model_cfg),
            "arm": args.arm,
            "data_meta": train_ds.meta,
            "step": total_steps,
            "val_loss": val_loss,
        },
        out / "model.pt",
    )
    with open(out / "train_config.json", "w") as f:
        json.dump({"config": cfg, "arm": args.arm, "data_dir": args.data_dir}, f, indent=2)
    print(f"saved checkpoint -> {out/'model.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", choices=["ar", "jepa"], default="ar")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=0)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
