"""Unified trainer.

``arm=ar`` trains the autoregressive baseline (Phase 1). ``arm=jepa`` is wired for
Phase 2 and dispatches to the JEPA training step once that arm lands. Checkpoints store
the model state, the model config, and the data meta so probing/eval can rebuild the
model without guessing hyperparameters.

On CUDA the training forward runs under bf16 autocast with TF32 matmuls enabled — a large
speedup on Ampere+ (A100/H100) over fp32, with no loss scaling needed for bf16. MPS/CPU
fall back to fp32. Checkpoints are saved periodically (every ``eval_every`` steps), so a
run is crash-safe and any checkpoint can be probed from a separate process (via
``cwm.probe.probe``) while training continues.

Usage:
    python -m cwm.train --config configs/small_gpu.yaml --arm ar \\
        --data-dir data/lichess --out checkpoints/ar
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
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
def evaluate(model, loader, device, autocast_ctx, max_batches: int = 20) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        with autocast_ctx():
            loss = model.compute_loss(batch["input_ids"].to(device))
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def train(args) -> None:
    cfg = load_config(args.config)
    device = pick_device(args.device)
    torch.manual_seed(cfg.get("seed", 0))

    use_amp = device.type == "cuda"
    if use_amp:
        torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+
        torch.backends.cudnn.allow_tf32 = True

    def autocast_ctx():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    train_ds = GameDataset(args.data_dir, "train", ctx=cfg["model"]["ctx"])
    val_ds = GameDataset(args.data_dir, "val", ctx=cfg["model"]["ctx"])
    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=use_amp,
    )
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    model_cfg = GPTConfig(vocab_size=train_ds.meta["vocab_size"], **cfg["model"])
    raw_model = build_model(args.arm, model_cfg).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model
    print(f"device={device}  arm={args.arm}  amp={'bf16' if use_amp else 'off'}  "
          f"compile={args.compile}  params={raw_model.backbone.num_params()/1e6:.1f}M  "
          f"train_games={len(train_ds)}  val_games={len(val_ds)}", flush=True)

    optim = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )

    total_steps = tcfg["max_steps"]
    warmup = tcfg.get("warmup_steps", max(1, total_steps // 20))
    log_every = tcfg.get("log_every", 50)
    eval_every = tcfg.get("eval_every", max(1, total_steps // 10))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_config.json", "w") as f:
        json.dump({"config": cfg, "arm": args.arm, "data_dir": args.data_dir}, f, indent=2)

    def save_checkpoint(step: int, val_loss: float | None) -> None:
        torch.save(
            {
                "model_state": raw_model.state_dict(),
                "model_cfg": vars(model_cfg),
                "arm": args.arm,
                "data_meta": train_ds.meta,
                "step": step,
                "val_loss": val_loss,
            },
            out / "model.pt",
        )

    batches = infinite_batches(train_loader)
    t0 = time.time()
    for step in range(total_steps):
        lr = lr_at(step, tcfg["lr"], warmup, total_steps)
        for g in optim.param_groups:
            g["lr"] = lr

        batch = next(batches)
        with autocast_ctx():
            loss = model.compute_loss(batch["input_ids"].to(device))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 1.0))
        optim.step()
        if hasattr(raw_model, "update_teacher"):
            raw_model.update_teacher()

        if step % log_every == 0 or step == total_steps - 1:
            dt = time.time() - t0
            print(f"step {step:6d}/{total_steps}  loss {loss.item():.4f}  lr {lr:.2e}  "
                  f"{dt:.1f}s", flush=True)

        # Periodic eval + checkpoint (crash-safe, probeable), optionally an inline probe.
        is_eval_step = (step + 1) % eval_every == 0 or step == total_steps - 1
        if is_eval_step:
            val_loss = evaluate(model, val_loader, device, autocast_ctx)
            save_checkpoint(step + 1, val_loss)
            print(f"  [eval] step {step+1}  val_loss {val_loss:.4f}  [checkpoint saved]",
                  flush=True)

    print(f"done. checkpoint -> {out/'model.pt'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", choices=["ar", "jepa"], default="ar")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--compile", action="store_true", help="wrap model in torch.compile")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
