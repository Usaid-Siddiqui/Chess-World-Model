"""Unified trainer.

``arm=ar`` trains the autoregressive baseline (Phase 1). ``arm=jepa`` is wired for
Phase 2 and dispatches to the JEPA training step once that arm lands.

Training length is **data-relative**: the config sets ``epochs`` and the trainer derives
``max_steps`` from the actual dataset size (``max_steps`` may be given to override). This
prevents the failure mode where an arbitrary step count silently becomes 100+ epochs and
the model memorizes the data. The header prints epochs/steps/tokens so the run's scale is
explicit.

On CUDA the forward runs under bf16 autocast with TF32 matmuls (a large speedup on
Ampere+; bf16 needs no loss scaling); MPS/CPU fall back to fp32.

Instrumentation, written under ``--out``:
  * ``train.log``   — a copy of everything printed,
  * ``metrics.csv`` — ``step,train_loss,val_loss,lr,elapsed_s`` appended each eval,
  * ``model.pt``    — rolling latest checkpoint (crash-safe / resume),
  * ``model_best.pt`` — checkpoint at the lowest val loss (for probing / use).

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


def build_model(arm: str, model_cfg: GPTConfig, jepa_cfg: dict):
    if arm == "ar":
        return ARModel(model_cfg)
    if arm == "jepa":
        from cwm.model.jepa import JEPAModel

        return JEPAModel(model_cfg, jepa_cfg)
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
def evaluate(model, loader, device, autocast_ctx, max_batches: int = 40) -> float:
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

    ctx = cfg["model"]["ctx"]
    train_ds = GameDataset(args.data_dir, "train", ctx=ctx)
    val_ds = GameDataset(args.data_dir, "val", ctx=ctx)
    tcfg = cfg["train"]
    batch_size = tcfg["batch_size"]
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=use_amp,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model_cfg = GPTConfig(vocab_size=train_ds.meta["vocab_size"], **cfg["model"])
    jepa_cfg = dict(cfg.get("jepa", {}))
    if args.vicreg is not None:  # CLI override for the anti-collapse experiment
        jepa_cfg["vicreg_weight"] = args.vicreg
    raw_model = build_model(args.arm, model_cfg, jepa_cfg).to(device)
    model = torch.compile(raw_model) if args.compile else raw_model

    # Data-relative schedule: derive steps from epochs and the real dataset size.
    steps_per_epoch = max(1, len(train_ds) // batch_size)
    max_steps = tcfg.get("max_steps")
    if max_steps is None:
        if "epochs" not in tcfg:
            raise ValueError("config train section needs either 'epochs' or 'max_steps'")
        max_steps = int(math.ceil(tcfg["epochs"] * steps_per_epoch))
    epochs = max_steps / steps_per_epoch
    warmup = tcfg.get("warmup_steps") or max(1, int(tcfg.get("warmup_frac", 0.05) * max_steps))
    log_every = tcfg.get("log_every", 50)
    eval_every = tcfg.get("eval_every", max(1, max_steps // 10))
    tokens_per_epoch = len(train_ds.tokens)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train_config.json", "w") as f:
        json.dump({"config": cfg, "arm": args.arm, "data_dir": args.data_dir}, f, indent=2)
    logf = open(out / "train.log", "a")
    metrics_path = out / "metrics.csv"
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    metricsf = open(metrics_path, "a")
    if write_header:
        metricsf.write("step,train_loss,val_loss,lr,elapsed_s\n")
        metricsf.flush()

    def log(msg: str) -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    def save_checkpoint(path: Path, step: int, val_loss: float | None) -> None:
        torch.save(
            {
                "model_state": raw_model.state_dict(),
                "model_cfg": vars(model_cfg),
                "jepa_cfg": jepa_cfg,
                "arm": args.arm,
                "data_meta": train_ds.meta,
                "step": step,
                "val_loss": val_loss,
            },
            path,
        )

    log(f"=== run {time.strftime('%Y-%m-%dT%H:%M:%S')}  device={device}  arm={args.arm}  "
        f"amp={'bf16' if use_amp else 'off'}  compile={args.compile} ===")
    log(f"model: params={raw_model.backbone.num_params()/1e6:.1f}M  dropout={model_cfg.dropout}")
    log(f"data:  train_games={len(train_ds)}  val_games={len(val_ds)}  "
        f"tokens/epoch={tokens_per_epoch/1e6:.1f}M")
    log(f"sched: epochs={epochs:.2f}  steps={max_steps}  steps/epoch={steps_per_epoch}  "
        f"warmup={warmup}  eval_every={eval_every}  tokens_seen={epochs*tokens_per_epoch/1e6:.0f}M")

    optim = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )

    best_val = float("inf")
    last_loss = float("nan")
    batches = infinite_batches(train_loader)
    t0 = time.time()
    for step in range(max_steps):
        lr = lr_at(step, tcfg["lr"], warmup, max_steps)
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
        last_loss = loss.item()

        if step % log_every == 0 or step == max_steps - 1:
            log(f"step {step:6d}/{max_steps}  loss {last_loss:.4f}  lr {lr:.2e}  "
                f"{time.time()-t0:.1f}s")

        if (step + 1) % eval_every == 0 or step == max_steps - 1:
            val_loss = evaluate(model, val_loader, device, autocast_ctx)
            elapsed = time.time() - t0
            metricsf.write(f"{step+1},{last_loss:.4f},{val_loss:.4f},{lr:.3e},{elapsed:.1f}\n")
            metricsf.flush()
            save_checkpoint(out / "model.pt", step + 1, val_loss)  # rolling latest
            tag = ""
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(out / "model_best.pt", step + 1, val_loss)
                tag = "  *best*"
            extra = ""
            if hasattr(raw_model, "last_latent_std"):  # JEPA collapse guard
                extra = f"  latent_std {raw_model.last_latent_std:.3f}"
            log(f"  [eval] step {step+1}  val_loss {val_loss:.4f}  "
                f"train_loss {last_loss:.4f}{tag}{extra}  [saved]")

    log(f"done. best val_loss {best_val:.4f}  ->  {out/'model_best.pt'}  (latest {out/'model.pt'})")
    logf.close()
    metricsf.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--arm", choices=["ar", "jepa"], default="ar")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--compile", action="store_true", help="wrap model in torch.compile")
    ap.add_argument("--vicreg", type=float, default=None,
                    help="override jepa.vicreg_weight (anti-collapse experiment)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
