"""Plot train vs val loss from a run's metrics.csv.

Reads the ``step,train_loss,val_loss,lr,elapsed_s`` file the trainer writes and produces a
loss-vs-step PNG — the quickest way to see whether a run is healthy (train and val fall
together) or overfitting (val turns up while train keeps dropping). Headless-safe (Agg
backend), so it runs on a cluster with no display.

Usage:
    python scripts/plot_metrics.py --metrics checkpoints/ar/metrics.csv \\
        --out results/phase1-ar-2016-04/loss.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402


def load_metrics(path: Path):
    steps, train, val = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]))
    return steps, train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="checkpoints/ar/metrics.csv")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside metrics.csv)")
    ap.add_argument("--title", default="training")
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    steps, train, val = load_metrics(metrics_path)
    if not steps:
        raise SystemExit(f"no rows in {metrics_path}")
    out = Path(args.out) if args.out else metrics_path.with_name("loss.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    best_i = min(range(len(val)), key=lambda i: val[i])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, train, label="train", color="#2563eb", marker="o", ms=3)
    ax.plot(steps, val, label="val", color="#dc2626", marker="o", ms=3)
    ax.scatter([steps[best_i]], [val[best_i]], color="#dc2626", zorder=5,
               label=f"best val {val[best_i]:.3f} @ {steps[best_i]}")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (cross-entropy)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}  ({len(steps)} points, best val {val[best_i]:.4f} @ step {steps[best_i]})")


if __name__ == "__main__":
    main()
