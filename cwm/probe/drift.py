"""Phase 3 — latent rollout drift.

Tests the JEPA arm's *dynamics*, not just its static representation. From every position t
we take the student's true latent ``s_t`` and roll the predictor ``g`` forward k steps on the
**real** moves (never re-reading the true board), decode each predicted latent through a
frozen board probe, and diff against the true board at ``t+k``. Plotting accuracy vs k shows
how fast the model's internal board drifts from reality when it runs on its own.

Design choices (intentional):
  * the probe decodes in the space ``g`` predicts — the **final-layer** latent (``layer=-1``),
    so the k=0 point of the drift curve equals the static final-layer probe;
  * the rollout is driven by the real actions (the played moves), isolating *world-model
    drift* from move choice;
  * an **oracle** line re-encodes the true prefix at each step (no drift, upper bound); the
    gap between the rollout curve and the oracle is the drift.

AR has no latent dynamics ``g`` to roll, so this is a JEPA-intrinsic measurement (an AR
self-rollout would generate its own moves and measure policy divergence — a different
question). Runs only on a JEPA checkpoint.

Usage:
    python -m cwm.probe.drift --checkpoint checkpoints/jepa_vicreg/model_best.pt \\
        --data-dir data/lichess --k-max 16 --out results/phase3-drift.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cwm import moves  # noqa: E402
from cwm.data.dataset import GameDataset  # noqa: E402
from cwm.probe import boards  # noqa: E402
from cwm.probe.probe import collect_features, fit_probe, load_model  # noqa: E402
from cwm.utils.config import pick_device  # noqa: E402


def _label_tensor(input_ids_cpu, lengths, relative):
    """(B,T,64) relative board labels aligned to sequence index; -1 where padded."""
    B, T = input_ids_cpu.shape
    labels = np.full((B, T, boards.NUM_SQUARES), -1, dtype=np.int64)
    for b in range(B):
        pos, lab = boards.game_labels(
            input_ids_cpu[b, :lengths[b]].numpy(), relative=relative, skip_start=False
        )
        labels[b, pos] = lab
    return torch.from_numpy(labels)


def _accum(correct, total, k, decoder, latents, labels, mask):
    B, n, C = latents.shape
    pred = decoder(latents.reshape(B * n, C)).argmax(-1).reshape(B, n, -1)  # (B,n,64)
    m = mask.unsqueeze(-1)  # (B,n,1)
    correct[k] += ((pred == labels) & m).sum().item()
    total[k] += m.expand_as(pred).sum().item()


@torch.no_grad()
def drift_eval(model, decoder, dataset, device, k_max, relative, max_positions, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    kc = k_max + 1
    correct_roll, total_roll = np.zeros(kc), np.zeros(kc)
    correct_oracle, total_oracle = np.zeros(kc), np.zeros(kc)
    seen = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["length"].tolist()
        T = input_ids.shape[1]
        K = min(k_max, T - 1)
        n = T - K
        s_full = model.encode(input_ids)  # (B,T,C) final-layer latents
        labels = _label_tensor(input_ids.cpu(), lengths, relative).to(device)
        real = input_ids != moves.PAD_ID

        s0 = s_full[:, :n, :]
        # k=0: decode the true start latent (rollout and oracle coincide).
        _accum(correct_roll, total_roll, 0, decoder, s0, labels[:, :n], real[:, :n])
        correct_oracle[0], total_oracle[0] = correct_roll[0], total_roll[0]

        h = s0
        for k in range(1, K + 1):
            h = model.step(h, input_ids[:, k:k + n])  # roll g on the real action at t+k
            tgt_lab, tgt_real = labels[:, k:k + n], real[:, k:k + n]
            _accum(correct_roll, total_roll, k, decoder, h, tgt_lab, tgt_real)
            _accum(correct_oracle, total_oracle, k, decoder,
                   s_full[:, k:k + n], tgt_lab, tgt_real)  # oracle: true latent, no drift
        seen += real[:, :n].sum().item()
        if seen >= max_positions:
            break

    roll = correct_roll / np.maximum(total_roll, 1)
    oracle = correct_oracle / np.maximum(total_oracle, 1)
    return roll, oracle


def run(args):
    device = pick_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    if ckpt["arm"] != "jepa":
        raise SystemExit("drift needs a JEPA checkpoint (it rolls the latent dynamics g)")
    ctx = ckpt["model_cfg"]["ctx"]
    rel = not args.absolute

    # Fit the board decoder on final-layer latents (the space g predicts).
    fit_ds = GameDataset(args.data_dir, args.fit_split, ctx=ctx)
    X, Y = collect_features(model, fit_ds, device, layer=-1, relative=rel,
                            max_positions=args.fit_positions)
    decoder = fit_probe(X, Y, "linear", device, epochs=args.epochs)
    print(f"decoder fit on {len(X)} final-layer positions")

    eval_ds = GameDataset(args.data_dir, args.split, ctx=ctx)
    roll, oracle = drift_eval(model, decoder, eval_ds, device, args.k_max, rel,
                              args.max_positions, args.batch_size)

    print(f"{'k':>3}  {'rollout':>8}  {'oracle':>8}  {'drift':>7}")
    for k in range(len(roll)):
        print(f"{k:>3}  {roll[k]*100:7.2f}%  {oracle[k]*100:7.2f}%  {(oracle[k]-roll[k])*100:6.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = np.arange(len(roll))
    ax.plot(ks, oracle * 100, "--", color="#16a34a", label="oracle (re-encode true prefix)")
    ax.plot(ks, roll * 100, "-o", color="#dc2626", ms=3, label="latent rollout g^k")
    ax.set_xlabel("rollout steps k")
    ax.set_ylabel("board accuracy (%)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="val", help="split to measure drift on")
    ap.add_argument("--fit-split", default="train", help="split to fit the decoder on")
    ap.add_argument("--k-max", type=int, default=16)
    ap.add_argument("--fit-positions", type=int, default=60000)
    ap.add_argument("--max-positions", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--absolute", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="results/phase3-drift.png")
    ap.add_argument("--title", default="JEPA latent rollout drift")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
