# Phase 2 — JEPA arm, run 1 (no VICReg)

Same backbone / data / probe as the AR baseline, trained with the pure JEPA objective
(action-conditioned latent prediction to an EMA teacher). **The JEPA arm builds a much
weaker world model than AR — and the run shows partial informational collapse.**

## Setup

- Data: Lichess `2016-04`, ~980k train / 20k val (identical shards to the AR run).
- Model: same ~26.5M backbone (student), + EMA teacher + action predictor. dropout 0.1.
- JEPA: `rollout_steps=4`, `ema_decay=0.999`, `predictor_layers=2`, `loss=cosine`,
  **`vicreg_weight=0.0`**.
- Training: bf16, 8 epochs / 20,408 steps, ~3.5 h. Cosine latent loss 0.637 → **0.0014**.

## Layer sweep (relative, val, 60k positions), vs AR

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 (final) |
|---|---|---|---|---|---|---|---|---|---|
| **JEPA** linear % | 73.97 | 76.26 | 76.26 | **77.06** | 76.75 | 76.53 | 75.84 | 75.60 | 72.95 |
| **AR** linear % | 84.70 | 89.81 | 91.59 | 94.31 | 95.28 | 96.29 | 98.24 | 98.94 | **99.01** |

| | AR | JEPA (run 1) |
|---|---|---|
| best linear probe | 99.01% (layer 8, final) | 77.06% (layer 3, mid) |
| MLP at best | 99.06% | 81.15% |
| linearity gap | 0.06 pts | 4.09 pts |
| layer profile | monotonic ↑, peaks at final | peaks mid-stack, **declines** to final |

JEPA's *best* layer (77%) is below AR's *worst* layer (84.7%).

## Diagnosis: partial collapse

`latent_std` fell monotonically the entire run: 0.918 → 0.835 → 0.757 → 0.695 → 0.646 →
0.611 → 0.584 → 0.566 → 0.554 → 0.545, while cosine loss went to ~0.0014. The model
minimized latent-prediction loss by making latents **smoother / lower-variance**, not more
board-faithful. In chess, consecutive states barely differ, so the next latent is easy to
predict from a compressed state — the objective never forces full board encoding, and the
model took the shortcut. EMA + predictor prevented *full* collapse (std ≠ 0) but not this
variance shrinkage — which is exactly what VICReg guards against, and it was off.

JEPA also pushes its best (weak) board rep to the **middle** layers and loses it toward the
final layer — the opposite of AR, whose next-token objective needs the board sharpest at the
end. Consistent with the two objectives.

## Next

- **Run 2: `--vicreg 0.1`** (and maybe 1.0) to fight the variance collapse. If the probe
  climbs toward 99%, run 1's weakness was mostly collapse; if it stays ~77%, it's a real
  limit of the objective. Either is a clean result.
- Regardless of static fidelity, JEPA has an explicit action-conditioned dynamics `g`, so
  **Phase 3 (rollout drift)** is where it may still win — worse static board, but slower
  drift over multi-step latent rollouts.

Artifacts: `metrics.csv`, `train.log`. Checkpoint kept on the server.
