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

## Run 2 — VICReg 0.1 (`--vicreg 0.1`)

VICReg **fixed the collapse**: `latent_std` held at **1.001** all run (vs 0.545). Cosine loss
settled higher (~0.032 vs 0.0014) — expected, since the low-variance shortcut is now blocked.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| linear % | 75.64 | 78.51 | 80.14 | **80.27** | 79.88 | 79.74 | 79.39 | 78.96 | 79.02 |

Best layer 3: linear **80.27%**, MLP **87.39%**, gap **7.12 pts**.

## Verdict (three-way)

| | AR | JEPA v1 (no VICReg) | JEPA v2 (VICReg 0.1) |
|---|---|---|---|
| latent_std (end) | — | 0.545 (collapsing) | 1.001 (healthy) |
| best linear | **99.01%** (L8) | 77.06% (L3) | 80.27% (L3) |
| MLP at best | 99.06% | 81.15% | 87.39% |
| linearity gap | 0.06 | 4.09 | 7.12 |

- **The gap is the objective, not collapse.** With collapse eliminated (std 1.0), JEPA still
  tops out ~80% linear / 87% MLP vs AR's 99%/99%. VICReg bought only +3 linear / +6 MLP.
- **VICReg → more present but more tangled**: MLP up to 87% (more board info recoverable),
  yet the linearity gap *widened* to 7.12 (less of it linearly accessible). AR's board is
  complete *and* linear; JEPA's is partial *and* tangled.
- **JEPA peaks mid-stack (layer 3)**, fading toward the final layer — opposite of AR.

**Conclusion:** on chess, next-token prediction builds a cleaner, more complete, more linear
internal board than pure latent-prediction — the strong form of "chess is a bad application
of JEPA," demonstrated with the ground-truth microscope after ruling out collapse.

## Next — Phase 3 (the fair rematch)

Static fidelity is only half the story. JEPA has an explicit action-conditioned dynamics `g`
that AR lacks. Phase 3 (`cwm/probe/drift.py`): unroll `g` k steps in latent space on real
actions, decode via the frozen probe, diff vs the true board at t+k. JEPA may drift *slower*
than an AR self-rollout even though its static board is worse — that's the property it was
actually built for, and where it could still win.

Artifacts (both runs): `metrics.csv`, `train.log`. Checkpoints kept on the server.
