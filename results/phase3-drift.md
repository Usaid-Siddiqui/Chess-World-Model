# Phase 3 — latent rollout drift (JEPA, VICReg model)

Tests JEPA's *dynamics*: from each position roll the predictor `g` k steps in latent space
on the **real** moves (never re-reading the board), decode each predicted latent with a
frozen final-layer probe, diff vs the true board at t+k. Compared to an **oracle** that
re-encodes the true prefix each step (no drift, upper bound).

Checkpoint: `checkpoints/jepa_vicreg/model_best.pt` (healthy `latent_std=1.0`, so results are
not a collapse artifact). Trained with `rollout_steps=4`; measured out to k=16.

## Curve (val, relative, decoder fit on train final-layer latents)

| k | 0 | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| rollout % | 76.60 | 77.47 | 77.39 | 76.83 | 75.53 | 74.77 | 74.22 |
| oracle %  | 76.60 | 77.50 | 77.22 | 76.55 | 75.15 | 74.38 | 73.84 |
| drift (oracle−rollout) | 0.00 | 0.03 | −0.18 | −0.28 | −0.38 | −0.39 | −0.38 |

- **k=0 anchor holds** (rollout == oracle = static final-layer probe).
- Both curves decay slowly *together* — a position-difficulty effect (deeper plies), which
  the oracle controls for.
- **Drift ≈ 0 across all 16 steps** (slightly negative, within noise). Rolling `g` forward
  **4× its training horizon loses essentially nothing** — the latent dynamics is faithful.

## Reading

`g` correctly applies each action's transition (rollout accuracy tracks the *evolving* true
board via the oracle, not a frozen start board), and it stays consistent far beyond the K=4
it was trained on. Caveat: it faithfully rolls a **mediocre** representation — it keeps the
~76% board at ~76%, it does not recover the ~20 pts JEPA lost to AR statically.

## Three-phase verdict

| Axis | AR (next-token) | JEPA (latent, VICReg) |
|---|---|---|
| static linear probe | **99.01%** (final layer) | 80.27% (mid-stack) |
| static MLP | **99.06%** | 87.39% |
| board storage | complete & linear (gap 0.06) | partial & tangled (gap 7.12) |
| latent dynamics `g` | none | yes |
| rollout drift (16 steps) | n/a | **≈0 (lossless)** |

Different kinds of world model: next-token prediction gives the **most accurate static
board** but nothing to roll; latent-prediction gives a **fuzzier board** but a **stable,
rollable dynamics** (the planning-relevant property AR can't offer). Chess let us measure
both against ground truth exactly.

## Optional follow-up

Run drift on the collapsed `checkpoints/jepa` (no-VICReg): if it is *also* ≈0 drift, some
flatness is degenerate (a low-variance latent is trivially predictable); contrast confirms
the VICReg model's ≈0 drift is a real dynamics property, not degeneracy.

Plot: `results/phase3-drift-jepa-vicreg.png`.
