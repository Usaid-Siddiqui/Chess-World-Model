# Phase 2b — JEPA objective ablations

Can a better *objective* force the strong board that pure JEPA (80%) missed? Same backbone,
data, and probe as everything else; only the JEPA training objective changes. Compared by
the layer-swept linear board probe (relative encoding, val, 60k positions; baseline 63.88%).

| variant | best linear | MLP | gap | peak layer | clean? |
|---|---|---|---|---|---|
| **AR** (control) | **99.01%** | 99.06% | 0.06 | 8 (final) | — |
| JEPA — cosine (no VICReg) | 77.06% | 81.15% | 4.09 | 3 | collapsed (latent_std 0.55) |
| JEPA — cosine + VICReg | 80.27% | 87.39% | 7.12 | 3 | ✓ |
| JEPA — + inverse dynamics | 83.89% | 88.00% | 4.11 | 6 | ✗ leaky (inv_acc 1.0) |
| **JEPA — contrastive (InfoNCE)** | **90.44%** | 91.09% | **0.65** | 7 | ✓ leak-free |

## What each objective did

- **cosine / VICReg** — the original pure JEPA. Regressing to the EMA teacher is an easy,
  low-entropy target; the board comes out partial (80%) and tangled (gap 7). VICReg only
  fixes collapse, not fidelity.
- **inverse dynamics** (`--inverse 1.0`) — predict the move `a_t` from the latent pair
  `(s_{t-1}, s_t)`. Reached 83.9%, but `inv_acc` pinned at **1.0**: the head reads the last
  move off `s_t` (residual-stream shortcut) rather than diffing boards, so the gain is a
  confounded side effect of move-grounding, not clean board pressure. It also predicts a
  move token — reintroducing AR's own signal — so it is a hybrid, not "JEPA."
- **contrastive** (`--loss contrastive`) — action-InfoNCE: among candidate actions applied
  to `s_{t-1}` via `g`, the true one's predicted latent must land nearest the true `s_t`.
  The query is `s_{t-1}` (before the move) so there is **no last token to read off** —
  leak-free. Best result by a wide margin: **90.44% linear, gap 0.65**, peak pushed to the
  final layers, an AR-like organization.

## Verdict

The pure-JEPA weakness was **largely fixable, and the honest fix won.** Making the target
*discriminative* (contrastive) — not regressive — jumped the board 80 → 90% and collapsed
the linearity gap 7.1 → 0.65 (approaching AR's 0.06), **leak-free**. This partly overturns
the "AR's edge is just move-grounding" reading: a better *latent-space* objective recovers
most of the gap without predicting moves. Notably the clean contrastive objective (90.4%)
**beat** the leaky inverse hybrid (83.9%).

Still ~9 points short of AR's 99%: latent-prediction gets close but doesn't fully match
next-token prediction for static board fidelity on chess.

Runs on the cluster: `checkpoints/jepa_inv`, `checkpoints/jepa_con` (both 20,408 steps).
