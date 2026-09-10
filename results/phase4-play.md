# Phase 4 — play / usefulness

Does the emergent world model let you *act*? AR plays by its native move policy (legal-masked
argmax). JEPA has no policy, so it plays by 1-ply latent lookahead: roll `g` one step for each
legal move, score the resulting latent with a value head, pick the best. 200-game
color-balanced matches (`jepa_con` is the 90%-board contrastive model).

Value head on `jepa_con` (material target): val **MSE 12.4 pawns², corr 0.441**.

| matchup | W–D–L | score |
|---|---|---|
| AR vs random | 183–17–0 | **95.8%** |
| JEPA planner vs random | 6–177–17 | **47.2%** |
| AR vs JEPA planner | 200–0–0 | **100.0%** |

## Verdict

**Auditability ≠ actionability.** AR plays genuine chess (96% vs random); the JEPA latent
planner plays at random level (47%) and loses every one of 200 games to AR. A representation
that *probes* the board at 90% is not thereby *usable* for play through this scheme.

## Why — errors compound into the value

The bottleneck is the value readout (**corr 0.441**, despite a 90% board probe). Material is a
weighted **sum over 64 squares**, so per-square probe errors *accumulate*: at 90% per-square
(~6 wrong squares/position) the material estimate is very noisy; AR's 99% (~0.6 wrong) would
give a clean value. **The 9-point board gap (99 vs 90) becomes a large value-quality gap for
any aggregate quantity** — so a "90% board" is far less useful for planning than a "99% board."
The planner then ranks moves by a noisy signal → near-random play (shuffling → 177 draws).
1-ply material-greedy is also a weak strategy even with a perfect value.

## Caveats / what this does and doesn't settle

This is a verdict on **material value + 1-ply latent lookahead**, not latent planning in
general. Open upgrades: a stronger **outcome value** (needs the data re-prep to store game
results); a **policy prior + deeper MuZero-style search**. But deeper search would *amplify*
the weak value, so value quality must be fixed first — and whether `g` stays faithful
**off-distribution** (down hypothetical search lines, untested) gates deep search entirely.

Artifacts: `checkpoints/{ar,jepa_con}` + `checkpoints/jepa_con/value_head.pt` on the cluster.
