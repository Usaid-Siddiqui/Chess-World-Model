# CWM — study summary

Using chess's **ground-truth board** as a microscope (the true board is known every ply, so
a learned latent can be audited by linear probing — impossible in vision), we compared what
internal world model different training objectives build, and whether it is useful. One
shared ~26M transformer backbone; real Lichess data (2016-04, ~1M games). Board = layer-swept
linear probe, relative encoding, baseline 63.9%.

## The five results

1. **AR (next-token) builds a near-perfect board — and plays.** 99.0% linear probe, gap 0.06,
   sharpening monotonically to the final layer. 95.8% vs random.

2. **Pure JEPA (latent-prediction) is much fuzzier.** 77–80% linear, gap ~7, peaking mid-stack.
   Collapse (fixed by VICReg) was minor; the *objective* was the cause.

3. **The gap is largely an objective-design problem.** A leak-free **contrastive** target
   reaches **90.4%, gap 0.65** (near-AR linear structure) — a better *latent* objective, not
   move-grounding. Inverse dynamics also helped (84%) but **leaks** in a causal token model
   (reads the last move off the after-state), so it is a confounded hybrid.

4. **JEPA's latent dynamics is self-consistent, not superior.** Rolling `g` on the real moves
   drifts ≈0 out to 4× its training horizon — but this is faithfulness vs its own encoder, not
   an edge over AR (AR re-encoding holds a ~99% board vs JEPA's ~76% latent rollout).

5. **Auditability ≠ actionability.** AR plays real chess; the JEPA latent planner (value +
   1-ply lookahead) plays at random (47% vs random) and loses 200–0 to AR. **Why:** value
   aggregates over the board, so per-square errors compound — AR's 99% board → value corr
   0.90, JEPA's 90% board → value corr 0.44. The modest board gap becomes a large value gap,
   and planning starves.

## Takeaways

- **Discriminative latent objectives capture far more world-state than regressive ones**
  (contrastive 90% vs cosine 80%), measured against ground truth — a statement vision/robotics
  make on faith but can't test directly.
- **A high per-element probe score does not imply usefulness.** 90% vs 99% board looks close,
  but errors compound in aggregates (value, policy), so the fuzzier world model can't be
  planned into competent play. Next-token prediction's edge here isn't exotic — it builds a
  board accurate enough that sums over it stay clean.

## Details
`phase1-ar-2016-04.md` · `phase2-jepa-2016-04.md` · `phase2-objectives.md` ·
`phase3-drift.md` · `phase4-play.md`

## Open (would-be a separate project)
Making JEPA actually play is MuZero-scale: outcome value (data re-prep for game results), a
policy prior (JEPA has none), MCTS, and a `g` validated **off-distribution** (drift was only
measured on real game lines) — on a value foundation that is already shaky at a 90% board.
