# CWM — Chess World Model

A JEPA-style latent world model for chess, built so its internal representation of the
board can be checked directly against the true board.

## Why chess

Chess dynamics are known and deterministic: the rules already give the exact next board, so
learning the transition function is pointless. The question worth asking is different — can a
model fed only a sequence of moves, trained only to predict the next one, build a correct
internal model of a board it is never shown?

The reason to want such a model is planning: to choose a move by imagining its consequences
and searching over those futures, an agent needs an internal world it can roll forward.
Autoregressive generation is the thin version of this — it looks ahead only by emitting the
next move and feeding it back. JEPA is the alternative: predict the future in latent space,
pair that with a latent transition, and you have a world you can roll forward and plan on.

As an *application*, JEPA is a poor fit for chess. Predicting in latent space is meant to let
a model ignore unpredictable low-level detail (pixels, textures) it would otherwise waste
capacity reconstructing; chess has no such detail — the state is small, discrete, and fully
described by tokens. What makes chess worth using is **verification**. In image or video JEPA
you cannot check whether a learned latent is correct: there is no ground-truth latent to
compare against. In chess the true state at every ply is a known board (`python-chess`
computes it exactly), so we can decode the model's latent, diff it against reality, and watch
whether that internal world stays faithful as the model runs forward on its own. The central
claim becomes directly measurable.

## The two arms (shared transformer backbone)

- **Autoregressive (AR) baseline (Chess-GPT)** — causal next-move prediction. The control
  and the harness validator: its hidden states must linearly probe to the board at ~99%
  before any JEPA result is trusted.
- **JEPA arm** *(Phase 2)* — **I-JEPA with a temporal mask, conditioned on the action.** A
  student encoder maps moves `m₁…mₜ → sₜ`; a predictor `g(sₜ, aₜ) → ŝₜ₊₁` advances the latent
  by the *withheld* next move `aₜ = mₜ₊₁`; a stop-grad **EMA teacher** (which saw `m₁…mₜ₊₁`)
  supplies the target `s̃ₜ₊₁`. The loss is computed in latent space; nothing is decoded back
  to moves. The predictor's form is MuZero/EfficientZero-style (action-conditioned latent
  dynamics, unrolled K steps); the objective is pure JEPA. None of MuZero's
  reward/value/policy/MCTS is kept.

The two arms look ahead in different spaces. An autoregressive model *generates*: it commits
to a move, appends the token, and re-encodes the whole sequence — lookahead in move space,
choosing its own moves, so its rollout wanders into a *different* game and what drifts is the
line of play, not its grip on the board. A JEPA model *predicts state*: its transition
`g(sₜ, aₜ) → ŝₜ₊₁` advances the latent directly, so you can drive it with the *real* moves and
check whether the predicted latent still matches the true board — isolating world-model drift
from move choice, and making latent-space search cheap where token regeneration is not.

## What we measure

The same probe harness scores both arms on the same held-out board labels:

1. **Fidelity** — does the latent linearly encode the true board? Target: ~99%.
2. **Linearity gap** — linear probe vs MLP probe. A small gap means the board is stored
   linearly rather than tangled.
3. **Rollout drift** *(Phase 3)* — advance the latent dynamics `k` steps using the real
   moves as actions (but never the real boards), decode, and compare to the true board at
   `t+k`. Accuracy against `k` shows how fast the internal board diverges from reality — a
   measurement chess makes possible and vision does not.

## Layout

```
cwm/moves.py          UCI move vocab (~1968 + specials); encode/decode; legal-move mask
cwm/data/             download (stream Lichess .pgn.zst), prepare (-> shards), dataset
cwm/model/gpt.py      shared backbone (causal | bidirectional), exposes hidden states
cwm/model/ar.py       AR head + next-move loss
cwm/model/jepa.py     JEPA arm  (Phase 2)
cwm/probe/boards.py   token seq -> ground-truth 13-class board labels (absolute/relative)
cwm/probe/probe.py    linear + MLP probes; majority-class baseline; per-square accuracy
cwm/probe/drift.py    latent rollout drift experiment  (Phase 3)
cwm/train.py          unified trainer (arm=ar|jepa)
configs/              dev_mps.yaml (laptop smoke test), small_gpu.yaml (real run)
```

## Quickstart

```bash
python -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt

# Phase 1, offline: random games + tiny model, validates the whole chain (minutes).
bash scripts/run_phase1.sh

# Phase 1, real: stream capped Lichess subset + train the small_gpu model (GPU).
MODE=real MONTH=2016-04 MAX_GAMES=200000 MIN_ELO=1600 bash scripts/run_phase1.sh
```

Boards are never fed to the model — the probe reconstructs them on demand, so the labels
are held out of the model's input.

## Results

Real Lichess run (2016-04, ~1M games, ~26M-param backbone). Board = layer-swept linear
probe (relative encoding), baseline 63.88%. **One-page synthesis:
[`results/SUMMARY.md`](results/SUMMARY.md)**; full write-ups in [`results/`](results/).

| model | best linear board | linearity gap | latent rollout (16 steps) |
|---|---|---|---|
| **AR baseline** (next-token) | **99.0%** | 0.06 | n/a (no latent dynamics fn) |
| JEPA (pure latent-prediction) | 80.3% | 7.1 | self-consistent (≈0 drift vs own encoder) |
| JEPA + contrastive objective | **90.4%** | 0.65 | — |

Findings:
- **AR builds the most accurate static board** — 99%, complete and linearly stored.
- **Pure JEPA is fuzzier** (80%, tangled). Its action-conditioned dynamics `g` rolls forward
  *self-consistently* (≈0 drift vs its own encoder, out to 4× its training horizon) — a
  faithfulness result, **not** a demonstrated edge over AR: driven by the same real moves, AR
  re-encoding holds a ~99% board vs JEPA's ~76% latent rollout. Whether latent rollout buys a
  *planning* advantage is untested (Phase 4).
- **The JEPA gap is largely an objective problem:** a leak-free *contrastive* target lifts the
  board to 90% with near-AR linear structure — a better latent objective, not move-grounding.

Detail: [`phase1-ar-2016-04.md`](results/phase1-ar-2016-04.md),
[`phase2-jepa-2016-04.md`](results/phase2-jepa-2016-04.md),
[`phase2-objectives.md`](results/phase2-objectives.md),
[`phase3-drift.md`](results/phase3-drift.md).

## Status

- [x] **Phase 1** — tokenizer, data pipeline, backbone, AR baseline (99.0% board probe), harness.
- [x] **Phase 2** — JEPA arm + AR-vs-JEPA comparison + objective ablations (contrastive → 90.4%).
- [x] **Phase 3** — rollout-drift experiment (JEPA dynamics ≈0 drift, self-consistent).
- [x] **Phase 4** — play/eval. AR plays real chess (95.8% vs random); the JEPA latent
      planner (material value, 1-ply) plays at ~random (47.2%) and loses 200–0 to AR.
      **Auditability ≠ actionability** — a 90% board probe doesn't make the world model
      usable for play; per-square errors compound into a noisy value (corr 0.44).
      See [`results/phase4-play.md`](results/phase4-play.md).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Covers move-vocab coverage over random games, shard→legal-game integrity, and exact
board-label agreement with python-chess.
