# CWM — Chess World Model

A JEPA-style **latent world model** for chess, built so its internal world can be
**measured against ground truth**.

## Why chess

Chess dynamics are known and deterministic, so *learning* the transition function is
pointless. The interesting object is the **emergent latent world model**: a network that
only ever sees a serialized stream of moves (never a board, never the rules) must
internally reconstruct the 8×8 board to do its job.

The decisive point: **chess is a bad application of JEPA but an ideal test rig for it.**
In image/video JEPA you can never verify whether a learned latent is "correct" — there is
no ground-truth latent. In chess the true world state at every ply is a known board
(`python-chess` gives it exactly), so we can **audit the latent directly**: decode it and
diff against reality. Vague claims about "abstract representations" become hard numbers.

## The two arms (shared transformer backbone)

- **AR baseline (Chess-GPT)** — causal next-move prediction. The control and harness
  validator: its hidden states must linearly probe to the board at ~99% before any JEPA
  result is trusted.
- **JEPA arm** *(Phase 2)* — **I-JEPA with a temporal mask, conditioned on the action.**
  A student encoder maps moves `m₁…mₜ → sₜ`; a predictor `g(sₜ, aₜ) → ŝₜ₊₁` advances the
  latent by the *withheld* next move `aₜ = mₜ₊₁`; a stop-grad **EMA teacher** (which saw
  `m₁…mₜ₊₁`) supplies the target `s̃ₜ₊₁`. Loss lives in latent space; nothing is decoded
  back to moves. The predictor's form is MuZero/EfficientZero-style (action-conditioned
  latent dynamics, unrolled K steps); the objective is pure JEPA. None of MuZero's
  reward/value/policy/MCTS is kept.

## What we measure (the microscope)

1. **Fidelity** — does the latent linearly encode the true board? (target: ~99%)
2. **Linearity gap** — linear vs MLP probe; a small gap = the board is stored *linearly*.
3. **Rollout drift** *(Phase 3)* — unroll the latent dynamics `k` steps on real actions
   (never real boards), decode, diff against the true board at `t+k`. The drift-vs-`k`
   curve is the headline result, only ground-truthable because this is chess.

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

Boards are **never fed to the model** — the probe reconstructs them on demand, so labels
are guaranteed held out of the model's input.

## Status

- [x] **Phase 1** — move tokenizer, data pipeline, backbone, AR baseline, probe harness
      (validated end-to-end; dev run beats the no-information baseline by a wide margin).
- [ ] **Phase 2** — JEPA arm + AR-vs-JEPA fidelity / linearity comparison.
- [ ] **Phase 3** — rollout-drift experiment + comparison plots.

A play/eval module (legal-move-masked sampling + Stockfish eval) is a clean later drop-in:
tokenization is move-level and `cwm.moves.legal_move_mask` already exists, so no changes
to the model, data, or tokenizer are needed.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Covers: move-vocab coverage over random games, shard→legal-game integrity, and exact
board-label agreement with python-chess.
