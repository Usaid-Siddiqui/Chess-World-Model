# results/

A tracked record of runs — the small, valuable artifacts, **not** model weights.

Commit here:
- run summaries (`*.md`),
- `metrics.csv` (step, train_loss, val_loss, lr, elapsed),
- `train.log`,
- plots (`*.png` — allowed under `results/` despite the global ignore).

Do **not** commit here:
- checkpoints (`model*.pt`, ~100MB) — keep on the server / object storage / git-LFS,
- tokenized shards (`*.bin`) or raw data.

Convention: one subfolder per run, e.g. `results/phase1-ar-2016-04/`.
