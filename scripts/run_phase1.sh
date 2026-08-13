#!/usr/bin/env bash
# Phase 1 end-to-end: data -> AR baseline -> board probe.
#
# MODE=dev (default): offline, random games, tiny model — validates the whole chain on
#   a laptop in minutes. The probe should beat the majority-class baseline by a wide
#   margin (real board state is present), but will NOT hit ~99% (random games, tiny model).
#
# MODE=real: streams a capped subset of a Lichess month, trains the small_gpu model.
#   This is the run that reproduces the ~99% linear board probe. Needs network + a GPU.
#
# Prepare is skipped when the shards already exist; set FORCE=1 to re-tokenize.
#
# Examples:
#   bash scripts/run_phase1.sh
#   MODE=real MONTH=2016-04 MIN_ELO=1600 bash scripts/run_phase1.sh   # MAX_GAMES defaults to 1M
set -euo pipefail

# Interpreter: defaults to the local venv, override for containers (e.g. PY=python).
PY="${PY:-.venv/bin/python}"
MODE="${MODE:-dev}"

if [[ "$MODE" == "dev" ]]; then
  DATA_DIR="${DATA_DIR:-data/dev}"
  CONFIG="${CONFIG:-configs/dev_mps.yaml}"
  OUT="${OUT:-checkpoints/dev_ar}"
  SRC_DESC="random games"
  prepare_data() {
    $PY -m cwm.data.prepare --source random --out-dir "$DATA_DIR" \
        --num-random "${NUM_RANDOM:-3000}" --min-plies 10 --max-plies 80 --val-frac 0.05
  }
else
  DATA_DIR="${DATA_DIR:-data/lichess}"
  CONFIG="${CONFIG:-configs/small_gpu.yaml}"
  OUT="${OUT:-checkpoints/ar}"
  SRC_DESC="Lichess ${MONTH:?set MONTH=YYYY-MM}"
  prepare_data() {
    $PY -m cwm.data.prepare --source lichess --month "$MONTH" --out-dir "$DATA_DIR" \
        --min-elo "${MIN_ELO:-1600}" --min-plies 10 --max-plies "${MAX_PLIES:-512}" \
        --max-games "${MAX_GAMES:-1000000}" --val-frac 0.02
  }
fi

# Tokenization is deterministic (month + filters + vocab), so reuse existing shards.
if [[ -f "$DATA_DIR/meta.json" && "${FORCE:-0}" != "1" ]]; then
  echo "== data exists at $DATA_DIR, skipping prepare (FORCE=1 to rebuild) =="
else
  echo "== preparing $SRC_DESC -> $DATA_DIR =="
  prepare_data
fi

echo "== training AR baseline =="
$PY -m cwm.train --config "$CONFIG" --arm ar --data-dir "$DATA_DIR" --out "$OUT"

echo "== probing board state =="
CKPT="$OUT/model_best.pt"; [[ -f "$CKPT" ]] || CKPT="$OUT/model.pt"
$PY -m cwm.probe.probe --checkpoint "$CKPT" --data-dir "$DATA_DIR" \
    --split val --layer -1 --max-positions "${MAX_POS:-60000}"
