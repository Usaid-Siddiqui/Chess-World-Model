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
# Examples:
#   bash scripts/run_phase1.sh
#   MODE=real MONTH=2016-04 MAX_GAMES=200000 MIN_ELO=1600 bash scripts/run_phase1.sh
set -euo pipefail

# Interpreter: defaults to the local venv, override for containers (e.g. PY=python).
PY="${PY:-.venv/bin/python}"
MODE="${MODE:-dev}"

if [[ "$MODE" == "dev" ]]; then
  DATA_DIR="${DATA_DIR:-data/dev}"
  CONFIG="${CONFIG:-configs/dev_mps.yaml}"
  OUT="${OUT:-checkpoints/dev_ar}"
  echo "== [dev] preparing random games =="
  $PY -m cwm.data.prepare --source random --out-dir "$DATA_DIR" \
      --num-random "${NUM_RANDOM:-3000}" --min-plies 10 --max-plies 80 --val-frac 0.05
else
  DATA_DIR="${DATA_DIR:-data/lichess}"
  CONFIG="${CONFIG:-configs/small_gpu.yaml}"
  OUT="${OUT:-checkpoints/ar}"
  echo "== [real] streaming Lichess ${MONTH:?set MONTH=YYYY-MM} =="
  $PY -m cwm.data.prepare --source lichess --month "$MONTH" --out-dir "$DATA_DIR" \
      --min-elo "${MIN_ELO:-1600}" --min-plies 10 --max-plies "${MAX_PLIES:-512}" \
      --max-games "${MAX_GAMES:-200000}" --val-frac 0.02
fi

echo "== training AR baseline =="
$PY -m cwm.train --config "$CONFIG" --arm ar --data-dir "$DATA_DIR" --out "$OUT"

echo "== probing board state =="
$PY -m cwm.probe.probe --checkpoint "$OUT/model.pt" --data-dir "$DATA_DIR" \
    --split val --layer -1 --max-positions "${MAX_POS:-60000}"
