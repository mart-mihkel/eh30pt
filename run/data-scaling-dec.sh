#!/usr/bin/env bash

DATA_PATH="./log/labels-all-dedup.parquet"
DATASET="discont-binary/extract-labels"
SAMPLES="100 900 1800 2700"

EVAL_STEPS=epoch
BATCH_SIZE=8
EPOCHS=50

TOLERANCE=0.05
PATIENCE=5

ML_EXP="eh30pt-data-scaling"

FS_FLAGS=(
  -d "$DATASET"
  -dp "$DATA_PATH"
  -b "$BATCH_SIZE"
  -mex "$ML_EXP"
)

FLAGS=(
  -d "$DATASET"
  -dp "$DATA_PATH"
  -b "$BATCH_SIZE"
  -e "$EPOCHS"
  -es "$EVAL_STEPS"
  -pat "$PATIENCE"
  -tol "$TOLERANCE"
  -mex "$ML_EXP"
)

MODELS=(
  "./models/medgemma-4b-it"
  "./models/Qwen3.5-9B"
)

for MODEL in "${MODELS[@]}"; do
  ehpt few-shot -n 5 -m "$MODEL" "${FS_FLAGS[@]}"
done

for MODEL in "${MODELS[@]}"; do
  for SAMP in $SAMPLES; do
    ehpt fine-tune -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 5e-5 -wr 0.05
  done
done

for MODEL in "${MODELS[@]}"; do
  for SAMP in $SAMPLES; do
    ehpt prompt-tune -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 1e-2 --save-model
  done
done
