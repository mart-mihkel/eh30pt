#!/usr/bin/env bash

DATA_PATH="./log/labels-all-dedup.parquet"
DATASET="discont-binary/extract-labels"
SAMPLES="100 900 1800 2700"

EVAL_STEPS=epoch
BATCH_SIZE=8
EPOCHS=50

PATIENCE=10

ML_EXP="eh30pt-data-scaling"

FLAGS=(
  -d "$DATASET"
  -dp "$DATA_PATH"
  -b "$BATCH_SIZE"
  -e "$EPOCHS"
  -es "$EVAL_STEPS"
  -pat "$PATIENCE"
  -mex "$ML_EXP"
)

MODELS=(
  "./models/mmBERT-base"
)

for MODEL in "${MODELS[@]}"; do
  for SAMP in $SAMPLES; do
    ehpt fine-tune -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 5e-4 -ho
    ehpt fine-tune -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 2e-5 -wr 0.05
    ehpt prompt-tune -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 1e-3 --save-model
  done
done
