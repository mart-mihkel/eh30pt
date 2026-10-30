#!/usr/bin/env bash

DATA_PATH="./log/labels-all-dedup.parquet"
DATASET="discont-binary/extract-labels"
SAMPLES="100 500 1000 1500 2000 2500"

EVAL_STEPS=epoch
BATCH_SIZE=8
WARMUP=0.06
PATIENCE=4
EPOCHS=20

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
  -wr "$WARMUP"
  -pat "$PATIENCE"
  -mex "$ML_EXP"
)

ENCODER_MODELS=(
  "./models/mmBERT-base"
)

DECODER_MODELS=(
  "./models/medgemma-4b-it"
  "./models/Qwen3.5-9B"
)

for MODEL in "${ENCODER_MODELS[@]}"; do
  for SAMP in $SAMPLES; do
    ehpt fine-tune   -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 5e-4 -ho
    ehpt fine-tune   -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 2e-5
    ehpt prompt-tune -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 1e-3
    # ehpt qlora     -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 2e-4
  done
done

for MODEL in "${DECODER_MODELS[@]}"; do
  ehpt few-shot -n 5 -m "$MODEL" "${FS_FLAGS[@]}"
  for SAMP in $SAMPLES; do
    ehpt fine-tune   -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 5e-5
    ehpt prompt-tune -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 1e-2
    # ehpt qlora     -m "$MODEL" "${FLAGS[@]}" -t "$SAMP" -lr 2e-4
  done
done
