#!/usr/bin/env bash

DATA_PATH="./log/labels-all-dedup.parquet"
DATASET="discont-binary/extract-labels"

EVAL_STEP=400
BATCH_SIZE=2
EPOCHS=1

ML_EXP="eh30pt-lr-search"

LEARNING_RATES=(
  0.3
  0.1
  0.05
  0.01
  0.001
)

# Decoders
DECODER_MODELS=(
  "./models/medgemma-4b-it"
  "./models/Llama-3.2-1B-Instruct"
  "./models/Llama-3.1-EstLLM-8B-Instruct-1125"
  "./models/Qwen3.5-9B"
)

for MODEL in "${DECODER_MODELS[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    ehpt prompt-tune -m "$MODEL" -d $DATASET -dp $DATA_PATH \
                     -b $BATCH_SIZE -e $EPOCHS -lr "$LR" -es $EVAL_STEP \
                     -mex $ML_EXP
  done
done

# Encoders
ENCODER_MODELS=(
  "./models/mmBERT-base"
)

for MODEL in "${ENCODER_MODELS[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    ehpt prompt-tune -m "$MODEL" -d $DATASET -dp $DATA_PATH \
                     -b $BATCH_SIZE -e $EPOCHS -lr "$LR" -es $EVAL_STEP \
                     -mex $ML_EXP
  done
done
