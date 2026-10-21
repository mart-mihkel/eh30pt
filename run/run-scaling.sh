#!/usr/bin/env bash

DATA_PATH="./log/labels-all-dedup.parquet"
DATASET="discont-binary/extract-labels"
SAMPLES="100 500 1000 2500"

EVAL_STEP=epoch
EPOCHS=1000
PATIENCE=4

ML_EXP="eh30pt-data-scaling"
SEED=42

# Decoders
DECODER_MODELS="./models/medgemma-4b-it ./models/Llama-3.1-8B-Instruct ./models/Llama-3.1-EstLLM-8B-Instruct-1125 ./models/Qwen3.5-9B"
BATCH_SIZE=2

for MODEL in $DECODER_MODELS; do
  ehpt few-shot   -n 5 -m "$MODEL" -d $DATASET -dp $DATA_PATH -b $BATCH_SIZE                                                     -mex $ML_EXP -s $SEED
  for SAMP in $SAMPLES; do
    ehpt qlora         -m "$MODEL" -d $DATASET -dp $DATA_PATH -b $BATCH_SIZE -e $EPOCHS -t "$SAMP" -es $EVAL_STEP -pat $PATIENCE -mex $ML_EXP -s $SEED
    ehpt fine-tune     -m "$MODEL" -d $DATASET -dp $DATA_PATH -b $BATCH_SIZE -e $EPOCHS -t "$SAMP" -es $EVAL_STEP -pat $PATIENCE -mex $ML_EXP -s $SEED
    ehpt prompt-tune   -m "$MODEL" -d $DATASET -dp $DATA_PATH -b $BATCH_SIZE -e $EPOCHS -t "$SAMP" -es $EVAL_STEP -pat $PATIENCE -mex $ML_EXP -s $SEED
  done
done

# Encoders
ENCODER_MODELS="./models/mmBERT-base"
BATCH_SIZE=16

for MODEL in $ENCODER_MODELS; do
  for SAMP in $SAMPLES; do
    ehpt qlora         -m "$MODEL" -d $DATASET -dp $DATA_PATH -b $BATCH_SIZE -e $EPOCHS -t "$SAMP" -es $EVAL_STEP -pat $PATIENCE -mex $ML_EXP -s $SEED
    ehpt fine-tune -ho -m "$MODEL" -d $DATASET -dp $DATA_PATH -b $BATCH_SIZE -e $EPOCHS -t "$SAMP" -es $EVAL_STEP -pat $PATIENCE -mex $ML_EXP -s $SEED
    ehpt fine-tune     -m "$MODEL" -d $DATASET -dp $DATA_PATH -b $BATCH_SIZE -e $EPOCHS -t "$SAMP" -es $EVAL_STEP -pat $PATIENCE -mex $ML_EXP -s $SEED
    ehpt prompt-tune   -m "$MODEL" -d $DATASET -dp $DATA_PATH -b $BATCH_SIZE -e $EPOCHS -t "$SAMP" -es $EVAL_STEP -pat $PATIENCE -mex $ML_EXP -s $SEED
  done
done
