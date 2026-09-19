#!/bin/bash
# Launch pretraining/anneal on a single DGX Spark.
# Usage: bash scripts/train.sh configs/150m.yaml pretrain
set -euo pipefail
CFG=${1:-configs/150m.yaml}
STAGE=${2:-pretrain}
TOK=${TOK:-tokenizer/}
OUT=${OUT:-checkpoints/mllm-150m-pt}

# DGX Spark tuning: allow TF32, fast flash-SDPA, max threads
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export TOKENIZERS_PARALLELISM=false

python -m mllm.train --config "$CFG" --stage "$STAGE" --tok "$TOK" --out "$OUT" ${RESUME:+--resume "$RESUME"}
