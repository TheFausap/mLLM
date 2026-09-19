#!/bin/bash
# Run the full no-API eval suite (+ optional MT-Bench-lite).
# Usage: bash scripts/eval.sh checkpoints/mllm-150m-sft
set -euo pipefail
CKPT=${1:?"checkpoint dir required"}
TOK=${TOK:-tokenizer/}
OUT=${OUT:-eval_results.json}

python -m mllm.eval --checkpoint "$CKPT" --tok "$TOK" --out "$OUT" --mtbench-lite ${JUDGE:+--judge "$JUDGE"}
