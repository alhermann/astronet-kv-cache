#!/usr/bin/env bash
# Full-FP16 KV cache reference eval across all 6 backbones.
# Each run produces logs/results/full_fp16_<model>.json with per-position accuracy
# (fact placed at non-query window 0/1/2/3).
#
# Smaller models run on cuda:1 (single GPU), Qwen 32B and Mistral 24B need --multi_gpu.
set -e

cd .
PY="${PY:-python3}"
SCRIPT=baselines/eval_full_fp16.py
LOGDIR=logs/full_fp16
mkdir -p "$LOGDIR"

echo "=== Stage 1: Qwen 7B (cuda:1, ~25 min) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/qwen2.5-7b \
  --device cuda:1 \
  --n_eval 100 --seed 42 2>&1 | tee "$LOGDIR/qwen7b.log"

echo "=== Stage 2: Llama 8B (cuda:1, ~25 min) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/llama-3.1-8b \
  --device cuda:1 \
  --n_eval 100 --seed 42 2>&1 | tee "$LOGDIR/llama8b.log"

echo "=== Stage 3: Mistral 7B (cuda:1, ~25 min) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/mistral-7b-v0.3 \
  --device cuda:1 \
  --n_eval 100 --seed 42 2>&1 | tee "$LOGDIR/mistral7b.log"

echo "=== Stage 4: Qwen 14B (cuda:1, ~40 min) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/qwen2.5-14b \
  --device cuda:1 \
  --n_eval 100 --seed 42 2>&1 | tee "$LOGDIR/qwen14b.log"

echo "=== Stage 5: Qwen 32B (multi-GPU, ~60 min) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/qwen2.5-32b \
  --multi_gpu \
  --n_eval 100 --seed 42 2>&1 | tee "$LOGDIR/qwen32b.log"

echo "=== Stage 6: Mistral 24B (multi-GPU, ~50 min) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/mistral-small-24b \
  --multi_gpu \
  --n_eval 100 --seed 42 2>&1 | tee "$LOGDIR/mistral24b.log"

echo
echo "=== All Full-FP16 evals complete ==="
ls -la logs/results/full_fp16_*.json
