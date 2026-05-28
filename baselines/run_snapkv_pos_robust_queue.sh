#!/bin/bash
# Sequential SnapKV pos-robust eval across single-GPU-fit models on cuda:0.
# Qwen 32B and Mistral 24B handled separately (need multi-GPU).
set -e
cd .

echo "[$(date)] === SnapKV pos-robust queue ==="

for MODEL in llama-3.1-8b mistral-7b-v0.3 qwen2.5-14b; do
  echo "[$(date)] === SnapKV pos-robust on $MODEL ==="
  PYTHONUNBUFFERED=1 "${PY:-python3}" baselines/eval_position_robustness.py \
    --model_path ./models/$MODEL \
    --n_samples 100 \
    --positions 0 1 2 3 \
    --methods snapkv multiplicative streaming_llm h2o \
    --device cuda:0 \
    --save_path logs/results/position_robustness_FAIR_${MODEL}.json \
    2>&1 | tee logs/training/snapkv_pos_robust_${MODEL}.log || echo "[$(date)] FAILED $MODEL, continuing"
done

echo "[$(date)] === SnapKV pos-robust queue complete ==="
