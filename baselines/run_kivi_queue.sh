#!/bin/bash
# Sequential KIVI baseline on remaining 3 single-GPU-fit models on cuda:1.
# Qwen 32B and Mistral 24B require multi-GPU and run separately later.
set -e
cd .

echo "[$(date)] === KIVI baseline queue (cuda:1) ==="

for MODEL in qwen2.5-14b llama-3.1-8b mistral-7b-v0.3; do
  SHORT=$(echo $MODEL | sed 's/[.-]/_/g')
  echo "[$(date)] === KIVI on $MODEL ==="
  PYTHONUNBUFFERED=1 "${PY:-python3}" baselines/eval_kivi.py \
    --model_path ./models/$MODEL \
    --n_samples 200 --seed 42 --k 300 \
    --device cuda:0 \
    --save_path logs/results/kivi_${SHORT}.json \
    2>&1 | tee logs/training/kivi_${SHORT}.log || echo "[$(date)] FAILED $MODEL, continuing"
done

echo "[$(date)] === KIVI queue complete ==="
