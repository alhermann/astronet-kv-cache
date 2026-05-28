#!/bin/bash
# Sequential retrain of all 5 remaining hybrid checkpoints with fixed S1.
# Run on cuda:0 in background; other tasks can use cuda:1.
set -e
cd .

# attn_dim=256 for all models below
echo "[$(date)] === Starting retrain queue ==="

# Mistral 7B (~1 hr)
echo "[$(date)] === Mistral 7B retrain ==="
PYTHONUNBUFFERED=1 "${PY:-python3}" training/train_hybrid.py \
  --model_path ./models/mistral-7b-v0.3 --n_train 5000 --epochs 3 --lr 5e-5 \
  --n_mem 16 --attn_dim 256 --k_real 284 --train_seed 42 --device cuda:0 \
  2>&1 | tee logs/training/hybrid_retrain_mistral7b.log

# Qwen 14B (~1.5 hr)
echo "[$(date)] === Qwen 14B retrain ==="
PYTHONUNBUFFERED=1 "${PY:-python3}" training/train_hybrid.py \
  --model_path ./models/qwen2.5-14b --n_train 5000 --epochs 3 --lr 5e-5 \
  --n_mem 16 --attn_dim 256 --k_real 284 --train_seed 42 --device cuda:0 \
  2>&1 | tee logs/training/hybrid_retrain_qwen14b.log

echo "[$(date)] === Retrain queue complete ==="
