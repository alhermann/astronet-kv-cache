#!/bin/bash
# 5-seed pos-robust CIs on Mistral 7B with retrained S2.
set -e
cd .

echo "[$(date)] === 5-seed CI Mistral 7B (retrained S2) ==="

for SEED in 7 123 999 2024; do
  echo "[$(date)] === Mistral 7B seed=$SEED ==="
  PYTHONUNBUFFERED=1 "${PY:-python3}" training/eval_hybrid_position_robust.py \
    --model_path ./models/mistral-7b-v0.3 \
    --checkpoint ./checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42.pt \
    --n_eval 100 --seed $SEED --attn_dim 256 \
    --device cuda:0 \
    --save_path logs/results/hybrid_pos_robust_RETRAINED_mistral7b_s${SEED}.json \
    2>&1 | tee logs/training/pos_robust_mistral7b_s${SEED}.log || echo "[$(date)] FAILED seed=$SEED"
done

echo "[$(date)] === 5-seed CI Mistral 7B complete ==="
