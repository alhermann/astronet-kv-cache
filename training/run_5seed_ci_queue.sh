#!/bin/bash
# Sequential 5-seed pos-robust CIs on Qwen 7B with retrained S2.
# Seed 42 already done -> needs 7, 123, 999, 2024 to make 5 seeds.
set -e
cd .

echo "[$(date)] === 5-seed CI queue (Qwen 7B, retrained S2) ==="

for SEED in 7 123 999 2024; do
  echo "[$(date)] === seed=$SEED ==="
  PYTHONUNBUFFERED=1 "${PY:-python3}" training/eval_hybrid_position_robust.py \
    --model_path ./models/qwen2.5-7b \
    --checkpoint ./checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt \
    --n_eval 100 --seed $SEED --attn_dim 512 \
    --device cuda:0 \
    --save_path logs/results/hybrid_pos_robust_RETRAINED_qwen7b_s${SEED}.json \
    2>&1 | tee logs/training/pos_robust_qwen7b_s${SEED}.log || echo "[$(date)] FAILED seed=$SEED"
done

echo "[$(date)] === 5-seed CI queue complete ==="
