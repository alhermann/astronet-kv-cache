#!/bin/bash
# n=10 mixed SQuAD/HotpotQA sensitivity check: Qwen 7B (cuda:0)
# Same recipe as the Mistral checkpoints; verifies the recipe does not
# degrade non-Mistral backbones, closing the "method shopping" objection.

set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3

MODEL=./models/qwen2.5-7b
mkdir -p logs/training
LOG=logs/training/n10mixed_qwen7b_$(date +%Y%m%d_%H%M%S).log
echo "[n10mixed-qwen7b] start $(date)" | tee -a "$LOG"

PYTHONUNBUFFERED=1 $PY training/train_hybrid.py \
    --model_path "$MODEL" \
    --n_train 5000 --n_eval 50 \
    --epochs 3 --lr 5e-5 \
    --n_mem 16 --k_real 284 \
    --n_windows 10 \
    --diverse_training squad+hotpotqa \
    --train_seed 42 \
    --device cuda:0 2>&1 | tee -a "$LOG"

echo "[n10mixed-qwen7b] done $(date)" | tee -a "$LOG"
echo "Expected checkpoint: checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt"
