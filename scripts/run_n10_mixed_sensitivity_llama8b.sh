#!/bin/bash
# n=10 mixed SQuAD/HotpotQA sensitivity check: Llama 8B (cuda:1, parallel with Qwen 7B)
# Same recipe as the Mistral checkpoints; verifies the recipe does not
# degrade non-Mistral backbones, closing the "method shopping" objection.

set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3

MODEL=./models/llama-3.1-8b
mkdir -p logs/training
LOG=logs/training/n10mixed_llama8b_$(date +%Y%m%d_%H%M%S).log
echo "[n10mixed-llama8b] start $(date)" | tee -a "$LOG"

PYTHONUNBUFFERED=1 $PY training/train_hybrid.py \
    --model_path "$MODEL" \
    --n_train 5000 --n_eval 50 \
    --epochs 3 --lr 5e-5 \
    --n_mem 16 --k_real 284 \
    --n_windows 10 \
    --diverse_training squad+hotpotqa \
    --train_seed 42 \
    --device cuda:1 2>&1 | tee -a "$LOG"

echo "[n10mixed-llama8b] done $(date)" | tee -a "$LOG"
echo "Expected checkpoint: checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42_w10_diverse.pt"
