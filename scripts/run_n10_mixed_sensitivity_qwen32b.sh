#!/bin/bash
# n=10 mixed SQuAD/HotpotQA sensitivity check: Qwen 32B (multi-GPU, after all others done)
# Same recipe as the Mistral checkpoints; verifies the recipe does not
# degrade non-Mistral backbones, closing the "method shopping" objection.
#
# Requires both Titan RTX cards (PyTorch already enumerates them as
# cuda:0 and cuda:1; the GT 1030 display GPU is cuda:2 and is unused).

set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3

MODEL=./models/qwen2.5-32b
mkdir -p logs/training
LOG=logs/training/n10mixed_qwen32b_$(date +%Y%m%d_%H%M%S).log
echo "[n10mixed-qwen32b] start $(date)" | tee -a "$LOG"

PYTHONUNBUFFERED=1 $PY training/train_hybrid.py \
    --model_path "$MODEL" \
    --n_train 5000 --n_eval 50 \
    --epochs 3 --lr 5e-5 \
    --n_mem 16 --k_real 284 \
    --n_windows 10 \
    --diverse_training squad+hotpotqa \
    --train_seed 42 \
    --multi_gpu \
    --device cuda:0 2>&1 | tee -a "$LOG"

echo "[n10mixed-qwen32b] done $(date)" | tee -a "$LOG"
echo "Expected checkpoint: checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42_w10_diverse.pt"
