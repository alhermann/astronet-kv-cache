#!/bin/bash
# Option C: re-evaluate the ORIGINAL AstroNet Mistral 7B checkpoint that
# was trained on n=5 segments + SQuAD-only (no _w10_diverse suffix).
# This MATCHES the training distribution of RMT-LoRA Mistral 7B and the
# eval distribution (5-segment SQuAD), giving an apples-to-apples
# comparison alongside the existing tab:squad_main number (which uses
# the n=10 mixed-corpus retrain for that backbone).
#
# Outputs:
#   logs/results/hybrid_pos_robust_mistral-7b-v0_3_n5squad_s{42,7,123,999,2024}.json
#
# These are NOT a substitute for the tab:squad_main 57.0% number; they
# are a separate, footnoted "matched training-distribution" comparison.

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

MODEL=./models/mistral-7b-v0.3
# Mistral 7B has hidden_size=4096, so attn_dim=256 (matches existing
# AstroNet Mistral 7B eval convention used elsewhere in the paper).
ATTN_DIM=256
CKPT=./checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42.pt
TAG=mistral-7b-v0_3_n5squad

mkdir -p logs/results logs/training
LOG=logs/training/astronet_eval_${TAG}_$(date +%Y%m%d_%H%M%S).log
echo "[astro-eval-${TAG}] start $(date)" | tee -a "$LOG"
echo "[astro-eval-${TAG}] checkpoint: $CKPT" | tee -a "$LOG"

for SEED in 42 7 123 999 2024; do
    SAVE=logs/results/hybrid_pos_robust_${TAG}_s${SEED}.json
    if [ -f "$SAVE" ]; then
        echo "[astro-eval-${TAG}] seed=$SEED already done, skipping." | tee -a "$LOG"
        continue
    fi
    echo "[astro-eval-${TAG}] seed=$SEED start $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY training/eval_hybrid_position_robust.py \
        --model_path "$MODEL" \
        --checkpoint "$CKPT" \
        --n_eval 100 \
        --seed "$SEED" \
        --k_real 284 \
        --n_mem 16 \
        --attn_dim $ATTN_DIM \
        --positions 0 1 2 3 \
        --device cuda:1 \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[astro-eval-${TAG}] seed=$SEED done $(date) -> $SAVE" | tee -a "$LOG"
done

echo "[astro-eval-${TAG}] all done $(date)" | tee -a "$LOG"
ls -la logs/results/hybrid_pos_robust_${TAG}_s*.json | tee -a "$LOG"
