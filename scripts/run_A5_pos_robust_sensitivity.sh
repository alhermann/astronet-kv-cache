#!/bin/bash
# A5: single-seed (seed=42) pos-robust SQuAD eval at 4 positions, n=100 per
# position, for each of the four n=10 mixed sensitivity checkpoints.
#
# CRITICAL: --attn_dim 256 is mandatory (the _w10_diverse checkpoints all
# use attn_dim=256, but eval_hybrid_position_robust.py defaults to 512).
# Without this override, the eval would strict=False-load and silently
# zero out half the projection.

set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3

mkdir -p logs/training logs/results
LOG=logs/training/A5_pos_robust_sensitivity_$(date +%Y%m%d_%H%M%S).log
echo "[A5] start $(date)" | tee -a "$LOG"

run_one() {
    local TAG="$1" MODEL="$2" CKPT="$3" DEV="$4"
    local SAVE=logs/results/hybrid_pos_robust_n10mixed_${TAG}.json
    if [ -f "$SAVE" ]; then
        echo "[A5] $TAG already done -> $SAVE; skipping" | tee -a "$LOG"
        return 0
    fi
    if [ ! -f "$CKPT" ]; then
        echo "[A5] $TAG checkpoint missing: $CKPT; skipping" | tee -a "$LOG"
        return 1
    fi
    echo "[A5] $TAG start $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY training/eval_hybrid_position_robust.py \
        --model_path "$MODEL" \
        --checkpoint "$CKPT" \
        --n_eval 100 --seed 42 \
        --attn_dim 256 \
        --positions 0 1 2 3 \
        --device "$DEV" \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[A5] $TAG done $(date) -> $SAVE" | tee -a "$LOG"
}

# Qwen 7B and Llama 8B can both run on cuda:0 sequentially (single-card)
run_one "qwen7b"  ./models/qwen2.5-7b \
        checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt \
        cuda:0

run_one "llama8b" ./models/llama-3.1-8b \
        checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42_w10_diverse.pt \
        cuda:0

run_one "qwen14b" ./models/qwen2.5-14b \
        checkpoints/astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42_w10_diverse.pt \
        cuda:0

# Qwen 32B requires --multi_gpu
echo "[A5] qwen32b start $(date)" | tee -a "$LOG"
SAVE32=logs/results/hybrid_pos_robust_n10mixed_qwen32b.json
CKPT32=checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42_w10_diverse.pt
if [ -f "$SAVE32" ]; then
    echo "[A5] qwen32b already done; skipping" | tee -a "$LOG"
elif [ ! -f "$CKPT32" ]; then
    echo "[A5] qwen32b checkpoint missing: $CKPT32; skipping" | tee -a "$LOG"
else
    PYTHONUNBUFFERED=1 $PY training/eval_hybrid_position_robust.py \
        --model_path ./models/qwen2.5-32b \
        --checkpoint "$CKPT32" \
        --n_eval 100 --seed 42 \
        --attn_dim 256 \
        --positions 0 1 2 3 \
        --multi_gpu \
        --device cuda:0 \
        --save_path "$SAVE32" 2>&1 | tee -a "$LOG"
    echo "[A5] qwen32b done $(date) -> $SAVE32" | tee -a "$LOG"
fi

echo "[A5] all done $(date)" | tee -a "$LOG"
ls -la logs/results/hybrid_pos_robust_n10mixed_*.json | tee -a "$LOG"
