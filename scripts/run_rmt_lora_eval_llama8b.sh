#!/bin/bash
# 5-seed x 4-position eval for the Llama 8B RMT-LoRA checkpoint.
# Mirrors scripts/run_rmt_lora_eval.sh (the Qwen 7B version).

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

MODEL=./models/llama-3.1-8b
LORA_DIR=./checkpoints/rmt_lora_llama8b_n16_r16_bptt2_t5000_s42/lora_final
MEM_CKPT=./checkpoints/rmt_lora_llama8b_n16_r16_bptt2_t5000_s42/final.pt
TAG=llama-3_1-8b

mkdir -p logs/results logs/training
LOG=logs/training/rmt_lora_eval_${TAG}_$(date +%Y%m%d_%H%M%S).log
echo "[rmt-lora-eval-${TAG}] start $(date)" | tee -a "$LOG"

for SEED in 42 7 123 999 2024; do
    SAVE=logs/results/rmt_lora_pos_robust_${TAG}_s${SEED}.json
    if [ -f "$SAVE" ]; then
        echo "[rmt-lora-eval-${TAG}] seed=$SEED already done, skipping." | tee -a "$LOG"
        continue
    fi
    echo "[rmt-lora-eval-${TAG}] seed=$SEED start $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY training/eval_rmt_lora_position_robust.py \
        --model_path "$MODEL" \
        --lora_dir "$LORA_DIR" \
        --mem_ckpt "$MEM_CKPT" \
        --n_mem 16 --n_windows 5 \
        --max_seg_tokens 256 --max_new_tokens 20 \
        --n_eval 100 --seed "$SEED" \
        --positions 0 1 2 3 \
        --device cuda:0 \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[rmt-lora-eval-${TAG}] seed=$SEED done $(date) -> $SAVE" | tee -a "$LOG"
done

echo "[rmt-lora-eval-${TAG}] all done $(date)" | tee -a "$LOG"
ls -la logs/results/rmt_lora_pos_robust_${TAG}_s*.json | tee -a "$LOG"
