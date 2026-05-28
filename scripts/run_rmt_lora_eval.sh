#!/bin/bash
# Five-seed × four-position RMT-LoRA evaluation, matching tab:squad_main
# protocol used for AstroNet S1+S2 on Qwen 7B.
#
# Outputs five JSONs:
#   logs/results/rmt_lora_pos_robust_qwen2_5-7b_s{7,42,123,999,2024}.json
# Each contains pos_0..pos_3 plus average. Aggregate offline for tab:squad_main.

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

MODEL=./models/qwen2.5-7b
LORA_DIR=./checkpoints/rmt_lora_qwen7b_n16_r16_bptt2_t5000_s42/lora_final
MEM_CKPT=./checkpoints/rmt_lora_qwen7b_n16_r16_bptt2_t5000_s42/final.pt

mkdir -p logs/results logs/training
LOG=logs/training/rmt_lora_eval_$(date +%Y%m%d_%H%M%S).log
echo "[rmt-lora-eval] start $(date)" | tee -a "$LOG"

for SEED in 42 7 123 999 2024; do
    SAVE=logs/results/rmt_lora_pos_robust_qwen2_5-7b_s${SEED}.json
    if [ -f "$SAVE" ]; then
        echo "[rmt-lora-eval] seed=$SEED already done ($SAVE), skipping." | tee -a "$LOG"
        continue
    fi
    echo "[rmt-lora-eval] seed=$SEED start $(date)" | tee -a "$LOG"
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
    echo "[rmt-lora-eval] seed=$SEED done $(date) -> $SAVE" | tee -a "$LOG"
done

echo "[rmt-lora-eval] all done $(date)" | tee -a "$LOG"
ls -la logs/results/rmt_lora_pos_robust_qwen2_5-7b_s*.json | tee -a "$LOG"
