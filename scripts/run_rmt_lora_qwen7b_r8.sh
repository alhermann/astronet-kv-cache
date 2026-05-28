#!/bin/bash
# Qwen 7B RMT-LoRA with LoRA rank=8 (half of r=16) — sweep upgrade 2.
# Same training corpus, same hyperparams except lora_r=8 -> 5.04M params
# (vs 10.1M at r=16). Defends the matched-budget claim: "even when LoRA
# is given fewer trainable parameters than AstroNet (5M vs ~14M),
# AstroNet still wins by some margin".

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

MODEL=./models/qwen2.5-7b
SAVE_DIR=checkpoints/rmt_lora_qwen7b_n16_r8_bptt2_t5000_s42

mkdir -p logs/training "$SAVE_DIR"
LOG=logs/training/rmt_lora_qwen7b_r8_full_$(date +%Y%m%d_%H%M%S).log
echo "[rmt-full-qwen7b-r8] start $(date)" | tee -a "$LOG"

PYTHONUNBUFFERED=1 $PY training/train_rmt_lora.py \
    --model_path "$MODEL" \
    --save_dir "$SAVE_DIR" \
    --n_mem 16 --n_windows 5 --n_train 5000 \
    --bptt_depth 2 --lr_lora 2e-4 --lr_mem 1e-4 \
    --epochs 3 --grad_accum 8 --warmup_steps 100 \
    --lora_r 8 --lora_alpha 16 \
    --watchdog_after_steps 500 --watchdog_mem_grad_min 1e-5 \
    --log_every 25 --save_every 2000 \
    --seed 42 --device cuda:0 2>&1 | tee -a "$LOG"

echo "[rmt-full-qwen7b-r8] done $(date)" | tee -a "$LOG"
