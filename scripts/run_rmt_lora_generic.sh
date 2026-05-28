#!/bin/bash
# Generic RMT-LoRA training wrapper. Trains the recurrent-memory + LoRA
# baseline on any backbone supported by transformers' 4-bit NF4 path.
#
# Usage:
#   ./scripts/run_rmt_lora_generic.sh <model_path> <save_tag> <cuda_device>
#
# Example:
#   ./scripts/run_rmt_lora_generic.sh ./models/llama-3.1-8b llama8b cuda:0
#
# Outputs:
#   checkpoints/rmt_lora_<save_tag>_n16_r16_bptt2_t5000_s42/{final.pt, lora_final/}
#   logs/training/rmt_lora_<save_tag>_full_<timestamp>.log

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model_path> <save_tag> <cuda_device>"
    exit 2
fi

MODEL_PATH="$1"
TAG="$2"
DEVICE="$3"

cd "$(dirname "$0")/.."
PY="${PY:-python3}"

SAVE_DIR=checkpoints/rmt_lora_${TAG}_n16_r16_bptt2_t5000_s42
mkdir -p logs/training "$SAVE_DIR"

LOG=logs/training/rmt_lora_${TAG}_full_$(date +%Y%m%d_%H%M%S).log
echo "[rmt-full-${TAG}] start $(date)" | tee -a "$LOG"
echo "[rmt-full-${TAG}] model=$MODEL_PATH device=$DEVICE save_dir=$SAVE_DIR" | tee -a "$LOG"

PYTHONUNBUFFERED=1 $PY training/train_rmt_lora.py \
    --model_path "$MODEL_PATH" \
    --save_dir "$SAVE_DIR" \
    --n_mem 16 --n_windows 5 --n_train 5000 \
    --bptt_depth 2 --lr_lora 2e-4 --lr_mem 1e-4 \
    --epochs 3 --grad_accum 8 --warmup_steps 100 \
    --lora_r 16 --lora_alpha 32 \
    --watchdog_after_steps 500 --watchdog_mem_grad_min 1e-5 \
    --log_every 25 --save_every 2000 \
    --seed 42 \
    --device "$DEVICE" 2>&1 | tee -a "$LOG"

echo "[rmt-full-${TAG}] done $(date)" | tee -a "$LOG"
