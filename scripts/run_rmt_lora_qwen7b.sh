#!/bin/bash
# Full RMT-LoRA training on Qwen 2.5-7B (~14h).
# Hyperparameters from the smoke-test-validated config:
#   --bptt_depth 2  (avoids 5-segment BPTT explosion)
#   --lr_mem 1e-4   (avoids early-step gradient blow-up)
#   --bf16 compute (avoids fp16 mem-init overflow)
# Wait for any running smoke test to finish first.

set -uo pipefail
cd .
PY="${PY:-python3}"

# Wait for the smoke test process to exit if still running.
SMOKE_LOG=$(ls -t logs/training/rmt_lora_smoketest*.log 2>/dev/null | head -1)
if [ -n "${SMOKE_LOG:-}" ]; then
    echo "[rmt-full] waiting for smoke test ($SMOKE_LOG) to finish..."
    until ! pgrep -f "train_rmt_lora.py.*_smoketest" >/dev/null; do
        sleep 30
    done
    echo "[rmt-full] smoke test done, starting full run."
fi

LOG=logs/training/rmt_lora_qwen7b_full_$(date +%Y%m%d_%H%M%S).log
echo "[rmt-full] start $(date)" | tee "$LOG"
PYTHONUNBUFFERED=1 $PY -u training/train_rmt_lora.py \
  --model_path ./models/qwen2.5-7b \
  --save_dir   ./checkpoints/rmt_lora_qwen7b_n16_r16_bptt2_t5000_s42 \
  --n_mem 16 --n_windows 5 --n_train 5000 \
  --bptt_depth 2 --lr_lora 2e-4 --lr_mem 1e-4 \
  --epochs 3 --grad_accum 8 --warmup_steps 100 \
  --lora_r 16 --lora_alpha 32 \
  --watchdog_after_steps 500 --watchdog_mem_grad_min 1e-5 \
  --log_every 25 --save_every 2000 \
  --seed 42 --device cuda:0  2>&1 | tee -a "$LOG"
echo "[rmt-full] done $(date)" | tee -a "$LOG"
