#!/bin/bash
# Reseed the remaining 4 cells of tab:kconfound for visual consistency:
#   Qwen 14B, Qwen 32B (multi-GPU), Llama 8B, Mistral 7B
# Three seeds each (offsets 10000/20000/30000), unique save_suffix per
# seed so canonical files are NOT overwritten.

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

Q14B=./models/qwen2.5-14b
Q14B_CKPT=checkpoints/astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42.pt

Q32B=./models/qwen2.5-32b
Q32B_CKPT=checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42.pt

L8B=./models/llama-3.1-8b
L8B_CKPT=checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42.pt

M7B=./models/mistral-7b-v0.3
M7B_CKPT=checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42_w10_diverse.pt

mkdir -p logs/results logs/training
LOG=logs/training/reseed_kconfound_remaining_$(date +%Y%m%d_%H%M%S).log
echo "[reseed-rem] start $(date)" | tee -a "$LOG"

for SEED_IDX in 1 2 3; do
    SEED_OFF=$((SEED_IDX * 10000))
    SUFFIX="_seed${SEED_IDX}"

    # --- Qwen 14B (single GPU) -----------------------------------------
    echo "[reseed-rem] Qwen 14B kconfound seed${SEED_IDX}" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/diag_mistral_hybrid.py \
        --model_path "$Q14B" --hybrid_checkpoint "$Q14B_CKPT" \
        --n_windows 20 --n_trials 20 \
        --save_tag "qwen14b" \
        --seed_offset "$SEED_OFF" --save_suffix "$SUFFIX" \
        --device cuda:0 2>&1 | tee -a "$LOG"

    # --- Qwen 32B (multi-GPU) ------------------------------------------
    echo "[reseed-rem] Qwen 32B kconfound seed${SEED_IDX} (multi-GPU)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY baselines/diag_mistral_hybrid.py \
        --model_path "$Q32B" --hybrid_checkpoint "$Q32B_CKPT" \
        --n_windows 20 --n_trials 20 \
        --save_tag "qwen32b" --multi_gpu \
        --seed_offset "$SEED_OFF" --save_suffix "$SUFFIX" 2>&1 | tee -a "$LOG"

    # --- Llama 8B (single GPU) -----------------------------------------
    echo "[reseed-rem] Llama 8B kconfound seed${SEED_IDX}" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/diag_mistral_hybrid.py \
        --model_path "$L8B" --hybrid_checkpoint "$L8B_CKPT" \
        --n_windows 20 --n_trials 20 \
        --save_tag "llama8b" \
        --seed_offset "$SEED_OFF" --save_suffix "$SUFFIX" \
        --device cuda:0 2>&1 | tee -a "$LOG"

    # --- Mistral 7B (single GPU, retrained checkpoint) -----------------
    echo "[reseed-rem] Mistral 7B kconfound seed${SEED_IDX}" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/diag_mistral_hybrid.py \
        --model_path "$M7B" --hybrid_checkpoint "$M7B_CKPT" \
        --n_windows 20 --n_trials 20 \
        --save_tag "mistral7b" \
        --seed_offset "$SEED_OFF" --save_suffix "$SUFFIX" \
        --device cuda:0 2>&1 | tee -a "$LOG"
done

echo "[reseed-rem] done $(date)" | tee -a "$LOG"
ls -la logs/results/diag_kconfound_{qwen14b,qwen32b,llama8b,mistral7b}_n20_seed*.json 2>&1 | tee -a "$LOG"
