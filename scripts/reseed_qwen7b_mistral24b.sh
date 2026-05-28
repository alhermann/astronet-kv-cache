#!/bin/bash
# Three-seed reseed to harden the three single-seed cells the reviewer
# critic flagged as sitting inside 1-2 binomial-SE:
#   1) k-confound diagnostic on Qwen 7B and Mistral-Small 24B
#   2) needle-in-a-haystack n=20 on Mistral-Small 24B
#
# Each run uses a unique seed_offset and save_suffix so existing
# canonical (seed_offset=0) result files are NOT overwritten.

set -uo pipefail  # no -e: a single failed run should not abort the rest
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

QWEN7B=./models/qwen2.5-7b
QWEN7B_CKPT=checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt

M24B=./models/mistral-small-24b
M24B_CKPT=checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42_w10_diverse.pt

mkdir -p logs/results logs/training

LOG=logs/training/reseed_$(date +%Y%m%d_%H%M%S).log
echo "[reseed] start $(date)" | tee -a "$LOG"

for SEED_IDX in 1 2 3; do
    SEED_OFF=$((SEED_IDX * 10000))
    SUFFIX="_seed${SEED_IDX}"

    # --- 1. Qwen 7B k-confound (attn_dim=512 for this checkpoint) ------
    echo "[reseed] Qwen 7B k-confound seed${SEED_IDX} offset=${SEED_OFF}" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/diag_mistral_hybrid.py \
        --model_path "$QWEN7B" \
        --hybrid_checkpoint "$QWEN7B_CKPT" \
        --n_windows 20 --n_trials 20 \
        --save_tag "qwen7b" \
        --attn_dim 512 \
        --seed_offset "$SEED_OFF" \
        --save_suffix "$SUFFIX" \
        --device cuda:0 2>&1 | tee -a "$LOG"

    # --- 2. Mistral-Small 24B k-confound (multi-GPU) -------------------
    echo "[reseed] Mistral-Small 24B k-confound seed${SEED_IDX} offset=${SEED_OFF}" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY baselines/diag_mistral_hybrid.py \
        --model_path "$M24B" \
        --hybrid_checkpoint "$M24B_CKPT" \
        --n_windows 20 --n_trials 20 \
        --save_tag "mistral24b" \
        --multi_gpu \
        --seed_offset "$SEED_OFF" \
        --save_suffix "$SUFFIX" 2>&1 | tee -a "$LOG"

    # --- 3. Mistral-Small 24B needle n=20 (multi-GPU) ------------------
    echo "[reseed] Mistral-Small 24B needle n=20 seed${SEED_IDX} offset=${SEED_OFF}" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY baselines/eval_needle.py \
        --model_path "$M24B" \
        --hybrid_checkpoint "$M24B_CKPT" \
        --k 300 \
        --n_windows_list 20 \
        --n_trials 20 \
        --methods multiplicative hybrid \
        --multi_gpu \
        --seed_offset "$SEED_OFF" \
        --save_suffix "${SUFFIX}_n20only" 2>&1 | tee -a "$LOG"
done

echo "[reseed] done $(date)" | tee -a "$LOG"
ls -la logs/results/diag_kconfound_qwen7b_n20_seed*.json \
       logs/results/diag_kconfound_mistral24b_n20_seed*.json \
       logs/results/needle_mistral-small-24b_k300_seed*.json 2>&1 | tee -a "$LOG"
