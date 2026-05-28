#!/bin/bash
# Pos-robust n=100 eval on 5 remaining v2 retrained models.
# Qwen 32B + Mistral 24B = multi-GPU; Qwen 14B + Llama 8B + Mistral 7B = single-GPU.
# Run multi-GPU jobs first (need both GPUs), then single-GPU.
set -e
cd .

LOGDIR=logs/training
RESDIR=logs/results

# --- Multi-GPU first (need both cuda:0 + cuda:1) ---
for SPEC in \
  "qwen32b ./models/qwen2.5-32b ./checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42.pt --multi_gpu" \
  "mistral24b ./models/mistral-small-24b ./checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42.pt --multi_gpu" ; do
  set -- $SPEC
  TAG=$1; MODEL=$2; CKPT=$3; FLAGS=$4
  echo "[$(date)] === pos-robust v2 $TAG (multi-GPU) ==="
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
    "${PY:-python3}" training/eval_hybrid_position_robust.py \
      --model_path $MODEL --checkpoint $CKPT \
      --n_eval 100 --seed 42 --attn_dim 256 $FLAGS \
      --save_path $RESDIR/hybrid_pos_robust_v2_${TAG}.json \
      2>&1 | tee $LOGDIR/pos_robust_v2_${TAG}.log || echo "[$(date)] FAILED $TAG"
done

# --- Single-GPU sequentially (override the device with CUDA_VISIBLE_DEVICES) ---
for SPEC in \
  "qwen14b ./models/qwen2.5-14b ./checkpoints/astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42.pt" \
  "llama8b ./models/llama-3.1-8b ./checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42.pt" \
  "mistral7b ./models/mistral-7b-v0.3 ./checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42.pt" ; do
  set -- $SPEC
  TAG=$1; MODEL=$2; CKPT=$3
  echo "[$(date)] === pos-robust v2 $TAG (single-GPU cuda:0) ==="
  PYTHONUNBUFFERED=1 \
    "${PY:-python3}" training/eval_hybrid_position_robust.py \
      --model_path $MODEL --checkpoint $CKPT \
      --n_eval 100 --seed 42 --attn_dim 256 \
      --device cuda:0 \
      --save_path $RESDIR/hybrid_pos_robust_v2_${TAG}.json \
      2>&1 | tee $LOGDIR/pos_robust_v2_${TAG}.log || echo "[$(date)] FAILED $TAG"
done

echo "[$(date)] === pos-robust v2 queue COMPLETE ==="
