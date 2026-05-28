#!/bin/bash
# Sequential Needle re-eval on multi-GPU models (Qwen 32B + Mistral 24B).
set -e
cd .

echo "[$(date)] === Needle multi-GPU queue with fixed S1+retrained S2 ==="

declare -A ATTN_DIMS=( ["qwen2.5-32b"]=256 ["mistral-small-24b"]=256 )
declare -A SENSE_LAYERS=( ["qwen2.5-32b"]=32 ["mistral-small-24b"]=20 )
declare -A CKPT_NAMES=( ["qwen2.5-32b"]="astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42.pt" ["mistral-small-24b"]="astro_hybrid_mistral-small-24b_n16_k284_t5000_s42.pt" )

for MODEL in qwen2.5-32b mistral-small-24b; do
  SHORT=$(echo $MODEL | sed 's/[.-]/_/g')
  echo "[$(date)] === Needle on $MODEL (multi-GPU) ==="
  ATTN=${ATTN_DIMS[$MODEL]}
  SENSE=${SENSE_LAYERS[$MODEL]}
  CKPT=${CKPT_NAMES[$MODEL]}
  CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 "${PY:-python3}" baselines/eval_needle.py \
    --model_path ./models/$MODEL \
    --k 300 --n_trials 20 --n_windows_list 5 10 20 \
    --methods streaming_llm h2o snapkv multiplicative hybrid \
    --hybrid_checkpoint ./checkpoints/$CKPT \
    --sense_layer $SENSE --n_mem 16 --attn_dim $ATTN \
    --multi_gpu \
    2>&1 | tee logs/training/needle_FIXED_${SHORT}.log || echo "[$(date)] FAILED $MODEL, continuing"
done

echo "[$(date)] === Needle multi-GPU queue complete ==="
