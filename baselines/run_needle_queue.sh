#!/bin/bash
# Sequential Needle re-eval with retrained S2 across all 4 single-GPU-fit models on cuda:0.
# Qwen 32B and Mistral 24B require multi-GPU and run separately later.
set -e
cd .

echo "[$(date)] === Needle queue with fixed S1+retrained S2 ==="

declare -A ATTN_DIMS=( ["qwen2.5-7b"]=512 ["qwen2.5-14b"]=256 ["llama-3.1-8b"]=256 ["mistral-7b-v0.3"]=256 )
declare -A SENSE_LAYERS=( ["qwen2.5-7b"]=14 ["qwen2.5-14b"]=24 ["llama-3.1-8b"]=16 ["mistral-7b-v0.3"]=16 )
declare -A CKPT_NAMES=( ["qwen2.5-7b"]="astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt" ["qwen2.5-14b"]="astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42.pt" ["llama-3.1-8b"]="astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42.pt" ["mistral-7b-v0.3"]="astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42.pt" )

for MODEL in qwen2.5-7b qwen2.5-14b llama-3.1-8b mistral-7b-v0.3; do
  SHORT=$(echo $MODEL | sed 's/[.-]/_/g')
  echo "[$(date)] === Needle on $MODEL ==="
  ATTN=${ATTN_DIMS[$MODEL]}
  SENSE=${SENSE_LAYERS[$MODEL]}
  CKPT=${CKPT_NAMES[$MODEL]}
  PYTHONUNBUFFERED=1 "${PY:-python3}" baselines/eval_needle.py \
    --model_path ./models/$MODEL \
    --k 300 --n_trials 20 --n_windows_list 5 10 20 \
    --methods streaming_llm h2o snapkv multiplicative hybrid \
    --hybrid_checkpoint ./checkpoints/$CKPT \
    --sense_layer $SENSE --n_mem 16 --attn_dim $ATTN \
    --device cuda:0 \
    2>&1 | tee logs/training/needle_FIXED_${SHORT}.log || echo "[$(date)] FAILED $MODEL, continuing"
done

echo "[$(date)] === Needle queue complete ==="
