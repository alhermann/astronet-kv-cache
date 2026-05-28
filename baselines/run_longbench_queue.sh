#!/bin/bash
# Sequential LongBench re-eval with NEW S1 (fixed) + retrained S2 across single-GPU-fit models.
set -e
cd .

echo "[$(date)] === LongBench queue with fixed S1+S2 ==="

declare -A ATTN_DIMS=( ["qwen2.5-7b"]=512 ["qwen2.5-14b"]=256 ["llama-3.1-8b"]=256 ["mistral-7b-v0.3"]=256 )
declare -A SENSE_LAYERS=( ["qwen2.5-7b"]=14 ["qwen2.5-14b"]=24 ["llama-3.1-8b"]=16 ["mistral-7b-v0.3"]=16 )
declare -A CKPT_NAMES=( ["qwen2.5-7b"]="astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt" ["qwen2.5-14b"]="astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42.pt" ["llama-3.1-8b"]="astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42.pt" ["mistral-7b-v0.3"]="astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42.pt" )

for MODEL in qwen2.5-7b qwen2.5-14b llama-3.1-8b mistral-7b-v0.3; do
  echo "[$(date)] === LongBench on $MODEL ==="
  ATTN=${ATTN_DIMS[$MODEL]}
  SENSE=${SENSE_LAYERS[$MODEL]}
  CKPT=${CKPT_NAMES[$MODEL]}
  PYTHONUNBUFFERED=1 "${PY:-python3}" baselines/eval_longbench.py \
    --model_path ./models/$MODEL \
    --tasks hotpotqa multifieldqa_en \
    --k 300 --max_samples 100 --chunk_size 384 \
    --methods streaming_llm h2o snapkv multiplicative hybrid \
    --hybrid_checkpoint ./checkpoints/$CKPT \
    --sense_layer $SENSE --n_mem 16 --attn_dim $ATTN \
    --device cuda:0 \
    2>&1 | tee logs/training/longbench_FIXED_${MODEL}.log || echo "[$(date)] FAILED $MODEL, continuing"
done

echo "[$(date)] === LongBench queue complete ==="
