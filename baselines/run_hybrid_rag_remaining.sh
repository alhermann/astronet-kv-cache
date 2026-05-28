#!/usr/bin/env bash
# Hybrid+RAG ensemble on LongBench HotpotQA for the remaining 4 backbones,
# chained after the running Qwen 7B + Mistral 7B queue releases cuda:1.
# The poll matches actual Python eval processes (not bash wrappers) to avoid
# self-reference deadlock.
set -eo pipefail
cd .
PY="${PY:-python3}"
SCRIPT=baselines/eval_hybrid_rag_longbench.py
LOGDIR=logs/hybrid_rag_v8
mkdir -p $LOGDIR

CKPT_L8B=./checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42.pt
CKPT_Q14B=./checkpoints/astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42.pt
CKPT_Q32B=./checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42.pt
CKPT_M24B=./checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42_w10_diverse.pt

echo "[$(date)] Starting remaining 4 backbones (GPU should already be free)."

echo "[$(date)] === Stage 1: Llama 3.1-8B (DONE on prior run; skipping) ==="
# Llama 8B result already saved in logs/results/hybrid_rag_longbench_hotpotqa_llama8b_v8.json

echo "[$(date)] === Stage 2: Qwen 2.5-14B (multi-GPU, ~1.5 h) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/qwen2.5-14b --multi_gpu \
  --hybrid_checkpoint $CKPT_Q14B \
  --task hotpotqa --n_samples 50 --k 300 --attn_dim 256 \
  --save_path logs/results/hybrid_rag_longbench_hotpotqa_qwen14b_v8.json \
  2>&1 | tee $LOGDIR/qwen14b.log

echo "[$(date)] === Stage 3: Qwen 2.5-32B (multi-GPU, ~2-3 h) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/qwen2.5-32b --multi_gpu \
  --hybrid_checkpoint $CKPT_Q32B \
  --task hotpotqa --n_samples 50 --k 300 --attn_dim 256 \
  --save_path logs/results/hybrid_rag_longbench_hotpotqa_qwen32b_v8.json \
  2>&1 | tee $LOGDIR/qwen32b.log

echo "[$(date)] === Stage 4: Mistral-Small 24B retrained (multi-GPU, ~3-4 h) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/mistral-small-24b --multi_gpu \
  --hybrid_checkpoint $CKPT_M24B \
  --task hotpotqa --n_samples 50 --k 300 --attn_dim 256 \
  --save_path logs/results/hybrid_rag_longbench_hotpotqa_mistral24b_v8.json \
  2>&1 | tee $LOGDIR/mistral24b.log

echo "[$(date)] === Remaining hybrid+RAG queue complete ==="
ls -la logs/results/hybrid_rag_longbench_hotpotqa_*_v8.json
