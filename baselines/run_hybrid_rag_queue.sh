#!/usr/bin/env bash
# Hybrid+RAG ensemble on LongBench HotpotQA, two backbones.
set -e
cd .
PY="${PY:-python3}"
SCRIPT=baselines/eval_hybrid_rag_longbench.py
LOGDIR=logs/hybrid_rag_v8
mkdir -p $LOGDIR

CKPT_Q7B=./checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt
CKPT_M7B=./checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42_w10_diverse.pt

echo "=== Stage 1: Qwen 7B, attn_dim=512 ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/qwen2.5-7b --device cuda:1 \
  --hybrid_checkpoint $CKPT_Q7B \
  --task hotpotqa --n_samples 50 --k 300 --attn_dim 512 \
  --save_path logs/results/hybrid_rag_longbench_hotpotqa_qwen7b_v8.json \
  2>&1 | tee $LOGDIR/qwen7b.log

echo "=== Stage 2: Mistral 7B retrained, attn_dim=256 ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/mistral-7b-v0.3 --device cuda:1 \
  --hybrid_checkpoint $CKPT_M7B \
  --task hotpotqa --n_samples 50 --k 300 --attn_dim 256 \
  --save_path logs/results/hybrid_rag_longbench_hotpotqa_mistral7b_v8.json \
  2>&1 | tee $LOGDIR/mistral7b.log

echo "=== Hybrid+RAG queue complete ==="
ls -la logs/results/hybrid_rag_longbench_*_v8.json
