#!/usr/bin/env bash
# Hybrid+RAG ensemble on LongBench HotpotQA, Llama 3.1-8B.
# Chained to start after the Qwen 7B + Mistral 7B queue finishes (cuda:1 release).
set -e
cd .
PY="${PY:-python3}"
SCRIPT=baselines/eval_hybrid_rag_longbench.py
LOGDIR=logs/hybrid_rag_v8
mkdir -p $LOGDIR

CKPT_L8B=./checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42.pt

# Wait for the running v8 queue (looks for any eval_hybrid_rag_longbench process not ourselves)
echo "Waiting for prior hybrid_rag eval to finish..."
while pgrep -af "eval_hybrid_rag_longbench.py.*qwen2\.5-7b\|eval_hybrid_rag_longbench.py.*mistral-7b-v0\.3" > /dev/null; do
  sleep 60
done
echo "GPU free at $(date). Launching Llama 8B hybrid+RAG."

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/llama-3.1-8b --device cuda:1 \
  --hybrid_checkpoint $CKPT_L8B \
  --task hotpotqa --n_samples 50 --k 300 --attn_dim 512 \
  --save_path logs/results/hybrid_rag_longbench_hotpotqa_llama8b_v8.json \
  2>&1 | tee $LOGDIR/llama8b.log

echo "=== Llama 8B hybrid+RAG complete ==="
ls -la logs/results/hybrid_rag_longbench_hotpotqa_llama8b_v8.json
