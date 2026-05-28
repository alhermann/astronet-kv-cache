#!/usr/bin/env bash
# Re-run LongBench eviction baselines (StreamingLLM, H2O, SnapKV) on the
# two Mistral backbones to match the post-retrain (w10_diverse) AstroNet
# columns in tab:longbench. Closes the methodological consistency gap
# flagged by the critic.
#
# AstroNet columns themselves are unchanged: only baseline rows are refreshed.
# Output overwrites the pre-retrain .bak entries with a new "_v2" suffix so
# the previous values remain auditable.
set -e

cd .
PY="${PY:-python3}"
SCRIPT=baselines/eval_longbench.py
LOGDIR=logs/longbench_mistral_baselines_v2
mkdir -p "$LOGDIR"

COMMON_ARGS="--tasks hotpotqa multifieldqa_en --max_samples 200 --k 300 \
             --methods streaming_llm h2o snapkv"

echo "=== Stage 1: Mistral 7B (cuda:1, ~30-45 min) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/mistral-7b-v0.3 \
  --device cuda:1 \
  --save_path logs/results/longbench_mistral-7b-v0.3_k300_baselines_v2.json \
  $COMMON_ARGS 2>&1 | tee "$LOGDIR/mistral7b.log"

echo "=== Stage 2: Mistral-Small 24B (multi-GPU, ~60-90 min) ==="
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY $SCRIPT \
  --model_path ./models/mistral-small-24b \
  --multi_gpu \
  --save_path logs/results/longbench_mistral-small-24b_k300_baselines_v2.json \
  $COMMON_ARGS 2>&1 | tee "$LOGDIR/mistral24b.log"

echo
echo "=== LongBench Mistral baseline re-eval complete ==="
ls -la logs/results/longbench_*_baselines_v2.json
