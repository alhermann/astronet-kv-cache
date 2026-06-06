#!/bin/bash
# X1 retry with ReZero gate --- the architectural fix from the 2026-06-06
# critic audit pass (§E.ii).
#
# Hypothesis tested: the original X1 catastrophic failure (hybrid=0% on
# SQuAD, -2.3pp on multi-hop) was caused by the cross-attention output
# residual having no scalar gate and no normalisation, so once training
# pushed out_proj off its zero init the residual scale was unbounded.
#
# Fix: ``astronet/x1_gapfill.py`` now defaults to ``use_rezero=True``
# which adds a learnable scalar ``gamma`` initialised at zero.  At init
# the residual contribution is identically zero regardless of
# out_proj.weight, so out_proj can be xavier-initialised safely; gamma
# then grows away from zero only when training demands a non-trivial
# residual.  ``training/train_hybrid.py``'s ``virtual_hidden.clamp(-100,
# 100)`` is also replaced with a soft tanh-shaped bound when X1 is
# attached, so the clamp can no longer silently kill X1's gradient
# signal when the residual approaches the boundary.
#
# Gate (decided in advance, see paper_npjai/AUDIT_BULLSHIT.md gate G7
# from the 2026-06-06 critic pass):
#   ReZero X1 hybrid_S1 >= 67.5%  (= baseline 62.5% + 5pp) --> paper-worthy
#   ReZero X1 hybrid_S1 <= 64.0%  (= baseline + 1.5pp)     --> X1 stays cut
#   in between                                              --> a §Ablation
#                                                              paragraph
#
# Wall-time on a single Titan: ~3-4h (5k SQuAD samples, 2 epochs, lr=1e-4
# matching the X2 that didn't catastrophically diverge).
#
# This script WAITS for:
#   - any remaining train_hybrid_x1.py procs
#   - the SQuAD budget sweep
#   - the KIVI rerun (if started)
# so it doesn't fight the higher-priority experiments for GPU.

set -euo pipefail
cd /home/alexander/Schreibtisch/AstroNet
PYTHONUNBUFFERED=1
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3

# Wait for ALL upstream workers (sweep, KIVI, LongBench, Needle, X1/X2).
WORKER_PATTERNS=(
    'python3 training/train_hybrid_x1\.py'
    'python3 baselines/eval_upstream_baselines\.py'
    'python3 training/eval_hybrid_position_robust\.py'
    'python3 baselines/eval_kivi\.py'
    'python3 baselines/eval_upstream_longbench\.py'
    'python3 baselines/eval_upstream_needle\.py'
    'python3 baselines/eval_longbench\.py'
    'python3 baselines/eval_needle\.py'
)
while true; do
    total=0
    for p in "${WORKER_PATTERNS[@]}"; do
        c=$(pgrep -fc "$p" 2>/dev/null || true)
        total=$((total + ${c:-0}))
    done
    if [ "$total" -eq 0 ]; then break; fi
    echo "[$(date +%H:%M)] X1-ReZero waiting: $total upstream worker(s)"
    sleep 600
done

echo "[$(date +%H:%M)] launching X1 ReZero retry on cuda:0"
mkdir -p logs/training
CUDA_VISIBLE_DEVICES=0 $PY \
    training/train_hybrid_x1.py \
    --model_path ./models/qwen2.5-7b \
    --init_checkpoint checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt \
    --n_train 5000 --epochs 2 --lr 1e-4 --data squad \
    --device cuda:0 \
    --save_path checkpoints/astro_x1_qwen7b_squad_rezero.pt \
    > logs/training/x1_rezero_train.log 2>&1

echo "[$(date +%H:%M)] training done; launching swap-selector eval"
CUDA_VISIBLE_DEVICES=0 $PY \
    training/eval_hybrid_swap_selector.py \
    --model_path ./models/qwen2.5-7b \
    --checkpoint checkpoints/astro_x1_qwen7b_squad_rezero.pt \
    --selector snapkv \
    --n_eval 100 \
    --device cuda:0 \
    --save_path logs/results/e3_x1_rezero_qwen7b.json \
    > logs/training/x1_rezero_eval.log 2>&1

echo "[$(date +%H:%M)] X1 ReZero retry pipeline done"
echo "  result: logs/results/e3_x1_rezero_qwen7b.json"
