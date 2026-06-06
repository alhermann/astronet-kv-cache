#!/bin/bash
# Retraining queue --- four Stage 2 variants in priority order.
#
# Priority based on (expected payoff x novelty x methodological soundness)
# after the 2026-06-07 single-prompt-Stage-1 diagnostic showed our
# scoring rule is streaming-native (single-prompt is WORSE for us).
#
# All variants train on Qwen 7B only (single backbone, ~6h each).
# If a variant produces a positive lift, extend to other backbones
# (~36h additional).
#
# IMPORTANT: queued behind all other GPU workers via the worker-pattern
# wait.  Will not fire while sweep / KIVI / LongBench / Needle / X1 ReZero
# / soft-prompt are still running.
#
# Each variant uses a DIFFERENT save_path so checkpoints don't collide.

set -euo pipefail
cd /home/alexander/Schreibtisch/AstroNet
PYTHONUNBUFFERED=1
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
mkdir -p logs/training/retraining checkpoints

WORKER_PATTERNS=(
    'python3 training/train_hybrid_x1\.py'
    'python3 training/train_hybrid\.py'
    'python3 training/train_softprompt\.py'
    'python3 baselines/eval_upstream_baselines\.py'
    'python3 training/eval_hybrid_position_robust\.py'
    'python3 baselines/eval_kivi\.py'
    'python3 baselines/eval_upstream_longbench\.py'
    'python3 baselines/eval_upstream_needle\.py'
    'python3 baselines/eval_longbench\.py'
    'python3 baselines/eval_needle\.py'
)
wait_for_workers() {
    while true; do
        local total=0 c
        for p in "${WORKER_PATTERNS[@]}"; do
            c=$(pgrep -fc "$p" 2>/dev/null || true)
            total=$((total + ${c:-0}))
        done
        if [ "$total" -eq 0 ]; then break; fi
        echo "[$(date +%H:%M)] retraining waiting: $total upstream worker(s)"
        sleep 600
    done
}

MODEL=./models/qwen2.5-7b

# ----------------------------------------------------------------------
# P0: quick diagnostic --- existing Stage 2 + single-prompt SnapKV-style.
# If this gives s1+s2 - s1 > +3pp, we have a no-retrain story.
# Eats one cuda:0 slot for ~15 min.
# ----------------------------------------------------------------------
wait_for_workers
echo "[$(date +%H:%M)] P0 quick test: existing S2 on single-prompt SnapKV-style"
CUDA_VISIBLE_DEVICES=0 $PY \
    training/diag_existing_s2_on_sp_snapkv.py \
    --model_path $MODEL \
    --checkpoint checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt \
    --k 300 --n_mem 16 --attn_dim 256 \
    --n_eval 100 --seed 42 --positions 0 1 2 3 \
    --device cuda:0 \
    --save_path logs/results/diag_existing_s2_on_sp_snapkv_qwen7b.json \
    > logs/training/retraining/diag_existing_s2_on_sp_snapkv.log 2>&1 || \
    echo "[WARN] P0 failed; continuing"

# ----------------------------------------------------------------------
# P1: Stage 2 with smaller n_mem (8 instead of 16).
# Addresses BUG-6 (displacement at tight k).  Same training corpus.
# ----------------------------------------------------------------------
wait_for_workers
echo "[$(date +%H:%M)] P1 retrain: n_mem=8 (less displacement at tight k)"
CUDA_VISIBLE_DEVICES=0 $PY \
    training/train_hybrid.py \
    --model_path $MODEL \
    --n_train 5000 --n_eval 100 --epochs 2 \
    --k_real 292 --n_mem 8 --sense_layer 14 --attn_dim 256 \
    --train_seed 42 --eval_seed 42 \
    --device cuda:0 \
    > logs/training/retraining/P1_train_nmem8.log 2>&1 || \
    echo "[WARN] P1 train failed; continuing"
# Eval P1 ckpt at k=300 (the canonical operating point)
P1_CKPT=$(ls -t checkpoints/*nmem8* 2>/dev/null | head -1)
if [ -n "$P1_CKPT" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY \
        training/eval_hybrid_position_robust.py \
        --model_path $MODEL --checkpoint "$P1_CKPT" \
        --n_eval 100 --seed 42 --k_real 292 --n_mem 8 --attn_dim 256 \
        --positions 0 1 2 3 \
        --save_path logs/results/retrain_P1_nmem8_qwen7b.json \
        > logs/training/retraining/P1_eval_nmem8.log 2>&1 || \
        echo "[WARN] P1 eval failed"
fi

# ----------------------------------------------------------------------
# P2: same Stage 2 with multi-position fact placement during training.
# Existing train_hybrid.py uses fact_position=0; this would require a
# small patch.  Document as TODO and skip if patch not yet applied.
# ----------------------------------------------------------------------
echo "[$(date +%H:%M)] P2 retrain: multi-position fact placement (SKIPPED --- requires patch to train_hybrid.py)"

# ----------------------------------------------------------------------
# P3: Stage 2 trained against SnapKV-style scoring (streaming protocol).
# REQUIRES new training script training/train_hybrid_snapkv_selector.py
# that swaps our multiplicative selector for max-pool kernel=7 over the
# end-of-prompt observation window during training.  Document as TODO.
# ----------------------------------------------------------------------
echo "[$(date +%H:%M)] P3 retrain: Stage 2 specialised for SnapKV selector (SKIPPED --- requires new training script)"

echo "[$(date +%H:%M)] retraining queue complete"
