#!/bin/bash
# KIVI K4V4 R=128 rerun, all six backbones, 5 seeds each.
#
# Replaces the current single-seed KIVI K4V4 column in tab:squad_main
# with a faithful, R=128-residual-buffer, 5-seed measurement.  Without
# the buffer the previous numbers slightly UNDER-estimated KIVI
# (recent tokens, which dominate decode attention, were being
# quantised when they shouldn't have been).  After the rerun the
# AstroNet-vs-KIVI gap will likely SHRINK by some pp on most models;
# the [PENDING] placeholder in the abstract (Overleaf 5620375) is
# unblocked by this script's output.
#
# Each (backbone, seed) cell runs all 4 KIVI configs the paper cites
# (k2v2, k4v2, k4v4, k8v4); ~30 min per cell, 6 backbones x 5 seeds =
# ~15 hours single-GPU.  Parallelised across both Titans -> ~7.5 h.
#
# Queued AFTER the SQuAD budget sweep (don't fight it for GPUs).

set -euo pipefail
cd /home/alexander/Schreibtisch/AstroNet
PYTHONUNBUFFERED=1
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
SAVE_DIR=logs/results/kivi_r128
mkdir -p "$SAVE_DIR" logs/training/kivi_r128

# Wait for the SQuAD budget sweep AND the LongBench/Needle siblings.
# Match the actual python worker processes since the bash parents may be
# inline ad-hoc subshells (the orphan-recovery pattern uses `bash -c '...'`).
SWEEP_PATTERNS=(
    'python3 baselines/eval_upstream_baselines\.py'
    'python3 training/eval_hybrid_position_robust\.py'
    'python3 baselines/eval_upstream_longbench\.py'
    'python3 baselines/eval_upstream_needle\.py'
    'python3 baselines/eval_longbench\.py'
    'python3 baselines/eval_needle\.py'
)
wait_for_sweep() {
    local total p
    while true; do
        total=0
        for p in "${SWEEP_PATTERNS[@]}"; do
            local c
            c=$(pgrep -fc "$p" 2>/dev/null || true)
            total=$((total + ${c:-0}))
        done
        if [ "$total" -eq 0 ]; then break; fi
        echo "[$(date +%H:%M)] kivi-r128 waiting: $total active sweep worker(s)"
        sleep 600
    done
}

# Backbones and their model dirs.
declare -A MODEL=(
    [qwen7b]="./models/qwen2.5-7b"
    [qwen14b]="./models/qwen2.5-14b"
    [qwen32b]="./models/qwen2.5-32b"
    [llama8b]="./models/llama-3.1-8b"
    [mistral7b]="./models/mistral-7b-v0.3"
    [mistral24b]="./models/mistral-small-24b"
)
SEEDS=(7 42 123 999 2024)   # match tab:squad_main's 5-seed protocol

run_cell() {
    local backbone=$1 seed=$2 device=$3
    local out="$SAVE_DIR/kivi_${backbone}_s${seed}_r128.json"
    [ -f "$out" ] && { echo "  [skip] $out"; return; }
    echo "  [run] $backbone seed=$seed (r=128) on cuda:$device"
    CUDA_VISIBLE_DEVICES=$device $PY baselines/eval_kivi.py \
        --model_path "${MODEL[$backbone]}" \
        --n_samples 100 --seed "$seed" --k 300 \
        --configs k2v2 k4v2 k4v4 k8v4 \
        --residual_length 128 \
        --device cuda:0 \
        --save_path "$out" \
        > "logs/training/kivi_r128/kivi_${backbone}_s${seed}.log" 2>&1 || \
        echo "  [WARN] $backbone seed=$seed failed (continuing)"
}

# qwen32b + mistral24b need multi-GPU (>22GB cache); skip for now,
# we already have hand-rolled numbers from before that we can amend
# AFTER seeing the smaller-model trends.
SINGLE_GPU=(qwen7b qwen14b llama8b mistral7b)
TWO_GPU=(qwen32b mistral24b)

main() {
    wait_for_sweep
    echo "[$(date +%H:%M)] launching KIVI R=128 rerun for single-GPU backbones"
    # Split across the two Titans: even-indexed seeds on cuda:0, odd on cuda:1.
    for backbone in "${SINGLE_GPU[@]}"; do
        for i in "${!SEEDS[@]}"; do
            seed=${SEEDS[$i]}
            device=$((i % 2))
            run_cell "$backbone" "$seed" "$device" &
            # Throttle to 2 in flight at once.
            while [ $(jobs -rp | wc -l) -ge 2 ]; do sleep 30; done
        done
        wait
        echo "  [done] $backbone all seeds"
    done
    echo "[$(date +%H:%M)] kivi-r128 single-GPU pass complete"
    # qwen32b and mistral24b need device_map='auto' across both Titans
    # (their NF4 weights alone don't fit on one 24 GB card with the
    # 100-sample eval buffer).  Run sequentially.
    for backbone in "${TWO_GPU[@]}"; do
        for seed in "${SEEDS[@]}"; do
            local out="$SAVE_DIR/kivi_${backbone}_s${seed}_r128.json"
            [ -f "$out" ] && { echo "  [skip] $out"; continue; }
            echo "  [run mGPU] $backbone seed=$seed (r=128) on cuda:0+cuda:1"
            CUDA_VISIBLE_DEVICES=0,1 $PY baselines/eval_kivi.py \
                --model_path "${MODEL[$backbone]}" \
                --n_samples 100 --seed "$seed" --k 300 \
                --configs k2v2 k4v2 k4v4 k8v4 \
                --residual_length 128 \
                --device cuda:0 \
                --save_path "$out" \
                > "logs/training/kivi_r128/kivi_${backbone}_s${seed}.log" 2>&1 || \
                echo "  [WARN] $backbone seed=$seed failed (continuing)"
        done
        echo "  [done] $backbone all seeds (mGPU)"
    done
    echo "[$(date +%H:%M)] kivi-r128 multi-GPU pass complete"
}

main "$@"
