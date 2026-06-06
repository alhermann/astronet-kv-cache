#!/bin/bash
# Needle-in-a-haystack faithful-baseline sweep.  Sibling of
# run_budget_sweep.sh and run_longbench_sweep.sh.
#
# Critic gate 3 redundancy: if both LongBench AND Needle show no crossover
# for AstroHybrid at tight k, the memory pivot is dead.  Needle is a
# different kind of held-out task (synthetic retrieval rather than natural
# QA), so it stress-tests transferability orthogonally to LongBench.
#
# Grid:
#   backbones   : qwen 7B + llama 8B
#   methods     : astrohybrid, snapkv, h2o, pyramidkv
#   k           : 150, 300                    (two operating points)
#   n_windows   : 20                          (the canonical 20-segment setup)
#   n_trials    : 20 per depth, 5 depths
#   seed_offset : 0, 1, 2                     (3 seed offsets ~ matched-power)

set -euo pipefail
cd /home/alexander/Schreibtisch/AstroNet
PYTHONUNBUFFERED=1
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
SAVE_DIR=logs/results/needle_sweep
mkdir -p "$SAVE_DIR" logs/training/needle_sweep

TRAIN_PATTERN='python3 training/train_hybrid_x1\.py'
SWEEP_PATTERN='bash scripts/run_budget_sweep\.sh'
LB_PATTERN='bash scripts/run_longbench_sweep\.sh'
wait_for_blockers() {
    for pat in "$TRAIN_PATTERN" "$SWEEP_PATTERN" "$LB_PATTERN"; do
        local procs
        procs=$(pgrep -fc "$pat" 2>/dev/null || true)
        procs=${procs:-0}
        while [ "$procs" -gt 0 ]; do
            echo "[$(date +%H:%M)] waiting for: $pat"
            sleep 600
            procs=$(pgrep -fc "$pat" 2>/dev/null || true)
            procs=${procs:-0}
        done
    done
}

declare -A MODEL_PATH=(
    [qwen7b]="./models/qwen2.5-7b"
    [llama8b]="./models/llama-3.1-8b"
)
declare -A AH_CKPT=(
    [qwen7b]="checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt"
    [llama8b]="checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42_w10_diverse.pt"
)

K_VALUES=(150 300)
SEED_OFFSETS=(0 1 2)
BASELINE_METHODS=(snapkv h2o pyramidkv)

run_baseline_cell() {
    local backbone=$1 method=$2 k=$3 seed=$4 device=$5
    local out="$SAVE_DIR/nd_${backbone}_${method}_k${k}_s${seed}.json"
    [ -f "$out" ] && { echo "  [skip] $out"; return; }
    echo "  [run] $backbone/$method k=$k seed_off=$seed on cuda:$device"
    CUDA_VISIBLE_DEVICES=$device $PY baselines/eval_upstream_needle.py \
        --model_path "${MODEL_PATH[$backbone]}" \
        --method "$method" \
        --k "$k" --n_windows_list 20 --n_trials 20 \
        --seed_offset "$seed" \
        --save_path "$out" \
        > "logs/training/needle_sweep/nd_${backbone}_${method}_k${k}_s${seed}.log" 2>&1
}

# For AstroHybrid on Needle we use baselines/eval_needle.py if it exists,
# which produced the original AstroNet Needle numbers in the paper.
# That harness DOES feed AstroNet's selection-plus-summary cache; it's the
# correct match-protocol for AstroHybrid vs faithful baselines on Needle.
run_astrohybrid_cell() {
    local backbone=$1 k=$2 seed=$3 device=$4
    local out="$SAVE_DIR/nd_${backbone}_astrohybrid_k${k}_s${seed}.json"
    [ -f "$out" ] && { echo "  [skip] $out"; return; }
    local k_real=$((k - 16))
    if [ -f baselines/eval_needle.py ]; then
        echo "  [run] $backbone/astrohybrid k=$k (k_real=$k_real) seed_off=$seed on cuda:$device"
        CUDA_VISIBLE_DEVICES=$device $PY baselines/eval_needle.py \
            --model_path "${MODEL_PATH[$backbone]}" \
            --checkpoint "${AH_CKPT[$backbone]}" \
            --k_real "$k_real" --n_mem 16 \
            --n_windows 20 --n_trials 20 \
            --seed_offset "$seed" \
            --save_path "$out" \
            > "logs/training/needle_sweep/nd_${backbone}_astrohybrid_k${k}_s${seed}.log" 2>&1
    else
        echo "  [skip-no-script] $backbone/astrohybrid (no baselines/eval_needle.py)"
    fi
}

run_backbone_grid() {
    local backbone=$1 device=$2
    echo
    echo "##  Backbone: $backbone on cuda:$device"
    for seed in "${SEED_OFFSETS[@]}"; do
        for k in "${K_VALUES[@]}"; do
            for method in "${BASELINE_METHODS[@]}"; do
                run_baseline_cell "$backbone" "$method" "$k" "$seed" "$device"
            done
            run_astrohybrid_cell "$backbone" "$k" "$seed" "$device"
        done
    done
}

main() {
    wait_for_blockers
    echo "[$(date +%H:%M)] launching Needle sweep (parallel across both GPUs)"
    run_backbone_grid qwen7b  0 > logs/training/needle_sweep/qwen7b.log  2>&1 &
    pid_q=$!
    run_backbone_grid llama8b 1 > logs/training/needle_sweep/llama8b.log 2>&1 &
    pid_l=$!
    wait $pid_q; rc_q=$?
    wait $pid_l; rc_l=$?
    echo "[$(date +%H:%M)] qwen7b rc=$rc_q  llama8b rc=$rc_l"
    echo "[$(date +%H:%M)] needle sweep done; raw JSONs in $SAVE_DIR"
}

main "$@"
