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

# Wait pattern note: match the actual python worker procs (the bash parents
# may be inline `bash -c` subshells from the orphan-recovery flow).
WORKER_PATTERNS=(
    'python3 training/train_hybrid_x1\.py'
    'python3 baselines/eval_upstream_baselines\.py'
    'python3 training/eval_hybrid_position_robust\.py'
    'python3 baselines/eval_kivi\.py'
    'python3 baselines/eval_upstream_longbench\.py'
    'python3 baselines/eval_longbench\.py'
)
wait_for_blockers() {
    while true; do
        local total=0 c
        for p in "${WORKER_PATTERNS[@]}"; do
            c=$(pgrep -fc "$p" 2>/dev/null || true)
            total=$((total + ${c:-0}))
        done
        if [ "$total" -eq 0 ]; then break; fi
        echo "[$(date +%H:%M)] needle waiting: $total upstream worker(s)"
        sleep 600
    done
}

declare -A MODEL_PATH=(
    [qwen7b]="./models/qwen2.5-7b"
    [qwen14b]="./models/qwen2.5-14b"
    [qwen32b]="./models/qwen2.5-32b"
    [llama8b]="./models/llama-3.1-8b"
    [mistral7b]="./models/mistral-7b-v0.3"
    [mistral24b]="./models/mistral-small-24b"
)
declare -A AH_CKPT=(
    [qwen7b]="checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt"
    [qwen14b]="checkpoints/astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42_w10_diverse.pt"
    [qwen32b]="checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42_w10_diverse.pt"
    [llama8b]="checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42_w10_diverse.pt"
    [mistral7b]="checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42_w10_diverse.pt"
    [mistral24b]="checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42_w10_diverse.pt"
)
LARGE_BACKBONES=(qwen32b mistral24b)

K_VALUES=(150 300)
SEED_OFFSETS=(0 1 2)
BASELINE_METHODS=(snapkv h2o pyramidkv)
# n_windows extended to RULER-range context lengths (substitute for
# full RULER eval; see baselines/eval_upstream_ruler_note.md for the
# faithfulness caveats).  n=22 ~= 8k tokens, n=44 ~= 16k, n=85 ~= 32k
# at the 384-token window size used throughout the paper.
N_WINDOWS_LIST=(20 22 44 85)

run_baseline_cell() {
    local backbone=$1 method=$2 k=$3 seed=$4 device=$5
    local out="$SAVE_DIR/nd_${backbone}_${method}_k${k}_s${seed}.json"
    [ -f "$out" ] && { echo "  [skip] $out"; return; }
    echo "  [run] $backbone/$method k=$k seed_off=$seed on cuda:$device"
    CUDA_VISIBLE_DEVICES=$device $PY baselines/eval_upstream_needle.py \
        --model_path "${MODEL_PATH[$backbone]}" \
        --method "$method" \
        --k "$k" --n_windows_list "${N_WINDOWS_LIST[@]}" --n_trials 20 \
        --seed_offset "$seed" \
        --save_path "$out" \
        > "logs/training/needle_sweep/nd_${backbone}_${method}_k${k}_s${seed}.log" 2>&1
}

# AstroHybrid on Needle via baselines/eval_needle.py with the 'hybrid' method.
# CLI quirks of that script (not the upstream wrapper):
#   - takes --k (total budget) not --k_real (it does the 16/284 split internally
#     based on --n_mem)
#   - takes --n_windows_list (plural) not --n_windows
#   - has --seed_offset, --attn_dim=256 default (matches the baseline ckpts)
#   - writes to a hardcoded path; uses --save_suffix instead of --save_path.
#     We sym-link the hardcoded output into our SAVE_DIR after the run.
run_astrohybrid_cell() {
    local backbone=$1 k=$2 seed=$3 device=$4
    local out="$SAVE_DIR/nd_${backbone}_astrohybrid_k${k}_s${seed}.json"
    [ -f "$out" ] && { echo "  [skip] $out"; return; }
    local suffix="_sweep_k${k}_s${seed}"
    local model_basename
    case $backbone in
        qwen7b)  model_basename="qwen2.5-7b" ;;
        llama8b) model_basename="llama-3.1-8b" ;;
    esac
    local hard_path="logs/results/needle_${model_basename}_k${k}${suffix}.json"
    if [ -f baselines/eval_needle.py ]; then
        echo "  [run] $backbone/astrohybrid k=$k seed_off=$seed on cuda:$device"
        CUDA_VISIBLE_DEVICES=$device $PY baselines/eval_needle.py \
            --model_path "${MODEL_PATH[$backbone]}" \
            --hybrid_checkpoint "${AH_CKPT[$backbone]}" \
            --methods hybrid \
            --k "$k" --n_mem 16 --attn_dim 256 \
            --n_windows_list 20 --n_trials 20 \
            --seed_offset "$seed" \
            --save_suffix "$suffix" \
            > "logs/training/needle_sweep/nd_${backbone}_astrohybrid_k${k}_s${seed}.log" 2>&1 || \
            echo "  [WARN] $backbone/astrohybrid k=$k s=$seed failed (continuing)"
        # Sym-link hardcoded output into the sweep dir for the aggregator.
        [ -f "$hard_path" ] && ln -sf "$(realpath "$hard_path")" "$out"
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
