#!/bin/bash
# Soft-prompt baseline --- defuses the strongest reviewer attack against
# Stage 2 ("your 16 virtual KV tokens are just learned soft prompts").
#
# We train a 16-vector and a 3900-vector (param-matched) soft prompt on
# the same 5000-sample SQuAD multi-segment corpus that trained Stage 2,
# then evaluate on the same position-robust protocol used in tab:squad_main.
# The paper will then report:
#   SoftPrompt-16   vs  Stage 1+2 (token-count comparison)
#   SoftPrompt-3900 vs  Stage 1+2 (param-count comparison)
# A soft-prompt that LOSES at matched params would defang the "this is
# just a soft prompt" attack; a soft-prompt that WINS would force us to
# either justify why our architecture matters or fold the baseline in.

set -euo pipefail
cd /home/alexander/Schreibtisch/AstroNet
PYTHONUNBUFFERED=1
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
mkdir -p logs/training/softprompt logs/results

# Wait for all prior GPU workers to free up (sweep, KIVI, LongBench,
# Needle, X1/X2, ReZero).  Same worker-pattern as the other queued
# drivers; matches python procs not bash parents.
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
    echo "[$(date +%H:%M)] softprompt waiting: $total upstream worker(s)"
    sleep 600
done

run_pair() {
    # Train + eval one (backbone, n_prompt) combination.
    local backbone=$1 model_path=$2 n_prompt=$3 device=$4
    local tag="${backbone}_np${n_prompt}"
    local ckpt="checkpoints/softprompt_${tag}.pt"
    local result="logs/results/softprompt_${tag}.json"
    [ -f "$result" ] && { echo "  [skip] $result"; return; }

    if [ ! -f "$ckpt" ]; then
        echo "  [train] $backbone n_prompt=$n_prompt on cuda:$device"
        CUDA_VISIBLE_DEVICES=$device $PY training/train_softprompt.py \
            --model_path "$model_path" \
            --n_train 5000 --n_prompt "$n_prompt" \
            --epochs 2 --lr 1e-3 \
            --device cuda:0 \
            --save_path "$ckpt" \
            > "logs/training/softprompt/train_${tag}.log" 2>&1 || \
            { echo "  [WARN] train failed: $tag"; return; }
    fi

    echo "  [eval] $backbone n_prompt=$n_prompt on cuda:$device"
    CUDA_VISIBLE_DEVICES=$device $PY training/eval_softprompt.py \
        --model_path "$model_path" \
        --prompt_path "$ckpt" \
        --n_eval 100 --seed 42 \
        --positions 0 1 2 3 \
        --device cuda:0 \
        --save_path "$result" \
        > "logs/training/softprompt/eval_${tag}.log" 2>&1 || \
        { echo "  [WARN] eval failed: $tag"; return; }
}

main() {
    echo "[$(date +%H:%M)] launching soft-prompt baseline (parallel on both Titans)"
    # Qwen 7B on cuda:0, Llama 8B on cuda:1 for parallelism.
    # Two operating points each: 16 vectors (token-count matched) and
    # 3900 vectors (param-count matched to Stage 2 ~= 14M params).
    {
        run_pair qwen7b  ./models/qwen2.5-7b     16   0
        run_pair qwen7b  ./models/qwen2.5-7b     3900 0
    } > logs/training/softprompt/qwen7b_driver.log 2>&1 &
    pid_q=$!
    {
        run_pair llama8b ./models/llama-3.1-8b   16   1
        run_pair llama8b ./models/llama-3.1-8b   3900 1
    } > logs/training/softprompt/llama8b_driver.log 2>&1 &
    pid_l=$!
    wait $pid_q; rc_q=$?
    wait $pid_l; rc_l=$?
    echo "[$(date +%H:%M)] qwen7b rc=$rc_q llama8b rc=$rc_l"
}

main "$@"
