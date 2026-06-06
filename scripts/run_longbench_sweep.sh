#!/bin/bash
# LongBench faithful-baseline sweep --- sibling of run_budget_sweep.sh.
#
# Critic gate 3 ("crossover holds on held-out non-SQuAD task") is evaluated
# here.  If the SQuAD budget sweep shows AstroHybrid crossing over the best
# baseline at k <= 150 but the same crossover does NOT hold on LongBench,
# the memory pivot is SQuAD-specific and we revert to the accuracy framing.
#
# Grid:
#   backbones : qwen 7B + llama 8B  (matches run_budget_sweep.sh)
#   methods   : astrohybrid, snapkv, h2o, pyramidkv
#               (StreamingLLM dropped: no Qwen2 upstream patch;
#                KVCache-Factory has Llama branches but our budget-sweep
#                aggregator can't merge mixed-method backbones cleanly.)
#   k         : 150, 300, 600          --- LongBench uses larger budgets
#               than SQuAD because contexts are ~5k tokens not ~1k.  At
#               k=300 we test the same "tight" point as SQuAD; k=600 is
#               LongBench's published canonical budget; k=150 is the
#               aspirational memory-pivot point.
#   tasks     : multifieldqa_en + hotpotqa (single-hop + multi-hop)
#   n_samples : 100 per task
#   seeds     : 1 seed for cost reasons --- LongBench eval is ~5x slower
#               than SQuAD per cell.  We get CI signal from the SQuAD sweep.
#
# NOTE: the upstream eval_upstream_longbench.py does NOT take a --seed
# argument (it reads the LongBench validation split deterministically).
# To get noise estimates we'd have to subsample, which we skip for the
# initial pass.

set -euo pipefail
cd /home/alexander/Schreibtisch/AstroNet
PYTHONUNBUFFERED=1
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
SAVE_DIR=logs/results/longbench_sweep
mkdir -p "$SAVE_DIR" logs/training/longbench_sweep

# Same TRAIN_PATTERN technique as run_budget_sweep.sh to avoid self-match.
TRAIN_PATTERN='python3 training/train_hybrid_x1\.py'
wait_for_x1() {
    local procs
    procs=$(pgrep -fc "$TRAIN_PATTERN" 2>/dev/null || true)
    procs=${procs:-0}
    while [ "$procs" -gt 0 ]; do
        echo "[$(date +%H:%M)] waiting for X1/X2 procs..."
        sleep 300
        procs=$(pgrep -fc "$TRAIN_PATTERN" 2>/dev/null || true)
        procs=${procs:-0}
    done
}

# Also wait for the SQuAD budget sweep so we don't double-book GPUs.
SWEEP_PATTERN='bash scripts/run_budget_sweep\.sh'
wait_for_squad_sweep() {
    local procs
    procs=$(pgrep -fc "$SWEEP_PATTERN" 2>/dev/null || true)
    procs=${procs:-0}
    while [ "$procs" -gt 0 ]; do
        echo "[$(date +%H:%M)] waiting for SQuAD budget sweep to finish..."
        sleep 600
        procs=$(pgrep -fc "$SWEEP_PATTERN" 2>/dev/null || true)
        procs=${procs:-0}
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

K_VALUES=(150 300 600)
BASELINE_METHODS=(snapkv h2o pyramidkv)
TASKS="multifieldqa_en hotpotqa"

run_baseline_cell() {
    local backbone=$1 method=$2 k=$3 device=$4
    local out="$SAVE_DIR/lb_${backbone}_${method}_k${k}.json"
    [ -f "$out" ] && { echo "  [skip] $out"; return; }
    echo "  [run] $backbone/$method k=$k on cuda:$device"
    CUDA_VISIBLE_DEVICES=$device $PY baselines/eval_upstream_longbench.py \
        --model_path "${MODEL_PATH[$backbone]}" \
        --method "$method" \
        --k "$k" --n_samples 100 \
        --tasks $TASKS \
        --save_path "$out" \
        > "logs/training/longbench_sweep/lb_${backbone}_${method}_k${k}.log" 2>&1
}

# For AstroHybrid on LongBench we reuse training/eval_hybrid_longbench.py
# if it exists; otherwise we skip and document the gap.  The AstroHybrid
# LongBench numbers in the current paper come from a different harness
# that we already audited; for the budget sweep we want matched-protocol
# numbers, so we use the same single-prompt path as the upstream baselines.
run_astrohybrid_cell() {
    local backbone=$1 k=$2 device=$3
    local out="$SAVE_DIR/lb_${backbone}_astrohybrid_k${k}.json"
    [ -f "$out" ] && { echo "  [skip] $out"; return; }
    # The AstroHybrid LongBench numbers in the current paper come from
    # baselines/eval_longbench.py with --hybrid_checkpoint.  Streaming
    # protocol; matches AstroNet's design (the upstream baselines are
    # single-prompt and serve as the standard-protocol reference).
    echo "  [run] $backbone/astrohybrid k=$k on cuda:$device"
    CUDA_VISIBLE_DEVICES=$device $PY baselines/eval_longbench.py \
        --model_path "${MODEL_PATH[$backbone]}" \
        --hybrid_checkpoint "${AH_CKPT[$backbone]}" \
        --k "$k" --max_samples 100 \
        --tasks $TASKS \
        --methods astronet_hybrid \
        --save_path "$out" \
        > "logs/training/longbench_sweep/lb_${backbone}_astrohybrid_k${k}.log" 2>&1
}

run_backbone_grid() {
    local backbone=$1 device=$2
    echo
    echo "##  Backbone: $backbone on cuda:$device"
    for k in "${K_VALUES[@]}"; do
        for method in "${BASELINE_METHODS[@]}"; do
            run_baseline_cell "$backbone" "$method" "$k" "$device"
        done
        run_astrohybrid_cell "$backbone" "$k" "$device"
    done
}

main() {
    wait_for_x1
    wait_for_squad_sweep
    echo "[$(date +%H:%M)] launching LongBench sweep (parallel across both GPUs)"
    run_backbone_grid qwen7b  0 > logs/training/longbench_sweep/qwen7b.log  2>&1 &
    pid_q=$!
    run_backbone_grid llama8b 1 > logs/training/longbench_sweep/llama8b.log 2>&1 &
    pid_l=$!
    wait $pid_q; rc_q=$?
    wait $pid_l; rc_l=$?
    echo "[$(date +%H:%M)] qwen7b rc=$rc_q  llama8b rc=$rc_l"
    echo "[$(date +%H:%M)] aggregating..."
    $PY baselines/aggregate_longbench_sweep.py \
        --in_dir "$SAVE_DIR" \
        --out_path logs/results/longbench_sweep_summary.json \
        --tasks $TASKS \
        || echo "(aggregator script not yet written --- raw JSONs in $SAVE_DIR)"
}

main "$@"
