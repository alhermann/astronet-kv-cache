#!/bin/bash
# Budget sweep --- memory-pivot evidence base.
#
# Critic acceptance gates (see paper_npjai/AUDIT_BULLSHIT.md 2026-06-06 update):
#   1. Iso-accuracy crossover budget gap >= 2x on at least 2/3 must-run tasks
#   2. CI separation at the crossover budget across >=3 seeds
#   3. Crossover holds on a held-out, non-SQuAD task (LongBench / Needle)
#   4. Per-head Lloyd-Max K8V4 composition verified on >=2 tasks, >=3 seeds
#   5. Param-cost-corrected memory savings still >=30% at session length 4k
#
# If any of (1)-(3) fails, the memory pivot does NOT happen and we stay on the
# accuracy framing.  This script produces the curve data needed to evaluate
# the gates; it does NOT itself commit the paper to the pivot.
#
# Grid (must-run):
#   backbones : qwen 7B + llama 8B  (two backbones; full sweep across 6 models
#                                    is "nice-to-have" per critic, not blocking)
#   methods   : astrohybrid, snapkv, h2o, pyramidkv, streamingllm
#               (RAG-k=3 and KIVI K4V4 already have anchor numbers; we add
#                them at k=300 only for the parity line)
#   k         : 50, 100, 150, 200, 250, 300, 400, 600
#   seeds     : 42, 1234, 7    (3 seeds for paired-bootstrap CIs)
#   datasets  : SQuAD pos-robust (n=100/pos x 4 positions = 400 evals/seed)
#               LongBench MultiFieldQA + HotpotQA (n>=150/task) --- separate script
#
# Estimated wall-time on 2x Titan RTX:  ~6h per backbone for SQuAD sweep,
# ~3h per backbone for LongBench, ~2h for Needle.  Total ~22h serialised; we
# can split SQuAD and LongBench across the two GPUs to halve that.
#
# CONTRACT WITH UPSTREAM CODE:
#   - SnapKV / H2O / PyramidKV / StreamingLLM go through the audited
#     `baselines/eval_upstream_baselines.py` which is a thin wrapper around
#     the official KVCache-Factory monkey-patches (NOT the hand-rolled
#     re-implementations from `eval_longbench.py` that were withdrawn from
#     the paper).
#   - AstroHybrid goes through `training/eval_hybrid_position_robust.py`,
#     the same harness that produced the verified E3 numbers.
#   - The output JSON schema is the same `pos_0..pos_3` + `averages` shape
#     across both code paths; the aggregator merges them by (model, method,
#     k, seed).

set -euo pipefail
cd /home/alexander/Schreibtisch/AstroNet
PYTHONUNBUFFERED=1
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
SAVE_DIR=logs/results/budget_sweep
mkdir -p $SAVE_DIR logs/training/budget_sweep

# Wait for X1/X2 training to finish so we have GPUs.  If they're already
# done this exits immediately.
wait_for_training() {
    local procs
    procs=$(pgrep -f "train_hybrid_x1.py" | wc -l)
    while [ "$procs" -gt 0 ]; do
        echo "[$(date +%H:%M)] waiting for $procs X1/X2 training procs to finish..."
        sleep 300
        procs=$(pgrep -f "train_hybrid_x1.py" | wc -l)
    done
    echo "[$(date +%H:%M)] X1/X2 done --- launching budget sweep"
}

# Per-backbone configuration
declare -A MODEL_PATH=(
    [qwen7b]="./models/qwen2.5-7b"
    [llama8b]="./models/llama-3.1-8b"
)
declare -A AH_CKPT=(
    [qwen7b]="checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt"
    [llama8b]="checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42_w10_diverse.pt"
)
declare -A AH_NAME=(
    [qwen7b]="qwen2.5-7b"
    [llama8b]="llama-3.1-8b"
)

K_VALUES=(50 100 150 200 250 300 400 600)
SEEDS=(42 1234 7)
# NOTE (2026-06-06, critic audit): StreamingLLM has NO Qwen2 monkey-patch in
# baselines/kvcache_factory/monkeypatch.py:154-179 (only snapkv/h2o/pyramidkv
# branches exist).  eval_upstream_baselines.py's argparse also restricts
# --method to {snapkv, h2o, pyramidkv}.  Including streamingllm here would
# crash every cell with an argparse error and produce empty JSONs that the
# aggregator would silently treat as "no streamingllm baseline available",
# skewing best_baseline selection.  Dropped from this sweep.
BASELINE_METHODS=(snapkv h2o pyramidkv)

run_baseline_cell() {
    local backbone=$1 method=$2 k=$3 seed=$4 device=$5
    local out="$SAVE_DIR/sq_${backbone}_${method}_k${k}_s${seed}.json"
    if [ -f "$out" ]; then
        echo "  [skip] $out exists"; return
    fi
    echo "  [run] $backbone/$method k=$k seed=$seed on $device"
    CUDA_VISIBLE_DEVICES=$device $PY baselines/eval_upstream_baselines.py \
        --model_path "${MODEL_PATH[$backbone]}" \
        --method "$method" \
        --k "$k" --n_eval 100 --seed "$seed" \
        --positions 0 1 2 3 \
        --save_path "$out" \
        > "logs/training/budget_sweep/sq_${backbone}_${method}_k${k}_s${seed}.log" 2>&1
}

run_astrohybrid_cell() {
    local backbone=$1 k=$2 seed=$3 device=$4
    local out="$SAVE_DIR/sq_${backbone}_astrohybrid_k${k}_s${seed}.json"
    if [ -f "$out" ]; then
        echo "  [skip] $out exists"; return
    fi
    # AstroHybrid k = real + virtual; --k_real subtracts the 16 virtual slots
    local k_real=$((k - 16))
    [ "$k_real" -lt 16 ] && return  # skip k<32 where virtual budget dominates
    echo "  [run] $backbone/astrohybrid k=$k (k_real=$k_real, n_mem=16) seed=$seed on $device"
    CUDA_VISIBLE_DEVICES=$device $PY training/eval_hybrid_position_robust.py \
        --model_path "${MODEL_PATH[$backbone]}" \
        --checkpoint "${AH_CKPT[$backbone]}" \
        --n_eval 100 --seed "$seed" \
        --k_real "$k_real" --n_mem 16 \
        --positions 0 1 2 3 \
        --save_path "$out" \
        > "logs/training/budget_sweep/sq_${backbone}_astrohybrid_k${k}_s${seed}.log" 2>&1
}

run_backbone_grid() {
    # Run the full (method x k x seed) grid for one backbone on the given GPU.
    # NOTE: critic 2026-06-06 audit: parallelise across the two Titans by
    # invoking this fn once per backbone in the background, halving wall-time.
    local backbone=$1 device=$2
    echo
    echo "############################################"
    echo "##  Backbone: $backbone  on cuda:$device"
    echo "############################################"
    for seed in "${SEEDS[@]}"; do
        for k in "${K_VALUES[@]}"; do
            for method in "${BASELINE_METHODS[@]}"; do
                run_baseline_cell "$backbone" "$method" "$k" "$seed" "$device"
            done
            run_astrohybrid_cell "$backbone" "$k" "$seed" "$device"
        done
    done
}

main() {
    wait_for_training

    # Parallelise across both Titans: Qwen 7B on cuda:0, Llama 8B on cuda:1.
    # Each backbone runs its own (method x k x seed) loop independently.
    run_backbone_grid qwen7b  0 > "logs/training/budget_sweep/qwen7b_driver.log"  2>&1 &
    pid_qwen=$!
    run_backbone_grid llama8b 1 > "logs/training/budget_sweep/llama8b_driver.log" 2>&1 &
    pid_llama=$!
    echo "[$(date +%H:%M)] launched qwen7b (pid=$pid_qwen) on cuda:0 + llama8b (pid=$pid_llama) on cuda:1"
    wait $pid_qwen
    qwen_rc=$?
    wait $pid_llama
    llama_rc=$?
    echo "[$(date +%H:%M)] qwen7b rc=$qwen_rc  llama8b rc=$llama_rc"

    echo
    echo "[$(date +%H:%M)] SQuAD budget sweep complete.  Aggregating..."
    $PY baselines/aggregate_budget_sweep.py \
        --in_dir "$SAVE_DIR" \
        --out_path logs/results/budget_sweep_summary.json \
        --pareto_path logs/results/pareto_data_v2.json
    echo "Done."
}

main "$@"
