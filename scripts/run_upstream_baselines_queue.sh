#!/bin/bash
# Re-run ALL baselines using the upstream kvcache_factory monkey-patches
# (SnapKV / H2O / PyramidKV).
#
# Replaces the unfaithful "faithful" baselines that we quarantined in
# logs/results/discarded_unfaithful_baselines/.
#
# Matrix: 6 models x 3 methods x 3 benchmarks = 54 cells.
# Per-cell scripts are in baselines/eval_upstream_{baselines,longbench,needle}.py.
# Each cell is one process invocation (monkey-patches are global).
# Skips cells whose JSON already exists.

set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
mkdir -p logs/training logs/results
LOG=logs/training/upstream_baselines_queue_$(date +%Y%m%d_%H%M%S).log
echo "[upstream-queue] start $(date)" | tee -a "$LOG"

# --- Per-benchmark cell runners --------------------------------------

sq_run() {  # SQuAD pos-robust, n=100, 4 positions, k=300
    local TAG="$1" MODEL="$2" METHOD="$3" EXTRA="${4:-}"
    local SAVE=logs/results/upstream_squad_${METHOD}_${TAG}.json
    if [ -f "$SAVE" ]; then
        echo "[skip] $TAG $METHOD squad already done" | tee -a "$LOG"; return 0
    fi
    echo "[run]  $TAG $METHOD squad $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/eval_upstream_baselines.py \
        --model_path "$MODEL" --method "$METHOD" --k 300 --n_eval 100 \
        --seed 42 --positions 0 1 2 3 $EXTRA \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[done] $TAG $METHOD squad $(date) -> $SAVE" | tee -a "$LOG"
}

nd_run() {  # Needle, n_windows {5,10,20}, n_trials=20, k=300
    local TAG="$1" MODEL="$2" METHOD="$3" EXTRA="${4:-}"
    local SAVE=logs/results/upstream_needle_${METHOD}_${TAG}.json
    if [ -f "$SAVE" ]; then
        echo "[skip] $TAG $METHOD needle already done" | tee -a "$LOG"; return 0
    fi
    echo "[run]  $TAG $METHOD needle $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/eval_upstream_needle.py \
        --model_path "$MODEL" --method "$METHOD" --k 300 \
        --n_windows_list 5 10 20 --n_trials 20 $EXTRA \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[done] $TAG $METHOD needle $(date) -> $SAVE" | tee -a "$LOG"
}

lb_run() {  # LongBench, n=100, k=300 (matches our other LongBench runs)
    local TAG="$1" MODEL="$2" METHOD="$3" EXTRA="${4:-}"
    local SAVE=logs/results/upstream_longbench_${METHOD}_${TAG}.json
    if [ -f "$SAVE" ]; then
        echo "[skip] $TAG $METHOD longbench already done" | tee -a "$LOG"; return 0
    fi
    echo "[run]  $TAG $METHOD longbench $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/eval_upstream_longbench.py \
        --model_path "$MODEL" --method "$METHOD" --k 300 \
        --tasks hotpotqa multifieldqa_en 2wikimqa musique \
        --n_samples 100 $EXTRA \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[done] $TAG $METHOD longbench $(date) -> $SAVE" | tee -a "$LOG"
}

# --- Models ----------------------------------------------------------

declare -a MODELS_SINGLE=(
    "qwen7b    ./models/qwen2.5-7b"
    "llama8b   ./models/llama-3.1-8b"
    "mistral7b ./models/mistral-7b-v0.3"
    "qwen14b   ./models/qwen2.5-14b"
)
declare -a MODELS_MULTI=(
    "qwen32b    ./models/qwen2.5-32b    --multi_gpu"
    "mistral24b ./models/mistral-small-24b --multi_gpu"
)
METHODS=(snapkv h2o pyramidkv)

# --- Order: SQuAD (fastest) -> Needle -> LongBench (slowest) ---------
# Within each benchmark, single-GPU models first, then multi-GPU.

echo "[upstream-queue] === Phase 1/3: SQuAD pos-robust ===" | tee -a "$LOG"
for ENTRY in "${MODELS_SINGLE[@]}"; do
    read -r TAG MODEL <<<"$ENTRY"
    for M in "${METHODS[@]}"; do sq_run "$TAG" "$MODEL" "$M"; done
done
for ENTRY in "${MODELS_MULTI[@]}"; do
    read -r TAG MODEL EXTRA <<<"$ENTRY"
    for M in "${METHODS[@]}"; do sq_run "$TAG" "$MODEL" "$M" "$EXTRA"; done
done

echo "[upstream-queue] === Phase 2/3: Needle-in-a-haystack ===" | tee -a "$LOG"
for ENTRY in "${MODELS_SINGLE[@]}"; do
    read -r TAG MODEL <<<"$ENTRY"
    for M in "${METHODS[@]}"; do nd_run "$TAG" "$MODEL" "$M"; done
done
for ENTRY in "${MODELS_MULTI[@]}"; do
    read -r TAG MODEL EXTRA <<<"$ENTRY"
    for M in "${METHODS[@]}"; do nd_run "$TAG" "$MODEL" "$M" "$EXTRA"; done
done

echo "[upstream-queue] === Phase 3/3: LongBench ===" | tee -a "$LOG"
for ENTRY in "${MODELS_SINGLE[@]}"; do
    read -r TAG MODEL <<<"$ENTRY"
    for M in "${METHODS[@]}"; do lb_run "$TAG" "$MODEL" "$M"; done
done
for ENTRY in "${MODELS_MULTI[@]}"; do
    read -r TAG MODEL EXTRA <<<"$ENTRY"
    for M in "${METHODS[@]}"; do lb_run "$TAG" "$MODEL" "$M" "$EXTRA"; done
done

echo "[upstream-queue] all done $(date)" | tee -a "$LOG"
ls -la logs/results/upstream_*.json 2>/dev/null | tee -a "$LOG"
