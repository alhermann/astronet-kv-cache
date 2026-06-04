#!/bin/bash
# Run faithful SnapKV / H2O / PyramidKV across all 6 backbones on three
# benchmarks: position-robust SQuAD, needle-in-a-haystack, and LongBench.
#
# Replaces the unfaithful baselines (4-layer "SnapKV", global cumulative
# "H2O", H2O-with-pyramid-budgets "PyramidKV") that the new faithful
# implementations supersede.  See baselines/faithful_*.py for the
# faithful selectors and the corresponding eval_faithful_*.py runners.
#
# Single-card models: Qwen 7B, Llama 8B, Qwen 14B, Mistral 7B.
# Multi-GPU models: Qwen 32B, Mistral-Small 24B (use --multi_gpu).

set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
mkdir -p logs/training logs/results
LOG=logs/training/faithful_baselines_queue_$(date +%Y%m%d_%H%M%S).log
echo "[faithful-baselines-queue] start $(date)" | tee -a "$LOG"

# Wait for any in-flight Qwen 7B SQuAD run (the smoke-finishing one) to
# release cuda:0 before kicking off the queue.
while pgrep -f "eval_faithful_baselines.py.*qwen2.5-7b" >/dev/null; do
    echo "[queue] waiting for in-flight Qwen 7B SQuAD ..." | tee -a "$LOG"
    sleep 60
done

# --- SQuAD: position-robust, n=100, 4 positions, k=300 ---
sq_run() {
    local TAG="$1" MODEL="$2" EXTRA="${3:-}"
    local SAVE=logs/results/faithful_squad_${TAG}.json
    if [ -f "$SAVE" ]; then
        echo "[queue] $TAG squad already done; skipping" | tee -a "$LOG"
        return 0
    fi
    echo "[queue] $TAG squad start $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/eval_faithful_baselines.py \
        --model_path "$MODEL" --n_eval 100 --seed 42 --k 300 \
        --positions 0 1 2 3 \
        --methods snapkv h2o pyramidkv \
        --device cuda:0 $EXTRA \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[queue] $TAG squad done $(date) -> $SAVE" | tee -a "$LOG"
}

needle_run() {
    local TAG="$1" MODEL="$2" EXTRA="${3:-}"
    local SAVE=logs/results/faithful_needle_${TAG}.json
    if [ -f "$SAVE" ]; then
        echo "[queue] $TAG needle already done; skipping" | tee -a "$LOG"
        return 0
    fi
    echo "[queue] $TAG needle start $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/eval_faithful_needle.py \
        --model_path "$MODEL" \
        --n_windows_list 5 10 20 \
        --n_trials 20 --k 300 \
        --methods snapkv h2o pyramidkv \
        --device cuda:0 $EXTRA \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[queue] $TAG needle done $(date) -> $SAVE" | tee -a "$LOG"
}

longbench_run() {
    local TAG="$1" MODEL="$2" EXTRA="${3:-}"
    local SAVE=logs/results/faithful_longbench_${TAG}.json
    if [ -f "$SAVE" ]; then
        echo "[queue] $TAG longbench already done; skipping" | tee -a "$LOG"
        return 0
    fi
    echo "[queue] $TAG longbench start $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/eval_faithful_longbench.py \
        --model_path "$MODEL" \
        --tasks hotpotqa multifieldqa_en 2wikimqa musique \
        --n_samples 100 --k 300 \
        --methods snapkv h2o pyramidkv \
        --device cuda:0 $EXTRA \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[queue] $TAG longbench done $(date) -> $SAVE" | tee -a "$LOG"
}

# Order: single-GPU models first, then multi-GPU.
# SQuAD across all 6 first (fastest), then needle (medium), then LongBench (slowest).

declare -a MODELS_SINGLE=(
    "qwen7b    ./models/qwen2.5-7b"
    "llama8b   ./models/llama-3.1-8b"
    "qwen14b   ./models/qwen2.5-14b"
    "mistral7b ./models/mistral-7b-v0.3"
)
declare -a MODELS_MULTI=(
    "qwen32b   ./models/qwen2.5-32b   --multi_gpu"
    "mistral24b ./models/mistral-small-24b --multi_gpu"
)

echo "[queue] === SQuAD across all 6 backbones ===" | tee -a "$LOG"
for ENTRY in "${MODELS_SINGLE[@]}"; do
    read -r TAG MODEL <<<"$ENTRY"
    sq_run "$TAG" "$MODEL"
done
for ENTRY in "${MODELS_MULTI[@]}"; do
    read -r TAG MODEL EXTRA <<<"$ENTRY"
    sq_run "$TAG" "$MODEL" "$EXTRA"
done

echo "[queue] === Needle across all 6 backbones ===" | tee -a "$LOG"
for ENTRY in "${MODELS_SINGLE[@]}"; do
    read -r TAG MODEL <<<"$ENTRY"
    needle_run "$TAG" "$MODEL"
done
for ENTRY in "${MODELS_MULTI[@]}"; do
    read -r TAG MODEL EXTRA <<<"$ENTRY"
    needle_run "$TAG" "$MODEL" "$EXTRA"
done

echo "[queue] === LongBench across all 6 backbones ===" | tee -a "$LOG"
for ENTRY in "${MODELS_SINGLE[@]}"; do
    read -r TAG MODEL <<<"$ENTRY"
    longbench_run "$TAG" "$MODEL"
done
for ENTRY in "${MODELS_MULTI[@]}"; do
    read -r TAG MODEL EXTRA <<<"$ENTRY"
    longbench_run "$TAG" "$MODEL" "$EXTRA"
done

echo "[faithful-baselines-queue] all done $(date)" | tee -a "$LOG"
ls -la logs/results/faithful_*_{qwen,llama,mistral}*.json | tee -a "$LOG"
