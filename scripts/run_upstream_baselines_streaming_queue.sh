#!/bin/bash
# Streaming-mode baseline queue: SnapKV / H2O / PyramidKV under the same
# AstroNet-S1 protocol (cumulative KV, question-conditioned scoring, per-
# method pooling+selection rule).  Apples-to-apples comparison vs hybrid.
#
# Matrix: 6 models x 3 methods x 2 budgets (k=300, k=284) x 3 benchmarks
#         = 108 cells.  Skip-if-exists.

set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
mkdir -p logs/training logs/results
LOG=logs/training/upstream_streaming_queue_$(date +%Y%m%d_%H%M%S).log
echo "[streaming-queue] start $(date)" | tee -a "$LOG"

DEVICE=${DEVICE:-cuda:1}   # default to cuda:1 to coexist with single-prompt queue
CUDA_DEV_VIS=${CUDA_DEV_VIS:-1}

sq_run() {
    local TAG="$1" MODEL="$2" METHOD="$3" K="$4" EXTRA="${5:-}"
    local SAVE=logs/results/upstream_squad_${METHOD}_${TAG}_k${K}_streaming.json
    if [ -f "$SAVE" ]; then
        echo "[skip] $TAG $METHOD k=$K squad already done" | tee -a "$LOG"; return 0
    fi
    echo "[run]  $TAG $METHOD k=$K squad $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$CUDA_DEV_VIS PYTHONUNBUFFERED=1 $PY \
        baselines/eval_upstream_baselines_streaming.py \
        --model_path "$MODEL" --method "$METHOD" --k "$K" --n_eval 100 \
        --seed 42 --positions 0 1 2 3 $EXTRA \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[done] $TAG $METHOD k=$K squad $(date) -> $SAVE" | tee -a "$LOG"
}

nd_run() {
    local TAG="$1" MODEL="$2" METHOD="$3" K="$4" EXTRA="${5:-}"
    local SAVE=logs/results/upstream_needle_${METHOD}_${TAG}_k${K}_streaming.json
    if [ -f "$SAVE" ]; then
        echo "[skip] $TAG $METHOD k=$K needle already done" | tee -a "$LOG"; return 0
    fi
    echo "[run]  $TAG $METHOD k=$K needle $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$CUDA_DEV_VIS PYTHONUNBUFFERED=1 $PY \
        baselines/eval_upstream_needle_streaming.py \
        --model_path "$MODEL" --method "$METHOD" --k "$K" \
        --n_windows_list 5 10 20 --n_trials 20 $EXTRA \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[done] $TAG $METHOD k=$K needle $(date) -> $SAVE" | tee -a "$LOG"
}

lb_run() {
    local TAG="$1" MODEL="$2" METHOD="$3" K="$4" EXTRA="${5:-}"
    local SAVE=logs/results/upstream_longbench_${METHOD}_${TAG}_k${K}_streaming.json
    if [ -f "$SAVE" ]; then
        echo "[skip] $TAG $METHOD k=$K longbench already done" | tee -a "$LOG"; return 0
    fi
    echo "[run]  $TAG $METHOD k=$K longbench $(date)" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=$CUDA_DEV_VIS PYTHONUNBUFFERED=1 $PY \
        baselines/eval_upstream_longbench_streaming.py \
        --model_path "$MODEL" --method "$METHOD" --k "$K" \
        --tasks hotpotqa multifieldqa_en 2wikimqa musique \
        --n_samples 100 --max_chunks 20 $EXTRA \
        --save_path "$SAVE" 2>&1 | tee -a "$LOG"
    echo "[done] $TAG $METHOD k=$K longbench $(date) -> $SAVE" | tee -a "$LOG"
}

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
BUDGETS=(300 284)

# Order: SQuAD (fastest, most important) -> Needle -> LongBench (slowest)
echo "[streaming-queue] === Phase 1/3: SQuAD pos-robust ===" | tee -a "$LOG"
for ENTRY in "${MODELS_SINGLE[@]}"; do
    read -r TAG MODEL <<<"$ENTRY"
    for K in "${BUDGETS[@]}"; do for M in "${METHODS[@]}"; do
        sq_run "$TAG" "$MODEL" "$M" "$K"
    done; done
done
for ENTRY in "${MODELS_MULTI[@]}"; do
    read -r TAG MODEL EXTRA <<<"$ENTRY"
    for K in "${BUDGETS[@]}"; do for M in "${METHODS[@]}"; do
        sq_run "$TAG" "$MODEL" "$M" "$K" "$EXTRA"
    done; done
done

echo "[streaming-queue] === Phase 2/3: Needle ===" | tee -a "$LOG"
for ENTRY in "${MODELS_SINGLE[@]}"; do
    read -r TAG MODEL <<<"$ENTRY"
    for K in "${BUDGETS[@]}"; do for M in "${METHODS[@]}"; do
        nd_run "$TAG" "$MODEL" "$M" "$K"
    done; done
done
for ENTRY in "${MODELS_MULTI[@]}"; do
    read -r TAG MODEL EXTRA <<<"$ENTRY"
    for K in "${BUDGETS[@]}"; do for M in "${METHODS[@]}"; do
        nd_run "$TAG" "$MODEL" "$M" "$K" "$EXTRA"
    done; done
done

echo "[streaming-queue] === Phase 3/3: LongBench ===" | tee -a "$LOG"
for ENTRY in "${MODELS_SINGLE[@]}"; do
    read -r TAG MODEL <<<"$ENTRY"
    for K in "${BUDGETS[@]}"; do for M in "${METHODS[@]}"; do
        lb_run "$TAG" "$MODEL" "$M" "$K"
    done; done
done
for ENTRY in "${MODELS_MULTI[@]}"; do
    read -r TAG MODEL EXTRA <<<"$ENTRY"
    for K in "${BUDGETS[@]}"; do for M in "${METHODS[@]}"; do
        lb_run "$TAG" "$MODEL" "$M" "$K" "$EXTRA"
    done; done
done

echo "[streaming-queue] all done $(date)" | tee -a "$LOG"
ls -la logs/results/upstream_*_streaming.json 2>/dev/null | tee -a "$LOG"
