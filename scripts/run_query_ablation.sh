#!/bin/bash
# Query-ablation: rerun pos-robust SQuAD with three scoring-query modes
# (real / trailing / empty) on the SAME checkpoints and SAME samples.
# Outputs land in logs/results/query_ablation_{real,trailing,empty}_{model}.json
#
# Two bugs fixed from the previous version (commit 69be8b9):
#   1. Empty $EXTRA was passed as literal '' to argparse; now we conditionally
#      append --multi_gpu via a bash array.
#   2. Multi-GPU evals (Qwen 32B, Mistral 24B) need CUDA_VISIBLE_DEVICES=0,1
#      to exclude the GT 1030 from device_map='auto' (bnb-4bit rejects
#      CPU-offload requests).

set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
mkdir -p logs/training logs/results
LOG=logs/training/query_ablation_$(date +%Y%m%d_%H%M%S).log
echo "[query-ablation] start $(date)" | tee -a "$LOG"

run_one() {
    # $1 tag, $2 model, $3 ckpt, $4 dev, $5 mode, $6 multi (0|1)
    local TAG="$1" MODEL="$2" CKPT="$3" DEV="$4" MODE="$5" MULTI="$6"
    local SAVE=logs/results/query_ablation_${MODE}_${TAG}.json
    if [ -f "$SAVE" ]; then
        echo "[query-ablation] $TAG $MODE already done -> $SAVE; skipping" | tee -a "$LOG"
        return 0
    fi
    if [ ! -f "$CKPT" ]; then
        echo "[query-ablation] $TAG $MODE checkpoint missing: $CKPT; skipping" | tee -a "$LOG"
        return 1
    fi
    echo "[query-ablation] $TAG $MODE start $(date)" | tee -a "$LOG"
    # Build argv as an array; conditionally append --multi_gpu.
    local args=(
        --model_path "$MODEL"
        --checkpoint "$CKPT"
        --n_eval 100 --seed 42
        --attn_dim 256
        --positions 0 1 2 3
        --query_mode "$MODE" --n_trailing 32
        --device "$DEV"
        --save_path "$SAVE"
    )
    if [ "$MULTI" = "1" ]; then
        args+=(--multi_gpu)
        # Restrict to the two Titan RTX cards (PyTorch order cuda:0, cuda:1)
        # so bnb's device_map='auto' doesn't try to dispatch onto the GT 1030.
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 \
            $PY training/eval_hybrid_query_ablation.py "${args[@]}" 2>&1 | tee -a "$LOG"
    else
        PYTHONUNBUFFERED=1 $PY training/eval_hybrid_query_ablation.py "${args[@]}" 2>&1 | tee -a "$LOG"
    fi
    echo "[query-ablation] $TAG $MODE done $(date) -> $SAVE" | tee -a "$LOG"
}

# Six backbones: (tag, model_path, ckpt, device, multi_gpu_flag)
declare -a MODELS=(
    "qwen7b     ./models/qwen2.5-7b        checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt        cuda:0 0"
    "llama8b    ./models/llama-3.1-8b      checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42_w10_diverse.pt      cuda:0 0"
    "qwen14b    ./models/qwen2.5-14b       checkpoints/astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42_w10_diverse.pt       cuda:0 0"
    "qwen32b    ./models/qwen2.5-32b       checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42_w10_diverse.pt       cuda:0 1"
    "mistral7b  ./models/mistral-7b-v0.3   checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42_w10_diverse.pt   cuda:0 0"
    "mistral24b ./models/mistral-small-24b checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42_w10_diverse.pt cuda:0 1"
)

# Mode order: trailing (most-likely-headline) -> real (control) -> empty (floor)
for MODE in trailing real empty; do
    for ENTRY in "${MODELS[@]}"; do
        read -r TAG MODEL CKPT DEV MULTI <<<"$ENTRY"
        run_one "$TAG" "$MODEL" "$CKPT" "$DEV" "$MODE" "$MULTI"
    done
done

echo "[query-ablation] all done $(date)" | tee -a "$LOG"
ls -la logs/results/query_ablation_*_*.json 2>/dev/null | tee -a "$LOG"
