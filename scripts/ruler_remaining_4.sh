#!/bin/bash
# Run RULER (n_windows = 22, 44, 85 -> ~8k, 16k, 32k tokens) on the four
# backbones not yet covered: Qwen 7B, Mistral 7B, Qwen 32B, Mistral-Small 24B.
#
# Single-seed (consistent with existing Qwen 14B + Llama 8B RULER cells),
# 10 trials per depth, 5 methods (streaming_llm, h2o, snapkv,
# multiplicative, hybrid). Output filenames mirror the existing pattern
# `ruler_{model_name}_k300.json` (renamed from the eval_needle.py default).
#
# Designed to start AFTER the current k-confound reseed finishes, by
# polling for the [reseed-rem] done marker in its log.

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

Q7B=./models/qwen2.5-7b
Q7B_CKPT=checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt

Q32B=./models/qwen2.5-32b
Q32B_CKPT=checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42.pt

M7B=./models/mistral-7b-v0.3
M7B_CKPT=checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42_w10_diverse.pt

M24B=./models/mistral-small-24b
M24B_CKPT=checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42_w10_diverse.pt

mkdir -p logs/results logs/training
LOG=logs/training/ruler_remaining4_$(date +%Y%m%d_%H%M%S).log
echo "[ruler-4] start $(date)" | tee -a "$LOG"

# --- 1. Wait for the in-flight k-confound reseed to finish ------------
KC_LOG=$(ls -t logs/training/reseed_kconfound_remaining_*.log 2>/dev/null | head -1)
if [ -n "${KC_LOG:-}" ] && [ -f "$KC_LOG" ]; then
    echo "[ruler-4] waiting for $KC_LOG to finish (poll every 60s)..." | tee -a "$LOG"
    until grep -q "\[reseed-rem\] done" "$KC_LOG"; do
        sleep 60
    done
    echo "[ruler-4] k-confound reseed done at $(date), proceeding." | tee -a "$LOG"
else
    echo "[ruler-4] no k-confound reseed log found, proceeding immediately." | tee -a "$LOG"
fi

run_ruler () {
    local label="$1" model_path="$2" ckpt="$3" attn_dim="$4" multi="$5"
    local model_name
    model_name=$(basename "$model_path")
    local out_default="logs/results/needle_${model_name}_k300_ruler.json"
    local out_final="logs/results/ruler_${model_name}_k300.json"

    echo "[ruler-4] $label start $(date)" | tee -a "$LOG"
    local cuda_env=""
    local extra=""
    if [ "$multi" = "1" ]; then
        cuda_env="CUDA_VISIBLE_DEVICES=0,1"
        extra="--multi_gpu"
    else
        extra="--device cuda:0"
    fi
    eval "PYTHONUNBUFFERED=1 $cuda_env $PY baselines/eval_needle.py \
        --model_path \"$model_path\" \
        --hybrid_checkpoint \"$ckpt\" \
        --k 300 \
        --n_windows_list 22 44 85 \
        --n_trials 10 \
        --methods streaming_llm h2o snapkv multiplicative hybrid \
        --attn_dim $attn_dim \
        --save_suffix _ruler \
        $extra 2>&1 | tee -a $LOG"

    # Rename to canonical ruler_*.json so the plot script picks it up.
    if [ -f "$out_default" ]; then
        mv -n "$out_default" "$out_final" && echo "[ruler-4] $label -> $out_final" | tee -a "$LOG"
    else
        echo "[ruler-4] WARNING: $out_default missing for $label" | tee -a "$LOG"
    fi
}

# --- 2. Run the 4 backbones in order of increasing GPU footprint ------
# Qwen 7B uses attn_dim=512 (per training config); all others use 256.
run_ruler "Qwen 7B"          "$Q7B"  "$Q7B_CKPT"  512 0
run_ruler "Mistral 7B"       "$M7B"  "$M7B_CKPT"  256 0
run_ruler "Qwen 32B"         "$Q32B" "$Q32B_CKPT" 256 1
run_ruler "Mistral-Small 24B" "$M24B" "$M24B_CKPT" 256 1

echo "[ruler-4] done $(date)" | tee -a "$LOG"
ls -la logs/results/ruler_*.json 2>&1 | tee -a "$LOG"
