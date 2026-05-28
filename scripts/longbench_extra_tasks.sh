#!/bin/bash
# Run LongBench MuSiQue + 2WikiMultihopQA on all six backbones, with
# all five methods (streaming_llm, h2o, snapkv, multiplicative, hybrid).
#
# Both tasks are extractive QA (consistent with the existing HotpotQA +
# MultiFieldQA scope). MuSiQue is a 2-4 hop multi-doc QA dataset
# designed to stress retrieval pipelines; 2WikiMultihopQA is a
# HotpotQA-family multi-hop benchmark with disambiguation distractors.
#
# Saves to unique filenames so existing longbench_*.json files are
# preserved:
#   logs/results/longbench_<model>_k300_extra2tasks.json
#
# Waits for the RULER queue to finish (its "[ruler-4] done" marker)
# before starting, so GPU contention is avoided.

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

Q7B=./models/qwen2.5-7b
Q14B=./models/qwen2.5-14b
Q32B=./models/qwen2.5-32b
L8B=./models/llama-3.1-8b
M7B=./models/mistral-7b-v0.3
M24B=./models/mistral-small-24b

Q7B_CKPT=checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt
Q14B_CKPT=checkpoints/astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42.pt
Q32B_CKPT=checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42.pt
L8B_CKPT=checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42.pt
M7B_CKPT=checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42_w10_diverse.pt
M24B_CKPT=checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42_w10_diverse.pt

TASKS="2wikimqa musique"
METHODS="streaming_llm h2o snapkv multiplicative hybrid"

mkdir -p logs/results logs/training
LOG=logs/training/longbench_extra2_$(date +%Y%m%d_%H%M%S).log
echo "[lb-extra] start $(date)" | tee -a "$LOG"

# --- Wait for the RULER expansion to finish ---------------------------
RULER_LOG=$(ls -t logs/training/ruler_remaining4_*.log 2>/dev/null | head -1)
if [ -n "${RULER_LOG:-}" ] && [ -f "$RULER_LOG" ]; then
    echo "[lb-extra] waiting for $RULER_LOG to finish (poll every 60s)..." | tee -a "$LOG"
    until grep -q "\[ruler-4\] done" "$RULER_LOG"; do
        sleep 60
    done
    echo "[lb-extra] RULER queue done at $(date), proceeding." | tee -a "$LOG"
else
    echo "[lb-extra] no RULER log found, proceeding immediately." | tee -a "$LOG"
fi

run_lb () {
    local label="$1" model_path="$2" ckpt="$3" attn_dim="$4" multi="$5"
    local model_name save_path
    model_name=$(basename "$model_path")
    save_path="logs/results/longbench_${model_name}_k300_extra2tasks.json"

    if [ -f "$save_path" ]; then
        echo "[lb-extra] $label already done ($save_path), skipping." | tee -a "$LOG"
        return
    fi

    echo "[lb-extra] $label start $(date)" | tee -a "$LOG"
    local cuda_env extra
    if [ "$multi" = "1" ]; then
        cuda_env="CUDA_VISIBLE_DEVICES=0,1"
        extra="--multi_gpu"
    else
        cuda_env=""
        extra="--device cuda:0"
    fi
    eval "PYTHONUNBUFFERED=1 $cuda_env $PY baselines/eval_longbench.py \
        --model_path \"$model_path\" \
        --hybrid_checkpoint \"$ckpt\" \
        --tasks $TASKS \
        --methods $METHODS \
        --k 300 \
        --max_samples 100 \
        --attn_dim $attn_dim \
        --save_path \"$save_path\" \
        $extra 2>&1 | tee -a $LOG"
    echo "[lb-extra] $label done $(date) -> $save_path" | tee -a "$LOG"
}

# --- Run in order of increasing footprint -----------------------------
# Qwen 7B uses attn_dim=512; all others use 256.
run_lb "Qwen 7B"           "$Q7B"  "$Q7B_CKPT"  512 0
run_lb "Mistral 7B"        "$M7B"  "$M7B_CKPT"  256 0
run_lb "Llama 8B"          "$L8B"  "$L8B_CKPT"  256 0
run_lb "Qwen 14B"          "$Q14B" "$Q14B_CKPT" 256 0
run_lb "Qwen 32B"          "$Q32B" "$Q32B_CKPT" 256 1
run_lb "Mistral-Small 24B" "$M24B" "$M24B_CKPT" 256 1

echo "[lb-extra] done $(date)" | tee -a "$LOG"
ls -la logs/results/longbench_*_k300_extra2tasks.json 2>&1 | tee -a "$LOG"
