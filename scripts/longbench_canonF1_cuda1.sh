#!/bin/bash
# Parallel arm of the LongBench canonical-F1 re-evaluation queue.
# Runs the four single-GPU backbones on cuda:1 (PyTorch index, =
# nvidia-smi index 2) while RMT-LoRA full training occupies cuda:0
# (PyTorch index, = nvidia-smi index 1).
#
# Outputs match the naming convention used by scripts/longbench_canonical_reeval.sh
# (logs/results/longbench_<model>_k300_canonF1.json), so the existing
# queue's "if [ -f $save_path ] then skip" guard prevents duplication
# when cuda:0 frees and the multi-GPU arm runs the two big models.

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

Q7B=./models/qwen2.5-7b
Q14B=./models/qwen2.5-14b
L8B=./models/llama-3.1-8b
M7B=./models/mistral-7b-v0.3

Q7B_CKPT=checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt
Q14B_CKPT=checkpoints/astro_hybrid_qwen2_5-14b_n16_k284_t5000_s42.pt
L8B_CKPT=checkpoints/astro_hybrid_llama-3_1-8b_n16_k284_t5000_s42.pt
M7B_CKPT=checkpoints/astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42_w10_diverse.pt

TASKS="hotpotqa multifieldqa_en 2wikimqa musique"
METHODS="streaming_llm h2o snapkv multiplicative hybrid"

mkdir -p logs/results logs/training
LOG=logs/training/longbench_canonF1_cuda1_$(date +%Y%m%d_%H%M%S).log
echo "[lb-canon-cuda1] start $(date)" | tee -a "$LOG"

run_lb () {
    local label="$1" model_path="$2" ckpt="$3" attn_dim="$4"
    local model_name save_path
    model_name=$(basename "$model_path")
    save_path="logs/results/longbench_${model_name}_k300_canonF1.json"

    if [ -f "$save_path" ]; then
        echo "[lb-canon-cuda1] $label already done ($save_path), skipping." | tee -a "$LOG"
        return
    fi

    echo "[lb-canon-cuda1] $label start $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 $PY baselines/eval_longbench.py \
        --model_path "$model_path" \
        --hybrid_checkpoint "$ckpt" \
        --tasks $TASKS \
        --methods $METHODS \
        --k 300 \
        --max_samples 100 \
        --attn_dim $attn_dim \
        --save_path "$save_path" \
        --device cuda:1 2>&1 | tee -a "$LOG"
    echo "[lb-canon-cuda1] $label done $(date) -> $save_path" | tee -a "$LOG"
}

# Small models first so cuda:1 doesn't sit idle if anything stalls
run_lb "Qwen 7B"    "$Q7B"  "$Q7B_CKPT"  512
run_lb "Mistral 7B" "$M7B"  "$M7B_CKPT"  256
run_lb "Llama 8B"   "$L8B"  "$L8B_CKPT"  256
run_lb "Qwen 14B"   "$Q14B" "$Q14B_CKPT" 256

echo "[lb-canon-cuda1] done $(date)" | tee -a "$LOG"
ls -la logs/results/longbench_*_k300_canonF1.json 2>&1 | tee -a "$LOG"
