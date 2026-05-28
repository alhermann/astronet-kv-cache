#!/bin/bash
# Re-run the two multi-GPU LongBench canonical-F1 evals that failed when
# cuda:1 was still busy with the duplicate Qwen 14B run.
#
# Waits for the duplicate Qwen 14B (cuda:1 single-GPU process) to exit,
# then runs Qwen 32B and Mistral-Small 24B sequentially with both Titans
# free.

set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"

Q32B=./models/qwen2.5-32b
M24B=./models/mistral-small-24b
Q32B_CKPT=checkpoints/astro_hybrid_qwen2_5-32b_n16_k284_t5000_s42.pt
M24B_CKPT=checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42_w10_diverse.pt

TASKS="hotpotqa multifieldqa_en 2wikimqa musique"
METHODS="streaming_llm h2o snapkv multiplicative hybrid"

mkdir -p logs/results logs/training
LOG=logs/training/longbench_canonF1_multigpu_$(date +%Y%m%d_%H%M%S).log
echo "[lb-multigpu] start $(date)" | tee -a "$LOG"

# Wait for the cuda:1 single-GPU duplicate Qwen 14B to exit
echo "[lb-multigpu] waiting for any single-GPU eval_longbench on cuda:1 to finish..." | tee -a "$LOG"
until ! pgrep -fa "eval_longbench.py.*--device cuda:1" >/dev/null; do
    sleep 30
done
echo "[lb-multigpu] cuda:1 free at $(date), proceeding." | tee -a "$LOG"

run_lb_multi () {
    local label="$1" model_path="$2" ckpt="$3"
    local model_name save_path
    model_name=$(basename "$model_path")
    save_path="logs/results/longbench_${model_name}_k300_canonF1.json"

    if [ -f "$save_path" ]; then
        echo "[lb-multigpu] $label already done, skipping." | tee -a "$LOG"
        return
    fi

    echo "[lb-multigpu] $label start $(date)" | tee -a "$LOG"
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 $PY baselines/eval_longbench.py \
        --model_path "$model_path" \
        --hybrid_checkpoint "$ckpt" \
        --tasks $TASKS \
        --methods $METHODS \
        --k 300 \
        --max_samples 100 \
        --attn_dim 256 \
        --save_path "$save_path" \
        --multi_gpu 2>&1 | tee -a "$LOG"
    echo "[lb-multigpu] $label done $(date) -> $save_path" | tee -a "$LOG"
}

run_lb_multi "Qwen 32B"          "$Q32B" "$Q32B_CKPT"
run_lb_multi "Mistral-Small 24B" "$M24B" "$M24B_CKPT"

echo "[lb-multigpu] done $(date)" | tee -a "$LOG"
ls -la logs/results/longbench_*_k300_canonF1.json 2>&1 | tee -a "$LOG"
