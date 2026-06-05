#!/bin/bash
# Master pipeline: launches X1 (arch only) on cuda:0 and X2 (arch+multihop)
# on cuda:1 once both GPUs are free.  Each takes ~10-14h on Qwen 7B.
# After both finish, evaluates with eval_hybrid_swap_selector and
# aggregates the deltas vs baseline.

set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
mkdir -p logs/training checkpoints
LOG=logs/training/x1_x2_pipeline_$(date +%Y%m%d_%H%M%S).log
echo "[x1x2] start $(date)" | tee -a "$LOG"

# --- 0) Wait until both Titan RTX GPUs are mostly free
# nvidia-smi numbering: 0=GT 1030 (display, always idle), 1+2 = Titans.
# Allow up to 2GB on the display-attached Titan (nvidia-smi GPU 2).
wait_for_gpus() {
    while true; do
        local titan1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
        local titan2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)
        if [ "$titan1" -lt 1000 ] && [ "$titan2" -lt 2000 ]; then
            echo "[wait] both Titans free (T1=${titan1}MiB, T2=${titan2}MiB)" | tee -a "$LOG"
            return 0
        fi
        echo "[wait] T1=${titan1}MiB T2=${titan2}MiB; sleeping 60s" | tee -a "$LOG"
        sleep 60
    done
}

wait_for_gpus

# --- 1) Launch X1 (arch only, SQuAD) on cuda:0 in background
echo "[x1x2] launching X1 (arch, SQuAD) on cuda:0" | tee -a "$LOG"
CKPT_X1=checkpoints/astro_x1_qwen7b_squad.pt
nohup bash -c "
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PY \
    training/train_hybrid_x1.py \
    --model_path ./models/qwen2.5-7b \
    --init_checkpoint checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt \
    --n_train 5000 --epochs 2 --lr 5e-4 \
    --data squad \
    --device cuda:0 \
    --save_path $CKPT_X1
" > logs/training/x1_train_squad.log 2>&1 &
X1_PID=$!
echo "[x1x2] X1 PID=$X1_PID, log=logs/training/x1_train_squad.log" | tee -a "$LOG"

# --- 2) Launch X2 (arch + multi-hop mix) on cuda:1 in background
echo "[x1x2] launching X2 (arch+multihop) on cuda:1" | tee -a "$LOG"
CKPT_X2=checkpoints/astro_x1_qwen7b_multihop.pt
nohup bash -c "
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 $PY \
    training/train_hybrid_x1.py \
    --model_path ./models/qwen2.5-7b \
    --init_checkpoint checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42_w10_diverse.pt \
    --n_train 5000 --epochs 2 --lr 5e-4 \
    --data multihop \
    --device cuda:0 \
    --save_path $CKPT_X2
" > logs/training/x1_train_multihop.log 2>&1 &
X2_PID=$!
echo "[x1x2] X2 PID=$X2_PID, log=logs/training/x1_train_multihop.log" | tee -a "$LOG"

# --- 3) Wait for both to finish
echo "[x1x2] waiting for X1 ($X1_PID) and X2 ($X2_PID) to finish..." | tee -a "$LOG"
wait $X1_PID; echo "[x1x2] X1 done $(date)" | tee -a "$LOG"
wait $X2_PID; echo "[x1x2] X2 done $(date)" | tee -a "$LOG"

# --- 4) Evaluate both on the swap-selector benchmark
# (compare pure_S1 / hybrid_S1 / hybrid_swap before vs after X1)
echo "[x1x2] evaluating X1 checkpoint" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PY \
    training/eval_hybrid_swap_selector.py \
    --model_path ./models/qwen2.5-7b \
    --checkpoint $CKPT_X1 \
    --selector snapkv --n_eval 100 \
    --positions 0 1 2 3 \
    --device cuda:0 \
    --save_path logs/results/e3_x1_qwen7b.json 2>&1 | tee -a "$LOG"

echo "[x1x2] evaluating X2 checkpoint" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 $PY \
    training/eval_hybrid_swap_selector.py \
    --model_path ./models/qwen2.5-7b \
    --checkpoint $CKPT_X2 \
    --selector snapkv --n_eval 100 \
    --positions 0 1 2 3 \
    --device cuda:0 \
    --save_path logs/results/e3_x2_qwen7b.json 2>&1 | tee -a "$LOG"

echo "[x1x2] pipeline done $(date)" | tee -a "$LOG"
ls -la logs/results/e3_x*_qwen7b.json | tee -a "$LOG"
