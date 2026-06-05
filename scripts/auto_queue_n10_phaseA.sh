#!/bin/bash
# Auto-launch A3 (Qwen 14B on cuda:0) when A1 (Qwen 7B) completes,
# then A4 (Qwen 32B multi-GPU) when both A2 (Llama 8B) and A3 finish.
# Runs in background; polls every 5 minutes.

set -uo pipefail
cd "$(dirname "$0")/.."
QLOG=/home/alexander/Schreibtisch/AstroNet/logs/training/auto_queue_phaseA.log
mkdir -p "$(dirname "$QLOG")"
echo "[auto-queue] start $(date)" >> "$QLOG"

# Wait for Qwen 7B
while pgrep -f "train_hybrid.*qwen2.5-7b.*cuda:0" >/dev/null; do
    sleep 300
done
echo "[auto-queue] $(date) — Qwen 7B done; launching Qwen 14B on cuda:0" >> "$QLOG"
nohup bash scripts/run_n10_mixed_sensitivity_qwen14b.sh >/dev/null 2>&1 &
disown
sleep 120  # give it time to start loading

# Wait for BOTH Qwen 14B and Llama 8B
while pgrep -f "train_hybrid.*qwen2.5-14b" >/dev/null \
      || pgrep -f "train_hybrid.*llama-3.1-8b" >/dev/null; do
    sleep 300
done
echo "[auto-queue] $(date) — Qwen 14B and Llama 8B done; launching Qwen 32B multi-GPU" >> "$QLOG"
nohup bash scripts/run_n10_mixed_sensitivity_qwen32b.sh >/dev/null 2>&1 &
disown
echo "[auto-queue] $(date) — Qwen 32B launched; queue complete" >> "$QLOG"
