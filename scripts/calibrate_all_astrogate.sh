#!/usr/bin/env bash
# Auto-calibrate λ for every AstroGate checkpoint we have.
# Writes <ckpt>.lam.json next to each .pt; eval scripts auto-load that
# sidecar when --lam_override is not passed, eliminating per-model
# manual λ tuning.
#
# Usage:
#   scripts/calibrate_all_astrogate.sh A    # cuda:0 lane (Qwen 7B, Mistral 7B)
#   scripts/calibrate_all_astrogate.sh B    # cuda:1 lane (Llama 8B, Qwen 14B)
#   scripts/calibrate_all_astrogate.sh BIG  # multi-GPU (Qwen 32B, Mistral 24B)
#
# Cost per ckpt: ~10–15 min for 7-14B (7 candidates × 10 trials × 4 Q).

set -u
cd /home/alexander/Schreibtisch/AstroNet
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
LOG=logs/training/lambda_calibration
mkdir -p "$LOG"

calibrate() {
    local NAME=$1 MP=$2 CKPT=$3 DEV=$4 EXTRA=$5
    if [ ! -f "$CKPT" ]; then echo "  MISS $CKPT — skip"; return; fi
    if [ -f "${CKPT}.lam.json" ]; then
        echo "  HAVE ${CKPT}.lam.json — skip"; return
    fi
    echo "[$(date +%H:%M)] calibrate $NAME ($DEV)"
    $PY -u scripts/calibrate_astrogate_lambda.py \
        --model_path "$MP" --checkpoint "$CKPT" \
        --device "$DEV" $EXTRA \
        > "$LOG/calibrate_${NAME}.log" 2>&1 \
        && echo "  -> ${CKPT}.lam.json" \
        || echo "  FAIL — see $LOG/calibrate_${NAME}.log"
}

LANE=${1:?usage: $0 A|B|BIG}
case "$LANE" in
    A)
        calibrate qwen7b    ./models/qwen2.5-7b      checkpoints/astro_gate_qwen7b.pt    cuda:0 ""
        calibrate mistral7b ./models/mistral-7b-v0.3 checkpoints/astro_gate_mistral7b.pt cuda:0 ""
        ;;
    B)
        calibrate llama8b ./models/llama-3.1-8b checkpoints/astro_gate_llama8b.pt cuda:1 ""
        calibrate qwen14b ./models/qwen2.5-14b checkpoints/astro_gate_qwen14b.pt cuda:1 ""
        ;;
    BIG)
        # 24B/32B need both GPUs — caller must set CUDA_VISIBLE_DEVICES=0,1
        # and the model must already use device_map='auto' in eval. Today
        # eval_multiquery_needle uses single-device device_map; that's
        # adequate for 14B but not 32B. Mark these as needing the multi-GPU
        # variant before running.
        calibrate qwen32b   ./models/qwen2.5-32b      checkpoints/astro_gate_qwen32b.pt   cuda:0 ""
        calibrate mistral24b ./models/mistral-small-24b checkpoints/astro_gate_mistral24b.pt cuda:0 ""
        ;;
    *) echo "unknown lane $LANE"; exit 1 ;;
esac

echo "[$(date +%H:%M)] $LANE done"
