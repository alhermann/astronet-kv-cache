#!/usr/bin/env bash
# AstroGate multiquery eval (2026-06-13). The trained gate emits a per-token
# bias on SnapKV scores; λ scales that bias. Distillation training doesn't
# set λ, so sweep at eval time. n=50 paragraphs to match the existing
# multiquery_squad_*_qwen7b_k300_pl.json cells.
#
#   Existing anchors on Qwen 7B (50p, max_q=4, k=300, perlayer):
#     full           = 84.07%   (ceiling)
#     snapkv_oracle  = 82.42%   (query-aware ref)
#     snapkv         = 42.86%   (query-agnostic baseline)
#     astrohybrid    = 34.07%   (the regression we're fixing)
#
#   AstroGate target: beat 42.86%.
#
# Usage:  scripts/run_astro_gate_sweep.sh <model_tag> <device>
#         e.g.   scripts/run_astro_gate_sweep.sh qwen7b cuda:0
set -u
cd /home/alexander/Schreibtisch/AstroNet
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
LOGDIR=logs/training/astro_gate
mkdir -p "$LOGDIR" logs/results

declare -A MP
MP[qwen7b]=./models/qwen2.5-7b
MP[llama8b]=./models/llama-3.1-8b
MP[qwen14b]=./models/qwen2.5-14b
MP[mistral7b]=./models/mistral-7b-v0.3

M=${1:?usage: $0 <model_tag> <device>}
DEV=${2:?usage: $0 <model_tag> <device>}
CKPT=checkpoints/astro_gate_${M}.pt
[ -f "$CKPT" ] || { echo "Missing $CKPT — train first."; exit 1; }

for LAM in 0.13 0.5 1.0 3.0 10.0; do
    TAG="lam${LAM/./p}"
    SAVE=logs/results/multiquery_squad_astrogate_${M}_k300_${TAG}_pl.json
    [ -f "$SAVE" ] && { echo "SKIP $SAVE"; continue; }
    echo "[$(date +%H:%M)] $M astrogate λ=$LAM"
    $PY -u baselines/eval_multiquery_squad.py \
        --model_path ${MP[$M]} --method astrogate \
        --checkpoint $CKPT --ckpt_tag $TAG \
        --n_paragraphs 50 --max_q 4 --n_ctx_windows 4 --seed 42 \
        --k 300 --selection perlayer --lam_override $LAM \
        --device $DEV --save_path $SAVE \
        > "$LOGDIR/eval_${M}_${TAG}.log" 2>&1
done
echo "[$(date +%H:%M)] $M sweep done"
