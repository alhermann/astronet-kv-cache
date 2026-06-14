#!/usr/bin/env bash
# 5-seed CI for AstroGainE2E (end-to-end NLL training) vs SnapKV.
# n=100 paragraphs to match the v2 protocol.
#   Usage:  scripts/run_astro_gain_e2e_seeds.sh <model_tag> <device>
set -u
cd /home/alexander/Schreibtisch/AstroNet
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
LOGDIR=logs/training/astro_gain_e2e_seeds
mkdir -p "$LOGDIR" logs/results

declare -A MP
MP[qwen7b]=./models/qwen2.5-7b
MP[llama8b]=./models/llama-3.1-8b

M=${1:?usage: $0 <model_tag> <device>}
DEV=${2:?usage}
CKPT=checkpoints/astro_gain_e2e_${M}.pt
[ -f "$CKPT" ] || { echo "Missing $CKPT"; exit 1; }

for SEED in 1 2 3 4 5; do
    SAVE_AG=logs/results/multiquery_squad_astrogain_e2e_${M}_k300_s${SEED}_n100_pl.json
    if [ ! -f "$SAVE_AG" ]; then
        echo "[$(date +%H:%M)] $M astrogain_e2e seed=$SEED"
        $PY -u baselines/eval_multiquery_squad.py \
            --model_path ${MP[$M]} --method astrogain_e2e \
            --checkpoint $CKPT --ckpt_tag e2e_s${SEED}_n100 \
            --n_paragraphs 100 --max_q 4 --n_ctx_windows 4 --seed $SEED \
            --k 300 --selection perlayer \
            --device $DEV --save_path $SAVE_AG \
            > "$LOGDIR/${M}_astrogain_e2e_s${SEED}.log" 2>&1
    else
        echo "SKIP $SAVE_AG"
    fi
    # Reuse the existing snapkv s_n100 baselines from v2 run
done
echo "[$(date +%H:%M)] $M e2e seeds done"
