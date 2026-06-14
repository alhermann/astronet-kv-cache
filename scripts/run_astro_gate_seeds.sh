#!/usr/bin/env bash
# 5-seed CI for AstroGate vs SnapKV on multiquery compress-once SQuAD.
# Same protocol (50p, max_q=4, n_ctx_windows=4, k=300, perlayer); only the
# sample seed varies. Fix best λ per model.
#   Usage:  scripts/run_astro_gate_seeds.sh <model_tag> <device> <lambda>
set -u
cd /home/alexander/Schreibtisch/AstroNet
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
LOGDIR=logs/training/astro_gate_seeds
mkdir -p "$LOGDIR" logs/results

declare -A MP
MP[qwen7b]=./models/qwen2.5-7b
MP[llama8b]=./models/llama-3.1-8b

M=${1:?usage: $0 <model_tag> <device> <lambda>}
DEV=${2:?usage}
LAM=${3:?usage}
CKPT=checkpoints/astro_gate_${M}.pt
[ -f "$CKPT" ] || { echo "Missing $CKPT"; exit 1; }

LTAG="lam${LAM/./p}"
for SEED in 1 2 3 4 5; do
    # AstroGate cell
    SAVE_AG=logs/results/multiquery_squad_astrogate_${M}_k300_${LTAG}_s${SEED}_pl.json
    if [ ! -f "$SAVE_AG" ]; then
        echo "[$(date +%H:%M)] $M astrogate seed=$SEED λ=$LAM"
        $PY -u baselines/eval_multiquery_squad.py \
            --model_path ${MP[$M]} --method astrogate \
            --checkpoint $CKPT --ckpt_tag ${LTAG}_s${SEED} \
            --n_paragraphs 50 --max_q 4 --n_ctx_windows 4 --seed $SEED \
            --k 300 --selection perlayer --lam_override $LAM \
            --device $DEV --save_path $SAVE_AG \
            > "$LOGDIR/${M}_astrogate_s${SEED}.log" 2>&1
    else
        echo "SKIP $SAVE_AG"
    fi
    # SnapKV cell (same seed) for paired comparison
    SAVE_S=logs/results/multiquery_squad_snapkv_${M}_k300_s${SEED}_pl.json
    if [ ! -f "$SAVE_S" ]; then
        echo "[$(date +%H:%M)] $M snapkv seed=$SEED"
        $PY -u baselines/eval_multiquery_squad.py \
            --model_path ${MP[$M]} --method snapkv \
            --n_paragraphs 50 --max_q 4 --n_ctx_windows 4 --seed $SEED \
            --k 300 --selection perlayer \
            --device $DEV --save_path $SAVE_S \
            > "$LOGDIR/${M}_snapkv_s${SEED}.log" 2>&1
    else
        echo "SKIP $SAVE_S"
    fi
done
echo "[$(date +%H:%M)] $M seeds done"
