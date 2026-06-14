#!/usr/bin/env bash
# Multi-query NIAH eval for additional backbones (Qwen 14B, Mistral 7B).
# Default config (5-seed CI) + k-scan + ctx-scan, mirroring the Qwen 7B /
# Llama 8B battery so we can include all four models in the same paper table.
# Lane A = cuda:0 (mistral7b), Lane B = cuda:1 (qwen14b).
set -u
cd /home/alexander/Schreibtisch/AstroNet
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
LOGDIR=logs/training/mqneedle
mkdir -p "$LOGDIR" logs/results

declare -A MP LAMG MGPU
MP[mistral7b]=./models/mistral-7b-v0.3
MP[qwen14b]=./models/qwen2.5-14b
MP[qwen32b]=./models/qwen2.5-32b
MP[mistral24b]=./models/mistral-small-24b
# Use the v1 default λ (0.5) as a starting point — we'll sweep at eval.
LAMG[mistral7b]=0.5
LAMG[qwen14b]=0.5
LAMG[qwen32b]=0.5
LAMG[mistral24b]=0.5
# 32B/24B need both GPUs split.
MGPU[qwen32b]=1
MGPU[mistral24b]=1

LANE=${1:?usage: $0 A|B|C|D}
case "$LANE" in
    A) M=mistral7b; DEV=cuda:0 ;;
    B) M=qwen14b;   DEV=cuda:1 ;;
    C) M=qwen32b;   DEV=cuda:0 ;;   # multi_gpu — DEV is "input" lane only
    D) M=mistral24b; DEV=cuda:0 ;;  # multi_gpu — DEV is "input" lane only
esac
EXTRA_LOAD=""
[ "${MGPU[$M]:-0}" = "1" ] && EXTRA_LOAD="--multi_gpu"
LAM=${LAMG[$M]}
CKPT=checkpoints/astro_gate_${M}.pt
[ -f "$CKPT" ] || { echo "Missing $CKPT — train first."; exit 1; }

run_cell() {
    local meth=$1 NW=$2 NN=$3 K=$4 SEED=$5 NT=$6 tag=$7 extra=$8
    local save=logs/results/mqneedle_${meth}_${M}_w${NW}_n${NN}_k${K}_s${SEED}_t${NT}_${tag}.json
    [ -f "$save" ] && { echo "  SKIP $save"; return; }
    echo "  [$(date +%H:%M)] $M $meth w${NW} n${NN} k${K} s${SEED} t${NT} $tag"
    $PY -u baselines/eval_multiquery_needle.py \
        --model_path ${MP[$M]} --method $meth \
        --n_windows $NW --n_needles $NN --n_trials $NT --seed $SEED \
        --k $K $extra $EXTRA_LOAD \
        --device $DEV --save_path $save \
        > "$LOGDIR/$(basename $save .json).log" 2>&1
}

echo "[$M lane $LANE] start $(date)"

# Quick anchor run first (seed=42 to match the original 80-question scale)
echo "=== anchor (seed=42 like Qwen 7B / Llama 8B) ==="
run_cell full           100 4 300 42 20 anchor ""
run_cell snapkv_oracle  100 4 300 42 20 anchor ""
run_cell snapkv         100 4 300 42 20 anchor ""
run_cell astrogate      100 4 300 42 20 anchor "--checkpoint $CKPT --ckpt_tag anchor --lam_override $LAM"

# λ sweep at seed=42 to find this model's optimum
echo "=== λ sweep at seed=42 ==="
for LAM_SWEEP in 0.13 1.0 3.0 10.0 30.0 100.0; do
    LTAG="lam${LAM_SWEEP/./p}"
    SAVE=logs/results/mqneedle_astrogate_${M}_w100_n4_k300_s42_t20_${LTAG}.json
    [ -f "$SAVE" ] && { echo "  SKIP $SAVE"; continue; }
    echo "  [$(date +%H:%M)] $M astrogate λ=$LAM_SWEEP"
    $PY -u baselines/eval_multiquery_needle.py \
        --model_path ${MP[$M]} --method astrogate \
        --checkpoint $CKPT --ckpt_tag $LTAG \
        --n_windows 100 --n_needles 4 --n_trials 20 --seed 42 \
        --k 300 --lam_override $LAM_SWEEP $EXTRA_LOAD \
        --device $DEV --save_path $SAVE \
        > "$LOGDIR/$(basename $SAVE .json).log" 2>&1
done

echo "[$M lane $LANE] done $(date)"
