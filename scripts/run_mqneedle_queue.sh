#!/usr/bin/env bash
# Multi-query NIAH compress-once eval (2026-06-14).
# All methods × both models × medium ctx (~10K tokens, 4 needles per ctx).
#   Lane A = cuda:0 (llama8b), Lane B = cuda:1 (qwen7b)
set -u
cd /home/alexander/Schreibtisch/AstroNet
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
LOGDIR=logs/training/mqneedle
mkdir -p "$LOGDIR" logs/results

declare -A MP
MP[qwen7b]=./models/qwen2.5-7b
MP[llama8b]=./models/llama-3.1-8b

NW=100   # ~10K tokens
NN=4
NT=20

LANE=${1:?usage: $0 A|B}
case "$LANE" in
    A) M=llama8b; DEV=cuda:0 ;;
    B) M=qwen7b; DEV=cuda:1 ;;
    *) echo bad lane; exit 1 ;;
esac

run_cell() {
    local meth=$1 tag=$2 extra=$3
    local save=logs/results/mqneedle_${meth}_${M}_w${NW}_n${NN}_${tag}.json
    [ -f "$save" ] && { echo "  SKIP $save"; return; }
    echo "  [$(date +%H:%M)] $M $meth $tag"
    $PY -u baselines/eval_multiquery_needle.py \
        --model_path ${MP[$M]} --method $meth \
        --n_windows $NW --n_needles $NN --n_trials $NT --seed 42 \
        --k 300 $extra \
        --device $DEV --save_path $save \
        > "$LOGDIR/${meth}_${M}_${tag}.log" 2>&1
}

echo "[mqneedle lane $LANE] start $(date)"
run_cell full           anchor  ""
run_cell snapkv_oracle  anchor  ""
run_cell snapkv         anchor  ""
run_cell astrogate      v1_l0p5 "--checkpoint checkpoints/astro_gate_${M}.pt --ckpt_tag v1 --lam_override 0.5"
# Per-model best λ from earlier sweep (Qwen prefers 0.5, Llama 100)
if [ "$M" = "llama8b" ]; then
    LAMG=100.0
else
    LAMG=0.5
fi
# Re-run astrogate at per-model best λ:
run_cell astrogate      v1_best "--checkpoint checkpoints/astro_gate_${M}.pt --ckpt_tag v1best --lam_override $LAMG"
run_cell astrogain_e2e  e2e     "--checkpoint checkpoints/astro_gain_e2e_${M}.pt --ckpt_tag e2e"
echo "[mqneedle lane $LANE] done $(date)"
