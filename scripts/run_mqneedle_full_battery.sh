#!/usr/bin/env bash
# Full validation battery for AstroGate on multi-query NIAH (2026-06-14).
# Lane A = cuda:0 (llama8b), Lane B = cuda:1 (qwen7b).
#
# Sections:
#   (1) 5-seed CI at default config — statistical confidence.
#   (2) k-budget scan {64, 100, 500} — robustness to compression budget.
#   (3) Context-length scan {n_windows = 50, 200, 400} — scaling claim.
#   (4) n_needles scan {2, 8} — multi-query stress.
#
# Default config: n_windows=100 (~10K), n_needles=4, k=300, 20 trials.
# Per-model best λ: Qwen 0.5, Llama 100.
set -u
cd /home/alexander/Schreibtisch/AstroNet
PY=/home/alexander/Schreibtisch/AstroNet/venv/bin/python3
LOGDIR=logs/training/mqneedle
mkdir -p "$LOGDIR" logs/results

declare -A MP LAMG
MP[qwen7b]=./models/qwen2.5-7b
MP[llama8b]=./models/llama-3.1-8b
LAMG[qwen7b]=0.5
LAMG[llama8b]=100.0

LANE=${1:?usage: $0 A|B}
case "$LANE" in
    A) M=llama8b; DEV=cuda:0 ;;
    B) M=qwen7b; DEV=cuda:1 ;;
esac
LAM=${LAMG[$M]}
CKPT=checkpoints/astro_gate_${M}.pt

run_cell() {
    local meth=$1 NW=$2 NN=$3 K=$4 SEED=$5 NT=$6 tag=$7 extra=$8
    local save=logs/results/mqneedle_${meth}_${M}_w${NW}_n${NN}_k${K}_s${SEED}_t${NT}_${tag}.json
    [ -f "$save" ] && { echo "  SKIP $save"; return; }
    echo "  [$(date +%H:%M)] $M $meth w${NW} n${NN} k${K} s${SEED} t${NT} $tag"
    $PY -u baselines/eval_multiquery_needle.py \
        --model_path ${MP[$M]} --method $meth \
        --n_windows $NW --n_needles $NN --n_trials $NT --seed $SEED \
        --k $K $extra \
        --device $DEV --save_path $save \
        > "$LOGDIR/$(basename $save .json).log" 2>&1
}

echo "[battery lane $LANE] start $(date)"

# ===== (1) 5-seed CI at default config (4 needles, w=100, k=300, 20 trials) =====
echo "=== (1) 5-seed CI ==="
for SEED in 1 2 3 4 5; do
    run_cell snapkv     100 4 300 $SEED 20 ci  ""
    run_cell astrogate  100 4 300 $SEED 20 ci  "--checkpoint $CKPT --ckpt_tag ci_$SEED --lam_override $LAM"
done

# ===== (2) k-budget scan (seed=1, 20 trials) =====
echo "=== (2) k-budget scan ==="
for K in 64 100 500; do
    run_cell full          100 4 $K 1 20 kscan ""
    run_cell snapkv_oracle 100 4 $K 1 20 kscan ""
    run_cell snapkv        100 4 $K 1 20 kscan ""
    run_cell astrogate     100 4 $K 1 20 kscan "--checkpoint $CKPT --ckpt_tag kscan --lam_override $LAM"
done

# ===== (3) Context-length scan (k=300, 4 needles, 15 trials at longer ctx) =====
echo "=== (3) Context-length scan ==="
for NW in 50 200 400; do
    NT=15
    [ $NW -gt 200 ] && NT=10
    run_cell full          $NW 4 300 1 $NT cscan ""
    run_cell snapkv_oracle $NW 4 300 1 $NT cscan ""
    run_cell snapkv        $NW 4 300 1 $NT cscan ""
    run_cell astrogate     $NW 4 300 1 $NT cscan "--checkpoint $CKPT --ckpt_tag cscan --lam_override $LAM"
done

# ===== (4) n_needles scan (k=300, w=100, seed=1) =====
echo "=== (4) n_needles scan ==="
for NN in 2 8; do
    NT=20
    [ $NN -eq 8 ] && NT=10
    run_cell full          100 $NN 300 1 $NT nnscan ""
    run_cell snapkv_oracle 100 $NN 300 1 $NT nnscan ""
    run_cell snapkv        100 $NN 300 1 $NT nnscan ""
    run_cell astrogate     100 $NN 300 1 $NT nnscan "--checkpoint $CKPT --ckpt_tag nnscan --lam_override $LAM"
done

echo "[battery lane $LANE] done $(date)"
