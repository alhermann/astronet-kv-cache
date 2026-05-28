# AstroNet

Hybrid KV-cache management for frozen, 4-bit-quantised long-context language
models. AstroNet is a drop-in inference-time module that combines a
parameter-free cross-segment token selector (Stage 1) with a small (~14M
parameter) learned summary module (Stage 2) that re-injects virtual
key/value pairs through the backbone's own dequantised projection weights.
The backbone is never updated and never sees a fine-tuning gradient; only
the auxiliary summary module is trained.

The pipeline is designed for the deployment-bound regime: a single
accelerator, the standard 4-bit serving stack intact, no architectural
modification of the backbone, and a fixed per-layer cache budget
(default: 16 virtual + 284 real = 300 tokens).

## Headline results

Single-seed unless stated. Full numbers, confidence intervals, and per-model
breakdowns live in `paper_npjai/`.

- **Controlled multi-segment QA (SQuAD, position-robust, 5 seeds).** Adding
  the Stage-2 summary to Stage-1 selection raises mean accuracy by an average
  of **+8.5 percentage points** across six backbones, with per-model gains
  of +3.6 to +12.9 pp.
- **LongBench extractive QA** (HotpotQA, MultiFieldQA, 2WikiMultihopQA,
  MuSiQue). AstroNet is top-1 against StreamingLLM / H2O / SnapKV on
  **13 of 24 task-backbone cells** under the canonical Counter-based F1
  with SQuAD-style normalisation, more than twice the uniform-null rate.
- **Needle-in-a-haystack at 20 segments.** Stage-1+2 beats SnapKV on
  all six backbones; the largest margins (19-50 pp) are on the Qwen and
  Llama backbones.
- **KIVI K4V4 baseline.** At a matched real-token budget of k=300, AstroNet
  outperforms KIVI K4V4 on every evaluated backbone.
- **Cross-turn amortisation.** A 1.32x cumulative speedup across five
  conversational turns at zero accuracy loss.

Evaluated backbones: Qwen 2.5-7B, Qwen 2.5-14B, Qwen 2.5-32B, Llama 3.1-8B,
Mistral 7B v0.3, Mistral-Small 24B.

## Repository layout

```
astronet/      Core library: AstroNet wrapper, hook injection, Stage-1 selector,
               Stage-2 summary module, Lloyd-Max and TurboQuant adaptations.
training/      Stage-2 training and hybrid evaluation scripts (SQuAD,
               position-robust, quantised eval, RMT-LoRA reference impl.).
baselines/     Faithful reimplementations of H2O, SnapKV, StreamingLLM,
               PyramidKV, KIVI, full-FP16 reference, plus LongBench, Needle,
               RULER, and latency drivers.
data/          Dataset builders (SQuAD, HotpotQA, synthetic multi-segment QA,
               natural QA, real_qa.py). No raw data is checked in.
scripts/       Shell wrappers that queue multi-model evaluations.
configs/       YAML configuration files for training runs.
docs/          Project-internal documentation (LaTeX source for the log).
examples/      Minimal demonstration scripts.
evaluation/    Evaluation helpers shared across pipelines.
paper_npjai/   LaTeX source for the npj AI manuscript (figures included).
```

## Installation

```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
# Install PyTorch matching your CUDA toolchain, e.g.:
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Tested with Python 3.8.10, PyTorch 2.4.1 + CUDA 11.8, transformers 4.46.3,
bitsandbytes 0.45.5. AstroNet expects 4-bit quantised backbones loaded
through `bitsandbytes` (nf4); other quantisation backends will work as long
as the backbone exposes the standard `model.model.layers[i]` accessor and
quantised `W_K`, `W_V` weights can be dequantised on demand.

## Downloading backbones

Backbones are not redistributed in this repository. Place them under
`./models/` (or symlink that directory at any storage location of your
choice). For example, for Qwen 2.5-7B:

```
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-7B --local-dir ./models/qwen2.5-7b
```

The training and evaluation scripts default to `./models/{model-name}` and
`./checkpoints/` paths; pass `--model_path` and `--checkpoint` to override.

## Quick start

The minimal demo loads Qwen 2.5-7B, runs Stage-1 selection plus the learned
Stage-2 summary on a single SQuAD-style multi-segment question, and prints
the predicted span:

```
python examples/demo_50line.py \
    --model_path ./models/qwen2.5-7b \
    --checkpoint ./checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt
```

To run a single-sample evaluation with the full hybrid pipeline:

```
python training/eval_hybrid_n200.py \
    --model_path ./models/qwen2.5-7b \
    --checkpoint ./checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt \
    --n_eval 1 --seed 42 --k_real 284 --n_mem 16 --attn_dim 512
```

All shell scripts respect the `PY` environment variable: set
`PY=/path/to/python3` if your interpreter is not on `$PATH`.

## Reproducing the paper

The canonical protocols for every experiment are described in
`CANONICAL_PROTOCOL.md`. The main entry points are:

| Experiment | Driver | Notes |
| --- | --- | --- |
| Main baselines table | `baselines/eval_all_baselines.py` | n=200, seed=42, k=300 |
| RAG k=3 | `training/eval_multiplicative.py --rag_k 3` | |
| Hybrid eval (paper Table 1) | `training/train_hybrid.py` evaluate() | n_eval=100 |
| LongBench | `baselines/eval_longbench.py` | 100 samples per task |
| Needle-in-a-haystack | `baselines/eval_needle.py` | 5 depths x 20 trials |
| RULER | `baselines/run_ruler.py` | |
| TurboQuant K8V4 | `training/eval_quantized.py` | |
| Position robust SQuAD CI | `training/eval_hybrid_position_robust.py` | seeds 42/7/123/999/2024 |
| Cross-turn amortisation | `baselines/eval_multiturn_amortization.py` | |

Per-model layer choices (sense layer, inject layers, attn_dim) are listed
in `CANONICAL_PROTOCOL.md`. Shell queues in `scripts/` and `baselines/`
chain the per-model runs for the full table.

## Citation

```
@article{astronet2026,
  title  = {AstroNet: hybrid KV-cache management for frozen long-context language models},
  author = {},
  year   = {2026},
  note   = {preprint}
}
```

## License

MIT (see `LICENSE`).
