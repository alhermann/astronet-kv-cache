# Results index

Every quantitative claim in the manuscript is reproducible from a file in this directory. 143 result JSONs cover six backbones across four evaluation regimes.

## Master aggregates

* `paper_results_complete.json` — full aggregated table across all backbones and methods.
* `pareto_data.json` — Pareto-frontier inputs (byte budgets + accuracies per backbone) consumed by the figure-regeneration script.

## Per-experiment files

| Pattern                                       | Coverage                                                      |
|-----------------------------------------------|---------------------------------------------------------------|
| `baselines_<model>_k300.json`                 | StreamingLLM, H2O, SnapKV, multiplicative, RAG k=1 at k=300   |
| `hybrid_pos_robust_v2_<model>*.json`          | Stage 1 / Stage 1+2, five seeds × four answer positions       |
| `hybrid_pos_robust_w10diverse_<model>*.json`  | Same, longer-context retraining recipe (Mistral)              |
| `kivi_<model>.json`                           | KIVI K2V2 / K4V4 / K8V4                                       |
| `longbench_<model>_k300.json`                 | LongBench HotpotQA + MultiFieldQA F1                          |
| `longbench_<model>_baselines_v2.json`         | Post-retrain eviction baselines for Mistral                   |
| `needle_<model>_k300.json`                    | Needle-in-a-haystack at n=5/10/20 windows                     |
| `needle_realsnapkv_<model>_k300.json`         | Faithful SnapKV reimplementation                              |
| `ruler_<model>_k300.json`                     | RULER 8k / 16k / 32k generalisation                           |
| `diag_kconfound_<model>_n20.json`             | Matched-budget Stage 2 contribution diagnostic                |
| `diag_bio_essentiality_<model>.json`          | Aggregator-form ablation (EMA / α=1 / α=0 / mean)             |
| `turboquant_cross_<model>.json`               | K8V4 single-seed verification                                 |
| `latency_<model>_v5.json`                     | Wall-clock latency (per method, per backbone)                 |
| `multiturn_amortization_qwen7b_v5.json`       | Multi-turn cache reuse                                        |
| `full_fp16_<model>.json`                      | Uncompressed reference for the Pareto upper anchor            |
| `pyramidkv_<model>.json`                      | PyramidKV remark in the Discussion                            |

## Schema

Each file shares a common shell:

```json
{
  "model": "qwen2.5-7b",
  "n_samples": 200,
  "seed": 42,
  "k": 300,
  "results": { ... }
}
```

The `results` sub-object's shape varies by experiment; see the loading code in the script that produced each file (typically in `baselines/eval_*.py` or `training/eval_*.py`).
