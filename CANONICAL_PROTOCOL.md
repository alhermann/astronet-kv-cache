# Canonical Experimental Protocol — ALL results must follow this

## Main Baselines (Table 1)
- **Script**: `baselines/eval_all_baselines.py`
- **n_samples**: 200
- **seed**: 42
- **k**: 300
- **Methods**: streaming_llm, h2o, snapkv, multiplicative, rag_k1
- **Output**: `logs/results/baselines_{model}_k300.json`
- **Status**: DONE for all 6 main models (7B-24B). 72B/70B use eval_large_model.py (streaming_llm, multiplicative, rag_k1, rag_k3 only — no h2o/snapkv).

## RAG k=3
- **Script**: `training/eval_multiplicative.py --rag_k 3 --skip_heuristic --skip_full_kv`
- **n_samples**: 200, **seed**: 42, **k**: 300
- **Output**: `logs/results/definitive_{model}_ragk3_n200.json` (NEW file, never overwrite definitive_*.json)
- **72B/70B**: included in large_model_*.json already

## Hybrid Eval
- **Script**: `training/train_hybrid.py` evaluate() function
- **n_eval**: 100 (STANDARDIZED — all models use 100)
- **seed**: 42
- **Source of truth**: training logs + `paper_results_complete.json`
- **72B/70B**: INFEASIBLE (documented)

## LongBench
- **Script**: `baselines/eval_longbench.py`
- **Tasks**: hotpotqa, multifieldqa_en (2 tasks — the common set across all models)
- **max_samples**: 100 per task
- **k**: 300, chunk_size: 384
- **Methods**: streaming_llm, h2o, snapkv, multiplicative, hybrid
- **--multi_gpu** for 14B+ models
- **--hybrid_checkpoint**: model-specific, **--sense_layer**: model-specific
- **Output**: `logs/results/longbench_{model}_k300.json`
- **72B/70B**: INFEASIBLE

## Needle-in-Haystack
- **Script**: `baselines/eval_needle.py`
- **n_windows_list**: 5 10 20 (3 sizes — matches existing 7B/8B/Mistral7B results)
- **n_trials**: 20
- **k**: 300
- **Methods**: streaming_llm, h2o, multiplicative (3 methods — matches existing)
- **--multi_gpu** for 14B+
- **Output**: `logs/results/needle_{model}_k300.json`
- **72B/70B**: INFEASIBLE

## TurboQuant K8V4
- **Script**: `training/eval_quantized.py` (7B-24B) or `baselines/eval_large_model.py` (72B/70B)
- **n_samples**: 200, **seed**: 42
- **Output**: `logs/results/turboquant_cross_{model}.json`

## Confidence Intervals
- **Script**: `baselines/eval_all_baselines.py` with different --seed (SAME script as main baselines!)
- **Seeds**: 42, 123, 999, 7, 2024
- **n_samples**: 200, **k**: 300
- **Methods**: streaming_llm, h2o, snapkv, multiplicative, rag_k1
- **Output**: `logs/results/ci_{model}_5seeds_baselines.json`
- **72B/70B**: use eval_large_model.py with --seed (streaming_llm, multiplicative, rag_k1, rag_k3)
- **NOTE**: This replaces the old CI protocol (which used train_hybrid evaluate). Using the SAME script as baselines ensures numbers are consistent.

## Vanilla TurboQuant (negative baseline)
- Same as TurboQuant but WITH random rotation — should give 0%
- Models: Qwen 7B (done), Llama 8B, Mistral 7B

## Per-head vs Global validation
- **Script**: `baselines/eval_perhead_baselines.py` (or similar)
- **n_samples**: 200, **k**: 300
- Models: Qwen 7B (done), Llama 8B, Mistral 7B

## Sense layers and inject layers per model
| Model | Layers | Inject | Sense | Hybrid ckpt |
|---|---|---|---|---|
| Qwen 7B | 28 | 7,14,21,26 | 14 | astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt |
| Qwen 14B | 48 | 12,24,36,46 | 24 | astro_hybrid_qwen2_5-14b.pt |
| Qwen 32B | 64 | 16,32,48,62 | 32 | astro_hybrid_qwen2_5-32b.pt |
| Llama 8B | 32 | 8,16,24,30 | 16 | astro_hybrid_llama-3_1-8b.pt |
| Mistral 7B | 32 | 8,16,24,30 | 16 | astro_hybrid_mistral-7b-v0_3_n16_k284_t5000_s42.pt |
| Mistral 24B | 40 | 10,20,30,38 | 20 | astro_hybrid_mistral-small-24b_n16_k284_t5000_s42.pt |
| Qwen 72B | 80 | 20,40,60,78 | 40 | NONE (infeasible) |
| Llama 70B | 80 | 20,40,60,78 | 40 | NONE (infeasible) |
