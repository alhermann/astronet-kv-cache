# AstroNet — drop-in hybrid KV-cache management for frozen LLMs

A small (~14M params) learned cross-window summary module + parameter-free token selector,
attached to a frozen quantised LLM. Reduces long-context KV cache to a fixed budget
(typically k=300 tokens + 16 learned summary tokens) while preserving QA accuracy.

## Quick start

```python
from astronet.wrapper import AstroNetWrapper

wrapper = AstroNetWrapper.from_pretrained(
    'Qwen/Qwen2.5-7B',                                       # local path or HF id
    astro_ckpt='checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt',
    attn_dim=512,                                            # Qwen 7B uses 512; others auto-detect
)

answer = wrapper.answer(
    windows=[doc_chunk_1, doc_chunk_2, ...],
    question='...',
    k=300,                                                   # total KV budget
    method='hybrid',                                         # or 'mult', 'snapkv', 'streaming'
)
```

See `examples/demo_50line.py` for a runnable example.

## Supported backbones

Released checkpoints (in `checkpoints/`):

| Model | attn_dim | Δ vs pure-S1 (SQuAD pos-robust, n=100×4 pos, 5-seed mean) |
|---|---|---|
| Qwen 2.5-7B | 512 | +8.5 pp |
| Qwen 2.5-14B | 256 | +10.1 pp ± 2.6 |
| Qwen 2.5-32B | 256 | +7.8 pp ± 2.9 |
| Llama 3.1-8B | 256 | +9.5 pp ± 4.0 |
| Mistral 7B v0.3 | 256 | +14.4 pp ± 3.2 |
| Mistral Small 24B | 256 | +3.9 pp ± 3.9 (marginal — use `method='mult'`) |

## Cache footprint

For an n=20-window context (≈7680 tokens) on Qwen 2.5-7B:

| Configuration | Cache size (KiB) | Compression vs full |
|---|---|---|
| Full FP16 cache | 430 080 | 1× |
| AstroNet hybrid k=300, FP16 | 16 800 | 25.6× |
| AstroNet hybrid k=300 + TurboQuant K8V4 | 6 300 | 68.3× |

## Methods

- `hybrid`: S1 (parameter-free cross-window selection, k_real=284) + S2 (16 learned summary tokens). Default.
- `mult`: S1 only, k=300. Use for Mistral 24B where S2 underperforms at long context.
- `snapkv`: Equivalent to `mult` in this implementation (same scoring with last-window mask).
- `streaming`: Sink + recent tokens. Baseline.

## Limitations

- Mistral 7B v0.3 (max_position=32k) has weak long-range attention; cross-window selection
  degrades sharply beyond ~10 windows.
- Mistral Small 24B (RoPE θ=1e8) exhibits OOD-context-length sensitivity; S2 module
  trained at n_windows=5 underperforms at n=20. Use `method='mult'` for this backbone.
