"""Generate the efficiency Pareto plot: KV-cache bytes vs accuracy across methods.

Bytes are computed analytically from cache structure:
- FP16 cache: k_total * n_layers * n_kv_heads * head_dim * 2 (K,V) * 2 bytes
- K8V4 (TurboQuant): k_total * n_layers * n_kv_heads * head_dim * (1 + 0.5) bytes  [K=8-bit, V=4-bit]
- K4V4 (KIVI):       k_total * n_layers * n_kv_heads * head_dim * (0.5 + 0.5) bytes

Pulls accuracy from existing JSONs in logs/results/.
Generates a CSV + a simple text-rendered ASCII Pareto for paper-prep.

Run: python baselines/generate_pareto.py
"""
import sys; sys.path.insert(0, '.')
import json
from pathlib import Path

# Model architecture specs (used for KV byte computation)
ARCH = {
    'qwen2.5-7b':       {'n_layers': 28, 'n_kv_heads': 4, 'head_dim': 128, 'native_max': 131072},
    'qwen2.5-14b':      {'n_layers': 48, 'n_kv_heads': 8, 'head_dim': 128, 'native_max': 131072},
    'qwen2.5-32b':      {'n_layers': 64, 'n_kv_heads': 8, 'head_dim': 128, 'native_max': 131072},
    'llama-3.1-8b':     {'n_layers': 32, 'n_kv_heads': 8, 'head_dim': 128, 'native_max': 131072},
    'mistral-7b-v0.3':  {'n_layers': 32, 'n_kv_heads': 8, 'head_dim': 128, 'native_max': 32768},
    'mistral-small-24b':{'n_layers': 40, 'n_kv_heads': 8, 'head_dim': 128, 'native_max': 32768},
}


def kv_bytes(model_name, k_tokens, dtype='fp16'):
    """Bytes for K+V cache across all layers."""
    a = ARCH[model_name]
    per_token_per_layer_bytes = {
        'fp16': 2 * a['n_kv_heads'] * a['head_dim'] * 2,        # K + V at fp16
        'k8v4': (1 + 0.5) * a['n_kv_heads'] * a['head_dim'],     # K=8-bit, V=4-bit
        'k4v4': (0.5 + 0.5) * a['n_kv_heads'] * a['head_dim'],   # K=V=4-bit
    }[dtype]
    return k_tokens * a['n_layers'] * per_token_per_layer_bytes


def needle_avg_at_n(jpath, n_key='n20'):
    if not Path(jpath).exists():
        return None
    d = json.load(open(jpath))['results'].get(n_key, {})
    if not d:
        return None
    out = {}
    for method, depths in d.items():
        avg = sum(depths.values()) / len(depths) if depths else None
        out[method] = avg
    return out


def turboquant_avg(jpath):
    if not Path(jpath).exists():
        return None
    d = json.load(open(jpath))
    return d.get('results', {})


def posrobust_seeds(model_tag):
    """Mean over seeds for hybrid_pos_robust_v2_<tag>_s*.json + default."""
    seeds = []
    for suff in ['', '_s7', '_s123', '_s999', '_s2024']:
        p = Path(f'logs/results/hybrid_pos_robust_v2_{model_tag}{suff}.json')
        if p.exists():
            d = json.load(open(p))['results']
            pure = sum(d[k]['pure300'] for k in d if k.startswith('pos_')) / 4
            hyb = sum(d[k]['hybrid'] for k in d if k.startswith('pos_')) / 4
            seeds.append((pure, hyb))
    if not seeds:
        return None, None
    n = len(seeds)
    mp = sum(s[0] for s in seeds) / n
    mh = sum(s[1] for s in seeds) / n
    return mp, mh


def main():
    # Map tag -> full model name
    tag2name = {
        'qwen7b': 'qwen2.5-7b', 'qwen14b': 'qwen2.5-14b', 'qwen32b': 'qwen2.5-32b',
        'llama8b': 'llama-3.1-8b', 'mistral7b': 'mistral-7b-v0.3', 'mistral24b': 'mistral-small-24b',
    }
    K = 300  # selection-method budget

    rows = []
    for tag, mn in tag2name.items():
        pure, hyb = posrobust_seeds(tag)
        ndl = needle_avg_at_n(f'logs/results/needle_{mn}_k300.json', 'n20')
        # TurboQuant SQuAD numbers
        tq = turboquant_avg(f'logs/results/turboquant_cross_{mn}.json')

        # FP16 K=300 cache
        b300_fp16 = kv_bytes(mn, K, 'fp16')
        # FP16 full-cache (use n=20 windows ≈ 7680 tokens for Needle reference)
        n20_tokens = 20 * 384  # chunk_size=384 in eval_needle.py
        b_full_fp16 = kv_bytes(mn, n20_tokens, 'fp16')
        # K8V4 TurboQuant K=300 cache
        b300_k8v4 = kv_bytes(mn, K, 'k8v4')
        # KIVI K4V4 K=300 cache
        b300_k4v4 = kv_bytes(mn, K, 'k4v4')

        row = {
            'tag': tag, 'model': mn,
            'pos_robust_pure': pure, 'pos_robust_hybrid': hyb,
            'needle_n20': ndl,
            'tq_squad': tq,
            'bytes_fp16_k300': b300_fp16,
            'bytes_fp16_full_n20': b_full_fp16,
            'bytes_k8v4_k300': b300_k8v4,
            'bytes_k4v4_k300': b300_k4v4,
            'compression_vs_full': b_full_fp16 / b300_fp16,
            'compression_vs_full_k8v4': b_full_fp16 / b300_k8v4,
        }
        rows.append(row)

    # Print summary
    print('\n=== EFFICIENCY PARETO (per model) ===')
    print(f'{"model":<20} {"K=300 FP16 KB":<14} {"Full FP16 KB":<14} {"Compression":<12} {"+K8V4 comp":<12}')
    for r in rows:
        print(f'{r["model"]:<20} {r["bytes_fp16_k300"]/1024:>10.1f}   {r["bytes_fp16_full_n20"]/1024:>10.1f}  {r["compression_vs_full"]:>8.1f}x  {r["compression_vs_full_k8v4"]:>8.1f}x')

    print('\n=== ACCURACY ANCHORS (Needle n=20, current numbers) ===')
    print(f'{"model":<20} {"streaming":<10} {"h2o":<6} {"mult":<6} {"hybrid":<7}')
    for r in rows:
        ndl = r['needle_n20'] or {}
        print(f'{r["model"]:<20} {ndl.get("streaming_llm",0)*100:>6.1f}    {ndl.get("h2o",0)*100:>3.1f}   {ndl.get("multiplicative",0)*100:>3.1f}   {ndl.get("hybrid",0)*100:>3.1f}')

    out = {'rows': rows}
    Path('logs/results/pareto_data.json').write_text(json.dumps(out, indent=2))
    print('\nSaved logs/results/pareto_data.json')


if __name__ == '__main__':
    main()
