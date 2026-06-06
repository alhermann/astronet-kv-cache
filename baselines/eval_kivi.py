"""KIVI 2-bit baseline for KV cache quantization.

KIVI (Liu et al. ICML 2024) uses asymmetric quantization:
- Keys: per-channel (per-head_dim) quantization, group size 32
- Values: per-token (per-position) quantization, group size 32

Applied here on top of multiplicative cross-window selection at k=300, so the
comparison against our adapted Lloyd-Max K8V4 is at matching cache budget.

Per-channel key quantization preserves the rotary-encoding structure per dim,
which is the same structural insight that motivates our per-head normalization
fix. KIVI uses uniform mid-rise quantization within each group.
"""

import sys, os, argparse, json, time
sys.path.insert(0, '.')
sys.path.insert(0, 'baselines')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from data.real_qa import generate_squad_dataset
from eval_all_baselines import (
    process_windows, build_cache_from_indices, generate_with_cache,
    select_multiplicative,
)


def kivi_quantize_per_channel(t, bits=2, group_size=32):
    """KIVI key quantization: per-channel (last dim), grouped along seq.
    t: (heads, seq, head_dim)
    Returns dequantized t with same shape.
    """
    H, S, D = t.shape
    # Pad seq to multiple of group_size
    pad = (group_size - S % group_size) % group_size
    if pad:
        t = torch.cat([t, torch.zeros(H, pad, D, dtype=t.dtype, device=t.device)], dim=1)
    Sp = S + pad
    n_groups = Sp // group_size
    # Reshape to (H, n_groups, group_size, D), per-channel = per-D-axis-min/max within group
    t_g = t.view(H, n_groups, group_size, D)
    # min/max per (H, n_groups, D) -> shape (H, n_groups, 1, D)
    mn = t_g.min(dim=2, keepdim=True).values
    mx = t_g.max(dim=2, keepdim=True).values
    levels = (1 << bits) - 1
    scale = (mx - mn).clamp(min=1e-8) / levels
    q = ((t_g - mn) / scale).round().clamp(0, levels)
    deq = q * scale + mn
    deq = deq.view(H, Sp, D)[:, :S, :]
    return deq


def kivi_quantize_per_token(t, bits=2, group_size=32):
    """KIVI value quantization: per-token (per-seq-position), grouped along D.
    t: (heads, seq, head_dim)
    """
    H, S, D = t.shape
    pad = (group_size - D % group_size) % group_size
    if pad:
        t = torch.cat([t, torch.zeros(H, S, pad, dtype=t.dtype, device=t.device)], dim=2)
    Dp = D + pad
    n_groups = Dp // group_size
    t_g = t.view(H, S, n_groups, group_size)
    mn = t_g.min(dim=3, keepdim=True).values
    mx = t_g.max(dim=3, keepdim=True).values
    levels = (1 << bits) - 1
    scale = (mx - mn).clamp(min=1e-8) / levels
    q = ((t_g - mn) / scale).round().clamp(0, levels)
    deq = q * scale + mn
    deq = deq.view(H, S, Dp)[:, :, :D]
    return deq


def apply_kivi_to_cache(cache, kbits, vbits, group_size=32, residual_length=128):
    """Apply KIVI quantization to a built cache, faithfully matching the
    published KIVI tuning-free recipe.

    Critical (2026-06-06 critic audit): the official KIVI keeps the last
    ``residual_length`` tokens of the prompt in FP16 alongside the quantised
    cache.  This is the "tuning-free" structural component --- recent context
    is the part the model attends to most heavily during decode, and
    quantising it hurts disproportionately.  Without this buffer, our KIVI
    numbers UNDER-estimate official KIVI (i.e., our reported gap between
    AstroNet and KIVI is overstated in our favour).

    Default ``residual_length=128`` matches the canonical KIVI setting
    (Liu et al., ICML 2024, Section 4.2).

    Args:
        cache: a HuggingFace DynamicCache to quantise
        kbits: bits per key scalar (KIVI K4V4 -> kbits=4)
        vbits: bits per value scalar
        group_size: KIVI default 32
        residual_length: number of most-recent tokens to keep in FP16.
            Pass ``0`` to reproduce the prior (under-estimating) behaviour.
    """
    new_cache = DynamicCache()
    for li in range(len(cache.key_cache)):
        K = cache.key_cache[li]    # (1, kv_heads, seq, head_dim)
        V = cache.value_cache[li]
        # Squeeze batch dim for the quantiser fns; (heads, seq, head_dim).
        K2 = K.squeeze(0)
        V2 = V.squeeze(0)
        S = K2.shape[1]
        # Split into "to be quantised" head + "kept in FP16" tail.
        # residual_length tokens at the END of the sequence stay FP16; the
        # rest are KIVI-quantised.  If seq <= residual_length, nothing is
        # quantised --- this matches KIVI's behaviour on short prompts.
        r = min(residual_length, S)
        if r > 0:
            K_head, K_tail = K2[:, :S - r, :], K2[:, S - r:, :]
            V_head, V_tail = V2[:, :S - r, :], V2[:, S - r:, :]
        else:
            K_head, K_tail = K2, K2[:, :0, :]
            V_head, V_tail = V2, V2[:, :0, :]
        # Quantise the (older) head, keep tail in FP16.
        if K_head.shape[1] > 0:
            K_head = kivi_quantize_per_channel(K_head, bits=kbits,
                                                 group_size=group_size)
            V_head = kivi_quantize_per_token(V_head, bits=vbits,
                                               group_size=group_size)
        K_combined = torch.cat([K_head, K_tail], dim=1)
        V_combined = torch.cat([V_head, V_tail], dim=1)
        new_cache.update(K_combined.unsqueeze(0), V_combined.unsqueeze(0), li)
    return new_cache


def run_eval(model, tokenizer, samples, k, kbits, vbits, device,
              residual_length=128):
    nl = model.config.num_hidden_layers
    correct = 0
    for si, s in enumerate(samples):
        fact_windows = s.windows[:-1]
        all_kv, all_attn, boundaries, total = process_windows(model, tokenizer, fact_windows, device)
        idx = select_multiplicative(all_attn, model, all_kv, s.question, tokenizer, k, device)
        cache = build_cache_from_indices(all_kv, idx, nl)
        cache = apply_kivi_to_cache(cache, kbits=kbits, vbits=vbits,
                                      residual_length=residual_length)
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        text = generate_with_cache(model, tokenizer, fq['input_ids'], cache, len(idx), device)
        if s.answer.lower() in text.lower():
            correct += 1
        if (si + 1) % 25 == 0:
            print(f'  KIVI {kbits}/{vbits}-bit {si+1}/{len(samples)}: {correct}/{si+1} = {correct/(si+1):.1%}',
                  flush=True)
    return correct / len(samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='models/qwen2.5-7b')
    p.add_argument('--n_samples', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path',
                   default='logs/results/kivi_qwen7b.json')
    p.add_argument('--configs', nargs='+', default=['k2v2', 'k4v2', 'k4v4', 'k8v4'],
                   help='Which (kbits, vbits) configurations to test')
    p.add_argument('--residual_length', type=int, default=128,
                   help='Number of most-recent tokens kept in FP16 alongside '
                        'the quantised cache.  KIVI default 128.  Pass 0 to '
                        'reproduce the prior (under-estimating) behaviour.')
    args = p.parse_args()

    print(f'Loading {args.model_path}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = model.get_input_embeddings().weight.device

    samples = generate_squad_dataset(n_samples=args.n_samples, n_windows=5,
                                      vary_distance=True, seed=args.seed,
                                      split='validation')

    config_map = {'k2v2': (2, 2), 'k4v2': (4, 2), 'k4v4': (4, 4), 'k8v4': (8, 4),
                  'k2v4': (2, 4)}
    results = {}
    for cfg in args.configs:
        kbits, vbits = config_map[cfg]
        print(f'\n=== KIVI {cfg} (keys={kbits}-bit per-channel, values={vbits}-bit per-token) ===',
              flush=True)
        t0 = time.time()
        acc = run_eval(model, tokenizer, samples, args.k, kbits, vbits, device,
                        residual_length=args.residual_length)
        elapsed = time.time() - t0
        results[cfg] = {'kbits': kbits, 'vbits': vbits, 'accuracy': acc,
                        'elapsed_s': elapsed,
                        'residual_length': args.residual_length}
        print(f'  KIVI {cfg} (r={args.residual_length}): {acc:.3f} ({elapsed:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'method': (f'multiplicative_k{args.k} + KIVI K?V? '
                        f'(per-channel keys, per-token values, group=32, '
                        f'r={args.residual_length} FP16 residual buffer)'),
            'n_samples': args.n_samples, 'seed': args.seed, 'k': args.k,
            'residual_length': args.residual_length,
            'results': results,
        }, f, indent=2)
    print(f'\nSaved to {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
