"""Compare four variants of the multiplicative selection rule across fact positions.

Variants:
  raw            : original h = sum over heads/layers/queries (the buggy version)
  zscore_window  : per-window z-score on h (Option A)
  causal_correct : divide h by (S - i) per window position (Option B)
  late_query     : use only last 32 queries' attention as h (Option C)

For each variant, evaluate at fact position 0..3 on Qwen 7B SQuAD (100 samples).
"""

import sys, os, argparse, json, time, math
sys.path.insert(0, '.')
sys.path.insert(0, 'baselines')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from data.real_qa import generate_squad_dataset, CrossContextSample
from eval_all_baselines import process_windows, build_cache_from_indices, generate_with_cache


def shuffle_fact_position(samples, position):
    out = []
    for s in samples:
        windows = list(s.windows)
        query_idx = s.query_window
        fact_idx = s.fact_window
        candidates = [i for i in range(len(windows)) if i != query_idx]
        if position >= len(candidates):
            position = len(candidates) - 1
        target = candidates[position]
        if target == fact_idx:
            out.append(s); continue
        new_windows = list(windows)
        new_windows[target], new_windows[fact_idx] = new_windows[fact_idx], new_windows[target]
        out.append(CrossContextSample(
            windows=new_windows, fact=s.fact, question=s.question, answer=s.answer,
            fact_window=target, query_window=query_idx,
            distance=s.distance, template_idx=s.template_idx,
        ))
    return out


def process_with_late_query(model, tokenizer, windows, device, late_q=32):
    """Like process_windows, but compute h using only the LAST `late_q` queries
    of each window for the cumulative-attention sum."""
    nl = model.config.num_hidden_layers
    all_kv = {li: ([], []) for li in range(nl)}
    h_per_window = []
    offset = 0
    boundaries = []
    for w in windows:
        ids = tokenizer(w, return_tensors='pt', max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True, output_attentions=True)
        sl = ids['input_ids'].shape[1]
        boundaries.append((offset, offset + sl))
        offset += sl
        imp = torch.zeros(sl, device=device)
        for la in out.attentions:
            a = la[0].to(device)
            if a.isnan().any(): continue
            # a: (heads, S, S). Take last `late_q` queries' attention rows.
            qs = min(late_q, sl)
            imp += a[:, -qs:, :].sum(dim=(0, 1))
        h_per_window.append(imp)
        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])
    return all_kv, h_per_window, boundaries, offset


def normalize_h(h_per_window, variant):
    """Apply chosen normalization to per-window h list, return concatenated tensor."""
    if variant == 'raw':
        return torch.cat(h_per_window)
    if variant == 'zscore_window':
        out = []
        for w in h_per_window:
            mu = w.mean()
            sd = w.std().clamp(min=1e-6)
            out.append((w - mu) / sd)
        return torch.cat(out)
    if variant == 'causal_correct':
        out = []
        for w in h_per_window:
            S = w.shape[0]
            denom = torch.arange(S, 0, -1, dtype=w.dtype, device=w.device)
            out.append(w / denom)
        return torch.cat(out)
    if variant == 'late_query':
        # Already-restricted h from process_with_late_query
        return torch.cat(h_per_window)
    raise ValueError(f'unknown variant {variant}')


def select_cross_only_softmax(model, all_kv, question, tokenizer, k, device, n_sink=4):
    """SnapKV-style cross-only score: post-softmax attention from question over cache.
    No `h` term; this isolates whether softmax cross-score alone is position-robust.
    """
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    total = torch.cat(all_kv[inject[0]][0], dim=2).shape[2]

    q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                      max_length=128, truncation=True).to(device)
    with torch.no_grad():
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)

    cross = torch.zeros(total, device=device)
    for li in inject:
        layer_device = all_kv[li][0][0].device
        Q = model.model.layers[li].self_attn.q_proj(
            q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
        K = torch.cat(all_kv[li][0], dim=2)[0]
        for hi in range(nkv):
            sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                              K[hi].float().T) / math.sqrt(hd)
            attn_w = torch.softmax(sc, dim=-1)
            cross += attn_w.sum(dim=(0, 1)).to(device)
    cross[:n_sink] = -1e9
    if total <= k:
        return torch.arange(total, device=device)
    _, idx = cross.topk(k)
    return idx.sort().values


def select_mult_softmax_cross(h_concat, model, all_kv, question, tokenizer, k, device, n_sink=4):
    """Multiplicative with SOFTMAX-normalized cross-score.
    Uses h_concat as the heuristic and softmax(Q@K) as the cross-score.
    This addresses the per-window K-norm bias by normalizing each query's attention
    distribution over cache positions.
    """
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    total = h_concat.shape[0]

    q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                      max_length=128, truncation=True).to(device)
    with torch.no_grad():
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
    cross = torch.zeros(total, device=device)
    for li in inject:
        layer_device = all_kv[li][0][0].device
        Q = model.model.layers[li].self_attn.q_proj(
            q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
        K = torch.cat(all_kv[li][0], dim=2)[0]
        for hi in range(nkv):
            sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                              K[hi].float().T) / math.sqrt(hd)
            attn_w = torch.softmax(sc, dim=-1)
            cross += attn_w.sum(dim=(0, 1)).to(device)
    cross[:n_sink] = -1e9

    mult = torch.clamp(h_concat, min=0) * torch.clamp(cross, min=0)
    mult[:n_sink] = -1e9
    if total <= k:
        return torch.arange(total, device=device)
    _, idx = mult.topk(k)
    return idx.sort().values


def select_mult_variant(h_concat, model, all_kv, question, tokenizer, k, device, n_sink=4):
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    total = h_concat.shape[0]

    q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                      max_length=128, truncation=True).to(device)
    with torch.no_grad():
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
    cross = torch.zeros(total, device=device)
    for li in inject:
        layer_device = all_kv[li][0][0].device
        Q = model.model.layers[li].self_attn.q_proj(
            q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
        K = torch.cat(all_kv[li][0], dim=2)[0]
        for hi in range(nkv):
            sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                              K[hi].float().T) / math.sqrt(hd)
            cross += sc.sum(dim=(0, 1)).to(device)
    cross[:n_sink] = -1e9

    mult = torch.clamp(h_concat, min=0) * torch.clamp(cross, min=0)
    mult[:n_sink] = -1e9
    if total <= k:
        return torch.arange(total, device=device)
    _, idx = mult.topk(k)
    return idx.sort().values


def eval_variant(model, tokenizer, samples, k, variant, device):
    nl = model.config.num_hidden_layers
    correct = 0
    for si, s in enumerate(samples):
        fact_windows = s.windows[:-1]
        if variant == 'late_query':
            all_kv, h_per_window, boundaries, total = process_with_late_query(
                model, tokenizer, fact_windows, device)
        else:
            all_kv, h_per_window, boundaries, total = process_windows(
                model, tokenizer, fact_windows, device)

        if variant == 'cross_only_softmax':
            idx = select_cross_only_softmax(model, all_kv, s.question, tokenizer, k, device)
        elif variant == 'mult_softmax_cross':
            h_concat = normalize_h(h_per_window, 'zscore_window')
            idx = select_mult_softmax_cross(h_concat, model, all_kv, s.question, tokenizer, k, device)
        else:
            h_concat = normalize_h(h_per_window, variant)
            idx = select_mult_variant(h_concat, model, all_kv, s.question, tokenizer, k, device)
        cache = build_cache_from_indices(all_kv, idx, nl)
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        text = generate_with_cache(model, tokenizer, fq['input_ids'], cache, len(idx), device)
        if s.answer.lower() in text.lower():
            correct += 1
        if (si + 1) % 25 == 0:
            print(f'    {variant} {si+1}/{len(samples)}: {correct}/{si+1} = {correct/(si+1):.1%}',
                  flush=True)
    return correct / len(samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='models/qwen2.5-7b')
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--variants', nargs='+',
                   default=['raw', 'zscore_window', 'causal_correct', 'late_query'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path',
                   default='logs/results/multiplicative_fix.json')
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

    base = generate_squad_dataset(n_samples=args.n_samples, n_windows=5,
                                   vary_distance=True, seed=args.seed,
                                   split='validation')

    results = {}
    for pos in args.positions:
        samples = shuffle_fact_position(base, pos)
        results[f'pos_{pos}'] = {}
        for v in args.variants:
            print(f'\n=== fact_pos={pos}, variant={v} ===', flush=True)
            t0 = time.time()
            acc = eval_variant(model, tokenizer, samples, args.k, v, device)
            elapsed = time.time() - t0
            results[f'pos_{pos}'][v] = acc
            print(f'  pos={pos} variant={v}: {acc:.3f} ({elapsed:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'n_samples': args.n_samples, 'seed': args.seed, 'k': args.k,
            'description': 'Compare multiplicative-selection variants across fact positions',
            'variants': {
                'raw': 'h_i = sum over heads/layers/queries (original buggy version)',
                'zscore_window': 'per-window z-score on h before multiplying with cross',
                'causal_correct': 'h_i divided by (S - i) per window to remove causal-mask bias',
                'late_query': 'h_i = sum over last 32 queries only (SnapKV-style observation window)',
            },
            'results': results,
        }, f, indent=2)
    print(f'\nSaved to {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
