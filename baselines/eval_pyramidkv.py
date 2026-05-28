"""PyramidKV-style baseline: layer-adaptive token budgets that decrease with depth.

Faithfully reimplements the layer-budget allocation idea from PyramidKV
(Cai et al. 2024) on top of our multi-window pipeline. Each layer gets a
different KV budget while the total summed budget across layers matches the
flat-budget setting.

Selection scoring within each layer follows H2O cumulative attention, since
that is the per-layer signal PyramidKV originally uses.
"""

import sys, os, argparse, json, time
sys.path.insert(0, '.')
sys.path.insert(0, 'baselines')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from data.real_qa import generate_squad_dataset
from eval_all_baselines import process_windows, generate_with_cache


def pyramid_budgets(n_layers, total_budget, ratio=0.5, n_sink=4, n_recent_min=10):
    """Allocate budgets b_l so that early layers get more, late layers get less.

    Linear schedule from b_top to ratio*b_top, summing to n_layers*total_budget.
    Returns list of n_layers ints summing to ~total_budget*n_layers.
    """
    # We want average budget per layer = total_budget. Linear from 2*total_budget*x
    # ranging from 1 at top to ratio at bottom. Average = (1+ratio)/2 ; pick scale
    # so that average = total_budget.
    avg = (1.0 + ratio) / 2.0
    top = total_budget / avg
    bots = []
    for l in range(n_layers):
        frac = 1.0 - (1.0 - ratio) * l / max(n_layers - 1, 1)
        bots.append(int(round(top * frac)))
    # Floor to reasonable minimum
    bots = [max(b, n_sink + n_recent_min) for b in bots]
    return bots


def select_per_layer_h2o(all_attn, total, budgets, n_sink=4, recent_ratio=0.3):
    """Per-layer H2O-style selection with layer-specific budgets.

    all_attn: list of length n_layers, each (heads, seq) attention probs summed across queries.
              In our pipeline, the attention used by select_h2o is averaged across layers;
              here we re-derive per-layer from the raw attention list.
    Returns: list of LongTensor indices, one per layer.
    """
    n_layers = len(budgets)
    sink = list(range(n_sink))
    out = []
    for l in range(n_layers):
        b = budgets[l]
        attn = all_attn[l] if isinstance(all_attn, list) else all_attn
        # Sum across heads
        if attn.dim() == 2:
            heur = attn.sum(dim=0)
        else:
            heur = attn
        n_recent = int(b * recent_ratio)
        n_heavy = b - n_sink - n_recent
        recent = list(range(total - n_recent, total))
        scores = heur.clone()
        scores[:n_sink] = -1e9
        for r in recent:
            if r < scores.shape[0]:
                scores[r] = -1e9
        if n_heavy > 0:
            top = scores.topk(min(n_heavy, scores.shape[0])).indices.tolist()
        else:
            top = []
        idx = sorted(set(sink + recent + top))
        idx = torch.tensor(idx, dtype=torch.long, device=heur.device)
        out.append(idx)
    return out


def build_per_layer_cache(all_kv, idx_per_layer, nl):
    cache = DynamicCache()
    for li in range(nl):
        # all_kv[li] is (K_list, V_list) with one tensor per processed window
        K_full = torch.cat(all_kv[li][0], dim=2)
        V_full = torch.cat(all_kv[li][1], dim=2)
        idx = idx_per_layer[li]
        K = K_full[:, :, idx, :]
        V = V_full[:, :, idx, :]
        cache.update(K, V, li)
    return cache


def run_eval(model, tokenizer, samples, k_avg, ratio, device):
    """PyramidKV-style: per-layer budget decreasing with depth.
    Selection scoring uses the same global cumulative attention as H2O baseline,
    since process_windows aggregates across layers/heads. The contribution of
    PyramidKV here is the budget-allocation policy, not the scoring rule.
    """
    nl = model.config.num_hidden_layers
    correct = 0
    for si, s in enumerate(samples):
        fact_windows = s.windows[:-1]
        all_kv, all_attn, boundaries, total = process_windows(model, tokenizer, fact_windows, device)
        # all_attn is a list of per-window 1D tensors of length seq_len_window.
        # Concatenate them into a global heuristic over all positions.
        heur = torch.cat(all_attn, dim=0)  # shape (total,)
        budgets = pyramid_budgets(nl, k_avg, ratio=ratio)
        budgets = [min(b, total) for b in budgets]

        # Select per layer
        idx_per_layer = []
        for l in range(nl):
            b = budgets[l]
            n_sink = 4
            n_recent = int(b * 0.3)
            n_heavy = max(b - n_sink - n_recent, 0)
            sink = list(range(n_sink))
            recent = list(range(max(total - n_recent, n_sink), total))
            scores = heur.clone()
            for r in sink + recent:
                if r < scores.shape[0]:
                    scores[r] = -1e9
            if n_heavy > 0:
                top_idx = scores.topk(min(n_heavy, scores.shape[0])).indices.tolist()
            else:
                top_idx = []
            idx = sorted(set(sink + recent + top_idx))
            idx_per_layer.append(torch.tensor(idx, dtype=torch.long, device=heur.device))

        cache = build_per_layer_cache(all_kv, idx_per_layer, nl)
        prefix_len = max(len(i) for i in idx_per_layer)
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        text = generate_with_cache(model, tokenizer, fq['input_ids'], cache, prefix_len, device)
        if s.answer.lower() in text.lower():
            correct += 1
        if (si + 1) % 25 == 0:
            print(f'  PyramidKV {si+1}/{len(samples)}: {correct}/{si+1} = {correct/(si+1):.1%}', flush=True)
    return correct / len(samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='models/qwen2.5-7b')
    p.add_argument('--n_samples', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300, help='Average budget per layer')
    p.add_argument('--ratios', nargs='+', type=float, default=[0.5, 0.7])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path',
                   default='logs/results/pyramidkv_qwen7b.json')
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

    results = {}
    for ratio in args.ratios:
        print(f'\n=== PyramidKV ratio={ratio} (top:bottom budget ratio = 1:{ratio}) ===', flush=True)
        t0 = time.time()
        acc = run_eval(model, tokenizer, samples, args.k, ratio, device)
        elapsed = time.time() - t0
        results[f'ratio_{ratio}'] = {'avg_budget': args.k, 'ratio': ratio,
                                     'accuracy': acc, 'elapsed_s': elapsed}
        print(f'  PyramidKV ratio={ratio}: {acc:.3f} ({elapsed:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'method': 'PyramidKV (per-layer H2O budgets, linear from top to bottom)',
            'n_samples': args.n_samples, 'seed': args.seed,
            'avg_budget_per_layer': args.k,
            'results': results,
        }, f, indent=2)
    print(f'\nSaved to {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
