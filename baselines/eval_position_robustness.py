"""Position-robustness sanity check: vary which window holds the fact context.

The default SQuAD multi-window builder always places the fact at window 0.
Reviewers may worry that some baselines are biased by this choice.
This script regenerates the dataset with the fact placed at a chosen position
among the non-query windows, then re-runs the four main KV selection methods
at fixed k=300, n_windows=5 on Qwen 7B.
"""

import sys, os, argparse, json, time
sys.path.insert(0, '.')
sys.path.insert(0, 'baselines')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from data.real_qa import generate_squad_dataset, CrossContextSample
from eval_all_baselines import (
    process_windows, build_cache_from_indices, generate_with_cache,
    select_streaming_llm, select_h2o, select_snapkv, select_multiplicative,
)


def shuffle_fact_position(samples, position):
    """Move the fact to position-th non-query window. position=0 is default."""
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


def eval_methods(model, tokenizer, samples, k, device, methods):
    nl = model.config.num_hidden_layers
    correct = {m: 0 for m in methods}
    for si, s in enumerate(samples):
        fact_windows = s.windows[:-1]
        all_kv, all_attn, boundaries, total = process_windows(model, tokenizer, fact_windows, device)
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        for method in methods:
            if method == 'streaming_llm':
                idx = select_streaming_llm(total, k).to(device)
            elif method == 'h2o':
                idx = select_h2o(all_attn, k, boundaries)
            elif method == 'snapkv':
                idx = select_snapkv(model, all_kv, s.question, tokenizer, k, device, boundaries)
            elif method == 'multiplicative':
                idx = select_multiplicative(all_attn, model, all_kv, s.question, tokenizer, k, device)
            cache = build_cache_from_indices(all_kv, idx, nl)
            text = generate_with_cache(model, tokenizer, fq['input_ids'], cache, len(idx), device)
            if s.answer.lower() in text.lower():
                correct[method] += 1
        if (si + 1) % 25 == 0:
            parts = [f'{m}={correct[m]}/{si+1}' for m in methods]
            print(f'    {si+1}/{len(samples)}: {", ".join(parts)}', flush=True)
    return {m: correct[m] / len(samples) for m in methods}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='models/qwen2.5-7b')
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--methods', nargs='+',
                   default=['streaming_llm', 'h2o', 'snapkv', 'multiplicative'])
    p.add_argument('--save_path',
                   default='logs/results/position_robustness.json')
    p.add_argument('--device', default='cuda:0')
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

    base_samples = generate_squad_dataset(n_samples=args.n_samples, n_windows=5,
                                           vary_distance=True, seed=args.seed,
                                           split='validation')

    results = {}
    for pos in args.positions:
        print(f'\n=== Fact at non-query window {pos} ===', flush=True)
        samples = shuffle_fact_position(base_samples, pos)
        t0 = time.time()
        accs = eval_methods(model, tokenizer, samples, args.k, device, args.methods)
        elapsed = time.time() - t0
        results[f'pos_{pos}'] = accs
        print(f'  fact_pos={pos}: ' + ', '.join([f'{m}={accs[m]:.3f}' for m in args.methods])
              + f' ({elapsed:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'n_samples': args.n_samples, 'seed': args.seed, 'k': args.k,
            'note': ('Fact context placed at the position-th non-query window. '
                     'pos=0 is the default setup; pos>=1 places fact deeper into the prefix. '
                     'Distractors fill the remaining slots, query is always last.'),
            'results': results,
        }, f, indent=2)
    print(f'\nSaved to {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
