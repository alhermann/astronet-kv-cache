"""Faithful-baseline needle-in-a-haystack evaluation.

Same haystack generator and depth/trial counts as ``eval_needle.py``,
but the SnapKV / H2O / PyramidKV columns are produced by the faithful
implementations in ``baselines/faithful_snapkv.py``,
``baselines/faithful_h2o.py``, and ``baselines/faithful_pyramidkv.py``.

Usage::

    python baselines/eval_faithful_needle.py \
        --model_path ./models/qwen2.5-7b \
        --n_windows_list 5 10 20 --n_trials 20 \
        --k 300 --methods snapkv h2o pyramidkv \
        --save_path logs/results/faithful_needle_qwen7b.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from baselines.eval_needle import generate_haystack, NEEDLES
from baselines.eval_faithful_baselines import load_backbone, _generate

from baselines.faithful_snapkv import select_snapkv_faithful
from baselines.faithful_h2o import (
    process_windows_perlayer, select_h2o_per_layer, build_cache_perlayer,
)
from baselines.faithful_pyramidkv import (
    select_pyramidkv_per_layer, build_pyramidkv_cache,
)


DEPTH_KEYS = ['start', '25%', '50%', '75%', 'end']


def depth_positions(n_windows: int):
    """Return (start, 25%, 50%, 75%, end) integer window indices."""
    if n_windows < 5:
        return list(range(n_windows))
    return [
        0,
        max(1, n_windows // 4),
        max(2, n_windows // 2),
        max(3, (3 * n_windows) // 4),
        n_windows - 1,
    ]


@torch.no_grad()
def eval_one_trial(model, tokenizer, windows, question, answer, k, device,
                   methods):
    """Run the three faithful selectors on one needle trial."""
    all_kv, attn_perlayer, total = process_windows_perlayer(
        model, tokenizer, windows, device)
    prompt = (f"Based on what you read earlier, answer the following question."
               f"\nQuestion: {question}\nAnswer:")
    nl = model.config.num_hidden_layers

    out = {}
    if 'snapkv' in methods:
        idx = select_snapkv_faithful(model, all_kv, question, tokenizer,
                                      k=k, device=device)
        cache = DynamicCache()
        for li in range(nl):
            K = torch.cat(all_kv[li][0], dim=2)
            V = torch.cat(all_kv[li][1], dim=2)
            li_idx = idx.to(K.device)
            cache.update(K[:, :, li_idx, :], V[:, :, li_idx, :], li)
        ans = _generate(model, tokenizer, prompt, cache, len(idx), device)
        out['snapkv'] = int(answer.lower() in ans.lower())

    if 'h2o' in methods:
        idx_per_layer = select_h2o_per_layer(attn_perlayer, total, k=k)
        cache = build_cache_perlayer(all_kv, idx_per_layer, nl)
        prefix_len = max(int(i.shape[0]) for i in idx_per_layer)
        ans = _generate(model, tokenizer, prompt, cache, prefix_len, device)
        out['h2o'] = int(answer.lower() in ans.lower())

    if 'pyramidkv' in methods:
        idx_per_layer = select_pyramidkv_per_layer(
            model, all_kv, question, tokenizer,
            avg_budget=k, device=device)
        cache = build_pyramidkv_cache(all_kv, idx_per_layer, nl)
        prefix_len = max(int(i.shape[0]) for i in idx_per_layer)
        ans = _generate(model, tokenizer, prompt, cache, prefix_len, device)
        out['pyramidkv'] = int(answer.lower() in ans.lower())

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--n_windows_list', nargs='+', type=int, default=[5, 10, 20])
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--seed_offset', type=int, default=0)
    p.add_argument('--methods', nargs='+', default=['snapkv', 'h2o', 'pyramidkv'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    model, tokenizer = load_backbone(args.model_path, args.device,
                                      args.multi_gpu)
    device = str(model.get_input_embeddings().weight.device)

    all_results = {}
    for n_win in args.n_windows_list:
        key = f'n{n_win}'
        depths = depth_positions(n_win)
        all_results[key] = {m: {dk: 0 for dk in DEPTH_KEYS}
                             for m in args.methods}

        for di, depth in enumerate(depths):
            dk = DEPTH_KEYS[di]
            for trial in range(args.n_trials):
                seed = trial + args.seed_offset
                needle_idx = seed % len(NEEDLES)
                windows, question, answer = generate_haystack(
                    n_win, depth, needle_idx=needle_idx, seed=seed)
                t0 = time.time()
                results = eval_one_trial(
                    model, tokenizer, windows, question, answer,
                    args.k, device, args.methods)
                for m, c in results.items():
                    all_results[key][m][dk] += c
            for m in args.methods:
                all_results[key][m][dk] /= args.n_trials
            line = '  '.join(f'{m}={all_results[key][m][dk]*100:.0f}%'
                              for m in args.methods)
            print(f'  n={n_win} depth={dk}: {line}', flush=True)

    # Per-n averages.
    averages = {key: {m: sum(all_results[key][m].values()) / len(DEPTH_KEYS)
                       for m in args.methods}
                 for key in all_results}

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'k': args.k,
            'n_trials': args.n_trials,
            'seed_offset': args.seed_offset,
            'methods': args.methods,
            'results': all_results,
            'average_per_n': averages,
        }, f, indent=2)
    print(f'\nSaved -> {args.save_path}')


if __name__ == '__main__':
    main()
