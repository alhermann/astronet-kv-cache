"""Upstream-faithful Needle-in-a-Haystack eval (single-prompt).

Concatenates all haystack windows into one prefill prompt + a final
question.  The upstream monkey-patch compresses the KV cache during
prefill; generation is greedy.

Same n_windows / depth / n_trials grid as ``eval_needle.py`` so columns
remain directly comparable.

Usage::

    python baselines/eval_upstream_needle.py \\
        --model_path ./models/qwen2.5-7b --method snapkv --k 300 \\
        --n_windows_list 5 10 20 --n_trials 20 \\
        --save_path logs/results/upstream_snapkv_needle_qwen7b.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.eval_upstream_baselines import (
    _family_from_path, _apply_monkey_patch, _set_config_hparams,
    load_backbone, _reset_kv_state,
)


DEPTH_KEYS = ['start', '25%', '50%', '75%', 'end']


def depth_positions(n_windows: int):
    if n_windows < 5:
        return list(range(n_windows))
    return [
        0,
        max(1, n_windows // 4),
        max(2, n_windows // 2),
        max(3, (3 * n_windows) // 4),
        n_windows - 1,
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--method', required=True,
                   choices=['snapkv', 'h2o', 'pyramidkv'])
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_windows_list', nargs='+', type=int, default=[5, 10, 20])
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--seed_offset', type=int, default=0)
    p.add_argument('--max_input_tokens', type=int, default=16384)
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    family = _family_from_path(args.model_path)
    print(f'[upstream-needle] family={family} method={args.method} '
          f'k={args.k}', flush=True)

    # 1) Monkey-patch BEFORE model load.
    _apply_monkey_patch(family, args.method)

    # 2) Load model.
    model, tokenizer = load_backbone(args.model_path, args.multi_gpu)

    # 3) Per-method config hyperparameters.
    _set_config_hparams(model.config, args.method, args.k)

    import torch
    from baselines.eval_needle import generate_haystack, NEEDLES

    embed_dev = model.get_input_embeddings().weight.device

    all_results = {}
    for n_win in args.n_windows_list:
        key = f'n{n_win}'
        depths = depth_positions(n_win)
        all_results[key] = {dk: 0 for dk in DEPTH_KEYS}

        for di, depth in enumerate(depths):
            dk = DEPTH_KEYS[di]
            hits = 0
            t0 = time.time()
            for trial in range(args.n_trials):
                seed = trial + args.seed_offset
                needle_idx = seed % len(NEEDLES)
                windows, question, answer = generate_haystack(
                    n_win, depth, needle_idx=needle_idx, seed=seed)
                # single-prompt: concat windows + question
                context = '\n\n'.join(windows)
                prompt = (f"{context}\n\nBased on what you read earlier, "
                           f"answer the following question.\n"
                           f"Question: {question}\nAnswer:")
                tokenizer.truncation_side = 'left'
                ids = tokenizer(prompt, return_tensors='pt',
                                 truncation=True,
                                 max_length=args.max_input_tokens).to(embed_dev)
                _reset_kv_state(model)
                with torch.no_grad():
                    out = model.generate(
                        **ids, max_new_tokens=32, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id, use_cache=True,
                    )
                gen_ids = out[0, ids['input_ids'].shape[1]:]
                pred = tokenizer.decode(
                    gen_ids, skip_special_tokens=True).strip()
                if answer.lower() in pred.lower():
                    hits += 1
            all_results[key][dk] = hits / args.n_trials
            print(f'  n={n_win} depth={dk}: '
                   f'{hits}/{args.n_trials} ({hits/args.n_trials*100:.0f}%)  '
                   f'({time.time()-t0:.0f}s)', flush=True)

    averages = {key: sum(all_results[key].values()) / len(DEPTH_KEYS)
                 for key in all_results}

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'family': family,
            'method': args.method,
            'k': args.k,
            'n_trials': args.n_trials,
            'seed_offset': args.seed_offset,
            'results': all_results,
            'average_per_n': averages,
            'source': 'upstream kvcache_factory monkey-patch (single-prompt)',
        }, f, indent=2)
    print(f'\nSaved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
