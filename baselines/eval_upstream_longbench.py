"""Upstream-faithful LongBench eval (single-prompt).

Runs SnapKV / H2O / PyramidKV from the upstream kvcache_factory
monkey-patches on bnb-4bit-loaded backbones, using the LongBench
canonical prompt template for each task.

One method per process invocation (monkey-patches are global).

Usage::

    python baselines/eval_upstream_longbench.py \\
        --model_path ./models/llama-3.1-8b \\
        --method snapkv --k 1024 \\
        --tasks hotpotqa multifieldqa_en 2wikimqa musique \\
        --n_samples 100 \\
        --save_path logs/results/upstream_snapkv_longbench_llama8b.json
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--method', required=True,
                   choices=['snapkv', 'h2o', 'pyramidkv'])
    p.add_argument('--k', type=int, default=1024,
                   help='max_capacity_prompt (avg per-layer for pyramidkv); '
                         'paper-default 1024 for the SnapKV/PyramidKV trend rows')
    p.add_argument('--tasks', nargs='+',
                   default=['hotpotqa', 'multifieldqa_en', '2wikimqa',
                             'musique'])
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--max_input_tokens', type=int, default=16384,
                   help='cap context to avoid OOM; LongBench paper uses 32k')
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    family = _family_from_path(args.model_path)
    print(f'[upstream-longbench] family={family} method={args.method} '
          f'k={args.k}', flush=True)

    # 1) Monkey-patch BEFORE model load.
    _apply_monkey_patch(family, args.method)

    # 2) Load model.
    model, tokenizer = load_backbone(args.model_path, args.multi_gpu)

    # 3) Apply per-method config hyperparameters.
    _set_config_hparams(model.config, args.method, args.k)

    # Now safe to import torch + datasets etc.
    import torch
    from datasets import load_dataset
    from baselines.eval_longbench import TASKS, PROMPTS
    from baselines.longbench_canonical_f1 import f1_score as canonical_f1

    embed_dev = model.get_input_embeddings().weight.device

    all_results = {}
    for task in args.tasks:
        if task not in TASKS:
            print(f'[upstream-longbench] skipping unknown task {task!r}',
                  flush=True)
            continue
        print(f'\n=== task={task} ===', flush=True)
        ds = load_dataset('THUDM/LongBench', task, split='test',
                           trust_remote_code=True)
        samples = list(ds)[:args.n_samples]
        max_gen = TASKS[task]['max_gen']
        prompt_tpl = PROMPTS[task]

        f1s = []
        t0 = time.time()
        for si, s in enumerate(samples):
            prompt = prompt_tpl.format(context=s['context'], input=s['input'])
            # Left-truncate so the question + "Answer:" tail is preserved
            # when context exceeds max_input_tokens (LongBench paper truncates
            # from the middle; left-trunc preserves the answer cue, which is
            # the part that catastrophically harms eval when dropped).
            tokenizer.truncation_side = 'left'
            ids = tokenizer(prompt, return_tensors='pt',
                             truncation=True,
                             max_length=args.max_input_tokens).to(embed_dev)
            _reset_kv_state(model)
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=max_gen, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id, use_cache=True,
                )
            gen_ids = out[0, ids['input_ids'].shape[1]:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred = pred.split('\n')[0].strip()
            f1 = canonical_f1(pred, s.get('answers', []))
            f1s.append(f1)
            if (si + 1) % 25 == 0:
                print(f'  {si+1}/{len(samples)} '
                       f'running mean f1 = {sum(f1s)/len(f1s):.2f}',
                       flush=True)
        mean_f1 = sum(f1s) / max(1, len(f1s))
        all_results[task] = mean_f1
        print(f'  task={task} mean f1 = {mean_f1:.2f}  '
               f'({time.time() - t0:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'family': family,
            'method': args.method,
            'k': args.k,
            'n_samples': args.n_samples,
            'tasks': args.tasks,
            'results_f1': all_results,
            'source': 'upstream kvcache_factory monkey-patch (single-prompt)',
        }, f, indent=2)
    print(f'\nSaved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
