"""Faithful-baseline LongBench evaluation.

Same task/prompt setup as ``eval_longbench.py`` (HotpotQA, MultiFieldQA,
2WikiMultihopQA, MuSiQue) but the SnapKV / H2O / PyramidKV columns are
produced by the faithful implementations in
``baselines/faithful_*.py``.

Usage::

    python baselines/eval_faithful_longbench.py \
        --model_path ./models/qwen2.5-7b \
        --tasks hotpotqa multifieldqa_en 2wikimqa musique \
        --n_samples 100 --k 300 \
        --methods snapkv h2o pyramidkv \
        --save_path logs/results/faithful_longbench_qwen7b.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from transformers import DynamicCache
from baselines.eval_longbench import TASKS, PROMPTS, chunk_text, f1_score
from baselines.eval_faithful_baselines import load_backbone

from baselines.faithful_snapkv import select_snapkv_faithful
from baselines.faithful_h2o import (
    process_windows_perlayer, select_h2o_per_layer, build_cache_perlayer,
)
from baselines.faithful_pyramidkv import (
    select_pyramidkv_per_layer, build_pyramidkv_cache,
)


@torch.no_grad()
def _gen_after_cache(model, tokenizer, prompt, cache, prefix_len, device,
                      max_gen: int):
    embed_dev = model.get_input_embeddings().weight.device
    pids = tokenizer(prompt, return_tensors='pt',
                      max_length=512, truncation=True).to(embed_dev)
    pos = torch.arange(prefix_len, prefix_len + pids['input_ids'].shape[1],
                       device=embed_dev).unsqueeze(0)
    cur, cc, gen = pids['input_ids'], cache, []
    for _ in range(max_gen):
        o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
        cc = o.past_key_values
        nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        gen.append(nxt[0, 0].item())
        cur = nxt.to(embed_dev)
        pos = torch.tensor(
            [[prefix_len + pids['input_ids'].shape[1] + len(gen) - 1]],
            device=embed_dev)
        if nxt[0, 0].item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def eval_one_sample(model, tokenizer, context_chunks, question, prompt_tpl,
                     k, device, methods, max_gen):
    all_kv, attn_perlayer, total = process_windows_perlayer(
        model, tokenizer, context_chunks, device)
    nl = model.config.num_hidden_layers

    # The full prompt re-includes the question after the context (the
    # context lives in the cache; the trailing question + answer marker
    # is what we generate from).  We strip the context placeholder from
    # the canonical prompt and feed only the question half.
    tail = prompt_tpl.split('{context}\n\n', 1)[-1]
    final_prompt = tail.replace('{input}', question)

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
        out['snapkv'] = _gen_after_cache(model, tokenizer, final_prompt,
                                          cache, len(idx), device, max_gen)

    if 'h2o' in methods:
        idx_per_layer = select_h2o_per_layer(attn_perlayer, total, k=k)
        cache = build_cache_perlayer(all_kv, idx_per_layer, nl)
        prefix_len = max(int(i.shape[0]) for i in idx_per_layer)
        out['h2o'] = _gen_after_cache(model, tokenizer, final_prompt, cache,
                                       prefix_len, device, max_gen)

    if 'pyramidkv' in methods:
        idx_per_layer = select_pyramidkv_per_layer(
            model, all_kv, question, tokenizer,
            avg_budget=k, device=device)
        cache = build_pyramidkv_cache(all_kv, idx_per_layer, nl)
        prefix_len = max(int(i.shape[0]) for i in idx_per_layer)
        out['pyramidkv'] = _gen_after_cache(model, tokenizer, final_prompt,
                                              cache, prefix_len, device,
                                              max_gen)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--tasks', nargs='+',
                   default=['hotpotqa', 'multifieldqa_en', '2wikimqa',
                             'musique'])
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--chunk_size', type=int, default=384)
    p.add_argument('--methods', nargs='+',
                   default=['snapkv', 'h2o', 'pyramidkv'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    model, tokenizer = load_backbone(args.model_path, args.device,
                                      args.multi_gpu)
    device = str(model.get_input_embeddings().weight.device)

    all_results = {}
    for task in args.tasks:
        print(f'\n=== task={task} ===', flush=True)
        data = load_dataset('THUDM/LongBench', task, split='test',
                              trust_remote_code=True)
        samples = list(data)[:args.n_samples]
        max_gen = TASKS[task]['max_gen']
        prompt_tpl = PROMPTS[task]

        scores = {m: [] for m in args.methods}
        for si, sample in enumerate(samples):
            context = sample['context']
            chunks = chunk_text(context, tokenizer, args.chunk_size)
            if len(chunks) < 2:
                continue
            question = sample['input']
            answers = sample.get('answers', [])

            t0 = time.time()
            preds = eval_one_sample(model, tokenizer, chunks, question,
                                      prompt_tpl, args.k, device,
                                      args.methods, max_gen)
            for m, pred in preds.items():
                scores[m].append(f1_score(pred, answers))
            if (si + 1) % 10 == 0:
                line = '  '.join(
                    f'{m}={sum(scores[m]) / len(scores[m]) * 100:.1f}'
                    for m in args.methods)
                print(f'  {si+1}/{len(samples)}: {line}  '
                       f'({time.time()-t0:.1f}s/sample)', flush=True)

        all_results[task] = {
            m: sum(scores[m]) / max(1, len(scores[m])) * 100
            for m in args.methods
        }
        print(f'  task={task} f1: ' + '  '.join(
            f'{m}={all_results[task][m]:.1f}' for m in args.methods))

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'k': args.k, 'n_samples': args.n_samples,
            'methods': args.methods,
            'results': all_results,
        }, f, indent=2)
    print(f'\nSaved -> {args.save_path}')


if __name__ == '__main__':
    main()
