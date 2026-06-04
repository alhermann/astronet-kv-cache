"""Unified runner for the faithful baseline reimplementations.

Replaces the lead role of the older ``select_snapkv`` (multi-layer + full
question text), ``select_h2o`` (global cumulative attention applied
uniformly across layers), and ``eval_pyramidkv.py`` (H2O scoring + pyramid
budgets) with faithful implementations that match the published
algorithms.  See ``faithful_snapkv.py``, ``faithful_h2o.py``, and
``faithful_pyramidkv.py`` for the rationale on each method.

Currently supports the position-robust multi-window SQuAD benchmark.
Needle and LongBench variants are added in companion scripts that share
the same selection backbones.

Usage::

    python baselines/eval_faithful_baselines.py \
        --model_path ./models/qwen2.5-7b \
        --n_eval 100 --seed 42 --k 300 \
        --methods snapkv h2o pyramidkv \
        --save_path logs/results/faithful_baselines_qwen7b.json
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from data.real_qa import generate_squad_dataset
from training.eval_hybrid_position_robust import shuffle_fact_position

from baselines.faithful_snapkv import select_snapkv_faithful
from baselines.faithful_h2o import (
    process_windows_perlayer, select_h2o_per_layer, build_cache_perlayer,
)
from baselines.faithful_pyramidkv import (
    select_pyramidkv_per_layer, build_pyramidkv_cache,
)


def load_backbone(model_path: str, device: str, multi_gpu: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    if multi_gpu:
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            if gib >= 16:
                max_memory[i] = '22GiB'
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map='auto',
            max_memory=max_memory, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb, device_map={'': device},
            torch_dtype=torch.float16)
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, query, cache, prefix_len, device,
              max_tokens: int = 20):
    embed_dev = model.get_input_embeddings().weight.device
    fq = tokenizer(query, return_tensors='pt', max_length=384,
                    truncation=True).to(embed_dev)
    pos = torch.arange(prefix_len, prefix_len + fq['input_ids'].shape[1],
                       device=embed_dev).unsqueeze(0)
    cur, cc, gen = fq['input_ids'], cache, []
    with torch.no_grad():
        for _ in range(max_tokens):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item())
            cur = nxt.to(embed_dev)
            pos = torch.tensor(
                [[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                device=embed_dev)
            if nxt[0, 0].item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate_at_position(model, tokenizer, samples, pos, k, device, methods):
    """Run each faithful baseline on the position-shuffled samples."""
    placed = shuffle_fact_position(samples, pos)
    correct = {m: 0 for m in methods}

    for si, s in enumerate(placed):
        windows = s.windows[:-1]  # everything except the query-prompt window

        # --- single pass that captures KV and per-(layer, head) attention ---
        all_kv, attn_perhead, total = process_windows_perlayer(
            model, tokenizer, windows, device)

        prompt = (f"Based on what you read earlier, answer the following "
                   f"question.\nQuestion: {s.question}\nAnswer:")

        # ---------------- faithful SnapKV ----------------
        if 'snapkv' in methods:
            idx = select_snapkv_faithful(model, all_kv, s.question,
                                          tokenizer, k=k, device=device)
            cache = DynamicCache()
            for li in range(model.config.num_hidden_layers):
                K = torch.cat(all_kv[li][0], dim=2)
                V = torch.cat(all_kv[li][1], dim=2)
                li_idx = idx.to(K.device)
                cache.update(K[:, :, li_idx, :], V[:, :, li_idx, :], li)
            ans = _generate(model, tokenizer, prompt, cache, len(idx), device)
            if s.answer.lower() in ans.lower():
                correct['snapkv'] += 1

        # ---------------- faithful H2O ----------------
        if 'h2o' in methods:
            idx_per_layer = select_h2o_per_layer(attn_perhead, total, k=k)
            cache = build_cache_perlayer(
                all_kv, idx_per_layer,
                model.config.num_hidden_layers)
            prefix_len = max(int(i.shape[0]) for i in idx_per_layer)
            ans = _generate(model, tokenizer, prompt, cache, prefix_len,
                             device)
            if s.answer.lower() in ans.lower():
                correct['h2o'] += 1

        # ---------------- faithful PyramidKV ----------------
        if 'pyramidkv' in methods:
            idx_per_layer = select_pyramidkv_per_layer(
                model, all_kv, s.question, tokenizer,
                avg_budget=k, device=device)
            cache = build_pyramidkv_cache(
                all_kv, idx_per_layer,
                model.config.num_hidden_layers)
            prefix_len = max(int(i.shape[0]) for i in idx_per_layer)
            ans = _generate(model, tokenizer, prompt, cache, prefix_len,
                             device)
            if s.answer.lower() in ans.lower():
                correct['pyramidkv'] += 1

        if (si + 1) % 25 == 0:
            line = ' '.join(f'{m}={correct[m]}/{si+1}' for m in methods)
            print(f'  pos={pos} {si+1}/{len(placed)} {line}', flush=True)

    return {m: correct[m] / max(1, len(placed)) for m in methods}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--methods', nargs='+', default=['snapkv', 'h2o',
                                                      'pyramidkv'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    print(f'[faithful-baselines] model={args.model_path}  k={args.k}  '
          f'methods={args.methods}', flush=True)

    model, tokenizer = load_backbone(args.model_path, args.device,
                                      args.multi_gpu)
    device = str(model.get_input_embeddings().weight.device)

    base = generate_squad_dataset(n_samples=args.n_eval, n_windows=5,
                                   vary_distance=True, seed=args.seed,
                                   split='validation')

    results = {}
    for pos in args.positions:
        print(f'\n=== pos={pos} ===', flush=True)
        t0 = time.time()
        accs = evaluate_at_position(model, tokenizer, base, pos, args.k,
                                     device, args.methods)
        results[f'pos_{pos}'] = accs
        line = '  '.join(f'{m}={accs[m]*100:.1f}%' for m in args.methods)
        print(f'  pos={pos}: {line}  ({time.time() - t0:.0f}s)',
              flush=True)

    averages = {m: sum(results[f'pos_{p}'][m]
                        for p in args.positions) / len(args.positions)
                 for m in args.methods}
    print('\nAverages:',
          '  '.join(f'{m}={averages[m]*100:.1f}%'
                     for m in args.methods))

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'n_eval': args.n_eval, 'seed': args.seed, 'k': args.k,
            'methods': args.methods,
            'results': results,
            'average': averages,
            'notes': 'Faithful baselines: SnapKV (single-layer L-1, '
                       'last-32 observation window), H2O (per-layer cumulative '
                       'attention, sinks + recent strip), '
                       'PyramidKV (per-layer SnapKV-style scoring + '
                       'decreasing per-layer budget).',
        }, f, indent=2)
    print(f'Saved -> {args.save_path}')


if __name__ == '__main__':
    main()
