"""Streaming-mode LongBench eval for SnapKV / H2O / PyramidKV.

Mirrors the SQuAD streaming protocol of
``eval_upstream_baselines_streaming.py``:

  1. Chunk the LongBench document into windows of ``chunk_size`` tokens.
  2. Process windows sequentially with ``use_cache=True``; collect the
     cumulative K / V (no per-window compression).
  3. Forward the question text separately to obtain question-conditioned
     Q at the inject layers.
  4. Score cumulative K against question Q with the method's
     pooling+selection rule.
  5. Build a fresh cache with selected K, V and decode greedily.

One method per process invocation.

Usage::

    python baselines/eval_upstream_longbench_streaming.py \\
        --model_path ./models/qwen2.5-7b --method snapkv --k 300 \\
        --tasks hotpotqa multifieldqa_en 2wikimqa musique \\
        --n_samples 100 \\
        --save_path logs/results/upstream_longbench_snapkv_qwen7b_streaming.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.eval_upstream_baselines_streaming import (
    _family_from_path, load_backbone, _inject_layers,
    _question_q_at_layers, _score_method, _select_indices, _select_kv,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--method', required=True,
                   choices=['snapkv', 'h2o', 'pyramidkv'])
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--tasks', nargs='+',
                   default=['hotpotqa', 'multifieldqa_en', '2wikimqa',
                             'musique'])
    p.add_argument('--n_samples', type=int, default=100)
    p.add_argument('--chunk_size', type=int, default=384)
    p.add_argument('--max_chunks', type=int, default=20,
                   help='Cap context length to control cumulative cache memory; '
                         '20 chunks * 384 tokens = ~7.7k tokens')
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    family = _family_from_path(args.model_path)
    print(f'[streaming-longbench] family={family} method={args.method} '
          f'k={args.k}', flush=True)
    model, tokenizer = load_backbone(args.model_path, args.multi_gpu)
    inject_layers = _inject_layers(model.config.num_hidden_layers)

    import torch
    from datasets import load_dataset
    from transformers import DynamicCache
    from baselines.eval_longbench import TASKS, PROMPTS, chunk_text
    from baselines.longbench_canonical_f1 import f1_score as canonical_f1

    cfg = model.config
    nl = cfg.num_hidden_layers
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // nq)
    embed_dev = model.get_input_embeddings().weight.device

    all_results = {}
    for task in args.tasks:
        if task not in TASKS:
            print(f'skipping unknown task {task!r}', flush=True); continue
        print(f'\n=== task={task} ===', flush=True)
        ds = load_dataset('THUDM/LongBench', task, split='test',
                           trust_remote_code=True)
        samples = list(ds)[:args.n_samples]
        max_gen = TASKS[task]['max_gen']
        prompt_tpl = PROMPTS[task]
        f1s = []
        t0 = time.time()
        for si, s in enumerate(samples):
          with torch.no_grad():  # noqa: E111 — keep memory bounded across windows
            chunks = chunk_text(s['context'], tokenizer, args.chunk_size)
            if len(chunks) < 2:
                # Too short to be meaningful for streaming compression; skip.
                continue
            # Cap context length so cumulative cache fits in 24 GB.
            # We left-truncate (keep recent chunks) since LongBench
            # questions tend to depend on later context for QA tasks.
            if len(chunks) > args.max_chunks:
                chunks = chunks[-args.max_chunks:]
            # 1) AstroNet S1 protocol: process each window FRESH
            #    (no past_key_values), then concat per-window K/V along
            #    the seq dim.  Each window's K is RoPE-rotated at
            #    positions [0, win_len), matching the question Q's
            #    positions for meaningful cross-attention scoring.
            per_K = {li: [] for li in range(nl)}
            per_V = {li: [] for li in range(nl)}
            last_w_size = 0
            for win in chunks:
                ids = tokenizer(win, return_tensors='pt', truncation=True,
                                 max_length=args.chunk_size).to(embed_dev)
                out = model(input_ids=ids['input_ids'], use_cache=True)
                pkv = out.past_key_values
                last_w_size = ids['input_ids'].shape[1]
                for li in range(nl):
                    per_K[li].append(pkv.key_cache[li])
                    per_V[li].append(pkv.value_cache[li])
                torch.cuda.empty_cache()
            all_kv = []
            for li in range(nl):
                K = torch.cat(per_K[li], dim=2)
                V = torch.cat(per_V[li], dim=2)
                all_kv.append((K, V))
            total_len = all_kv[0][0].shape[-2]

            # 2) Question Q.  Use the tail of the LongBench prompt
            #    (Question: ... Answer:) so scoring is conditioned on
            #    the same string used during decode (critic fix #10).
            tail0 = prompt_tpl.split('{context}', 1)[-1].replace(
                '{input}', s['input'])
            q_states = _question_q_at_layers(
                model, tokenizer, tail0, inject_layers, embed_dev)

            # 3) Score with method-specific pooling.
            if args.method == 'snapkv':
                cross = _score_method(all_kv, q_states, inject_layers, nq, nkv,
                                       hd, total_len, 'snapkv', 7, 'maxpool')
            elif args.method == 'h2o':
                cross = _score_method(all_kv, q_states, inject_layers, nq, nkv,
                                       hd, total_len, 'h2o', 1, 'avgpool')
            else:
                cross = _score_method(all_kv, q_states, inject_layers, nq, nkv,
                                       hd, total_len, 'pyramidkv', 5, 'maxpool')
            sel = _select_indices(cross, total_len, args.k, args.method,
                                    last_w_size, nl, beta=20)

            # 4) Build fresh cache (single selection across all layers).
            new_cache = DynamicCache()
            for li in range(nl):
                K, V = _select_kv(all_kv, li, sel)
                new_cache.update(K, V, li)
            prefix_len = sel.shape[0]

            # Question prompt (tail of the LongBench template).
            tail = prompt_tpl.split('{context}', 1)[-1]
            qp = tail.replace('{input}', s['input'])
            qids = tokenizer(qp, return_tensors='pt', truncation=True,
                              max_length=512).to(embed_dev)
            pos = torch.arange(prefix_len, prefix_len + qids['input_ids'].shape[1],
                                device=embed_dev).unsqueeze(0)
            cur = qids['input_ids']; gen = []
            for _ in range(max_gen):
                out = model(input_ids=cur, past_key_values=new_cache,
                              position_ids=pos, use_cache=True)
                new_cache = out.past_key_values
                nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                tok_id = int(nxt[0, 0].item())
                gen.append(tok_id)
                if tok_id == tokenizer.eos_token_id: break
                cur = nxt.to(embed_dev)
                pos = torch.tensor(
                    [[prefix_len + qids['input_ids'].shape[1] + len(gen) - 1]],
                    device=embed_dev)
            pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
            pred = pred.split('\n')[0].strip()
            f1 = canonical_f1(pred, s.get('answers', []))
            f1s.append(f1)
            if (si + 1) % 25 == 0:
                print(f'  {si+1}/{len(samples)} mean f1 = {sum(f1s)/len(f1s):.2f}',
                      flush=True)
        mean_f1 = sum(f1s) / max(1, len(f1s))
        all_results[task] = mean_f1
        print(f'  task={task} f1 = {mean_f1:.2f}  '
               f'({time.time() - t0:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'family': family, 'method': args.method, 'k': args.k,
            'n_samples': args.n_samples, 'tasks': args.tasks,
            'results_f1': all_results,
            'source': 'streaming-mode LongBench (cumulative KV, '
                       'question-conditioned scoring, AstroNet S1 protocol)',
        }, f, indent=2)
    print(f'\nSaved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
