"""DIAGNOSTIC: AstroNet Stage 1 selection in SINGLE-PROMPT mode.

Hypothesis (Cause 1, 2026-06-07): the streaming protocol throws away
inter-segment positional information that single-prompt baselines preserve.
This script tests it by applying our exact multiplicative Stage 1 score
to a SINGLE-PROMPT prefill cache (all segments concatenated into one
input) at k=300, then running greedy decode on the selected real tokens.

If single-prompt Stage 1 reaches ~75-80% on Qwen 7B (matching upstream
SnapKV), the streaming protocol was the dominant cause of AstroHybrid's
underperformance.  If it stays near streaming's ~58%, our scoring rule
itself is the problem.

NO Stage 2 (no virtual KV tokens) --- we are testing Stage 1 alone.
Comparison anchor (streaming Stage 1 on Qwen 7B at k=300, seed 42):
    pos_0=65, pos_1=65, pos_2=65, pos_3=36 -> avg = 57.8%
SnapKV upstream on Qwen 7B at k=150 has already landed at 79.8%; we
expect single-prompt Stage 1 to fall between those two anchors.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           BitsAndBytesConfig, DynamicCache)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.real_qa import generate_squad_dataset
from training.eval_hybrid_position_robust import shuffle_fact_position


def _find_answer(text: str, answer: str) -> bool:
    return answer.strip().lower() in text.strip().lower()


def evaluate_at_position(model, tokenizer, samples, position, k_total,
                          device, sense_layers, max_tokens=20):
    """Score single-prompt prefill with our multiplicative Stage 1 rule.

    For each sample:
      1) Concatenate all context segments into one prompt.
      2) Prefill in one forward pass with output_attentions=True.
      3) Multiplicative score:
             score[i] = sum_{layer in sense_layers} sum_{h, q} attn[layer][h, q, i]
             - i.e., cumulative attention received by token i at the
               four scoring layers (same recipe as streaming Stage 1).
      4) Query-conditioned cross-attention score from the QUESTION onto
         each cached position:
             cross[i] = sum_{layer in sense_layers} sum_{q_head} Q_q @ K[i]
      5) Final score = clamp(attn_score, 0) * clamp(cross_score, 0)
      6) Top-k selection with sink + recent reservation (matches the
         eval_hybrid_swap_selector.py recipe).
      7) Rebuild cache from selected indices, generate answer.

    Reports per-position accuracy on `samples`.
    """
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim',
                 model.config.hidden_size // nq)
    qpk = nq // nkv

    placed = shuffle_fact_position(samples, position)
    correct = 0
    n_sink = 4

    for si, s in enumerate(placed):
        # Concatenate ALL context segments (single-prompt regime).
        context = '\n\n'.join(s.windows[:-1])
        query_template = s.windows[-1]
        full_prompt = context + '\n\n' + query_template

        ctx_ids = tokenizer(context, return_tensors='pt',
                              max_length=8192, truncation=True).to(device)
        ctx_len = ctx_ids['input_ids'].shape[1]

        # 1) Single-prompt prefill on the CONTEXT only --- the cache we
        # then compress lives over context positions.
        with torch.no_grad():
            out = model(input_ids=ctx_ids['input_ids'],
                        use_cache=True,
                        output_attentions=True)
        kvs = out.past_key_values
        attns = out.attentions

        # 2) Cumulative attention score across the four sensing layers.
        attn_score = torch.zeros(ctx_len, device=device)
        for li in sense_layers:
            a = attns[li][0]  # (n_q_heads, ctx_len, ctx_len)
            if a.isnan().any(): continue
            attn_score += a.sum(dim=(0, 1)).to(device)
        attn_score[:n_sink] = -1e9

        # 3) Query-conditioned cross-attention.
        q_text = f"Question: {s.question}\nAnswer:"
        q_ids = tokenizer(q_text, return_tensors='pt', max_length=128,
                            truncation=True).to(device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'],
                          output_hidden_states=True)
        cross = torch.zeros(ctx_len, device=device)
        for li in sense_layers:
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float()).half().view(-1, nq, hd)
            K_full = kvs[li][0][0]  # (n_kv_heads, ctx_len, head_dim)
            for hi in range(nkv):
                sc = torch.matmul(
                    Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                    K_full[hi].float().T) / math.sqrt(hd)
                attn_w = torch.softmax(sc, dim=-1)
                cross += attn_w.sum(dim=(0, 1)).to(device)

        # 4) Multiplicative score with clamping.
        score = (attn_score.clamp(min=0).float()
                 * cross.clamp(min=0).float())
        score[:n_sink] = -1e9

        # 5) Reserve sinks + recent strip, then top-k the middle.
        n_recent = min(int(k_total * 0.2), ctx_len - n_sink)
        recent_start = ctx_len - n_recent
        recent_idx = torch.arange(recent_start, ctx_len, device=device)
        score[recent_start:ctx_len] = -1e9
        n_top = max(k_total - n_sink - n_recent, 0)
        n_avail = (score > -1e8).sum().item()
        n_top = min(n_top, n_avail)
        if n_top > 0:
            _, top = score.topk(n_top)
        else:
            top = torch.empty(0, dtype=torch.long, device=device)
        sink_idx = torch.arange(n_sink, device=device)
        idx = torch.cat([sink_idx, top, recent_idx]).unique().sort().values
        idx = idx[:k_total]

        # 6) Rebuild cache from selected indices.
        cache = DynamicCache()
        for li in range(nl):
            K_sel = kvs[li][0][:, :, idx, :]
            V_sel = kvs[li][1][:, :, idx, :]
            cache.update(K_sel, V_sel, li)
        cache_len = idx.shape[0]

        # 7) Manual greedy decode with the selected cache.
        # `model.generate` with pre-populated past_key_values has a
        # cache_position bug in transformers 4.46 (IndexError on empty
        # cache_position).  Doing the loop ourselves is more reliable.
        q_full = tokenizer(query_template,
                              return_tensors='pt', max_length=256,
                              truncation=True).to(device)
        q_ids = q_full['input_ids']
        running_cache = cache
        with torch.no_grad():
            # Feed the query prompt in one shot, then sample tokens one by one.
            pos = torch.arange(cache_len, cache_len + q_ids.shape[1],
                                device=device).unsqueeze(0)
            o = model(input_ids=q_ids,
                       past_key_values=running_cache,
                       position_ids=pos,
                       use_cache=True)
            running_cache = o.past_key_values
            next_pos = cache_len + q_ids.shape[1]
            gen_tokens = []
            for _ in range(max_tokens):
                next_id = o.logits[0, -1].argmax().item()
                if next_id == tokenizer.eos_token_id:
                    break
                gen_tokens.append(next_id)
                next_input = torch.tensor([[next_id]], device=device,
                                           dtype=torch.long)
                pos_t = torch.tensor([[next_pos]], device=device,
                                       dtype=torch.long)
                o = model(input_ids=next_input,
                           past_key_values=running_cache,
                           position_ids=pos_t,
                           use_cache=True)
                running_cache = o.past_key_values
                next_pos += 1
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        if _find_answer(text, s.answer):
            correct += 1
        if (si + 1) % 25 == 0:
            print(f'  pos={position} {si+1}/{len(placed)}: '
                   f'acc={correct}/{si+1}', flush=True)

    return correct / max(1, len(placed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    print(f'[diag-sp-s1] model={args.model_path}  k={args.k}  '
          f'n_eval={args.n_eval}  seed={args.seed}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = str(model.get_input_embeddings().weight.device)

    nl = model.config.num_hidden_layers
    sense_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    print(f'[diag-sp-s1] sense_layers = {sense_layers}', flush=True)

    base = generate_squad_dataset(
        n_samples=args.n_eval, n_windows=5, vary_distance=True,
        seed=args.seed, split='validation')

    results = {}
    for pos in args.positions:
        print(f'\n=== pos={pos} ===', flush=True)
        t0 = time.time()
        acc = evaluate_at_position(model, tokenizer, base, pos, args.k,
                                     device, sense_layers)
        results[f'pos_{pos}'] = acc
        print(f'  pos={pos}: acc={acc * 100:.1f}%  '
              f'({time.time() - t0:.0f}s)', flush=True)

    avg = sum(results.values()) / max(1, len(results))
    print(f'\nAVG: {avg * 100:.1f}%', flush=True)
    print(f'\nFor comparison:')
    print(f'  streaming Stage 1 on Qwen 7B k=300 s=42: 57.8%')
    print(f'  upstream SnapKV on Qwen 7B k=150 s=42:   79.8%')

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'diagnostic': 'single-prompt stage 1 (no S2)',
            'model': os.path.basename(args.model_path),
            'k': args.k, 'n_eval': args.n_eval, 'seed': args.seed,
            'positions': args.positions,
            'sense_layers': sense_layers,
            'results': results,
            'average': avg,
            'comparison_anchors': {
                'streaming_stage1_qwen7b_k300_s42': 0.578,
                'upstream_snapkv_qwen7b_k150_s42': 0.798,
            },
        }, f, indent=2)
    print(f'\nSaved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
