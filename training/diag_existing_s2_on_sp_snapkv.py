"""DIAGNOSTIC: drop the EXISTING Stage 2 checkpoint onto a single-prompt
SnapKV-style selector and see if it helps.

If YES (Stage 2 adds +3-5pp on top of single-prompt SnapKV-style): the
existing Stage 2 checkpoint generalises across protocols.  We have a
story without retraining: "Stage 2 is a drop-in summary that complements
any heavy-hitter selector at single-prompt OR streaming protocols."

If NO (Stage 2 is neutral or hurts): the existing Stage 2 is co-adapted
to its streaming-protocol Stage 1.  Retraining Stage 2 for single-prompt
SnapKV-style is then well-motivated.

This is a quick test (~15 min on Qwen 7B, n=100, 4 positions).  Eats one
sweep cell's worth of GPU time on cuda:0.

Methodology (no BS):
  * Single-prompt prefill of the whole context (positions coherent under RoPE).
  * SnapKV-style score = max-pool kernel-7 of the END-OF-PROMPT attention
    window onto the cached context.
  * Reserve 4 sinks + 20% recent strip; top-k the middle.
  * For the +S2 condition: replace 16 of the 284 selected real tokens with
    the existing Stage 2 module's virtual KV pairs (matches the budget
    accounting used in tab:squad_main).
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
from training.train_hybrid import AstroHybrid


def _find(text, ans):
    return ans.strip().lower() in text.strip().lower()


def evaluate_at_position(model, tokenizer, samples, position, k_total,
                          device, sense_layers, astro, n_mem, max_tokens=20):
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim',
                 model.config.hidden_size // nq)
    qpk = nq // nkv

    placed = shuffle_fact_position(samples, position)
    correct_s1, correct_s1s2 = 0, 0
    n_sink = 4
    obs_window = 32  # SnapKV's end-of-prompt observation window
    kernel = 7
    half_k = kernel // 2

    for si, s in enumerate(placed):
        context = '\n\n'.join(s.windows[:-1])
        query_template = s.windows[-1]

        ctx_ids = tokenizer(context, return_tensors='pt',
                              max_length=8192, truncation=True).to(device)
        ctx_len = ctx_ids['input_ids'].shape[1]

        # Single-prompt prefill on context only.
        with torch.no_grad():
            out = model(input_ids=ctx_ids['input_ids'],
                        use_cache=True,
                        output_attentions=True)
        kvs = out.past_key_values
        attns = out.attentions

        # SnapKV-style scoring: end-of-prompt observation window (last
        # `obs_window` tokens) attends back to ALL ctx positions, score
        # is the max-pooled (kernel=7) sum across layers/heads.
        score = torch.zeros(ctx_len, device=device)
        for li in sense_layers:
            a = attns[li][0]  # (n_q_heads, ctx_len, ctx_len)
            if a.isnan().any(): continue
            # rows = query positions, cols = key positions.  Take the
            # last `obs_window` rows -> attention from end-of-prompt
            # tokens onto the rest of the context.
            obs = a[:, -obs_window:, :].sum(dim=(0, 1))  # (ctx_len,)
            score += obs.to(device)
        # Max-pool kernel=7 (matches published SnapKV).
        score_pad = F.pad(score.view(1, 1, -1).float(), (half_k, half_k),
                            mode='constant', value=-1e9)
        score = F.max_pool1d(score_pad, kernel_size=kernel,
                              stride=1).squeeze()
        score[:n_sink] = -1e9
        n_recent = min(int(k_total * 0.2), ctx_len - n_sink)
        recent_start = ctx_len - n_recent
        recent_idx = torch.arange(recent_start, ctx_len, device=device)
        score[recent_start:ctx_len] = -1e9

        for cond, want_s2 in [('s1', False), ('s1s2', True)]:
            k_real = k_total - n_mem if want_s2 else k_total
            n_top = max(k_real - n_sink - n_recent, 0)
            n_avail = (score > -1e8).sum().item()
            n_top = min(n_top, n_avail)
            if n_top > 0:
                _, top = score.topk(n_top)
            else:
                top = torch.empty(0, dtype=torch.long, device=device)
            sink_idx = torch.arange(n_sink, device=device)
            idx = torch.cat([sink_idx, top, recent_idx]).unique().sort().values
            idx = idx[:k_real]

            # Build cache: optional virtual KV from astro + selected real KV.
            cache = DynamicCache()
            if want_s2:
                # Drive astro's EMA state with the unified single-prompt
                # hidden state at the sense layer.  Same forward as
                # update_state(); we approximate "EMA over windows" with
                # "EMA of equal-sized chunks of the single prefill" to
                # match the existing module without retraining.
                astro.reset_state()
                sense_h = model.config.hidden_size  # unused, kept for clarity
                # Use the END-of-prompt hidden state as the summary input.
                # (Existing Stage 2 was trained on per-window hidden states;
                # using the whole prompt's last layer is an approximation.)
                with torch.no_grad():
                    h_prompt = model(input_ids=ctx_ids['input_ids'],
                                       output_hidden_states=True
                                       ).hidden_states[sense_layers[1]]
                    # Chunk into ~5 fake "windows" of equal length.
                    chunk = max(1, h_prompt.shape[1] // 5)
                    for ci in range(5):
                        start = ci * chunk
                        end = min((ci + 1) * chunk, h_prompt.shape[1])
                        if start >= end: break
                        sensed = astro.sense(h_prompt[:, start:end, :])
                        astro.update_state(sensed, keep_grad=False)
            for li in range(nl):
                K_sel = kvs[li][0][:, :, idx, :]
                V_sel = kvs[li][1][:, :, idx, :]
                if want_s2:
                    K_mem, V_mem = astro.generate_kv(li, (K_sel, V_sel))
                    K_mem = K_mem.to(K_sel.device); V_mem = V_mem.to(V_sel.device)
                    K_cat = torch.cat([K_mem, K_sel], dim=2)
                    V_cat = torch.cat([V_mem, V_sel], dim=2)
                else:
                    K_cat, V_cat = K_sel, V_sel
                cache.update(K_cat, V_cat, li)
            cache_len = idx.shape[0] + (n_mem if want_s2 else 0)

            # Manual greedy decode.
            q_full = tokenizer(query_template,
                                  return_tensors='pt', max_length=256,
                                  truncation=True).to(device)
            q_ids = q_full['input_ids']
            with torch.no_grad():
                pos = torch.arange(cache_len, cache_len + q_ids.shape[1],
                                    device=device).unsqueeze(0)
                o = model(input_ids=q_ids,
                           past_key_values=cache,
                           position_ids=pos, use_cache=True)
                running = o.past_key_values
                next_pos = cache_len + q_ids.shape[1]
                gen = []
                for _ in range(max_tokens):
                    nid = o.logits[0, -1].argmax().item()
                    if nid == tokenizer.eos_token_id: break
                    gen.append(nid)
                    nin = torch.tensor([[nid]], device=device, dtype=torch.long)
                    npos = torch.tensor([[next_pos]], device=device,
                                          dtype=torch.long)
                    o = model(input_ids=nin, past_key_values=running,
                               position_ids=npos, use_cache=True)
                    running = o.past_key_values
                    next_pos += 1
                text = tokenizer.decode(gen, skip_special_tokens=True)
            if _find(text, s.answer):
                if cond == 's1': correct_s1 += 1
                else: correct_s1s2 += 1
        if (si + 1) % 25 == 0:
            print(f'  pos={position} {si+1}/{len(placed)}: '
                   f's1={correct_s1}/{si+1}  s1+s2={correct_s1s2}/{si+1}',
                  flush=True)
    return (correct_s1 / max(1, len(placed)),
            correct_s1s2 / max(1, len(placed)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=256)
    p.add_argument('--n_eval', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

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
    inject_layers = sense_layers

    astro = AstroHybrid(
        hidden_dim=model.config.hidden_size, n_mem_tokens=args.n_mem,
        attn_dim=args.attn_dim,
        n_kv_heads=model.config.num_key_value_heads,
        head_dim=getattr(model.config, 'head_dim',
                          model.config.hidden_size // model.config.num_attention_heads),
        n_layers=nl, inject_layers=inject_layers).to(device)
    astro.extract_model_weights(model, device)
    raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
    astro.load_state_dict(raw, strict=False)
    astro.eval()
    print(f'[diag] loaded astro (n_mem={args.n_mem}, attn_dim={args.attn_dim})',
          flush=True)

    base = generate_squad_dataset(
        n_samples=args.n_eval, n_windows=5, vary_distance=True,
        seed=args.seed, split='validation')

    results = {}
    for pos in args.positions:
        print(f'\n=== pos={pos} ===', flush=True)
        t0 = time.time()
        s1, s1s2 = evaluate_at_position(model, tokenizer, base, pos,
                                          args.k, device, sense_layers,
                                          astro, args.n_mem)
        results[f'pos_{pos}'] = {'s1': s1, 's1s2': s1s2, 'delta': s1s2 - s1}
        print(f'  pos={pos}: s1={s1*100:.1f}%  s1+s2={s1s2*100:.1f}%  '
               f'delta={(s1s2-s1)*100:+.1f}pp  ({time.time()-t0:.0f}s)',
              flush=True)
    avg_s1 = sum(r['s1'] for r in results.values()) / len(results)
    avg_s1s2 = sum(r['s1s2'] for r in results.values()) / len(results)
    print(f'\nAVG s1={avg_s1*100:.1f}%  s1+s2={avg_s1s2*100:.1f}%  '
          f'delta={(avg_s1s2-avg_s1)*100:+.1f}pp', flush=True)
    print(f'\nFor comparison:')
    print(f'  streaming Stage 1+2 on Qwen 7B (paper): 66.8%')
    print(f'  upstream SnapKV on Qwen 7B k=150 s=42:  79.8%')

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'diagnostic': 'existing Stage 2 on single-prompt SnapKV-style scoring',
            'model': os.path.basename(args.model_path),
            'k': args.k, 'n_mem': args.n_mem, 'n_eval': args.n_eval,
            'seed': args.seed,
            'results': results,
            'average': {'s1': avg_s1, 's1s2': avg_s1s2,
                        'delta': avg_s1s2 - avg_s1},
        }, f, indent=2)
    print(f'\nSaved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
