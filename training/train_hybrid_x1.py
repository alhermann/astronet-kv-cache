"""Train an AstroHybrid with the X1 augmentation — S2 cross-attends to
the hidden states of the S1-selected tokens at each inject layer.

Compared to ``train_hybrid.py`` this adds, per sample:
  * collection of hidden states at the four inject layers per window
    (not just sense_layer)
  * after S1 selects 284 indices, slice those hidden states per layer
    and stash them on the astro module via ``set_selected_hidden``
  * the rest of the forward + loss is unchanged

Initialise the X1 module with zero output projection so the freshly
constructed AstroHybrid+X1 reproduces the baseline AstroHybrid
behaviour at step 0.  Lift only appears after gradient updates push
the gap-fill projection away from zero.

Usage::

    python training/train_hybrid_x1.py \\
        --model_path ./models/qwen2.5-7b \\
        --init_checkpoint checkpoints/astro_hybrid_qwen2_5-7b_..._w10_diverse.pt \\
        --n_train 5000 --n_eval 100 --epochs 2 \\
        --save_path checkpoints/astro_x1_qwen2_5-7b.pt
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache,
)
from training.train_hybrid import AstroHybrid, _select_kv
from astronet.x1_gapfill import GapFillPerInjectLayer
from data.real_qa import generate_squad_dataset


def train_step_x1(model, tokenizer, astro, optimizer, s,
                   inject_layers, sense_layer, k_real, device):
    """One sample's training step with X1 plumbing."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv

    astro.reset_state()
    # all_kv[li] = ([K_per_window], [V_per_window]) along seq dim.
    # all_hidden[li] = [H_per_window] for inject layers only (memory cost).
    all_kv = {li: ([], []) for li in range(nl)}
    all_hidden_inject = {li: [] for li in inject_layers}

    # --- 1) Process fact windows + collect hidden states per inject layer
    for wi in range(len(s.windows) - 1):
        ids = tokenizer(s.windows[wi], return_tensors='pt',
                         max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True,
                          output_hidden_states=True)
        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0].detach())
            all_kv[li][1].append(out.past_key_values[li][1].detach())
        for li in inject_layers:
            all_hidden_inject[li].append(
                out.hidden_states[li][0].detach())   # (sl, hidden)
        hidden = out.hidden_states[sense_layer].detach()
        sensed = astro.sense(hidden)
        n_grad_windows = max(2, len(s.windows) // 2)
        astro.update_state(
            sensed, keep_grad=(wi >= len(s.windows) - 1 - n_grad_windows))

    total = sum(h.shape[0] for h in all_hidden_inject[inject_layers[0]])

    # --- 2) Cross-attention scoring against question Q
    q_text = f'Question: {s.question}\nAnswer:'
    q_ids = tokenizer(q_text, return_tensors='pt', max_length=128,
                        truncation=True).to(device)
    with torch.no_grad():
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
    cross = torch.zeros(total, device=device)
    for li in inject_layers:
        Q = model.model.layers[li].self_attn.q_proj(
            q_out.hidden_states[li][0].float()).half().view(-1, nq, hd)
        K = torch.cat(all_kv[li][0], dim=2)[0]
        for hi in range(nkv):
            sc = torch.matmul(
                Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                K[hi].float().T) / math.sqrt(hd)
            attn_w = torch.softmax(sc, dim=-1)
            cross += attn_w.sum(dim=(0, 1)).to(device)
    if total > 5:
        cross = F.avg_pool1d(cross.unsqueeze(0).unsqueeze(0),
                                 kernel_size=5, padding=2,
                                 stride=1).squeeze()
    cross[:4] = -1e9

    # --- 3) S1 selection
    n_sink = 4
    last_window_size = all_kv[inject_layers[0]][0][-1].shape[2]
    last_start = total - last_window_size
    n_recent = min(int(k_real * 0.2), last_window_size)
    recent = torch.arange(last_start, last_start + n_recent, device=device)
    scores = cross.clone()
    scores[last_start:total] = -1e9
    n_select = max(k_real - n_sink - len(recent), 0)
    n_avail = (scores > -1e8).sum().item()
    n_select = min(n_select, n_avail, scores.shape[0])
    _, top = scores.topk(n_select) if n_select > 0 else (
        None, torch.empty(0, dtype=torch.long, device=device))
    sink_idx = torch.arange(n_sink, device=device)
    idx = torch.cat([sink_idx, top, recent]).unique().sort().values
    idx = idx[:k_real]

    # --- 4) Stash per-layer selected hidden states for X1
    for li in inject_layers:
        full_h = torch.cat(all_hidden_inject[li], dim=0)   # (total, hidden)
        h_sel = full_h.index_select(0, idx).unsqueeze(0)    # (1, k_real, hidden)
        astro.set_selected_hidden(li, h_sel)

    # --- 5) Build hybrid cache + teacher forcing forward
    n_mem = astro.n_mem_tokens
    cache = DynamicCache()
    for li in range(nl):
        K_real, V_real = _select_kv(all_kv, li, idx)
        K_mem, V_mem = astro.generate_kv(li, (K_real, V_real))
        K_mem, V_mem = K_mem.to(K_real.device), V_mem.to(V_real.device)
        cache.update(torch.cat([K_mem, K_real], dim=2),
                       torch.cat([V_mem, V_real], dim=2), li)

    query = (f'Based on what you read earlier, answer the following '
             f'question.\nQuestion: {s.question}\nAnswer: {s.answer}')
    fq = tokenizer(query, return_tensors='pt', max_length=384,
                    truncation=True).to(device)
    input_ids = fq['input_ids']
    total_cache_len = n_mem + len(idx)
    pos = torch.arange(total_cache_len, total_cache_len + input_ids.shape[1],
                         device=device).unsqueeze(0)

    out = model(input_ids=input_ids, past_key_values=cache, position_ids=pos)
    logits = out.logits

    # NLL on the answer span only
    # Find token positions of " Answer: " marker — loss only on what follows.
    full_ids = input_ids[0]
    # Locate ' Answer:' (or 'Answer:') in the tokenised query.
    marker = tokenizer(' Answer:', add_special_tokens=False).input_ids
    if not marker:
        marker = tokenizer('Answer:', add_special_tokens=False).input_ids
    ans_start = None
    for i in range(full_ids.shape[0] - len(marker) + 1):
        if torch.equal(full_ids[i:i+len(marker)],
                        torch.tensor(marker, device=device)):
            ans_start = i + len(marker)
            break
    if ans_start is None:
        ans_start = full_ids.shape[0] - 5

    target = full_ids[ans_start:]
    pred = logits[0, ans_start - 1:full_ids.shape[0] - 1]
    loss = F.cross_entropy(pred, target)
    return loss


def train_epoch_x1(model, tokenizer, astro, optimizer, samples,
                    inject_layers, sense_layer, k_real, device,
                    accum_steps=4, max_grad_norm=1.0):
    astro.train()
    total_loss = 0.0; n_total = 0
    optimizer.zero_grad()
    for si, s in enumerate(samples):
        try:
            loss = train_step_x1(model, tokenizer, astro, optimizer, s,
                                  inject_layers, sense_layer, k_real, device)
        except Exception as e:
            print(f'  [warn] sample {si} failed: {e}', flush=True)
            continue
        (loss / accum_steps).backward()
        total_loss += loss.item(); n_total += 1
        if (si + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in astro.parameters() if p.requires_grad],
                max_grad_norm)
            optimizer.step(); optimizer.zero_grad()
        if (si + 1) % 50 == 0:
            print(f'  train {si+1}/{len(samples)}: '
                   f'mean loss = {total_loss/n_total:.3f}', flush=True)
    if n_total > 0:
        optimizer.step(); optimizer.zero_grad()
    return total_loss / max(1, n_total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--init_checkpoint', required=True,
                   help='Baseline AstroHybrid checkpoint to initialise from')
    p.add_argument('--n_train', type=int, default=5000)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--k_real', type=int, default=284)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=256)
    p.add_argument('--x1_attn_dim', type=int, default=256)
    p.add_argument('--n_windows', type=int, default=10)
    p.add_argument('--accum_steps', type=int, default=4)
    p.add_argument('--data', default='squad',
                   choices=['squad', 'multihop'],
                   help='squad = pure SQuAD; multihop = X2 mix '
                         '(40% SQuAD / 40% HotpotQA / 20% 2WikiMQA)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    print(f'[x1-train] model={args.model_path}  init={args.init_checkpoint}',
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()

    cfg = model.config
    nl = cfg.num_hidden_layers
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    inject_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    sense_layer = nl // 2

    # 1) Baseline AstroHybrid loaded from existing checkpoint
    astro = AstroHybrid(
        hidden_dim=cfg.hidden_size, n_mem_tokens=args.n_mem,
        attn_dim=args.attn_dim, n_kv_heads=nkv, head_dim=hd,
        n_layers=nl, inject_layers=inject_layers).to(args.device)
    sd = torch.load(args.init_checkpoint, map_location=args.device,
                      weights_only=False)
    astro.load_state_dict(sd, strict=True)
    astro.extract_model_weights(model, args.device)

    # 2) Attach X1 gap-fill, identity at init
    x1 = GapFillPerInjectLayer(inject_layers, hidden_dim=cfg.hidden_size,
                                  attn_dim=args.x1_attn_dim).to(args.device)
    astro.attach_x1(x1)
    print(f'  AstroHybrid params: {astro.parameter_count():,}', flush=True)
    print(f'  X1 params         : {x1.parameter_count():,}', flush=True)

    # 3) Optimiser over AstroHybrid + X1 params
    train_params = list(astro.parameters()) + list(x1.parameters())
    optimizer = torch.optim.AdamW(train_params, lr=args.lr,
                                     weight_decay=0.01)

    # 4) Data
    if args.data == 'squad':
        train_samples = generate_squad_dataset(
            n_samples=args.n_train, n_windows=args.n_windows,
            vary_distance=True, seed=42, split='train')
    else:
        from data.multihop_mix import multihop_mix_samples
        train_samples = multihop_mix_samples(
            n_samples=args.n_train, n_windows=args.n_windows,
            seed=42, split='train')
        print(f'[x1-train] multi-hop mix produced '
               f'{len(train_samples)} samples', flush=True)

    # 5) Training loop
    for epoch in range(args.epochs):
        t0 = time.time()
        mean_loss = train_epoch_x1(
            model, tokenizer, astro, optimizer, train_samples,
            inject_layers, sense_layer, args.k_real, args.device,
            accum_steps=args.accum_steps)
        print(f'epoch {epoch+1}/{args.epochs}: '
               f'mean loss = {mean_loss:.4f}  '
               f'({time.time()-t0:.0f}s)', flush=True)
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        # Save BOTH astro and x1 state dicts in one file
        torch.save({
            'astro': astro.state_dict(),
            'x1': x1.state_dict(),
            'args': vars(args),
        }, args.save_path)
        print(f'  saved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
