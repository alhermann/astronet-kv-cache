"""Experiment 3 (load-bearing) — S2 transfer test.

Take a trained AstroNet hybrid checkpoint (S1+S2 jointly trained).  At
inference, the S1 selector that picks 284 real KV indices is REPLACED
with a different scoring rule (faithful SnapKV / H2O / no-selector).
S2 still emits its 16 learned tokens.

The question this experiment answers:
  - If we swap S1 for SnapKV's selection rule and the +5 pt S2 boost
    SURVIVES on multiple backbones, then S2 is a transferable additive
    augmenter — the headline reframe holds.
  - If the boost EVAPORATES, S2 was co-adapted to S1's selection
    statistics and does not transfer.  We then must scope the paper
    claim to "co-trained S1+S2 system" (Fallback B, task #136).

Protocol mirrors ``train_hybrid.evaluate`` (same windows-fresh + concat
+ separate question forward + question-conditioned scoring) except for
the per-method scoring rule.  All four conditions in one run for
direct comparison on the same samples:
  pure_S1        = AstroNet S1 selection alone (no S2)
  pure_swap      = swapped selector alone (no S2)
  hybrid_S1      = AstroNet S1 selection + S2 16 tokens
  hybrid_swap    = swapped selector + S2 16 tokens

Δ_S1   = hybrid_S1   - pure_S1
Δ_swap = hybrid_swap - pure_swap

If Δ_swap ≈ Δ_S1 (within seed noise) across backbones → S2 transfers.

Usage::

    python training/eval_hybrid_swap_selector.py \\
        --model_path ./models/qwen2.5-7b \\
        --checkpoint checkpoints/astro_hybrid_qwen2_5-7b_n16_k284_t5000_s42.pt \\
        --selector snapkv --n_eval 100 --seed 42 \\
        --positions 0 1 2 3 \\
        --save_path logs/results/e3_swap_snapkv_qwen7b_s42.json
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, '/home/alexander/Schreibtisch/AstroNet')

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from training.train_hybrid import AstroHybrid, _select_kv
from training.eval_hybrid_position_robust import shuffle_fact_position
from data.real_qa import generate_squad_dataset


# ---- selector rules -----------------------------------------------------

def select_astronet(cross, total_len, k, last_window_size, n_sink=4):
    """AstroNet multiplicative S1: avg-pool kernel=5 (applied in caller),
    recent = first 20% of last fact window, mask the entire last fact
    window from heavy-hit scoring."""
    last_start = total_len - last_window_size
    n_recent = min(int(k * 0.2), last_window_size)
    recent = torch.arange(last_start, last_start + n_recent,
                            device=cross.device)
    scores = cross.clone()
    scores[:n_sink] = -1e9
    scores[last_start:total_len] = -1e9
    n_select = max(k - n_sink - len(recent), 0)
    n_avail = (scores > -1e8).sum().item()
    n_select = min(n_select, n_avail)
    _, top = scores.topk(n_select) if n_select > 0 else (
        None, torch.empty(0, dtype=torch.long, device=cross.device))
    sink = torch.arange(n_sink, device=cross.device)
    idx = torch.cat([sink, top, recent]).unique().sort().values
    return idx[:k]


def select_snapkv(cross, total_len, k, last_window_size, n_sink=4):
    """Faithful SnapKV: max-pool kernel=7 (applied in caller; pooling
    differs from AstroNet's avg-pool), recent = last 32 of cumulative
    (NOT the last fact window), no last-window mask, sink reserve."""
    window_size = 32
    recent_start = total_len - window_size
    recent = torch.arange(recent_start, total_len, device=cross.device)
    scores = cross.clone()
    scores[:n_sink] = -1e9
    # SnapKV also excludes the recent strip from heavy-hit competition
    # (otherwise the last 32 tokens would be picked twice).
    scores[recent_start:total_len] = -1e9
    n_select = max(k - n_sink - window_size, 0)
    n_avail = (scores > -1e8).sum().item()
    n_select = min(n_select, n_avail)
    _, top = scores.topk(n_select) if n_select > 0 else (
        None, torch.empty(0, dtype=torch.long, device=cross.device))
    sink = torch.arange(n_sink, device=cross.device)
    idx = torch.cat([sink, top, recent]).unique().sort().values
    return idx[:k]


def select_h2o(cross, heur, total_len, k, last_window_size, n_sink=4):
    """H2O: cumulative-attention heuristic (heur) + last-window-recent
    strip. heur is collected from per-window prefill attention sums
    (pre-question-conditioned), recent = last fact window 30% of k."""
    last_start = total_len - last_window_size
    n_recent = int(k * 0.3)
    n_recent = min(n_recent, last_window_size)
    recent = torch.arange(last_start, last_start + n_recent,
                            device=heur.device)
    scores = heur.clone()
    scores[:n_sink] = -1e9
    scores[last_start:total_len] = -1e9
    n_select = max(k - n_sink - len(recent), 0)
    n_avail = (scores > -1e8).sum().item()
    n_select = min(n_select, n_avail)
    _, top = scores.topk(n_select) if n_select > 0 else (
        None, torch.empty(0, dtype=torch.long, device=heur.device))
    sink = torch.arange(n_sink, device=heur.device)
    idx = torch.cat([sink, top, recent]).unique().sort().values
    return idx[:k]


# ---- evaluation core ----------------------------------------------------

def make_pool(method: str):
    """Returns the per-method pooling function applied to cross before
    selection.  Identity for h2o (which uses heur directly)."""
    if method == 'astronet':
        return lambda c: F.avg_pool1d(c.unsqueeze(0).unsqueeze(0),
                                         kernel_size=5, padding=2,
                                         stride=1).squeeze()
    if method == 'snapkv':
        return lambda c: F.max_pool1d(c.unsqueeze(0).unsqueeze(0),
                                         kernel_size=7, padding=3,
                                         stride=1).squeeze()
    return lambda c: c   # h2o: no pooling on cross (uses heur)


@torch.no_grad()
def evaluate_position(model, tokenizer, astro, samples, pos, selector,
                       device, k=300, n_mem=16, max_gen=20):
    """Evaluate four conditions at a single fact-position: pure_S1,
    pure_swap, hybrid_S1, hybrid_swap.  Each sample is run THROUGH FOUR
    decodes so we share the cumulative KV + cross score computation."""
    inject = astro.inject_layers
    cfg = model.config
    nl = cfg.num_hidden_layers
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // nq)
    qpk = nq // nkv
    # Use nl // 2 to match training/eval_hybrid_position_robust.py
    # (the script that produced the published S1/hybrid numbers).
    sense_layer = getattr(astro, 'sense_layer', nl // 2)

    placed = shuffle_fact_position(samples, pos)
    correct = {'pure_S1': 0, 'pure_swap': 0, 'hybrid_S1': 0,
                'hybrid_swap': 0}
    using_x1 = getattr(astro, '_x1', None) is not None

    for si, s in enumerate(placed):
        astro.reset_state()
        all_kv = {li: ([], []) for li in range(nl)}
        all_hidden_inject = {li: [] for li in inject} if using_x1 else None
        all_attn = []
        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt',
                             max_length=384, truncation=True).to(device)
            out = model(input_ids=ids['input_ids'], use_cache=True,
                          output_attentions=True, output_hidden_states=True)
            sl = ids['input_ids'].shape[1]
            imp = torch.zeros(sl, device=device)
            for la in out.attentions:
                a = la[0].to(device)
                if not a.isnan().any():
                    imp += a.sum(dim=(0, 1))
            imp[:4] = -1e9
            all_attn.append(imp)
            for li in range(nl):
                all_kv[li][0].append(out.past_key_values[li][0])
                all_kv[li][1].append(out.past_key_values[li][1])
            if using_x1:
                for li in inject:
                    all_hidden_inject[li].append(out.hidden_states[li][0])
            hidden = out.hidden_states[sense_layer]
            sensed = astro.sense(hidden)
            astro.update_state(sensed)

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn)

        # Question-conditioned cross score
        q_ids = tokenizer(f'Question: {s.question}\nAnswer:',
                            return_tensors='pt', max_length=128,
                            truncation=True).to(device)
        q_out = model(input_ids=q_ids['input_ids'],
                       output_hidden_states=True)
        cross = torch.zeros(total, device=device)
        for li in inject:
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float()).half().view(-1, nq, hd)
            K_full = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                    K_full[hi].float().T) / math.sqrt(hd)
                attn_w = torch.softmax(sc, dim=-1)
                cross += attn_w.sum(dim=(0, 1)).to(device)

        last_window_size = all_kv[inject[0]][0][-1].shape[2]
        # Pool per method
        cross_S1 = make_pool('astronet')(cross) if total > 5 else cross
        cross_S1[:4] = -1e9
        if selector == 'snapkv':
            cross_swap = make_pool('snapkv')(cross) if total > 5 else cross
            cross_swap[:4] = -1e9
        else:
            cross_swap = cross   # h2o uses heur

        # Select 284 real-KV indices for each rule
        k_real = k - n_mem
        idx_S1 = select_astronet(cross_S1, total, k_real,
                                    last_window_size)
        if selector == 'snapkv':
            idx_swap = select_snapkv(cross_swap, total, k_real,
                                        last_window_size)
        else:
            idx_swap = select_h2o(cross_swap, heur, total, k_real,
                                    last_window_size)

        # Also a pure_S1 at full k (no S2)
        idx_S1_pure = select_astronet(cross_S1, total, k,
                                         last_window_size)
        idx_swap_pure = (select_snapkv(cross_swap, total, k, last_window_size)
                           if selector == 'snapkv'
                           else select_h2o(cross_swap, heur, total, k,
                                            last_window_size))

        # Decode each of the four conditions
        for cond, idx_real, attach_S2 in [
            ('pure_S1', idx_S1_pure, False),
            ('pure_swap', idx_swap_pure, False),
            ('hybrid_S1', idx_S1, True),
            ('hybrid_swap', idx_swap, True),
        ]:
            # X1: stash per-inject-layer selected hidden states for the
            # cross-attn gap-fill module to consume during generate_kv.
            if attach_S2 and using_x1:
                for li in inject:
                    full_h = torch.cat(all_hidden_inject[li], dim=0)
                    h_sel = full_h.index_select(0, idx_real).unsqueeze(0)
                    astro.set_selected_hidden(li, h_sel)
            cache = DynamicCache()
            for li in range(nl):
                K_real, V_real = _select_kv(all_kv, li, idx_real)
                if attach_S2:
                    K_mem, V_mem = astro.generate_kv(li, (K_real, V_real))
                    K_mem, V_mem = K_mem.to(K_real.device), V_mem.to(V_real.device)
                    K_combined = torch.cat([K_mem, K_real], dim=2)
                    V_combined = torch.cat([V_mem, V_real], dim=2)
                else:
                    K_combined, V_combined = K_real, V_real
                cache.update(K_combined, V_combined, li)
            total_len = (n_mem if attach_S2 else 0) + idx_real.shape[0]
            query = (f'Based on what you read earlier, answer the '
                       f'following question.\nQuestion: {s.question}\n'
                       f'Answer:')
            fq = tokenizer(query, return_tensors='pt', max_length=384,
                              truncation=True).to(device)
            pos_t = torch.arange(total_len,
                                    total_len + fq['input_ids'].shape[1],
                                    device=device).unsqueeze(0)
            cur = fq['input_ids']; cc = cache; gen = []
            for _ in range(max_gen):
                o = model(input_ids=cur, past_key_values=cc,
                              position_ids=pos_t)
                cc = o.past_key_values
                nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                gen.append(nxt[0, 0].item()); cur = nxt
                pos_t = torch.tensor(
                    [[total_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                    device=device)
                if nxt[0, 0].item() == tokenizer.eos_token_id: break
            ans = tokenizer.decode(gen, skip_special_tokens=True).strip()
            if s.answer.lower() in ans.lower():
                correct[cond] += 1

        if (si + 1) % 25 == 0:
            line = ' '.join(f'{c}={correct[c]}/{si+1}' for c in correct)
            print(f'  pos={pos} {si+1}/{len(placed)}  {line}', flush=True)

    return {c: correct[c] / max(1, len(placed)) for c in correct}


# ---- main ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--selector', required=True,
                   choices=['snapkv', 'h2o'])
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    print(f'[E3 swap] model={args.model_path}  ckpt={args.checkpoint}  '
          f'selector={args.selector}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    # NB: do NOT pass attn_implementation='eager' here.  Empirically that
    # path produces materially different (much lower) accuracy than the
    # default SDPA-with-manual-fallback that the original training and
    # evaluation used (May 30 / Jun 1 logs).  The fallback is invoked
    # whenever output_attentions=True (which our evaluate() needs for the
    # heuristic).  Reproducing the published checkpoint numbers requires
    # this default; explicit eager mode silently breaks the eval.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()

    cfg = model.config
    nl = cfg.num_hidden_layers
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    astro = AstroHybrid(hidden_dim=cfg.hidden_size, n_mem_tokens=args.n_mem,
                          n_kv_heads=nkv, head_dim=hd, n_layers=nl,
                          inject_layers=inject, model=model).to(args.device)
    sd = torch.load(args.checkpoint, map_location=args.device,
                      weights_only=False)
    # Support both legacy ({'<param_name>': tensor}) and X1-wrapped
    # ({'astro': sd, 'x1': sd, 'args': dict}) checkpoint formats.
    if isinstance(sd, dict) and 'astro' in sd and 'x1' in sd:
        astro.load_state_dict(sd['astro'], strict=False)
        from astronet.x1_gapfill import GapFillPerInjectLayer
        x1_attn_dim = sd.get('args', {}).get('x1_attn_dim', 256)
        x1 = GapFillPerInjectLayer(inject, hidden_dim=cfg.hidden_size,
                                       attn_dim=x1_attn_dim).to(args.device)
        x1.load_state_dict(sd['x1'])
        astro.attach_x1(x1)
        print(f'  loaded X1-wrapped checkpoint: '
               f'astro+X1 ({sum(p.numel() for p in x1.parameters()):,} '
               f'X1 params)', flush=True)
    else:
        astro.load_state_dict(sd, strict=False)
    astro.extract_model_weights(model, args.device)
    astro.eval()

    samples = generate_squad_dataset(
        n_samples=args.n_eval, n_windows=5, vary_distance=True,
        seed=args.seed, split='validation')

    all_results = {}
    for pos in args.positions:
        print(f'\n=== pos={pos} ===', flush=True)
        t0 = time.time()
        accs = evaluate_position(model, tokenizer, astro, samples, pos,
                                    args.selector, args.device,
                                    args.k, args.n_mem)
        all_results[f'pos_{pos}'] = accs
        line = '  '.join(f'{c}={v*100:.1f}%' for c, v in accs.items())
        print(f'  pos={pos}: {line}  ({time.time() - t0:.0f}s)', flush=True)

    averages = {c: sum(all_results[f'pos_{p}'][c]
                          for p in args.positions) / len(args.positions)
                 for c in ['pure_S1', 'pure_swap', 'hybrid_S1', 'hybrid_swap']}
    delta_S1 = averages['hybrid_S1'] - averages['pure_S1']
    delta_swap = averages['hybrid_swap'] - averages['pure_swap']
    print(f'\nAverages: pure_S1={averages["pure_S1"]*100:.1f}%  '
          f'pure_swap={averages["pure_swap"]*100:.1f}%  '
          f'hybrid_S1={averages["hybrid_S1"]*100:.1f}%  '
          f'hybrid_swap={averages["hybrid_swap"]*100:.1f}%')
    print(f'Δ_S1   (S2 boost on AstroNet selector)   = {delta_S1*100:+.1f}pp')
    print(f'Δ_swap (S2 boost on {args.selector} selector) = '
           f'{delta_swap*100:+.1f}pp')

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'checkpoint': args.checkpoint, 'selector': args.selector,
            'k': args.k, 'n_mem': args.n_mem, 'n_eval': args.n_eval,
            'seed': args.seed, 'positions': args.positions,
            'results': all_results, 'averages': averages,
            'delta_S1': delta_S1, 'delta_swap': delta_swap,
            'verdict': ('S2 transfers' if abs(delta_S1 - delta_swap) < 0.03
                          else 'S2 selector-specific'),
        }, f, indent=2)
    print(f'Saved -> {args.save_path}')


if __name__ == '__main__':
    main()
