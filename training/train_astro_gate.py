"""Distill AstroGate from query-aware SnapKV oracle (query-agnostic regime).

For each multi-query SQuAD sample:
  1. Forward ctx (no Q) with output_hidden_states; capture sense-layer hidden.
  2. Astro: sense + EMA-update fast/slow buffers (keep_grad=True so
     pattern_head receives gradient via state).
  3. For each Q (1..max_q): forward Q on ctx_cache with AttentionCapture;
     compute per-layer SnapKV scores (query-aware oracle).
  4. Aggregate (Q, layer) into a per-token relevance target:
        target_i = fraction of (Q, layer) pairs where ctx token i is in top-K.
     Top-K is taken with the same n_sink=4 head used at eval time.
  5. Astro.gate_logits(ctx_hidden) → student logits.
  6. Loss = BCE(sigmoid(gate_logit), target).

Result: astro learns to predict query-aware relevance from ctx alone — a
context-conditioned prior that biases SnapKV's query-agnostic selection at
inference, without consuming any budget slots.

Usage (smoke):
    python training/train_astro_gate.py \
        --model_path ./models/qwen2.5-7b --device cuda:1 \
        --n_train 100 --epochs 1 --k 300 \
        --save_path checkpoints/astro_gate_qwen7b_smoke.pt
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import defaultdict
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from baselines.streaming_kv_core import AttentionCapture
from baselines.eval_faithful_streaming_needle import (
    score_per_layer_snapkv_layerwise, OBS_WINDOW)
from data.real_qa import _load_squad, _filter_squad
from astronet.astro_gate import AstroGate

QUERY_TMPL = ("Based on what you read earlier, answer the following "
              "question.\nQuestion: {q}\nAnswer:")
N_SINK = 4


def build_multiquery_samples_split(split, n_paragraphs, n_ctx_windows, max_q, seed):
    rng = random.Random(seed)
    filtered = _filter_squad(_load_squad(split))
    by_ctx = defaultdict(list)
    for ex in filtered:
        by_ctx[ex['context']].append(ex)
    groups = [(ctx, exs) for ctx, exs in by_ctx.items() if len(exs) >= 2]
    rng.shuffle(groups)
    by_title = defaultdict(list)
    for ex in filtered:
        by_title[ex['title']].append(ex)
    titles = list(by_title.keys())
    samples = []
    for ctx, exs in groups[:n_paragraphs]:
        fact_title = exs[0]['title']
        distractors = []
        attempts = 0
        while len(distractors) < n_ctx_windows - 1 and attempts < 100:
            t = rng.choice(titles)
            if t != fact_title and by_title[t]:
                cand = rng.choice(by_title[t])['context']
                if cand != ctx and cand not in distractors:
                    distractors.append(cand)
            attempts += 1
        windows = list(distractors)
        fact_pos = rng.randrange(n_ctx_windows)
        windows.insert(fact_pos, ctx)
        qa = [(ex['question'], ex['answers']['text'][0]) for ex in exs[:max_q]]
        samples.append({'windows': windows, 'fact_pos': fact_pos, 'qa': qa})
    return samples


def _select_topk_idx(scores, k, n_sink=N_SINK):
    ctx_len = scores.shape[0]
    s = scores.clone()
    s[:n_sink] = -1e9
    n_heavy = max(min(k - n_sink, ctx_len - n_sink), 0)
    if n_heavy > 0:
        _, top = s.topk(n_heavy)
    else:
        top = torch.empty(0, dtype=torch.long, device=scores.device)
    sink = torch.arange(min(n_sink, ctx_len), device=scores.device)
    return torch.cat([sink, top]).unique()


@torch.no_grad()
def compute_oracle_target(model, tokenizer, capture, ctx_cache, ctx_len,
                          question, k, n_layers, device):
    """Per-token relevance from query-aware SnapKV oracle, per layer.
    Returns: target [T_ctx] in [0,1] = fraction of layers where token i is
             in the layer's top-k."""
    q_text = QUERY_TMPL.format(q=question)
    q_ids = tokenizer(q_text, return_tensors='pt',
                      add_special_tokens=False).input_ids.to(device)
    q_pos = torch.arange(ctx_len, ctx_len + q_ids.shape[1],
                          device=device).unsqueeze(0)
    capture.clear()
    capture.enabled = True
    cloned = DynamicCache()
    for li in range(n_layers):
        K, V = ctx_cache[li]
        cloned.update(K.clone(), V.clone(), li)
    out = model(input_ids=q_ids, past_key_values=cloned,
                position_ids=q_pos, use_cache=True)
    full_q = out.past_key_values
    q_len = q_ids.shape[1]
    scores = score_per_layer_snapkv_layerwise(
        capture, full_q, ctx_len, q_len, n_layers)
    counts = torch.zeros(ctx_len, device=device)
    for s in scores:
        idx = _select_topk_idx(s, k)
        counts[idx] += 1.0
    return counts / float(n_layers)


@torch.no_grad()
def compute_snapkv_qa_target(capture, ctx_cache, ctx_len, obs, k, n_layers,
                              device):
    """Per-layer query-agnostic SnapKV selection (the inference-time signal
    we are trying to ADD to). Returns [T_ctx] in [0,1] = fraction of layers
    where token i is in the per-layer top-k under ctx-tail observation."""
    scores = score_per_layer_snapkv_layerwise(
        capture, ctx_cache, ctx_len, obs, n_layers)
    counts = torch.zeros(ctx_len, device=device)
    for s in scores:
        idx = _select_topk_idx(s, k)
        counts[idx] += 1.0
    return counts / float(n_layers)


def train_step(model, tokenizer, capture, astro, optimizer, sample,
               sense_layer, k, n_layers, device):
    astro.train()
    astro.reset_state()
    ctx_text = '\n\n'.join(sample['windows']) + '\n\n'
    ctx_ids = tokenizer(ctx_text, return_tensors='pt',
                        max_length=8192, truncation=True
                        ).input_ids.to(device)
    ctx_len = ctx_ids.shape[1]
    obs = min(OBS_WINDOW, ctx_len - 1)

    capture.clear()
    capture.enabled = False
    with torch.no_grad():
        out1 = model(input_ids=ctx_ids[:, :-obs],
                     past_key_values=DynamicCache(),
                     use_cache=True, output_hidden_states=True)
        h1 = out1.hidden_states[sense_layer].detach()
        # Chunk 2 with capture enabled — this is the query-agnostic obs
        # window AttentionCapture will score against for snapkv_qa.
        capture.enabled = True
        out2 = model(input_ids=ctx_ids[:, -obs:],
                     past_key_values=out1.past_key_values,
                     use_cache=True, output_hidden_states=True)
        h2 = out2.hidden_states[sense_layer].detach()
        ctx_cache = out2.past_key_values
    ctx_hidden = torch.cat([h1, h2], dim=1)             # [1, T_ctx, D]

    sensed = astro.sense(ctx_hidden)
    astro.update_state(sensed, keep_grad=True)

    # Query-agnostic SnapKV target — what selection looks like WITHOUT us.
    snap_qa = compute_snapkv_qa_target(
        capture, ctx_cache, ctx_len, obs, k, n_layers, device)  # [T_ctx]

    targets = []
    for question, _ in sample['qa']:
        # Each oracle pass disables capture mid-run; re-enable for next.
        tgt = compute_oracle_target(model, tokenizer, capture, ctx_cache,
                                     ctx_len, question, k, n_layers, device)
        targets.append(tgt)
    oracle = torch.stack(targets, dim=0).mean(dim=0)     # [T_ctx] in [0,1]

    # CORRECTIVE target: what we want the gate to ADD to snapkv scores.
    # +1 → "oracle picks but query-agnostic snapkv misses, BOOST it"
    # −1 → "snapkv picks but oracle says irrelevant, SUPPRESS it"
    #  0 → "snapkv and oracle agree, no correction needed"
    target = oracle - snap_qa                            # [T_ctx] in [-1,1]

    gate_logit = astro.gate_logits(ctx_hidden)           # [T_ctx]
    # Map logit to [-1, 1] via tanh, MSE against corrective target.
    # Soft tanh (logit/2) gives smooth gradients while still reaching ±1
    # at logit≈±5, so the eval-time additive bias bias∈roughly ±5.
    pred = torch.tanh(gate_logit / 2.0)
    mse = F.mse_loss(pred, target)
    var_pen = F.relu(0.05 - pred.std())
    loss = mse + 0.5 * var_pen

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(astro.parameters(), 1.0)
    optimizer.step()
    return loss.item(), target.abs().mean().item(), gate_logit.detach()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--n_train', type=int, default=2000)
    p.add_argument('--n_ctx_windows', type=int, default=4)
    p.add_argument('--max_q', type=int, default=4)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_astro', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=256)
    p.add_argument('--n_patterns', type=int, default=8)
    p.add_argument('--sense_layer', type=int, default=-1,
                   help='-1 → n_layers // 2')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    print(f'[train-astro-gate] model={args.model_path} n_train={args.n_train} '
          f'epochs={args.epochs} k={args.k}', flush=True)
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
    capture = AttentionCapture(model)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    sense_layer = args.sense_layer if args.sense_layer >= 0 else n_layers // 2

    astro = AstroGate(
        hidden_dim=cfg.hidden_size, n_astro=args.n_astro,
        attn_dim=args.attn_dim, n_patterns=args.n_patterns,
    ).to(device)
    print(f'AstroGate params: {astro.parameter_count():,}  '
          f'sense_layer={sense_layer}', flush=True)

    samples = build_multiquery_samples_split(
        'train', args.n_train, args.n_ctx_windows, args.max_q, args.seed)
    n_q = sum(len(s['qa']) for s in samples)
    print(f'{len(samples)} train paragraphs ({n_q} questions)', flush=True)

    optimizer = torch.optim.AdamW(astro.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    t0 = time.time()
    step = 0
    losses = []
    for ep in range(args.epochs):
        print(f'\n=== epoch {ep+1}/{args.epochs} ===', flush=True)
        random.Random(args.seed + ep).shuffle(samples)
        for si, s in enumerate(samples):
            try:
                loss, tgt_sum, glog = train_step(
                    model, tokenizer, capture, astro, optimizer, s,
                    sense_layer, args.k, n_layers, device)
                losses.append(loss)
                step += 1
            except Exception as e:
                print(f'  paragraph {si} ERROR: {type(e).__name__}: {str(e)[:120]}',
                      flush=True)
                continue
            if step <= 3 or step % 25 == 0:
                recent = sum(losses[-25:]) / max(len(losses[-25:]), 1)
                print(f'  step {step:5d}: loss={recent:.4f}  '
                      f'λ={astro.lam.item():.3f}  '
                      f'αf={astro.alpha_fast.item():.3f}  '
                      f'αs={astro.alpha_slow.item():.3f}  '
                      f'|tgt|={tgt_sum:.3f}  '
                      f'gate_mean={glog.mean().item():.3f}  '
                      f'gate_std={glog.std().item():.3f}  '
                      f'({(time.time()-t0)/60:.1f}m)', flush=True)
        ep_save = args.save_path.replace('.pt', f'_ep{ep+1}.pt')
        torch.save({'astro': astro.state_dict(),
                    'config': vars(args),
                    'sense_layer': sense_layer},
                   ep_save)
        print(f'Saved epoch ckpt: {ep_save}', flush=True)
    torch.save({'astro': astro.state_dict(),
                'config': vars(args),
                'sense_layer': sense_layer},
               args.save_path)
    print(f'Saved final: {args.save_path}', flush=True)
    print(f'Final λ={astro.lam.item():.3f}  '
          f'αf={astro.alpha_fast.item():.3f}  αs={astro.alpha_slow.item():.3f}',
          flush=True)


if __name__ == '__main__':
    main()
