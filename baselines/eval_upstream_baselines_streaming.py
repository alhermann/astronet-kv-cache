"""Streaming-mode SQuAD pos-robust eval for SnapKV / H2O / PyramidKV.

This puts the published baselines in the SAME REGIME AstroNet operates
in (matches the protocol used by ``training/train_hybrid.py::evaluate``
which produces our ``pure300``/hybrid numbers):

  1. Process all fact windows sequentially through the model with
     ``use_cache=True``; collect the cumulative K / V (no compression
     in between).
  2. Forward the question text in a SEPARATE pass to obtain
     question-conditioned Q at the inject layers.
  3. Score cumulative K against question Q using the method's rule:
        * SnapKV:   max-pool kernel-7 + topk
        * H2O:      cumulative-attention-sum + topk + recent strip
        * PyramidKV: max-pool kernel-5 + per-layer decreasing budget
  4. Select indices, build a fresh cache, run the question with the
     compressed cache + greedy decode.

This is the SAME protocol as AstroNet's S1 ("pure300").  The only
difference between methods is the per-layer scoring rule.

Usage::

    python baselines/eval_upstream_baselines_streaming.py \\
        --model_path ./models/qwen2.5-7b --method snapkv --k 300 \\
        --n_eval 100 --positions 0 1 2 3 \\
        --save_path logs/results/upstream_squad_snapkv_qwen7b_streaming.json
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _family_from_path(model_path: str) -> str:
    name = os.path.basename(model_path.rstrip('/')).lower()
    if 'llama' in name:  return 'llama'
    if 'mistral' in name: return 'mistral'
    if 'qwen' in name:   return 'qwen'
    raise ValueError(f'cannot infer model family from {model_path!r}')


def load_backbone(model_path: str, multi_gpu: bool):
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    common = dict(quantization_config=bnb, torch_dtype=torch.float16,
                   attn_implementation='sdpa')
    if multi_gpu:
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            if gib >= 16:
                max_memory[i] = '22GiB'
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto', max_memory=max_memory, **common)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map={'': 'cuda:0'}, **common)
    model.eval()
    return model, tokenizer


# ---------------------------- core ---------------------------------------

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _inject_layers(num_hidden_layers: int):
    """Layers used for scoring (matches train_hybrid.py choice)."""
    nl = num_hidden_layers
    return [nl // 4, nl // 2, 3 * nl // 4, nl - 2]


def _gather_kv_cumulative(model, tokenizer, sample, device, max_w_tokens=4096):
    """Process each fact window FRESH (no past_key_values), then
    concatenate per-window K / V across the sequence dim — matches
    ``training/train_hybrid.evaluate`` lines 514-530 exactly.

    This is the AstroNet-S1 streaming protocol: each window is
    RoPE-rotated starting at position 0 (not at cumulative_len).
    Combined with a question Q that is also RoPE-rotated at positions
    [0, q_len), this puts queries and keys in the same RoPE range so
    cross-attention scoring is meaningful — necessary for the per-
    window KV-selection step to identify the correct heavy hitters.

    Returns ``(all_kv, total_len, last_window_size)``.

    * ``all_kv[li] = (K_layer, V_layer)`` with shape
      ``(bsz=1, num_kv_heads, total_len, head_dim)`` — concatenation of
      the per-window K / V along seq dim.
    * ``last_window_size`` = size of the LAST fact window's K
      contribution to the concatenation; needed by the recent-strip.
    """
    nl = model.config.num_hidden_layers
    embed_dev = model.get_input_embeddings().weight.device
    per_window_K = {li: [] for li in range(nl)}
    per_window_V = {li: [] for li in range(nl)}
    last_w_size = 0
    for win in sample.windows[:-1]:
        ids = tokenizer(win, return_tensors='pt', truncation=True,
                         max_length=max_w_tokens).to(embed_dev)
        # Fresh forward — no past_key_values.  Each window's K is
        # RoPE'd at positions [0, win_len).
        out = model(input_ids=ids['input_ids'], use_cache=True)
        pkv = out.past_key_values
        last_w_size = ids['input_ids'].shape[1]
        for li in range(nl):
            per_window_K[li].append(
                pkv.key_cache[li] if hasattr(pkv, 'key_cache')
                else pkv[li][0])
            per_window_V[li].append(
                pkv.value_cache[li] if hasattr(pkv, 'value_cache')
                else pkv[li][1])
    # Concatenate along seq dim per layer.
    import torch
    all_kv = []
    for li in range(nl):
        K = torch.cat(per_window_K[li], dim=2)
        V = torch.cat(per_window_V[li], dim=2)
        all_kv.append((K, V))
    total = all_kv[0][0].shape[-2]
    return all_kv, total, last_w_size


def _question_q_at_layers(model, tokenizer, question_text, inject_layers, device):
    """Forward the question separately, return Q (post-q_proj) at each
    of the inject layers."""
    embed_dev = model.get_input_embeddings().weight.device
    q_ids = tokenizer(question_text, return_tensors='pt',
                        max_length=128, truncation=True).to(embed_dev)
    with torch.no_grad():
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
    cfg = model.config
    nq = cfg.num_attention_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // nq)
    q_states = {}
    for li in inject_layers:
        h = q_out.hidden_states[li][0].float()  # (T, hidden)
        q = model.model.layers[li].self_attn.q_proj(h).half()  # (T, nq*hd)
        q_states[li] = q.view(-1, nq, hd)  # (T, nq, hd)
    return q_states


def _score_method(all_kv, q_states, inject_layers, num_q_heads,
                   num_kv_heads, head_dim, total_len, method: str,
                   kernel_size: int, pooling: str):
    """Compute a single cross-window importance vector over total_len
    using the method's pooling.  Layer-wise aggregation summed."""
    qpk = num_q_heads // num_kv_heads
    cross = torch.zeros(total_len, device=q_states[inject_layers[0]].device)
    for li in inject_layers:
        Q = q_states[li]  # (T, nq, hd)
        K = all_kv[li][0][0]  # (num_kv, total_len, hd)
        for hi in range(num_kv_heads):
            sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                K[hi].float().T) / math.sqrt(head_dim)
            attn_w = torch.softmax(sc, dim=-1)
            cross += attn_w.sum(dim=(0, 1)).to(cross.device)
    # Pool
    if total_len > kernel_size:
        if pooling == 'maxpool':
            cross_p = F.max_pool1d(
                cross.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                padding=kernel_size // 2, stride=1,
            ).squeeze()
        else:
            cross_p = F.avg_pool1d(
                cross.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                padding=kernel_size // 2, stride=1,
            ).squeeze()
        cross_p = cross_p[:total_len]
    else:
        cross_p = cross
    return cross_p


def _select_indices(cross, total_len, k, method, last_window_size,
                     num_layers, beta=20):
    """Per-method index selection — matches AstroNet S1 protocol
    (``training/train_hybrid.evaluate`` lines 566-581).

    Recent strip = first ``n_recent`` positions of the LAST FACT window
    (NOT the last ``window_size`` positions of the cumulative cache).
    Heavy-hitter region = everything before the last fact window.

    ``last_window_size`` is the size of the last fact window's K
    contribution to the cumulative cache.

    For SnapKV / H2O: uniform budget k, returns a single tensor.
    For PyramidKV: PER-LAYER uniform budget k (we disabled per-layer
    variation to avoid the DynamicCache-padding RoPE bug;
    see docstring of ``stream_then_answer`` for full explanation).
    Returns a single tensor in all cases.
    """
    n_sink = 4
    last_start = total_len - last_window_size
    n_recent = min(int(k * 0.2), last_window_size)
    recent = torch.arange(last_start, last_start + n_recent,
                            device=cross.device)
    scores = cross.clone()
    scores[:n_sink] = -1e9
    # Mask the ENTIRE last fact window from heavy-hit scoring (it is
    # already represented via the recent strip).
    scores[last_start:total_len] = -1e9
    n_select = max(k - n_sink - len(recent), 0)
    n_avail = (scores > -1e8).sum().item()
    n_select = min(n_select, n_avail)
    _, top = scores.topk(n_select) if n_select > 0 else (
        None, torch.empty(0, dtype=torch.long, device=cross.device))
    sink = torch.arange(n_sink, device=cross.device)
    idx = torch.cat([sink, top, recent]).unique().sort().values
    return idx[:k]


def _select_kv(all_kv, li, idx):
    K, V = all_kv[li]
    li_idx = idx.to(K.device)
    return K[:, :, li_idx, :], V[:, :, li_idx, :]


@torch.no_grad()
def stream_then_answer(model, tokenizer, sample, method: str, k: int,
                       inject_layers, max_new_tokens=20):
    from transformers import DynamicCache
    cfg = model.config
    nl = cfg.num_hidden_layers
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // nq)
    embed_dev = model.get_input_embeddings().weight.device

    # 1) Collect cumulative KV across fact windows.
    all_kv, total_len, last_window_size = _gather_kv_cumulative(
        model, tokenizer, sample, embed_dev)

    # 2) Question-conditioned Q for scoring.
    q_text = f'Question: {sample.question}\nAnswer:'
    q_states = _question_q_at_layers(model, tokenizer, q_text,
                                       inject_layers, embed_dev)

    # 3) Method-specific scoring (only the pooling+kernel differs).
    #    The recent-strip + heavy-hit selection happens in
    #    _select_indices using last_window_size.
    if method == 'snapkv':
        cross = _score_method(all_kv, q_states, inject_layers, nq, nkv, hd,
                                total_len, 'snapkv', kernel_size=7,
                                pooling='maxpool')
    elif method == 'h2o':
        # In our one-shot question-conditioned protocol, "H2O" reduces to
        # SnapKV-without-pooling (cumulative attention sum equivalent
        # under a single observation). Different from published H2O which
        # accumulates across DECODING steps; flag in paper.
        cross = _score_method(all_kv, q_states, inject_layers, nq, nkv, hd,
                                total_len, 'h2o', kernel_size=1,
                                pooling='avgpool')
    elif method == 'pyramidkv':
        cross = _score_method(all_kv, q_states, inject_layers, nq, nkv, hd,
                                total_len, 'pyramidkv', kernel_size=5,
                                pooling='maxpool')
    else:
        raise ValueError(method)

    # 4) Indices (uniform budget for all methods — per-layer pyramidkv
    #    requires custom HF cache handling we don't yet have; flag).
    sel = _select_indices(cross, total_len, k, method, last_window_size,
                            nl, beta=20)

    # 5) Build fresh cache (single selection across all layers).
    cache = DynamicCache()
    for li in range(nl):
        K, V = _select_kv(all_kv, li, sel)
        cache.update(K, V, li)
    prefix_len = sel.shape[0]

    # 6) Question prefill + greedy decode (manual loop).
    query = (f'Based on what you read earlier, answer the following '
             f'question.\nQuestion: {sample.question}\nAnswer:')
    fq = tokenizer(query, return_tensors='pt', max_length=384,
                    truncation=True).to(embed_dev)
    pos = torch.arange(prefix_len,
                         prefix_len + fq['input_ids'].shape[1],
                         device=embed_dev).unsqueeze(0)
    cur = fq['input_ids']
    gen = []
    for _ in range(max_new_tokens):
        out = model(input_ids=cur, past_key_values=cache,
                     position_ids=pos, use_cache=True)
        cache = out.past_key_values
        nxt = out.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
        tok_id = int(nxt[0, 0].item())
        gen.append(tok_id)
        if tok_id == tokenizer.eos_token_id:
            break
        cur = nxt.to(embed_dev)
        pos = torch.tensor(
            [[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]],
            device=embed_dev)
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--method', required=True,
                   choices=['snapkv', 'h2o', 'pyramidkv'])
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--max_new_tokens', type=int, default=20)
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    family = _family_from_path(args.model_path)
    print(f'[streaming-baseline] family={family} method={args.method} '
          f'k={args.k}  n_eval={args.n_eval}  seed={args.seed}', flush=True)

    model, tokenizer = load_backbone(args.model_path, args.multi_gpu)
    inject_layers = _inject_layers(model.config.num_hidden_layers)
    print(f'  layers={model.config.num_hidden_layers}  '
           f'inject_layers={inject_layers}', flush=True)

    from data.real_qa import generate_squad_dataset
    from training.eval_hybrid_position_robust import shuffle_fact_position
    samples = generate_squad_dataset(
        n_samples=args.n_eval, n_windows=5, vary_distance=True,
        seed=args.seed, split='validation')

    results = {}
    for pos in args.positions:
        print(f'\n=== pos={pos} ===', flush=True)
        placed = shuffle_fact_position(samples, pos)
        correct = 0
        t0 = time.time()
        for si, s in enumerate(placed):
            ans = stream_then_answer(
                model, tokenizer, s, args.method, args.k,
                inject_layers, max_new_tokens=args.max_new_tokens)
            if s.answer.lower() in ans.lower():
                correct += 1
            if (si + 1) % 25 == 0:
                print(f'  pos={pos} {si+1}/{len(placed)} '
                       f'acc={correct}/{si+1}', flush=True)
        acc = correct / max(1, len(placed))
        results[f'pos_{pos}'] = acc
        print(f'  pos={pos}: acc={acc * 100:.1f}%  '
               f'({time.time() - t0:.0f}s)', flush=True)

    avg = sum(results.values()) / max(1, len(results))
    print(f'\nAVG across positions: {avg * 100:.1f}%', flush=True)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'family': family,
            'method': args.method,
            'k': args.k,
            'n_eval': args.n_eval,
            'seed': args.seed,
            'positions': args.positions,
            'results': results,
            'average': avg,
            'source': 'streaming-mode baselines (cumulative KV, '
                       'question-conditioned scoring, method-specific '
                       'pooling+selection — matches AstroNet S1 protocol)',
        }, f, indent=2)
    print(f'Saved -> {args.save_path}')


if __name__ == '__main__':
    main()
