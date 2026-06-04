"""Faithful per-layer H2O implementation.

The published H2O (Zhang et al., NeurIPS 2023) scores cached tokens by
the cumulative post-softmax attention they receive, with the scoring
computed \emph{at each transformer layer separately} (not by a single
attention map averaged across the whole stack).  The earlier
``select_h2o`` in ``eval_all_baselines.py`` summed attention across all
layers and all heads into a single global heuristic and applied the
resulting indices uniformly to every layer; that is structurally weaker
than the published method.

We provide a per-layer H2O implementation here.  At every layer we sum
attention across that layer's heads, accumulate across windows, and
retain the top-k tokens for that layer.  Different layers therefore
retain different index sets, but within a layer all KV heads share the
same selection.  This matches the published H2O paper's reported
ablation that within-layer head variation is minor, and it avoids the
per-head bookkeeping that would otherwise force every layer to share a
common cache time axis.

Usage::

    from baselines.faithful_h2o import (
        process_windows_perlayer, select_h2o_per_layer, build_cache_perlayer,
    )
    all_kv, attn_perlayer, total = process_windows_perlayer(
        model, tokenizer, windows, device)
    idx_per_layer = select_h2o_per_layer(attn_perlayer, total, k=300)
    cache = build_cache_perlayer(all_kv, idx_per_layer, nl)
"""
from __future__ import annotations
import torch
from transformers import DynamicCache


@torch.no_grad()
def process_windows_perlayer(model, tokenizer, windows, device, max_len: int = 384):
    """Process ``windows`` and return per-layer attention sums.

    For each cached token position, sum the post-softmax attention it
    received from every query at every head, separately at every
    transformer layer.

    Returns
    -------
    all_kv : dict[int, ([K_chunks], [V_chunks])]
        Per-layer KV chunks, one per processed window.
    attn_perlayer : dict[int, list[torch.Tensor]]
        ``attn_perlayer[layer]`` is a list of per-window 1-D tensors
        whose entries are the per-position attention sum at that layer.
    total : int
        Total cached positions across windows.
    """
    nl = model.config.num_hidden_layers
    all_kv = {li: ([], []) for li in range(nl)}
    attn_perlayer = {li: [] for li in range(nl)}
    total = 0
    for window in windows:
        ids = tokenizer(window, return_tensors='pt',
                        max_length=max_len, truncation=True).to(device)
        out = model(input_ids=ids['input_ids'], use_cache=True,
                    output_attentions=True)
        sl = ids['input_ids'].shape[1]
        total += sl
        for li, la in enumerate(out.attentions):
            # la: (1, n_q_heads, q_len, k_len)
            a = la[0]
            if a.isnan().any():
                a = torch.nan_to_num(a, nan=0.0)
            # Sum across all heads and across the query axis ->
            # per-position attention received at this layer (1-D, sl).
            attn_per_pos = a.sum(dim=(0, 1)).to(device).float()
            attn_perlayer[li].append(attn_per_pos)
        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])
    return all_kv, attn_perlayer, total


def select_h2o_per_layer(
    attn_perlayer,
    total: int,
    k: int,
    n_sink: int = 4,
    recent_ratio: float = 0.3,
):
    """Per-layer H2O selection.

    Each layer's top-k is computed from its own accumulated attention
    sum.  Attention sinks at the front and a recent strip at the end are
    reserved as in the published implementation.

    Returns
    -------
    idx_per_layer : list[LongTensor]
        ``idx_per_layer[layer]`` is a sorted 1-D index tensor of length
        at most ``k`` for that layer.
    """
    n_recent = min(int(k * recent_ratio), total)
    n_heavy = max(k - n_sink - n_recent, 0)
    sink = torch.arange(n_sink)
    recent = torch.arange(total - n_recent, total)
    idx_per_layer = []
    for li in sorted(attn_perlayer.keys()):
        chunks = attn_perlayer[li]
        # Concatenate per-window scores along the key axis -> (total,).
        scores = torch.cat(chunks, dim=0)
        scores[:n_sink] = -float('inf')
        if n_recent > 0:
            scores[total - n_recent:] = -float('inf')
        if n_heavy > 0:
            _, top = scores.topk(min(n_heavy, scores.shape[0]))
            top = top.cpu()
        else:
            top = torch.empty(0, dtype=torch.long)
        idx = torch.cat([sink, top, recent]).unique().sort().values[:k]
        idx_per_layer.append(idx)
    return idx_per_layer


def build_cache_perlayer(all_kv, idx_per_layer, nl):
    """Build a per-layer DynamicCache with possibly different per-layer
    index sets.  HF's DynamicCache supports variable per-layer seq sizes
    natively.
    """
    cache = DynamicCache()
    for li in range(nl):
        K = torch.cat(all_kv[li][0], dim=2)
        V = torch.cat(all_kv[li][1], dim=2)
        idx = idx_per_layer[li].to(K.device)
        cache.update(K[:, :, idx, :], V[:, :, idx, :], li)
    return cache
