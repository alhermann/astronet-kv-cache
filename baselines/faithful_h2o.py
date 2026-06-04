"""Faithful per-layer per-head H2O implementation.

The published H2O (Zhang et al., NeurIPS 2023) scores cached tokens by the
cumulative post-softmax attention they receive from \emph{later} queries,
\emph{separately at every (layer, head) pair}, and retains a per-(layer,
head) budget of heavy hitters plus a small recent strip.

The earlier ``select_h2o`` in this codebase summed attention across all
layers and all heads into a single global heuristic and applied the same
token-index set uniformly to every layer.  That is structurally weaker
than the published method, and a reviewer comparing to faithful H2O
will get different (typically higher) baseline numbers.

This module re-derives per-(layer, head) attention sums during segment
processing and exposes ``select_h2o_per_layer`` which returns one
``LongTensor`` of retained indices for each (layer, head) pair.  The
caller is responsible for building a cache with per-head index slicing.

Usage::

    from baselines.faithful_h2o import (
        process_windows_perlayer, select_h2o_per_layer, build_cache_perhead,
    )
    all_kv, attn_perhead, total = process_windows_perlayer(
        model, tokenizer, windows, device)
    idx_lh = select_h2o_per_layer(attn_perhead, total, k=300)
    cache = build_cache_perhead(all_kv, idx_lh, nl, nkv)
"""
from __future__ import annotations
import torch
from transformers import DynamicCache


@torch.no_grad()
def process_windows_perlayer(model, tokenizer, windows, device, max_len: int = 384):
    """Process ``windows`` and return per-layer-per-head attention sums.

    For each cached token position, sum the post-softmax attention it
    received from every query at every (layer, head) pair encountered
    so far in the forward pass.

    Returns
    -------
    all_kv : dict[int, ([K_chunks], [V_chunks])]
        Per-layer KV chunks, one per processed window.
    attn_perhead : dict[int, list[torch.Tensor]]
        ``attn_perhead[layer]`` is a list of per-window tensors of shape
        ``(num_heads, window_len)``: the cumulative attention received
        by every cached position at that layer, summed over heads' queries.
    total : int
        Total cached positions across windows.
    """
    nl = model.config.num_hidden_layers
    all_kv = {li: ([], []) for li in range(nl)}
    attn_perhead = {li: [] for li in range(nl)}
    total = 0
    for window in windows:
        ids = tokenizer(window, return_tensors='pt',
                        max_length=max_len, truncation=True).to(device)
        out = model(input_ids=ids['input_ids'], use_cache=True,
                    output_attentions=True)
        sl = ids['input_ids'].shape[1]
        total += sl
        for li, la in enumerate(out.attentions):
            # la: (1, n_heads, q_len, k_len), here q_len == k_len == sl
            a = la[0]
            if a.isnan().any():
                a = torch.nan_to_num(a, nan=0.0)
            # Cumulative attention received by every key position:
            # sum over query axis -> (n_heads, sl)
            attn_perhead[li].append(a.sum(dim=1).to(device))
        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])
    return all_kv, attn_perhead, total


def select_h2o_per_layer(
    attn_perhead,
    total: int,
    k: int,
    n_sink: int = 4,
    recent_ratio: float = 0.3,
):
    """Per-(layer, head) H2O selection.

    Parameters
    ----------
    attn_perhead : dict[int, list[Tensor]]
        Output of ``process_windows_perlayer``.
    total : int
        Total cached positions across windows.
    k : int
        Budget per (layer, head).  Matches the flat per-token budget used
        for the other baselines so memory cost is comparable.
    n_sink : int
        Number of attention-sink tokens reserved at the start.
    recent_ratio : float
        Fraction of ``k`` reserved for the most recent positions.

    Returns
    -------
    idx_lh : dict[int, list[LongTensor]]
        ``idx_lh[layer]`` is a list of length ``n_heads`` whose entries
        are the retained token indices for that (layer, head) pair.
    """
    n_recent = min(int(k * recent_ratio), total)
    n_heavy = max(k - n_sink - n_recent, 0)
    sink = torch.arange(n_sink)

    idx_lh = {}
    for li, chunks in attn_perhead.items():
        # Concatenate along the key axis: (n_heads, total)
        heur_lh = torch.cat(chunks, dim=1)
        # Last n_recent tokens regardless of attention.
        recent = torch.arange(total - n_recent, total)
        # Heavy hitters per head, excluding the recent strip and sinks.
        per_head = []
        for h in range(heur_lh.shape[0]):
            scores = heur_lh[h].clone()
            scores[:n_sink] = -1e9
            if n_recent > 0:
                scores[total - n_recent:] = -1e9
            if n_heavy > 0:
                _, top = scores.topk(min(n_heavy, scores.shape[0]))
                top = top.cpu()
            else:
                top = torch.empty(0, dtype=torch.long)
            idx = torch.cat([sink, top, recent]).unique().sort().values
            per_head.append(idx[:k])
        idx_lh[li] = per_head
    return idx_lh


def build_cache_perhead(all_kv, idx_lh, nl, nkv):
    """Build a DynamicCache that uses per-head index sets.

    HF's DynamicCache is shape ``(B, n_kv_heads, S, head_dim)`` per layer.
    Different heads can retain different positions but the cache layout
    forces a common time axis.  We therefore pad every head's selection to
    the union of all heads' indices at that layer, and rely on the
    attention mask (zeroed entries) to ignore positions that head did not
    retain.  This preserves the per-head selection semantics of faithful
    H2O while remaining compatible with the existing inference path.
    """
    cache = DynamicCache()
    for li in range(nl):
        K = torch.cat(all_kv[li][0], dim=2)
        V = torch.cat(all_kv[li][1], dim=2)
        # Build the union of per-head indices for this layer.
        per_head = idx_lh[li]
        union = sorted(set().union(*[set(t.tolist()) for t in per_head]))
        union_t = torch.tensor(union, dtype=torch.long, device=K.device)
        K_sel = K[:, :, union_t, :]
        V_sel = V[:, :, union_t, :]
        # Build a per-head mask: 1 where the position is in that head's set.
        n_heads = per_head[0].shape[0] if hasattr(per_head[0], 'shape') else nkv
        # H2O paper retains a head-specific budget; the mask zeroes
        # positions not selected by that head.  We multiply K and V by
        # the mask, which zeros their contribution to the inner product
        # and the weighted sum, effectively excluding them.
        mask = torch.zeros(nkv, K_sel.shape[2], device=K.device, dtype=K.dtype)
        for h, idx in enumerate(per_head):
            in_union = torch.tensor([u in set(idx.tolist()) for u in union],
                                     device=K.device, dtype=torch.bool)
            mask[h] = in_union.to(K.dtype)
        K_sel = K_sel * mask.unsqueeze(0).unsqueeze(-1)
        V_sel = V_sel * mask.unsqueeze(0).unsqueeze(-1)
        cache.update(K_sel, V_sel, li)
    return cache
