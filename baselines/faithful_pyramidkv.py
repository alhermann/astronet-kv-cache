"""Faithful PyramidKV implementation in our multi-window pipeline.

The published PyramidKV (Cai et al., 2024) has two ingredients:

  1. Per-layer SnapKV-style scoring: at every transformer layer, score
     cached tokens by the attention they receive from a short observation
     window (default W=32) at that layer, with avg-pool kernel-5 smoothing.
  2. A decreasing per-layer budget schedule: the early (input-near)
     layers get a larger KV budget and the late (output-near) layers
     get a smaller one, with the total summing to a fixed flat budget.

The earlier ``eval_pyramidkv.py`` in this codebase used ingredient (2) but
replaced ingredient (1) with a global cumulative-attention heuristic
shared across all layers (the H2O scoring rule).  That is not the
published PyramidKV.  A reviewer who runs the upstream PyramidKV repo on
the same models will get different numbers.

This module implements both ingredients faithfully and integrates with
our multi-window pipeline.  The upstream PyramidKV monkey-patches the
attention layers of Llama and Mistral but not Qwen; re-implementing at
the selection level (rather than at the attention forward) avoids that
limitation while preserving the algorithm.

Usage::

    from baselines.faithful_pyramidkv import (
        select_pyramidkv_per_layer, build_pyramidkv_cache,
    )
    idx_per_layer = select_pyramidkv_per_layer(
        model, all_kv, question, tokenizer,
        avg_budget=k, device=device, observation_window=32,
    )
    cache = build_pyramidkv_cache(all_kv, idx_per_layer, nl)
"""
from __future__ import annotations
import math
import torch
from transformers import DynamicCache


def pyramid_budget_schedule(n_layers: int, avg_budget: int,
                             top_to_bottom_ratio: float = 0.5,
                             min_budget: int | None = None):
    """Decreasing per-layer budget summing to ``n_layers * avg_budget``.

    Layer 0 receives the largest budget; layer ``n_layers - 1`` the
    smallest.  ``top_to_bottom_ratio`` controls the ratio of the bottom
    budget to the top.
    """
    avg = (1.0 + top_to_bottom_ratio) / 2.0
    top = avg_budget / avg
    bots = []
    for li in range(n_layers):
        frac = 1.0 - (1.0 - top_to_bottom_ratio) * li / max(n_layers - 1, 1)
        bots.append(int(round(top * frac)))
    if min_budget is not None:
        bots = [max(b, min_budget) for b in bots]
    return bots


@torch.no_grad()
def select_pyramidkv_per_layer(
    model,
    all_kv,
    question: str,
    tokenizer,
    avg_budget: int,
    device,
    observation_window: int = 32,
    n_sink: int = 4,
    recent_ratio: float = 0.2,
    smooth_kernel: int = 5,
    top_to_bottom_ratio: float = 0.5,
):
    """Per-layer PyramidKV selection.

    At every transformer layer, recompute query projections from the last
    ``observation_window`` tokens of ``"Question: ...\\nAnswer:"``,
    score the cached keys at that layer, apply avg-pool smoothing, and
    retain a layer-specific top-k determined by the pyramid schedule.
    """
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, "head_dim",
                 model.config.hidden_size // nq)
    qpk = nq // nkv

    # Observation window: last W tokens of the question prompt.
    q_ids_full = tokenizer(f"Question: {question}\nAnswer:",
                            return_tensors="pt",
                            max_length=128, truncation=True).to(device)
    full_ids = q_ids_full["input_ids"]
    if full_ids.shape[1] > observation_window:
        obs_ids = full_ids[:, -observation_window:]
    else:
        obs_ids = full_ids
    q_out = model(input_ids=obs_ids, output_hidden_states=True)

    total = torch.cat(all_kv[0][0], dim=2).shape[2]
    budgets = pyramid_budget_schedule(
        nl, avg_budget,
        top_to_bottom_ratio=top_to_bottom_ratio,
        min_budget=n_sink + max(int(avg_budget * recent_ratio), 1),
    )
    budgets = [min(b, total) for b in budgets]

    idx_per_layer = []
    last_window_size = all_kv[0][0][-1].shape[2]
    last_start = total - last_window_size

    for li in range(nl):
        layer_dev = all_kv[li][0][0].device
        Q = (model.model.layers[li]
             .self_attn.q_proj(
                 q_out.hidden_states[li][0].float().to(layer_dev)
             )
             .half()
             .view(-1, nq, hd))
        K = torch.cat(all_kv[li][0], dim=2)[0]
        score = torch.zeros(total, device=device)
        for hi in range(nkv):
            sc = torch.matmul(
                Q[:, hi * qpk:(hi + 1) * qpk, :].float(),
                K[hi].float().T,
            ) / math.sqrt(hd)
            attn_w = torch.softmax(sc, dim=-1)
            score += attn_w.sum(dim=(0, 1)).to(device)

        if smooth_kernel and total > smooth_kernel:
            score = torch.nn.functional.avg_pool1d(
                score.unsqueeze(0).unsqueeze(0),
                kernel_size=smooth_kernel,
                padding=smooth_kernel // 2,
                stride=1,
            ).squeeze()

        score[:n_sink] = -1e9
        b = budgets[li]
        n_recent = min(int(b * recent_ratio), last_window_size)
        recent = torch.arange(last_start, last_start + n_recent,
                              device=device)
        scores = score.clone()
        scores[last_start:total] = -1e9
        n_select = max(b - n_sink - n_recent, 0)
        n_avail = (scores > -1e8).sum().item()
        n_select = min(n_select, n_avail, scores.shape[0])
        if n_select > 0:
            _, top = scores.topk(n_select)
        else:
            top = torch.empty(0, dtype=torch.long, device=device)
        sink = torch.arange(n_sink, device=device)
        idx = torch.cat([sink, top, recent]).unique().sort().values[:b]
        idx_per_layer.append(idx)
    return idx_per_layer


def build_pyramidkv_cache(all_kv, idx_per_layer, nl):
    """Build a per-layer DynamicCache that respects PyramidKV's variable
    per-layer budgets.

    Each layer's cache is sliced to its own index set.  HF's DynamicCache
    accepts different ``seq`` per layer natively.
    """
    cache = DynamicCache()
    for li in range(nl):
        K = torch.cat(all_kv[li][0], dim=2)
        V = torch.cat(all_kv[li][1], dim=2)
        idx = idx_per_layer[li].to(K.device)
        cache.update(K[:, :, idx, :], V[:, :, idx, :], li)
    return cache
