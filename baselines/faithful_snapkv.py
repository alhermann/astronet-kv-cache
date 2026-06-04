"""Faithful single-layer SnapKV implementation.

The published SnapKV (Li et al., NeurIPS 2024) scores cached tokens by
attention received from a short observation window of the prompt-tail
at a single attention layer (the last layer in the reference
implementation), with avg-pool smoothing and a recent-window keep.

Our earlier ``select_snapkv`` averaged scores across four scoring layers,
which is the multi-layer variant introduced as AstroNet Stage 1.  This
module provides the genuinely single-layer SnapKV baseline so the paper
can compare against an unmodified Stage 1 ancestor.

Usage::

    from baselines.faithful_snapkv import select_snapkv_faithful
    idx = select_snapkv_faithful(model, all_kv, question, tokenizer,
                                 k=300, device=device)
"""
from __future__ import annotations
import math
import torch


def select_snapkv_faithful(
    model,
    all_kv,
    question: str,
    tokenizer,
    k: int,
    device,
    layer_idx: int | None = None,
    observation_window: int = 32,
    n_sink: int = 4,
    recent_ratio: float = 0.2,
    smooth_kernel: int = 5,
):
    """Faithful SnapKV: single-attention-layer scoring against observation window.

    Implements SnapKV (Li et al., NeurIPS 2024) as published:

    * Score every cached token by the attention it receives from a short
      observation window of length ``observation_window`` (default 32, the
      SnapKV default).
    * Compute that attention at a SINGLE attention layer (default ``L-1``,
      matching the reference implementation).
    * Apply a 1-D avg-pool of kernel ``smooth_kernel`` to the per-token
      score before top-k selection (SnapKV default kernel = 5).
    * Reserve attention sinks at the front and a recent strip at the end.

    Our multi-segment setting does not have a single concatenated prompt,
    so the natural ``prompt tail'' is the user's question segment.  The
    function therefore constructs an observation window by taking the
    last ``observation_window`` tokens of ``"Question: ...\\nAnswer:"``.
    """
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, "head_dim",
                 model.config.hidden_size // nq)
    qpk = nq // nkv

    if layer_idx is None:
        layer_idx = nl - 1

    # ---- compute the score at the SINGLE chosen layer ----
    # Tokenise the question prompt; trim to the last observation_window
    # tokens to match SnapKV's spec (default W=32).
    q_ids_full = tokenizer(f"Question: {question}\nAnswer:",
                            return_tensors="pt",
                            max_length=128,
                            truncation=True).to(device)
    full_ids = q_ids_full["input_ids"]
    if full_ids.shape[1] > observation_window:
        obs_ids = full_ids[:, -observation_window:]
    else:
        obs_ids = full_ids
    with torch.no_grad():
        q_out = model(input_ids=obs_ids, output_hidden_states=True)

    total = torch.cat(all_kv[layer_idx][0], dim=2).shape[2]
    cross = torch.zeros(total, device=device)

    layer_dev = all_kv[layer_idx][0][0].device
    Q = (model.model.layers[layer_idx]
         .self_attn.q_proj(
             q_out.hidden_states[layer_idx][0].float().to(layer_dev)
         )
         .half()
         .view(-1, nq, hd))
    K = torch.cat(all_kv[layer_idx][0], dim=2)[0]
    for hi in range(nkv):
        sc = torch.matmul(
            Q[:, hi * qpk:(hi + 1) * qpk, :].float(),
            K[hi].float().T,
        ) / math.sqrt(hd)
        attn_w = torch.softmax(sc, dim=-1)
        cross += attn_w.sum(dim=(0, 1)).to(device)

    # ---- avg-pool smoothing (SnapKV default) ----
    if smooth_kernel and total > smooth_kernel:
        cross = torch.nn.functional.avg_pool1d(
            cross.unsqueeze(0).unsqueeze(0),
            kernel_size=smooth_kernel,
            padding=smooth_kernel // 2,
            stride=1,
        ).squeeze()

    # ---- sinks at the front ----
    cross[:n_sink] = -1e9

    # ---- recent strip from the last cached segment ----
    last_window_size = all_kv[layer_idx][0][-1].shape[2]
    last_start = total - last_window_size
    n_recent = min(int(k * recent_ratio), last_window_size)
    recent_idx = torch.arange(last_start, last_start + n_recent,
                               device=device)

    # ---- top-k from the non-recent, non-sink region ----
    scores = cross.clone()
    scores[last_start:total] = -1e9
    n_select = max(k - n_sink - n_recent, 0)
    n_avail = (scores > -1e8).sum().item()
    n_select = min(n_select, n_avail, scores.shape[0])
    if n_select > 0:
        _, top_idx = scores.topk(n_select)
    else:
        top_idx = torch.empty(0, dtype=torch.long, device=device)

    sink_idx = torch.arange(n_sink, device=device)
    idx = torch.cat([sink_idx, top_idx, recent_idx]).unique().sort().values
    return idx[:k]
