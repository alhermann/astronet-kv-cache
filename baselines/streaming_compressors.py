"""Streaming-mode KV-cache compressors for SnapKV / H2O / PyramidKV.

These mirror the algorithms in ``baselines/kvcache_factory/pyramidkv_utils``
but support the **multi-window streaming regime**: the cumulative KV
cache may be much longer than the most recently prefilled chunk's
query, so the assumption ``q_len == k_len`` baked into the upstream
clusters cannot be reused.

The compressor sees, after each window's prefill:

  * ``key_cache``  — shape (bsz, num_q_heads, cumulative_len, head_dim)
  * ``value_cache`` — same
  * ``query_obs`` — shape (bsz, num_q_heads, window_obs, head_dim),
    the last ``window_obs`` query tokens of the just-prefilled window,
    used as the SnapKV-style observation set.

It returns ``key_cache``, ``value_cache`` truncated to the configured
budget for that layer.

Same scoring rules as upstream (max-pool / avg-pool over per-head
attention sums + topk).  H2O additionally keeps a running per-(layer,
kv_token) accumulator across windows — that is the "cumulative
attention" the H2O paper compresses by.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


class StreamingCompressor:
    """Per-method, multi-layer streaming KV-cache compressor.

    Holds H2O's running accumulator when ``method == 'h2o'``.  Reset
    between samples via :meth:`reset`.

    Layer-budget rules:
      * snapkv / h2o      : uniform budget = ``k`` for every layer.
      * pyramidkv         : linearly decreasing budget across layers,
                            mean = ``k``, top/bottom ratio = ``beta``.
    """

    def __init__(self, method: str, k: int, window_size: int,
                 kernel_size: int, pooling: str, num_layers: int,
                 beta: int = 20):
        assert method in ('snapkv', 'h2o', 'pyramidkv'), method
        assert pooling in ('maxpool', 'avgpool'), pooling
        self.method = method
        self.k = int(k)
        self.window_size = int(window_size)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling
        self.num_layers = int(num_layers)
        self.beta = int(beta)
        # Precompute per-layer budgets.
        self._budgets = [self._layer_budget(li) for li in range(num_layers)]
        # H2O running accumulator: list of (bsz, num_q_heads, cumulative_len)
        # tensors per layer.  None until first observation.
        self._h2o_accum = [None] * num_layers

    def reset(self):
        self._h2o_accum = [None] * self.num_layers

    def _layer_budget(self, layer_idx: int) -> int:
        if self.method != 'pyramidkv':
            return self.k
        N = self.num_layers
        if N <= 1:
            return self.k
        # Solve b_0 = beta * b_{N-1}, mean = k.
        # mean of linear interp between b_0 and b_{N-1} = (b_0 + b_{N-1}) / 2 = k
        # => b_{N-1} = 2k / (beta + 1); b_0 = 2k * beta / (beta + 1).
        b_top = 2.0 * self.k / (self.beta + 1)        # b_{N-1}: upper layers
        b_bot = 2.0 * self.k * self.beta / (self.beta + 1)  # b_0: lower layers
        # Linear interp: layer 0 → b_bot, layer N-1 → b_top
        return max(1, int(round(b_bot + (b_top - b_bot) * layer_idx / (N - 1))))

    @torch.no_grad()
    def compress(self, key_cache: torch.Tensor, value_cache: torch.Tensor,
                  query_obs: torch.Tensor, layer_idx: int):
        """Compress one layer's cumulative cache to its budget.

        Args:
            key_cache:   (bsz, num_q_heads, cumulative_len, head_dim).
                         Already repeated to q heads.
            value_cache: same shape.
            query_obs:   (bsz, num_q_heads, window_obs, head_dim).
            layer_idx:   used for pyramidkv per-layer budget.

        Returns:
            (compressed_K, compressed_V) — shape
            (bsz, num_q_heads, layer_budget, head_dim).
        """
        budget = self._budgets[layer_idx]
        cumulative_len = key_cache.shape[-2]
        head_dim = key_cache.shape[-1]
        if cumulative_len <= budget:
            return key_cache, value_cache

        w = self.window_size
        # Cannot keep more recent than we have.
        recent = min(w, cumulative_len)
        # Past-history budget = budget - recent.
        hist_budget = budget - recent
        if hist_budget <= 0:
            # Pathological config; keep only the recent slice.
            return key_cache[..., -budget:, :], value_cache[..., -budget:, :]

        # 1) attention-score the observation queries against ALL keys
        #    (use the last min(w, q_obs_len) of the observation window).
        obs = query_obs[..., -min(w, query_obs.shape[-2]):, :]
        scale = math.sqrt(head_dim)
        attn_w = torch.matmul(obs, key_cache.transpose(-2, -1)) / scale
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(
            query_obs.dtype)
        # Sum across the observation-token axis: shape (b, h, cumulative_len)
        attn_sum = attn_w.sum(dim=-2)

        # 2) H2O accumulates across windows; SnapKV / PyramidKV do not.
        if self.method == 'h2o':
            prior = self._h2o_accum[layer_idx]
            if prior is not None:
                # Pad prior to current cumulative_len (new tokens get score 0).
                if prior.shape[-1] < cumulative_len:
                    pad = cumulative_len - prior.shape[-1]
                    prior = torch.cat([
                        prior,
                        torch.zeros(*prior.shape[:-1], pad,
                                     device=prior.device, dtype=prior.dtype),
                    ], dim=-1)
                attn_sum = attn_sum + prior

        # 3) Pool to smooth (SnapKV / PyramidKV trick).
        if self.method in ('snapkv', 'pyramidkv'):
            if self.pooling == 'maxpool':
                attn_pool = F.max_pool1d(
                    attn_sum, kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2, stride=1)
            else:
                attn_pool = F.avg_pool1d(
                    attn_sum, kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2, stride=1)
        else:
            attn_pool = attn_sum

        # 4) Pick top hist_budget heavy hitters from the past
        #    (positions BEFORE the recent slice).
        history_scores = attn_pool[..., : cumulative_len - recent]
        if history_scores.shape[-1] == 0:
            # Past has been chopped completely; keep only recent.
            return (key_cache[..., -recent:, :],
                     value_cache[..., -recent:, :])
        if hist_budget > history_scores.shape[-1]:
            hist_budget = history_scores.shape[-1]
        topk = history_scores.topk(hist_budget, dim=-1).indices
        # 5) Plus the recent strip — positions
        #    [cumulative_len - recent, cumulative_len).
        recent_idx = torch.arange(
            cumulative_len - recent, cumulative_len,
            device=key_cache.device).view(1, 1, -1).expand(
            topk.shape[0], topk.shape[1], -1)
        all_idx = torch.cat([topk, recent_idx], dim=-1)  # (b, h, budget)
        # 6) Gather K/V.
        idx4 = all_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        compressed_K = key_cache.gather(dim=-2, index=idx4)
        compressed_V = value_cache.gather(dim=-2, index=idx4)
        # 7) For H2O, also gather the running accumulator down to the
        #    same selected indices so it persists into the next window.
        if self.method == 'h2o':
            self._h2o_accum[layer_idx] = attn_sum.gather(dim=-1, index=all_idx)
        return compressed_K, compressed_V
