"""AstroGainE2E — end-to-end gain modulation trained via answer NLL.

Architectural difference from AstroGate v1/v2:
  - v1/v2 trained the gate via DISTILLATION (BCE/MSE against a relevance label).
    The trained gate was used at inference as an additive bias on SnapKV
    scores. The mismatch — training optimized a proxy, inference applied
    the result to selection — capped the win at +1.86pp on Qwen 7B (CI
    excludes zero) and +1.41pp on Llama 8B (CI grazes zero). λ never
    received gradient because the distillation loss never saw it.

  - This module trains END-TO-END via answer NLL. The gain modulates K
    vectors of SnapKV-selected ctx tokens (post-RoPE, in cache form);
    that's an attention-logit modulation Q·(gain·K) = gain·(Q·K). The
    model's frozen attention machinery handles the rest. Gradient flows
    NLL → attention → K_modulated → gain → astro params + λ. Every
    relevant parameter, including the ReZero λ, receives signal from
    the actual objective.

  - Same multi-timescale Ca²⁺ buffers as v1/v2 (fast α≈0.5 + slow α≈0.05).
    Same cross-attention sense pooling.

Borrowed scaffolding from the archived `astronet/astrocyte_gain.py`
(2026-06-08 design that was tested on the now-known-broken eval); the
sensing, EMA buffers, and per-layer projection layout are the same.
What's new here is the training regime (end-to-end NLL, not distillation),
the eval-time selection bias derived from the trained gain, and that the
ReZero λ initializes at 0 so the model starts at identity and only
deviates if NLL improves.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AstroGainE2E(nn.Module):
    def __init__(self, hidden_dim, n_astro=16, attn_dim=256,
                 n_kv_heads=4, head_dim=128, n_layers=28,
                 alpha_fast_init=0.5, alpha_slow_init=0.05):
        super().__init__()
        assert n_astro % 2 == 0
        self.hidden_dim = hidden_dim
        self.n_astro = n_astro
        self.attn_dim = attn_dim
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers

        # --- Sensing: cross-attention pooling ---
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.queries = nn.Parameter(torch.randn(1, n_astro, attn_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, attn_dim, bias=False)

        # --- Multi-timescale Ca²⁺ buffers ---
        def _logit(a): return math.log(a / (1 - a))
        self.alpha_fast_logit = nn.Parameter(torch.tensor(_logit(alpha_fast_init)))
        self.alpha_slow_logit = nn.Parameter(torch.tensor(_logit(alpha_slow_init)))
        self.n_fast = n_astro // 2
        self.n_slow = n_astro - self.n_fast
        self.register_buffer('g_fast', torch.zeros(1, self.n_fast, hidden_dim))
        self.register_buffer('g_slow', torch.zeros(1, self.n_slow, hidden_dim))

        # --- Per-layer projection: state hidden_dim → (n_kv × head_dim) ---
        # Each layer's projection produces n_astro "K-space probes" against
        # which K[ctx_token] is dot-product-scored to give per-head gain.
        # Bottlenecked through attn_dim to keep params manageable on big
        # models — naive Linear(D, n_kv·hd) per layer is ~1.8M for Qwen 7B
        # × 28 layers = 51M.
        self.state_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.layer_proj = nn.ModuleDict({
            str(li): nn.Linear(attn_dim, n_kv_heads * head_dim, bias=False)
            for li in range(n_layers)
        })

        # ReZero λ. Init 0 → gain=1 at start → forward pass is the
        # unmodulated SnapKV baseline, so the model starts at the right
        # operating point and only deviates if NLL improves.
        self.log_lambda = nn.Parameter(torch.tensor(-3.0))   # softplus(-3)≈0.049

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.state_proj.weight, gain=1.0)
        for li in self.layer_proj.values():
            nn.init.xavier_uniform_(li.weight, gain=0.5)

    @property
    def alpha_fast(self): return torch.sigmoid(self.alpha_fast_logit)
    @property
    def alpha_slow(self): return torch.sigmoid(self.alpha_slow_logit)
    @property
    def lam(self): return F.softplus(self.log_lambda)

    def reset_state(self):
        self.g_fast = torch.zeros_like(self.g_fast)
        self.g_slow = torch.zeros_like(self.g_slow)

    def sense(self, hidden_states):
        """hidden_states: [B, L, D] (any dtype)
        Returns: sensed [B, n_astro, D] (fp32)."""
        B = hidden_states.shape[0]
        x = self.input_norm(hidden_states.float())
        K = self.key_proj(x)
        Q = self.queries.expand(B, -1, -1)
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.attn_dim)
        weights = F.softmax(scores, dim=-1)
        return weights @ x

    def update_state(self, sensed, keep_grad=False):
        af, as_ = self.alpha_fast, self.alpha_slow
        s_f = sensed[:, :self.n_fast, :]
        s_s = sensed[:, self.n_fast:, :]
        if not keep_grad:
            s_f = s_f.detach(); s_s = s_s.detach()
        self.g_fast = (1 - af) * self.g_fast + af * s_f
        self.g_slow = (1 - as_) * self.g_slow + as_ * s_s

    def gain(self, layer_idx, K_layer):
        """Per-(head, position) multiplicative gain for K vectors.

        K_layer: [1, n_kv, T, head_dim] (post-RoPE K from cache; any T,
                  e.g. full ctx_len at inference or k_selected at training).
        Returns: gain [1, n_kv, T, 1] in [1−λ, 1+λ] cast to K_layer.dtype.
        """
        g = torch.cat([self.g_fast, self.g_slow], dim=1)        # [1, A, D]
        g_proj = self.state_proj(g)                              # [1, A, attn_dim]
        proj = self.layer_proj[str(layer_idx)](g_proj)           # [1, A, n_kv·hd]
        proj = proj.view(1, self.n_astro, self.n_kv_heads, self.head_dim)
        scores = torch.einsum('banh,bnTh->banT', proj, K_layer.float())
        gain_logits = scores.mean(dim=1) / math.sqrt(self.head_dim)  # [1, n_kv, T]
        gain = 1.0 + self.lam * torch.tanh(gain_logits)
        return gain.unsqueeze(-1).to(K_layer.dtype)              # [1, n_kv, T, 1]

    @torch.no_grad()
    def selection_bias(self, K_per_layer):
        """For inference: derive a per-token additive bias for SnapKV
        selection scores from the trained gain. The aggregated (across
        layer, head) gain deviation from 1 quantifies how much astrocyte
        wants this token amplified.

        K_per_layer: list of [1, n_kv, T_ctx, hd] for each layer.
        Returns: bias [T_ctx] to add to per-layer SnapKV score (or to
                  cross-layer aggregate score).
        """
        bias = None
        n = 0
        for li, K_layer in enumerate(K_per_layer):
            gain_logits = self._gain_logits_only(li, K_layer)    # [1, n_kv, T]
            contrib = gain_logits.mean(dim=1).squeeze(0)         # [T]
            bias = contrib if bias is None else bias + contrib
            n += 1
        return (bias / max(n, 1)) if bias is not None else None

    def _gain_logits_only(self, layer_idx, K_layer):
        """Same dot product as gain() but returns raw logits (no tanh,
        no λ scaling) — used as a bias signal for selection."""
        g = torch.cat([self.g_fast, self.g_slow], dim=1)
        g_proj = self.state_proj(g)
        proj = self.layer_proj[str(layer_idx)](g_proj)
        proj = proj.view(1, self.n_astro, self.n_kv_heads, self.head_dim)
        scores = torch.einsum('banh,bnTh->banT', proj, K_layer.float())
        return scores.mean(dim=1) / math.sqrt(self.head_dim)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
