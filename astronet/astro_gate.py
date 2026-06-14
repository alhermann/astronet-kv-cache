"""AstroGate — gain modulation BEFORE selection (query-agnostic regime).

Restructure rationale (2026-06-13):
  AstroHybrid added 16 virtual KV tokens that cost budget. In the
  compress-once / answer-many regime that's biologically the natural
  fit for astrocyte function, AH loses to plain SnapKV by ~9pp on
  Qwen 7B. Two structural problems:
    1. 16-slot budget tax (k_real = k - 16, vs k for SnapKV).
    2. sense() was trained with the question in context (Fix B); in
       query-agnostic deployment the sensed vector is OOD.

Biological re-anchoring:
  Astrocytes don't add synapses, they modulate them. Tripartite-synapse
  Ca²⁺ → gliotransmitter release → presynaptic release probability.
  That's GAIN on existing inputs, not new inputs.

  Multi-timescale: astrocyte Ca²⁺ has fast microdomain events (~100ms)
  and slow syncytium waves (seconds-minutes). One EMA collapses both;
  we need at least a fast/slow split.

AstroGate operates BEFORE selection:
  1. sense ctx-body (no question seen) → updates fast + slow Ca²⁺ buffers
  2. for each ctx token, gate_logit = max-pattern-match against learned
     context-conditioned "relevance patterns" derived from the astrocyte
     state
  3. snapkv_score'[i] = snapkv_score[i] + λ · gate_logit[i]
  4. top-k selects under the modulated score — all k slots are real
     tokens, no virtual budget tax.

  All k slots stay real; the astrocyte's only effect is biasing WHICH
  k tokens SnapKV keeps. ReZero λ-gate so init is identity.

Trained via distillation from the query-aware SnapKV oracle: given a
ctx body, predict the per-token relevance that the query-aware oracle
would assign once it sees the question. Aggregate across multiple
questions per paragraph → a query-independent relevance prior.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AstroGate(nn.Module):
    def __init__(self, hidden_dim, n_astro=16, attn_dim=256, n_patterns=8,
                 alpha_fast_init=0.5, alpha_slow_init=0.05):
        super().__init__()
        assert n_astro % 2 == 0, 'n_astro must be even (half fast, half slow)'
        self.hidden_dim = hidden_dim
        self.n_astro = n_astro
        self.attn_dim = attn_dim
        self.n_patterns = n_patterns

        # --- Sensing: cross-attention pooling over hidden states ---
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.queries = nn.Parameter(torch.randn(1, n_astro, attn_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, attn_dim, bias=False)

        # --- Multi-timescale Ca²⁺ buffers ---
        def _logit(a):
            return math.log(a / (1.0 - a))
        self.alpha_fast_logit = nn.Parameter(torch.tensor(_logit(alpha_fast_init)))
        self.alpha_slow_logit = nn.Parameter(torch.tensor(_logit(alpha_slow_init)))
        self.n_fast = n_astro // 2
        self.n_slow = n_astro - self.n_fast
        self.register_buffer('g_fast', torch.zeros(1, self.n_fast, hidden_dim))
        self.register_buffer('g_slow', torch.zeros(1, self.n_slow, hidden_dim))

        # --- Pattern head ---
        # State (n_astro × hidden_dim) is heavy; project to attn_dim per
        # astrocyte first so the head stays small. Pattern head then maps
        # n_astro·attn_dim → n_patterns·attn_dim. Tokens get their own
        # hidden→attn_dim projection and we compare in attn space.
        self.state_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.token_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.state_norm = nn.LayerNorm(hidden_dim)
        # Norm + GELU + norm + linear keeps the pre-activation in a stable
        # range; the bare GELU + linear stack was prone to "dying GELU"
        # collapse at moderate LR.
        head_hidden = attn_dim * 2
        self.pattern_head = nn.Sequential(
            nn.Linear(attn_dim * n_astro, head_hidden, bias=False),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, attn_dim * n_patterns, bias=False),
        )

        # ReZero λ-gate. Init small so the gate barely shifts SnapKV at
        # the start; the bias can still receive gradient because we use
        # softplus(log_lambda).
        self.log_lambda = nn.Parameter(torch.tensor(-2.0))   # softplus(-2)≈0.13

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.state_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.token_proj.weight, gain=1.0)
        for m in self.pattern_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    @property
    def alpha_fast(self):
        return torch.sigmoid(self.alpha_fast_logit)

    @property
    def alpha_slow(self):
        return torch.sigmoid(self.alpha_slow_logit)

    @property
    def lam(self):
        return F.softplus(self.log_lambda)

    def reset_state(self):
        self.g_fast = torch.zeros_like(self.g_fast)
        self.g_slow = torch.zeros_like(self.g_slow)

    def sense(self, hidden_states):
        """hidden_states: [B, L, hidden_dim] (any dtype)
        Returns: sensed [B, n_astro, hidden_dim] (fp32)."""
        B = hidden_states.shape[0]
        x = self.input_norm(hidden_states.float())
        K = self.key_proj(x)
        Q = self.queries.expand(B, -1, -1)
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.attn_dim)
        weights = F.softmax(scores, dim=-1)
        sensed = weights @ x
        return sensed

    def update_state(self, sensed, keep_grad=False):
        af = self.alpha_fast
        as_ = self.alpha_slow
        s_f = sensed[:, :self.n_fast, :]
        s_s = sensed[:, self.n_fast:, :]
        if not keep_grad:
            s_f = s_f.detach()
            s_s = s_s.detach()
        self.g_fast = (1 - af) * self.g_fast + af * s_f
        self.g_slow = (1 - as_) * self.g_slow + as_ * s_s

    def gate_logits(self, ctx_hidden):
        """Per-token relevance logits in context-body space.

        ctx_hidden: [1, T_ctx, hidden_dim] — hidden states at sense layer
                    over the context body (no question seen).
        Returns: [T_ctx] additive bias for SnapKV scores.
                 (Sign indicates promote/suppress; magnitude scaled by λ.)
        """
        g = torch.cat([self.g_fast, self.g_slow], dim=1)        # [1, n_astro, D]
        g = self.state_norm(g)
        g = self.state_proj(g)                                   # [1, n_astro, A]
        g_flat = g.flatten(start_dim=1)                          # [1, n_astro*A]
        patterns = self.pattern_head(g_flat)                     # [1, A*n_patterns]
        patterns = patterns.view(1, self.n_patterns, self.attn_dim)
        h_proj = self.token_proj(ctx_hidden.float())             # [1, T, A]
        sim = h_proj @ patterns.transpose(-1, -2)                # [1, T, n_patterns]
        sim = sim / math.sqrt(self.attn_dim)
        # Max-pattern match: token is "relevant" if it matches ANY pattern.
        # That captures heterogeneity without forcing one global motif.
        gate, _ = sim.max(dim=-1)                                # [1, T]
        return gate.squeeze(0)                                   # [T]

    def modulate_scores(self, snap_scores, ctx_hidden):
        """Apply ReZero-gated additive bias to SnapKV scores.

        snap_scores: [T_ctx] aggregate or per-layer SnapKV score tensor.
        ctx_hidden: [1, T_ctx, hidden_dim] sense-layer hidden states
                    over the context body.
        Returns: [T_ctx] modulated scores.
        """
        bias = self.gate_logits(ctx_hidden)
        return snap_scores + self.lam * bias.to(snap_scores.device)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
