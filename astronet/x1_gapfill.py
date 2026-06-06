"""X1 augmentation — cross-attention from S2 to S1-selected hidden states.

The S2 module of AstroHybrid emits 16 virtual KV per layer from an
EMA-summarised state ``g``.  That summary is *selector-blind*: S2
doesn't know which 284 real KV indices S1 picked.  X1 makes S2
selector-aware by letting its per-layer virtual hidden states attend
over the hidden states of the *selected* tokens at each inject layer
before going through the backbone's W_K / W_V.

The module is **initialised as identity** (output projection zeroed)
so freshly-instantiating an X1-augmented AstroHybrid reproduces the
baseline AstroHybrid behaviour exactly.  Lift only appears after
training pushes the gap-fill projection away from zero.

Per-layer parameter cost: hidden_dim**2 * 2 + hidden_dim * attn_dim * 2.
For Qwen 7B (hidden=3584, attn=256), that's about 28 × (12.8M + 1.8M)
= 410M if applied to every layer.  We therefore apply it ONLY at the
inject layers (4 by default) → 4 × 14.6M = 58M extra params total.
Acceptable on top of the existing AstroHybrid 5-10M.

Reference: critic's recommended X1 approach (lift +3-5 pp).  Paper
mechanism is "S2 learns to fill the gaps S1 systematically misses."
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GapFillCrossAttn(nn.Module):
    """Single-layer gap-fill cross-attention.

    Inputs:
        h_virt: (B, n_mem, hidden_dim) — current S2 virtual hidden
                states for this layer (output of layer_up/shared_up).
        h_selected: (B, k_real, hidden_dim) — hidden states of the
                S1-selected tokens at this layer.

    Output: (B, n_mem, hidden_dim) — augmented S2 hidden states.
            Equals h_virt initially (output proj is zero-init); lifts
            only after training.
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 256,
                  use_rezero: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Output projection.  In the ReZero variant (2026-06-06 critic fix)
        # this is xavier-initialised at small scale and gated by a learnable
        # scalar gamma initialised at zero, so the residual contribution
        # starts as exactly zero and grows only when training pushes gamma
        # off zero.  In the legacy zero-out_proj variant the projection
        # itself starts at zero; we keep the legacy path for reproducibility
        # of the catastrophic failure mode in tools/checkpoints from the
        # 2026-06-06 X1/X2 experiments.
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_scale = 1.0 / math.sqrt(attn_dim)
        # Layer norm on inputs for stable gradients.
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.k_norm = nn.LayerNorm(hidden_dim)
        self.use_rezero = use_rezero
        if use_rezero:
            # ReZero scalar gate, init 0.  At init the residual contribution
            # is identically zero regardless of out_proj.weight, so out_proj
            # can be xavier-initialised safely; the gate then grows away
            # from zero only when training demands a non-trivial residual.
            self.gamma = nn.Parameter(torch.zeros(1))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.5)
        if self.use_rezero:
            # With ReZero gate at zero, residual contribution is gated
            # off at init regardless of out_proj scale.  Use a small
            # xavier init so the residual magnitude stays bounded.
            nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        else:
            # Legacy: zero-init out_proj for identity start.
            nn.init.zeros_(self.out_proj.weight)

    def forward(self, h_virt: torch.Tensor,
                h_selected: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(self.q_norm(h_virt.float()))   # (B, n_mem, attn_dim)
        k = self.k_proj(self.k_norm(h_selected.float()))  # (B, k_real, attn_dim)
        v = self.v_proj(h_selected.float())             # (B, k_real, hidden_dim)
        # Scaled dot-product attention.
        sc = torch.matmul(q, k.transpose(-2, -1)) * self.attn_scale
        attn = F.softmax(sc, dim=-1)
        residual = self.out_proj(torch.matmul(attn, v))  # (B, n_mem, hidden_dim)
        if self.use_rezero:
            residual = self.gamma * residual
        return h_virt + residual


class GapFillPerInjectLayer(nn.Module):
    """One ``GapFillCrossAttn`` per inject layer, indexed by layer id."""

    def __init__(self, inject_layers, hidden_dim: int, attn_dim: int = 256,
                  use_rezero: bool = True):
        super().__init__()
        self.inject_layers = list(inject_layers)
        self.attns = nn.ModuleDict()
        for li in self.inject_layers:
            self.attns[str(li)] = GapFillCrossAttn(hidden_dim, attn_dim,
                                                     use_rezero=use_rezero)

    def forward(self, layer_idx: int, h_virt: torch.Tensor,
                h_selected: torch.Tensor) -> torch.Tensor:
        key = str(layer_idx)
        if key not in self.attns:
            return h_virt
        return self.attns[key](h_virt, h_selected)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
