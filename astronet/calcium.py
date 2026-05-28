"""
Astrocytic calcium dynamics modules.

Phase 0: AstroStateV0 — single-timescale Ca2+ state with EMA integration.
         AstroStateV1 — attention-pooled sensing + multi-token memory.
Phase 1: CalciumChannel, AstroNetV1 — multi-timescale with surprise gating.

Biology:
  - g = intracellular Ca2+ concentration (persistent state)
  - alpha = Ca2+ uptake rate (how fast new info is absorbed)
  - W_sense = astrocyte process sensitivity (what signals to detect)
  - W_feedback = gliotransmitter release (how to modulate neural activity)
  - lambda_mod = gliotransmitter release strength
  - Sigmoid = Ca2+ concentration bounds (buffering capacity)
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class AstroStateV0(nn.Module):
    """
    Minimal astrocytic working memory state for Phase 0.

    Wraps around a frozen LLM's attention layer. Maintains a persistent
    state vector g that integrates hidden states via exponential moving
    average and feeds back as a bias on attention key projections.

    Dynamics:
        sense:  s(t) = sigmoid(W_sense @ mean_pool(h(t)))
        update: g(t+1) = (1 - alpha) * g(t) + alpha * s(t)
        feed:   bias = lambda * W_feedback(g(t))

    Args:
        hidden_dim: Dimension of the LLM's hidden states (e.g. 4096 for Llama 8B)
        state_dim: Dimension of the astrocytic state vector (compression)
        target_layer: Which transformer layer to modulate (0-indexed)
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: Optional[int] = None,
        target_layer: int = 15,
        target_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim or hidden_dim
        self.target_layer = target_layer
        self.target_scale = target_scale

        # Sensing: project hidden states into astrocytic space
        # Analogous to astrocyte processes detecting neurotransmitter release
        self.W_sense = nn.Linear(hidden_dim, self.state_dim, bias=False)

        # Feedback: project astrocytic state back to hidden-dim space
        # Analogous to gliotransmitter release modulating synaptic efficacy
        self.W_feedback = nn.Linear(self.state_dim, hidden_dim, bias=False)

        # Learnable dynamics parameters (stored in log-space for unconstrained optimization)
        # alpha ~ 0.5 initial uptake rate (balanced integration/decay)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        # lambda ~ 0.5 initial modulation strength
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

        # Persistent state: NOT a parameter — survives across forward passes
        # but is NOT updated by the optimizer. This IS the "working memory."
        self.register_buffer('g', torch.zeros(1, self.state_dim))

        # Tanh: bounded [-1,1], gradient=1 at origin (unlike sigmoid's 0.25)
        # Allows negative state values while keeping bounded range
        self.activation = nn.Tanh()

        # Initialize projections
        # W_sense: larger gain so different inputs produce distinguishable states
        nn.init.xavier_uniform_(self.W_sense.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_feedback.weight, gain=0.1)

    @property
    def alpha(self) -> torch.Tensor:
        """Ca2+ uptake rate, constrained to (0, 1)."""
        return torch.sigmoid(self.log_alpha)

    @property
    def lambda_mod(self) -> torch.Tensor:
        """Gliotransmitter modulation strength, constrained to (0, 1)."""
        return torch.sigmoid(self.log_lambda)

    def sense(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Sense neural activity: project hidden states into astrocytic space.

        Mean-pools across the sequence dimension (astrocyte integrates over
        many synapses simultaneously) then projects and saturates.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) from target layer
        Returns:
            sensed signal: (batch, state_dim)
        """
        # Cast to float32 for AstroNet computation (model may output float16)
        h_mean = hidden_states.float().mean(dim=1)  # (batch, hidden_dim)
        return self.activation(self.W_sense(h_mean))  # (batch, state_dim)

    def update_state(self, sensed: torch.Tensor, keep_grad: bool = False) -> None:
        """
        Update Ca2+ state with exponential integration.

        g(t+1) = (1 - alpha) * g(t) + alpha * sensed(t)

        This is the core "slow timescale" dynamic. The state exponentially
        decays old information while integrating new signals. With alpha ~ 0.13,
        information half-life is about 5 context windows.

        Args:
            sensed: (batch, state_dim) — output of self.sense()
            keep_grad: If True, keep gradients on the state (for truncated BPTT).
                       Set True for the last N windows to allow W_sense/alpha to learn.
        """
        alpha = self.alpha
        g_expanded = self.g.expand(sensed.shape[0], -1)
        new_g = (1 - alpha) * g_expanded + alpha * sensed
        # Store mean across batch (single persistent state vector)
        new_g_mean = new_g.mean(dim=0, keepdim=True)
        if keep_grad:
            self.g = new_g_mean  # keep gradients for BPTT
        else:
            self.g = new_g_mean.detach()  # detach to prevent unbounded graph growth

    def get_feedback(self) -> torch.Tensor:
        """
        Compute the feedback bias to inject into the LLM (legacy hook method).

        Returns:
            feedback: (1, hidden_dim) — scaled bias vector
        """
        raw = self.W_feedback(self.g)  # (1, hidden_dim)
        normalized = raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)
        return self.lambda_mod * normalized * self.target_scale

    def get_memory_embedding(self) -> torch.Tensor:
        """
        Project the astrocytic state into a virtual token embedding.

        This embedding is prepended to the query window's input, allowing
        the model's attention at ALL layers to attend to the memory state.
        Much more effective than single-layer additive bias.

        Returns:
            mem_embed: (1, 1, hidden_dim) — virtual token embedding
        """
        embed = self.W_feedback(self.g)  # (1, hidden_dim)
        return embed.unsqueeze(1)  # (1, 1, hidden_dim)

    def reset_state(self) -> None:
        """Reset Ca2+ to baseline (start of new session/sequence)."""
        self.g.zero_()

    def get_state_norm(self) -> float:
        """Get the L2 norm of the current state (for monitoring)."""
        return self.g.norm().item()

    def get_state_stats(self) -> dict:
        """Get diagnostic statistics about the current state."""
        return {
            'g_norm': self.g.norm().item(),
            'g_mean': self.g.mean().item(),
            'g_std': self.g.std().item(),
            'g_min': self.g.min().item(),
            'g_max': self.g.max().item(),
            'alpha': self.alpha.item(),
            'lambda': self.lambda_mod.item(),
        }

    def parameter_count(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, state_dim={self.state_dim}, "
            f"target_layer={self.target_layer}, "
            f"params={self.parameter_count():,}"
        )


class AstroStateV1(nn.Module):
    """
    Cross-attention memory encoder with multi-token output.

    Informed by ICAE, Perceiver IO, and soft prompt tuning literature.

    Architecture:
      1. K learnable queries cross-attend over hidden states (low-rank keys)
      2. Cross-attention outputs are transformed via low-rank projection
      3. Result: K virtual tokens in embedding space

    Key design choices:
      - Low-rank key projection (attn_dim << hidden_dim) keeps params small
      - Output projection transforms weighted averages into embedding space
      - High initial alpha (0.88) for near-direct state assignment
      - For 2-window POC, use set_state_direct() to bypass EMA entirely
    """

    def __init__(
        self,
        hidden_dim: int,
        n_mem_tokens: int = 4,
        target_layer: int = 15,
        attn_dim: int = 256,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mem_tokens = n_mem_tokens
        self.target_layer = target_layer

        # Cross-attention: K queries attend over sequence
        self.queries = nn.Parameter(torch.randn(1, n_mem_tokens, attn_dim) * 0.02)
        self.key_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.attn_scale = 1.0 / math.sqrt(attn_dim)

        # Layer norm on hidden states before cross-attention (model-agnostic scaling)
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Low-rank output projection: hidden_dim -> proj_dim -> hidden_dim
        # Cross-attention outputs are weighted averages of normed hidden states.
        # This projection compresses and re-expands, controlling output scale
        # via initialization gain. No hard norm constraints — gradients flow freely.
        self.out_down = nn.Linear(hidden_dim, proj_dim, bias=False)
        self.out_up = nn.Linear(proj_dim, hidden_dim, bias=False)

        # EMA rate (high initial = near-direct assignment)
        self.log_alpha = nn.Parameter(torch.tensor(2.0))  # alpha ≈ 0.88

        # Persistent state: K vectors in hidden_dim space
        self.register_buffer('g', torch.zeros(1, n_mem_tokens, hidden_dim))

        # Initialization — scale out_up so initial output norms ≈ 5-10
        # (larger than embedding norm 0.58, so model attention pays attention)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_down.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_up.weight, gain=1.0)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha)

    def sense(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention pooling: K queries attend over sequence positions.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            sensed: (batch, n_mem_tokens, hidden_dim)
        """
        h = self.input_norm(hidden_states.float())  # (B, S, D) — normalized
        B = h.shape[0]
        keys = self.key_proj(h)  # (B, S, attn_dim)

        # (B, K, attn_dim) x (B, attn_dim, S) -> (B, K, S)
        attn_scores = torch.bmm(
            self.queries.expand(B, -1, -1),
            keys.transpose(1, 2),
        ) * self.attn_scale
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, K, S)

        # (B, K, S) x (B, S, D) -> (B, K, D)
        pooled = torch.bmm(attn_weights, h)
        return pooled

    def update_state(self, sensed: torch.Tensor, keep_grad: bool = False) -> None:
        """EMA update of the state (K vectors in hidden_dim)."""
        alpha = self.alpha
        g_expanded = self.g.expand(sensed.shape[0], -1, -1)
        new_g = (1 - alpha) * g_expanded + alpha * sensed
        new_g_mean = new_g.mean(dim=0, keepdim=True)
        if keep_grad:
            self.g = new_g_mean
        else:
            self.g = new_g_mean.detach()

    def set_state_direct(self, sensed: torch.Tensor, keep_grad: bool = True) -> None:
        """Directly set state without EMA (for 2-window / single-fact case)."""
        new_g = sensed.mean(dim=0, keepdim=True)
        if keep_grad:
            self.g = new_g
        else:
            self.g = new_g.detach()

    def get_memory_embedding(self, target_norm: float = 0.0) -> torch.Tensor:
        """
        Transform state into memory token embeddings via low-rank projection.

        Args:
            target_norm: If > 0, normalize output tokens to this L2 norm.
                         Set to model's typical embedding norm for compatibility.
                         If 0, no normalization (controlled by soft regularization).

        Returns:
            mem_embeds: (1, n_mem_tokens, hidden_dim)
        """
        mem = self.out_up(self.out_down(self.g))
        if target_norm > 0:
            mem = mem * (target_norm / (mem.norm(dim=-1, keepdim=True) + 1e-8))
        return mem

    def reset_state(self) -> None:
        self.g.zero_()

    def get_state_norm(self) -> float:
        return self.g.norm().item()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"n_mem_tokens={self.n_mem_tokens}, "
            f"target_layer={self.target_layer}, "
            f"params={self.parameter_count():,}"
        )


class KVInjectionHead(nn.Module):
    """Projects astrocyte memory state to per-layer Key-Value pairs for
    attention-level injection (tripartite synapse model).

    Instead of perturbing the residual stream (additive/gain), this module
    produces Key and Value vectors that get injected into the attention
    mechanism via past_key_value cache.  The model's attention naturally
    decides how much to attend to memory tokens vs. real tokens.

    Biologically: the astrocyte provides extra synaptic inputs at the
    tripartite synapse.  The synaptic weight (attention score) is
    determined by the model's own Q projections — the astrocyte doesn't
    force a fixed signal, it offers information that the neuron can
    choose to use.

    Args:
        hidden_dim: Model hidden dimension
        n_kv_heads: Number of KV heads in the model (GQA)
        head_dim: Dimension per attention head
        inject_layers: List of layer indices to inject at
        n_mem_kv: Number of memory KV pairs to inject per layer
        bottleneck_dim: Bottleneck for projection
    """

    def __init__(self, hidden_dim, n_kv_heads, head_dim, inject_layers,
                 n_mem_kv=4, bottleneck_dim=256):
        super().__init__()
        self.inject_layers = inject_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_mem_kv = n_mem_kv
        kv_dim = n_kv_heads * head_dim  # total KV dimension

        # Per-layer projections: memory_state → K and V
        self.k_projections = nn.ModuleDict()
        self.v_projections = nn.ModuleDict()
        for layer_idx in inject_layers:
            self.k_projections[str(layer_idx)] = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim, bias=False),
                nn.Linear(bottleneck_dim, kv_dim * n_mem_kv, bias=False),
            )
            self.v_projections[str(layer_idx)] = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim, bias=False),
                nn.Linear(bottleneck_dim, kv_dim * n_mem_kv, bias=False),
            )

    def forward(self, mem_state):
        """Project memory state to per-layer K, V pairs.

        Args:
            mem_state: (batch, n_mem_tokens, hidden_dim) — AstroNet state

        Returns:
            dict mapping layer_idx → (key, value) tensors
                key: (batch, n_kv_heads, n_mem_kv, head_dim)
                value: (batch, n_kv_heads, n_mem_kv, head_dim)
        """
        # Pool memory tokens
        pooled = mem_state.mean(dim=1)  # (batch, hidden_dim)
        kv_pairs = {}
        for layer_idx in self.inject_layers:
            k = self.k_projections[str(layer_idx)](pooled)
            v = self.v_projections[str(layer_idx)](pooled)
            # Reshape to (batch, n_kv_heads, n_mem_kv, head_dim)
            B = pooled.shape[0]
            k = k.view(B, self.n_mem_kv, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, self.n_mem_kv, self.n_kv_heads, self.head_dim).transpose(1, 2)
            kv_pairs[layer_idx] = (k, v)
        return kv_pairs


class AstroStateV2(AstroStateV1):
    """Astrocyte state with gated memory update (Ca2+ microdomain model).

    Extends V1 with per-dimension forget/input gates, inspired by the
    observation that different astrocytic compartments (soma, processes,
    microdomains) have distinct Ca2+ time constants.  Fast microdomains
    capture recent events; slow somatic Ca2+ retains older information.

    The update rule replaces the scalar EMA with:
        f = sigmoid(W_f · [g, s] + b_f)     # forget gate (what to keep)
        i = sigmoid(W_i · [g, s] + b_i)     # input gate (what to update)
        g' = f * g + i * s                   # new state

    This allows per-dimension selective retention vs. overwrite, resolving
    the alpha-collapse problem where a single scalar cannot simultaneously
    retain old facts and incorporate new ones.

    Biologically: IP3R-mediated Ca2+ release has different activation
    thresholds in different compartments.  The forget/input gates model
    this heterogeneity — some dimensions retain (slow somatic Ca2+),
    others update (fast microdomain Ca2+).
    """

    def __init__(
        self,
        hidden_dim: int,
        n_mem_tokens: int = 4,
        target_layer: int = 15,
        attn_dim: int = 256,
        proj_dim: int = 256,
    ):
        super().__init__(hidden_dim, n_mem_tokens, target_layer, attn_dim, proj_dim)

        # Per-dimension gates operating on each memory token independently
        # Input: concatenation of [g, sensed] along last dim → 2*hidden_dim
        self.gate_f = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_i = nn.Linear(hidden_dim * 2, hidden_dim)

        # Forget gate bias +2: sigmoid(2) ≈ 0.88 → retain ~88% by default
        nn.init.zeros_(self.gate_f.weight)
        self.gate_f.bias.data.fill_(2.0)
        # Input gate bias -2: sigmoid(-2) ≈ 0.12 → write ~12% by default
        nn.init.zeros_(self.gate_i.weight)
        self.gate_i.bias.data.fill_(-2.0)

    @property
    def alpha(self) -> torch.Tensor:
        # For logging compatibility — return mean input gate activation
        # (not actually used in update_state)
        return torch.sigmoid(self.gate_i.bias.data.mean())

    def update_state(self, sensed: torch.Tensor, keep_grad: bool = False) -> None:
        """Gated update: per-dimension forget/input gates (Ca2+ microdomain model)."""
        g_expanded = self.g.expand(sensed.shape[0], -1, -1)
        combined = torch.cat([g_expanded, sensed], dim=-1)  # (B, K, 2*D)
        f = torch.sigmoid(self.gate_f(combined))  # forget gate
        i = torch.sigmoid(self.gate_i(combined))  # input gate
        new_g = f * g_expanded + i * sensed
        new_g_mean = new_g.mean(dim=0, keepdim=True)
        if keep_grad:
            self.g = new_g_mean
        else:
            self.g = new_g_mean.detach()
