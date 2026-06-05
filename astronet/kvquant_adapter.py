"""Faithful KVQuant K4 adapter for AstroNet's serving pipeline.

KVQuant (Hooper et al., NeurIPS 2024) is a KV-cache quantisation scheme
that combines three innovations to push KV memory down to 4 bits per
scalar with sub-PPL accuracy loss on Llama-class backbones:

    1. Per-channel pre-RoPE key quantisation.  Each head_dim channel of
       the K projection output (BEFORE RoPE is applied) gets its own
       scale and 4-bit non-uniform codebook.
    2. Per-token V quantisation.  Each (layer, head, token) position
       gets its own scale, with a 16-entry codebook shared across the
       head_dim dimension (per-head in our adapter).
    3. Dense-and-Sparse outlier extraction.  Values outside the symmetric
       outer percentile range (e.g. lower 0.5% and upper 0.5% with
       --sparsity 0.01) are passed through unchanged in fp16; the dense
       99% middle is quantised to the 16-entry NUQ codebook.  Plus the
       first few tokens are unconditionally kept in fp16
       (attention-sink-aware quantisation).

Reference: github.com/SqueezeAILab/KVQuant; arXiv:2401.18079.

Our adapter is independent of the reference codebase: we implement the
algorithm from scratch, mirroring quant/kvquant/simquant_module_quantizer.py
but using simpler bookkeeping suited to AstroNet's serving wrapper
(bitsandbytes 4-bit backbones via astronet.wrapper.AstroWrapper).

Calibration strategy.
  We collect k_proj and v_proj output statistics by attaching forward
  hooks while running representative SQuAD multi-segment instances
  through the backbone.  For each (layer, head) we fit a 16-entry
  non-uniform codebook via 1-D k-means on the dense 99% portion (after
  removing the top 1% of |value| as outliers).  The codebook for K is
  per-channel (one codebook per head_dim channel); for V it is shared
  across the head_dim channels.

  We do NOT compute Fisher information weights.  The reference
  implementation uses Fisher weighting to upweight high-sensitivity
  positions during k-means, which improves perplexity by ~0.1 in their
  setting.  We document this deviation in the paper.

Quantisation at inference.
  After calibration, we replace each layer's self_attn.k_proj and
  self_attn.v_proj with KVQuantWrap modules that (a) forward through
  the original (possibly 4-bit bitsandbytes) linear, (b) quantise and
  immediately dequantise the output using the calibrated codebook and
  outlier mask.  The downstream attention path sees fp16 tensors with
  K4V4-equivalent distortion baked in.

  This is "simulated quantisation": memory savings are not realised
  during the experiment, but the accuracy effect matches what a real
  K4V4-packed cache would produce.  This is the same protocol used by
  KVQuant's llama_simquant.py.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
#  Core quantisation primitives
# ----------------------------------------------------------------------

def _kmeans_1d(values: np.ndarray, n_levels: int, max_iter: int = 50,
               seed: int = 42) -> np.ndarray:
    """Plain 1-D k-means; deterministic init by quantile spacing.

    Returns the sorted codebook centres of length n_levels.  We use
    quantile-spaced init rather than sklearn KMeans to avoid the heavy
    dependency and to keep the routine pure-numpy."""
    if values.size == 0:
        # degenerate case: return symmetric levels at zero
        return np.linspace(-1, 1, n_levels, dtype=np.float32)

    # initialise centres at evenly spaced quantiles of the data
    quantiles = np.linspace(0.0, 1.0, n_levels + 2, dtype=np.float64)[1:-1]
    centres = np.quantile(values, quantiles).astype(np.float64)

    # ensure centres are distinct (avoids divide-by-zero on flat regions)
    eps = 1e-6 * (values.std() + 1e-9)
    for i in range(1, len(centres)):
        if centres[i] <= centres[i - 1]:
            centres[i] = centres[i - 1] + eps

    for _ in range(max_iter):
        # assignment: each value to nearest centre
        d2 = (values[:, None] - centres[None, :]) ** 2
        assign = d2.argmin(axis=1)
        # update: mean of assigned values per centre
        new_centres = centres.copy()
        moved = False
        for i in range(n_levels):
            mask = assign == i
            if mask.any():
                m = values[mask].mean()
                if abs(m - centres[i]) > 1e-7:
                    moved = True
                new_centres[i] = m
        centres = new_centres
        if not moved:
            break

    # final sort
    centres.sort()
    return centres.astype(np.float32)


def _quantize_dequantize_to_codebook(
    x: torch.Tensor,
    centres: torch.Tensor,
    outlier_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Round x to nearest codebook centre; preserve outliers in fp16.

    x: (..., n_quantised_axis) tensor of fp16/fp32 values
    centres: (n_levels,) tensor of codebook centres (same shape as the
             quantised axis broadcasted)
    outlier_mask: boolean tensor same shape as x.  True positions are
                  passed through unchanged.

    Returns the quantise-then-dequantise output."""
    # nearest centre per element (broadcast over the leading axes)
    # centres may be (n_channels, n_levels) for per-channel quantisation
    # or (n_levels,) for global.  We handle both by broadcasting.
    if centres.dim() == 1:
        # global codebook: (..., n) -> (..., n, 1) - (1,..,1,L) -> (..,n,L)
        d2 = (x.unsqueeze(-1) - centres) ** 2  # (..., n, L)
    else:
        # per-channel codebook: x shape (..., C); centres shape (C, L)
        # broadcast to (..., C, L)
        d2 = (x.unsqueeze(-1) - centres) ** 2

    idx = d2.argmin(dim=-1)  # (..., C)

    if centres.dim() == 1:
        recon = centres[idx]
    else:
        # gather per channel: centres has shape (C, L); idx has shape (..., C)
        # We need to index into centres[c, idx[..., c]]
        # Use torch.gather along the last dim of centres after broadcasting
        # Easiest: expand centres to (..., C, L) and gather
        expanded_centres = centres.expand(*idx.shape, centres.size(-1))
        recon = expanded_centres.gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    if outlier_mask is not None:
        recon = torch.where(outlier_mask, x, recon)

    return recon


# ----------------------------------------------------------------------
#  Calibration: per-layer codebooks fitted from observed K, V tensors
# ----------------------------------------------------------------------

@dataclass
class LayerQuantizer:
    """Calibrated KVQuant codebook for one transformer layer.

    K: per-channel non-uniform 4-bit (one codebook per head_dim channel,
       outlier_threshold per channel).
    V: per-token-group non-uniform 4-bit (codebook shared across head_dim,
       outlier_threshold per token-group).

    Per-token in V is implemented as a single shared codebook across all
    head_dim channels of V, with per-token scaling normalised by the
    token-level rms.  This matches the KVQuant convention where v_proj
    uses qchannel=-1 (last-dim quantisation)."""

    n_kv_heads: int
    head_dim: int
    # K side: (n_kv_heads, head_dim, n_levels) codebooks
    k_centres: torch.Tensor = field(default=None)
    # K outlier thresholds (per kv_head per channel, upper and lower)
    k_outlier_lo: torch.Tensor = field(default=None)  # (n_kv_heads, head_dim)
    k_outlier_hi: torch.Tensor = field(default=None)
    # V side: (n_kv_heads, n_levels) shared across head_dim
    v_centres: torch.Tensor = field(default=None)
    v_outlier_lo: torch.Tensor = field(default=None)  # (n_kv_heads,)
    v_outlier_hi: torch.Tensor = field(default=None)


def calibrate_layer(
    k_samples: torch.Tensor,
    v_samples: torch.Tensor,
    n_kv_heads: int,
    head_dim: int,
    n_levels: int = 16,
    sparsity: float = 0.01,
    seed: int = 42,
) -> LayerQuantizer:
    """Fit a per-layer KVQuant codebook from observed K, V tensors.

    k_samples: (n_total_tokens, n_kv_heads * head_dim) - pre-RoPE K output
    v_samples: (n_total_tokens, n_kv_heads * head_dim) - V output
    n_levels: codebook size (16 for 4-bit)
    sparsity: outlier fraction (0.01 = top 1% in magnitude treated as outliers)"""
    assert k_samples.dim() == 2 and v_samples.dim() == 2
    assert k_samples.size(1) == n_kv_heads * head_dim

    # Reshape to (n_tokens, n_kv_heads, head_dim) so we calibrate per head
    K = k_samples.reshape(-1, n_kv_heads, head_dim).float().cpu()
    V = v_samples.reshape(-1, n_kv_heads, head_dim).float().cpu()

    n_tokens = K.size(0)
    quantizer = LayerQuantizer(n_kv_heads=n_kv_heads, head_dim=head_dim)

    # --- K: per-channel codebook ---
    quantizer.k_centres = torch.zeros(n_kv_heads, head_dim, n_levels)
    quantizer.k_outlier_lo = torch.zeros(n_kv_heads, head_dim)
    quantizer.k_outlier_hi = torch.zeros(n_kv_heads, head_dim)

    for h in range(n_kv_heads):
        for c in range(head_dim):
            vals = K[:, h, c].numpy()
            # outlier thresholds: symmetric percentile
            if sparsity > 0 and n_tokens > 100:
                lo = float(np.quantile(vals, sparsity / 2.0))
                hi = float(np.quantile(vals, 1.0 - sparsity / 2.0))
                dense = vals[(vals >= lo) & (vals <= hi)]
            else:
                lo, hi = float(vals.min() - 1), float(vals.max() + 1)
                dense = vals
            quantizer.k_outlier_lo[h, c] = lo
            quantizer.k_outlier_hi[h, c] = hi
            centres = _kmeans_1d(dense, n_levels, seed=seed + h * head_dim + c)
            quantizer.k_centres[h, c] = torch.from_numpy(centres)

    # --- V: per-head codebook shared across head_dim ---
    # KVQuant's V quantisation is "per-token" meaning each token gets its
    # own scale.  We implement this via a shared per-head codebook and an
    # explicit per-token rms normalisation applied at inference time.
    quantizer.v_centres = torch.zeros(n_kv_heads, n_levels)
    quantizer.v_outlier_lo = torch.zeros(n_kv_heads)
    quantizer.v_outlier_hi = torch.zeros(n_kv_heads)

    for h in range(n_kv_heads):
        # Normalise each token by its rms; then fit codebook to normalised dist.
        v_h = V[:, h, :]  # (n_tokens, head_dim)
        rms = v_h.float().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        v_norm = (v_h / rms).flatten().numpy()
        if sparsity > 0 and v_norm.size > 100:
            lo = float(np.quantile(v_norm, sparsity / 2.0))
            hi = float(np.quantile(v_norm, 1.0 - sparsity / 2.0))
            dense = v_norm[(v_norm >= lo) & (v_norm <= hi)]
        else:
            lo, hi = float(v_norm.min() - 1), float(v_norm.max() + 1)
            dense = v_norm
        quantizer.v_outlier_lo[h] = lo
        quantizer.v_outlier_hi[h] = hi
        centres = _kmeans_1d(dense, n_levels, seed=seed + 10000 + h)
        quantizer.v_centres[h] = torch.from_numpy(centres)

    return quantizer


# ----------------------------------------------------------------------
#  Inference-time quantise-dequantise application
# ----------------------------------------------------------------------

def apply_kvquant_to_K(
    K: torch.Tensor,  # (batch, n_kv_heads, seq, head_dim)
    q: LayerQuantizer,
    first_few_fp16: int = 4,
) -> torch.Tensor:
    """Quantise-dequantise K per channel per head with outlier preservation.

    K shape conventions follow PyTorch standard: (batch, n_kv_heads, seq, head_dim).
    The codebook is per (head, channel) so we broadcast over batch and seq."""
    assert K.dim() == 4
    B, H, T, D = K.shape
    assert H == q.n_kv_heads and D == q.head_dim
    dev = K.device
    dtype = K.dtype

    # move codebook to device once per call (cheap)
    centres = q.k_centres.to(dev, dtype=torch.float32)        # (H, D, L)
    lo = q.k_outlier_lo.to(dev, dtype=torch.float32)          # (H, D)
    hi = q.k_outlier_hi.to(dev, dtype=torch.float32)          # (H, D)

    K_f = K.to(torch.float32)
    # outlier mask: broadcast (H, D) -> (1, H, 1, D)
    om = (K_f < lo.view(1, H, 1, D)) | (K_f > hi.view(1, H, 1, D))
    # for the dense portion, round to nearest centre per (head, channel)
    # reshape to (B*T, H*D) -> (B*T, H, D) so we can use the (H, D, L) codebook
    flat = K_f.transpose(1, 2).reshape(B * T, H, D)   # (B*T, H, D)
    om_flat = om.transpose(1, 2).reshape(B * T, H, D)

    # per (h, d) nearest centre lookup
    expanded_centres = centres.unsqueeze(0).expand(B * T, H, D, centres.size(-1))
    d2 = (flat.unsqueeze(-1) - expanded_centres) ** 2
    idx = d2.argmin(dim=-1, keepdim=True)               # (B*T, H, D, 1)
    recon = expanded_centres.gather(-1, idx).squeeze(-1)  # (B*T, H, D)

    # outliers passed through
    recon = torch.where(om_flat, flat, recon)
    out = recon.view(B, T, H, D).transpose(1, 2).to(dtype)

    # attention sink: first few tokens kept in fp16
    if first_few_fp16 > 0 and T > 0:
        n_sink = min(first_few_fp16, T)
        out[:, :, :n_sink, :] = K[:, :, :n_sink, :]

    return out


def apply_kvquant_to_V(
    V: torch.Tensor,  # (batch, n_kv_heads, seq, head_dim)
    q: LayerQuantizer,
    first_few_fp16: int = 4,
) -> torch.Tensor:
    """Quantise-dequantise V per token per head with outlier preservation.

    For each (batch, head, token) we (a) compute the rms of the head_dim
    vector, (b) normalise to unit rms, (c) quantise via the per-head
    codebook, (d) denormalise by the rms.  This realises 'per-token'
    quantisation while keeping the codebook small."""
    assert V.dim() == 4
    B, H, T, D = V.shape
    assert H == q.n_kv_heads and D == q.head_dim
    dev = V.device
    dtype = V.dtype

    centres = q.v_centres.to(dev, dtype=torch.float32)        # (H, L)
    lo = q.v_outlier_lo.to(dev, dtype=torch.float32)          # (H,)
    hi = q.v_outlier_hi.to(dev, dtype=torch.float32)          # (H,)

    V_f = V.to(torch.float32)
    # per-token rms per head: (B, H, T, 1)
    rms = V_f.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    V_norm = V_f / rms

    # per-head codebook lookup; outliers detected in normalised space
    # om shape (B, H, T, D)
    om = (V_norm < lo.view(1, H, 1, 1)) | (V_norm > hi.view(1, H, 1, 1))

    # nearest centre per element using the (H, L) codebook
    # broadcast: V_norm (B, H, T, D, 1) - centres (1, H, 1, 1, L)
    d2 = (V_norm.unsqueeze(-1) - centres.view(1, H, 1, 1, -1)) ** 2
    idx = d2.argmin(dim=-1, keepdim=True)                     # (B, H, T, D, 1)
    expanded_centres = centres.view(1, H, 1, 1, -1).expand(B, H, T, D, centres.size(-1))
    recon_norm = expanded_centres.gather(-1, idx).squeeze(-1)  # (B, H, T, D)

    recon_norm = torch.where(om, V_norm, recon_norm)
    # denormalise
    recon = recon_norm * rms
    out = recon.to(dtype)

    if first_few_fp16 > 0 and T > 0:
        n_sink = min(first_few_fp16, T)
        out[:, :, :n_sink, :] = V[:, :, :n_sink, :]

    return out


# ----------------------------------------------------------------------
#  Calibration collector: forward-hook based statistics gathering
# ----------------------------------------------------------------------

class CalibrationCollector:
    """Attaches forward hooks to a backbone's k_proj and v_proj modules
    and accumulates their outputs.  After enough samples, calibrate_all()
    fits per-layer quantisers."""

    def __init__(self, model, n_kv_heads: int, head_dim: int, n_layers: int):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_layers = n_layers
        self.k_buffers: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
        self.v_buffers: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
        self._handles = []

        # register hooks
        for li in range(n_layers):
            layer = model.model.layers[li].self_attn
            self._handles.append(layer.k_proj.register_forward_hook(
                self._make_hook(self.k_buffers[li])))
            self._handles.append(layer.v_proj.register_forward_hook(
                self._make_hook(self.v_buffers[li])))

    def _make_hook(self, buf):
        def hook(_module, _inp, out):
            # out shape: (batch, seq, n_kv_heads * head_dim)
            # store to CPU to keep VRAM low
            buf.append(out.detach().to(torch.float16).cpu())
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def total_tokens(self) -> int:
        if not self.k_buffers[0]:
            return 0
        return sum(t.shape[0] * t.shape[1] for t in self.k_buffers[0])

    def calibrate_all(self, sparsity: float = 0.01, seed: int = 42) -> Dict[int, LayerQuantizer]:
        """Fit per-layer codebooks from the accumulated samples."""
        quantizers: Dict[int, LayerQuantizer] = {}
        for li in range(self.n_layers):
            ks = torch.cat([t.reshape(-1, t.shape[-1]) for t in self.k_buffers[li]], dim=0)
            vs = torch.cat([t.reshape(-1, t.shape[-1]) for t in self.v_buffers[li]], dim=0)
            quantizers[li] = calibrate_layer(
                ks, vs,
                n_kv_heads=self.n_kv_heads,
                head_dim=self.head_dim,
                sparsity=sparsity,
                seed=seed,
            )
        return quantizers


def save_quantizers(quantizers: Dict[int, LayerQuantizer], path: str):
    with open(path, 'wb') as f:
        pickle.dump(quantizers, f)


def load_quantizers(path: str) -> Dict[int, LayerQuantizer]:
    with open(path, 'rb') as f:
        return pickle.load(f)


# ----------------------------------------------------------------------
#  Cache-side quantisation: apply to (K, V) pairs produced by the model
# ----------------------------------------------------------------------

def quantise_cache_pair(
    K: torch.Tensor, V: torch.Tensor,
    quantizer: LayerQuantizer,
    first_few_fp16: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply KVQuant simulated quantisation to a (K, V) pair.

    K, V shape: (batch, n_kv_heads, seq, head_dim) -- standard HuggingFace
    past_key_values layout."""
    K_q = apply_kvquant_to_K(K, quantizer, first_few_fp16=first_few_fp16)
    V_q = apply_kvquant_to_V(V, quantizer, first_few_fp16=first_few_fp16)
    return K_q, V_q


# ----------------------------------------------------------------------
#  Pre-RoPE wrapping: install QuantLinearWrapper on backbone's
#  k_proj and v_proj so that quantisation happens inside the attention
#  forward, BEFORE RoPE is applied (matching KVQuant Hooper et al. 2024).
# ----------------------------------------------------------------------

class _KQuantWrapper(nn.Module):
    """Wraps a k_proj module so that the output is quantised pre-RoPE.

    Forward:  y = orig(x)                        # fp16 output of bnb Linear4bit
              y' = quantise-dequantise(y, q_k)   # per-channel NUQ + outliers
              return y'                          # downstream RoPE sees y'
    """
    def __init__(self, orig_linear: nn.Module, quantizer: 'LayerQuantizer',
                 first_few_fp16: int = 4):
        super().__init__()
        self.orig = orig_linear
        # store quantiser tensors as buffers so .to() moves them along
        self.register_buffer('k_centres', quantizer.k_centres)
        self.register_buffer('k_lo', quantizer.k_outlier_lo)
        self.register_buffer('k_hi', quantizer.k_outlier_hi)
        self.n_kv_heads = quantizer.n_kv_heads
        self.head_dim = quantizer.head_dim
        self.first_few_fp16 = first_few_fp16

    def forward(self, x):
        y = self.orig(x)                                          # (B, S, H*D)
        B = y.size(0); S = y.size(1)
        H, D = self.n_kv_heads, self.head_dim
        # reshape to (B, H, S, D) so apply_kvquant_to_K works
        y_resh = y.view(B, S, H, D).transpose(1, 2).contiguous()
        # build a transient quantizer object referencing our buffers
        _q = LayerQuantizer(n_kv_heads=H, head_dim=D,
                            k_centres=self.k_centres,
                            k_outlier_lo=self.k_lo,
                            k_outlier_hi=self.k_hi)
        yq = apply_kvquant_to_K(y_resh, _q, first_few_fp16=self.first_few_fp16)
        return yq.transpose(1, 2).contiguous().view(B, S, H * D)


class _VQuantWrapper(nn.Module):
    """Wraps a v_proj module so that the output is quantised at the
    projection stage (V has no RoPE so the timing does not matter
    numerically, but doing it here keeps the integration symmetric
    with the k_proj wrapper and means downstream code is unchanged)."""
    def __init__(self, orig_linear: nn.Module, quantizer: 'LayerQuantizer',
                 first_few_fp16: int = 4):
        super().__init__()
        self.orig = orig_linear
        self.register_buffer('v_centres', quantizer.v_centres)
        self.register_buffer('v_lo', quantizer.v_outlier_lo)
        self.register_buffer('v_hi', quantizer.v_outlier_hi)
        self.n_kv_heads = quantizer.n_kv_heads
        self.head_dim = quantizer.head_dim
        self.first_few_fp16 = first_few_fp16

    def forward(self, x):
        y = self.orig(x)
        B = y.size(0); S = y.size(1)
        H, D = self.n_kv_heads, self.head_dim
        y_resh = y.view(B, S, H, D).transpose(1, 2).contiguous()
        _q = LayerQuantizer(n_kv_heads=H, head_dim=D,
                            v_centres=self.v_centres,
                            v_outlier_lo=self.v_lo,
                            v_outlier_hi=self.v_hi)
        yq = apply_kvquant_to_V(y_resh, _q, first_few_fp16=self.first_few_fp16)
        return yq.transpose(1, 2).contiguous().view(B, S, H * D)


def install_kvquant_simquant(
    model,
    quantizers: Dict[int, LayerQuantizer],
    first_few_fp16: int = 4,
):
    """Wrap every layer's self_attn.k_proj and self_attn.v_proj with
    KVQuant simulated quantisation modules.  After this call, every
    forward pass through the model applies pre-RoPE K and per-token V
    quantisation in place.

    Returns a dict {layer_idx: (orig_k_proj, orig_v_proj)} that can be
    passed to uninstall_kvquant_simquant() to restore the model."""
    originals = {}
    for li, q in quantizers.items():
        layer = model.model.layers[li].self_attn
        originals[li] = (layer.k_proj, layer.v_proj)
        layer.k_proj = _KQuantWrapper(layer.k_proj, q, first_few_fp16=first_few_fp16)
        layer.v_proj = _VQuantWrapper(layer.v_proj, q, first_few_fp16=first_few_fp16)
    return originals


def uninstall_kvquant_simquant(model, originals: Dict[int, Tuple[nn.Module, nn.Module]]):
    """Restore original k_proj and v_proj modules on every wrapped layer."""
    for li, (orig_k, orig_v) in originals.items():
        layer = model.model.layers[li].self_attn
        layer.k_proj = orig_k
        layer.v_proj = orig_v
