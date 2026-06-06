"""Memory accounting for the AstroNet memory pivot.

Critic A1 (2026-06-06 ruthless review): "Your n=300 virtual+real is not a memory
saving --- it is a parameter-count obfuscation. When you claim 'AstroHybrid at
k=200 matches SnapKV at k=300', you are silently spending megabytes of trained
Stage-2 parameters that SnapKV does not." This file is the response: ALL memory
accounting for the Pareto / iso-accuracy claims goes through here, and includes
both (i) per-request KV bytes and (ii) amortised Stage-2 parameter bytes per
inference at a given session length.

Usage:
    from baselines.memory_accounting import total_bytes_per_request

    bytes_at_4k  = total_bytes_per_request("qwen2.5-7b",
                                            method="astrohybrid",
                                            k=300, dtype="fp16",
                                            session_length=4096)
    bytes_at_16k = total_bytes_per_request("qwen2.5-7b",
                                            method="astrohybrid",
                                            k=300, dtype="fp16",
                                            session_length=16384)

The split (KV bytes vs amortised Stage-2 bytes) is preserved so reviewers can
audit each component independently.  The function never silently swaps units;
returns explicit (kv_bytes, params_amortised_bytes, total_bytes).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

# Backbone geometry --- copy of the ARCH table from baselines/generate_pareto.py
# kept in sync manually.  See `paper_npjai/AUDIT.md` for the source-of-truth
# verification that these numbers match the HuggingFace configs of the
# checkpoints in `./models/`.
ARCH = {
    "qwen2.5-7b":         {"n_layers": 28, "n_kv_heads": 4,  "head_dim": 128, "hidden_dim": 3584},
    "qwen2.5-14b":        {"n_layers": 48, "n_kv_heads": 8,  "head_dim": 128, "hidden_dim": 5120},
    "qwen2.5-32b":        {"n_layers": 64, "n_kv_heads": 8,  "head_dim": 128, "hidden_dim": 5120},
    "llama-3.1-8b":       {"n_layers": 32, "n_kv_heads": 8,  "head_dim": 128, "hidden_dim": 4096},
    "mistral-7b-v0.3":    {"n_layers": 32, "n_kv_heads": 8,  "head_dim": 128, "hidden_dim": 4096},
    "mistral-small-24b":  {"n_layers": 40, "n_kv_heads": 8,  "head_dim": 128, "hidden_dim": 5120},
}

# Stage-2 parameter count per backbone, in millions.  Source:
# `astro.parameter_count()` printed at the head of every Stage-2 training run.
# These are the FP16 params (the AstroHybrid module is trained in FP16); the
# byte cost below assumes FP16 storage at deploy time.
STAGE2_PARAMS_M = {
    "qwen2.5-7b":         14.0,
    "qwen2.5-14b":        18.0,
    "qwen2.5-32b":        18.0,
    "llama-3.1-8b":       16.0,
    "mistral-7b-v0.3":    16.0,
    "mistral-small-24b":  22.0,
}

# Per-token KV bytes by dtype.  Cross-check against generate_pareto.py:25-32.
def _per_token_per_layer_bytes(arch: dict, dtype: str) -> float:
    """K+V bytes per token per layer at the given precision."""
    d = arch["n_kv_heads"] * arch["head_dim"]
    return {
        "fp16":  2 * d * 2,         # 2 bytes/elem for K, 2 for V
        "k8v4":  1.0 * d + 0.5 * d, # K=8b=1 byte, V=4b=0.5 byte (excludes scales; see note)
        "k4v4":  0.5 * d + 0.5 * d, # K=V=4b
    }[dtype]

# Per-method "what counts as a slot" --- a virtual KV slot occupies the same
# K/V tensor footprint as a real KV slot at the same precision.  AstroNet's
# 16 learned virtual tokens are 16 slots; the total cache budget k counts
# real + virtual.
METHODS_WITH_VIRTUAL = {"astrohybrid"}   # add new ones here if/when we have them


@dataclass
class MemoryBreakdown:
    kv_bytes: int               # per-request KV cache, K + V
    params_amortised_bytes: int # Stage-2 params amortised at session_length
    total_bytes: int            # sum
    components: dict            # transparent dict of contributions for audit

    def __str__(self) -> str:
        kb = self.kv_bytes / 1024
        pb = self.params_amortised_bytes / 1024
        tb = self.total_bytes / 1024
        return (f"KV={kb:.1f} KiB  +  params/sess={pb:.1f} KiB  "
                f"=  total={tb:.1f} KiB")


def total_bytes_per_request(model: str,
                            method: str,
                            k: int,
                            dtype: str = "fp16",
                            session_length: int = 4096,
                            include_quant_metadata: bool = True
                            ) -> MemoryBreakdown:
    """Total per-request KV memory.

    Parameters
    ----------
    model : one of the keys in ``ARCH``
    method : one of {"astrohybrid", "snapkv", "h2o", "pyramidkv",
                     "streamingllm", "kivi", "rag", "stage1_only", "full"}
        Anything in ``METHODS_WITH_VIRTUAL`` pays the Stage-2 amortised cost;
        all others are KV-only.  ``"full"`` ignores k and uses the full cache.
    k : total cache budget (real + virtual for methods with virtual slots);
        for ``"full"`` this is unused.
    dtype : "fp16" | "k8v4" | "k4v4"
    session_length : amortisation horizon in tokens.  At session_length=oo the
        Stage-2 params are free (asymptote).  At session_length=k the
        amortisation is over a single full cache.
    include_quant_metadata : if True, adds a rough 5% overhead for per-(layer,
        head) scales/shifts that K8V4 / K4V4 need to store.  Negligible in
        practice but reviewers will ask.
    """
    if model not in ARCH:
        raise ValueError(f"unknown model {model}; known: {list(ARCH)}")
    a = ARCH[model]

    # --- 1) KV bytes ---
    per_tok_per_layer = _per_token_per_layer_bytes(a, dtype)
    if method == "full":
        # full cache --- caller responsibility to set k to n_tokens upstream;
        # we treat k as n_tokens here.
        n_slots = k
    else:
        n_slots = k
    kv_bytes = int(n_slots * a["n_layers"] * per_tok_per_layer)
    if include_quant_metadata and dtype != "fp16":
        # 5% overhead for codebook + per-(layer, head) scales.  Conservative;
        # actual is closer to 1-2% for Lloyd-Max with shared per-head codebooks.
        kv_bytes = int(kv_bytes * 1.05)

    # --- 2) Amortised Stage-2 parameter bytes ---
    if method in METHODS_WITH_VIRTUAL:
        params_bytes = int(STAGE2_PARAMS_M[model] * 1e6 * 2)  # FP16, 2 B/param
        # Per-request amortisation: each session processes `session_length`
        # tokens; the params are loaded once and serve every request in the
        # session.  We report bytes/request at the given session_length.
        params_amortised = params_bytes // max(1, session_length // n_slots)
    else:
        params_amortised = 0

    total = kv_bytes + params_amortised
    return MemoryBreakdown(
        kv_bytes=kv_bytes,
        params_amortised_bytes=params_amortised,
        total_bytes=total,
        components=dict(model=model, method=method, k=k, dtype=dtype,
                         session_length=session_length,
                         per_token_per_layer_bytes=per_tok_per_layer,
                         n_layers=a["n_layers"],
                         stage2_params_M=STAGE2_PARAMS_M.get(model, 0.0)
                         if method in METHODS_WITH_VIRTUAL else 0.0))


def iso_accuracy_crossover(baseline_k_to_acc: dict,
                           astrohybrid_k_to_acc: dict,
                           target_acc: float,
                           tol_pp: float = 1.0
                           ) -> Optional[int]:
    """Given two dictionaries {k: accuracy}, find the smallest k_AstroHybrid
    that matches a target accuracy.

    Returns the smallest budget k at which AstroHybrid >= ``target_acc - tol_pp``;
    None if no such budget is in the grid.  ``target_acc`` is typically the
    best baseline at k=300 (the parity anchor).
    """
    for k in sorted(astrohybrid_k_to_acc):
        if astrohybrid_k_to_acc[k] >= target_acc - tol_pp / 100.0:
            return k
    return None


# Quick self-check
if __name__ == "__main__":
    for sess in (4096, 16384):
        print(f"\n=== session_length = {sess} tokens ===")
        for method in ("snapkv", "astrohybrid"):
            for dtype in ("fp16", "k8v4"):
                m = total_bytes_per_request("qwen2.5-7b", method, k=300,
                                            dtype=dtype, session_length=sess)
                print(f"  qwen2.5-7b  {method:12s}  k=300  {dtype}: {m}")
