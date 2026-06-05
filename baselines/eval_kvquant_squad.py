"""KVQuant K4 NUQ-1% evaluation on the AstroNet pos-robust SQuAD protocol.

Pipeline
========
1. Load a 4-bit NF4 quantised backbone (same path as other eval scripts).
2. Construct an AstroNetWrapper without any AstroNet Stage 2 module.
3. Calibrate per-layer KVQuant codebooks (per-channel K, per-head V,
   1% symmetric outlier extraction, 4 first-token fp16 attention-sink
   positions) by collecting k_proj/v_proj outputs from N SQuAD train
   samples processed through the wrapper's window pipeline.
4. Install pre-RoPE quantising wrappers (_KQuantWrapper, _VQuantWrapper)
   on every layer's self_attn.k_proj and self_attn.v_proj.
5. Run the position-robust SQuAD eval at each of the requested answer
   positions, using AstroNetWrapper.answer with method='mult' (Stage 1
   selection only, no AstroNet S2).
6. Save results to logs/results/kvquant_k4_<model_tag>.json.

CLI
===
    python baselines/eval_kvquant_squad.py \\
        --model_path ./models/qwen2.5-7b \\
        --n_calib 64 \\
        --n_eval 100 \\
        --positions 0 1 2 3 \\
        --seed 42 \\
        --device cuda:0 \\
        --save_path logs/results/kvquant_k4_qwen2_5-7b.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from astronet.wrapper import AstroNetWrapper
from astronet.kvquant_adapter import (
    CalibrationCollector, install_kvquant_simquant, uninstall_kvquant_simquant,
)
from data.real_qa import generate_squad_dataset
from training.eval_hybrid_position_robust import shuffle_fact_position


def load_backbone_4bit(model_path: str, device: str):
    """Load a 4-bit NF4 quantised backbone exactly as other eval scripts do."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def _normalise_answer(s: str) -> str:
    """SQuAD-style answer normalisation: lowercase, strip articles and
    punctuation, collapse whitespace.  Matches the SQuAD canonical eval."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def _is_correct(pred: str, gold: str) -> bool:
    p = _normalise_answer(pred)
    g = _normalise_answer(gold)
    if not g:
        return False
    return g in p or p == g


def calibrate(model, tokenizer, wrapper, calib_samples, seed: int = 42):
    """Collect K/V statistics by running calibration samples through the
    backbone's window-processing path.  Returns per-layer quantizers."""
    cfg = model.config
    nl = cfg.num_hidden_layers
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

    collector = CalibrationCollector(model, n_kv_heads=nkv, head_dim=hd,
                                     n_layers=nl)
    try:
        for samp in calib_samples:
            with torch.no_grad():
                _ = wrapper._process_windows(samp.windows)
        n_tokens = collector.total_tokens()
        print(f"  Collected {n_tokens} tokens of K/V across {len(calib_samples)} samples")
        quantizers = collector.calibrate_all(sparsity=0.01, seed=seed)
    finally:
        collector.remove()
    return quantizers


def eval_position_robust(wrapper, samples, position: int, seed: int) -> float:
    """Run the standard pos-robust eval (method='mult' = Stage 1 selection)
    at a fixed answer position.  Returns accuracy in [0, 1]."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    placed = shuffle_fact_position(samples, position)
    n_correct = 0
    for samp in placed:
        with torch.no_grad():
            pred = wrapper.answer(
                samp.windows[:-1], samp.question,
                k=300, method='mult', max_new_tokens=20,
            )
        if _is_correct(pred, samp.answer):
            n_correct += 1
    return n_correct / max(1, len(placed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--n_calib", type=int, default=64)
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--positions", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save_path", required=True)
    p.add_argument("--first_few_fp16", type=int, default=4)
    args = p.parse_args()

    print(f"[kvquant-eval] model={args.model_path}  device={args.device}")
    print(f"[kvquant-eval] n_calib={args.n_calib}  n_eval={args.n_eval}  "
          f"positions={args.positions}")

    print(f"[kvquant-eval] loading backbone...")
    model, tokenizer = load_backbone_4bit(args.model_path, args.device)
    cfg = model.config
    print(f"[kvquant-eval] model: {cfg.num_hidden_layers} layers, "
          f"{cfg.hidden_size} hidden, {cfg.num_key_value_heads} kv heads")

    wrapper = AstroNetWrapper(model, tokenizer, astro=None, n_mem=16)

    print(f"[kvquant-eval] loading calibration data ({args.n_calib} train samples)...")
    calib_samples = generate_squad_dataset(
        split="train", n_samples=args.n_calib, n_windows=5,
        vary_distance=True, seed=4242,
    )

    print(f"[kvquant-eval] calibrating per-layer codebooks...")
    t0 = time.time()
    quantizers = calibrate(model, tokenizer, wrapper, calib_samples,
                           seed=args.seed)
    print(f"[kvquant-eval] calibration: {time.time() - t0:.1f}s "
          f"({len(quantizers)} layer codebooks fitted)")

    print(f"[kvquant-eval] installing KVQuant K4 NUQ-1% wrappers "
          f"(first_few_fp16={args.first_few_fp16})...")
    originals = install_kvquant_simquant(model, quantizers,
                                         first_few_fp16=args.first_few_fp16)

    eval_samples = generate_squad_dataset(
        split="validation", n_samples=args.n_eval, n_windows=5,
        vary_distance=True, seed=args.seed,
    )
    accs = {}
    for pos in args.positions:
        print(f"[kvquant-eval] eval at position {pos} (n={args.n_eval})...")
        t0 = time.time()
        acc = eval_position_robust(wrapper, eval_samples, pos,
                                   seed=args.seed + pos)
        accs[f"pos_{pos}"] = {"acc": acc}
        print(f"  pos={pos}: acc={acc:.3f}  ({time.time() - t0:.1f}s)")

    uninstall_kvquant_simquant(model, originals)

    avg_acc = float(np.mean([accs[f"pos_{p}"]["acc"] for p in args.positions]))

    result = {
        "model": Path(args.model_path).name,
        "method": "kvquant_k4_nuq1pct_per_head_v",
        "n_calib": args.n_calib,
        "n_eval": args.n_eval,
        "positions": args.positions,
        "seed": args.seed,
        "first_few_fp16": args.first_few_fp16,
        "results": accs,
        "average": {"acc": avg_acc},
        "notes": (
            "KVQuant K4 NUQ-1% sim-quant adapted to AstroNet's serving wrapper. "
            "K is quantised pre-RoPE via _KQuantWrapper installed on every "
            "layer's k_proj. V uses per-head NUQ with per-token RMS "
            "normalisation (the reference uses per-layer codebook with per-token "
            "min-max). Fisher-weighted k-means skipped (reference reports ~0.1 PPL "
            "improvement; defensible to omit at K4). Calibration: SQuAD train, "
            "n_calib samples, 5 windows each, sparsity=0.01."
        ),
    }
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[kvquant-eval] avg acc = {avg_acc:.3f}; saved -> {args.save_path}")


if __name__ == "__main__":
    main()
