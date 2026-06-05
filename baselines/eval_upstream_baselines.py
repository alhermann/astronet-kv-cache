"""Upstream-faithful SQuAD position-robust baseline eval.

Drives the upstream ``kvcache_factory`` monkey-patches (SnapKV, H2O,
PyramidKV) on bnb-4bit-loaded backbones for the same position-robust
SQuAD benchmark used by AstroNet hybrid.

Each method is run in a fresh process (monkey-patches are global), so
one invocation = one method.  Bash drives the matrix.

The eval feeds the full multi-window context + question as a single
prompt.  The monkey-patched attention forward intercepts at prefill and
compresses the KV cache to the configured budget.  Continuation is
generated greedily from the compressed cache.

Usage::

    python baselines/eval_upstream_baselines.py \\
        --model_path ./models/llama-3.1-8b \\
        --method snapkv --k 300 --n_eval 100 \\
        --positions 0 1 2 3 \\
        --save_path logs/results/upstream_snapkv_squad_llama8b.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _family_from_path(model_path: str) -> str:
    name = os.path.basename(model_path.rstrip('/')).lower()
    if 'llama' in name:
        return 'llama'
    if 'mistral' in name:
        return 'mistral'
    if 'qwen' in name:
        return 'qwen'
    raise ValueError(f'cannot infer model family from {model_path!r}')


def _apply_monkey_patch(family: str, method: str) -> None:
    """Apply the upstream attention-forward monkey-patch for the given family.

    Must be called BEFORE the model is loaded.
    """
    from baselines.kvcache_factory.monkeypatch import (
        replace_llama, replace_mistral, replace_qwen2,
    )
    if family == 'llama':
        replace_llama(method)
    elif family == 'mistral':
        replace_mistral(method)
    elif family == 'qwen':
        replace_qwen2(method)
    else:
        raise ValueError(family)


def _set_config_hparams(model_config, method: str, k: int) -> None:
    """Apply paper hyperparameters for each method.

    The upstream init_X(self) helpers read from self.config the first
    time the patched forward runs.
    """
    if method == 'snapkv':
        # SnapKV paper Table 1: W=32, kernel=7, max-pool.
        model_config.window_size = 32
        model_config.max_capacity_prompt = k
        model_config.kernel_size = 7
        model_config.pooling = 'maxpool'
        model_config.merge = None
    elif method == 'h2o':
        # H2O paper (Zhang et al. NeurIPS 2023, Algorithm 1): heavy + recent =
        # total budget, with recent = total / 2 in the canonical setup.
        # ``window_size`` in this codebase IS the recent budget; heavy is
        # ``max_capacity_prompt - window_size``.
        model_config.window_size = k // 2
        model_config.max_capacity_prompt = k
    elif method == 'pyramidkv':
        # PyramidKV paper: alpha=8 observation window, beta=20 top/bottom
        # ratio, max-pool kernel-5 over avg-pool, max_capacity_prompt is
        # the AVERAGE per-layer budget.
        model_config.window_size = 8
        model_config.max_capacity_prompt = k
        model_config.kernel_size = 5
        model_config.pooling = 'avgpool'
        model_config.beta = 20
        model_config.merge = None
    else:
        raise ValueError(method)


def load_backbone(model_path: str, multi_gpu: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    common = dict(quantization_config=bnb, torch_dtype=torch.float16,
                   attn_implementation='sdpa')
    if multi_gpu:
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            if gib >= 16:
                max_memory[i] = '22GiB'
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map='auto', max_memory=max_memory, **common)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map={'': 'cuda:0'}, **common)
    model.eval()
    return model, tokenizer


def _build_prompt(sample) -> str:
    """Concatenate context windows + question into a single prefill string.

    ``sample.windows[-1]`` is the canonical query template (with the
    'Answer:' marker).  Everything before it is context.
    """
    context = '\n\n'.join(sample.windows[:-1])
    query = sample.windows[-1]
    return f'{context}\n\n{query}'


def _reset_kv_state(model):
    """Reset per-layer ``kv_seq_len`` so the next prefill is detected
    correctly by the upstream monkey-patched forward.

    For Llama and Mistral the upstream ``replace_X`` already patches
    ``prepare_inputs_for_generation`` to do this; for Qwen2 we do not
    patch the PIFG, so the runner has to reset manually before every
    ``model.generate(...)``.  Always safe to call (no-op if the
    attribute isn't present yet).
    """
    for layer in model.model.layers:
        if hasattr(layer.self_attn, 'kv_seq_len'):
            layer.self_attn.kv_seq_len = 0


def _generate(model, tokenizer, prompt: str, max_tokens: int):
    """Single-prompt greedy generation.

    The monkey-patched attention forward compresses the KV cache during
    prefill; subsequent decoded tokens go through the standard attention
    path with the compressed cache.
    """
    import torch
    embed_dev = model.get_input_embeddings().weight.device
    ids = tokenizer(prompt, return_tensors='pt').to(embed_dev)
    _reset_kv_state(model)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    gen = out[0, ids['input_ids'].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def evaluate_at_position(model, tokenizer, samples, pos: int, max_tokens: int):
    from training.eval_hybrid_position_robust import shuffle_fact_position

    placed = shuffle_fact_position(samples, pos)
    correct = 0
    for si, s in enumerate(placed):
        prompt = _build_prompt(s)
        ans = _generate(model, tokenizer, prompt, max_tokens)
        if s.answer.lower() in ans.lower():
            correct += 1
        if (si + 1) % 25 == 0:
            print(f'  pos={pos} {si+1}/{len(placed)} acc={correct}/{si+1}',
                  flush=True)
    return correct / max(1, len(placed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--method', required=True,
                   choices=['snapkv', 'h2o', 'pyramidkv'])
    p.add_argument('--k', type=int, default=300,
                   help='max_capacity_prompt (avg per-layer for pyramidkv)')
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--max_tokens', type=int, default=20)
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    family = _family_from_path(args.model_path)
    print(f'[upstream-baseline] family={family} method={args.method} '
          f'k={args.k}  n_eval={args.n_eval}  seed={args.seed}', flush=True)

    # 1) Monkey-patch BEFORE model load.
    _apply_monkey_patch(family, args.method)

    # 2) Load model.
    model, tokenizer = load_backbone(args.model_path, args.multi_gpu)

    # 3) Apply config-driven hyperparameters.
    _set_config_hparams(model.config, args.method, args.k)

    # 4) SQuAD samples.
    from data.real_qa import generate_squad_dataset
    samples = generate_squad_dataset(
        n_samples=args.n_eval, n_windows=5, vary_distance=True,
        seed=args.seed, split='validation')

    # 5) Per-position eval.
    results = {}
    for pos in args.positions:
        print(f'\n=== pos={pos} ===', flush=True)
        t0 = time.time()
        acc = evaluate_at_position(model, tokenizer, samples, pos,
                                     args.max_tokens)
        results[f'pos_{pos}'] = acc
        print(f'  pos={pos}: acc={acc * 100:.1f}%  '
              f'({time.time() - t0:.0f}s)', flush=True)

    avg = sum(results.values()) / max(1, len(results))
    print(f'\nAVG across positions: {avg * 100:.1f}%', flush=True)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'family': family,
            'method': args.method,
            'k': args.k,
            'n_eval': args.n_eval,
            'seed': args.seed,
            'positions': args.positions,
            'results': results,
            'average': avg,
            'source': 'upstream kvcache_factory monkey-patch '
                       '(SnapKV/H2O/PyramidKV faithful to paper '
                       'algorithm + paper hyperparameters)',
        }, f, indent=2)
    print(f'Saved -> {args.save_path}')


if __name__ == '__main__':
    main()
