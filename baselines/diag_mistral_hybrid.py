"""Diagnostic for the mult-vs-hybrid confound (paper-wide, not just Mistral 24B).

The confound: `mult` uses k=300 (300 real tokens), `hybrid` uses k=300 = 16 mem + 284 real.
Every "hybrid beats mult" claim conflates "S2 mem tokens help" with "k=300 real > k=284 real".

This script tests 3 conditions per model at n_windows=20, using the same haystack seeds:
  1. mult_k300        (matches paper's "mult" column)
  2. mult_k284        (NEW — controls for real-token count)
  3. hybrid_k300      (matches paper's "hybrid" column = 16 mem + 284 real)

Interpretation per model:
  - If mult_k284 ≈ mult_k300: real-token count doesn't matter, hybrid win is from S2 mem tokens (good for paper).
  - If mult_k284 << mult_k300: hybrid handicap is real, and S2 must overcome it (still good).
  - If mult_k284 ≈ hybrid_k300: S2 contributes nothing (bad — kills the central claim).
"""
import sys; sys.path.insert(0, '.')
import os, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from baselines.eval_needle import (
    generate_haystack, process_and_select, NEEDLES
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--hybrid_checkpoint', required=True)
    parser.add_argument('--n_windows', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--sense_layer', type=int, default=None)
    parser.add_argument('--n_mem', type=int, default=16)
    parser.add_argument('--attn_dim', type=int, default=256)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_tag', default=None,
                        help='Tag for output file; default derived from model_path basename')
    parser.add_argument('--seed_offset', type=int, default=0,
                        help='Offset added to per-trial seed; 0 reproduces original results')
    parser.add_argument('--save_suffix', default='',
                        help='Suffix appended to output filename (e.g. _seed1) to avoid overwriting')
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path)
    tag = args.save_tag or model_name.replace('.', '_')

    print(f"[diag] Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = model.get_input_embeddings().weight.device

    # Load AstroNet (matches eval_needle.py loading pattern)
    from training.train_hybrid import AstroHybrid
    nl = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
    sense_layer = args.sense_layer or nl // 2
    astro = AstroHybrid(
        hidden_dim=hidden_dim, n_mem_tokens=args.n_mem, attn_dim=args.attn_dim,
        n_kv_heads=nkv, head_dim=hd, n_layers=nl,
        inject_layers=[nl // 4, nl // 2, 3 * nl // 4, nl - 2],
    ).to(device)
    astro.extract_model_weights(model, device)
    astro.load_state_dict(torch.load(args.hybrid_checkpoint, map_location=device), strict=False)
    astro.eval()
    print(f"[diag] AstroNet loaded: {astro.parameter_count():,} params", flush=True)

    # Conditions
    depths = [0, args.n_windows // 4, args.n_windows // 2, 3 * args.n_windows // 4, args.n_windows - 1]
    depth_labels = ['start', '25%', '50%', '75%', 'end']
    conditions = [
        ('mult_k300',   {'method': 'multiplicative', 'k': 300, 'astro': None}),
        ('mult_k284',   {'method': 'multiplicative', 'k': 284, 'astro': None}),
        ('hybrid_k300', {'method': 'hybrid',         'k': 300, 'astro': astro}),
    ]
    results = {c[0]: {lab: [] for lab in depth_labels} for c in conditions}

    print(f"[diag] n_windows={args.n_windows}, n_trials={args.n_trials}, depths={depth_labels}", flush=True)

    for trial in range(args.n_trials):
        for di, (depth, label) in enumerate(zip(depths, depth_labels)):
            seed = 42 + trial * 100 + di + args.seed_offset
            needle_idx = (trial + di + args.seed_offset) % len(NEEDLES)
            windows, question, answer = generate_haystack(
                args.n_windows, depth, needle_idx=needle_idx, seed=seed
            )
            for cname, ckwargs in conditions:
                pred = process_and_select(
                    model, tokenizer, windows, question,
                    k=ckwargs['k'], method=ckwargs['method'],
                    device=device, astro=ckwargs['astro'],
                    sense_layer=sense_layer,
                )
                ok = int(answer.lower() in pred.lower())
                results[cname][label].append(ok)
                if trial < 3:
                    print(f"  trial={trial} depth={label} {cname}: pred='{pred[:50]}...' ans='{answer[:40]}' ok={ok}", flush=True)

    # Aggregate
    print("\n=== AGGREGATE ===", flush=True)
    summary = {}
    for cname, _ in conditions:
        per_d = {k: sum(v) / len(v) for k, v in results[cname].items()}
        avg = sum(per_d.values()) / len(per_d)
        summary[cname] = {'per_depth': per_d, 'avg': avg}
        print(f"{cname}: {per_d} avg={avg:.3f}", flush=True)

    out_path = f'logs/results/diag_kconfound_{tag}_n{args.n_windows}{args.save_suffix}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'model': model_name, 'n_windows': args.n_windows,
            'n_trials': args.n_trials, 'seed_offset': args.seed_offset,
            'conditions': summary, 'raw': results
        }, f, indent=2)
    print(f"\n[diag] Saved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
