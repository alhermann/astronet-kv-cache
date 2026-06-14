"""Auto-calibrate λ for an AstroGate checkpoint.

Why: distillation loss never sees λ — it operates on the additive bias,
λ is only applied at eval. Different backbones land at different optimal
λ (Qwen 7B λ=0.5; Llama 8B λ=100; Mistral 7B λ=10). Requiring a manual
sweep per model is a production-time cost. This script runs that sweep
ONCE on a small held-out probe slice and caches the result.

Output: <checkpoint>.lam.json sidecar with the best λ and the grid it
came from. The eval scripts auto-load this if --lam_override is not set.

Cost: small. 10 probe trials × 4 questions × len(candidates).
Typical: ~5–15 min on a 7B–14B; ~20–30 min on a 24B/32B (dual-GPU).
"""
from __future__ import annotations
import argparse, json, os, sys, time, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from baselines.streaming_kv_core import AttentionCapture, SenseCapture
from baselines.eval_multiquery_needle import (
    build_multineedle_haystack, compress_once, answer_question)
from baselines.eval_multiquery_squad import _find_answer
from astronet.astro_gate import AstroGate


DEFAULT_CANDIDATES = (0.13, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0)


def _set_lambda(astro, lam):
    new_log = math.log(math.expm1(lam))     # inverse softplus
    astro.log_lambda.data.fill_(new_log)


@torch.no_grad()
def score_one_lambda(model, tokenizer, capture, sense_cap, astro,
                     probe_seeds, n_windows, n_needles, k, n_layers, device):
    correct, total = 0, 0
    for seed in probe_seeds:
        windows, qa = build_multineedle_haystack(n_windows, n_needles, seed=seed)
        compressed, full_layers, ctx_len = compress_once(
            model, tokenizer, capture, sense_cap, astro, 'astrogate',
            windows, k, 16, n_layers, device)
        for q, a in qa:
            gen = answer_question(model, tokenizer, capture, 'astrogate',
                                   compressed, full_layers, ctx_len, q,
                                   k, 16, n_layers, device)
            if _find_answer(gen, a): correct += 1
            total += 1
    return correct, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--n_windows', type=int, default=100)
    p.add_argument('--n_needles', type=int, default=4)
    p.add_argument('--n_probe_trials', type=int, default=10,
                   help='trials per λ candidate (each = n_needles questions)')
    p.add_argument('--probe_seed_base', type=int, default=9_999_000,
                   help='probe seeds = base+0..n-1; deliberately distinct '
                        'from eval seeds 42/seed*1000+t to avoid leakage')
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=256)
    p.add_argument('--candidates', default=','.join(str(c) for c in DEFAULT_CANDIDATES))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path', default=None,
                   help='default: <checkpoint>.lam.json')
    p.add_argument('--force', action='store_true',
                   help='overwrite existing sidecar')
    args = p.parse_args()

    save_path = args.save_path or (args.checkpoint + '.lam.json')
    if os.path.exists(save_path) and not args.force:
        print(f'SKIP — {save_path} already exists. Pass --force to overwrite.')
        with open(save_path) as f: print(json.dumps(json.load(f), indent=2))
        return

    candidates = [float(c) for c in args.candidates.split(',') if c.strip()]
    print(f'[calibrate] ckpt={args.checkpoint}', flush=True)
    print(f'[calibrate] candidates λ ∈ {candidates}', flush=True)
    print(f'[calibrate] probe = {args.n_probe_trials} trials × {args.n_needles} Q '
          f'= {args.n_probe_trials*args.n_needles} questions per λ', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = str(model.get_input_embeddings().weight.device)
    capture = AttentionCapture(model)
    cfg = model.config
    n_layers = cfg.num_hidden_layers

    raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_cfg = raw.get('config', {}) if isinstance(raw, dict) else {}
    astro = AstroGate(
        hidden_dim=cfg.hidden_size,
        n_astro=saved_cfg.get('n_astro', args.n_mem),
        attn_dim=saved_cfg.get('attn_dim', args.attn_dim),
        n_patterns=saved_cfg.get('n_patterns', 8),
    ).to(device)
    astro.load_state_dict(
        raw['astro'] if isinstance(raw, dict) and 'astro' in raw else raw,
        strict=False)
    astro.eval()
    sl = raw.get('sense_layer', n_layers // 2) if isinstance(raw, dict) else n_layers // 2
    sense_cap = SenseCapture(model, sl)

    probe_seeds = [args.probe_seed_base + i for i in range(args.n_probe_trials)]
    grid = {}
    t0 = time.time()
    for lam in candidates:
        _set_lambda(astro, lam)
        correct, total = score_one_lambda(
            model, tokenizer, capture, sense_cap, astro,
            probe_seeds, args.n_windows, args.n_needles, args.k,
            n_layers, device)
        acc = 100.0 * correct / max(1, total)
        grid[f'{lam:g}'] = {'acc': acc, 'correct': correct, 'total': total}
        elapsed = time.time() - t0
        print(f'  λ={lam:>6g}  acc={acc:5.2f}% ({correct}/{total})  '
              f'[+{elapsed:.0f}s]', flush=True)

    best_key = max(grid, key=lambda k: grid[k]['acc'])
    best_lam = float(best_key)
    best_acc = grid[best_key]['acc']

    out = {
        'lambda': best_lam,
        'best_acc': best_acc,
        'grid': grid,
        'candidates': candidates,
        'n_probe_trials': args.n_probe_trials,
        'n_needles': args.n_needles,
        'n_windows': args.n_windows,
        'k': args.k,
        'probe_seed_base': args.probe_seed_base,
        'model_path': args.model_path,
        'checkpoint': args.checkpoint,
        'calibrated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_s': round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f: json.dump(out, f, indent=2)
    print(f'\nBest λ = {best_lam:g}  ({best_acc:.2f}% probe-acc)')
    print(f'Wrote -> {save_path}')


if __name__ == '__main__':
    main()
