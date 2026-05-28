"""Position-robust hybrid eval: pure-S1 vs S1+S2 across all four fact positions.

Loads a trained AstroNet checkpoint and runs evaluate() on randomized-position
SQuAD samples (fact at window 0/1/2/3 in turn).
"""
import sys, os, json, argparse, torch
sys.path.insert(0, '.')
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from training.train_hybrid import AstroHybrid, evaluate
from data.real_qa import generate_squad_dataset, CrossContextSample


def shuffle_fact_position(samples, position):
    out = []
    for s in samples:
        windows = list(s.windows)
        query_idx = s.query_window
        fact_idx = s.fact_window
        candidates = [i for i in range(len(windows)) if i != query_idx]
        if position >= len(candidates):
            position = len(candidates) - 1
        target = candidates[position]
        if target == fact_idx:
            out.append(s); continue
        new_w = list(windows)
        new_w[target], new_w[fact_idx] = new_w[fact_idx], new_w[target]
        out.append(CrossContextSample(
            windows=new_w, fact=s.fact, question=s.question, answer=s.answer,
            fact_window=target, query_window=query_idx,
            distance=s.distance, template_idx=s.template_idx,
        ))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k_real', type=int, default=284)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=512)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', default=None)
    args = p.parse_args()

    model_name = os.path.basename(args.model_path)
    print(f'Loading {args.model_path}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type='nf4')
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = str(model.get_input_embeddings().weight.device)

    hidden_dim = model.config.hidden_size
    nl = model.config.num_hidden_layers
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
    inject_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    sense_layer = nl // 2

    astro = AstroHybrid(
        hidden_dim=hidden_dim, n_mem_tokens=args.n_mem, attn_dim=args.attn_dim,
        n_kv_heads=nkv, head_dim=hd, n_layers=nl, inject_layers=inject_layers,
    ).to(device)
    astro.extract_model_weights(model, device)
    astro.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False),
                           strict=False)
    astro.eval()
    print(f'Loaded checkpoint: {args.checkpoint} ({astro.parameter_count():,} params)', flush=True)

    base = generate_squad_dataset(n_samples=args.n_eval, n_windows=5,
                                   vary_distance=True, seed=args.seed, split='validation')

    results = {}
    for pos in args.positions:
        samples = shuffle_fact_position(base, pos)
        print(f'\n=== fact_pos={pos} ({len(samples)} samples) ===', flush=True)
        pure, hyb = evaluate(model, tokenizer, astro, samples,
                              inject_layers, sense_layer, args.k_real, device)
        results[f'pos_{pos}'] = {'pure300': pure, 'hybrid': hyb, 'delta': hyb - pure}
        print(f'  pos={pos}: pure300={pure:.3f} hybrid={hyb:.3f} delta={hyb-pure:+.3f}', flush=True)

    avg_pure = sum(v['pure300'] for v in results.values()) / len(results)
    avg_hyb = sum(v['hybrid'] for v in results.values()) / len(results)
    print(f'\nAverage across positions: pure300={avg_pure:.3f} hybrid={avg_hyb:.3f} '
          f'delta={avg_hyb-avg_pure:+.3f}', flush=True)

    save_path = args.save_path or f'logs/results/hybrid_position_robust_{model_name.replace(".", "_")}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'model': model_name, 'checkpoint': args.checkpoint,
            'n_eval': args.n_eval, 'seed': args.seed,
            'k_real': args.k_real, 'n_mem': args.n_mem,
            'attn_dim': args.attn_dim,
            'results': results,
            'average': {'pure300': avg_pure, 'hybrid': avg_hyb, 'delta': avg_hyb - avg_pure},
        }, f, indent=2)
    print(f'Saved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
