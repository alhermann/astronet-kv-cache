"""Eval-only script for hybrid (S1+S2) at n=200 samples.
Loads a trained AstroNet checkpoint and runs evaluate() on 200 SQuAD validation samples.
No training — just evaluation. Saves results with unique filenames."""
import sys; sys.path.insert(0, '.')
import os, json, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from training.train_hybrid import AstroHybrid, evaluate
from data.real_qa import generate_squad_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--checkpoint', required=True, help='Path to AstroNet .pt checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_eval', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k_real', type=int, default=284)
    parser.add_argument('--n_mem', type=int, default=16)
    parser.add_argument('--attn_dim', type=int, default=256)
    parser.add_argument('--multi_gpu', action='store_true')
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path)
    print(f"Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print(f"Model: {nl} layers, {hidden_dim} hidden, inject={inject_layers}, sense={sense_layer}", flush=True)

    astro = AstroHybrid(
        hidden_dim=hidden_dim,
        n_mem_tokens=args.n_mem,
        attn_dim=args.attn_dim,
        n_kv_heads=nkv,
        head_dim=hd,
        n_layers=nl,
        inject_layers=inject_layers,
    ).to(device)
    astro.extract_model_weights(model, device)
    astro.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False), strict=False)
    astro.eval()
    print(f"Loaded checkpoint: {args.checkpoint} ({astro.parameter_count():,} params)", flush=True)

    samples = generate_squad_dataset(n_samples=args.n_eval, n_windows=5,
                                      vary_distance=True, seed=args.seed, split='validation')
    print(f"Eval: {len(samples)} samples, k_real={args.k_real}, seed={args.seed}", flush=True)

    pure_acc, hybrid_acc = evaluate(model, tokenizer, astro, samples,
                                     inject_layers, sense_layer, args.k_real, device)
    print(f"\nRESULTS {model_name}:", flush=True)
    print(f"  pure300 (S1):     {pure_acc:.1%}", flush=True)
    print(f"  hybrid (S1+S2):   {hybrid_acc:.1%}", flush=True)

    save_path = f'logs/results/hybrid_n200_{model_name.replace(".", "_")}_s{args.seed}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'model': model_name,
            'checkpoint': args.checkpoint,
            'n_eval': args.n_eval,
            'seed': args.seed,
            'k_real': args.k_real,
            'n_mem': args.n_mem,
            'results': {
                'pure300': pure_acc,
                'hybrid': hybrid_acc,
            }
        }, f, indent=2)
    print(f"Saved to {save_path}", flush=True)


if __name__ == '__main__':
    main()
