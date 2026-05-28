"""Full-FP16 KV cache reference: no compression, no eviction, no retrieval.

Evaluates each pos-robust SQuAD sample with the entire concatenated multi-window
context fed to the model in one pass, so the model attends over the full FP16
KV cache. This is the upper-right anchor of the Pareto frontier in Figure 1
(compression ratio = 1x, accuracy = whatever the backbone achieves on the
uncompressed task).

Matches eval_hybrid_position_robust.py settings: n_windows=5, n_eval=100,
seed=42, positions [0,1,2,3] for the fact context.
"""
import sys, os, json, argparse, time

sys.path.insert(0, '.')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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


@torch.no_grad()
def eval_full(model, tokenizer, samples, device, max_input_tokens=8192):
    correct = 0
    truncated = 0
    for si, s in enumerate(samples):
        prompt = "\n\n".join(s.windows)
        enc = tokenizer(prompt, return_tensors='pt',
                         max_length=max_input_tokens, truncation=True)
        input_ids = enc['input_ids'].to(device)
        if enc['input_ids'].shape[1] >= max_input_tokens:
            truncated += 1
        out = model.generate(
            input_ids, max_new_tokens=24, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        if s.answer.lower() in gen.lower():
            correct += 1
        if (si + 1) % 25 == 0:
            print(f'    {si+1}/{len(samples)}: {correct}/{si+1}'
                  f' ({correct/(si+1):.3f})', flush=True)
    return correct / len(samples), truncated


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', default=None)
    args = p.parse_args()

    model_name = os.path.basename(args.model_path)
    print(f'Loading {args.model_path}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type='nf4')
    device_map = 'auto' if args.multi_gpu else {'': args.device}
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
        quantization_config=bnb, device_map=device_map, torch_dtype=torch.float16)
    model.eval()
    device = str(model.get_input_embeddings().weight.device)

    base = generate_squad_dataset(n_samples=args.n_eval, n_windows=5,
                                   vary_distance=True, seed=args.seed,
                                   split='validation')

    results = {}
    overall_correct = 0
    overall_total = 0
    truncated_total = 0
    for pos in args.positions:
        samples = shuffle_fact_position(base, pos)
        print(f'\n=== fact_pos={pos} ({len(samples)} samples) ===', flush=True)
        t0 = time.time()
        acc, trunc = eval_full(model, tokenizer, samples, device)
        elapsed = time.time() - t0
        results[f'pos_{pos}'] = {'full_fp16': acc, 'truncated': trunc}
        overall_correct += round(acc * len(samples))
        overall_total += len(samples)
        truncated_total += trunc
        print(f'  pos={pos}: full_fp16={acc:.3f} (truncated={trunc}/{len(samples)}) '
              f'({elapsed:.0f}s)', flush=True)

    avg = sum(v['full_fp16'] for v in results.values()) / len(results)
    print(f'\nAverage across positions: full_fp16={avg:.3f} '
          f'(truncated total {truncated_total}/{overall_total})', flush=True)

    save_path = args.save_path or (
        f'logs/results/'
        f'full_fp16_{model_name.replace(".", "_")}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'model': model_name,
            'n_eval': args.n_eval, 'seed': args.seed,
            'method': 'full_fp16',
            'note': ('Whole concatenated multi-window context fed to the model '
                     'in one pass; no KV compression, no eviction. Backbone is '
                     'still 4-bit nf4 quantised, only the KV cache is FP16 (the '
                     'natural state).'),
            'n_windows': 5,
            'results': results,
            'average': {'full_fp16': avg, 'truncated_total': truncated_total},
        }, f, indent=2)
    print(f'Saved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
