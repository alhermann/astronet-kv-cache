"""Position-robust SQuAD evaluation of a trained soft prompt.

Mirrors `training/eval_hybrid_position_robust.py` but with a SoftPrompt
adapter in place of AstroHybrid: the trained 16-vector (or larger) prompt
is prepended in front of the input embeddings of each evaluation example.
Reports per-position accuracy + the across-position average.

This is the defense-against-soft-prompt experiment.  The paper will
report SoftPrompt vs AstroHybrid vs pure Stage 1 on the same protocol.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           BitsAndBytesConfig)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.real_qa import generate_squad_dataset
from training.eval_hybrid_position_robust import shuffle_fact_position
from training.train_softprompt import SoftPrompt


def _find_answer(prediction: str, answer: str) -> bool:
    return answer.strip().lower() in prediction.strip().lower()


def evaluate_at_position(model, tokenizer, sp, samples, position, max_tokens=20):
    embed = model.get_input_embeddings()
    placed = shuffle_fact_position(samples, position)
    correct = 0
    for si, s in enumerate(placed):
        full_text = '\n\n'.join(s.windows[:-1]) + '\n\n' + s.windows[-1]
        ids = tokenizer(full_text, return_tensors='pt', max_length=2048,
                          truncation=True).to(model.device)
        with torch.no_grad():
            tok_embeds = embed(ids['input_ids']).float()
            inputs_embeds = sp.prepend(tok_embeds).half()
            generated = model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        if _find_answer(text, s.answer):
            correct += 1
        if (si + 1) % 25 == 0:
            print(f'  pos={position} {si+1}/{len(placed)}: acc={correct}/{si+1}',
                  flush=True)
    return correct / max(1, len(placed))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--prompt_path', required=True,
                   help='Path to .pt saved by training/train_softprompt.py')
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    print(f'[softprompt-eval] model={args.model_path}  prompt={args.prompt_path}',
          flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()

    sd = torch.load(args.prompt_path, map_location=args.device,
                     weights_only=False)
    n_prompt = sd['n_prompt']
    hidden_dim = sd['hidden_dim']
    sp = SoftPrompt(n_prompt, hidden_dim).to(args.device).float()
    with torch.no_grad():
        sp.prompt.copy_(sd['prompt'].to(args.device).float())
    sp.eval()
    print(f'[softprompt-eval] loaded {sd["param_count"]:,} prompt params',
          flush=True)

    samples = generate_squad_dataset(
        n_samples=args.n_eval, n_windows=5, vary_distance=True,
        seed=args.seed, split='validation')

    results = {}
    for pos in args.positions:
        print(f'\n=== pos={pos} ===', flush=True)
        t0 = time.time()
        acc = evaluate_at_position(model, tokenizer, sp, samples, pos)
        results[f'pos_{pos}'] = acc
        print(f'  pos={pos}: acc={acc * 100:.1f}%  ({time.time() - t0:.0f}s)',
              flush=True)

    avg = sum(results.values()) / max(1, len(results))
    print(f'\nAVG: {avg * 100:.1f}%', flush=True)
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'prompt_path': args.prompt_path,
            'n_prompt': n_prompt,
            'param_count': sd['param_count'],
            'n_eval': args.n_eval, 'seed': args.seed,
            'positions': args.positions,
            'results': results,
            'average': avg,
        }, f, indent=2)
    print(f'Saved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
