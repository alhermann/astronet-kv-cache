"""Verification: upstream SnapKV monkey-patch on Llama 8B vs full context.

Run two conditions on the same model + benchmark with single-prompt
LongBench HotpotQA generation:

  * mode=full     - no monkey-patch, no compression
  * mode=snapkv   - upstream replace_llama('snapkv') with k=1024

Expected if the upstream implementation is faithful: the SnapKV f1
should be within ~3 pts of the full-context f1 on this task (the
published SnapKV paper reports ~1pt gap on Mistral 7B Instruct).

Run in TWO SEPARATE processes (monkey-patches are global; can't toggle
in-process).  Save one JSON per condition; the comparison is done by
loading both files.

Usage::

    python baselines/verify_upstream_snapkv.py --mode full \\
        --n_samples 50 \\
        --save_path logs/results/verify_full_llama8b_hotpot.json
    python baselines/verify_upstream_snapkv.py --mode snapkv --k 1024 \\
        --n_samples 50 \\
        --save_path logs/results/verify_snapkv1024_llama8b_hotpot.json
"""
from __future__ import annotations
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['full', 'snapkv'])
    p.add_argument('--model_path', default='./models/llama-3.1-8b')
    p.add_argument('--task', default='hotpotqa')
    p.add_argument('--n_samples', type=int, default=50)
    p.add_argument('--k', type=int, default=1024,
                   help='max_capacity_prompt for snapkv (ignored for full)')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    # 1) Monkey-patch (or not) BEFORE model load.
    if args.mode == 'snapkv':
        from baselines.kvcache_factory.monkeypatch import replace_llama
        replace_llama('snapkv')
        print('[verify] replace_llama(snapkv) applied', flush=True)
    else:
        print('[verify] no monkey-patch (full-context baseline)', flush=True)

    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    from datasets import load_dataset
    from baselines.eval_longbench import TASKS, PROMPTS
    from baselines.longbench_canonical_f1 import f1_score as canonical_f1

    print(f'[verify] loading {args.model_path} ...', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': 'cuda:0'},
        torch_dtype=torch.float16,
        attn_implementation='sdpa')
    model.eval()
    print('[verify] model loaded', flush=True)

    if args.mode == 'snapkv':
        # SnapKV paper hyperparameters: W=32, kernel=7, max-pool.
        model.config.window_size = 32
        model.config.max_capacity_prompt = args.k
        model.config.kernel_size = 7
        model.config.pooling = 'maxpool'
        model.config.merge = None
        print(f'[verify] SnapKV budget = {args.k} (W=32, kernel=7, max-pool)',
              flush=True)

    print(f'[verify] loading LongBench {args.task} ...', flush=True)
    ds = load_dataset('THUDM/LongBench', args.task, split='test',
                       trust_remote_code=True)
    samples = list(ds)[:args.n_samples]
    max_gen = TASKS[args.task]['max_gen']
    prompt_tpl = PROMPTS[args.task]

    embed_dev = model.get_input_embeddings().weight.device

    f1s = []
    t0 = time.time()
    for si, s in enumerate(samples):
        prompt = prompt_tpl.format(context=s['context'], input=s['input'])
        ids = tok(prompt, return_tensors='pt', truncation=True,
                   max_length=16384).to(embed_dev)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=max_gen,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
        gen_ids = out[0, ids['input_ids'].shape[1]:]
        pred = tok.decode(gen_ids, skip_special_tokens=True).strip()
        # Some models include trailing junk; truncate at first newline.
        pred = pred.split('\n')[0].strip()
        f1 = canonical_f1(pred, s.get('answers', []))
        f1s.append(f1)
        if (si + 1) % 10 == 0:
            print(f'  {si+1}/{len(samples)} '
                   f'running mean f1 = {sum(f1s)/len(f1s):.2f}', flush=True)

    mean_f1 = sum(f1s) / max(1, len(f1s))
    print(f'\n[verify] mode={args.mode}  task={args.task}  '
           f'n={len(f1s)}  mean_f1 = {mean_f1:.2f}  '
           f'({time.time() - t0:.0f}s)', flush=True)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'mode': args.mode,
            'model': os.path.basename(args.model_path),
            'task': args.task,
            'n_samples': len(f1s),
            'k': args.k if args.mode == 'snapkv' else None,
            'mean_f1': mean_f1,
            'per_sample_f1': f1s,
            'wall_time_s': time.time() - t0,
        }, f, indent=2)
    print(f'[verify] saved -> {args.save_path}')


if __name__ == '__main__':
    main()
