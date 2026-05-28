"""Wall-clock latency comparison for the multi-window SQuAD pipeline.

Measures per-sample time for each method, broken into:
- Window processing (forward passes on the fact windows)
- Selection or retrieval step
- Generation step

Reports mean and standard deviation over n samples on Qwen 7B at k=300.
"""

import sys, os, argparse, json, time
sys.path.insert(0, '.')
sys.path.insert(0, 'baselines')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from data.real_qa import generate_squad_dataset
from eval_all_baselines import (
    process_windows, build_cache_from_indices, generate_with_cache,
    select_streaming_llm, select_h2o, select_snapkv, select_multiplicative,
)


def time_block():
    torch.cuda.synchronize()
    return time.time()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='models/qwen2.5-7b')
    p.add_argument('--n_samples', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path',
                   default='logs/results/latency.json')
    args = p.parse_args()

    print(f'Loading {args.model_path}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = model.get_input_embeddings().weight.device
    nl = model.config.num_hidden_layers

    encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

    samples = generate_squad_dataset(n_samples=args.n_samples, n_windows=5,
                                      vary_distance=True, seed=args.seed,
                                      split='validation')

    # Warmup: run two samples through everything
    for s in samples[:2]:
        fw = s.windows[:-1]
        _ = process_windows(model, tokenizer, fw, device)

    methods = ['streaming_llm', 'h2o', 'snapkv', 'multiplicative', 'rag_k1', 'rag_k3']
    timings = {m: {'select': [], 'generate': [], 'total': []} for m in methods}
    timings['_window_processing'] = []

    eval_samples = samples[2:]
    print(f'Timing {len(eval_samples)} samples across {len(methods)} methods', flush=True)

    for si, s in enumerate(eval_samples):
        fact_windows = s.windows[:-1]
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)

        # Window processing — shared across KV-selection methods
        t0 = time_block()
        all_kv, all_attn, boundaries, total = process_windows(model, tokenizer, fact_windows, device)
        win_time = time_block() - t0
        timings['_window_processing'].append(win_time)

        # KV selection methods
        for method in ['streaming_llm', 'h2o', 'snapkv', 'multiplicative']:
            t1 = time_block()
            if method == 'streaming_llm':
                idx = select_streaming_llm(total, args.k).to(device)
            elif method == 'h2o':
                idx = select_h2o(all_attn, args.k, boundaries)
            elif method == 'snapkv':
                idx = select_snapkv(model, all_kv, s.question, tokenizer, args.k, device, boundaries)
            elif method == 'multiplicative':
                idx = select_multiplicative(all_attn, model, all_kv, s.question, tokenizer, args.k, device)
            sel_time = time_block() - t1
            cache = build_cache_from_indices(all_kv, idx, nl)
            t2 = time_block()
            _ = generate_with_cache(model, tokenizer, fq['input_ids'], cache, len(idx), device)
            gen_time = time_block() - t2
            timings[method]['select'].append(sel_time)
            timings[method]['generate'].append(gen_time)
            timings[method]['total'].append(win_time + sel_time + gen_time)

        # RAG: re-encode windows + retrieve + run model on selected windows + question
        for rag_k in [1, 3]:
            method = f'rag_k{rag_k}'
            t1 = time_block()
            # Encode fact windows + question
            window_texts = fact_windows
            window_emb = encoder.encode(window_texts, convert_to_tensor=True, show_progress_bar=False)
            q_emb = encoder.encode([s.question], convert_to_tensor=True, show_progress_bar=False)
            sims = (window_emb @ q_emb.T).squeeze(-1)
            top = sims.topk(min(rag_k, len(window_texts))).indices.tolist()
            top.sort()
            sel_time = time_block() - t1
            # Generate by re-feeding the retrieved windows + question to the model
            t2 = time_block()
            retrieved_text = '\n\n'.join(window_texts[i] for i in top) + '\n\n' + query
            inputs = tokenizer(retrieved_text, return_tensors='pt', truncation=True, max_length=4096).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen_time = time_block() - t2
            timings[method]['select'].append(sel_time)
            timings[method]['generate'].append(gen_time)
            # RAG total: ONLY this rag-k pipeline, no shared window-processing pre-compute
            timings[method]['total'].append(sel_time + gen_time)

        if (si + 1) % 10 == 0:
            print(f'  {si+1}/{len(eval_samples)} done', flush=True)

    # Aggregate
    import statistics
    summary = {'k': args.k, 'n_samples': len(eval_samples),
               'window_processing_mean_s': statistics.mean(timings['_window_processing']),
               'window_processing_std_s': statistics.stdev(timings['_window_processing'])
                   if len(timings['_window_processing']) > 1 else 0.0}
    for m in methods:
        summary[m] = {
            'select_mean_s': statistics.mean(timings[m]['select']),
            'generate_mean_s': statistics.mean(timings[m]['generate']),
            'total_mean_s': statistics.mean(timings[m]['total']),
            'total_std_s': statistics.stdev(timings[m]['total'])
                          if len(timings[m]['total']) > 1 else 0.0,
        }
    summary['note'] = ('AstroNet S1 reuses window_processing across methods. '
                       'Reported total for KV methods = window_processing + select + generate. '
                       'RAG total includes encoder forward + retrieval + a separate generation pass. '
                       'AstroNet S1+S2 latency = AstroNet S1 + Stage-2 forward (negligible: small module).')

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n=== Summary ===', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Saved to {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
