"""LongBench evaluation with AstroNet multi-window pipeline.

Chunks long documents into windows, processes sequentially with KV selection,
then answers the question. Compares: Full KV, StreamingLLM, H2O, SnapKV,
Multiplicative, and AstroNet Hybrid.

Focus on QA tasks most relevant to our method:
  - NarrativeQA: story comprehension
  - HotpotQA: multi-hop reasoning
  - MultiFieldQA: diverse domain QA
"""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache


TASKS = {
    'narrativeqa': {'max_gen': 128, 'metric': 'f1'},
    'hotpotqa': {'max_gen': 32, 'metric': 'f1'},
    'multifieldqa_en': {'max_gen': 64, 'metric': 'f1'},
    '2wikimqa': {'max_gen': 32, 'metric': 'f1'},
    'musique': {'max_gen': 32, 'metric': 'f1'},
}

PROMPTS = {
    'narrativeqa': "You are given a story and a question. Answer the question as concisely as you can, using a single phrase if possible.\n\nStory: {context}\n\nQuestion: {input}\nAnswer:",
    'hotpotqa': "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    'multifieldqa_en': "Read the following text and answer briefly.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    '2wikimqa': "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    'musique': "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n{context}\n\nQuestion: {input}\nAnswer:",
}


def chunk_text(text, tokenizer, max_chunk_tokens=384):
    """Split text into windows of roughly max_chunk_tokens each."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_chunk_tokens):
        chunk_ids = tokens[i:i + max_chunk_tokens]
        chunks.append(tokenizer.decode(chunk_ids))
    return chunks


from baselines.longbench_canonical_f1 import f1_score as _canonical_f1


def f1_score(prediction, ground_truths):
    """Canonical LongBench token-level F1 (Counter-based, SQuAD-style
    normalisation: lowercase, strip articles a/an/the, strip
    punctuation). Returns the value in the 0--1 range to match the
    existing caller convention in this script; the canonical scorer
    itself returns in 0--100 so we divide.
    """
    return _canonical_f1(prediction, ground_truths) / 100.0


def process_windows_and_select(model, tokenizer, windows, question, k, method, device,
                                astro=None, sense_layer=14):
    """Process text windows and select top-k KV pairs using specified method."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    all_kv = {li: ([], []) for li in range(nl)}
    all_attn = []
    window_boundaries = []
    offset = 0

    # Reset AstroNet state if using hybrid
    if astro is not None:
        astro.reset_state()

    for window in windows:
        ids = tokenizer(window, return_tensors='pt', max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True,
                        output_attentions=True,
                        output_hidden_states=(astro is not None))
        sl = ids['input_ids'].shape[1]
        window_boundaries.append((offset, offset + sl))
        offset += sl

        imp = torch.zeros(sl, device=device)
        for la in out.attentions:
            a = la[0]
            if not a.isnan().any():
                imp += a.sum(dim=(0, 1))
        all_attn.append(imp)

        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])

        # AstroNet sensing + Ca2+ update
        if astro is not None:
            hidden = out.hidden_states[sense_layer].detach()
            sensed = astro.sense(hidden)
            astro.update_state(sensed)

    total = offset
    if total == 0:
        return None, 0

    k = min(k, total)
    heur = torch.cat(all_attn)
    n_sink = 4

    if method == 'full_kv':
        idx = torch.arange(total, device=device)
        k = total
    elif method == 'streaming_llm':
        if total <= k:
            idx = torch.arange(total, device=device)
        else:
            sink = torch.arange(n_sink, device=device)
            recent = torch.arange(total - (k - n_sink), total, device=device)
            idx = torch.cat([sink, recent])
    elif method == 'h2o':
        n_recent = int(k * 0.3)
        n_heavy = k - n_sink - n_recent
        last_s, last_e = window_boundaries[-1]
        n_recent = min(n_recent, last_e - last_s)
        recent_idx = torch.arange(last_s, last_s + n_recent, device=device)
        scores = heur.clone()
        scores[:n_sink] = -1e9
        scores[last_s:last_e] = -1e9
        n_avail = (scores > -1e8).sum().item()
        _, heavy_idx = scores.topk(min(n_heavy, n_avail))
        idx = torch.cat([torch.arange(n_sink, device=device), heavy_idx, recent_idx]).unique().sort().values[:k]
    elif method == 'snapkv':
        q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                          max_length=128, truncation=True).to(device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
        cross = torch.zeros(total, device=device)
        for li in inject:
            layer_dev = all_kv[li][0][0].device
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float().to(layer_dev)).half().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(), K[hi].float().T) / math.sqrt(hd)
                attn_w = torch.softmax(sc, dim=-1)
                cross += attn_w.sum(dim=(0, 1)).to(device)
        if total > 5:
            cross = torch.nn.functional.avg_pool1d(cross.unsqueeze(0).unsqueeze(0), 5, 1, 2).squeeze()
        cross[:n_sink] = -1e9
        _, top_idx = cross.topk(k - n_sink)
        idx = torch.cat([torch.arange(n_sink, device=device), top_idx]).sort().values
    elif method == 'multiplicative':
        # Patched S1: softmax + multi-layer + last-window mask + recent-keep
        q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                          max_length=128, truncation=True).to(device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
        cross = torch.zeros(total, device=device)
        for li in inject:
            layer_dev = all_kv[li][0][0].device
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float().to(layer_dev)).half().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(), K[hi].float().T) / math.sqrt(hd)
                attn_w = torch.softmax(sc, dim=-1)
                cross += attn_w.sum(dim=(0, 1)).to(device)
        if total > 5:
            cross = torch.nn.functional.avg_pool1d(cross.unsqueeze(0).unsqueeze(0), 5, 1, 2).squeeze()
        cross[:n_sink] = -1e9
        last_window_size = all_kv[inject[0]][0][-1].shape[2]
        last_start = total - last_window_size
        n_recent = min(int(k * 0.2), last_window_size)
        recent_idx = torch.arange(last_start, last_start + n_recent, device=device)
        scores = cross.clone()
        scores[last_start:total] = -1e9
        n_select = max(k - n_sink - n_recent, 0)
        n_avail = (scores > -1e8).sum().item()
        n_select = min(n_select, n_avail, scores.shape[0])
        _, top = scores.topk(n_select) if n_select > 0 else (None, torch.empty(0, dtype=torch.long, device=device))
        sink_idx = torch.arange(n_sink, device=device)
        idx = torch.cat([sink_idx, top, recent_idx]).unique().sort().values[:k]
    elif method == 'hybrid' and astro is not None:
        # Patched S1 selection for real tokens (k - n_mem)
        n_mem = astro.n_mem_tokens
        k_real = k - n_mem
        q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                          max_length=128, truncation=True).to(device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
        cross = torch.zeros(total, device=device)
        for li in inject:
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float()).half().view(-1, nq, hd)
            K_cat = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(), K_cat[hi].float().T) / math.sqrt(hd)
                attn_w = torch.softmax(sc, dim=-1)
                cross += attn_w.sum(dim=(0, 1)).to(device)
        if total > 5:
            cross = torch.nn.functional.avg_pool1d(cross.unsqueeze(0).unsqueeze(0), 5, 1, 2).squeeze()
        cross[:n_sink] = -1e9
        last_window_size = all_kv[inject[0]][0][-1].shape[2]
        last_start = total - last_window_size
        n_recent = min(int(k_real * 0.2), last_window_size)
        recent_idx = torch.arange(last_start, last_start + n_recent, device=device)
        scores = cross.clone()
        scores[last_start:total] = -1e9
        n_select = max(min(k_real, total) - n_sink - n_recent, 0)
        n_avail = (scores > -1e8).sum().item()
        n_select = min(n_select, n_avail, scores.shape[0])
        _, top = scores.topk(n_select) if n_select > 0 else (None, torch.empty(0, dtype=torch.long, device=device))
        sink_idx = torch.arange(n_sink, device=device)
        idx = torch.cat([sink_idx, top, recent_idx]).unique().sort().values[:min(k_real, total)]

        # Build hybrid cache: [AstroNet KV | real selected KV]
        cache = DynamicCache()
        for li in range(nl):
            K_cat_li = torch.cat(all_kv[li][0], dim=2)
            li_dev = K_cat_li.device
            li_idx = idx.to(li_dev)
            K_real = K_cat_li[:, :, li_idx, :]
            V_real = torch.cat(all_kv[li][1], dim=2)[:, :, li_idx, :]
            K_mem, V_mem = astro.generate_kv(li, (K_real.to(device), V_real.to(device)))
            cache.update(torch.cat([K_mem.to(li_dev), K_real], dim=2),
                         torch.cat([V_mem.to(li_dev), V_real], dim=2), li)
        return cache, n_mem + len(idx)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build cache (move idx to each layer's device for multi-GPU)
    cache = DynamicCache()
    for li in range(nl):
        K_cat = torch.cat(all_kv[li][0], dim=2)
        li_idx = idx.to(K_cat.device)
        K = K_cat[:, :, li_idx, :]
        V = torch.cat(all_kv[li][1], dim=2)[:, :, li_idx, :]
        cache.update(K, V, li)

    return cache, len(idx)


def generate_answer(model, tokenizer, question, cache, prefix_len, device, max_gen=32):
    """Generate answer with KV cache."""
    prompt = f"Question: {question}\nAnswer:"
    ids = tokenizer(prompt, return_tensors='pt', max_length=256, truncation=True).to(device)
    pos = torch.arange(prefix_len, prefix_len + ids['input_ids'].shape[1], device=device).unsqueeze(0)
    cur = ids['input_ids']; cc = cache; gen = []
    with torch.no_grad():
        for _ in range(max_gen):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item())
            cur = nxt
            pos = torch.tensor([[prefix_len + ids['input_ids'].shape[1] + len(gen) - 1]], device=device)
            if nxt[0, 0].item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    # Truncate at first newline — standard for extractive QA
    # Prevents verbose models from generating follow-up questions that dilute F1
    text = text.split('\n')[0].strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tasks', nargs='+', default=['narrativeqa', 'hotpotqa', 'multifieldqa_en'])
    parser.add_argument('--k', type=int, default=300)
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--methods', nargs='+',
                        default=['streaming_llm', 'h2o', 'snapkv', 'multiplicative'])
    parser.add_argument('--chunk_size', type=int, default=384)
    parser.add_argument('--hybrid_checkpoint', default=None,
                        help='Path to AstroNet hybrid checkpoint for "hybrid" method')
    parser.add_argument('--sense_layer', type=int, default=14)
    parser.add_argument('--n_mem', type=int, default=16, help='Number of memory tokens')
    parser.add_argument('--attn_dim', type=int, default=256, help='Attention/bottleneck dimension')
    parser.add_argument('--alpha_override', type=float, default=None,
                        help='Override learned alpha for EMA (e.g. 0.05 for long docs)')
    parser.add_argument('--memory_bank', action='store_true',
                        help='Enable memory bank mode (per-window summaries + cross-attention)')
    parser.add_argument('--gated', action='store_true',
                        help='Enable gated Ca2+ update (per-dimension forget/input gates)')
    parser.add_argument('--additive_mem', action='store_true',
                        help='Add memory tokens on top of k real tokens (k+16) instead of replacing')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--save_path', default=None,
                        help='Override default output path logs/results/longbench_<model>_k<k>.json')
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path)
    print(f"Loading {args.model_path}", flush=True)
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

    # Load AstroNet hybrid if requested
    astro = None
    if 'hybrid' in args.methods and args.hybrid_checkpoint:
        from training.train_hybrid import AstroHybrid
        import bitsandbytes as bnb_lib
        hidden_dim = model.config.hidden_size
        nl = model.config.num_hidden_layers
        nkv = model.config.num_key_value_heads
        hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
        astro = AstroHybrid(
            hidden_dim=hidden_dim, n_mem_tokens=args.n_mem, attn_dim=args.attn_dim,
            n_kv_heads=nkv, head_dim=hd, n_layers=nl,
            inject_layers=[nl//4, nl//2, 3*nl//4, nl-2],
        ).to(device)
        astro.extract_model_weights(model, device)
        astro.load_state_dict(torch.load(args.hybrid_checkpoint, map_location=device), strict=False)
        if hasattr(args, 'alpha_override') and args.alpha_override is not None:
            import math
            with torch.no_grad():
                logit_val = math.log(args.alpha_override / (1 - args.alpha_override))
                astro.log_alpha.fill_(logit_val)
            print(f"  Alpha overridden to {astro.alpha.item():.4f}", flush=True)
        if hasattr(args, 'memory_bank') and args.memory_bank:
            astro.use_memory_bank = True
            print(f"  Memory bank mode ENABLED", flush=True)
        if hasattr(args, 'gated') and args.gated:
            astro.use_gated = True
            print(f"  Gated Ca2+ mode ENABLED", flush=True)
        astro.eval()
        print(f"Loaded AstroNet hybrid: {astro.parameter_count():,} params (alpha={astro.alpha.item():.3f})", flush=True)

    all_results = {}

    for task in args.tasks:
        print(f"\n{'='*50}\nTask: {task}\n{'='*50}", flush=True)

        # Load LongBench data
        data = load_dataset('THUDM/LongBench', task, split='test', trust_remote_code=True)
        if len(data) > args.max_samples:
            data = data.select(range(args.max_samples))
        print(f"  {len(data)} samples", flush=True)

        task_results = {}
        task_raw = {}  # method -> list of {pred, gold, f1} for offline rescoring
        for method in args.methods:
            scores = []
            raw_records = []
            for si, sample in enumerate(data):
                context = sample['context']
                question = sample['input']
                answers = sample['answers']

                # Chunk context into windows
                windows = chunk_text(context, tokenizer, max_chunk_tokens=args.chunk_size)
                if len(windows) == 0:
                    continue

                cache, prefix_len = process_windows_and_select(
                    model, tokenizer, windows, question, args.k, method, device,
                    astro=astro, sense_layer=args.sense_layer)
                if cache is None:
                    continue

                pred = generate_answer(model, tokenizer, question, cache, prefix_len,
                                       device, max_gen=TASKS[task]['max_gen'])
                score = f1_score(pred, answers)
                scores.append(score)
                raw_records.append({
                    'idx': si,
                    'question': question,
                    'pred': pred,
                    'gold': answers,
                    'f1': float(score) * 100.0,
                })

                if (si + 1) % 25 == 0:
                    mean_f1 = np.mean(scores) * 100
                    print(f'    {method} {si+1}/{len(data)}: F1={mean_f1:.1f}', flush=True)

            mean_f1 = np.mean(scores) * 100 if scores else 0
            task_results[method] = mean_f1
            task_raw[method] = raw_records
            print(f'  {method}: F1={mean_f1:.1f}', flush=True)

        all_results[task] = task_results
        if 'raw' not in all_results:
            all_results['raw'] = {}
        all_results['raw'][task] = task_raw

    # Save
    save_path = args.save_path or f'logs/results/longbench_{model_name}_k{args.k}.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({'model': model_name, 'k': args.k, 'results': all_results}, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)

    # Summary
    print(f"\n{'='*50}\nSUMMARY\n{'='*50}", flush=True)
    for task, res in all_results.items():
        print(f"\n{task}:")
        for method, f1 in sorted(res.items()):
            print(f"  {method:20s}: {f1:.1f}")


if __name__ == '__main__':
    main()
