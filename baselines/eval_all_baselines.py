"""Unified baseline evaluation for multi-window SQuAD.

Implements all standard KV cache compression baselines in our multi-window
sequential processing setup, for fair comparison with AstroNet.

Baselines:
  - No Memory: question only, no context
  - Full KV Cache: all windows concatenated (oracle upper bound)
  - StreamingLLM: first 4 sink tokens + last k-4 tokens
  - H2O: cumulative attention scoring (heavy hitters + recent)
  - SnapKV: observation-window-driven attention (question Q @ fact K)
  - Multiplicative: H2O × SnapKV (our zero-shot method)
  - RAG k=1: retrieve best window via sentence-transformer
  - AstroNet Hybrid: 16 learned KV + 284 multiplicative (needs checkpoint)

Reference papers:
  - H2O: Zhang et al., NeurIPS 2023
  - SnapKV: Li et al., NeurIPS 2024
  - StreamingLLM: Xiao et al., ICLR 2024
"""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from sentence_transformers import SentenceTransformer
from data.real_qa import generate_squad_dataset


def generate_with_cache(model, tokenizer, input_ids, cache, prefix_len, device, max_tokens=20):
    """Generate tokens with pre-filled KV cache."""
    embed_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(embed_device)
    pos = torch.arange(prefix_len, prefix_len + input_ids.shape[1], device=embed_device).unsqueeze(0)
    cur = input_ids; cc = cache; gen = []
    with torch.no_grad():
        for _ in range(max_tokens):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item())
            cur = nxt.to(embed_device)
            pos = torch.tensor([[prefix_len + input_ids.shape[1] + len(gen) - 1]], device=embed_device)
            if nxt[0, 0].item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def process_windows(model, tokenizer, windows, device):
    """Process fact windows, collecting KV cache and attention scores."""
    nl = model.config.num_hidden_layers
    all_kv = {li: ([], []) for li in range(nl)}
    all_attn = []
    window_boundaries = []
    offset = 0

    for wi, window in enumerate(windows):
        ids = tokenizer(window, return_tensors='pt', max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True, output_attentions=True)
        sl = ids['input_ids'].shape[1]
        window_boundaries.append((offset, offset + sl))
        offset += sl

        imp = torch.zeros(sl, device=device)
        for la in out.attentions:
            a = la[0].to(device)
            if not a.isnan().any():
                imp += a.sum(dim=(0, 1))
        all_attn.append(imp)

        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])

    return all_kv, all_attn, window_boundaries, offset


def build_cache_from_indices(all_kv, idx, nl):
    """Build DynamicCache from selected token indices."""
    cache = DynamicCache()
    for li in range(nl):
        K = torch.cat(all_kv[li][0], dim=2)
        V = torch.cat(all_kv[li][1], dim=2)
        li_idx = idx.to(K.device)
        cache.update(K[:, :, li_idx, :], V[:, :, li_idx, :], li)
    return cache


# ============================================================
# Baseline implementations
# ============================================================

def select_streaming_llm(total, k, n_sink=4):
    """StreamingLLM: keep first n_sink tokens + last (k - n_sink) tokens."""
    if total <= k:
        return torch.arange(total)
    sink = torch.arange(n_sink)
    recent = torch.arange(total - (k - n_sink), total)
    return torch.cat([sink, recent])


def select_h2o(all_attn, k, window_boundaries, n_sink=4, recent_ratio=0.3):
    """H2O: cumulative attention scoring (heavy hitters + recent window).
    Faithful to Zhang et al. NeurIPS 2023: keeps attention sinks, recent tokens,
    and heavy hitters by cumulative post-softmax attention scores.
    In multi-window: 'recent' = tokens from the last fact window."""
    heur = torch.cat(all_attn)
    total = heur.shape[0]
    if total <= k:
        return torch.arange(total, device=heur.device)

    n_recent = int(k * recent_ratio)
    n_heavy = k - n_sink - n_recent
    if n_heavy < 0:
        n_heavy = 0
        n_recent = k - n_sink

    # Recent: last fact window
    last_start, last_end = window_boundaries[-1]
    n_recent = min(n_recent, last_end - last_start)
    recent_idx = torch.arange(last_start, last_start + n_recent, device=heur.device)

    # Heavy hitters: exclude sinks and recent window
    scores = heur.clone()
    scores[:n_sink] = -1e9
    scores[last_start:last_end] = -1e9
    n_available = (scores > -1e8).sum().item()
    _, heavy_idx = scores.topk(min(n_heavy, n_available))

    sink_idx = torch.arange(n_sink, device=heur.device)
    idx = torch.cat([sink_idx, heavy_idx, recent_idx]).unique().sort().values
    return idx[:k]


def select_snapkv(model, all_kv, question, tokenizer, k, device,
                   window_boundaries, n_sink=4, recent_ratio=0.2):
    """SnapKV: observation-window-driven selection with softmax + pooling.
    Faithful to Li et al. NeurIPS 2024: uses post-softmax attention from
    observation window (= question), avg-pooling for smoothing.
    In multi-window: observation window = question tokens."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                      max_length=128, truncation=True).to(device)
    with torch.no_grad():
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)

    total = torch.cat(all_kv[inject[0]][0], dim=2).shape[2]
    cross = torch.zeros(total, device=device)

    for li in inject:
        layer_device = all_kv[li][0][0].device
        Q = model.model.layers[li].self_attn.q_proj(
            q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
        K = torch.cat(all_kv[li][0], dim=2)[0]
        for hi in range(nkv):
            sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                              K[hi].float().T) / math.sqrt(hd)
            # Post-softmax (faithful to SnapKV)
            attn_w = torch.softmax(sc, dim=-1)
            cross += attn_w.sum(dim=(0, 1)).to(device)

    # Avg-pool smoothing (kernel_size=5, like SnapKV)
    if total > 5:
        cross_smooth = torch.nn.functional.avg_pool1d(
            cross.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2, stride=1
        ).squeeze()
    else:
        cross_smooth = cross

    cross_smooth[:n_sink] = -1e9

    # Keep recent window (like SnapKV keeps observation window)
    n_recent = int(k * recent_ratio)
    n_select = k - n_sink - n_recent

    last_start, last_end = window_boundaries[-1]
    n_recent = min(n_recent, last_end - last_start)
    recent_idx = torch.arange(last_start, last_start + n_recent, device=device)

    scores = cross_smooth.clone()
    scores[last_start:last_end] = -1e9
    n_available = (scores > -1e8).sum().item()
    _, top_idx = scores.topk(min(n_select, n_available))

    sink_idx = torch.arange(n_sink, device=device)
    idx = torch.cat([sink_idx, top_idx, recent_idx]).unique().sort().values
    return idx[:k]


def select_multiplicative(all_attn, model, all_kv, question, tokenizer, k, device,
                          n_sink=4, recent_ratio=0.2, smooth_kernel=5):
    """AstroNet S1 selection (position-robust formulation).

    The original multiplicative form h*c was found to be position-fragile because
    cumulative attention h has a within-window bias from the causal mask: tokens
    early in any window receive attention from many subsequent queries, creating
    a per-window position artifact unrelated to query relevance.

    The position-robust formulation uses softmax-normalized cross-window query-key
    relevance summed across multiple injection layers, with optional avg-pool
    smoothing and a recent-window keep heuristic, structurally similar to SnapKV
    but with multi-layer aggregation.
    """
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    total = torch.cat(all_kv[inject[0]][0], dim=2).shape[2]

    q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                      max_length=128, truncation=True).to(device)
    with torch.no_grad():
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)

    cross = torch.zeros(total, device=device)
    for li in inject:
        layer_device = all_kv[li][0][0].device
        Q = model.model.layers[li].self_attn.q_proj(
            q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
        K = torch.cat(all_kv[li][0], dim=2)[0]
        for hi in range(nkv):
            sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                              K[hi].float().T) / math.sqrt(hd)
            attn_w = torch.softmax(sc, dim=-1)
            cross += attn_w.sum(dim=(0, 1)).to(device)

    if smooth_kernel and total > smooth_kernel:
        cross = torch.nn.functional.avg_pool1d(
            cross.unsqueeze(0).unsqueeze(0),
            kernel_size=smooth_kernel,
            padding=smooth_kernel // 2,
            stride=1,
        ).squeeze()

    cross[:n_sink] = -1e9
    last_window_size = all_kv[inject[0]][0][-1].shape[2]
    last_start = total - last_window_size
    n_recent = min(int(k * recent_ratio), last_window_size)
    if n_recent > 0:
        recent = torch.arange(last_start, last_start + n_recent, device=device)
    else:
        recent = torch.empty(0, dtype=torch.long, device=device)
    scores = cross.clone()
    scores[last_start:total] = -1e9
    n_select = k - n_sink - n_recent
    n_avail = (scores > -1e8).sum().item()
    if n_avail < n_select:
        n_select = n_avail
    _, top = scores.topk(min(n_select, scores.shape[0]))
    sink_idx = torch.arange(n_sink, device=device)
    idx = torch.cat([sink_idx, top, recent]).unique().sort().values
    if total <= k:
        return torch.arange(total, device=device)
    return idx[:k]


def eval_rag(model, tokenizer, samples, encoder, device, rag_k=1):
    """RAG baseline: retrieve top-k windows via sentence embedding."""
    correct = 0
    for s in samples:
        past = s.windows[:-1]
        q_emb = encoder.encode(s.question, convert_to_numpy=True)
        p_embs = encoder.encode(past, convert_to_numpy=True)
        sims = (p_embs / (np.linalg.norm(p_embs, axis=1, keepdims=True) + 1e-8)) @ \
               (q_emb / (np.linalg.norm(q_emb) + 1e-8))
        top_indices = np.argsort(sims)[-rag_k:][::-1]
        context = "\n\n".join([past[i] for i in sorted(top_indices)])
        text = f'{context}\n\nBased on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        r_ids = tokenizer(text, return_tensors='pt', max_length=768, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(input_ids=r_ids['input_ids'], max_new_tokens=20,
                                  do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out[0][r_ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if s.answer.lower() in gen.lower():
            correct += 1
    return correct / len(samples)


def eval_no_memory(model, tokenizer, samples, device):
    """No memory baseline."""
    embed_device = model.get_input_embeddings().weight.device
    correct = 0
    for s in samples:
        text = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        ids = tokenizer(text, return_tensors='pt', max_length=384, truncation=True).to(embed_device)
        with torch.no_grad():
            out = model.generate(input_ids=ids['input_ids'], max_new_tokens=20,
                                  do_sample=False, pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if s.answer.lower() in gen.lower():
            correct += 1
    return correct / len(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=300, help='KV budget for all methods')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--methods', nargs='+',
                        default=['no_memory', 'streaming_llm', 'h2o', 'snapkv',
                                 'multiplicative', 'rag_k1'],
                        help='Which baselines to run')
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
    nl = model.config.num_hidden_layers

    encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') if 'rag_k1' in args.methods else None
    samples = generate_squad_dataset(n_samples=args.n_samples, n_windows=5,
                                      vary_distance=True, seed=args.seed, split='validation')
    print(f"Loaded {len(samples)} samples, k={args.k}", flush=True)

    results = {}

    # No memory
    if 'no_memory' in args.methods:
        print(f"\n{'='*50}\nNo Memory\n{'='*50}", flush=True)
        t0 = time.time()
        results['no_memory'] = eval_no_memory(model, tokenizer, samples, device)
        print(f"  => {results['no_memory']:.1%} ({time.time()-t0:.0f}s)", flush=True)

    # KV-cache methods: process windows once, then select differently
    kv_methods = [m for m in args.methods if m in ['streaming_llm', 'h2o', 'snapkv', 'multiplicative']]
    if kv_methods:
        correct = {m: 0 for m in kv_methods}

        for si, s in enumerate(samples):
            fact_windows = s.windows[:-1]
            all_kv, all_attn, boundaries, total = process_windows(
                model, tokenizer, fact_windows, device)

            query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
            fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)

            for method in kv_methods:
                if method == 'streaming_llm':
                    idx = select_streaming_llm(total, args.k).to(device)
                elif method == 'h2o':
                    idx = select_h2o(all_attn, args.k, boundaries)
                elif method == 'snapkv':
                    idx = select_snapkv(model, all_kv, s.question, tokenizer, args.k, device, boundaries)
                elif method == 'multiplicative':
                    idx = select_multiplicative(all_attn, model, all_kv, s.question,
                                                 tokenizer, args.k, device)

                cache = build_cache_from_indices(all_kv, idx, nl)
                k = len(idx)
                text = generate_with_cache(model, tokenizer, fq['input_ids'], cache, k, device)
                if s.answer.lower() in text.lower():
                    correct[method] += 1

            if (si + 1) % 50 == 0:
                parts = [f'{m}={correct[m]}/{si+1}' for m in kv_methods]
                print(f"  {si+1}/{len(samples)}: {', '.join(parts)}", flush=True)

        for m in kv_methods:
            results[m] = correct[m] / len(samples)
            print(f"\n{m}: {results[m]:.1%}", flush=True)

    # RAG
    if 'rag_k1' in args.methods:
        print(f"\n{'='*50}\nRAG k=1\n{'='*50}", flush=True)
        t0 = time.time()
        results['rag_k1'] = eval_rag(model, tokenizer, samples, encoder, device, rag_k=1)
        print(f"  => {results['rag_k1']:.1%} ({time.time()-t0:.0f}s)", flush=True)

    # Save results
    save_path = f'./logs/results/baselines_{model_name}_k{args.k}.json'
    save_data = {
        'model': model_name, 'n_samples': len(samples), 'seed': args.seed,
        'k': args.k, 'results': results,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)

    print(f"\n{'='*50}\nSUMMARY: {model_name} (k={args.k})\n{'='*50}", flush=True)
    for m, acc in sorted(results.items()):
        print(f"  {m:20s}: {acc:.1%}", flush=True)


if __name__ == '__main__':
    main()
