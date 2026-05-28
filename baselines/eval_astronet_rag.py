"""AstroNet + RAG complementarity test.

Hypothesis: AstroNet's token-level KV selection is orthogonal to RAG's passage-level
retrieval. Combining them should outperform either alone, especially when the budget
is constrained.

Conditions (all on SQuAD-style multi-window context):
  1. full_context    — model sees all windows concatenated (upper bound)
  2. rag_k1          — BM25 retrieves top-1 window, model sees only that (no KV compression)
  3. astronet_hybrid — AstroNet S1+S2 on all windows, k=300
  4. rag_then_astro  — BM25 retrieves top-1 window, then AstroNet hybrid is applied (k=300, but window already ~384 tokens → mostly trivial)
  5. astro_with_rag_anchor — AstroNet S1 but RAG-retrieved window is guaranteed in selection (anchor)

BM25 is used to avoid sentence-transformer dependency. Question is the query.
"""
import sys; sys.path.insert(0, '.')
import os, json, math, argparse, random, re
from collections import Counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from baselines.eval_needle import generate_haystack, _select_kv, NEEDLES


def tokenize_simple(text):
    return re.findall(r'\b\w+\b', text.lower())


def bm25_scores(query, docs, k1=1.5, b=0.75):
    """Compute BM25 scores for query over docs (list of strings)."""
    q_toks = tokenize_simple(query)
    doc_toks = [tokenize_simple(d) for d in docs]
    avgdl = sum(len(d) for d in doc_toks) / max(len(doc_toks), 1)
    df = Counter()
    for d in doc_toks:
        for w in set(d):
            df[w] += 1
    N = len(doc_toks)
    scores = []
    for d in doc_toks:
        tf = Counter(d)
        dl = len(d)
        s = 0.0
        for q in q_toks:
            if q not in tf: continue
            idf = math.log(1 + (N - df[q] + 0.5) / (df[q] + 0.5))
            num = tf[q] * (k1 + 1)
            den = tf[q] + k1 * (1 - b + b * dl / max(avgdl, 1))
            s += idf * num / max(den, 1e-9)
        scores.append(s)
    return scores


def gen_pred(model, tokenizer, cache, prefix_len, question, device, max_new=30):
    query = f'Based on what you read, answer the question.\nQuestion: {question}\nAnswer:'
    fq = tokenizer(query, return_tensors='pt', max_length=256, truncation=True).to(device)
    pos = torch.arange(prefix_len, prefix_len + fq['input_ids'].shape[1], device=device).unsqueeze(0)
    cur = fq['input_ids']; cc = cache; gen = []
    with torch.no_grad():
        for _ in range(max_new):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item()); cur = nxt
            pos = torch.tensor([[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]], device=device)
            if nxt[0, 0].item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def run_full_context(model, tokenizer, windows, question, device):
    full = ' '.join(windows)
    ids = tokenizer(full, return_tensors='pt', max_length=8192, truncation=True).to(device)
    with torch.no_grad():
        out = model(input_ids=ids['input_ids'], use_cache=True)
    cache = out.past_key_values
    prefix_len = ids['input_ids'].shape[1]
    return gen_pred(model, tokenizer, cache, prefix_len, question, device)


def run_rag_only(model, tokenizer, windows, question, device, top_k=1):
    scores = bm25_scores(question, windows)
    top_idx = sorted(range(len(windows)), key=lambda i: scores[i], reverse=True)[:top_k]
    retrieved = ' '.join(windows[i] for i in sorted(top_idx))
    ids = tokenizer(retrieved, return_tensors='pt', max_length=2048, truncation=True).to(device)
    with torch.no_grad():
        out = model(input_ids=ids['input_ids'], use_cache=True)
    cache = out.past_key_values
    prefix_len = ids['input_ids'].shape[1]
    return gen_pred(model, tokenizer, cache, prefix_len, question, device)


def run_astronet(model, tokenizer, windows, question, k, device, astro, sense_layer, rag_anchor_idx=None, all_windows_for_anchor=None):
    """Standard hybrid selection. If rag_anchor_idx given, ensure that window's tokens
    are guaranteed in the selection (anchor)."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl//4, nl//2, 3*nl//4, nl-2]

    all_kv = {li: ([], []) for li in range(nl)}
    window_boundaries = []
    offset = 0
    astro.reset_state()
    for window in windows:
        ids = tokenizer(window, return_tensors='pt', max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True, output_hidden_states=True)
        sl = ids['input_ids'].shape[1]
        window_boundaries.append((offset, offset + sl))
        offset += sl
        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])
        with torch.no_grad():
            hidden = out.hidden_states[sense_layer]
            sensed = astro.sense(hidden)
            astro.update_state(sensed)
    total = offset

    q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt', max_length=128, truncation=True).to(device)
    with torch.no_grad():
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
    cross = torch.zeros(total, device=device)
    for li in inject:
        layer_device = all_kv[li][0][0].device
        Q = model.model.layers[li].self_attn.q_proj(
            q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
        K = torch.cat(all_kv[li][0], dim=2)[0]
        for hi in range(nkv):
            sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(), K[hi].float().T) / math.sqrt(hd)
            cross += torch.softmax(sc, dim=-1).sum(dim=(0, 1)).to(device)
    if total > 5:
        cross = torch.nn.functional.avg_pool1d(
            cross.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2, stride=1
        ).squeeze()
    n_sink = 4
    cross[:n_sink] = -1e9
    last_start = window_boundaries[-1][0]
    last_end = window_boundaries[-1][1]
    last_window_size = last_end - last_start

    # If rag_anchor: boost scores of the retrieved window
    if rag_anchor_idx is not None and rag_anchor_idx < len(window_boundaries):
        s, e = window_boundaries[rag_anchor_idx]
        cross[s:e] += 1e5  # forces selection

    n_mem = astro.n_mem_tokens
    k_real = k - n_mem
    n_recent = min(int(k_real * 0.2), last_window_size)
    recent_idx = torch.arange(last_start, last_start + n_recent, device=device)
    scores = cross.clone()
    scores[last_start:total] = -1e9
    n_select = max(k_real - n_sink - n_recent, 0)
    n_avail = (scores > -1e8).sum().item()
    n_select = min(n_select, n_avail, scores.shape[0])
    _, top = scores.topk(n_select) if n_select > 0 else (None, torch.empty(0, dtype=torch.long, device=device))
    sink_idx = torch.arange(n_sink, device=device)
    idx = torch.cat([sink_idx, top, recent_idx]).unique().sort().values[:k_real]

    cache = DynamicCache()
    for li in range(nl):
        K_real, V_real = _select_kv(all_kv, li, idx)
        li_dev = K_real.device
        K_mem, V_mem = astro.generate_kv(li, (K_real.to(device), V_real.to(device)))
        cache.update(torch.cat([K_mem.to(li_dev), K_real], dim=2),
                     torch.cat([V_mem.to(li_dev), V_real], dim=2), li)
    prefix_len = n_mem + len(idx)
    return gen_pred(model, tokenizer, cache, prefix_len, question, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--hybrid_checkpoint', required=True)
    parser.add_argument('--n_windows', type=int, default=10)
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--k', type=int, default=300)
    parser.add_argument('--sense_layer', type=int, default=None)
    parser.add_argument('--attn_dim', type=int, default=256)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--save_tag', required=True)
    args = parser.parse_args()

    print(f"[astro-rag] {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=bnb, device_map={'':args.device}, torch_dtype=torch.float16)
    model.eval()
    device = model.get_input_embeddings().weight.device

    from training.train_hybrid import AstroHybrid
    nl = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
    sense_layer = args.sense_layer or nl // 2
    astro = AstroHybrid(hidden_dim=hidden_dim, n_mem_tokens=16, attn_dim=args.attn_dim,
                       n_kv_heads=nkv, head_dim=hd, n_layers=nl,
                       inject_layers=[nl//4, nl//2, 3*nl//4, nl-2]).to(device)
    astro.extract_model_weights(model, device)
    astro.load_state_dict(torch.load(args.hybrid_checkpoint, map_location=device), strict=False)
    astro.eval()

    depths = [0, args.n_windows//4, args.n_windows//2, 3*args.n_windows//4, args.n_windows-1]
    conditions = ['full_context', 'rag_k1', 'astronet_hybrid', 'astro_with_rag_anchor']
    results = {c: [] for c in conditions}

    for trial in range(args.n_trials):
        for di, depth in enumerate(depths):
            seed = 42 + trial*100 + di
            needle_idx = (trial + di) % len(NEEDLES)
            windows, question, answer = generate_haystack(args.n_windows, depth, needle_idx=needle_idx, seed=seed)

            # Find RAG anchor via BM25
            bm = bm25_scores(question, windows)
            rag_top = max(range(len(windows)), key=lambda i: bm[i])

            try:
                pred_full = run_full_context(model, tokenizer, windows, question, device)
                results['full_context'].append(int(answer.lower() in pred_full.lower()))
            except Exception as e:
                results['full_context'].append(0)
                print(f"full_context err: {e}", flush=True)

            try:
                pred_rag = run_rag_only(model, tokenizer, windows, question, device, top_k=1)
                results['rag_k1'].append(int(answer.lower() in pred_rag.lower()))
            except Exception as e:
                results['rag_k1'].append(0)

            try:
                pred_astro = run_astronet(model, tokenizer, windows, question, args.k, device, astro, sense_layer)
                results['astronet_hybrid'].append(int(answer.lower() in pred_astro.lower()))
            except Exception as e:
                results['astronet_hybrid'].append(0)

            try:
                pred_combo = run_astronet(model, tokenizer, windows, question, args.k, device, astro, sense_layer,
                                          rag_anchor_idx=rag_top)
                results['astro_with_rag_anchor'].append(int(answer.lower() in pred_combo.lower()))
            except Exception as e:
                results['astro_with_rag_anchor'].append(0)

            if trial < 2:
                print(f"  trial={trial} d={depth} rag_top={rag_top} ans='{answer}': "
                      f"full={results['full_context'][-1]} rag={results['rag_k1'][-1]} "
                      f"astro={results['astronet_hybrid'][-1]} combo={results['astro_with_rag_anchor'][-1]}", flush=True)

    print(f"\n=== AGGREGATE astronet+rag {args.save_tag} ===", flush=True)
    for c in conditions:
        acc = sum(results[c]) / max(len(results[c]), 1)
        print(f"  {c}: {acc:.3f} ({sum(results[c])}/{len(results[c])})", flush=True)

    with open(f'logs/results/diag_astronet_rag_{args.save_tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved.", flush=True)


if __name__ == '__main__':
    main()
