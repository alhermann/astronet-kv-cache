"""Hybrid + RAG complementarity on LongBench HotpotQA.

Tests whether combining AstroNet's compressed hybrid cache with RAG-retrieved
passages outperforms either alone on a real multi-hop benchmark (HotpotQA),
where RAG retrieval is imperfect (multi-hop questions need information from
multiple passages, not all in the top-k retrieved set).

Compares:
    - hybrid_only       (k=300, AstroNet S1+S2 cache)
    - rag_k1            (top-1 retrieved passage in context, no cache eviction)
    - rag_k3            (top-3 retrieved passages in context, no cache eviction)
    - hybrid_plus_rag1  (AstroNet hybrid k=300 cache + top-1 RAG passage)
    - hybrid_plus_rag3  (AstroNet hybrid k=300 cache + top-3 RAG passages)
    - full_context      (full document, truncated to 4096 tokens, upper bound)
"""
import sys, os, json, argparse, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from datasets import load_dataset
from training.train_hybrid import AstroHybrid
from baselines.metrics import f1_score


def chunk_text(text, tokenizer, chunk_size=384):
    """Tokenize text and split into chunks of chunk_size tokens."""
    ids = tokenizer(text, return_tensors='pt').input_ids[0]
    chunks = []
    for i in range(0, ids.shape[0], chunk_size):
        chunk_ids = ids[i:i+chunk_size]
        if chunk_ids.shape[0] < 32:
            continue
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
    return chunks


def encode_query(encoder_model, encoder_tok, query, device):
    """Get a single embedding from sentence-transformer-style encoder."""
    inputs = encoder_tok(query, return_tensors='pt', truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = encoder_model(**inputs)
    emb = out.last_hidden_state[:, 0]
    return F.normalize(emb, dim=-1)[0]


def retrieve_topk(encoder_model, encoder_tok, query, chunks, device, top_k):
    """Score each chunk against query, return top_k indices."""
    q = encode_query(encoder_model, encoder_tok, query, device)
    scores = []
    for chunk in chunks:
        c = encode_query(encoder_model, encoder_tok, chunk, device)
        scores.append((q @ c).item())
    ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return ranked[:top_k]


def process_windows_with_kv(model, tokenizer, windows, device, max_window_len=384):
    """Forward-pass each window, accumulate KV across all windows."""
    nl = model.config.num_hidden_layers
    all_kv = {li: ([], []) for li in range(nl)}
    boundaries = []
    offset = 0
    hidden_states_per_window = []
    for w in windows:
        ids = tokenizer(w, return_tensors='pt', max_length=max_window_len,
                        truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True,
                        output_hidden_states=True)
        sl = ids['input_ids'].shape[1]
        boundaries.append((offset, offset + sl))
        offset += sl
        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])
        hidden_states_per_window.append(out.hidden_states)
    return all_kv, offset, boundaries, hidden_states_per_window


def compute_cross_score(model, tokenizer, all_kv, total, question, inject_layers, device):
    """Cross-window query-key score (the multiplicative selection's cross component)."""
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    q_ids = tokenizer(f'Question: {question}\nAnswer:', return_tensors='pt',
                       max_length=128, truncation=True).to(device)
    with torch.no_grad():
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
    cross = torch.zeros(total, device=device)
    for li in inject_layers:
        layer_dev = all_kv[li][0][0].device
        Q = model.model.layers[li].self_attn.q_proj(
            q_out.hidden_states[li][0].float().to(layer_dev)).half().view(-1, nq, hd)
        K = torch.cat(all_kv[li][0], dim=2)[0]
        for hi in range(nkv):
            sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                              K[hi].float().T) / math.sqrt(hd)
            cross += torch.softmax(sc, dim=-1).sum(dim=(0, 1)).to(device)
    if total > 5:
        cross = F.avg_pool1d(cross.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2, stride=1).squeeze()
    cross[:4] = -1e9
    return cross


def generate(model, tokenizer, past_key_values, prefix_len, question, device, max_new_tokens=64):
    q = f'Based on what you read, answer the question.\nQuestion: {question}\nAnswer:'
    fq = tokenizer(q, return_tensors='pt', max_length=256, truncation=True).to(device)
    pos = torch.arange(prefix_len, prefix_len + fq['input_ids'].shape[1],
                       device=device).unsqueeze(0)
    cur, cc, gen = fq['input_ids'], past_key_values, []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item())
            cur = nxt
            pos = torch.tensor([[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                               device=device)
            if nxt[0, 0].item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def hybrid_select_and_generate(model, tokenizer, astro, all_kv, total, last_start,
                                question, inject_layers, k, n_mem, device, extra_anchor_windows=None):
    """Stage 1 selection + Stage 2 generated tokens. Optionally prepend retrieved anchors."""
    cross = compute_cross_score(model, tokenizer, all_kv, total, question, inject_layers, device)
    nl = model.config.num_hidden_layers
    last_window_size = all_kv[inject_layers[0]][0][-1].shape[2]
    n_sink = 4
    k_real = k - n_mem
    n_recent = min(int(k_real * 0.2), last_window_size)
    recent_idx = torch.arange(last_start, last_start + n_recent, device=device)
    scores = cross.clone()
    scores[last_start:total] = -1e9
    n_select = max(k_real - n_sink - n_recent, 0)
    n_avail = (scores > -1e8).sum().item()
    n_select = min(n_select, n_avail, scores.shape[0])
    _, top = scores.topk(n_select) if n_select > 0 else \
             (None, torch.empty(0, dtype=torch.long, device=device))
    sink_idx = torch.arange(n_sink, device=device)
    idx = torch.cat([sink_idx, top, recent_idx]).unique().sort().values[:k_real]

    cache = DynamicCache()
    for li in range(nl):
        K = torch.cat(all_kv[li][0], dim=2)
        V = torch.cat(all_kv[li][1], dim=2)
        K_real, V_real = K[:, :, idx.to(K.device), :], V[:, :, idx.to(V.device), :]
        K_mem, V_mem = astro.generate_kv(li, (K_real.to(device), V_real.to(device)))
        cache.update(torch.cat([K_mem.to(K_real.device), K_real], dim=2),
                     torch.cat([V_mem.to(V_real.device), V_real], dim=2), li)
    prefix_len = n_mem + len(idx)

    # If RAG anchors are provided, prepend them to the question prompt
    q_prefix = ''
    if extra_anchor_windows:
        q_prefix = '\n'.join(extra_anchor_windows) + '\n'
    full_q = q_prefix + f'Question: {question}\nAnswer:'
    fq = tokenizer(full_q, return_tensors='pt', max_length=2048, truncation=True).to(device)
    pos = torch.arange(prefix_len, prefix_len + fq['input_ids'].shape[1],
                       device=device).unsqueeze(0)
    cur, cc, gen = fq['input_ids'], cache, []
    with torch.no_grad():
        for _ in range(64):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item())
            cur = nxt
            pos = torch.tensor([[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                               device=device)
            if nxt[0, 0].item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def rag_only_generate(model, tokenizer, retrieved_chunks, question, device):
    """RAG baseline: retrieved chunks + question in context, no cache eviction."""
    ctx = '\n'.join(retrieved_chunks)
    full = f'{ctx}\n\nQuestion: {question}\nAnswer:'
    ids = tokenizer(full, return_tensors='pt', max_length=4096, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(ids['input_ids'], max_new_tokens=64, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    ans = tokenizer.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return ans


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--hybrid_checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--task', default='hotpotqa')
    p.add_argument('--n_samples', type=int, default=50)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=256)
    p.add_argument('--chunk_size', type=int, default=384)
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', default=None)
    args = p.parse_args()

    print(f'Loading {args.model_path}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = str(model.get_input_embeddings().weight.device)

    # Sentence encoder for RAG retrieval
    print('Loading sentence-transformer for RAG retrieval', flush=True)
    from transformers import AutoModel as HFAutoModel
    enc_path = 'sentence-transformers/all-MiniLM-L6-v2'
    enc_tok = AutoTokenizer.from_pretrained(enc_path)
    enc_model = HFAutoModel.from_pretrained(enc_path).to(device).eval()

    # AstroNet
    nl = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
    sense_layer = nl // 2
    inject_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    from training.train_hybrid import AstroHybrid
    astro = AstroHybrid(hidden_dim=hidden_dim, n_mem_tokens=args.n_mem, attn_dim=args.attn_dim,
                         n_kv_heads=nkv, head_dim=hd, n_layers=nl,
                         inject_layers=inject_layers).to(device)
    astro.extract_model_weights(model, device)
    astro.load_state_dict(torch.load(args.hybrid_checkpoint, map_location=device), strict=False)
    astro.eval()
    print(f'AstroNet: {sum(p.numel() for p in astro.parameters())} params', flush=True)

    # Dataset
    print(f'Loading LongBench {args.task}', flush=True)
    ds = load_dataset('THUDM/LongBench', args.task, split='test')

    methods = ['hybrid_only', 'rag_k1', 'rag_k3', 'hybrid_plus_rag1', 'hybrid_plus_rag3']
    scores = {m: [] for m in methods}
    raw = []

    for idx, ex in enumerate(ds):
        if idx >= args.n_samples:
            break
        context, question, answers = ex['context'], ex['input'], ex['answers']
        chunks = chunk_text(context, tokenizer, chunk_size=args.chunk_size)
        if len(chunks) < 2:
            continue

        # Retrieve top-k passages
        ranked = retrieve_topk(enc_model, enc_tok, question, chunks, device, top_k=3)
        rag1 = [chunks[i] for i in ranked[:1]]
        rag3 = [chunks[i] for i in ranked[:3]]

        # Process windows for hybrid (Stage 2 sensing)
        astro.reset_state()
        all_kv, total, boundaries, _ = process_windows_with_kv(
            model, tokenizer, chunks, device, max_window_len=args.chunk_size)
        # Stage 2 sensing pass (re-run if needed; here we just use generated kv)
        # Run a sense pass on the last hidden state to populate state for hybrid
        with torch.no_grad():
            for w in chunks:
                ids = tokenizer(w, return_tensors='pt', max_length=args.chunk_size,
                                 truncation=True).to(device)
                out = model(input_ids=ids['input_ids'], use_cache=True,
                            output_hidden_states=True)
                hidden = out.hidden_states[sense_layer]
                sensed = astro.sense(hidden)
                astro.update_state(sensed)
        last_start = boundaries[-1][0]

        # 1) hybrid_only
        ans_h = hybrid_select_and_generate(
            model, tokenizer, astro, all_kv, total, last_start, question,
            inject_layers, args.k, args.n_mem, device, extra_anchor_windows=None)

        # 2) rag_k1
        ans_r1 = rag_only_generate(model, tokenizer, rag1, question, device)

        # 3) rag_k3
        ans_r3 = rag_only_generate(model, tokenizer, rag3, question, device)

        # 4) hybrid + rag1 anchor
        ans_hr1 = hybrid_select_and_generate(
            model, tokenizer, astro, all_kv, total, last_start, question,
            inject_layers, args.k, args.n_mem, device, extra_anchor_windows=rag1)

        # 5) hybrid + rag3 anchor
        ans_hr3 = hybrid_select_and_generate(
            model, tokenizer, astro, all_kv, total, last_start, question,
            inject_layers, args.k, args.n_mem, device, extra_anchor_windows=rag3)

        # Score F1 against gold answers
        record = {'idx': idx, 'question': question, 'gold': answers}
        for method_name, ans in [('hybrid_only', ans_h), ('rag_k1', ans_r1),
                                  ('rag_k3', ans_r3), ('hybrid_plus_rag1', ans_hr1),
                                  ('hybrid_plus_rag3', ans_hr3)]:
            ans_trunc = ans.split('\n')[0]
            best_f1 = max(f1_score(ans_trunc, gold) for gold in answers)
            scores[method_name].append(best_f1)
            record[method_name] = {'answer': ans_trunc, 'f1': best_f1}
        raw.append(record)

        if (idx + 1) % 5 == 0:
            mean_str = '  '.join(f"{m}={sum(scores[m])/len(scores[m]):.2f}" for m in methods if scores[m])
            print(f'[{idx+1}/{args.n_samples}] {mean_str}', flush=True)

    # Aggregate
    print('\n=== AGGREGATE hybrid+RAG on LongBench HotpotQA ===')
    summary = {}
    for m in methods:
        if scores[m]:
            mean = sum(scores[m]) / len(scores[m])
            summary[m] = {'mean_f1': mean, 'n': len(scores[m])}
            print(f'  {m}: F1={mean*100:.2f} (n={len(scores[m])})')

    out = {'config': vars(args), 'summary': summary, 'raw': raw}
    if args.save_path:
        with open(args.save_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Saved to {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
