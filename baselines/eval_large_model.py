"""Memory-efficient evaluation for 70B+ models.
SDPA path (no output_attentions) + pre-hooks on inject layers only:
1. Forward uses default SDPA which never materializes the full attention tensor
2. Pre-hooks capture hidden states for the 4 inject layers only
3. Heuristic = manual Q @ K^T computed post-hoc for those 4 layers, sum across heads
4. Skips the all-80-layers attention sum (irrelevant on big models per prior ablations)

This is the only path that fits a 72B 4-bit model on 2x24 GiB hardware."""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from sentence_transformers import SentenceTransformer
from data.real_qa import generate_squad_dataset


def eval_with_hooks(model, tokenizer, samples, k, device, methods=['multiplicative', 'rag_k1']):
    """SDPA forward + pre-hooks on inject layers only — fits 72B on 2x24 GiB."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    inject_set = set(inject)
    embed_device = model.get_input_embeddings().weight.device

    needs_rag = any(m.startswith('rag_') for m in methods)
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu') if needs_rag else None
    rag_ks = sorted({int(m.split('_k')[1]) for m in methods if m.startswith('rag_k')})

    # Per-forward storage: pre-hooks save hidden states for inject layers only.
    captured_hidden = {}

    def make_pre_hook(layer_idx):
        def pre_hook(module, args, kwargs):
            # First positional arg is hidden_states (1, sl, hidden_dim)
            h = args[0] if args else kwargs.get('hidden_states')
            if h is not None:
                captured_hidden[layer_idx] = h.detach()
        return pre_hook

    # Register pre-hooks ONLY on inject layers.
    hook_handles = []
    for li in inject:
        h = model.model.layers[li].self_attn.register_forward_pre_hook(
            make_pre_hook(li), with_kwargs=True)
        hook_handles.append(h)

    try:
        results = {m: 0 for m in methods}

        for si, s in enumerate(samples):
            fact_windows = s.windows[:-1]
            all_kv = {li: ([], []) for li in range(nl)}
            all_imp = []
            total_tokens = 0

            for wi, window in enumerate(fact_windows):
                ids = tokenizer(window, return_tensors='pt', max_length=256, truncation=True).to(embed_device)
                sl = ids['input_ids'].shape[1]

                captured_hidden.clear()
                with torch.no_grad():
                    # SDPA forward — no output_attentions, no full attn tensor materialized.
                    out = model(input_ids=ids['input_ids'], use_cache=True, output_attentions=False)

                    # Compute heuristic from inject layers only via manual Q @ K^T.
                    imp = torch.zeros(sl, device=embed_device)
                    for li in inject:
                        if li not in captured_hidden:
                            continue
                        h = captured_hidden[li]  # (1, sl, hidden)
                        layer_dev = model.model.layers[li].self_attn.q_proj.weight.device
                        Q = model.model.layers[li].self_attn.q_proj(
                            h.to(layer_dev).float()).half()  # (1, sl, nq*hd)
                        Q = Q.view(1, sl, nq, hd).transpose(1, 2)[0]  # (nq, sl, hd)
                        K = out.past_key_values[li][0][0]  # (nkv, sl, hd)
                        for hi in range(nkv):
                            sc = torch.matmul(
                                Q[hi*qpk:(hi+1)*qpk].float(),
                                K[hi].float().T) / math.sqrt(hd)  # (qpk, sl, sl)
                            imp += sc.sum(dim=(0, 1)).to(embed_device)
                        del Q, K
                    all_imp.append(imp)
                    captured_hidden.clear()

                    # Keep only KV pairs (small).
                    for li in range(nl):
                        all_kv[li][0].append(out.past_key_values[li][0])
                        all_kv[li][1].append(out.past_key_values[li][1])

                    del out
                    torch.cuda.empty_cache()

                total_tokens += sl

            heur = torch.cat(all_imp)

            if 'multiplicative' in methods:
                # Cross-window Q@K scoring.
                q_ids = tokenizer(f'Question: {s.question}\nAnswer:', return_tensors='pt',
                                  max_length=128, truncation=True).to(embed_device)
                captured_hidden.clear()
                with torch.no_grad():
                    q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True,
                                  output_attentions=False)

                cross = torch.zeros(total_tokens, device=embed_device)
                for li in inject:
                    layer_device = all_kv[li][0][0].device
                    Q = model.model.layers[li].self_attn.q_proj(
                        q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
                    K = torch.cat(all_kv[li][0], dim=2)[0]
                    for hi in range(nkv):
                        sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                          K[hi].float().T) / math.sqrt(hd)
                        cross += sc.sum(dim=(0, 1)).to(embed_device)
                del q_out; torch.cuda.empty_cache()

                cross[:4] = -1e9
                mult = torch.clamp(heur, min=0) * torch.clamp(cross, min=0)
                mult[:4] = -1e9
                _, idx = mult.topk(min(k, total_tokens))
                idx = idx.sort().values

                # Build cache.
                cache = DynamicCache()
                for li in range(nl):
                    K_cat = torch.cat(all_kv[li][0], dim=2)
                    V_cat = torch.cat(all_kv[li][1], dim=2)
                    li_idx = idx.to(K_cat.device)
                    cache.update(K_cat[:, :, li_idx, :], V_cat[:, :, li_idx, :], li)
                del K_cat, V_cat; torch.cuda.empty_cache()

                # Generate.
                query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
                fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(embed_device)
                prefix_len = len(idx)
                pos = torch.arange(prefix_len, prefix_len + fq['input_ids'].shape[1],
                                   device=embed_device).unsqueeze(0)
                cur = fq['input_ids']; cc = cache; gen = []
                captured_hidden.clear()
                with torch.no_grad():
                    for _ in range(20):
                        o = model(input_ids=cur, past_key_values=cc, position_ids=pos,
                                  output_attentions=False)
                        cc = o.past_key_values
                        nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                        gen.append(nxt[0, 0].item())
                        cur = nxt.to(embed_device)
                        pos = torch.tensor([[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                                           device=embed_device)
                        if nxt[0, 0].item() == tokenizer.eos_token_id:
                            break
                text = tokenizer.decode(gen, skip_special_tokens=True).strip()
                if s.answer.lower() in text.lower():
                    results['multiplicative'] += 1
                del cache, cc; torch.cuda.empty_cache()

            # === streaming_llm: zero-cost selection — first n_sink + last (k-n_sink) tokens.
            if 'streaming_llm' in methods:
                n_sink = 4
                if total_tokens <= k:
                    sl_idx = torch.arange(total_tokens, device=embed_device)
                else:
                    sink_idx = torch.arange(n_sink, device=embed_device)
                    recent_idx = torch.arange(total_tokens - (k - n_sink), total_tokens, device=embed_device)
                    sl_idx = torch.cat([sink_idx, recent_idx]).unique().sort().values

                cache = DynamicCache()
                for li in range(nl):
                    K_cat = torch.cat(all_kv[li][0], dim=2)
                    V_cat = torch.cat(all_kv[li][1], dim=2)
                    li_idx = sl_idx.to(K_cat.device)
                    cache.update(K_cat[:, :, li_idx, :], V_cat[:, :, li_idx, :], li)
                del K_cat, V_cat; torch.cuda.empty_cache()

                query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
                fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(embed_device)
                prefix_len = len(sl_idx)
                pos = torch.arange(prefix_len, prefix_len + fq['input_ids'].shape[1],
                                   device=embed_device).unsqueeze(0)
                cur = fq['input_ids']; cc = cache; gen = []
                with torch.no_grad():
                    for _ in range(20):
                        o = model(input_ids=cur, past_key_values=cc, position_ids=pos,
                                  output_attentions=False)
                        cc = o.past_key_values
                        nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                        gen.append(nxt[0, 0].item())
                        cur = nxt.to(embed_device)
                        pos = torch.tensor([[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                                           device=embed_device)
                        if nxt[0, 0].item() == tokenizer.eos_token_id:
                            break
                text = tokenizer.decode(gen, skip_special_tokens=True).strip()
                if s.answer.lower() in text.lower():
                    results['streaming_llm'] += 1
                del cache, cc; torch.cuda.empty_cache()

            # === RAG (k=1, k=3, ...): retrieve top-rag_k windows by SBERT similarity.
            if rag_ks:
                past = s.windows[:-1]
                q_emb = encoder.encode(s.question, convert_to_numpy=True)
                p_embs = encoder.encode(past, convert_to_numpy=True)
                sims = (p_embs / (np.linalg.norm(p_embs, axis=1, keepdims=True) + 1e-8)) @ \
                       (q_emb / (np.linalg.norm(q_emb) + 1e-8))
                for rk in rag_ks:
                    method_name = f'rag_k{rk}'
                    if method_name not in methods:
                        continue
                    top_idx = np.argsort(sims)[-rk:][::-1]
                    top_windows = '\n\n'.join([past[i] for i in top_idx])
                    rag_text = f'{top_windows}\n\nBased on what you read, answer the following question.\nQuestion: {s.question}\nAnswer:'
                    r_ids = tokenizer(rag_text, return_tensors='pt', max_length=512 + (rk-1)*256, truncation=True).to(embed_device)
                    with torch.no_grad():
                        r_out = model.generate(input_ids=r_ids['input_ids'], max_new_tokens=20,
                                                do_sample=False, pad_token_id=tokenizer.pad_token_id)
                    text = tokenizer.decode(r_out[0][r_ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    if s.answer.lower() in text.lower():
                        results[method_name] += 1
                    del r_out; torch.cuda.empty_cache()

            del all_kv, all_imp, heur
            torch.cuda.empty_cache()

            if (si + 1) % 10 == 0:
                parts = [f'{m}={results[m]}/{si+1}' for m in methods]
                print(f'  {si+1}/{len(samples)}: {", ".join(parts)}', flush=True)

        n = len(samples)
        final = {m: results[m] / n for m in methods}
        return final
    finally:
        for h in hook_handles:
            h.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=300)
    parser.add_argument('--methods', nargs='+',
                        default=['streaming_llm', 'multiplicative', 'rag_k1', 'rag_k3'])
    parser.add_argument('--gpu0_mem', default='22GiB')
    parser.add_argument('--gpu1_mem', default='23GiB')
    parser.add_argument('--tag', default='full')
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path)
    print(f"Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    # SDPA is the default and most memory-efficient kernel — never materializes the
    # full attention tensor. We compute the heuristic via pre-hooks on inject layers
    # only (manual Q @ K^T post-hoc), avoiding the all-80-layers attention storage.
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
        quantization_config=bnb, device_map='sequential',
        max_memory={0: args.gpu0_mem, 1: args.gpu1_mem},
        torch_dtype=torch.float16,
        attn_implementation='sdpa')
    model.eval()

    samples = generate_squad_dataset(n_samples=args.n_samples, n_windows=5,
                                      vary_distance=True, seed=args.seed, split='validation')
    print(f"Loaded {len(samples)} samples (seed={args.seed}, k={args.k})", flush=True)

    results = eval_with_hooks(model, tokenizer, samples, args.k,
                               model.get_input_embeddings().weight.device, args.methods)

    print(f"\nRESULTS {model_name}:", flush=True)
    for m, acc in results.items():
        print(f"  {m}: {acc:.1%}", flush=True)

    save_path = f'logs/results/large_model_{model_name}_k{args.k}_n{args.n_samples}_s{args.seed}_{args.tag}.json'
    with open(save_path, 'w') as f:
        json.dump({'model': model_name, 'n_samples': args.n_samples, 'k': args.k,
                   'seed': args.seed, 'tag': args.tag, 'methods': args.methods,
                   'results': results}, f, indent=2)
    print(f"Saved to {save_path}", flush=True)


if __name__ == '__main__':
    main()
