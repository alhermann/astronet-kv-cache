"""Definitive cross-model evaluation: multiplicative KV selection vs RAG vs baselines.
Zero-shot, zero parameters. Evaluates on SQuAD validation set."""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from sentence_transformers import SentenceTransformer
from data.real_qa import generate_squad_dataset


def multiplicative_eval(model, tokenizer, samples, k_total=300, n_sink=4):
    """Multiplicative cross-window KV selection (Ca2+ × stimulus modulation)."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    device = model.get_input_embeddings().weight.device
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    correct = 0
    for si, s in enumerate(samples):
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []

        # Process fact windows
        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=384, truncation=True).to(device)
            with torch.no_grad():
                out = model(input_ids=ids['input_ids'], use_cache=True, output_attentions=True)
            sl = ids['input_ids'].shape[1]
            imp = torch.zeros(sl, device=device)
            for la in out.attentions:
                a = la[0]
                if not a.isnan().any():
                    imp += a.sum(dim=(0, 1))
            imp[:n_sink] = -1e9
            all_attn.append(imp)
            for li in range(nl):
                all_kv[li][0].append(out.past_key_values[li][0])
                all_kv[li][1].append(out.past_key_values[li][1])

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn)

        # Cross-window scoring from question
        q_ids = tokenizer(f'Question: {s.question}\nAnswer:', return_tensors='pt',
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
                sc = torch.matmul(
                    Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                    K[hi].float().T) / math.sqrt(hd)
                cross += sc.sum(dim=(0, 1)).to(device)
        cross[:n_sink] = -1e9

        # Multiplicative combination
        mult = torch.clamp(heur, min=0) * torch.clamp(cross, min=0)
        mult[:n_sink] = -1e9
        k = min(k_total, total)
        _, idx = mult.topk(k)
        idx = idx.sort().values

        # Build cache and generate
        cache = DynamicCache()
        for li in range(nl):
            K = torch.cat(all_kv[li][0], dim=2)
            V = torch.cat(all_kv[li][1], dim=2)
            li_idx = idx.to(K.device)
            cache.update(K[:, :, li_idx, :], V[:, :, li_idx, :], li)

        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        gen_device = next(model.parameters()).device
        text = _generate_with_cache(model, tokenizer, fq['input_ids'], cache, k, gen_device)
        if s.answer.lower() in text.lower():
            correct += 1

        if (si + 1) % 50 == 0:
            print(f'  mult {si+1}/{len(samples)}: {correct}/{si+1} = {correct/(si+1):.1%}', flush=True)

    return correct / len(samples)


def heuristic_eval(model, tokenizer, samples, k_total=300, n_sink=4):
    """Heuristic-only (cumulative attention) KV selection."""
    nl = model.config.num_hidden_layers
    device = model.get_input_embeddings().weight.device
    correct = 0

    for si, s in enumerate(samples):
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []

        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=384, truncation=True).to(device)
            with torch.no_grad():
                out = model(input_ids=ids['input_ids'], use_cache=True, output_attentions=True)
            sl = ids['input_ids'].shape[1]
            imp = torch.zeros(sl, device=device)
            for la in out.attentions:
                a = la[0]
                if not a.isnan().any():
                    imp += a.sum(dim=(0, 1))
            imp[:n_sink] = -1e9
            all_attn.append(imp)
            for li in range(nl):
                all_kv[li][0].append(out.past_key_values[li][0])
                all_kv[li][1].append(out.past_key_values[li][1])

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn)
        k = min(k_total, total)
        _, idx = heur.topk(k)
        idx = idx.sort().values

        cache = DynamicCache()
        for li in range(nl):
            K = torch.cat(all_kv[li][0], dim=2)
            V = torch.cat(all_kv[li][1], dim=2)
            li_idx = idx.to(K.device)
            cache.update(K[:, :, li_idx, :], V[:, :, li_idx, :], li)

        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        gen_device = next(model.parameters()).device
        text = _generate_with_cache(model, tokenizer, fq['input_ids'], cache, k, gen_device)
        if s.answer.lower() in text.lower():
            correct += 1

        if (si + 1) % 50 == 0:
            print(f'  heur {si+1}/{len(samples)}: {correct}/{si+1} = {correct/(si+1):.1%}', flush=True)

    return correct / len(samples)


def rag_eval(model, tokenizer, samples, encoder, rag_k=1):
    """RAG baseline: retrieve top-k windows via sentence embedding similarity."""
    device = model.get_input_embeddings().weight.device
    correct = 0

    for si, s in enumerate(samples):
        past = s.windows[:-1]
        q_emb = encoder.encode(s.question, convert_to_numpy=True)
        p_embs = encoder.encode(past, convert_to_numpy=True)
        sims = (p_embs / (np.linalg.norm(p_embs, axis=1, keepdims=True) + 1e-8)) @ \
               (q_emb / (np.linalg.norm(q_emb) + 1e-8))

        top_indices = np.argsort(sims)[-rag_k:][::-1]
        rag_context = "\n\n".join([past[i] for i in sorted(top_indices)])
        rag_text = f'{rag_context}\n\nBased on what you read, answer the following question.\nQuestion: {s.question}\nAnswer:'
        r_ids = tokenizer(rag_text, return_tensors='pt', max_length=768, truncation=True).to(device)
        with torch.no_grad():
            r_out = model.generate(input_ids=r_ids['input_ids'], max_new_tokens=20,
                                    do_sample=False, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(r_out[0][r_ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if s.answer.lower() in text.lower():
            correct += 1

        if (si + 1) % 50 == 0:
            print(f'  rag_k{rag_k} {si+1}/{len(samples)}: {correct}/{si+1} = {correct/(si+1):.1%}', flush=True)

    return correct / len(samples)


def no_memory_eval(model, tokenizer, samples):
    """No memory baseline — question only, no context."""
    device = model.get_input_embeddings().weight.device
    correct = 0
    for si, s in enumerate(samples):
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        q_ids = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(input_ids=q_ids['input_ids'], max_new_tokens=20,
                                  do_sample=False, pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][q_ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if s.answer.lower() in text.lower():
            correct += 1
    return correct / len(samples)


def full_kv_eval(model, tokenizer, samples):
    """Full KV cache — all windows concatenated (upper bound)."""
    nl = model.config.num_hidden_layers
    device = model.get_input_embeddings().weight.device
    correct = 0

    for si, s in enumerate(samples):
        all_text = "\n\n".join(s.windows[:-1])
        ids = tokenizer(all_text, return_tensors='pt', max_length=2048, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True)
        seq_len = ids['input_ids'].shape[1]
        cache = DynamicCache()
        for li in range(nl):
            cache.update(out.past_key_values[li][0], out.past_key_values[li][1], li)

        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        text = _generate_with_cache(model, tokenizer, fq['input_ids'], cache, seq_len, device)
        if s.answer.lower() in text.lower():
            correct += 1

        if (si + 1) % 50 == 0:
            print(f'  full_kv {si+1}/{len(samples)}: {correct}/{si+1} = {correct/(si+1):.1%}', flush=True)

    return correct / len(samples)


def _generate_with_cache(model, tokenizer, input_ids, cache, prefix_len, device, max_tokens=20):
    """Generate tokens with a pre-filled KV cache."""
    # For multi-GPU, input_ids go to the embedding device
    embed_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(embed_device)
    pos = torch.arange(prefix_len, prefix_len + input_ids.shape[1], device=embed_device).unsqueeze(0)
    cur = input_ids
    cc = cache
    gen = []
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=300, help='KV budget for multiplicative/heuristic')
    parser.add_argument('--rag_k', type=int, default=1, help='Number of retrieved windows for RAG')
    parser.add_argument('--skip_heuristic', action='store_true')
    parser.add_argument('--skip_full_kv', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true', help='Use both GPUs for large models')
    parser.add_argument('--save_path', default=None)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path)
    if args.save_path is None:
        args.save_path = f'./logs/definitive_{model_name}_k{args.k}.json'

    print(f"Loading: {args.model_path}", flush=True)
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

    samples = generate_squad_dataset(n_samples=args.n_samples, n_windows=5,
                                      vary_distance=True, seed=args.seed, split='validation')
    print(f"Loaded {len(samples)} samples, k={args.k}", flush=True)

    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    results = {}

    # No memory
    print(f"\n{'='*50}\nNo Memory\n{'='*50}", flush=True)
    t0 = time.time()
    nm = no_memory_eval(model, tokenizer, samples)
    print(f"  => {nm:.1%} ({time.time()-t0:.0f}s)", flush=True)
    results['no_memory'] = nm

    # Heuristic
    if not args.skip_heuristic:
        print(f"\n{'='*50}\nHeuristic k={args.k}\n{'='*50}", flush=True)
        t0 = time.time()
        h = heuristic_eval(model, tokenizer, samples, k_total=args.k)
        print(f"  => {h:.1%} ({time.time()-t0:.0f}s)", flush=True)
        results['heuristic'] = h

    # Multiplicative
    print(f"\n{'='*50}\nMultiplicative k={args.k}\n{'='*50}", flush=True)
    t0 = time.time()
    m = multiplicative_eval(model, tokenizer, samples, k_total=args.k)
    print(f"  => {m:.1%} ({time.time()-t0:.0f}s)", flush=True)
    results['multiplicative'] = m

    # RAG
    print(f"\n{'='*50}\nRAG k={args.rag_k}\n{'='*50}", flush=True)
    t0 = time.time()
    r = rag_eval(model, tokenizer, samples, encoder, rag_k=args.rag_k)
    print(f"  => {r:.1%} ({time.time()-t0:.0f}s)", flush=True)
    results[f'rag_k{args.rag_k}'] = r

    # Full KV cache
    if not args.skip_full_kv:
        print(f"\n{'='*50}\nFull KV Cache\n{'='*50}", flush=True)
        t0 = time.time()
        fkv = full_kv_eval(model, tokenizer, samples)
        print(f"  => {fkv:.1%} ({time.time()-t0:.0f}s)", flush=True)
        results['full_kv_cache'] = fkv

    save_data = {
        'model': model_name, 'n_samples': len(samples), 'seed': args.seed,
        'k': args.k, 'rag_k': args.rag_k, 'results': results
    }
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {args.save_path}", flush=True)
    print(f"\n{'='*50}\nSUMMARY: {model_name}\n{'='*50}", flush=True)
    for method, acc in results.items():
        print(f"  {method:20s}: {acc:.1%}", flush=True)


if __name__ == '__main__':
    main()
