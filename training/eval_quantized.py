"""Cross-model eval: multiplicative KV selection + adapted TurboQuant (K8V4 Lloyd-Max).
Per-head normalization + Lloyd-Max optimal codebook. No random rotation."""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from sentence_transformers import SentenceTransformer
from data.real_qa import generate_squad_dataset
import scipy.integrate as integrate


def solve_lm_normal(bits):
    """Solve Lloyd-Max optimal quantizer for N(0,1)."""
    n = 2 ** bits
    pdf = lambda x: (1.0 / math.sqrt(2 * math.pi)) * math.exp(-x * x / 2)
    c = [-3.5 + 7.0 * (i + 0.5) / n for i in range(n)]
    for _ in range(200):
        b = [(c[i] + c[i + 1]) / 2 for i in range(n - 1)]
        edges = [-10] + b + [10]
        nc = []
        for i in range(n):
            num, _ = integrate.quad(lambda x: x * pdf(x), edges[i], edges[i + 1])
            den, _ = integrate.quad(pdf, edges[i], edges[i + 1])
            nc.append(num / den if den > 1e-15 else c[i])
        if max(abs(nc[i] - c[i]) for i in range(n)) < 1e-10:
            break
        c = nc
    return torch.tensor(c, dtype=torch.float32)


def lm_compress(tensor, codebook):
    """Per-head Lloyd-Max quantization."""
    B, H, S, D = tensor.shape
    dt = tensor.dtype
    result = torch.zeros_like(tensor)
    cb = codebook.to(tensor.device)
    for h in range(H):
        head = tensor[0, h].float()
        mean = head.mean(0, keepdim=True)
        std = head.std(0, keepdim=True).clamp(min=1e-8)
        normed = (head - mean) / std
        indices = (normed.unsqueeze(-1) - cb).abs().argmin(dim=-1)
        recon = cb[indices]
        result[0, h] = (recon * std + mean).to(dt)
    return result


def eval_model(model_path, device, n_samples, seed, multi_gpu=False):
    """Run full eval: no-memory, multiplicative FP16, multiplicative K8V4, RAG k=1."""
    model_name = os.path.basename(model_path)
    print(f"\n{'='*60}\n{model_name}\n{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    if multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(model_path,
            quantization_config=bnb, device_map='auto',
            max_memory={0: '23GiB', 1: '23GiB'}, torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,
            quantization_config=bnb, device_map={'': device}, torch_dtype=torch.float16)
    model.eval()

    embed_device = model.get_input_embeddings().weight.device
    encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    samples = generate_squad_dataset(n_samples=n_samples, n_windows=5,
                                      vary_distance=True, seed=seed, split='validation')

    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    # Lloyd-Max codebooks
    cb8 = solve_lm_normal(8)
    cb4 = solve_lm_normal(4)
    print(f"Lloyd-Max codebooks ready (8-bit: {len(cb8)} levels, 4-bit: {len(cb4)} levels)", flush=True)

    correct_mult = 0
    correct_tq = 0
    correct_rag = 0

    for si, s in enumerate(samples):
        # === Build KV cache and attention scores ===
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []
        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=384,
                            truncation=True).to(embed_device)
            with torch.no_grad():
                out = model(input_ids=ids['input_ids'], use_cache=True, output_attentions=True)
            sl = ids['input_ids'].shape[1]
            imp = torch.zeros(sl, device=embed_device)
            for la in out.attentions:
                a = la[0].to(embed_device)
                if not a.isnan().any():
                    imp += a.sum(dim=(0, 1))
            imp[:4] = -1e9
            all_attn.append(imp)
            for li in range(nl):
                all_kv[li][0].append(out.past_key_values[li][0])
                all_kv[li][1].append(out.past_key_values[li][1])

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn)

        # Cross-window scoring
        q_ids = tokenizer(f'Question: {s.question}\nAnswer:', return_tensors='pt',
                          max_length=128, truncation=True).to(embed_device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)

        cross = torch.zeros(total, device=embed_device)
        for li in inject:
            layer_device = all_kv[li][0][0].device
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(
                    Q[:, hi * qpk:(hi + 1) * qpk, :].float(),
                    K[hi].float().T) / math.sqrt(hd)
                cross += sc.sum(dim=(0, 1)).to(embed_device)
        cross[:4] = -1e9

        mult = torch.clamp(heur, min=0) * torch.clamp(cross, min=0)
        mult[:4] = -1e9
        k = min(300, total)
        _, idx = mult.topk(k)
        idx = idx.sort().values

        # === Multiplicative FP16 ===
        cache_fp16 = DynamicCache()
        for li in range(nl):
            K_cat = torch.cat(all_kv[li][0], dim=2)
            V_cat = torch.cat(all_kv[li][1], dim=2)
            li_idx = idx.to(K_cat.device)
            cache_fp16.update(K_cat[:, :, li_idx, :], V_cat[:, :, li_idx, :], li)

        text_fp16 = _generate(model, tokenizer, cache_fp16, s.question, k, embed_device)
        if s.answer.lower() in text_fp16.lower():
            correct_mult += 1

        # === Multiplicative + K8V4 Lloyd-Max ===
        cache_tq = DynamicCache()
        for li in range(nl):
            K_cat = torch.cat(all_kv[li][0], dim=2)
            V_cat = torch.cat(all_kv[li][1], dim=2)
            li_idx = idx.to(K_cat.device)
            K_sel = K_cat[:, :, li_idx, :]
            V_sel = V_cat[:, :, li_idx, :]
            cache_tq.update(lm_compress(K_sel, cb8), lm_compress(V_sel, cb4), li)

        text_tq = _generate(model, tokenizer, cache_tq, s.question, k, embed_device)
        if s.answer.lower() in text_tq.lower():
            correct_tq += 1

        # === RAG k=1 ===
        past = s.windows[:-1]
        q_emb = encoder.encode(s.question, convert_to_numpy=True)
        p_embs = encoder.encode(past, convert_to_numpy=True)
        sims = (p_embs / (np.linalg.norm(p_embs, axis=1, keepdims=True) + 1e-8)) @ \
               (q_emb / (np.linalg.norm(q_emb) + 1e-8))
        rag_idx = np.argmax(sims)
        rag_text = f'{past[rag_idx]}\n\nBased on what you read, answer the following question.\nQuestion: {s.question}\nAnswer:'
        r_ids = tokenizer(rag_text, return_tensors='pt', max_length=768, truncation=True).to(embed_device)
        with torch.no_grad():
            r_out = model.generate(input_ids=r_ids['input_ids'], max_new_tokens=20,
                                    do_sample=False, pad_token_id=tokenizer.pad_token_id)
        r_text = tokenizer.decode(r_out[0][r_ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if s.answer.lower() in r_text.lower():
            correct_rag += 1

        if (si + 1) % 50 == 0:
            print(f'  {si+1}/{n_samples}: mult={correct_mult}/{si+1}={correct_mult/(si+1):.1%}  '
                  f'tq_k8v4={correct_tq}/{si+1}={correct_tq/(si+1):.1%}  '
                  f'rag={correct_rag}/{si+1}={correct_rag/(si+1):.1%}', flush=True)

    results = {
        'multiplicative_fp16': correct_mult / n_samples,
        'multiplicative_k8v4_lm': correct_tq / n_samples,
        'rag_k1': correct_rag / n_samples,
    }
    print(f'\nRESULT {model_name}: mult={results["multiplicative_fp16"]:.1%}  '
          f'tq_k8v4={results["multiplicative_k8v4_lm"]:.1%}  '
          f'rag={results["rag_k1"]:.1%}', flush=True)
    return model_name, results


def _generate(model, tokenizer, cache, question, k, device):
    """Generate answer with pre-filled KV cache."""
    query = f'Based on what you read earlier, answer the following question.\nQuestion: {question}\nAnswer:'
    fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
    input_ids = fq['input_ids']
    pos = torch.arange(k, k + input_ids.shape[1], device=device).unsqueeze(0)
    cur = input_ids
    cc = cache
    gen = []
    with torch.no_grad():
        for _ in range(20):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item())
            cur = nxt.to(device)
            pos = torch.tensor([[k + input_ids.shape[1] + len(gen) - 1]], device=device)
            if nxt[0, 0].item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--multi_gpu', action='store_true')
    args = parser.parse_args()

    model_name, results = eval_model(args.model_path, args.device, args.n_samples, args.seed, args.multi_gpu)

    save_path = f'./logs/turboquant_cross_{model_name}.json'
    save_data = {
        'model': model_name, 'n_samples': args.n_samples, 'seed': args.seed,
        'k': 300, 'method': 'multiplicative + adapted TurboQuant K8V4 Lloyd-Max',
        'results': results,
    }
    os.makedirs('logs', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f'Saved to {save_path}', flush=True)


if __name__ == '__main__':
    main()
