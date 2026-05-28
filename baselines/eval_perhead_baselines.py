"""Per-head H2O and SnapKV baselines — faithful to original papers.

Each attention head independently scores and selects tokens, then we take
the UNION across heads (capped at k). This is more generous than global
selection, making these stronger baselines.

Compares: global H2O/SnapKV (our impl) vs per-head H2O/SnapKV (faithful).
"""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from data.real_qa import generate_squad_dataset


def process_windows(model, tokenizer, windows, device):
    """Process windows, collecting per-head attention scores and KV cache."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads

    all_kv = {li: ([], []) for li in range(nl)}
    # Per-head attention: list of [n_heads, seq_len] tensors per window
    all_attn_perhead = []
    window_boundaries = []
    offset = 0

    for window in windows:
        ids = tokenizer(window, return_tensors='pt', max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True, output_attentions=True)
        sl = ids['input_ids'].shape[1]
        window_boundaries.append((offset, offset + sl))
        offset += sl

        # Per-head attention scores: sum over query positions, keep head dimension
        # out.attentions[layer] shape: [batch, n_heads, q_len, kv_len]
        head_imp = torch.zeros(nq, sl, device=device)
        for la in out.attentions:
            a = la[0]  # [n_heads, q_len, kv_len]
            if not a.isnan().any():
                head_imp += a.sum(dim=1)  # sum over query positions → [n_heads, kv_len]
        all_attn_perhead.append(head_imp)

        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])

    return all_kv, all_attn_perhead, window_boundaries, offset


def select_h2o_perhead(all_attn_perhead, k, window_boundaries, n_sink=4, recent_ratio=0.3):
    """Per-head H2O: each head selects its own heavy hitters, then take union."""
    per_head = torch.cat(all_attn_perhead, dim=1)  # [n_heads, total]
    n_heads, total = per_head.shape
    if total <= k:
        return torch.arange(total, device=per_head.device)

    n_recent = int(k * recent_ratio)
    per_head_k = max((k - n_sink - n_recent) // n_heads, 1)

    last_start, last_end = window_boundaries[-1]
    recent_idx = set(range(last_start, min(last_end, last_start + n_recent)))
    sink_idx = set(range(n_sink))

    # Per-head heavy hitter selection
    all_selected = set()
    all_selected.update(sink_idx)
    all_selected.update(recent_idx)

    for h in range(n_heads):
        scores = per_head[h].clone()
        scores[:n_sink] = -1e9
        scores[last_start:last_end] = -1e9
        _, top = scores.topk(min(per_head_k, (scores > -1e8).sum().item()))
        all_selected.update(top.tolist())

    idx = sorted(list(all_selected))[:k]  # cap at k
    return torch.tensor(idx, device=per_head.device)


def select_snapkv_perhead(model, all_kv, all_attn_perhead, question, tokenizer,
                           k, device, window_boundaries, n_sink=4, recent_ratio=0.2):
    """Per-head SnapKV: each head uses its own Q@K scores, then take union."""
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

    # Per-head cross-window scores
    per_head_cross = torch.zeros(nq, total, device=device)

    for li in inject:
        layer_device = all_kv[li][0][0].device
        Q = model.model.layers[li].self_attn.q_proj(
            q_out.hidden_states[li][0].float().to(layer_device)).half().view(-1, nq, hd)
        K = torch.cat(all_kv[li][0], dim=2)[0]  # [n_kv, total, hd]

        for hi in range(nkv):
            # Each Q head in this KV group
            for qi in range(qpk):
                head_idx = hi * qpk + qi
                sc = torch.matmul(Q[:, head_idx, :].float(), K[hi].float().T) / math.sqrt(hd)
                # Post-softmax (faithful to SnapKV)
                attn_w = torch.softmax(sc, dim=-1)
                per_head_cross[head_idx] += attn_w.sum(dim=0).to(device)

    # Per-head avg-pool smoothing
    if total > 5:
        per_head_cross = F.avg_pool1d(
            per_head_cross.unsqueeze(0), kernel_size=5, padding=2, stride=1
        ).squeeze(0)

    n_recent = int(k * recent_ratio)
    per_head_k = max((k - n_sink - n_recent) // nq, 1)

    last_start, last_end = window_boundaries[-1]
    recent_idx = set(range(last_start, min(last_end, last_start + n_recent)))
    sink_idx = set(range(n_sink))

    all_selected = set()
    all_selected.update(sink_idx)
    all_selected.update(recent_idx)

    for h in range(nq):
        scores = per_head_cross[h].clone()
        scores[:n_sink] = -1e9
        scores[last_start:last_end] = -1e9
        n_avail = (scores > -1e8).sum().item()
        _, top = scores.topk(min(per_head_k, n_avail))
        all_selected.update(top.tolist())

    idx = sorted(list(all_selected))[:k]
    return torch.tensor(idx, device=device)


def generate_with_cache(model, tokenizer, all_kv, idx, k, question, nl, device):
    """Build cache and generate."""
    cache = DynamicCache()
    for li in range(nl):
        K = torch.cat(all_kv[li][0], dim=2)[:, :, idx, :]
        V = torch.cat(all_kv[li][1], dim=2)[:, :, idx, :]
        cache.update(K, V, li)

    query = f'Based on what you read earlier, answer the following question.\nQuestion: {question}\nAnswer:'
    fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
    pos = torch.arange(k, k + fq['input_ids'].shape[1], device=device).unsqueeze(0)
    cur = fq['input_ids']; cc = cache; gen = []
    with torch.no_grad():
        for _ in range(20):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item()); cur = nxt
            pos = torch.tensor([[k + fq['input_ids'].shape[1] + len(gen) - 1]], device=device)
            if nxt[0, 0].item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=300)
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path)
    print(f"Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
        quantization_config=bnb, device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = model.get_input_embeddings().weight.device
    nl = model.config.num_hidden_layers

    samples = generate_squad_dataset(n_samples=args.n_samples, n_windows=5,
                                      vary_distance=True, seed=args.seed, split='validation')
    print(f"Loaded {len(samples)} samples", flush=True)

    correct = {'h2o_global': 0, 'h2o_perhead': 0, 'snapkv_global': 0, 'snapkv_perhead': 0}

    for si, s in enumerate(samples):
        fact_windows = s.windows[:-1]
        all_kv, all_attn_perhead, boundaries, total = process_windows(
            model, tokenizer, fact_windows, device)

        # Global attention (sum across heads)
        all_attn_global = [a.sum(dim=0) for a in all_attn_perhead]

        # H2O global
        from baselines.eval_all_baselines import select_h2o
        idx_h2o_g = select_h2o(all_attn_global, args.k, boundaries)
        text = generate_with_cache(model, tokenizer, all_kv, idx_h2o_g, len(idx_h2o_g),
                                    s.question, nl, device)
        if s.answer.lower() in text.lower():
            correct['h2o_global'] += 1

        # H2O per-head
        idx_h2o_ph = select_h2o_perhead(all_attn_perhead, args.k, boundaries)
        text = generate_with_cache(model, tokenizer, all_kv, idx_h2o_ph, len(idx_h2o_ph),
                                    s.question, nl, device)
        if s.answer.lower() in text.lower():
            correct['h2o_perhead'] += 1

        # SnapKV global
        from baselines.eval_all_baselines import select_snapkv
        idx_snap_g = select_snapkv(model, all_kv, s.question, tokenizer, args.k, device, boundaries)
        text = generate_with_cache(model, tokenizer, all_kv, idx_snap_g, len(idx_snap_g),
                                    s.question, nl, device)
        if s.answer.lower() in text.lower():
            correct['snapkv_global'] += 1

        # SnapKV per-head
        idx_snap_ph = select_snapkv_perhead(model, all_kv, all_attn_perhead, s.question,
                                             tokenizer, args.k, device, boundaries)
        text = generate_with_cache(model, tokenizer, all_kv, idx_snap_ph, len(idx_snap_ph),
                                    s.question, nl, device)
        if s.answer.lower() in text.lower():
            correct['snapkv_perhead'] += 1

        if (si + 1) % 50 == 0:
            n = si + 1
            parts = [f'{m}={correct[m]}/{n}' for m in correct]
            print(f"  {n}/{len(samples)}: {', '.join(parts)}", flush=True)

    n = len(samples)
    results = {m: correct[m] / n for m in correct}
    print(f"\nRESULTS ({model_name}, k={args.k}):", flush=True)
    for m, acc in results.items():
        print(f"  {m:20s}: {acc:.1%}", flush=True)

    save_path = f'logs/results/perhead_baselines_{model_name}.json'
    with open(save_path, 'w') as f:
        json.dump({'model': model_name, 'k': args.k, 'n_samples': n, 'results': results}, f, indent=2)
    print(f"Saved to {save_path}", flush=True)


if __name__ == '__main__':
    main()
