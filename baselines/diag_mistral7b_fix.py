"""Mistral 7B specific S1 fixes — diagnose which architectural adaptation helps.

Hypothesis: Mistral 7B's multiplicative S1 collapses because its long-range
cross-window attention is noisy (trained at 32k max-position, much less than
Qwen 7B's 131k).

Test 5 S1 variants on Mistral 7B Needle n=20:
  1. mult_default        (current: softmax + kernel=5 + sense layer nl/2)
  2. mult_widekern        (kernel=11 smoothing)
  3. mult_earlyonly       (use first 2 inject layers only)
  4. mult_lateonly        (use last 2 inject layers only)
  5. mult_nosmooth        (no avg-pool)
  6. mult_eager           (attn_implementation=eager)
"""
import sys; sys.path.insert(0, '.')
import os, json, math, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from baselines.eval_needle import generate_haystack, _select_kv, NEEDLES


def score_and_predict(model, tokenizer, windows, question, answer, k, device, variant='default'):
    """Process windows and select tokens using a specific S1 variant, then predict."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv

    if variant == 'earlyonly':
        inject = [nl // 8, nl // 4]
    elif variant == 'lateonly':
        inject = [3 * nl // 4, nl - 2]
    else:
        inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    all_kv = {li: ([], []) for li in range(nl)}
    offset = 0
    for window in windows:
        ids = tokenizer(window, return_tensors='pt', max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True)
        offset += ids['input_ids'].shape[1]
        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])
    total = offset

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

    # Smoothing variants
    if variant == 'widekern' and total > 11:
        cross = torch.nn.functional.avg_pool1d(
            cross.unsqueeze(0).unsqueeze(0), kernel_size=11, padding=5, stride=1
        ).squeeze()
    elif variant == 'nosmooth':
        pass
    else:
        if total > 5:
            cross = torch.nn.functional.avg_pool1d(
                cross.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2, stride=1
            ).squeeze()

    n_sink = 4
    cross[:n_sink] = -1e9
    last_window_size = all_kv[inject[0]][0][-1].shape[2]
    last_start = total - last_window_size

    k_sel = min(k, total)
    n_recent = min(int(k_sel * 0.2), last_window_size)
    recent_idx = torch.arange(last_start, last_start + n_recent, device=device)
    scores = cross.clone()
    scores[last_start:total] = -1e9
    n_select = max(k_sel - n_sink - n_recent, 0)
    n_avail = (scores > -1e8).sum().item()
    n_select = min(n_select, n_avail, scores.shape[0])
    _, top = scores.topk(n_select) if n_select > 0 else (None, torch.empty(0, dtype=torch.long, device=device))
    sink_idx = torch.arange(n_sink, device=device)
    idx = torch.cat([sink_idx, top, recent_idx]).unique().sort().values[:k_sel]

    cache = DynamicCache()
    for li in range(nl):
        K, V = _select_kv(all_kv, li, idx)
        cache.update(K, V, li)
    prefix_len = len(idx)

    query = f'Based on what you read, answer the question.\nQuestion: {question}\nAnswer:'
    fq = tokenizer(query, return_tensors='pt', max_length=256, truncation=True).to(device)
    pos = torch.arange(prefix_len, prefix_len + fq['input_ids'].shape[1], device=device).unsqueeze(0)
    cur = fq['input_ids']; cc = cache; gen = []
    with torch.no_grad():
        for _ in range(30):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item()); cur = nxt
            pos = torch.tensor([[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]], device=device)
            if nxt[0, 0].item() == tokenizer.eos_token_id:
                break
    pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return int(answer.lower() in pred.lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./models/mistral-7b-v0.3')
    parser.add_argument('--n_windows', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--k', type=int, default=300)
    args = parser.parse_args()

    print(f"[diag-mistral7b] Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
        quantization_config=bnb, device_map={'': 'cuda:0'}, torch_dtype=torch.float16)
    model.eval()
    device = model.get_input_embeddings().weight.device

    variants = ['default', 'widekern', 'earlyonly', 'lateonly', 'nosmooth']
    depths = [0, args.n_windows // 4, args.n_windows // 2, 3 * args.n_windows // 4, args.n_windows - 1]

    results = {v: [] for v in variants}
    for trial in range(args.n_trials):
        for di, depth in enumerate(depths):
            seed = 42 + trial * 100 + di
            needle_idx = (trial + di) % len(NEEDLES)
            windows, question, answer = generate_haystack(args.n_windows, depth,
                                                         needle_idx=needle_idx, seed=seed)
            for v in variants:
                ok = score_and_predict(model, tokenizer, windows, question, answer,
                                       args.k, device, variant=v)
                results[v].append(ok)
                if trial < 2:
                    print(f"  trial={trial} d={depth} {v}: ok={ok}", flush=True)

    print("\n=== AGGREGATE (Mistral 7B n=20) ===", flush=True)
    for v in variants:
        acc = sum(results[v]) / len(results[v])
        print(f"  {v}: {acc:.3f} ({sum(results[v])}/{len(results[v])})", flush=True)

    with open('logs/results/diag_mistral7b_fix.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to logs/results/diag_mistral7b_fix.json", flush=True)


if __name__ == '__main__':
    main()
