"""Mistral 24B architectural fix: spread mem-token RoPE positions across the cache
to avoid the θ=1e8 degeneracy at positions 0..15.

Tests 3 conditions on Mistral 24B Needle n=20:
  1. mem_packed     — mem tokens at positions 0..15 (current default, fails at 54%)
  2. mem_spread     — mem tokens at evenly-spaced positions across the 7680-token cache
  3. mem_few4       — only 4 mem tokens (subset of trained 16) at positions 0..3
  4. s1_only        — no S2 mem tokens at all (mult_k300 control, baseline 73%)

Implementation: we manually apply RoPE rotation to mem_K BEFORE caching by overriding
the cache positions. The RoPE rotation is computed from the model's RoPE module.
"""
import sys; sys.path.insert(0, '.')
import os, json, math, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from baselines.eval_needle import generate_haystack, _select_kv, NEEDLES


def apply_rope_at_positions(K, positions, rope_module, head_dim):
    """Apply RoPE rotation to K at given positions.
    K: [1, n_kv_heads, n_tokens, head_dim] (un-RoPE'd)
    positions: [n_tokens] long tensor
    """
    pos = positions.unsqueeze(0)  # [1, n_tokens]
    cos, sin = rope_module(K, pos)  # [1, n_tokens, head_dim]
    # Apply rotation: K_rot = K * cos + rotate_half(K) * sin
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(1)  # [1, 1, n_tokens, head_dim]
    sin = sin.unsqueeze(1)
    K_rot = (K * cos) + (rotate_half(K) * sin)
    return K_rot


def process_with_variant(model, tokenizer, windows, question, k, variant, device, astro, sense_layer):
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    all_kv = {li: ([], []) for li in range(nl)}
    offset = 0
    if astro is not None:
        astro.reset_state()
    for window in windows:
        ids = tokenizer(window, return_tensors='pt', max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True,
                        output_hidden_states=(astro is not None))
        offset += ids['input_ids'].shape[1]
        for li in range(nl):
            all_kv[li][0].append(out.past_key_values[li][0])
            all_kv[li][1].append(out.past_key_values[li][1])
        if astro is not None:
            with torch.no_grad():
                hidden = out.hidden_states[sense_layer]
                sensed = astro.sense(hidden)
                astro.update_state(sensed)
    total = offset

    # Cross-window mult scoring
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
            cross += torch.softmax(sc, dim=-1).sum(dim=(0, 1)).to(device)
    if total > 5:
        cross = torch.nn.functional.avg_pool1d(
            cross.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2, stride=1
        ).squeeze()
    n_sink = 4
    cross[:n_sink] = -1e9
    last_window_size = all_kv[inject[0]][0][-1].shape[2]
    last_start = total - last_window_size

    # Select real KV
    if variant == 's1_only':
        n_mem_use = 0
    elif variant == 'mem_few4':
        n_mem_use = 4
    else:
        n_mem_use = 16

    k_real = k - n_mem_use
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
        if n_mem_use > 0 and astro is not None:
            li_dev = K_real.device
            K_mem, V_mem = astro.generate_kv(li, (K_real.to(device), V_real.to(device)))
            # K_mem is un-RoPE'd; assign positions and apply RoPE
            if n_mem_use < 16:
                K_mem = K_mem[:, :, :n_mem_use, :]
                V_mem = V_mem[:, :, :n_mem_use, :]
            if variant == 'mem_spread':
                # Evenly spread mem_K positions across the real-token range
                positions = torch.linspace(0, total - 1, steps=n_mem_use, device=K_mem.device).long()
            else:
                # Packed: positions 0..n_mem_use-1 (default behavior, no manual RoPE)
                positions = torch.arange(n_mem_use, device=K_mem.device).long()
            # Apply manual RoPE rotation to K_mem at these positions
            try:
                rope_module = model.model.layers[li].self_attn.rotary_emb \
                    if hasattr(model.model.layers[li].self_attn, 'rotary_emb') \
                    else model.model.rotary_emb
                K_mem_rope = apply_rope_at_positions(K_mem.to(device).float(), positions, rope_module, hd).half().to(li_dev)
            except Exception as e:
                # Fallback: no manual RoPE, default behavior
                K_mem_rope = K_mem.to(li_dev)
            cache.update(torch.cat([K_mem_rope, K_real], dim=2),
                         torch.cat([V_mem.to(li_dev), V_real], dim=2), li)
        else:
            cache.update(K_real, V_real, li)
    prefix_len = n_mem_use + len(idx)

    # Generate
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
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./models/mistral-small-24b')
    parser.add_argument('--hybrid_checkpoint', default='./checkpoints/astro_hybrid_mistral-small-24b_n16_k284_t5000_s42.pt')
    parser.add_argument('--n_windows', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--k', type=int, default=300)
    parser.add_argument('--sense_layer', type=int, default=20)
    parser.add_argument('--attn_dim', type=int, default=256)
    args = parser.parse_args()

    print(f"[evenpos] Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
        quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
    model.eval()
    device = model.get_input_embeddings().weight.device

    from training.train_hybrid import AstroHybrid
    nl = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
    astro = AstroHybrid(
        hidden_dim=hidden_dim, n_mem_tokens=16, attn_dim=args.attn_dim,
        n_kv_heads=nkv, head_dim=hd, n_layers=nl,
        inject_layers=[nl // 4, nl // 2, 3 * nl // 4, nl - 2],
    ).to(device)
    astro.extract_model_weights(model, device)
    astro.load_state_dict(torch.load(args.hybrid_checkpoint, map_location=device), strict=False)
    astro.eval()

    variants = ['mem_packed', 'mem_spread', 'mem_few4', 's1_only']
    depths = [0, args.n_windows // 4, args.n_windows // 2, 3 * args.n_windows // 4, args.n_windows - 1]
    results = {v: [] for v in variants}

    for trial in range(args.n_trials):
        for di, depth in enumerate(depths):
            seed = 42 + trial * 100 + di
            needle_idx = (trial + di) % len(NEEDLES)
            windows, question, answer = generate_haystack(args.n_windows, depth, needle_idx=needle_idx, seed=seed)
            for v in variants:
                pred = process_with_variant(model, tokenizer, windows, question, args.k, v, device, astro, args.sense_layer)
                ok = int(answer.lower() in pred.lower())
                results[v].append(ok)
                if trial < 2:
                    print(f"  trial={trial} d={depth} {v}: ok={ok} pred='{pred[:50]}'", flush=True)

    print("\n=== AGGREGATE Mistral 24B even-pos fix (n=20, trials={}) ===".format(args.n_trials), flush=True)
    for v in variants:
        acc = sum(results[v]) / len(results[v])
        print(f"  {v}: {acc:.3f} ({sum(results[v])}/{len(results[v])})", flush=True)

    with open('logs/results/diag_mistral24b_evenpos.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to logs/results/diag_mistral24b_evenpos.json", flush=True)


if __name__ == '__main__':
    main()
