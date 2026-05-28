"""Needle-in-a-Haystack evaluation.

Hide a fact at varying depths in distractor text, then ask about it.
Tests whether KV selection methods can find the needle across different
positions in long context. Uses our multi-window pipeline.
"""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse, random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache


DISTRACTOR_TOPICS = [
    "The history of classical music spans several centuries and includes many notable composers.",
    "Marine biology studies the diverse ecosystems found in oceans and other saltwater environments.",
    "Modern architecture emphasizes clean lines, open spaces, and the integration of natural light.",
    "The development of renewable energy sources has accelerated significantly in recent decades.",
    "Advances in materials science have led to the creation of stronger and lighter composites.",
    "The study of linguistics reveals the complex structures underlying human language.",
    "Agricultural innovations have dramatically increased crop yields around the world.",
    "The field of astronomy continues to expand our understanding of the universe.",
    "Transportation systems have evolved from horse-drawn carriages to high-speed rail.",
    "The printing press revolutionized the dissemination of knowledge across societies.",
]

NEEDLES = [
    ("The secret code for the vault is ALPHA-7829.", "What is the secret code for the vault?", "ALPHA-7829"),
    ("The meeting is scheduled for March 15th at 3pm in Room 204.", "When and where is the meeting scheduled?", "March 15th at 3pm in Room 204"),
    ("Professor Henderson discovered the high-temperature superconductor in 1987.", "Who discovered the high-temperature superconductor?", "Professor Henderson"),
    ("The population of the island nation is exactly 4,231,567 people.", "What is the population of the island nation?", "4,231,567"),
    ("The antidote requires exactly three drops of the blue serum.", "How many drops of the blue serum does the antidote require?", "three"),
]


def generate_haystack(n_windows, needle_position, needle_idx=0, seed=42):
    """Generate a haystack with a needle at a specific position.

    Args:
        n_windows: total number of windows
        needle_position: which window (0-indexed) contains the needle
        needle_idx: which needle to use
        seed: random seed
    Returns:
        windows: list of text windows
        question: the question about the needle
        answer: the expected answer
    """
    rng = random.Random(seed)
    needle_text, question, answer = NEEDLES[needle_idx % len(NEEDLES)]

    windows = []
    for i in range(n_windows):
        if i == needle_position:
            # Needle window: embed the fact among some distractor text
            prefix = rng.choice(DISTRACTOR_TOPICS)
            suffix = rng.choice(DISTRACTOR_TOPICS)
            windows.append(f"{prefix} {needle_text} {suffix}")
        else:
            # Pure distractor: 2-3 random topic sentences
            n_sents = rng.randint(2, 3)
            sents = [rng.choice(DISTRACTOR_TOPICS) for _ in range(n_sents)]
            windows.append(" ".join(sents))

    return windows, question, answer


def _select_kv(all_kv, li, idx):
    """Select KV pairs by index, handling multi-GPU device placement."""
    K = torch.cat(all_kv[li][0], dim=2)
    V = torch.cat(all_kv[li][1], dim=2)
    li_idx = idx.to(K.device)
    return K[:, :, li_idx, :], V[:, :, li_idx, :]


def process_and_select(model, tokenizer, windows, question, k, method, device, astro=None, sense_layer=14):
    """Process windows and select tokens — same as in eval_all_baselines."""
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    inject = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    all_kv = {li: ([], []) for li in range(nl)}
    all_attn = []
    offset = 0

    # Reset AstroNet if using hybrid
    if astro is not None:
        astro.reset_state()

    for window in windows:
        ids = tokenizer(window, return_tensors='pt', max_length=384, truncation=True).to(device)
        with torch.no_grad():
            out = model(input_ids=ids['input_ids'], use_cache=True, output_attentions=True,
                        output_hidden_states=(astro is not None))
        sl = ids['input_ids'].shape[1]
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

        # AstroNet sensing
        if astro is not None:
            with torch.no_grad():
                hidden = out.hidden_states[sense_layer]
                sensed = astro.sense(hidden)
                astro.update_state(sensed)

    total = offset
    heur = torch.cat(all_attn)

    # Cross-window scoring with patched S1 (softmax + multi-layer + last-window mask)
    if method in ('multiplicative', 'hybrid', 'snapkv'):
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
        if total > 5:
            cross = torch.nn.functional.avg_pool1d(
                cross.unsqueeze(0).unsqueeze(0), kernel_size=5, padding=2, stride=1
            ).squeeze()
        n_sink = 4
        cross[:n_sink] = -1e9
        last_window_size = all_kv[inject[0]][0][-1].shape[2]
        last_start = total - last_window_size

    if method == 'hybrid' and astro is not None:
        n_mem = astro.n_mem_tokens
        k_real = min(k - n_mem, total)
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

        # Build hybrid cache: [AstroNet KV | real selected KV]
        cache = DynamicCache()
        for li in range(nl):
            K_real, V_real = _select_kv(all_kv, li, idx)
            li_dev = K_real.device
            K_mem, V_mem = astro.generate_kv(li, (K_real.to(device), V_real.to(device)))
            cache.update(torch.cat([K_mem.to(li_dev), K_real], dim=2),
                         torch.cat([V_mem.to(li_dev), V_real], dim=2), li)
        prefix_len = n_mem + k_real
    elif method == 'multiplicative':
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
    elif method == 'h2o':
        k_sel = min(k, total)
        scores = heur.clone(); scores[:4] = -1e9
        _, idx = scores.topk(k_sel)
        idx = idx.sort().values
        cache = DynamicCache()
        for li in range(nl):
            K, V = _select_kv(all_kv, li, idx)
            cache.update(K, V, li)
        prefix_len = k_sel
    elif method == 'streaming_llm':
        k_sel = min(k, total)
        sink = torch.arange(4, device=device)
        recent = torch.arange(total - (k_sel - 4), total, device=device)
        idx = torch.cat([sink, recent])
        cache = DynamicCache()
        for li in range(nl):
            K, V = _select_kv(all_kv, li, idx)
            cache.update(K, V, li)
        prefix_len = len(idx)
    elif method == 'snapkv':
        # Faithful SnapKV (Li et al. 2024): same scoring as multiplicative
        # (post-softmax cross-Q*K + avg-pool smoothing), but selection scope
        # is the last *observation* window only (the question). In our multi-window
        # setup this matches `multiplicative`: same scoring; recent-window kept.
        # We reuse the `cross` already computed and apply the standard top-k.
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
    else:
        idx = torch.arange(total, device=device)
        cache = DynamicCache()
        for li in range(nl):
            K, V = _select_kv(all_kv, li, idx)
            cache.update(K, V, li)
        prefix_len = total

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
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--k', type=int, default=300)
    parser.add_argument('--n_windows_list', nargs='+', type=int, default=[5, 10, 20, 40])
    parser.add_argument('--n_trials', type=int, default=20, help='Trials per depth per n_windows')
    parser.add_argument('--methods', nargs='+', default=['streaming_llm', 'h2o', 'multiplicative'])
    parser.add_argument('--hybrid_checkpoint', default=None, help='AstroNet checkpoint for hybrid method')
    parser.add_argument('--sense_layer', type=int, default=None)
    parser.add_argument('--n_mem', type=int, default=16)
    parser.add_argument('--attn_dim', type=int, default=256)
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--seed_offset', type=int, default=0,
                        help='Offset added to per-trial seed; 0 reproduces original results')
    parser.add_argument('--save_suffix', default='',
                        help='Suffix appended to output filename (e.g. _seed1) to avoid overwriting')
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

    # Load AstroNet if hybrid requested
    astro = None
    sense_layer = args.sense_layer or model.config.num_hidden_layers // 2
    if args.hybrid_checkpoint and 'hybrid' in args.methods:
        from training.train_hybrid import AstroHybrid
        nl = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        nkv = model.config.num_key_value_heads
        hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
        astro = AstroHybrid(
            hidden_dim=hidden_dim, n_mem_tokens=args.n_mem, attn_dim=args.attn_dim,
            n_kv_heads=nkv, head_dim=hd, n_layers=nl,
            inject_layers=[nl//4, nl//2, 3*nl//4, nl-2],
        ).to(device)
        astro.extract_model_weights(model, device)
        astro.load_state_dict(torch.load(args.hybrid_checkpoint, map_location=device), strict=False)
        astro.eval()
        print(f"Loaded AstroNet: {astro.parameter_count():,} params", flush=True)

    all_results = {}

    for n_win in args.n_windows_list:
        print(f"\n{'='*50}\nn_windows={n_win}\n{'='*50}", flush=True)

        # Test needle at different depths: start, 25%, 50%, 75%, end
        depths = [0, n_win // 4, n_win // 2, 3 * n_win // 4, n_win - 1]
        depth_labels = ['start', '25%', '50%', '75%', 'end']

        win_results = {}
        for method in args.methods:
            depth_acc = {}
            for di, (depth, label) in enumerate(zip(depths, depth_labels)):
                correct = 0
                for trial in range(args.n_trials):
                    needle_idx = (trial + args.seed_offset) % len(NEEDLES)
                    windows, question, answer = generate_haystack(
                        n_win, depth, needle_idx,
                        seed=42 + trial * 100 + di + args.seed_offset)
                    pred = process_and_select(model, tokenizer, windows, question,
                                              args.k, method, device, astro=astro,
                                              sense_layer=sense_layer)
                    if answer.lower() in pred.lower():
                        correct += 1
                acc = correct / args.n_trials
                depth_acc[label] = acc
            win_results[method] = depth_acc
            avg = np.mean(list(depth_acc.values()))
            print(f"  {method}: {depth_acc} avg={avg:.1%}", flush=True)

        all_results[f'n{n_win}'] = win_results

    save_path = f'logs/results/needle_{model_name}_k{args.k}{args.save_suffix}.json'
    with open(save_path, 'w') as f:
        json.dump({'model': model_name, 'k': args.k,
                   'seed_offset': args.seed_offset,
                   'results': all_results}, f, indent=2)
    print(f"\nSaved to {save_path}", flush=True)


if __name__ == '__main__':
    main()
