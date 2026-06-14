"""Multi-Query Needle-in-a-Haystack — compress-once protocol.

RULER-style multi-key/multi-query NIAH. Long ctx contains K distinct needles
at random depths; the same compressed cache must answer K questions, one per
needle. SnapKV's query-agnostic obs-window scoring has no way to predict
which questions will come, so its single top-k must somehow cover all K
needles. AstroGate's integrated context state, in principle, can be biased
toward "things that look like inserted facts" rather than "what's at the
ctx tail".

Methods scored:
  full           — no compression (ceiling)
  snapkv_oracle  — query-aware ref (cheats: per-question recompression)
  snapkv         — query-agnostic (the actual baseline we want to beat)
  astrogate      — v1 (oracle distillation, λ override at eval)
  astrogain_e2e  — end-to-end NLL-trained K-gain

For each (model, method, trial):
  1. Sample K needles at K random positions in n_windows haystack.
  2. Build ctx; compress ONCE (no question).
  3. For each needle's (Q, A): clone compressed cache, forward Q, decode,
     score by substring containment.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from baselines.streaming_kv_core import AttentionCapture, SenseCapture, decode_from_cache
from baselines.eval_faithful_streaming_needle import (
    score_per_layer_snapkv_layerwise, score_per_layer_snapkv,
    select_topk_global, OBS_WINDOW)
from baselines.eval_needle import NEEDLES, generate_haystack
from baselines.eval_multiquery_squad import (
    _clone_cache, _find_answer, QUERY_TMPL)
from training.train_hybrid import AstroHybrid
from astronet.astro_gate import AstroGate
from astronet.astro_gain_e2e import AstroGainE2E

# Reuse existing distractor windows but build multi-needle haystacks.
def build_multineedle_haystack(n_windows, n_needles, seed):
    """Insert n_needles distinct needles at distinct random positions.
    Returns: windows, list of (question, answer) tuples in needle order."""
    rng = random.Random(seed)
    assert n_needles <= len(NEEDLES), f'only {len(NEEDLES)} needles available'
    needle_idxs = rng.sample(range(len(NEEDLES)), n_needles)
    positions = sorted(rng.sample(range(n_windows), n_needles))
    # Start from a single-needle generation, then patch in the rest.
    windows, _, _ = generate_haystack(n_windows, positions[0], needle_idxs[0],
                                       seed=seed)
    qa = [(NEEDLES[needle_idxs[0]][1], NEEDLES[needle_idxs[0]][2])]
    for p, ni in zip(positions[1:], needle_idxs[1:]):
        needle_text, q, a = NEEDLES[ni]
        prefix = windows[p].split(' ', 1)
        # Replace this window with a needle window.
        windows[p] = f"{prefix[0]} {needle_text} {prefix[1] if len(prefix) > 1 else ''}"
        qa.append((q, a))
    return windows, qa


def select_topk(scores, k, n_sink=4):
    return select_topk_global(scores, k, n_sink=n_sink)


@torch.no_grad()
def compress_once(model, tokenizer, capture, sense_cap, astro, method,
                  windows, k, n_mem, n_layers, device, selection='perlayer'):
    """Compress ctx (no question)."""
    ctx_text = '\n\n'.join(windows) + '\n\n'
    ctx_ids = tokenizer(ctx_text, return_tensors='pt',
                        max_length=131072, truncation=True
                        ).input_ids.to(device)
    ctx_len = ctx_ids.shape[1]
    obs = min(OBS_WINDOW, ctx_len - 1)

    capture.clear()
    cache = DynamicCache()
    hidden_chunks = []
    capture.enabled = False
    if sense_cap is not None:
        sense_cap.clear()
    out1 = model(input_ids=ctx_ids[:, :-obs], past_key_values=cache, use_cache=True)
    if sense_cap is not None and sense_cap.hidden is not None:
        hidden_chunks.append(sense_cap.hidden.detach())
        sense_cap.clear()
    capture.enabled = True
    out2 = model(input_ids=ctx_ids[:, -obs:],
                  past_key_values=out1.past_key_values, use_cache=True)
    ctx_cache = out2.past_key_values
    if sense_cap is not None and sense_cap.hidden is not None:
        hidden_chunks.append(sense_cap.hidden.detach())

    full_layers = [(ctx_cache[li][0], ctx_cache[li][1]) for li in range(n_layers)]

    if method in ('full', 'snapkv_oracle'):
        return None, full_layers, ctx_len

    ctx_hidden_full = None
    if astro is not None:
        astro.reset_state()
        ctx_hidden_full = torch.cat(hidden_chunks, dim=1) if hidden_chunks else None
        if ctx_hidden_full is not None:
            astro.update_state(astro.sense(ctx_hidden_full))

    k_keep = k - n_mem if method == 'astrohybrid' else k
    scores = score_per_layer_snapkv_layerwise(
        capture, ctx_cache, ctx_len, obs, n_layers)
    if method == 'astrogate' and astro is not None and ctx_hidden_full is not None:
        scores = [astro.modulate_scores(s, ctx_hidden_full) for s in scores]
    elif method == 'astrogain_e2e' and astro is not None:
        ctx_K_per_layer = [full_layers[li][0][:, :, :ctx_len, :]
                            for li in range(n_layers)]
        biased = []
        for li, s in enumerate(scores):
            gl = astro._gain_logits_only(li, ctx_K_per_layer[li])
            bias = gl.mean(dim=1).squeeze(0).to(s.device)
            biased.append(s + astro.lam.item() * bias)
        scores = biased
    per_layer_idx = [select_topk(s, k_keep) for s in scores]

    compressed = []
    for li in range(n_layers):
        K, V = full_layers[li]
        i = per_layer_idx[li].to(K.device)
        K_real, V_real = K[:, :, i, :], V[:, :, i, :]
        if method == 'astrogain_e2e':
            gain = astro.gain(li, K_real)
            K_real = K_real * gain
        compressed.append((K_real, V_real))
    return compressed, full_layers, ctx_len


@torch.no_grad()
def answer_question(model, tokenizer, capture, method, compressed, full_layers,
                    ctx_len, question, k, n_mem, n_layers, device):
    q_text = f'Question: {question}\nAnswer:'
    q_ids = tokenizer(q_text, return_tensors='pt',
                      add_special_tokens=False).input_ids.to(device)
    q_len = q_ids.shape[1]
    q_pos = torch.arange(ctx_len, ctx_len + q_len, device=device).unsqueeze(0)

    if method == 'snapkv_oracle':
        capture.clear(); capture.enabled = True
        cache = _clone_cache(full_layers)
        out_q = model(input_ids=q_ids, past_key_values=cache,
                       position_ids=q_pos, use_cache=True)
        full_q = out_q.past_key_values
        last_logit = out_q.logits[0, -1].clone()
        scores = score_per_layer_snapkv_layerwise(
            capture, full_q, ctx_len, q_len, n_layers)
        per_layer_idx = [select_topk(s, k) for s in scores]
        new_cache = DynamicCache()
        for li in range(n_layers):
            K, V = full_q[li][0], full_q[li][1]
            i = per_layer_idx[li].to(K.device)
            new_cache.update(
                torch.cat([K[:, :, i, :], K[:, :, ctx_len:, :]], dim=2),
                torch.cat([V[:, :, i, :], V[:, :, ctx_len:, :]], dim=2), li)
        gen = decode_from_cache(model, new_cache, ctx_len + q_len, last_logit,
                                 tokenizer.eos_token_id, device, max_new=30)
        return tokenizer.decode(gen, skip_special_tokens=True)

    capture.enabled = False
    if method == 'full':
        cache = _clone_cache(full_layers); offset = 0
    else:
        cache = _clone_cache(compressed); offset = 0
    out_q = model(input_ids=q_ids, past_key_values=cache,
                   position_ids=q_pos, use_cache=True)
    last_logit = out_q.logits[0, -1].clone()
    gen = decode_from_cache(model, out_q.past_key_values,
                              offset + ctx_len + q_len, last_logit,
                              tokenizer.eos_token_id, device, max_new=30)
    return tokenizer.decode(gen, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--method', required=True,
                   choices=['snapkv', 'snapkv_oracle', 'full',
                            'astrogate', 'astrogain_e2e'])
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--ckpt_tag', default='')
    p.add_argument('--n_windows', type=int, default=100,
                   help='~100 ≈ 10K tokens; 200 ≈ 20K')
    p.add_argument('--n_needles', type=int, default=4)
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=256)
    p.add_argument('--lam_override', type=float, default=None)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    if os.path.exists(args.save_path):
        print(f'SKIP {args.save_path}'); return

    print(f'[mq-needle] method={args.method} k={args.k} n_windows={args.n_windows} '
          f'n_needles={args.n_needles} trials={args.n_trials}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = str(model.get_input_embeddings().weight.device)
    capture = AttentionCapture(model)
    cfg = model.config
    nl = cfg.num_hidden_layers; nkv = cfg.num_key_value_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)

    astro = None; sense_cap = None
    if args.method in ('astrogate', 'astrogain_e2e'):
        assert args.checkpoint
        raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_cfg = raw.get('config', {}) if isinstance(raw, dict) else {}
        if args.method == 'astrogate':
            astro = AstroGate(
                hidden_dim=cfg.hidden_size,
                n_astro=saved_cfg.get('n_astro', args.n_mem),
                attn_dim=saved_cfg.get('attn_dim', args.attn_dim),
                n_patterns=saved_cfg.get('n_patterns', 8),
            ).to(device)
        else:
            astro = AstroGainE2E(
                hidden_dim=cfg.hidden_size,
                n_astro=saved_cfg.get('n_astro', args.n_mem),
                attn_dim=saved_cfg.get('attn_dim', args.attn_dim),
                n_kv_heads=nkv, head_dim=hd, n_layers=nl,
            ).to(device)
        astro.load_state_dict(
            raw['astro'] if isinstance(raw, dict) and 'astro' in raw else raw,
            strict=False)
        astro.eval()
        if args.lam_override is not None:
            import math
            new_log_lam = math.log(math.exp(args.lam_override) - 1.0)
            astro.log_lambda.data.fill_(new_log_lam)
        sl = raw.get('sense_layer', nl // 2) if isinstance(raw, dict) else nl // 2
        sense_cap = SenseCapture(model, sl)
        print(f'Loaded {args.method} ckpt; λ={astro.lam.item():.4f}', flush=True)

    correct = 0
    total = 0
    n_err = 0
    t0 = time.time()
    for t in range(args.n_trials):
        try:
            windows, qa = build_multineedle_haystack(
                args.n_windows, args.n_needles, seed=args.seed * 1000 + t)
            compressed, full_layers, ctx_len = compress_once(
                model, tokenizer, capture, sense_cap, astro, args.method,
                windows, args.k, args.n_mem, nl, device)
            for question, answer in qa:
                gen = answer_question(model, tokenizer, capture, args.method,
                                       compressed, full_layers, ctx_len, question,
                                       args.k, args.n_mem, nl, device)
                if _find_answer(gen, answer):
                    correct += 1
                total += 1
        except Exception as e:
            n_err += 1
            total += args.n_needles
            print(f'  trial {t} ERR: {type(e).__name__}: {str(e)[:120]}',
                   flush=True)
        if (t + 1) % 5 == 0:
            print(f'  {t+1}/{args.n_trials}: {correct}/{total} '
                   f'({time.time()-t0:.0f}s)', flush=True)
    acc = correct / max(1, total)
    print(f'ACC: {correct}/{total} = {acc*100:.2f}%  errors={n_err}', flush=True)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'protocol': 'multiquery_needle',
            'method': args.method,
            'model': os.path.basename(args.model_path),
            'checkpoint': args.checkpoint,
            'ckpt_tag': args.ckpt_tag,
            'k': args.k,
            'n_windows': args.n_windows,
            'n_needles': args.n_needles,
            'n_trials': args.n_trials,
            'n_questions': total,
            'seed': args.seed,
            'correct': correct,
            'accuracy': acc,
            'n_trial_errors': n_err,
        }, f, indent=2)
    print(f'Saved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
