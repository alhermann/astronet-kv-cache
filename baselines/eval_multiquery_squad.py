"""Multi-query compress-once SQuAD eval (query-agnostic compression regime).

The faithful-streaming SQuAD/LongBench protocol hands the compressor the
question BEFORE selection: SnapKV's observation window is the query itself.
That is generous to query-aware selection and unrepresentative of cache
reuse: in a compress-once / answer-many setting (prompt caching, streaming
ingestion) the question is unknown at compression time.

Protocol here:
  1. Build a context of paragraphs (1 fact paragraph with >=2 SQuAD
     questions + distractor paragraphs, fact at seeded random position).
     NO question is appended.
  2. Forward the context in two chunks; only the last OBS_WINDOW tokens run
     with AttentionCapture, giving published-SnapKV scoring with a
     context-tail observation window (the query-agnostic variant).
  3. Compress ONCE:  snapkv keeps top-k ctx tokens; astrohybrid keeps its
     n_mem ctx-sensed virtual tokens + top-(k-n_mem) ctx tokens.
  4. For EACH question of the fact paragraph: clone the compressed cache,
     forward the query at positions [ctx_len..), greedy decode, score by
     substring containment.

Reference methods:
  snapkv_oracle: per-question query-aware recompression of the full ctx
      cache (the standard protocol; needs the query before selection).
  full: no compression (ceiling).

Usage::

    python baselines/eval_multiquery_squad.py \
        --model_path ./models/qwen2.5-7b --method astrohybrid \
        --checkpoint checkpoints/astro_hybrid_snapkv_s1_qwen7b.pt \
        --ckpt_tag fixb --k 300 --n_paragraphs 50 --seed 42 \
        --save_path logs/results/multiquery_squad_astrohybrid_qwen7b_k300_fixb.json
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from collections import defaultdict
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from baselines.streaming_kv_core import (
    AttentionCapture, SenseCapture, decode_from_cache)
from baselines.eval_faithful_streaming_needle import (
    score_per_layer_snapkv, score_per_layer_snapkv_layerwise,
    select_topk_global, OBS_WINDOW)
from data.real_qa import _load_squad, _filter_squad
from training.train_hybrid import AstroHybrid
from astronet.astro_gate import AstroGate
from astronet.astro_gain_e2e import AstroGainE2E

QUERY_TMPL = ("Based on what you read earlier, answer the following "
              "question.\nQuestion: {q}\nAnswer:")


def _find_answer(text: str, answer: str) -> bool:
    return answer.strip().lower() in text.strip().lower()


def build_multiquery_samples(n_paragraphs, n_ctx_windows, max_q, seed):
    """Group SQuAD val by paragraph; keep paragraphs with >=2 questions."""
    rng = random.Random(seed)
    filtered = _filter_squad(_load_squad('validation'))
    by_ctx = defaultdict(list)
    for ex in filtered:
        by_ctx[ex['context']].append(ex)
    groups = [(ctx, exs) for ctx, exs in by_ctx.items() if len(exs) >= 2]
    rng.shuffle(groups)
    by_title = defaultdict(list)
    for ex in filtered:
        by_title[ex['title']].append(ex)
    titles = list(by_title.keys())

    samples = []
    for ctx, exs in groups[:n_paragraphs]:
        fact_title = exs[0]['title']
        distractors = []
        attempts = 0
        while len(distractors) < n_ctx_windows - 1 and attempts < 100:
            t = rng.choice(titles)
            if t != fact_title and by_title[t]:
                cand = rng.choice(by_title[t])['context']
                if cand != ctx and cand not in distractors:
                    distractors.append(cand)
            attempts += 1
        windows = list(distractors)
        fact_pos = rng.randrange(n_ctx_windows)
        windows.insert(fact_pos, ctx)
        qa = [(ex['question'], ex['answers']['text'][0]) for ex in exs[:max_q]]
        samples.append({'windows': windows, 'fact_pos': fact_pos, 'qa': qa})
    return samples


def _clone_cache(src_layers, device=None):
    c = DynamicCache()
    for li, (K, V) in enumerate(src_layers):
        c.update(K.clone(), V.clone(), li)
    return c


@torch.no_grad()
def compress_once(model, tokenizer, capture, sense_cap, astro, method,
                  windows, k, n_mem, n_layers, device, selection):
    """Forward ctx (no query), compress once.  Returns (compressed_layers,
    full_layers, ctx_len).  compressed_layers is None for oracle/full."""
    ctx_text = '\n\n'.join(windows) + '\n\n'
    ctx_ids = tokenizer(ctx_text, return_tensors='pt',
                        max_length=131072, truncation=True
                        ).input_ids.to(device)
    ctx_len = ctx_ids.shape[1]
    obs = min(OBS_WINDOW, ctx_len - 1)

    capture.clear()
    cache = DynamicCache()
    hidden_chunks = []
    # Chunk 1: context body, capture disabled (memory: no Q storage).
    capture.enabled = False
    if sense_cap is not None:
        sense_cap.clear()
    out1 = model(input_ids=ctx_ids[:, :-obs], past_key_values=cache,
                 use_cache=True)
    if sense_cap is not None and sense_cap.hidden is not None:
        hidden_chunks.append(sense_cap.hidden.detach())
        sense_cap.clear()
    # Chunk 2: last obs tokens, capture enabled -> query-agnostic obs window.
    capture.enabled = True
    out2 = model(input_ids=ctx_ids[:, -obs:], past_key_values=out1.past_key_values,
                 use_cache=True)
    ctx_cache = out2.past_key_values
    if sense_cap is not None and sense_cap.hidden is not None:
        hidden_chunks.append(sense_cap.hidden.detach())

    full_layers = [(ctx_cache[li][0], ctx_cache[li][1])
                   for li in range(n_layers)]

    if method in ('full', 'snapkv_oracle'):
        return None, full_layers, ctx_len

    ctx_hidden_full = None
    if astro is not None:
        astro.reset_state()
        ctx_hidden_full = torch.cat(hidden_chunks, dim=1)
        sensed = astro.sense(ctx_hidden_full)
        astro.update_state(sensed)

    k_keep = k - n_mem if method == 'astrohybrid' else k
    if selection == 'perlayer':
        scores = score_per_layer_snapkv_layerwise(
            capture, ctx_cache, ctx_len, obs, n_layers)
        if method == 'astrogate' and astro is not None:
            scores = [astro.modulate_scores(s, ctx_hidden_full) for s in scores]
        elif method == 'astrogain_e2e' and astro is not None:
            # Use trained gain as a per-layer selection bias: gain logits
            # for full ctx K → averaged over heads, added to per-layer SnapKV.
            ctx_K_per_layer = [full_layers[li][0][:, :, :ctx_len, :]
                                for li in range(n_layers)]
            biased = []
            with torch.no_grad():
                for li, s in enumerate(scores):
                    gl = astro._gain_logits_only(li, ctx_K_per_layer[li])
                    bias = gl.mean(dim=1).squeeze(0).to(s.device)
                    biased.append(s + astro.lam.item() * bias)
            scores = biased
        per_layer_idx = [select_topk_global(s, k_keep) for s in scores]
    else:
        cross = score_per_layer_snapkv(capture, ctx_cache, ctx_len, obs, n_layers)
        if method == 'astrogate' and astro is not None:
            cross = astro.modulate_scores(cross, ctx_hidden_full)
        per_layer_idx = [select_topk_global(cross, k_keep)] * n_layers

    compressed = []
    for li in range(n_layers):
        K, V = full_layers[li]
        i = per_layer_idx[li].to(K.device)
        K_real, V_real = K[:, :, i, :], V[:, :, i, :]
        if method == 'astrohybrid':
            K_mem, V_mem = astro.generate_kv(li, (K_real, V_real))
            K_real = torch.cat([K_mem.to(K_real.device), K_real], dim=2)
            V_real = torch.cat([V_mem.to(V_real.device), V_real], dim=2)
        elif method == 'astrogain_e2e':
            # Apply trained gain on selected K (the modulation that was
            # trained against answer NLL).
            with torch.no_grad():
                gain = astro.gain(li, K_real)
            K_real = K_real * gain
        compressed.append((K_real, V_real))
    return compressed, full_layers, ctx_len


@torch.no_grad()
def answer_question(model, tokenizer, capture, method, compressed_layers,
                    full_layers, ctx_len, question, k, n_mem, n_layers,
                    device, selection):
    q_ids = tokenizer(QUERY_TMPL.format(q=question), return_tensors='pt',
                      add_special_tokens=False).input_ids.to(device)
    q_len = q_ids.shape[1]
    q_pos = torch.arange(ctx_len, ctx_len + q_len, device=device).unsqueeze(0)

    if method == 'snapkv_oracle':
        # Standard query-aware protocol, per question, on the full cache.
        capture.clear()
        capture.enabled = True
        cache = _clone_cache(full_layers)
        out_q = model(input_ids=q_ids, past_key_values=cache,
                      position_ids=q_pos, use_cache=True)
        full_q = out_q.past_key_values
        last_logit = out_q.logits[0, -1].clone()
        if selection == 'perlayer':
            scores = score_per_layer_snapkv_layerwise(
                capture, full_q, ctx_len, q_len, n_layers)
            per_layer_idx = [select_topk_global(s, k) for s in scores]
        else:
            cross = score_per_layer_snapkv(capture, full_q, ctx_len, q_len, n_layers)
            per_layer_idx = [select_topk_global(cross, k)] * n_layers
        new_cache = DynamicCache()
        for li in range(n_layers):
            K, V = full_q[li][0], full_q[li][1]
            i = per_layer_idx[li].to(K.device)
            new_cache.update(
                torch.cat([K[:, :, i, :], K[:, :, ctx_len:, :]], dim=2),
                torch.cat([V[:, :, i, :], V[:, :, ctx_len:, :]], dim=2), li)
        gen = decode_from_cache(model, new_cache, ctx_len + q_len, last_logit,
                                tokenizer.eos_token_id, device, max_new=20)
        return tokenizer.decode(gen, skip_special_tokens=True)

    capture.enabled = False
    if method == 'full':
        cache = _clone_cache(full_layers)
        offset = 0
    else:
        cache = _clone_cache(compressed_layers)
        offset = n_mem if method == 'astrohybrid' else 0
    # astrogate: no virtual tokens, offset = 0 (handled by the else above)
    out_q = model(input_ids=q_ids, past_key_values=cache,
                  position_ids=q_pos, use_cache=True)
    last_logit = out_q.logits[0, -1].clone()
    gen = decode_from_cache(model, out_q.past_key_values,
                            offset + ctx_len + q_len, last_logit,
                            tokenizer.eos_token_id, device, max_new=20)
    return tokenizer.decode(gen, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--method', required=True,
                   choices=['snapkv', 'astrohybrid', 'astrogate',
                            'astrogain_e2e', 'snapkv_oracle', 'full'])
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--ckpt_tag', default='')
    p.add_argument('--n_paragraphs', type=int, default=50)
    p.add_argument('--max_q', type=int, default=4)
    p.add_argument('--n_ctx_windows', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=256)
    p.add_argument('--selection', default='perlayer',
                   choices=['global', 'perlayer'])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--lam_override', type=float, default=None,
                   help='astrogate only: override the learned λ (softplus(log_lambda)) '
                        'to test how much modulation the trained gate logits warrant.')
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    if os.path.exists(args.save_path):
        print(f'SKIP: {args.save_path} already exists', flush=True)
        return

    print(f'[multiquery-squad] method={args.method} k={args.k} '
          f'n_paragraphs={args.n_paragraphs} ckpt_tag={args.ckpt_tag}',
          flush=True)
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
    nl = cfg.num_hidden_layers
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    inject_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]

    astro = None
    sense_cap = None
    if args.method == 'astrohybrid':
        assert args.checkpoint, '--checkpoint required for astrohybrid'
        astro = AstroHybrid(
            hidden_dim=cfg.hidden_size, n_mem_tokens=args.n_mem,
            attn_dim=args.attn_dim, n_kv_heads=nkv, head_dim=hd,
            n_layers=nl, inject_layers=inject_layers,
        ).to(device)
        astro.extract_model_weights(model, device)
        # CPU-offload the fp32 kv projection bank (2GB on 48-layer models).
        for li in list(astro._kv_weights.keys()):
            astro._kv_weights[li] = {
                'k': astro._kv_weights[li]['k'].cpu(),
                'v': astro._kv_weights[li]['v'].cpu()}
        torch.cuda.empty_cache()
        raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
        astro.load_state_dict(
            raw['astro'] if isinstance(raw, dict) and 'astro' in raw else raw,
            strict=False)
        astro.eval()
        sense_cap = SenseCapture(model, nl // 2)
        print(f'Loaded astro ckpt: {args.checkpoint}', flush=True)
    elif args.method == 'astrogain_e2e':
        assert args.checkpoint, '--checkpoint required for astrogain_e2e'
        raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_cfg = raw.get('config', {}) if isinstance(raw, dict) else {}
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
        sl = raw.get('sense_layer', nl // 2) if isinstance(raw, dict) else nl // 2
        sense_cap = SenseCapture(model, sl)
        print(f'Loaded AstroGainE2E ckpt: {args.checkpoint}  sense_layer={sl}  '
              f'lambda={astro.lam.item():.4f}  '
              f'alpha_fast={astro.alpha_fast.item():.3f}  '
              f'alpha_slow={astro.alpha_slow.item():.3f}', flush=True)
    elif args.method == 'astrogate':
        assert args.checkpoint, '--checkpoint required for astrogate'
        raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_cfg = raw.get('config', {}) if isinstance(raw, dict) else {}
        astro = AstroGate(
            hidden_dim=cfg.hidden_size,
            n_astro=saved_cfg.get('n_astro', args.n_mem),
            attn_dim=saved_cfg.get('attn_dim', args.attn_dim),
            n_patterns=saved_cfg.get('n_patterns', 8),
        ).to(device)
        astro.load_state_dict(
            raw['astro'] if isinstance(raw, dict) and 'astro' in raw else raw,
            strict=False)
        astro.eval()
        if args.lam_override is not None:
            import math
            # Invert softplus: log_lambda = log(exp(λ) - 1)
            new_log_lam = math.log(math.exp(args.lam_override) - 1.0)
            astro.log_lambda.data.fill_(new_log_lam)
        sl = raw.get('sense_layer', nl // 2) if isinstance(raw, dict) else nl // 2
        sense_cap = SenseCapture(model, sl)
        print(f'Loaded AstroGate ckpt: {args.checkpoint}  sense_layer={sl}  '
              f'lambda={astro.lam.item():.3f}  '
              f'alpha_fast={astro.alpha_fast.item():.3f}  '
              f'alpha_slow={astro.alpha_slow.item():.3f}', flush=True)

    samples = build_multiquery_samples(args.n_paragraphs, args.n_ctx_windows,
                                        args.max_q, args.seed)
    n_q_total = sum(len(s['qa']) for s in samples)
    print(f'{len(samples)} paragraphs, {n_q_total} questions', flush=True)

    correct = 0
    total = 0
    n_err = 0
    t0 = time.time()
    for si, s in enumerate(samples):
        try:
            compressed, full_layers, ctx_len = compress_once(
                model, tokenizer, capture, sense_cap, astro, args.method,
                s['windows'], args.k, args.n_mem, nl, device, args.selection)
            for question, answer in s['qa']:
                text = answer_question(
                    model, tokenizer, capture, args.method, compressed,
                    full_layers, ctx_len, question, args.k, args.n_mem,
                    nl, device, args.selection)
                if _find_answer(text, answer): correct += 1
                total += 1
        except Exception as e:
            n_err += 1
            total += len(s['qa'])
            print(f'  paragraph {si} ERROR: {type(e).__name__}: {e}'[:200],
                  flush=True)
        if (si + 1) % 10 == 0:
            print(f'  {si+1}/{len(samples)} paragraphs: {correct}/{total}  '
                  f'({time.time()-t0:.0f}s)', flush=True)
    acc = correct / max(1, total)
    print(f'ACC: {correct}/{total} = {acc*100:.2f}%  errors={n_err}', flush=True)

    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'protocol': 'multiquery_compress_once',
            'method': args.method,
            'model': os.path.basename(args.model_path),
            'checkpoint': args.checkpoint,
            'ckpt_tag': args.ckpt_tag,
            'k': args.k,
            'n_mem': args.n_mem if args.method == 'astrohybrid' else None,
            'selection': args.selection,
            'n_paragraphs': args.n_paragraphs,
            'max_q': args.max_q,
            'n_ctx_windows': args.n_ctx_windows,
            'n_questions': total,
            'seed': args.seed,
            'obs_window': 'last %d ctx tokens (query-agnostic)' % OBS_WINDOW,
            'correct': correct,
            'accuracy': acc,
            'n_paragraph_errors': n_err,
        }, f, indent=2)
    print(f'Saved -> {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
