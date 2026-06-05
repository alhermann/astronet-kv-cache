"""Query-ablation pos-robust eval.

Tests whether Stage 1's gains depend on having an explicit user question.
The scoring query is replaced by the trailing N tokens of the last context
segment (a distractor paragraph in our SQuAD setup, deliberately unrelated
to the answer).  Generation still receives the real question -- the model
must still produce an answer; we are only ablating what Stage 1 uses to
RANK the cache, not what the model is asked to do.

Three predictions to test:
  (a) AstroNet-real  >> AstroNet-fallback   -> the explicit question is
      doing most of the work for Stage 1 selection (paper's claim).
  (b) AstroNet-fallback ~ SnapKV-fallback   -> when both methods use the
      same query, AstroNet's machinery is no worse than SnapKV.
  (c) AstroNet-fallback < AstroNet-real but > random-selection
      -> Stage 2 EMA still helps even without a good selection query.

This script measures (a) directly.  (b) and (c) come from cross-referencing
with the existing SnapKV and pure-recency results.
"""
import sys, os, json, argparse, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import DynamicCache

sys.path.insert(0, '.')
from training.train_hybrid import AstroHybrid, _select_kv
from training.eval_hybrid_position_robust import shuffle_fact_position
from data.real_qa import generate_squad_dataset


def make_fallback_query(text: str, tokenizer, n_tokens: int) -> str:
    """Take the LAST n_tokens tokens of `text` and decode back to a string."""
    ids = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
    if ids.shape[0] <= n_tokens:
        return text
    tail = ids[-n_tokens:]
    return tokenizer.decode(tail, skip_special_tokens=True)


@torch.no_grad()
def evaluate_with_query_override(model, tokenizer, astro, samples,
                                  inject_layers, sense_layer, k_real, device,
                                  query_mode: str, n_trailing: int):
    """Like train_hybrid.evaluate(), but the scoring query is overridable.

    query_mode:
      'real'        : use s.question (paper's setting; sanity check)
      'trailing'    : use last `n_trailing` tokens of s.windows[-2]
                       (the last context segment that the cache loop saw)
    Generation always uses s.question.
    """
    astro.eval()
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv
    n_mem = astro.n_mem_tokens
    correct_hybrid = 0
    correct_pure = 0

    for si, s in enumerate(samples):
        astro.reset_state()
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []

        # --- forward each context window once, build K,V cache + Stage 2 state
        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=384,
                            truncation=True).to(device)
            out = model(input_ids=ids['input_ids'], use_cache=True,
                        output_attentions=True, output_hidden_states=True)
            sl = ids['input_ids'].shape[1]
            imp = torch.zeros(sl, device=device)
            for la in out.attentions:
                a = la[0].to(device)
                if not a.isnan().any():
                    imp += a.sum(dim=(0, 1))
            imp[:4] = -1e9
            all_attn.append(imp)
            for li in range(nl):
                all_kv[li][0].append(out.past_key_values[li][0])
                all_kv[li][1].append(out.past_key_values[li][1])
            hidden = out.hidden_states[sense_layer]
            sensed = astro.sense(hidden)
            astro.update_state(sensed)

        total = sum(a.shape[0] for a in all_attn)

        # --- choose what text to use as the SCORING query
        if query_mode == 'real':
            scoring_query_text = s.question
        elif query_mode == 'trailing':
            # IMPORTANT: at pos_3 the fact paragraph is s.windows[-2] (the
            # last context window).  Using its trailing tokens as the
            # fallback query would leak the answer's vocabulary into S1
            # scoring.  Pick a window that is GUARANTEED to be a distractor
            # for every pos in {0,1,2,3}: with 5 windows and pos in 0..3,
            # the fact lives at exactly one of windows[0..3] and
            # windows[-3] = windows[2] is a distractor whenever the fact
            # is not there; when the fact IS at pos=2 we fall back to
            # windows[1] (also a distractor by construction).  We pick the
            # earliest non-fact context window so the trailing-32 tokens
            # are always from unrelated SQuAD text.
            ctx_idx = max(0, len(s.windows) - 3)
            if ctx_idx == s.fact_window:
                ctx_idx = max(0, ctx_idx - 1)
            scoring_query_text = make_fallback_query(s.windows[ctx_idx], tokenizer, n_trailing)
        elif query_mode == 'empty':
            scoring_query_text = ''
        else:
            raise ValueError(f"unknown query_mode {query_mode}")

        # --- Stage 1 multiplicative scoring (uses scoring_query_text)
        q_ids = tokenizer(f'Question: {scoring_query_text}\nAnswer:',
                          return_tensors='pt', max_length=128, truncation=True).to(device)
        q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
        cross = torch.zeros(total, device=device)
        for li in inject_layers:
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float()).half().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                  K[hi].float().T) / math.sqrt(hd)
                attn_w = torch.softmax(sc, dim=-1)
                cross += attn_w.sum(dim=(0, 1)).to(device)
        if total > 5:
            cross = torch.nn.functional.avg_pool1d(
                cross.unsqueeze(0).unsqueeze(0),
                kernel_size=5, padding=2, stride=1,
            ).squeeze()
        cross[:4] = -1e9

        last_window_size = all_kv[inject_layers[0]][0][-1].shape[2]
        last_start = total - last_window_size

        # ===== Pure-300 path (S1 only) =====
        k_pure = min(300, total)
        n_recent_p = min(int(k_pure * 0.2), last_window_size)
        recent_p = torch.arange(last_start, last_start + n_recent_p, device=device)
        scores_p = cross.clone()
        scores_p[last_start:total] = -1e9
        n_select_p = max(k_pure - 4 - len(recent_p), 0)
        n_avail_p = (scores_p > -1e8).sum().item()
        n_select_p = min(n_select_p, n_avail_p, scores_p.shape[0])
        top_p = scores_p.topk(n_select_p)[1] if n_select_p > 0 else torch.empty(0, dtype=torch.long, device=device)
        idx_pure = torch.cat([torch.arange(4, device=device), top_p, recent_p]).unique().sort().values[:k_pure]
        k_pure = len(idx_pure)
        cache_pure = DynamicCache()
        for li in range(nl):
            K_p, V_p = _select_kv(all_kv, li, idx_pure)
            cache_pure.update(K_p, V_p, li)
        q_text = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
        fq_p = tokenizer(q_text, return_tensors='pt', max_length=384, truncation=True).to(device)
        pos_p = torch.arange(k_pure, k_pure + fq_p['input_ids'].shape[1], device=device).unsqueeze(0)
        cur_p, cc_p, gen_p = fq_p['input_ids'], cache_pure, []
        for _ in range(20):
            o_p = model(input_ids=cur_p, past_key_values=cc_p, position_ids=pos_p)
            cc_p = o_p.past_key_values
            nxt_p = o_p.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen_p.append(nxt_p[0, 0].item()); cur_p = nxt_p
            pos_p = torch.tensor([[k_pure + fq_p['input_ids'].shape[1] + len(gen_p) - 1]], device=device)
            if nxt_p[0, 0].item() == tokenizer.eos_token_id: break
        text_pure = tokenizer.decode(gen_p, skip_special_tokens=True).strip()
        if s.answer.lower() in text_pure.lower():
            correct_pure += 1

        # ===== Hybrid path (S1 + S2 virtual KV) =====
        k_hyb_target = min(k_real, total)
        n_recent_h = min(int(k_hyb_target * 0.2), last_window_size)
        recent_h = torch.arange(last_start, last_start + n_recent_h, device=device)
        scores_h = cross.clone()
        scores_h[last_start:total] = -1e9
        n_select_h = max(k_hyb_target - 4 - len(recent_h), 0)
        n_avail_h = (scores_h > -1e8).sum().item()
        n_select_h = min(n_select_h, n_avail_h, scores_h.shape[0])
        top_h = scores_h.topk(n_select_h)[1] if n_select_h > 0 else torch.empty(0, dtype=torch.long, device=device)
        idx_hyb = torch.cat([torch.arange(4, device=device), top_h, recent_h]).unique().sort().values[:k_hyb_target]
        k_hyb = len(idx_hyb)
        cache_hyb = DynamicCache()
        for li in range(nl):
            K_real_l, V_real_l = _select_kv(all_kv, li, idx_hyb)
            K_mem, V_mem = astro.generate_kv(li, (K_real_l, V_real_l))
            K_mem, V_mem = K_mem.to(K_real_l.device), V_mem.to(V_real_l.device)
            cache_hyb.update(torch.cat([K_mem, K_real_l], dim=2),
                              torch.cat([V_mem, V_real_l], dim=2), li)
        total_len = n_mem + k_hyb
        fq = tokenizer(q_text, return_tensors='pt', max_length=384, truncation=True).to(device)
        pos = torch.arange(total_len, total_len + fq['input_ids'].shape[1], device=device).unsqueeze(0)
        cur, cc, gen = fq['input_ids'], cache_hyb, []
        for _ in range(20):
            o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
            cc = o.past_key_values
            nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            gen.append(nxt[0, 0].item()); cur = nxt
            pos = torch.tensor([[total_len + fq['input_ids'].shape[1] + len(gen) - 1]], device=device)
            if nxt[0, 0].item() == tokenizer.eos_token_id: break
        text_hyb = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if s.answer.lower() in text_hyb.lower():
            correct_hybrid += 1

        if (si + 1) % 25 == 0:
            print(f'  eval {si+1}/{len(samples)}: pure300={correct_pure}/{si+1}  '
                  f'hybrid={correct_hybrid}/{si+1}', flush=True)

    return correct_pure / len(samples), correct_hybrid / len(samples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_eval', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k_real', type=int, default=284)
    p.add_argument('--n_mem', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=512)
    p.add_argument('--positions', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--query_mode', choices=['real', 'trailing', 'empty'], default='trailing',
                   help="real = paper setting; trailing = last N tokens of an "
                        "always-distractor context window; empty = no query text")
    p.add_argument('--n_trailing', type=int, default=32)
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    print(f'[query-ablation] model={args.model_path}  query_mode={args.query_mode}  '
          f'n_trailing={args.n_trailing}', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type='nf4')
    if args.multi_gpu:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map='auto', torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path,
            quantization_config=bnb, device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = str(model.get_input_embeddings().weight.device)

    hidden_dim = model.config.hidden_size
    nl = model.config.num_hidden_layers
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
    inject_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    sense_layer = nl // 2

    astro = AstroHybrid(
        hidden_dim=hidden_dim, n_mem_tokens=args.n_mem, attn_dim=args.attn_dim,
        n_kv_heads=nkv, head_dim=hd, n_layers=nl, inject_layers=inject_layers,
    ).to(device)
    astro.extract_model_weights(model, device)
    astro.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False),
                          strict=False)
    astro.eval()
    print(f'Loaded checkpoint: {args.checkpoint} ({astro.parameter_count():,} params)', flush=True)

    base = generate_squad_dataset(n_samples=args.n_eval, n_windows=5,
                                   vary_distance=True, seed=args.seed, split='validation')
    results = {}
    for pos in args.positions:
        samples = shuffle_fact_position(base, pos)
        print(f'\n=== fact_pos={pos} ({len(samples)} samples) ===', flush=True)
        pure, hyb = evaluate_with_query_override(
            model, tokenizer, astro, samples, inject_layers, sense_layer,
            args.k_real, device, args.query_mode, args.n_trailing,
        )
        results[f'pos_{pos}'] = {'pure300': pure, 'hybrid': hyb, 'delta': hyb - pure}
        print(f'  pos={pos}: pure300={pure:.3f}  hybrid={hyb:.3f}  delta={hyb-pure:+.3f}', flush=True)

    avg_pure = sum(v['pure300'] for v in results.values()) / len(results)
    avg_hyb = sum(v['hybrid'] for v in results.values()) / len(results)
    print(f'\nAverage: pure300={avg_pure:.3f} hybrid={avg_hyb:.3f} '
          f'delta={avg_hyb-avg_pure:+.3f}', flush=True)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump({
            'model': os.path.basename(args.model_path),
            'checkpoint': args.checkpoint,
            'query_mode': args.query_mode,
            'n_trailing': args.n_trailing,
            'n_eval': args.n_eval, 'seed': args.seed,
            'k_real': args.k_real, 'n_mem': args.n_mem,
            'results': results,
            'average': {'pure300': avg_pure, 'hybrid': avg_hyb, 'delta': avg_hyb - avg_pure},
        }, f, indent=2)
    print(f'Saved to {args.save_path}', flush=True)


if __name__ == '__main__':
    main()
