"""End-to-end training for AstroGainE2E: NLL on answer through K-gain.

Training step (per multi-query SQuAD sample):
  1. ctx forward (no_grad). Capture sense-layer hidden state, full ctx_cache.
  2. astro.sense + EMA update buffers (keep_grad=True via sense path).
  3. Pick ONE question per paragraph (we shuffle through them across epochs).
  4. Q forward on ctx_cache (no_grad, capture enabled) → SnapKV scores.
  5. SnapKV per-layer top-k selection (vanilla, no astro involvement).
  6. For each layer ℓ:
        K_sel = full_cache[ℓ][0][:, :, idx[ℓ], :].detach()  # leaf, no grad on K itself
        V_sel = full_cache[ℓ][1][:, :, idx[ℓ], :].detach()
        gain  = astro.gain(ℓ, K_sel)                        # depends on astro state
        K_mod = K_sel * gain                                 # gradient through gain
        K_q   = full_cache[ℓ][0][:, :, ctx_len:, :].detach()
        V_q   = full_cache[ℓ][1][:, :, ctx_len:, :].detach()
        new_cache[ℓ] = (cat[K_mod, K_q], cat[V_sel, V_q])
  7. Forward A on new_cache with position_ids resuming at (ctx_len+q_len)
     and enable_grad: the 4-bit model has no trainable params but the cache
     K_mod is a leaf with grad → autograd traces logits back to gain.
  8. NLL on answer tokens. Backprop to astro.

Saves: checkpoints/astro_gain_e2e_<model>.pt
"""
from __future__ import annotations
import argparse, os, random, sys, time
from collections import defaultdict
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from baselines.streaming_kv_core import AttentionCapture
from baselines.eval_faithful_streaming_needle import (
    score_per_layer_snapkv_layerwise, OBS_WINDOW)
from data.real_qa import _load_squad, _filter_squad
from astronet.astro_gain_e2e import AstroGainE2E
from training.train_astro_gate import (
    build_multiquery_samples_split, _select_topk_idx, N_SINK)

QUERY_TMPL = ("Based on what you read earlier, answer the following "
              "question.\nQuestion: {q}\nAnswer:")


def train_step(model, tokenizer, capture, astro, optimizer, sample,
               sense_layer, k, n_layers, device):
    astro.train()
    astro.reset_state()
    ctx_text = '\n\n'.join(sample['windows']) + '\n\n'
    ctx_ids = tokenizer(ctx_text, return_tensors='pt',
                        max_length=8192, truncation=True
                        ).input_ids.to(device)
    ctx_len = ctx_ids.shape[1]
    obs = min(OBS_WINDOW, ctx_len - 1)

    # --- 1. Forward ctx, sense ---
    capture.clear(); capture.enabled = False
    with torch.no_grad():
        out1 = model(input_ids=ctx_ids[:, :-obs],
                     past_key_values=DynamicCache(),
                     use_cache=True, output_hidden_states=True)
        h1 = out1.hidden_states[sense_layer].detach()
        # Chunk 2 with capture enabled → query-agnostic obs window for SnapKV
        capture.enabled = True
        out2 = model(input_ids=ctx_ids[:, -obs:],
                     past_key_values=out1.past_key_values,
                     use_cache=True, output_hidden_states=True)
        h2 = out2.hidden_states[sense_layer].detach()
        ctx_cache = out2.past_key_values
    ctx_hidden = torch.cat([h1, h2], dim=1)
    sensed = astro.sense(ctx_hidden)
    astro.update_state(sensed, keep_grad=True)

    # --- 2. Pick ONE question (random per step, shuffled outside) ---
    question, answer = sample['qa'][sample.get('_q_idx', 0) % len(sample['qa'])]
    q_text = QUERY_TMPL.format(q=question)
    q_ids = tokenizer(q_text, return_tensors='pt',
                      add_special_tokens=False).input_ids.to(device)
    q_pos = torch.arange(ctx_len, ctx_len + q_ids.shape[1],
                          device=device).unsqueeze(0)

    # --- 3. Forward Q to get SnapKV scores (no_grad) ---
    capture.clear(); capture.enabled = True
    with torch.no_grad():
        cloned = DynamicCache()
        for li in range(n_layers):
            K, V = ctx_cache[li]
            cloned.update(K.clone(), V.clone(), li)
        out_q = model(input_ids=q_ids, past_key_values=cloned,
                      position_ids=q_pos, use_cache=True)
        full_q = out_q.past_key_values
        q_len = q_ids.shape[1]

    # --- 4. Per-layer SnapKV top-k indices ---
    scores = score_per_layer_snapkv_layerwise(
        capture, full_q, ctx_len, q_len, n_layers)
    per_layer_idx = [_select_topk_idx(s, k) for s in scores]

    # --- 5. Build new_cache with gain·K_sel + Q tail (grad-enabled) ---
    new_cache = DynamicCache()
    for li in range(n_layers):
        K_full = full_q[li][0].detach()
        V_full = full_q[li][1].detach()
        i = per_layer_idx[li].to(K_full.device)
        K_sel = K_full[:, :, i, :]
        V_sel = V_full[:, :, i, :]
        gain = astro.gain(li, K_sel)                # [1, n_kv, k, 1] WITH GRAD
        K_mod = K_sel * gain                         # GRAD THROUGH gain
        K_q = K_full[:, :, ctx_len:, :]
        V_q = V_full[:, :, ctx_len:, :]
        K_new = torch.cat([K_mod, K_q], dim=2)
        V_new = torch.cat([V_sel, V_q], dim=2)
        new_cache.update(K_new, V_new, li)

    # --- 6. Forward A through model on new_cache; NLL on answer ---
    # Use the first answer token as target — single-token NLL is enough
    # signal and avoids decoding-loop overhead.
    a_tok = tokenizer(' ' + answer, return_tensors='pt',
                       add_special_tokens=False).input_ids[0, 0]
    a_pos = torch.tensor([[ctx_len + q_len]], device=device)
    # Need a 1-token input to pass into the model so it produces a logit
    # at position ctx_len+q_len. We use a dummy padding token; the
    # logit we read is the prediction for the NEXT token, which is what
    # we score against. Actually simpler: use the LAST captured Q logit
    # from out_q above — but that used full_q (no gain). We need a
    # forward through the MODULATED cache to get the gain-affected logit.
    # Solution: take Q's last token, re-forward it on new_cache.
    last_q_tok = q_ids[:, -1:]
    last_q_pos = torch.tensor([[ctx_len + q_len - 1]], device=device)
    out_a = model(input_ids=last_q_tok, past_key_values=new_cache,
                  position_ids=last_q_pos, use_cache=True)
    logit = out_a.logits[0, -1, :]                  # [V]
    nll = F.cross_entropy(logit.unsqueeze(0), a_tok.unsqueeze(0).to(device))

    optimizer.zero_grad()
    nll.backward()
    torch.nn.utils.clip_grad_norm_(astro.parameters(), 1.0)
    optimizer.step()
    return nll.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_train', type=int, default=1500)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--n_astro', type=int, default=16)
    p.add_argument('--attn_dim', type=int, default=256)
    p.add_argument('--sense_layer', type=int, default=-1)
    p.add_argument('--save_path', required=True)
    args = p.parse_args()

    print(f'[train-astro-gain-e2e] {args.model_path} '
          f'n_train={args.n_train} epochs={args.epochs} k={args.k}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb,
        device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()
    device = str(model.get_input_embeddings().weight.device)
    capture = AttentionCapture(model)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    sense_layer = args.sense_layer if args.sense_layer >= 0 else n_layers // 2

    astro = AstroGainE2E(
        hidden_dim=cfg.hidden_size, n_astro=args.n_astro,
        attn_dim=args.attn_dim, n_kv_heads=n_kv,
        head_dim=head_dim, n_layers=n_layers,
    ).to(device)
    print(f'AstroGainE2E params: {astro.parameter_count():,}  '
          f'sense_layer={sense_layer}', flush=True)

    samples = build_multiquery_samples_split(
        'train', args.n_train, 4, 4, args.seed)
    print(f'{len(samples)} train paragraphs', flush=True)

    optimizer = torch.optim.AdamW(astro.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    t0 = time.time()
    step = 0
    losses = []
    for ep in range(args.epochs):
        print(f'\n=== epoch {ep+1}/{args.epochs} ===', flush=True)
        random.Random(args.seed + ep).shuffle(samples)
        for si, s in enumerate(samples):
            # Rotate which question we use each epoch so all Q/A get touched
            s['_q_idx'] = (ep + si) % len(s['qa'])
            try:
                loss = train_step(model, tokenizer, capture, astro,
                                   optimizer, s, sense_layer, args.k,
                                   n_layers, device)
                losses.append(loss)
                step += 1
            except Exception as e:
                print(f'  para {si} ERROR: {type(e).__name__}: {str(e)[:120]}',
                      flush=True)
                continue
            if step <= 3 or step % 25 == 0:
                recent = sum(losses[-25:]) / max(len(losses[-25:]), 1)
                print(f'  step {step:5d}: nll={recent:.4f}  '
                      f'λ={astro.lam.item():.4f}  '
                      f'αf={astro.alpha_fast.item():.3f}  '
                      f'αs={astro.alpha_slow.item():.3f}  '
                      f'({(time.time()-t0)/60:.1f}m)', flush=True)
        ep_save = args.save_path.replace('.pt', f'_ep{ep+1}.pt')
        torch.save({'astro': astro.state_dict(),
                    'config': vars(args),
                    'sense_layer': sense_layer}, ep_save)
        print(f'Saved epoch ckpt: {ep_save}', flush=True)
    torch.save({'astro': astro.state_dict(),
                'config': vars(args),
                'sense_layer': sense_layer}, args.save_path)
    print(f'Saved final: {args.save_path}', flush=True)
    print(f'Final λ={astro.lam.item():.4f}  '
          f'αf={astro.alpha_fast.item():.3f}  αs={astro.alpha_slow.item():.3f}',
          flush=True)


if __name__ == '__main__':
    main()
