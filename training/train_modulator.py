"""AstroNet S2-mod: Memory-modulated token selection.

Instead of generating virtual KV pairs, the memory state modulates which
real tokens are selected. Biologically: astrocyte modulates synaptic
conductance rather than creating new synapses.

score_i = clamp(h_i) * clamp(c_i) * clamp(m_i)
where m_i = f(memory_state, hidden_i) is the memory-derived relevance.

All k tokens are real — memory only improves selection.
"""
import sys; sys.path.insert(0, '.')
import os, json, math, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache
from data.real_qa import generate_squad_dataset
from training.train_hybrid import AstroModulator, _select_kv


def train_epoch(model, tokenizer, modulator, optimizer, samples,
                inject_layers, sense_layer, k, device):
    """Train one epoch. The modulator learns to improve token selection."""
    modulator.train()
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv

    total_loss = 0
    n_total = 0

    for si, s in enumerate(samples):
        modulator.reset_state()
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []
        all_hidden = []  # store hidden states for memory scoring

        # Process fact windows
        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=384,
                            truncation=True).to(device)
            with torch.no_grad():
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
                all_kv[li][0].append(out.past_key_values[li][0].detach())
                all_kv[li][1].append(out.past_key_values[li][1].detach())

            # Store hidden states for memory scoring
            hidden = out.hidden_states[sense_layer].detach()
            all_hidden.append(hidden[0])  # (seq_len, hidden_dim)

            # Modulator senses and updates memory
            sensed = modulator.sense(hidden)
            n_grad_windows = max(2, len(s.windows) // 2)
            modulator.update_state(sensed, keep_grad=(wi >= len(s.windows) - 1 - n_grad_windows))

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn)

        # Cross-window Q@K scoring (same as S1)
        q_text = f'Question: {s.question}\nAnswer:'
        q_ids = tokenizer(q_text, return_tensors='pt', max_length=128,
                          truncation=True).to(device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)

        cross = torch.zeros(total, device=device)
        for li in inject_layers:
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float()).half().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                  K[hi].float().T) / math.sqrt(hd)
                cross += sc.sum(dim=(0, 1)).to(device)
        cross[:4] = -1e9

        # Select top-k tokens using S1 ONLY (pure multiplicative, no learning)
        mult = torch.clamp(heur, min=0) * torch.clamp(cross, min=0)
        mult[:4] = -1e9
        k_actual = min(k, total)
        _, idx = mult.topk(k_actual)
        idx = idx.sort().values

        # Memory modulation: compute per-token weights (THIS IS THE LEARNED PART)
        mem_scores = modulator.compute_memory_scores(all_hidden, device)
        mod_weights = torch.sigmoid(mem_scores[idx])  # (k_actual,) in [0, 1]

        # Build cache with VALUE MODULATION — astrocyte modulates synaptic strength
        cache = DynamicCache()
        for li in range(nl):
            K_sel, V_sel = _select_kv(all_kv, li, idx)
            # Modulate values: V * w (gradient flows through mod_weights!)
            V_mod = V_sel * mod_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(V_sel.device)
            cache.update(K_sel, V_mod.to(V_sel.dtype), li)

        # Forward pass with cache — teacher forcing
        query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer: {s.answer}'
        fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
        input_ids = fq['input_ids']
        pos = torch.arange(k_actual, k_actual + input_ids.shape[1],
                           device=device).unsqueeze(0)

        out = model(input_ids=input_ids, past_key_values=cache, position_ids=pos)

        logits = out.logits[0, :-1]
        targets = input_ids[0, 1:]
        loss = F.cross_entropy(logits.float(), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modulator.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_total += 1

        if (si + 1) % 10 == 0:
            print(f'  train {si+1}/{len(samples)}: loss={total_loss/n_total:.3f} '
                  f'alpha={modulator.alpha.item():.3f}',
                  flush=True)

    return total_loss / max(n_total, 1)


def evaluate(model, tokenizer, modulator, samples, inject_layers, sense_layer, k, device):
    """Evaluate: compare modulated (S1+S2) vs pure (S1)."""
    modulator.eval()
    nl = model.config.num_hidden_layers
    nq = model.config.num_attention_heads
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', model.config.hidden_size // nq)
    qpk = nq // nkv

    correct_mod = 0
    correct_pure = 0

    for si, s in enumerate(samples):
        modulator.reset_state()
        all_kv = {li: ([], []) for li in range(nl)}
        all_attn = []
        all_hidden = []

        for wi in range(len(s.windows) - 1):
            ids = tokenizer(s.windows[wi], return_tensors='pt', max_length=384,
                            truncation=True).to(device)
            with torch.no_grad():
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
            with torch.no_grad():
                hidden = out.hidden_states[sense_layer]
                all_hidden.append(hidden[0])
                sensed = modulator.sense(hidden)
                modulator.update_state(sensed)

        total = sum(a.shape[0] for a in all_attn)
        heur = torch.cat(all_attn)

        # Cross-window scoring
        q_ids = tokenizer(f'Question: {s.question}\nAnswer:', return_tensors='pt',
                          max_length=128, truncation=True).to(device)
        with torch.no_grad():
            q_out = model(input_ids=q_ids['input_ids'], output_hidden_states=True)
        cross = torch.zeros(total, device=device)
        for li in inject_layers:
            Q = model.model.layers[li].self_attn.q_proj(
                q_out.hidden_states[li][0].float()).half().view(-1, nq, hd)
            K = torch.cat(all_kv[li][0], dim=2)[0]
            for hi in range(nkv):
                sc = torch.matmul(Q[:, hi*qpk:(hi+1)*qpk, :].float(),
                                  K[hi].float().T) / math.sqrt(hd)
                cross += sc.sum(dim=(0, 1)).to(device)
        cross[:4] = -1e9

        mult_pure = torch.clamp(heur, min=0) * torch.clamp(cross, min=0)
        mult_pure[:4] = -1e9

        # Select tokens with S1 (same for both pure and modulated)
        k_actual = min(k, total)
        _, idx = mult_pure.topk(k_actual)
        idx = idx.sort().values

        # Memory modulation weights
        with torch.no_grad():
            mem_scores = modulator.compute_memory_scores(all_hidden, device)
            mod_weights = torch.sigmoid(mem_scores[idx])

        # Generate with both: pure (no modulation) and modulated (value scaling)
        for label in ['pure', 'modulated']:
            cache = DynamicCache()
            for li in range(nl):
                K_sel, V_sel = _select_kv(all_kv, li, idx)
                if label == 'modulated':
                    V_sel = V_sel * mod_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(V_sel.device)
                cache.update(K_sel, V_sel.to(K_sel.dtype), li)

            query = f'Based on what you read earlier, answer the following question.\nQuestion: {s.question}\nAnswer:'
            fq = tokenizer(query, return_tensors='pt', max_length=384, truncation=True).to(device)
            pos = torch.arange(k_actual, k_actual + fq['input_ids'].shape[1],
                               device=device).unsqueeze(0)
            cur = fq['input_ids']; cc = cache; gen = []
            with torch.no_grad():
                for _ in range(20):
                    o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
                    cc = o.past_key_values
                    nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    gen.append(nxt[0, 0].item()); cur = nxt
                    pos = torch.tensor([[k_actual + fq['input_ids'].shape[1] + len(gen) - 1]],
                                       device=device)
                    if nxt[0, 0].item() == tokenizer.eos_token_id:
                        break
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            if s.answer.lower() in text.lower():
                if label == 'pure':
                    correct_pure += 1
                else:
                    correct_mod += 1

        if (si + 1) % 25 == 0:
            print(f'  eval {si+1}/{len(samples)}: pure={correct_pure}/{si+1}  '
                  f'modulated={correct_mod}/{si+1}', flush=True)

    pure_acc = correct_pure / len(samples)
    mod_acc = correct_mod / len(samples)
    return pure_acc, mod_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./models/qwen2.5-7b')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_eval', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--k', type=int, default=300, help='Total KV budget (all real tokens)')
    parser.add_argument('--n_queries', type=int, default=16)
    parser.add_argument('--attn_dim', type=int, default=256)
    parser.add_argument('--sense_layer', type=int, default=None)
    parser.add_argument('--n_windows', type=int, default=0, help='0=variable 5-15')
    parser.add_argument('--memory_bank', action='store_true')
    parser.add_argument('--train_seed', type=int, default=42)
    parser.add_argument('--eval_seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                              bnb_4bit_quant_type='nf4')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
        quantization_config=bnb, device_map={'': args.device}, torch_dtype=torch.float16)
    model.eval()

    hidden_dim = model.config.hidden_size
    nl = model.config.num_hidden_layers
    nkv = model.config.num_key_value_heads
    hd = getattr(model.config, 'head_dim', hidden_dim // model.config.num_attention_heads)
    inject_layers = [nl // 4, nl // 2, 3 * nl // 4, nl - 2]
    sense_layer = args.sense_layer or nl // 2

    print(f"Model: {nl} layers, {hidden_dim} hidden, inject={inject_layers}, sense={sense_layer}", flush=True)

    modulator = AstroModulator(
        hidden_dim=hidden_dim,
        n_queries=args.n_queries,
        attn_dim=args.attn_dim,
    ).to(args.device)
    if args.memory_bank:
        modulator.use_memory_bank = True
        print("Memory bank mode ENABLED", flush=True)
    print(f"AstroModulator: {modulator.parameter_count():,} params", flush=True)

    optimizer = torch.optim.AdamW(modulator.parameters(), lr=args.lr, weight_decay=0.01)

    # Data
    if args.n_windows == 0:
        from data.longctx_qa import generate_longctx_dataset
        train_samples = generate_longctx_dataset(tokenizer, n_samples=args.n_train,
                                                  n_windows='variable', chunk_size=384,
                                                  seed=args.train_seed, split='train',
                                                  include_answer=True)
    else:
        train_samples = generate_squad_dataset(n_samples=args.n_train, n_windows=args.n_windows,
                                                vary_distance=True, seed=args.train_seed,
                                                split='train', include_answer=True)
    eval_samples = generate_squad_dataset(n_samples=args.n_eval, n_windows=5,
                                           vary_distance=True, seed=args.eval_seed, split='validation')
    print(f"Train: {len(train_samples)}, Eval: {len(eval_samples)}", flush=True)

    best_mod = 0
    results = []

    for epoch in range(args.epochs + 1):
        if epoch == 0:
            print(f"\n{'='*50}\nEpoch 0 (before training)\n{'='*50}", flush=True)
        else:
            print(f"\n{'='*50}\nEpoch {epoch}\n{'='*50}", flush=True)
            t0 = time.time()
            train_loss = train_epoch(model, tokenizer, modulator, optimizer,
                                      train_samples, inject_layers, sense_layer,
                                      args.k, args.device)
            print(f"  train: loss={train_loss:.3f} ({time.time()-t0:.0f}s)", flush=True)

        pure_acc, mod_acc = evaluate(model, tokenizer, modulator, eval_samples,
                                      inject_layers, sense_layer, args.k, args.device)
        print(f"  eval: pure={pure_acc:.1%}  modulated={mod_acc:.1%}", flush=True)
        results.append({'epoch': epoch, 'pure': pure_acc, 'modulated': mod_acc})

        if mod_acc > best_mod:
            best_mod = mod_acc
            os.makedirs('checkpoints', exist_ok=True)
            model_name = os.path.basename(args.model_path).replace('.', '_')
            bank_tag = '_bank' if args.memory_bank else ''
            win_tag = f'_w{args.n_windows}' if args.n_windows != 5 else ''
            ckpt_path = f'checkpoints/astro_modulator_{model_name}_k{args.k}_q{args.n_queries}{win_tag}{bank_tag}.pt'
            torch.save(modulator.state_dict(), ckpt_path)
            print(f"  Saved to {ckpt_path}", flush=True)
            print(f"  NEW BEST: {best_mod:.1%}", flush=True)

    print(f"\nBest modulated: {best_mod:.1%}", flush=True)


if __name__ == '__main__':
    main()
