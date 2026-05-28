"""Multi-turn KV cache amortization benchmark.

Setup: each instance is a 5-window factual context where each window contains
one ground-truth fact. The user asks 5 sequential turns of questions, one per
window. We compare three cache policies:

1) **fresh_full**: rebuild the full FP16 KV cache from scratch every turn.
   This is the upper bound on accuracy and a lower bound on cost.
2) **fresh_hybrid**: rebuild the AstroNet S1+S2 cache (k=300) from scratch
   every turn. Our default 'single-shot' deployment.
3) **amortized_hybrid**: process the windows once, freeze the S2 summary
   tokens, cache the full KV. On each turn, re-run only Stage-1 cross-window
   query-key scoring against the cached keys, take top-(k-K) real tokens,
   and prepend the cached S2 summary. The expensive per-window forward pass
   is performed only once across the entire 5-turn conversation.

Metrics:
- Mean per-turn forward-pass wall-clock time.
- Mean per-turn accuracy (exact substring match of ground-truth fact token).
- Cumulative 5-turn compute cost.

The interesting comparison is amortized_hybrid vs fresh_hybrid: same final
accuracy, but the cumulative cost should be ~1.2x single-turn rather than ~5x.
"""
import sys, os, json, time, math, argparse, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from astronet.wrapper import AstroNetWrapper


# ---------------------------------------------------------------------------
# Synthetic 5-window factoid instances.
# Each instance: 5 windows, each with a unique fact (a name + a code-like value).
# Five turn questions, one per fact, in random order.
# ---------------------------------------------------------------------------

CATEGORIES = [
    ('access code', ['ALPHA-7829-OMEGA', 'BRAVO-3194-DELTA', 'CHARLIE-5582-ZULU',
                     'ECHO-9043-LIMA', 'FOXTROT-6178-KILO', 'GAMMA-2461-SIERRA',
                     'HOTEL-7314-TANGO', 'JULIET-8825-VICTOR']),
    ('serial number', ['SN-49217', 'SN-78531', 'SN-31096', 'SN-65724', 'SN-90148',
                       'SN-23687', 'SN-41530', 'SN-58292']),
    ('badge ID', ['BG-1129', 'BG-5837', 'BG-2604', 'BG-9716', 'BG-4082',
                  'BG-3375', 'BG-6948', 'BG-8120']),
    ('passphrase', ['QUARTZ-7', 'GRANITE-3', 'OBSIDIAN-9', 'BASALT-5',
                    'PUMICE-2', 'MARBLE-8', 'SLATE-4', 'JADE-6']),
    ('lock combination', ['12-47-93', '24-08-67', '36-19-85', '48-72-14',
                          '51-90-26', '63-35-78', '79-04-52', '85-61-20']),
]
DISTRACTORS = [
    "Marine biology studies organisms in the sea and their ecological interactions.",
    "Solar energy capacity doubled between 2020 and 2024 due to falling panel costs.",
    "Classical music spans several centuries from Bach to Stravinsky.",
    "Modern architecture emphasises clean lines and open spaces.",
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose.",
    "The Roman Empire reached its greatest territorial extent under Trajan.",
    "Quantum entanglement remains one of the most counterintuitive results of physics.",
    "Plate tectonics explains continental drift and mountain formation.",
]


def make_instance(rng):
    categories = rng.sample(CATEGORIES, 5)  # pick 5 distinct categories
    windows = []
    facts = []
    for cat_name, values in categories:
        value = rng.choice(values)
        distractor = rng.choice(DISTRACTORS)
        # Embed the fact in a longer window with a distractor sentence.
        window_text = (f"{distractor} The {cat_name} is {value}. "
                       f"This information should be retained for later use.")
        windows.append(window_text)
        facts.append((cat_name, value))
    rng.shuffle(windows)  # randomise position of each fact
    # Recompute which window holds each fact after shuffle.
    # Easier: shuffle (window, cat_name, value) triples together.
    triples = list(zip(windows, [c for c, v in facts], [v for c, v in facts]))
    rng.shuffle(triples)
    windows = [t[0] for t in triples]
    cat_value = [(t[1], t[2]) for t in triples]
    # Five turns: one question per fact, asked in random order.
    turn_order = list(range(5))
    rng.shuffle(turn_order)
    turns = []
    for idx in turn_order:
        cat, val = cat_value[idx]
        question = f"What is the {cat}?"
        turns.append({'question': question, 'answer_token': val})
    return {'windows': windows, 'turns': turns}


def evaluate(args):
    rng = random.Random(args.seed)
    instances = [make_instance(rng) for _ in range(args.n_samples)]

    device_map = 'auto' if args.multi_gpu else {'': args.device}
    print(f"Loading {args.model_path}", flush=True)
    wrapper = AstroNetWrapper.from_pretrained(
        args.model_path, astro_ckpt=args.hybrid_checkpoint,
        attn_dim=args.attn_dim, device_map=device_map,
    )
    model = wrapper.model
    tok = wrapper.tokenizer
    device = wrapper.device

    results = {'config': vars(args), 'instances': []}

    for inst_idx, inst in enumerate(instances):
        windows = inst['windows']
        turns = inst['turns']
        inst_rec = {'idx': inst_idx, 'turns': []}

        # --------------------------------------------------------------
        # Condition (1) fresh_hybrid: rebuild the hybrid cache every turn.
        # --------------------------------------------------------------
        per_turn_fresh = []
        for turn in turns:
            t0 = time.time()
            ans = wrapper.answer(windows, turn['question'], k=args.k,
                                 method='hybrid', max_new_tokens=24)
            elapsed = time.time() - t0
            correct = turn['answer_token'] in ans
            per_turn_fresh.append({'time_sec': elapsed, 'correct': correct, 'answer': ans})

        # --------------------------------------------------------------
        # Condition (2) amortized_hybrid: process windows once; per-turn
        # only the cross-window query-key score + generation.
        # --------------------------------------------------------------
        # Pre-process windows once.
        t0 = time.time()
        all_kv, total, last_start = wrapper._process_windows(windows)
        preprocess_time = time.time() - t0

        per_turn_amort = []
        for turn in turns:
            t0 = time.time()
            # Re-score Stage 1 against this turn's question.
            cross = wrapper._compute_cross(all_kv, total, turn['question'])
            # Pick top-(k - n_mem) tokens.
            last_window_size = all_kv[wrapper.inject_layers[0]][0][-1].shape[2]
            n_sink = 4
            k_real = args.k - wrapper.n_mem
            n_recent = min(int(k_real * 0.2), last_window_size)
            recent_idx = torch.arange(last_start, last_start + n_recent, device=device)
            scores = cross.clone()
            scores[last_start:total] = -1e9
            n_select = max(k_real - n_sink - n_recent, 0)
            n_avail = (scores > -1e8).sum().item()
            n_select = min(n_select, n_avail, scores.shape[0])
            _, top = scores.topk(n_select) if n_select > 0 else \
                     (None, torch.empty(0, dtype=torch.long, device=device))
            sink_idx = torch.arange(n_sink, device=device)
            idx = torch.cat([sink_idx, top, recent_idx]).unique().sort().values[:k_real]
            from transformers import DynamicCache
            nl = model.config.num_hidden_layers
            cache = DynamicCache()
            for li in range(nl):
                K_real, V_real = wrapper._select_kv(all_kv, idx, li)
                K_mem, V_mem = wrapper.astro.generate_kv(
                    li, (K_real.to(device), V_real.to(device)))
                cache.update(torch.cat([K_mem.to(K_real.device), K_real], dim=2),
                             torch.cat([V_mem.to(V_real.device), V_real], dim=2), li)
            prefix_len = wrapper.n_mem + len(idx)
            q = f'Based on what you read, answer the question.\nQuestion: {turn["question"]}\nAnswer:'
            fq = tok(q, return_tensors='pt', max_length=256, truncation=True).to(device)
            pos = torch.arange(prefix_len, prefix_len + fq['input_ids'].shape[1],
                               device=device).unsqueeze(0)
            cur, cc, gen = fq['input_ids'], cache, []
            with torch.no_grad():
                for _ in range(24):
                    o = model(input_ids=cur, past_key_values=cc, position_ids=pos)
                    cc = o.past_key_values
                    nxt = o.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
                    gen.append(nxt[0, 0].item())
                    cur = nxt
                    pos = torch.tensor([[prefix_len + fq['input_ids'].shape[1] + len(gen) - 1]],
                                       device=device)
                    if nxt[0, 0].item() == tok.eos_token_id:
                        break
            ans = tok.decode(gen, skip_special_tokens=True).strip()
            elapsed = time.time() - t0
            correct = turn['answer_token'] in ans
            per_turn_amort.append({'time_sec': elapsed, 'correct': correct, 'answer': ans})

        inst_rec['preprocess_time_sec'] = preprocess_time
        inst_rec['fresh'] = per_turn_fresh
        inst_rec['amortized'] = per_turn_amort
        results['instances'].append(inst_rec)
        if (inst_idx + 1) % 5 == 0:
            print(f"  [{inst_idx+1}/{len(instances)}] inst done", flush=True)

    # --------------------------------------------------------------
    # Aggregate.
    # --------------------------------------------------------------
    fresh_times = [t['time_sec'] for inst in results['instances'] for t in inst['fresh']]
    amort_times = [t['time_sec'] for inst in results['instances'] for t in inst['amortized']]
    fresh_corr = [t['correct'] for inst in results['instances'] for t in inst['fresh']]
    amort_corr = [t['correct'] for inst in results['instances'] for t in inst['amortized']]
    preprocess_times = [inst['preprocess_time_sec'] for inst in results['instances']]

    summary = {
        'fresh_hybrid_mean_per_turn_sec': sum(fresh_times) / max(len(fresh_times), 1),
        'fresh_hybrid_accuracy': sum(fresh_corr) / max(len(fresh_corr), 1),
        'amortized_hybrid_mean_per_turn_sec': sum(amort_times) / max(len(amort_times), 1),
        'amortized_hybrid_accuracy': sum(amort_corr) / max(len(amort_corr), 1),
        'amortized_preprocess_sec': sum(preprocess_times) / max(len(preprocess_times), 1),
        'cumulative_5turn_fresh': 5 * sum(fresh_times) / max(len(fresh_times), 1),
        'cumulative_5turn_amortized': (
            sum(preprocess_times) / max(len(preprocess_times), 1)
            + 5 * sum(amort_times) / max(len(amort_times), 1)
        ),
    }
    summary['amortization_speedup'] = (
        summary['cumulative_5turn_fresh'] / summary['cumulative_5turn_amortized']
        if summary['cumulative_5turn_amortized'] > 0 else float('nan')
    )
    results['summary'] = summary

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.save_path}", flush=True)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--hybrid_checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--n_samples', type=int, default=50)
    p.add_argument('--n_turns', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=300)
    p.add_argument('--attn_dim', type=int, default=512)
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--save_path', default=None)
    args = p.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
