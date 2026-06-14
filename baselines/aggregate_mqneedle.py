"""Aggregate the multi-query NIAH full battery into a single results JSON."""
from __future__ import annotations
import json, os, statistics, math, glob

OUT = 'logs/results/mqneedle_summary.json'


def load(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)


def acc(p):
    d = load(p)
    return d['accuracy'] * 100 if d else None


def get_method(model, method, w, n, k, seed, t, tag):
    suffix = f'_s{seed}_t{t}_{tag}'
    p = f'logs/results/mqneedle_{method}_{model}_w{w}_n{n}_k{k}{suffix}.json'
    return acc(p), p


def main():
    out = {'multi_query_NIAH': {}}

    # --- (1) 5-seed CI: default config (w=100, n=4, k=300, 20 trials) ---
    ci = {}
    for m in ('qwen7b', 'llama8b'):
        snap = []; ag = []; ds = []
        for s in range(1, 6):
            sp = f'logs/results/mqneedle_snapkv_{m}_w100_n4_k300_s{s}_t20_ci.json'
            ap = f'logs/results/mqneedle_astrogate_{m}_w100_n4_k300_s{s}_t20_ci.json'
            sv = acc(sp); av = acc(ap)
            if sv is not None and av is not None:
                snap.append(sv); ag.append(av); ds.append(av - sv)
        if not ds: continue
        sm, am, dm = statistics.mean(snap), statistics.mean(ag), statistics.mean(ds)
        ssd = statistics.stdev(snap) if len(snap) > 1 else 0
        asd = statistics.stdev(ag) if len(ag) > 1 else 0
        dsd = statistics.stdev(ds) if len(ds) > 1 else 0
        sem = dsd / math.sqrt(len(ds)); ci95 = sem * 1.96
        ci[m] = {
            'snapkv_mean': sm, 'snapkv_sem': ssd / math.sqrt(len(snap)),
            'astrogate_mean': am, 'astrogate_sem': asd / math.sqrt(len(ag)),
            'delta_mean': dm,
            'delta_95ci_low': dm - ci95, 'delta_95ci_high': dm + ci95,
            'seeds_positive': f'{sum(1 for x in ds if x > 0)}/{len(ds)}',
            'per_seed_delta': ds,
        }
    out['multi_query_NIAH']['five_seed_CI'] = ci

    # --- (2) k-budget scan (seed=1, 20 trials) ---
    kscan = {}
    for m in ('qwen7b', 'llama8b'):
        per_k = {}
        for k in (64, 100, 300, 500):
            cells = {}
            for method in ('full', 'snapkv_oracle', 'snapkv', 'astrogate'):
                if k == 300:
                    p = f'logs/results/mqneedle_{method}_{m}_w100_n4_k300_s1_t20_ci.json'
                else:
                    p = f'logs/results/mqneedle_{method}_{m}_w100_n4_k{k}_s1_t20_kscan.json'
                v = acc(p)
                if v is not None: cells[method] = v
            if 'snapkv' in cells and 'astrogate' in cells:
                cells['delta'] = cells['astrogate'] - cells['snapkv']
            per_k[str(k)] = cells
        kscan[m] = per_k
    out['multi_query_NIAH']['k_budget_scan'] = kscan

    # --- (3) Context-length scan (k=300, n=4, seed=1) ---
    cscan = {}
    for m in ('qwen7b', 'llama8b'):
        per_w = {}
        for w, t in ((50, 15), (100, 20), (200, 15), (400, 10)):
            cells = {}
            for method in ('full', 'snapkv_oracle', 'snapkv', 'astrogate'):
                if w == 100:
                    p = f'logs/results/mqneedle_{method}_{m}_w100_n4_k300_s1_t20_ci.json'
                else:
                    p = f'logs/results/mqneedle_{method}_{m}_w{w}_n4_k300_s1_t{t}_cscan.json'
                v = acc(p)
                if v is not None: cells[method] = v
            if 'snapkv' in cells and 'astrogate' in cells:
                cells['delta'] = cells['astrogate'] - cells['snapkv']
            per_w[str(w)] = cells
        cscan[m] = per_w
    out['multi_query_NIAH']['context_length_scan'] = cscan

    # --- (4) n_needles scan (w=100, k=300, seed=1) ---
    nscan = {}
    for m in ('qwen7b', 'llama8b'):
        per_n = {}
        for n, t in ((2, 20), (4, 20), (8, 10)):
            cells = {}
            for method in ('full', 'snapkv_oracle', 'snapkv', 'astrogate'):
                if n == 4:
                    p = f'logs/results/mqneedle_{method}_{m}_w100_n4_k300_s1_t20_ci.json'
                else:
                    p = f'logs/results/mqneedle_{method}_{m}_w100_n{n}_k300_s1_t{t}_nnscan.json'
                v = acc(p)
                if v is not None: cells[method] = v
            if 'snapkv' in cells and 'astrogate' in cells:
                cells['delta'] = cells['astrogate'] - cells['snapkv']
            per_n[str(n)] = cells
        nscan[m] = per_n
    out['multi_query_NIAH']['n_needles_scan'] = nscan

    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {OUT}')
    print('\n=== Headline (5-seed CI) ===')
    for m, c in ci.items():
        print(f"  {m}: SnapKV {c['snapkv_mean']:.2f}±{c['snapkv_sem']:.2f}% → "
              f"AstroGate {c['astrogate_mean']:.2f}±{c['astrogate_sem']:.2f}% "
              f"Δ={c['delta_mean']:+.2f}pp "
              f"[{c['delta_95ci_low']:+.2f}, {c['delta_95ci_high']:+.2f}] "
              f"({c['seeds_positive']} seeds +)")


if __name__ == '__main__':
    main()
