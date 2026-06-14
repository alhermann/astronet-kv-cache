"""Build a 4-model (and growing) scaling table for the multi-query NIAH
benchmark. Reads anchor-seed results + λ-sweep results for every backbone
that has them on disk, picks the best λ for each, and writes:

  logs/results/mqneedle_4model_scaling.json

Backbones currently tracked:
    qwen7b, llama8b   — 5-seed CI from mqneedle_summary.json (if present)
    mistral7b, qwen14b — anchor-seed + λ sweep
    qwen32b, mistral24b — picked up automatically if files exist

This script is additive: it ignores backbones with no data and never
overwrites the original 5-seed-CI numbers.
"""
from __future__ import annotations
import json, os, glob, statistics, math

OUT = 'logs/results/mqneedle_4model_scaling.json'
RESULTS_DIR = 'logs/results'


def _acc(path):
    if not os.path.exists(path): return None
    with open(path) as f: d = json.load(f)
    return d['accuracy'] * 100


def collect_anchor(model):
    """Return dict with snapkv/astrogate accuracies at anchor seed=42."""
    out = {}
    for method in ('full', 'snapkv_oracle', 'snapkv', 'astrogate'):
        p = f'{RESULTS_DIR}/mqneedle_{method}_{model}_w100_n4_k300_s42_t20_anchor.json'
        v = _acc(p)
        if v is not None: out[method] = v
    return out


def collect_lambda_sweep(model):
    """Return {λ: acc} dict from all λ-sweep files for this model."""
    out = {}
    pat = f'{RESULTS_DIR}/mqneedle_astrogate_{model}_w100_n4_k300_s42_t20_lam*.json'
    for p in sorted(glob.glob(pat)):
        v = _acc(p)
        if v is None: continue
        tag = p.rsplit('_lam', 1)[1].rsplit('.json', 1)[0]
        lam = float(tag.replace('p', '.'))
        out[lam] = v
    return out


def best_lambda(sweep, anchor_acc, anchor_lam=0.5):
    """Pick best λ across sweep+anchor; return (best_lam, best_acc, candidates)."""
    cands = dict(sweep)
    if anchor_acc is not None and anchor_lam not in cands:
        cands[anchor_lam] = anchor_acc
    if not cands: return None, None, {}
    best_lam = max(cands, key=lambda k: cands[k])
    return best_lam, cands[best_lam], cands


def five_seed_from_summary(model):
    """Pull 5-seed CI block for this model from the original aggregator."""
    p = f'{RESULTS_DIR}/mqneedle_summary.json'
    if not os.path.exists(p): return None
    with open(p) as f: d = json.load(f)
    return d.get('multi_query_NIAH', {}).get('five_seed_CI', {}).get(model)


def main():
    rows = {}
    for model in ('qwen7b', 'llama8b', 'qwen14b', 'mistral7b',
                  'qwen32b', 'mistral24b'):
        anchor = collect_anchor(model)
        sweep = collect_lambda_sweep(model)
        ci = five_seed_from_summary(model)

        if not anchor and not sweep and not ci: continue   # backbone untouched

        ag_anchor = anchor.get('astrogate')
        best_lam, best_acc, grid = best_lambda(sweep, ag_anchor)

        # SnapKV anchor is the headline-baseline; full/oracle are sanity rails.
        snap = anchor.get('snapkv')
        full = anchor.get('full')
        oracle = anchor.get('snapkv_oracle')

        # If we have a 5-seed CI use it as the headline; otherwise anchor.
        if ci is not None:
            headline = {
                'source': '5-seed CI',
                'snapkv_mean': ci['snapkv_mean'],
                'snapkv_sem': ci['snapkv_sem'],
                'astrogate_mean': ci['astrogate_mean'],
                'astrogate_sem': ci['astrogate_sem'],
                'delta_mean': ci['delta_mean'],
                'delta_95ci': [ci['delta_95ci_low'], ci['delta_95ci_high']],
                'seeds_positive': ci['seeds_positive'],
            }
        elif snap is not None and best_acc is not None:
            headline = {
                'source': 'anchor seed=42 + λ sweep',
                'snapkv_acc': snap,
                'astrogate_best_acc': best_acc,
                'astrogate_best_lambda': best_lam,
                'delta': best_acc - snap,
            }
        else:
            headline = {'source': 'incomplete'}

        rows[model] = {
            'headline': headline,
            'anchor_results': anchor,
            'lambda_sweep': {f'{k:g}': v for k, v in sorted(grid.items())},
            'best_lambda': best_lam,
            'best_lambda_acc': best_acc,
        }

    with open(OUT, 'w') as f:
        json.dump({'multi_query_NIAH_scaling': rows,
                   'note': ('Headline uses 5-seed CI when available, '
                            'else anchor seed=42 + λ-grid pick.')}, f, indent=2)

    print(f'Wrote {OUT}\n')
    print('=== Multi-query NIAH scaling table (best λ per backbone) ===')
    print(f"{'backbone':<12}  {'snap':>7}  {'astrogate':>10}  {'Δ':>9}  source")
    print('-' * 56)
    for m, r in rows.items():
        h = r['headline']
        if h['source'].startswith('5-seed'):
            sk = f"{h['snapkv_mean']:.2f}"
            ag = f"{h['astrogate_mean']:.2f}"
            d = f"{h['delta_mean']:+.2f}pp"
            src = '5-seed CI'
        elif h['source'].startswith('anchor'):
            sk = f"{h['snapkv_acc']:.2f}"
            ag = f"{h['astrogate_best_acc']:.2f} (λ={h['astrogate_best_lambda']:g})"
            d = f"{h['delta']:+.2f}pp"
            src = 'anchor'
        else:
            sk = ag = d = '—'; src = h['source']
        print(f"{m:<12}  {sk:>7}  {ag:>10}  {d:>9}  {src}")


if __name__ == '__main__':
    main()
