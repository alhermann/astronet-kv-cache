"""Figure 1 v3: rich Pareto frontier with ALL baselines included.

Compared to v2:
- Adds StreamingLLM, H2O (was: dropped to supplement)
- Adds KIVI K4V4 (was: missing)
- Extends y-axis 0-100% so weak baselines are visible
- Tightens x-axis lower bound (no Full-FP16 marker yet)
- Pareto envelope, AstroNet traversal line preserved

Data sources per panel:
- baselines_<model>_k300.json:   streaming_llm, h2o, snapkv, multiplicative, rag_k1
- kivi_<model>.json:              k4v4 accuracy (at our k=300 + KIVI quant)
- pareto_data.json:               pos_robust_pure, pos_robust_hybrid, byte budgets
- Hard-coded RAG k=3 dict         (computed elsewhere, n=200)
"""
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
PARETO = ROOT / 'logs/results/pareto_data.json'
OUT = Path(__file__).resolve().parent / 'fig1_pareto.pdf'

# ---- per-model file mapping ------------------------------------------------
BASELINE_FILE = {
    'qwen7b':     'baselines_qwen2.5-7b_k300.json',
    'qwen14b':    'baselines_qwen2.5-14b_k300.json',
    'qwen32b':    'baselines_qwen2.5-32b_k300.json',
    'llama8b':    'baselines_llama-3.1-8b_k300.json',
    'mistral7b':  'baselines_mistral-7b-v0.3_k300.json',
    'mistral24b': 'baselines_mistral-small-24b_k300.json',
}
KIVI_FILE = {
    'qwen7b':     'kivi_qwen7b.json',
    'qwen14b':    'kivi_qwen2_5_14b.json',
    'qwen32b':    'kivi_qwen2_5_32b_v3.json',
    'llama8b':    'kivi_llama_3_1_8b.json',
    'mistral7b':  'kivi_mistral_7b_v0_3.json',
    'mistral24b': 'kivi_mistral_small_24b.json',
}
FULL_FP16_FILE = {
    'qwen7b':     'full_fp16_qwen2_5-7b.json',
    'qwen14b':    'full_fp16_qwen2_5-14b.json',
    'qwen32b':    'full_fp16_qwen2_5-32b.json',
    'llama8b':    'full_fp16_llama-3_1-8b.json',
    'mistral7b':  'full_fp16_mistral-7b-v0_3.json',
    'mistral24b': 'full_fp16_mistral-small-24b.json',
}

# ---- styling: Okabe-Ito colourblind-safe palette ---------------------------
COLORS = {
    'full_fp16':      '#000000',   # black (reference)
    'streaming_llm':  '#999999',   # grey
    'h2o':            '#0072B2',   # blue
    'snapkv':         '#56B4E9',   # light blue
    'kivi_k4v4':      '#CC79A7',   # pink
    'multiplicative': '#F0E442',   # yellow
    'hybrid':         '#D55E00',   # orange
    'hybrid_k8v4':    '#E69F00',   # gold
    'rag_k1':         '#009E73',   # green
    'rag_k3':         '#117733',   # dark green
}
MARKERS = {
    'full_fp16': 'h',
    'streaming_llm': 'v', 'h2o': 'o', 'snapkv': 's', 'kivi_k4v4': 'X',
    'multiplicative': '^', 'hybrid': 'D', 'hybrid_k8v4': '*',
    'rag_k1': 'P', 'rag_k3': 'p',
}
LABELS = {
    'full_fp16':      'Full FP16 (no compression)',
    'streaming_llm':  'StreamingLLM',
    'h2o':            r'H${}_2$O',
    'snapkv':         'SnapKV',
    'kivi_k4v4':      'KIVI K4V4',
    'multiplicative': 'AstroNet S1',
    'hybrid':         'AstroNet S1+S2',
    'hybrid_k8v4':    'AstroNet S1+S2+K8V4',
    'rag_k1':         r'RAG $k{=}1$',
    'rag_k3':         r'RAG $k{=}3$',
}

ORDER = ['qwen7b', 'qwen14b', 'qwen32b', 'llama8b', 'mistral7b', 'mistral24b']
PRETTY = {
    'qwen7b': 'Qwen 2.5-7B', 'qwen14b': 'Qwen 2.5-14B', 'qwen32b': 'Qwen 2.5-32B',
    'llama8b': 'Llama 3.1-8B', 'mistral7b': 'Mistral 7B v0.3',
    'mistral24b': 'Mistral-Small 24B',
}
RAG_K3 = {'qwen7b': 0.80, 'qwen14b': 0.765, 'qwen32b': 0.795,
          'llama8b': 0.835, 'mistral7b': 0.68, 'mistral24b': 0.805}

mpl.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 100,
})


def load_baselines(tag):
    p = ROOT / 'logs/results' / BASELINE_FILE[tag]
    return json.load(open(p))['results']


def load_full_fp16(tag):
    p = ROOT / 'logs/results' / FULL_FP16_FILE[tag]
    if not p.exists():
        print(f'  WARNING: Full-FP16 file missing for {tag}: {p.name}', flush=True)
        return None
    data = json.load(open(p))
    avg = data.get('average', {}).get('full_fp16')
    return avg


def load_kivi_k4v4(tag):
    p = ROOT / 'logs/results' / KIVI_FILE[tag]
    if not p.exists():
        print(f'  WARNING: KIVI file missing for {tag}: {p.name}', flush=True)
        return None
    data = json.load(open(p))
    r = data.get('results') or data
    if 'k4v4' in r:
        v = r['k4v4']
        return v['accuracy'] if isinstance(v, dict) else v
    print(f'  WARNING: no k4v4 key in {p.name}', flush=True)
    return None


def compression_kivi_k4v4(comp_k300_fp16):
    """KIVI K4V4 applied on top of our k=300 selection: bytes are k=300 * (4+4)/(16+16) = 1/4."""
    return comp_k300_fp16 * (32 / 8)


def build_points(row, tag):
    """Return list of (method, compression_ratio, accuracy_pct) tuples for this backbone."""
    base = load_baselines(tag)
    bytes_full = row['bytes_fp16_full_n20']
    bytes_k300_fp16 = row['bytes_fp16_k300']
    bytes_k300_k8v4 = row['bytes_k8v4_k300']
    bytes_rag1 = (384 / 300) * bytes_k300_fp16
    bytes_rag3 = 3 * bytes_rag1

    comp_k300_fp16 = bytes_full / bytes_k300_fp16
    comp_k300_k8v4 = bytes_full / bytes_k300_k8v4
    comp_rag1 = bytes_full / bytes_rag1
    comp_rag3 = bytes_full / bytes_rag3

    acc_streaming = base['streaming_llm'] * 100
    acc_h2o = base['h2o'] * 100
    acc_snapkv = base['snapkv'] * 100
    acc_s1 = row['pos_robust_pure'] * 100
    acc_s1s2 = row['pos_robust_hybrid'] * 100
    acc_rag1 = base.get('rag_k1', 0.775) * 100
    acc_rag3 = RAG_K3[tag] * 100

    pts = [
        ('streaming_llm', comp_k300_fp16, acc_streaming),
        ('h2o',           comp_k300_fp16, acc_h2o),
        ('snapkv',        comp_k300_fp16, acc_snapkv),
        ('multiplicative', comp_k300_fp16, acc_s1),
        ('hybrid',        comp_k300_fp16, acc_s1s2),
        ('hybrid_k8v4',   comp_k300_k8v4, acc_s1s2),
        ('rag_k1',        comp_rag1,      acc_rag1),
        ('rag_k3',        comp_rag3,      acc_rag3),
    ]

    kivi_acc = load_kivi_k4v4(tag)
    if kivi_acc is not None:
        comp_kivi = compression_kivi_k4v4(comp_k300_fp16)
        pts.append(('kivi_k4v4', comp_kivi, kivi_acc * 100))

    full_acc = load_full_fp16(tag)
    if full_acc is not None:
        # Compression ratio = 1.0 (no compression — that's the definition of full FP16)
        pts.append(('full_fp16', 1.0, full_acc * 100))

    return pts


def dominated(p, others):
    x, y = p[1], p[2]
    for o in others:
        if o is p: continue
        if o[1] >= x and o[2] >= y and (o[1] > x or o[2] > y):
            return True
    return False


def render():
    rows = json.load(open(PARETO))['rows']

    fig, axes = plt.subplots(2, 3, figsize=(18 / 2.54, 13 / 2.54), sharey=True)

    for i, tag in enumerate(ORDER):
        ax = axes[i // 3, i % 3]
        row = next(r for r in rows if r['tag'] == tag)
        pts = build_points(row, tag)

        # ---- Pareto envelope + shaded dominated region -----------------------
        frontier = [p for p in pts if not dominated(p, pts)]
        frontier = sorted(frontier, key=lambda p: p[1])
        if frontier:
            xs = [p[1] for p in frontier] + [200]
            ys = [p[2] for p in frontier] + [frontier[-1][2]]
            ax.fill_between(xs, [0] * len(xs), ys,
                             color='#cccccc', alpha=0.18, zorder=0)
            ax.plot([p[1] for p in frontier], [p[2] for p in frontier],
                    color='#444444', lw=1.3, alpha=0.85, ls='--', zorder=2)

        # ---- AstroNet traversal (S1 -> S1+S2 -> S1+S2+K8V4) -----------------
        astro_pts = sorted(
            [p for p in pts if p[0] in ('multiplicative', 'hybrid', 'hybrid_k8v4')],
            key=lambda p: p[1])
        ax.plot([p[1] for p in astro_pts], [p[2] for p in astro_pts],
                color=COLORS['hybrid'], lw=1.4, alpha=0.55, zorder=3)

        # ---- markers --------------------------------------------------------
        for method, x, y in pts:
            if method == 'hybrid_k8v4':
                edge, lw, size = 'black', 1.0, 100
            elif method == 'full_fp16':
                edge, lw, size = 'black', 0.8, 80
            else:
                edge, lw, size = COLORS[method], 0.5, 48
            ax.scatter(x, y, c=COLORS[method], marker=MARKERS[method],
                       s=size, edgecolors=edge, linewidths=lw, zorder=5,
                       label=LABELS[method] if i == 0 else None)

        # ---- axis cosmetics -------------------------------------------------
        ax.set_xscale('log')
        ax.set_xlim(0.8, 130)
        ax.set_ylim(0, 100)
        ax.set_xticks([1, 3, 7, 25, 70])
        ax.set_xticklabels(['1', '3', '7', '25', '70'])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, which='both', alpha=0.2, zorder=0)
        ax.set_title(PRETTY[tag], fontsize=10, fontweight='normal', pad=4)
        if i % 3 == 0:
            ax.set_ylabel('SQuAD accuracy (%)')
        if i // 3 == 1:
            ax.set_xlabel(r'Cache compression factor ($\times$)')

    # ---- shared legend below the panels ------------------------------------
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.04), frameon=False,
               columnspacing=1.4, handletextpad=0.4)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUT, bbox_inches='tight', dpi=300)
    print(f'Saved {OUT}')
    print(f'  Size: {OUT.stat().st_size / 1024:.1f} KB')


if __name__ == '__main__':
    render()
