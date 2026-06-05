"""Build the query-ablation slope chart (replaces or accompanies tab:queryablation).

Three categorical x-positions (Real, Empty, Trailing-32), one line per
backbone.  The parallel downward slopes make the "graceful degradation"
claim visual; the Mistral lines visibly flatten.

All values from logs/results/query_ablation_<mode>_<model>.json (verified).
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'fig_queryablation_slope.pdf'

# Verified hybrid accuracies (single seed=42, n=100, 4 positions averaged).
QA = {
    'Qwen 2.5 7B':       {'Real': 62.5, 'Empty': 43.2, 'Trailing-32': 39.8},
    'Qwen 2.5 14B':      {'Real': 65.8, 'Empty': 48.2, 'Trailing-32': 42.2},
    'Qwen 2.5 32B':      {'Real': 67.0, 'Empty': 53.0, 'Trailing-32': 47.2},
    'Llama 3.1 8B':      {'Real': 57.5, 'Empty': 47.0, 'Trailing-32': 43.5},
    'Mistral 7B':        {'Real': 53.8, 'Empty': 50.5, 'Trailing-32': 47.0},
    'Mistral-Small 24B': {'Real': 63.0, 'Empty': 56.5, 'Trailing-32': 53.5},
}

X_ORDER = ['Real', 'Empty', 'Trailing-32']
COLORS = {
    'Qwen 2.5 7B':       '#1f77b4',
    'Qwen 2.5 14B':      '#5fa8e0',
    'Qwen 2.5 32B':      '#08306b',
    'Llama 3.1 8B':      '#2ca02c',
    'Mistral 7B':        '#d62728',
    'Mistral-Small 24B': '#8c564b',
}


def render():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.linewidth': 0.7,
        'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = [0, 1, 2]
    for name, row in QA.items():
        ys = [row[m] for m in X_ORDER]
        ax.plot(x, ys, color=COLORS[name], marker='o', markersize=5,
                lw=1.8, label=name, alpha=0.9)
        # Annotate endpoints
        ax.annotate(f'{ys[0]:.1f}', (x[0]-0.06, ys[0]), va='center', ha='right',
                    fontsize=8, color=COLORS[name])
        ax.annotate(f'{ys[2]:.1f}', (x[2]+0.06, ys[2]), va='center', ha='left',
                    fontsize=8, color=COLORS[name])

    ax.set_xticks(x)
    ax.set_xticklabels(X_ORDER)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel('Hybrid accuracy (\%)')
    ax.set_xlabel('Scoring query mode')
    ax.set_ylim(30, 72)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    plt.subplots_adjust(left=0.13, right=0.92, top=0.95, bottom=0.27)
    plt.savefig(OUT)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    render()
