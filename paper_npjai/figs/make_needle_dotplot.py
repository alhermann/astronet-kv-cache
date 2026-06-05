"""Build the needle-at-n=20 dot plot (replaces tab:needle).

Horizontal Y axis = backbone, X = accuracy.  Five method dots per
row (StreamingLLM, H2O, SnapKV, AstroNet S1, AstroNet S1+S2).
A faint connector from SnapKV to AstroNet S1+S2 makes the gain visible.
"""
from __future__ import annotations
import json
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'fig_needle_dotplot.pdf'

# Disk values verified earlier: from needle_realsnapkv_<model>_k300.json
# n20 averages across 5 depths.  Plus paper_results_complete.json for
# StreamingLLM and H2O.
NEEDLE = {
    'Qwen 2.5 7B':       {'StreamingLLM': 40, 'H2O': 15, 'SnapKV': 72, 'AstroNet S1': 72, 'AstroNet S1+S2': 91},
    'Qwen 2.5 14B':      {'StreamingLLM': 35, 'H2O':  2, 'SnapKV': 79, 'AstroNet S1': 79, 'AstroNet S1+S2': 100},
    'Qwen 2.5 32B':      {'StreamingLLM': 40, 'H2O': 19, 'SnapKV': 50, 'AstroNet S1': 50, 'AstroNet S1+S2': 100},
    'Llama 3.1 8B':      {'StreamingLLM': 38, 'H2O':  2, 'SnapKV': 58, 'AstroNet S1': 58, 'AstroNet S1+S2': 95},
    'Mistral 7B':        {'StreamingLLM': 22, 'H2O':  0, 'SnapKV':  5, 'AstroNet S1':  5, 'AstroNet S1+S2': 32},
    'Mistral-Small 24B': {'StreamingLLM': 40, 'H2O':  0, 'SnapKV': 73, 'AstroNet S1': 73, 'AstroNet S1+S2': 81},
}

METHOD_ORDER = ['StreamingLLM', 'H2O', 'SnapKV', 'AstroNet S1', 'AstroNet S1+S2']
METHOD_COLOR = {
    'StreamingLLM':   '#6c757d',  # grey
    'H2O':            '#1f77b4',  # blue
    'SnapKV':         '#17becf',  # cyan
    'AstroNet S1':    '#ffbb33',  # yellow/orange
    'AstroNet S1+S2': '#d62728',  # red
}
METHOD_MARKER = {
    'StreamingLLM':   'v',
    'H2O':            'o',
    'SnapKV':         's',
    'AstroNet S1':    '^',
    'AstroNet S1+S2': '*',
}


def render():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.linewidth': 0.7,
        'pdf.fonttype': 42,
    })
    backbones = list(NEEDLE.keys())
    y_pos = list(range(len(backbones)))[::-1]  # top to bottom

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    # Connector from SnapKV to AstroNet S1+S2 (the headline gain)
    for y, name in zip(y_pos, backbones):
        x_snap = NEEDLE[name]['SnapKV']
        x_hyb  = NEEDLE[name]['AstroNet S1+S2']
        ax.plot([x_snap, x_hyb], [y, y], color='#cccccc', lw=1.0, zorder=1)

    # Method dots
    for method in METHOD_ORDER:
        xs = [NEEDLE[name][method] for name in backbones]
        ax.scatter(
            xs, y_pos,
            c=METHOD_COLOR[method],
            marker=METHOD_MARKER[method],
            s=68 if method == 'AstroNet S1+S2' else 38,
            label=method,
            zorder=3,
            edgecolors='black',
            linewidths=0.4,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(backbones)
    ax.set_xlabel('Needle-in-a-haystack accuracy at $n{=}20$ segments (\%)')
    ax.set_xlim(-3, 105)
    ax.grid(axis='x', linestyle=':', linewidth=0.5, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=5,
        frameon=False,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.3,
    )
    plt.subplots_adjust(left=0.21, right=0.97, top=0.97, bottom=0.22)
    plt.savefig(OUT)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    render()
