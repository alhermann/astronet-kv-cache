"""Build the kconfound dumbbell plot (replaces tab:kconfound).

One row per backbone.  Dumbbell from S1 @ k=284 (matched-budget
control) to S1+S2 (16+284), with the multi-seed delta annotated
to the right.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'fig_kconfound_dumbbell.pdf'

# Verified disk values, n=20 needle, 100 trials per cell.
# S1@300, S1@284, S1+S2 (16+284), Δ multi-seed (mean ± std).
KCONFOUND = {
    'Qwen 2.5 7B':       {'s1_300':  74, 's1_284': 82,  's12':  88, 'delta':  9.0, 'std': 2.2, 'n_seeds': 4},
    'Qwen 2.5 14B':      {'s1_300':  78, 's1_284': 80,  's12': 100, 'delta': 19.8, 'std': 0.5, 'n_seeds': 4},
    'Qwen 2.5 32B':      {'s1_300':  48, 's1_284': 60,  's12': 100, 'delta': 40.8, 'std': 4.1, 'n_seeds': 4},
    'Llama 3.1 8B':      {'s1_300':  60, 's1_284': 40,  's12':  92, 'delta': 45.0, 'std': 4.8, 'n_seeds': 4},
    'Mistral 7B':        {'s1_300':   8, 's1_284': 10,  's12':  32, 'delta': 20.3, 'std': 0.6, 'n_seeds': 3},
    'Mistral-Small 24B': {'s1_300':  72, 's1_284': 72,  's12':  80, 'delta': 10.0, 'std': 4.4, 'n_seeds': 3},
}


def render():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.linewidth': 0.7,
        'pdf.fonttype': 42,
    })
    backbones = list(KCONFOUND.keys())
    y_pos = list(range(len(backbones)))[::-1]

    fig, ax = plt.subplots(figsize=(5.8, 3.4))

    # Dumbbell line from S1@284 to S1+S2; rounded square + circle endpoints
    for y, name in zip(y_pos, backbones):
        d = KCONFOUND[name]
        ax.plot([d['s1_284'], d['s12']], [y, y], color='#999999', lw=2.5, zorder=2)
        ax.scatter([d['s1_284']], [y], c='#ffbb33', marker='s', s=70,
                   edgecolors='black', linewidths=0.5, zorder=3, label='Stage 1 @ $k{=}284$' if y == y_pos[0] else None)
        ax.scatter([d['s12']], [y], c='#d62728', marker='o', s=80,
                   edgecolors='black', linewidths=0.5, zorder=3, label='Stage 1+2 ($16{+}284$)' if y == y_pos[0] else None)
        # Unmatched S1 @ k=300 as a smaller faded x
        ax.scatter([d['s1_300']], [y], c='#cccccc', marker='x', s=44,
                   linewidths=1.3, zorder=2, label='Stage 1 @ $k{=}300$' if y == y_pos[0] else None)
        # Δ annotation to the right
        x_right = max(d['s1_300'], d['s1_284'], d['s12'])
        ax.annotate(
            f"$+{d['delta']:.1f}\\,\\pm\\,{d['std']:.1f}$",
            (x_right + 2, y),
            va='center', ha='left', fontsize=9, color='#222222',
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(backbones)
    ax.set_xlabel('Needle-in-a-haystack accuracy at $n{=}20$ segments (\%)')
    ax.set_xlim(-3, 130)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis='x', linestyle=':', linewidth=0.5, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=1.5,
        handletextpad=0.5,
    )
    plt.subplots_adjust(left=0.21, right=0.97, top=0.97, bottom=0.22)
    plt.savefig(OUT)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    render()
