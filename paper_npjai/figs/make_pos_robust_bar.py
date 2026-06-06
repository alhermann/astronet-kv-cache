"""Per-position SQuAD bar chart for the Stage 1 vs Stage 1+2 comparison.

Surfaces the pos=3 honesty: pure_S1 drops to ~36-40% at fact-at-end
while the S1+S2 hybrid recovers ~+4 to +8pp.  The averaged number in
tab:squad_main hides this structure.  Critic gate "per-position
reporting" is satisfied by this figure.

Data sources:
    logs/results/hybrid_pos_robust_v2_{model}{_seed}.json (5 seeds each)

Output:
    paper_npjai/figs/fig_pos_robust_bar.pdf
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "logs" / "results"
OUT = Path(__file__).resolve().parent / "fig_pos_robust_bar.pdf"

# Backbones and the seeds we have files for.
BACKBONES = [
    ("qwen7b",   "Qwen 2.5-7B"),
    ("qwen14b",  "Qwen 2.5-14B"),
    ("qwen32b",  "Qwen 2.5-32B"),
    ("llama8b",  "Llama 3.1-8B"),
    ("mistral7b",  "Mistral 7B"),
    ("mistral24b", "Mistral-Small 24B"),
]
SEEDS = ["", "_s7", "_s123", "_s999", "_s2024"]  # default file = seed 42

POSITIONS = [0, 1, 2, 3]


def load_one(backbone: str) -> dict | None:
    """Returns {pos: {'pure': [acc_per_seed], 'hybrid': [acc_per_seed]}}."""
    per_pos = defaultdict(lambda: {"pure": [], "hybrid": []})
    found_any = False
    for seed_suffix in SEEDS:
        path = RESULTS_DIR / f"hybrid_pos_robust_v2_{backbone}{seed_suffix}.json"
        if not path.exists():
            continue
        found_any = True
        with open(path) as f:
            d = json.load(f)
        results = d.get("results", {})
        for pos in POSITIONS:
            cell = results.get(f"pos_{pos}", {})
            if "pure300" in cell:
                per_pos[pos]["pure"].append(cell["pure300"])
            if "hybrid" in cell:
                per_pos[pos]["hybrid"].append(cell["hybrid"])
    if not found_any:
        return None
    return dict(per_pos)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(POSITIONS))
    width = 0.36

    for ax, (backbone, label) in zip(axes, BACKBONES):
        per_pos = load_one(backbone)
        if per_pos is None:
            ax.text(0.5, 0.5, f"{label}\n(no data)", ha="center", va="center",
                     transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=10)
            continue

        pure_mean = [np.mean(per_pos[p]["pure"]) if per_pos[p]["pure"] else 0
                     for p in POSITIONS]
        pure_se   = [np.std(per_pos[p]["pure"]) / np.sqrt(max(1, len(per_pos[p]["pure"])))
                     for p in POSITIONS]
        hyb_mean  = [np.mean(per_pos[p]["hybrid"]) if per_pos[p]["hybrid"] else 0
                     for p in POSITIONS]
        hyb_se    = [np.std(per_pos[p]["hybrid"]) / np.sqrt(max(1, len(per_pos[p]["hybrid"])))
                     for p in POSITIONS]
        n_seeds   = max(len(per_pos[p]["pure"]) for p in POSITIONS)

        ax.bar(x - width/2, pure_mean, width, yerr=pure_se, capsize=2,
                label="Stage 1", color="#9ec5fe", edgecolor="#1f4e9e", linewidth=0.6)
        ax.bar(x + width/2, hyb_mean, width, yerr=hyb_se,  capsize=2,
                label="Stage 1+2", color="#f5b87a", edgecolor="#c25e1a", linewidth=0.6)
        # Annotate per-position delta.
        for i, p in enumerate(POSITIONS):
            delta = (hyb_mean[i] - pure_mean[i]) * 100
            ax.annotate(f"+{delta:.0f}" if delta >= 0 else f"{delta:.0f}",
                         xy=(i, max(pure_mean[i], hyb_mean[i]) + 0.04),
                         ha="center", fontsize=8,
                         color="#c25e1a" if delta >= 0 else "#a01515")

        ax.set_title(f"{label}  (n={n_seeds} seeds)", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"pos {p}" for p in POSITIONS], fontsize=9)
        ax.set_ylim(0, 1.0)
        if ax in (axes[0], axes[3]):
            ax.set_ylabel("SQuAD accuracy", fontsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        # Highlight pos=3 (hardest, fact at end of context).
        ax.axvspan(2.5, 3.5, alpha=0.06, color="red")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle("", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
